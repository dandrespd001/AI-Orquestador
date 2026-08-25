"""Common MCP delegation entrypoint over the orchestrator's existing policy."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from ai_orchestrator.adapters import execute
from ai_orchestrator.admission import AdmissionController
from ai_orchestrator.config import (
    config_root,
    load_gates,
    load_models,
    load_profile,
    load_roles,
)
from ai_orchestrator.models import ContractError, TaskSpec
from ai_orchestrator.routing import select_route
from ai_orchestrator.telemetry import TelemetryStore

MAX_MCP_TASK_OUTPUT_BYTES = 1_048_576
_PROFILE_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class OrchestratorTaskRunner:
    """Execute an explicit task spec through every configured core adapter.

    MCP is never an apply surface. Writer models operate in isolated worktrees
    and can return only a reviewed `PATCH_READY` artifact.
    """

    def __init__(
        self,
        *,
        configuration_root: Path | None = None,
        executor: Callable[..., Any] = execute,
        telemetry: Any | None = None,
        admission: Any | None = None,
    ) -> None:
        self.configuration_root = (configuration_root or config_root()).resolve()
        self.executor = executor
        self.telemetry = telemetry
        self.admission = admission

    def run(
        self,
        *,
        spec_path: str,
        profile: str = "default",
        model: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(spec_path, str) or not spec_path.strip():
            raise ContractError("spec_path must be a non-empty string")
        if not isinstance(profile, str) or not _PROFILE_NAME.fullmatch(profile):
            raise ContractError("invalid profile name")
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise ContractError("model pin must be a non-empty string")

        unresolved = Path(spec_path).expanduser()
        if unresolved.is_symlink():
            raise ContractError("task spec must not be a symlink")
        try:
            source = unresolved.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ContractError("task spec does not exist") from exc
        if not source.is_file() or source.stat().st_size > 1_000_000:
            raise ContractError("task spec must be a regular file of at most 1 MB")

        task = TaskSpec.read(source)
        if task.write_mode == "validated_apply":
            raise ContractError(
                "MCP delegation requires read_only or validated_patch; "
                "validated_apply is forbidden"
            )
        if task.prompt_path() is None:
            raise ContractError("MCP delegation requires prompt_file in the task spec")

        models = load_models(self.configuration_root)
        roles = load_roles(self.configuration_root)
        selected_profile = load_profile(profile, self.configuration_root)
        telemetry = self.telemetry or TelemetryStore()
        observation_loader = getattr(telemetry, "routing_observations", None)
        observations = observation_loader() if observation_loader else None
        decision = select_route(
            task,
            models,
            roles,
            selected_profile,
            observations=observations,
            pin=model,
        )
        telemetry.record_route(task, decision)
        if not decision.auto_execute or decision.status != "READY":
            return {
                "schema_version": 1,
                "status": "HANDOFF_REQUIRED",
                "decision": decision.as_dict(),
                "fallback_allowed": False,
            }

        if task.gate_profile == "profile":
            gate_commands = selected_profile.deterministic_gates
            gate_timeout_seconds = task.timeout_seconds
        else:
            gate_profiles = load_gates(self.configuration_root)
            if task.gate_profile not in gate_profiles:
                raise ContractError(f"unknown gate_profile: {task.gate_profile}")
            gate = gate_profiles[task.gate_profile]
            gate_commands = gate.commands
            gate_timeout_seconds = gate.timeout_seconds

        admission = self.admission or AdmissionController()
        with admission.acquire(
            model=str(decision.selected_model),
            max_concurrency=decision.max_concurrency,
            repo=task.repo,
            allow_parallel=task.allow_parallel,
            lease_seconds=task.timeout_seconds + 60,
            wait_timeout_seconds=min(30, task.timeout_seconds),
        ):
            result, output = self.executor(
                task,
                decision,
                gate_commands=gate_commands,
                gate_timeout_seconds=gate_timeout_seconds,
                protected_prefixes=selected_profile.protected_path_prefixes,
            )
        if len(output) > MAX_MCP_TASK_OUTPUT_BYTES:
            result = replace(
                result,
                status="OUTPUT_LIMIT_EXCEEDED",
                output_sha256=None,
                output_bytes=len(output),
                exit_code=125,
                error_code="OUTPUT_LIMIT_EXCEEDED",
            )
            telemetry.record_run(result)
            raise ContractError("delegated output exceeds the 1 MB MCP limit")
        telemetry.record_run(result)
        return {
            "schema_version": 1,
            "status": result.status,
            "decision": decision.as_dict(),
            "run": result.as_dict(),
            "output": output.decode("utf-8", errors="replace"),
            "source_repo_modified": result.status == "APPLIED",
            "review_required": result.status == "PATCH_READY",
            "fallback_allowed": False,
        }

    def run_inline(
        self,
        *,
        task: TaskSpec,
        prompt: bytes,
        profile: str,
        model: str,
    ) -> dict[str, Any]:
        """Execute bytes held by the caller; persist only their SHA-256."""

        if not isinstance(prompt, bytes) or not prompt:
            raise ContractError("inline prompt must be non-empty bytes")
        if len(prompt) > 100_000:
            raise ContractError("inline prompt exceeds the 100000-byte limit")
        if not isinstance(profile, str) or not _PROFILE_NAME.fullmatch(profile):
            raise ContractError("invalid profile name")
        if not isinstance(model, str) or not model.strip():
            raise ContractError("model pin must be a non-empty string")
        if task.write_mode == "validated_apply":
            raise ContractError("inline workflow cannot auto-apply a patch")
        task = replace(task, inline_prompt_sha256=hashlib.sha256(prompt).hexdigest())
        models = load_models(self.configuration_root)
        roles = load_roles(self.configuration_root)
        selected_profile = load_profile(profile, self.configuration_root)
        telemetry = self.telemetry or TelemetryStore()
        observation_loader = getattr(telemetry, "routing_observations", None)
        observations = observation_loader() if observation_loader else None
        decision = select_route(
            task,
            models,
            roles,
            selected_profile,
            observations=observations,
            pin=model,
        )
        telemetry.record_route(task, decision)
        if not decision.auto_execute or decision.status != "READY":
            return {
                "schema_version": 1,
                "status": "HANDOFF_REQUIRED",
                "decision": decision.as_dict(),
                "fallback_allowed": False,
            }
        if task.gate_profile == "profile":
            gate_commands = selected_profile.deterministic_gates
            gate_timeout_seconds = task.timeout_seconds
        else:
            gate_profiles = load_gates(self.configuration_root)
            if task.gate_profile not in gate_profiles:
                raise ContractError(f"unknown gate_profile: {task.gate_profile}")
            gate = gate_profiles[task.gate_profile]
            gate_commands = gate.commands
            gate_timeout_seconds = gate.timeout_seconds
        admission = self.admission or AdmissionController()
        with admission.acquire(
            model=str(decision.selected_model),
            max_concurrency=decision.max_concurrency,
            repo=task.repo,
            allow_parallel=task.allow_parallel,
            lease_seconds=task.timeout_seconds + 60,
            wait_timeout_seconds=min(30, task.timeout_seconds),
        ):
            result, output = self.executor(
                task,
                decision,
                gate_commands=gate_commands,
                gate_timeout_seconds=gate_timeout_seconds,
                protected_prefixes=selected_profile.protected_path_prefixes,
                prompt_bytes=prompt,
            )
        if len(output) > MAX_MCP_TASK_OUTPUT_BYTES:
            result = replace(
                result,
                status="OUTPUT_LIMIT_EXCEEDED",
                output_sha256=None,
                output_bytes=len(output),
                exit_code=125,
                error_code="OUTPUT_LIMIT_EXCEEDED",
            )
            telemetry.record_run(result)
            raise ContractError("delegated output exceeds the 1 MB MCP limit")
        telemetry.record_run(result)
        return {
            "schema_version": 1,
            "status": result.status,
            "decision": decision.as_dict(),
            "run": result.as_dict(),
            "output": output.decode("utf-8", errors="replace"),
            "source_repo_modified": False,
            "review_required": result.status == "PATCH_READY",
            "fallback_allowed": False,
        }
