"""Abacus CLI policy: human login, Pro quota first, never auto-purchase."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ai_orchestrator.models import ContractError
from ai_orchestrator.supervisor import ProcessOutcome, run_supervised

ABACUS_CLI_VERSION = "2.6.9"
MAX_ABACUS_PROMPT_BYTES = 100_000
MAX_ABACUS_OUTPUT_BYTES = 256 * 1024
ABACUS_TASK_ROUTES = {
    "architecture": "Grok 4.6",
    "review": "Grok 4.6",
    "reasoning": "Grok 4.6",
    "research": "Grok 4.6",
    "security": "Grok 4.6",
    "agentic_implementation": "Qwen3.8 Max",
    "implementation": "Qwen3.8 Max",
    "bulk_refactor": "Qwen3.8 Max",
    "tests": "Qwen3.8 Max",
    "docs": "Qwen3.8 Max",
    "boilerplate": "Qwen3.8 Max",
    "long_context": "Qwen3.8 Max",
}
ABACUS_MODELS = {
    "Grok 4.6": {"status": "available", "effort": "medium"},
    # VulcanBench observed lower success as Qwen3.8 Max effort increased.
    "Qwen3.8 Max": {"status": "available", "effort": "low_or_medium"},
    # Retained for explicit bounded trials; external project-level evidence is weak.
    "Qwen3.8 27B": {"status": "available", "effort": "medium"},
    # New vendor release with insufficient independent evaluation as of 2026-08-20.
    "ZAI GLM 5.3": {"status": "experimental", "effort": "medium"},
}


def select_abacus_model(
    task_type: str,
    *,
    requested_model: str | None = None,
    allow_experimental: bool = False,
) -> str:
    """Select an allowlisted Abacus model without a silent fallback."""

    normalized_task = task_type.strip().lower()
    if normalized_task not in ABACUS_TASK_ROUTES:
        raise ContractError(f"unsupported Abacus task type: {task_type!r}")
    model = requested_model or ABACUS_TASK_ROUTES[normalized_task]
    metadata = ABACUS_MODELS.get(model)
    if metadata is None:
        raise ContractError(f"unsupported Abacus model: {model!r}")
    if metadata["status"] == "experimental" and not allow_experimental:
        raise ContractError(f"experimental Abacus model requires opt-in: {model}")
    return model


def _default_abacus_executable() -> str:
    pinned = Path.home() / ".abacusai" / "versions" / ABACUS_CLI_VERSION / "abacusai"
    return str(pinned if pinned.is_file() else (shutil.which("abacusai") or pinned))


class AbacusDelegator:
    """Run a no-tools Abacus query in an empty disposable directory."""

    def __init__(
        self,
        *,
        executable: str | None = None,
        runner: Callable[..., ProcessOutcome] = run_supervised,
    ) -> None:
        self.executable = executable or _default_abacus_executable()
        self.runner = runner

    def delegate(
        self,
        *,
        prompt: str,
        task_type: str,
        model: str | None = None,
        sensitivity: str = "internal",
        timeout_seconds: int = 300,
        allow_experimental: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ContractError("Abacus prompt must be a non-empty string")
        if not isinstance(task_type, str):
            raise ContractError("Abacus task type must be a string")
        if type(allow_experimental) is not bool:
            raise ContractError("allow_experimental must be a boolean")
        prompt_bytes = prompt.encode("utf-8")
        if len(prompt_bytes) > MAX_ABACUS_PROMPT_BYTES:
            raise ContractError("Abacus prompt exceeds the 100000-byte limit")
        if sensitivity not in {"public", "internal"}:
            raise ContractError(
                "Abacus delegation accepts public or internal data only"
            )
        if type(timeout_seconds) is not int or not 10 <= timeout_seconds <= 600:
            raise ContractError("Abacus timeout must be between 10 and 600 seconds")

        selected_model = select_abacus_model(
            task_type,
            requested_model=model,
            allow_experimental=allow_experimental,
        )
        command = [
            self.executable,
            "--print",
            "--model",
            selected_model,
            "--output-format",
            "text",
            "--permission-mode",
            "normal",
            "--disallowed-tools",
            "*",
            "--no-agents-md",
        ]
        try:
            with tempfile.TemporaryDirectory(prefix="ai-orchestrator-abacus-") as work:
                os.chmod(work, 0o700)
                outcome = self.runner(
                    command,
                    input_data=prompt_bytes,
                    cwd=work,
                    env=abacus_environment(),
                    timeout_seconds=timeout_seconds,
                )
        except OSError as exc:
            raise RuntimeError("Abacus CLI could not be started") from exc

        if outcome.timed_out:
            raise RuntimeError("Abacus CLI timed out")
        if outcome.returncode != 0:
            raise RuntimeError(f"Abacus CLI failed with status {outcome.returncode}")
        if len(outcome.stdout) > MAX_ABACUS_OUTPUT_BYTES:
            raise ContractError("Abacus output exceeds the 262144-byte limit")
        return {
            "schema_version": 1,
            "provider": "abacus",
            "model": selected_model,
            "task_type": task_type,
            "result": outcome.stdout.decode("utf-8", errors="replace"),
            "authentication": "human_cli_login",
            "quota_policy": "pro_included_credits_only",
            "automatic_purchase": False,
            "incremental_spend_allowed": False,
            "allowed_tools": [],
            "filesystem_access": "disposable_empty_directory",
            "fallback_allowed": False,
        }


@dataclass(frozen=True)
class AbacusPolicy:
    cli_version: str = ABACUS_CLI_VERSION
    authentication: str = "human_cli_login"
    preferred_quota: str = "pro_included_credits"
    routellm_api_enabled: bool = False
    supercomputer_included_credits_only: bool = True
    automatic_purchases: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def abacus_environment() -> dict[str, str]:
    allowed = {
        "HOME",
        "USER",
        "LOGNAME",
        "PATH",
        "LANG",
        "LC_ALL",
        "TERM",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
        "NO_COLOR",
    }
    result = {key: value for key, value in os.environ.items() if key in allowed}
    result.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    result["NO_COLOR"] = "1"
    result["AI_ORCHESTRATOR_AUTO_PURCHASE"] = "0"
    return result


def validate_abacus_version(output: str) -> dict[str, Any]:
    observed = output.strip().lstrip("v")
    return {
        "required": ABACUS_CLI_VERSION,
        "observed": observed or None,
        "matches": observed == ABACUS_CLI_VERSION,
    }


def plan_supercomputer_use(
    *,
    available_included_credits: float,
    estimated_minutes: int,
    human_approved: bool = False,
) -> dict[str, Any]:
    if available_included_credits < 0 or estimated_minutes < 1:
        raise ContractError("invalid Abacus SuperComputer budget")
    estimated_credits = math.ceil(estimated_minutes / 5)
    if estimated_credits > available_included_credits:
        raise ContractError("included Abacus credits are insufficient")
    if not human_approved:
        raise ContractError("SuperComputer use requires a human approval")
    return {
        "schema_version": 1,
        "approved": True,
        "estimated_minutes": estimated_minutes,
        "estimated_credits": estimated_credits,
        "included_credits_only": True,
        "automatic_purchase": False,
    }
