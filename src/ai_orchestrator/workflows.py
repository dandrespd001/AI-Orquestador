"""Strict multi-provider workflow policy and durable attempt accounting."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import uuid
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path

from ai_orchestrator.abacus import AbacusDelegator
from ai_orchestrator.admission import AdmissionController
from ai_orchestrator.artifacts import ArtifactStore
from ai_orchestrator.config import config_root, load_gates, load_profile
from ai_orchestrator.delegation import OrchestratorTaskRunner
from ai_orchestrator.models import (
    ContractError,
    Profile,
    TaskSpec,
    ensure_private_directory,
)
from ai_orchestrator.telemetry import TelemetryStore, default_state_root


@dataclass(frozen=True)
class WorkflowStage:
    name: str
    model: str
    provider: str
    fallback_allowed: bool = False


@dataclass(frozen=True)
class StageExecution:
    status: str
    output: bytes = b""
    verdict: str | None = None
    run_id: str | None = None
    budget_tokens: int | None = None
    budget_credits: float | None = None


_ABACUS_ROUTE_MODEL = {
    "abacus-qwen3.8-max": "Qwen3.8 Max",
    "abacus-grok-4.6": "Grok 4.6",
}


def _verdict(output: bytes) -> str | None:
    try:
        payload = __import__("json").loads(output.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    value = str(payload.get("verdict", "")).strip().lower()
    return value if value in {"approve", "reject"} else None


class DefaultStageExecutor:
    """Bridge workflow stages to the existing bounded provider adapters."""

    def __init__(
        self,
        *,
        profile: str,
        task_runner: OrchestratorTaskRunner | object | None = None,
        abacus: AbacusDelegator | object | None = None,
        admission: AdmissionController | None = None,
    ) -> None:
        self.profile = profile
        self.task_runner = task_runner or OrchestratorTaskRunner()
        self.abacus = abacus or AbacusDelegator()
        self.admission = admission or AdmissionController()

    def __call__(
        self, stage: WorkflowStage, task: TaskSpec, prompt: bytes
    ) -> StageExecution:
        if stage.provider == "abacus":
            route_model = _ABACUS_ROUTE_MODEL.get(stage.model)
            if route_model is None:
                raise ContractError(f"unsupported Abacus workflow model: {stage.model}")
            task_type = (
                "agentic_implementation"
                if stage.name == "plan"
                else "review"
                if stage.name == "review"
                else task.task_type
            )
            with self.admission.acquire(
                model=stage.model,
                max_concurrency=1,
                repo=task.repo,
                allow_parallel=True,
                lease_seconds=task.timeout_seconds + 60,
                wait_timeout_seconds=min(30, task.timeout_seconds),
            ):
                receipt = self.abacus.delegate(
                    prompt=prompt.decode("utf-8", errors="strict"),
                    task_type=task_type,
                    model=route_model,
                    sensitivity=task.sensitivity,
                    timeout_seconds=min(600, task.timeout_seconds),
                )
            raw_output = str(receipt.get("result", "")).encode("utf-8")
            return StageExecution(
                status=(
                    "PATCH_READY"
                    if stage.name == "execute" and task.write_mode != "read_only"
                    else "SUCCESS"
                ),
                output=raw_output,
                verdict=_verdict(raw_output) if stage.name == "review" else None,
            )

        selected_task = task
        if stage.name == "review":
            selected_task = replace(
                task,
                task_type="review",
                write_mode="read_only",
                allowed_write_paths=(),
            )
        receipt = self.task_runner.run_inline(
            task=selected_task,
            prompt=prompt,
            profile=self.profile,
            model=stage.model,
        )
        output = str(receipt.get("output", "")).encode("utf-8")
        run = receipt.get("run")
        run_id = (
            str(run.get("run_id"))
            if isinstance(run, dict) and run.get("run_id")
            else None
        )
        return StageExecution(
            status=str(receipt.get("status", "FAILED")),
            output=output,
            verdict=_verdict(output) if stage.name == "review" else None,
            run_id=run_id,
        )


_STAGE = {
    "abacus-qwen3.8-max": ("abacus",),
    "abacus-grok-4.6": ("abacus",),
    "claude-opus": ("anthropic",),
    "claude-sonnet": ("anthropic",),
    "gpt-5.6-sol": ("openai",),
    "gpt-5.6-terra": ("openai",),
    "gpt-5.6-luna": ("openai",),
    "deepseek-v4-flash": ("deepseek",),
    "deepseek-v4-pro": ("deepseek",),
    "kimi-official": ("moonshot",),
}


def _stage(name: str, model: str) -> WorkflowStage:
    return WorkflowStage(name=name, model=model, provider=_STAGE[model][0])


_WORKFLOW_ROUTES: dict[str, tuple[WorkflowStage, ...]] = {
    "architecture": (
        _stage("execute", "claude-opus"),
        _stage("review", "abacus-grok-4.6"),
    ),
    "implementation": (
        _stage("plan", "abacus-qwen3.8-max"),
        _stage("execute", "claude-sonnet"),
        _stage("review", "abacus-grok-4.6"),
    ),
    "bulk_refactor": (
        _stage("plan", "abacus-qwen3.8-max"),
        _stage("execute", "gpt-5.6-sol"),
        _stage("review", "abacus-grok-4.6"),
    ),
    "tests": (
        _stage("plan", "abacus-qwen3.8-max"),
        _stage("execute", "deepseek-v4-flash"),
        _stage("review", "gpt-5.6-terra"),
    ),
    "docs": (
        _stage("plan", "abacus-qwen3.8-max"),
        _stage("execute", "claude-sonnet"),
        _stage("review", "abacus-grok-4.6"),
    ),
    "bounded_cpp": (
        _stage("execute", "deepseek-v4-flash"),
        _stage("review", "gpt-5.6-sol"),
    ),
    "determinism": (
        _stage("execute", "gpt-5.6-sol"),
        _stage("review", "claude-opus"),
    ),
    "godot_ui": (
        _stage("plan", "abacus-qwen3.8-max"),
        _stage("execute", "kimi-official"),
        _stage("review", "claude-sonnet"),
    ),
    "security": (
        _stage("execute", "abacus-grok-4.6"),
        _stage("review", "gpt-5.6-terra"),
    ),
    "research": (
        _stage("execute", "abacus-grok-4.6"),
        _stage("review", "claude-opus"),
    ),
}


def workflow_route(
    task_type: str,
    *,
    short_task: bool = False,
    qwen_plan: bool = True,
    qwen_patch: bool = False,
) -> tuple[WorkflowStage, ...]:
    normalized = str(task_type).strip().lower()
    if normalized not in _WORKFLOW_ROUTES:
        raise ContractError(f"unsupported CHUNSA workflow task type: {task_type!r}")
    route = _WORKFLOW_ROUTES[normalized]
    if qwen_patch:
        if normalized not in {"implementation", "bulk_refactor", "tests", "docs"}:
            raise ContractError("Qwen patch mode is unsupported for this task type")
        route = (
            _stage("execute", "abacus-qwen3.8-max"),
            _stage("review", "gpt-5.6-terra"),
        )
    if normalized == "bounded_cpp" and short_task:
        route = (_stage("execute", "deepseek-v4-pro"), route[-1])
    if normalized == "godot_ui" and not qwen_plan:
        route = tuple(row for row in route if row.name != "plan")
    if any(left.provider == right.provider for left, right in pairwise(route)):
        raise ContractError("adjacent workflow stages must use distinct providers")
    return route


MODEL_ATTEMPT_LIMITS = {
    "abacus-qwen3.8-max": 12,
    "abacus-grok-4.6": 6,
}


class WorkflowStore:
    """Private SQLite ledger for rolling limits and canary promotion."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = (path or default_state_root() / "workflows.sqlite").resolve()
        ensure_private_directory(self.path.parent)
        if self.path.is_symlink():
            raise ValueError("workflow database must not be a symlink")
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS model_attempts (
                    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    outcome TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_model_attempt_window
                    ON model_attempts(model, created_at);
                CREATE TABLE IF NOT EXISTS canary_results (
                    canary_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    passed INTEGER NOT NULL CHECK (passed IN (0, 1))
                );
                CREATE INDEX IF NOT EXISTS idx_canary_model
                    ON canary_results(model, canary_id);
                """
            )
        os.chmod(self.path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def reserve_attempt(
        self, model: str, *, now: datetime | None = None
    ) -> dict[str, object]:
        if model not in MODEL_ATTEMPT_LIMITS:
            return {"model": model, "limited": False, "remaining": None}
        observed = now or datetime.now(UTC)
        if observed.tzinfo is None:
            raise ValueError("attempt timestamp must be timezone-aware")
        cutoff = observed - timedelta(hours=24)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            used = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM model_attempts
                    WHERE model = ? AND created_at > ?
                    """,
                    (model, cutoff.isoformat()),
                ).fetchone()[0]
            )
            limit = MODEL_ATTEMPT_LIMITS[model]
            if used >= limit:
                connection.rollback()
                raise ContractError(f"24-hour attempt limit exhausted for {model}")
            connection.execute(
                "INSERT INTO model_attempts(model, created_at, outcome) VALUES(?,?,?)",
                (model, observed.isoformat(), "RESERVED"),
            )
            connection.commit()
        finally:
            connection.close()
        return {
            "model": model,
            "limited": True,
            "limit": limit,
            "remaining": limit - used - 1,
        }

    def record_attempt(
        self,
        model: str,
        *,
        outcome: str,
        created_at: datetime | None = None,
    ) -> None:
        observed = created_at or datetime.now(UTC)
        if observed.tzinfo is None:
            raise ValueError("attempt timestamp must be timezone-aware")
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT INTO model_attempts(model, created_at, outcome) VALUES(?,?,?)",
                (model, observed.isoformat(), str(outcome)),
            )

    def record_canary(self, model: str, *, passed: bool) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT INTO canary_results(model, created_at, passed) VALUES(?,?,?)",
                (model, datetime.now(UTC).isoformat(), int(bool(passed))),
            )

    def qwen_patch_enabled(self) -> bool:
        return self._last_five_canaries_passed("abacus-qwen3.8-max")

    def _last_five_canaries_passed(self, model: str) -> bool:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT passed FROM canary_results
                WHERE model = ?
                ORDER BY canary_id DESC LIMIT 5
                """,
                (model,),
            ).fetchall()
        return len(rows) == 5 and all(int(row[0]) == 1 for row in rows)

    def workflow_model_enabled(self, model: str) -> bool:
        """Keep a configured model candidate after any incomplete/failed batch."""

        if model not in MODEL_ATTEMPT_LIMITS:
            return True
        with closing(self._connect()) as connection:
            count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM canary_results WHERE model = ?", (model,)
                ).fetchone()[0]
            )
        return count == 0 or self._last_five_canaries_passed(model)


def _head(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise ContractError("workflow repository must be a Git worktree")
    return completed.stdout.decode().strip()


class WorkflowRunner:
    """Run the strict workflow without representing a prompt on disk."""

    def __init__(
        self,
        *,
        configuration_root: Path | None = None,
        profile_loader: Callable[[str], Profile] | None = None,
        stage_executor: Callable[[WorkflowStage, TaskSpec, bytes], StageExecution]
        | None = None,
        workflow_store: WorkflowStore | None = None,
        telemetry: TelemetryStore | None = None,
        artifacts: ArtifactStore | None = None,
    ) -> None:
        self.configuration_root = (configuration_root or config_root()).resolve()
        self.profile_loader = profile_loader or (
            lambda name: load_profile(name, self.configuration_root)
        )
        self.workflow_store = workflow_store or WorkflowStore()
        self.telemetry = telemetry or TelemetryStore()
        self.artifacts = artifacts or ArtifactStore(telemetry=self.telemetry)
        self.stage_executor = stage_executor

    def _gate_commands(
        self, task: TaskSpec, selected_profile: Profile
    ) -> tuple[tuple[str, ...], int]:
        if task.gate_profile == "profile":
            return selected_profile.deterministic_gates, task.timeout_seconds
        profiles = load_gates(self.configuration_root)
        gate = profiles.get(task.gate_profile)
        if gate is None:
            raise ContractError(f"unknown gate_profile: {task.gate_profile}")
        return gate.commands, gate.timeout_seconds

    @staticmethod
    def _prompt_for_stage(
        stage: WorkflowStage,
        original: bytes,
        *,
        plan: bytes,
        execution: bytes,
    ) -> bytes:
        if stage.name == "plan":
            prefix = b"Produce a bounded implementation plan only.\n\n"
            return prefix + original
        if stage.name == "execute":
            suffix = b""
            if plan:
                suffix = b"\n\nINDEPENDENT PLAN:\n" + plan
            return original + suffix
        return (
            b"Review the result independently. Return exactly one JSON object: "
            b'{"verdict":"approve"} or {"verdict":"reject"}. Reject on '
            b"uncertainty.\n\nORIGINAL TASK:\n"
            + original
            + b"\n\nEXECUTION RESULT:\n"
            + execution
        )

    def submit(
        self, *, prompt: str, metadata: dict[str, object], profile: str
    ) -> dict[str, object]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ContractError("workflow prompt must be a non-empty string")
        prompt_bytes = prompt.encode("utf-8")
        if len(prompt_bytes) > 100_000:
            raise ContractError("workflow prompt exceeds the 100000-byte limit")
        if not isinstance(metadata, dict):
            raise ContractError("workflow metadata must be an object")
        forbidden = {"repo", "profile", "prompt", "prompt_file"} & metadata.keys()
        if forbidden:
            raise ContractError("workflow repository and prompt come from the profile")
        selected_profile = self.profile_loader(profile)
        if selected_profile.name != profile or not selected_profile.repo_root:
            raise ContractError("workflow profile must bind exactly one repository")
        options = dict(metadata)
        short_task = bool(options.pop("short_task", False))
        qwen_plan = bool(options.pop("qwen_plan", True))
        qwen_patch = bool(options.pop("qwen_patch", False))
        if qwen_patch and not self.workflow_store.qwen_patch_enabled():
            raise ContractError(
                "Qwen patch mode requires five consecutive passing canaries"
            )
        options.update(
            {
                "task_id": str(uuid.uuid4()),
                "repo": selected_profile.repo_root,
                "sensitivity": options.get("sensitivity", "internal"),
            }
        )
        task = TaskSpec.from_mapping(options)
        route = workflow_route(
            task.task_type,
            short_task=short_task,
            qwen_plan=qwen_plan,
            qwen_patch=qwen_patch,
        )
        if task.write_mode != "read_only" and any(
            stage.name == "execute" and stage.model == "abacus-grok-4.6"
            for stage in route
        ):
            raise ContractError("Grok 4.6 is read-only in CHUNSA workflows")
        candidate_models = sorted(
            {
                stage.model
                for stage in route
                if not self.workflow_store.workflow_model_enabled(stage.model)
            }
        )
        if candidate_models:
            raise ContractError(
                "workflow model remains candidate after canary results: "
                + ", ".join(candidate_models)
            )
        stage_executor = self.stage_executor or DefaultStageExecutor(
            profile=profile,
            task_runner=OrchestratorTaskRunner(
                configuration_root=self.configuration_root,
                telemetry=self.telemetry,
            ),
        )
        workflow_id = str(uuid.uuid4())
        prompt_sha256 = __import__("hashlib").sha256(prompt_bytes).hexdigest()
        self.telemetry.record_workflow(
            workflow_id=workflow_id,
            task_id=task.task_id,
            profile=profile,
            task_type=task.task_type,
            prompt_sha256=prompt_sha256,
            status="RUNNING",
            budget_tokens=task.max_context_tokens,
            budget_credits=None,
        )
        base_head = _head(Path(task.repo))
        plan_output = b""
        execution_output = b""
        execution_status = ""
        stage_receipts: list[dict[str, object]] = []
        reviewer: WorkflowStage | None = None
        review_verdict: str | None = None
        for stage in route:
            stage_prompt = self._prompt_for_stage(
                stage,
                prompt_bytes,
                plan=plan_output,
                execution=execution_output,
            )
            try:
                if stage.model in MODEL_ATTEMPT_LIMITS:
                    self.workflow_store.reserve_attempt(stage.model)
                outcome = stage_executor(stage, task, stage_prompt)
                if not isinstance(outcome, StageExecution):
                    raise TypeError("workflow executor returned an invalid outcome")
                if (
                    qwen_patch
                    and stage.name == "execute"
                    and outcome.status == "PATCH_READY"
                ):
                    gate_commands, gate_timeout = self._gate_commands(
                        task, selected_profile
                    )
                    self.artifacts.validate_candidate(
                        task=task,
                        patch=outcome.output,
                        protected_prefixes=selected_profile.protected_path_prefixes,
                        gate_commands=gate_commands,
                        gate_timeout_seconds=gate_timeout,
                    )
            except Exception:
                self.telemetry.record_workflow_stage(
                    workflow_id=workflow_id,
                    stage=stage.name,
                    provider=stage.provider,
                    model=stage.model,
                    status="FAILED",
                    verdict="reject",
                )
                self.telemetry.finalize_workflow(
                    workflow_id, status="FAILED", verdict="reject"
                )
                raise
            normalized_verdict = (
                outcome.verdict.strip().lower() if outcome.verdict else None
            )
            self.telemetry.record_workflow_stage(
                workflow_id=workflow_id,
                stage=stage.name,
                provider=stage.provider,
                model=stage.model,
                status=outcome.status,
                verdict=normalized_verdict,
                run_id=outcome.run_id,
                budget_tokens=outcome.budget_tokens,
                budget_credits=outcome.budget_credits,
            )
            stage_receipts.append(
                {
                    "stage": stage.name,
                    "provider": stage.provider,
                    "model": stage.model,
                    "status": outcome.status,
                    "verdict": normalized_verdict,
                    "budget_tokens": outcome.budget_tokens,
                    "budget_credits": outcome.budget_credits,
                    "fallback_allowed": False,
                }
            )
            accepted_statuses = (
                {"SUCCESS"} if stage.name != "execute" else {"SUCCESS", "PATCH_READY"}
            )
            if outcome.status not in accepted_statuses:
                self.telemetry.finalize_workflow(
                    workflow_id, status="FAILED", verdict="reject"
                )
                return {
                    "schema_version": 1,
                    "workflow_id": workflow_id,
                    "status": "FAILED",
                    "artifact_id": None,
                    "stages": stage_receipts,
                    "fallback_allowed": False,
                }
            if stage.name == "plan":
                plan_output = outcome.output
            elif stage.name == "execute":
                execution_output = outcome.output
                execution_status = outcome.status
            else:
                reviewer = stage
                review_verdict = normalized_verdict

        if reviewer is None or review_verdict != "approve":
            self.telemetry.finalize_workflow(
                workflow_id, status="REJECTED", verdict="reject"
            )
            return {
                "schema_version": 1,
                "workflow_id": workflow_id,
                "status": "REJECTED",
                "artifact_id": None,
                "stages": stage_receipts,
                "fallback_allowed": False,
            }

        artifact_id: str | None = None
        final_status = "COMPLETED"
        if task.write_mode != "read_only":
            if execution_status != "PATCH_READY" or not execution_output:
                self.telemetry.finalize_workflow(
                    workflow_id, status="REJECTED", verdict="reject"
                )
                return {
                    "schema_version": 1,
                    "workflow_id": workflow_id,
                    "status": "REJECTED",
                    "artifact_id": None,
                    "stages": stage_receipts,
                    "fallback_allowed": False,
                }
            gate_commands, gate_timeout = self._gate_commands(task, selected_profile)
            manifest = self.artifacts.stage_patch(
                workflow_id=workflow_id,
                task_id=task.task_id,
                repo=Path(task.repo),
                base_head=base_head,
                patch=execution_output,
                allowed_write_paths=task.allowed_write_paths,
                protected_prefixes=selected_profile.protected_path_prefixes,
                gate_commands=gate_commands,
                gate_timeout_seconds=gate_timeout,
                reviewer_provider=reviewer.provider,
                reviewer_model=reviewer.model,
                reviewer_verdict="approve",
            )
            artifact_id = str(manifest["artifact_id"])
            final_status = "PATCH_READY"
        self.telemetry.finalize_workflow(
            workflow_id,
            status=final_status,
            verdict="approve",
            artifact_id=artifact_id,
        )
        receipt: dict[str, object] = {
            "schema_version": 1,
            "workflow_id": workflow_id,
            "status": final_status,
            "artifact_id": artifact_id,
            "stages": stage_receipts,
            "fallback_allowed": False,
            "human_acceptance_required": artifact_id is not None,
        }
        if task.write_mode == "read_only":
            receipt["result"] = execution_output.decode("utf-8", errors="replace")
        return receipt
