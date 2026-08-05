from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_orchestrator.models import RouteDecision, RunResult, TaskSpec, ensure_private_directory


class AdapterError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _minimal_environment(transport: str) -> dict[str, str]:
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
        "CODEX_HOME",
        "CLAUDE_CONFIG_DIR",
    }
    if transport == "omniroute":
        allowed.update({"OMNIROUTE_API_KEY", "OMNIROUTE_BASE_URL", "OMNIROUTE_CONTEXT"})
    result = {key: value for key, value in os.environ.items() if key in allowed}
    result.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    result["PYTHONDONTWRITEBYTECODE"] = "1"
    return result


def _command(decision: RouteDecision, prompt_path: Path) -> list[str]:
    if decision.transport == "codex_cli":
        return [
            "codex",
            "exec",
            "--json",
            "--model",
            str(decision.requested_route_model),
            "-",
        ]
    if decision.transport == "claude_cli":
        return [
            "claude",
            "--print",
            "--model",
            str(decision.requested_route_model),
            "--output-format",
            "json",
            "--tools",
            "",
        ]
    if decision.transport == "minimax_official_cli":
        return [
            "opencode", "run", "--pure", "--title", "ai-orchestrator-minimax-readonly",
            "--agent", "auditor", "-m",
            str(decision.requested_route_model),
        ]
    if decision.transport == "omniroute":
        return [
            "omniroute",
            "--output",
            "json",
            "--timeout",
            "900000",
            "chat",
            "--model",
            str(decision.requested_route_model),
            "--file",
            str(prompt_path),
        ]
    raise AdapterError("HANDOFF_REQUIRED", f"unsupported supervised transport: {decision.transport}")


def _find_reported_model(payload: Any) -> str | None:
    if isinstance(payload, dict):
        for key in ("model", "requestedModel", "requested_model"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
        for value in payload.values():
            found = _find_reported_model(value)
            if found:
                return found
    if isinstance(payload, list):
        for value in payload:
            found = _find_reported_model(value)
            if found:
                return found
    return None


def _reported_model(stdout: bytes, decision: RouteDecision) -> tuple[str | None, bool]:
    if decision.transport in {"codex_cli", "claude_cli", "minimax_official_cli"}:
        # Native clients receive an exact model argv and own their authenticated
        # connection. The argv is the provenance boundary for those transports.
        return decision.selected_model, True
    try:
        payload = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, False
    reported = _find_reported_model(payload)
    if not reported:
        return None, False
    expected = str(decision.requested_route_model)
    return reported, reported in {expected, expected.split("/", 1)[-1], decision.selected_model}


def _atomic_private_write(path: Path, data: bytes) -> None:
    parent = path.parent.resolve()
    ensure_private_directory(parent)
    if path.exists() and path.is_symlink():
        raise AdapterError("UNSAFE_OUTPUT", "output path must not be a symlink")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb", closefd=True) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def execute(
    task: TaskSpec,
    decision: RouteDecision,
    *,
    output_path: Path | None = None,
) -> tuple[RunResult, bytes]:
    if not decision.auto_execute or decision.status != "READY":
        raise AdapterError("HANDOFF_REQUIRED", decision.reason)
    prompt_path = task.prompt_path()
    if prompt_path is None:
        raise AdapterError("PROMPT_REQUIRED", "run requires prompt_file")

    command = _command(decision, prompt_path)
    prompt = prompt_path.read_bytes()
    started = datetime.now(UTC)
    start_ns = time.monotonic_ns()
    try:
        completed = subprocess.run(
            command,
            input=prompt if decision.transport != "omniroute" else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=task.repo,
            env=_minimal_environment(str(decision.transport)),
            timeout=task.timeout_seconds,
            check=False,
        )
        exit_code: int | None = completed.returncode
        stdout = completed.stdout
        if completed.returncode != 0:
            status = "FAILED"
            error_code = "MODEL_PROCESS_FAILED"
        else:
            status = "SUCCESS"
            error_code = None
    except subprocess.TimeoutExpired:
        exit_code = None
        stdout = b""
        status = "AMBIGUOUS_TIMEOUT"
        error_code = "AMBIGUOUS_TIMEOUT"

    finished = datetime.now(UTC)
    latency_ms = (time.monotonic_ns() - start_ns) // 1_000_000
    reported, provenance = _reported_model(stdout, decision)
    if status == "SUCCESS" and not provenance:
        status = "PROVENANCE_UNVERIFIED"
        error_code = "PROVENANCE_UNVERIFIED"

    digest = hashlib.sha256(stdout).hexdigest() if stdout else None
    if output_path is not None and stdout:
        resolved_output = output_path.expanduser().resolve()
        if resolved_output.is_relative_to(Path(task.repo).resolve()):
            raise AdapterError("OUTPUT_IN_REPO_FORBIDDEN", "model output must remain in staging")
        _atomic_private_write(resolved_output, stdout)

    result = RunResult(
        schema_version=1,
        run_id=str(uuid.uuid4()),
        decision_id=decision.decision_id,
        task_id=task.task_id,
        status=status,
        requested_model=str(decision.selected_model),
        reported_model=reported,
        provider=str(decision.provider),
        transport=str(decision.transport),
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        latency_ms=int(latency_ms),
        output_sha256=digest,
        output_bytes=len(stdout),
        exit_code=exit_code,
        provenance_verified=provenance,
        error_code=error_code,
    )
    return result, stdout
