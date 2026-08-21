"""Secure Promptfoo wrapper and metadata-only quality receipts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ai_orchestrator.benchmark import atomic_private_json
from ai_orchestrator.models import ContractError, ensure_private_directory

PROMPTFOO_VERSION = "0.122.0"
PROMPTFOO_SECURITY_ENV = {
    "PROMPTFOO_CACHE_TYPE": "memory",
    "PROMPTFOO_DISABLE_TELEMETRY": "1",
    "PROMPTFOO_DISABLE_UPDATE": "1",
    "PROMPTFOO_DISABLE_REMOTE_GENERATION": "true",
    "PROMPTFOO_DISABLE_REDTEAM_REMOTE_GENERATION": "true",
    "PROMPTFOO_DISABLE_SHARING": "true",
    "PROMPTFOO_SELF_HOSTED": "1",
}


def _network_denied_command(command: list[str]) -> list[str]:
    """Run a child in an empty network namespace or fail closed."""

    unshare = shutil.which("unshare")
    if not unshare:
        raise ContractError("Promptfoo requires unshare network isolation")
    return [unshare, "--user", "--map-root-user", "--net", *command]


def promptfoo_binary(project_root: Path) -> Path:
    candidate = (
        project_root / "evals" / "promptfoo" / "node_modules" / ".bin" / "promptfoo"
    )
    if candidate.is_symlink():
        resolved = candidate.resolve()
    else:
        resolved = candidate
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ContractError("pinned project-local Promptfoo is not installed")
    return resolved


def promptfoo_environment(
    home: Path, inherited: Mapping[str, str] | None = None
) -> dict[str, str]:
    source = inherited or os.environ
    allowed = {"PATH", "LANG", "LC_ALL", "TERM", "NO_COLOR"}
    environment = {key: value for key, value in source.items() if key in allowed}
    environment.update(PROMPTFOO_SECURITY_ENV)
    environment["HOME"] = str(home)
    environment["PROMPTFOO_LOG_DIR"] = str(home / "logs")
    environment["PROMPTFOO_CACHE_PATH"] = str(home / "cache")
    environment["PROMPTFOO_CONFIG_DIR"] = str(home / "config")
    environment["NODE_OPTIONS"] = "--disable-warning=ExperimentalWarning"
    environment.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    return environment


def _stats(payload: Any) -> tuple[int, int, int]:
    if not isinstance(payload, dict):
        return 0, 0, 0
    results = payload.get("results")
    if isinstance(results, dict):
        stats = results.get("stats")
        if isinstance(stats, dict):
            successes = int(stats.get("successes", stats.get("passes", 0)) or 0)
            failures = int(stats.get("failures", 0) or 0)
            errors = int(stats.get("errors", 0) or 0)
            return successes, failures, errors
    return 0, 0, 0


def _validate_config(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ContractError("Promptfoo config must be a regular file")
    raw = path.read_text(encoding="utf-8")
    lowered = raw.lower()
    forbidden = (
        "sharing: true",
        "api.promptfoo.app",
        "api.promptfoo.dev",
        "promptfoo.app/welcome",
        "openrouter",
        "helicone",
        "langfuse",
    )
    matches = [value for value in forbidden if value in lowered]
    if matches:
        raise ContractError("unsafe Promptfoo config: " + ",".join(matches))
    if "sharing: false" not in lowered:
        raise ContractError("Promptfoo config must declare sharing: false")


def run_quality_eval(
    *,
    project_root: Path,
    config: Path,
    output: Path,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    root = project_root.resolve()
    config = config.resolve()
    if not config.is_relative_to(root):
        raise ContractError("Promptfoo config must be inside the project")
    _validate_config(config)
    if not 1 <= timeout_seconds <= 1800:
        raise ContractError("quality eval timeout must be between 1 and 1800 seconds")

    executable = promptfoo_binary(root)
    state_parent = Path(
        os.environ.get("AI_ORCHESTRATOR_TMP", f"/tmp/ai-orchestrator-{os.getuid()}")
    )
    ensure_private_directory(state_parent)
    with tempfile.TemporaryDirectory(prefix="quality-", dir=state_parent) as temporary:
        home = Path(temporary)
        for name in ("logs", "cache", "config"):
            ensure_private_directory(home / name)
        raw_output = home / "promptfoo-results.json"
        command = [
            str(executable),
            "eval",
            "--config",
            str(config),
            "--no-share",
            "--no-cache",
            "--output",
            str(raw_output),
        ]
        started = time.monotonic_ns()
        completed = subprocess.run(
            _network_denied_command(command),
            cwd=config.parent,
            env=promptfoo_environment(home),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        latency_ms = (time.monotonic_ns() - started) // 1_000_000
        parsed: Any = {}
        if raw_output.is_file() and not raw_output.is_symlink():
            try:
                parsed = json.loads(raw_output.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                parsed = {}
        successes, failures, errors = _stats(parsed)
        receipt = {
            "schema_version": 1,
            "kind": "quality_eval",
            "engine": "promptfoo",
            "engine_version": PROMPTFOO_VERSION,
            "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "exit_code": completed.returncode,
            "latency_ms": int(latency_ms),
            "successes": successes,
            "failures": failures,
            "errors": errors,
            "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
            "raw_results_retained": False,
            "status": "PASSED"
            if completed.returncode == 0 and failures == 0 and errors == 0
            else "FAILED",
        }
        atomic_private_json(output, receipt)
        return receipt


def promptfoo_preflight(project_root: Path) -> dict[str, Any]:
    binary = promptfoo_binary(project_root.resolve())
    state_parent = Path(
        os.environ.get("AI_ORCHESTRATOR_TMP", f"/tmp/ai-orchestrator-{os.getuid()}")
    )
    ensure_private_directory(state_parent)
    with tempfile.TemporaryDirectory(
        prefix="promptfoo-preflight-", dir=state_parent
    ) as raw:
        completed = subprocess.run(
            _network_denied_command([str(binary), "--version"]),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            env=promptfoo_environment(Path(raw)),
            timeout=30,
            check=False,
        )
    observed = completed.stdout.decode("utf-8", "replace").strip()
    return {
        "available": completed.returncode == 0,
        "expected_version": PROMPTFOO_VERSION,
        "observed_version": observed,
        "version_matches": PROMPTFOO_VERSION in observed,
        "network_isolation": "unshare-user-net",
        "security_environment": dict(PROMPTFOO_SECURITY_ENV),
    }
