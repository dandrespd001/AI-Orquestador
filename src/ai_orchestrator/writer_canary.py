"""Disposable write-capability canaries for promotion decisions."""

from __future__ import annotations

import hashlib
import secrets
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from ai_orchestrator.adapters import execute
from ai_orchestrator.config import load_models
from ai_orchestrator.models import ContractError, RouteDecision, TaskSpec


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=20,
        check=True,
    )


def run_writer_canaries(
    model_id: str, *, attempts: int = 5, timeout_seconds: int = 120
) -> dict[str, Any]:
    if not 1 <= attempts <= 10:
        raise ValueError("writer canary attempts must be between 1 and 10")
    models = load_models()
    if model_id not in models:
        raise ContractError(f"unknown model: {model_id}")
    model = models[model_id]
    if "write" not in model.capabilities or model.execution_profile == "read_only":
        raise ContractError(f"model is not configured for writing: {model_id}")

    rows: list[dict[str, Any]] = []
    for attempt in range(1, attempts + 1):
        expected = "CANARY_" + secrets.token_hex(8)
        started = time.monotonic_ns()
        with tempfile.TemporaryDirectory(prefix="ai-orchestrator-writer-canary-") as directory:
            repo = Path(directory)
            _git(repo, "init", "-q")
            _git(repo, "config", "user.email", "canary@example.invalid")
            _git(repo, "config", "user.name", "AI Orchestrator Canary")
            (repo / "prompt.txt").write_text(
                f"Create canary.txt containing exactly {expected} followed by a newline.",
                encoding="utf-8",
            )
            _git(repo, "add", "prompt.txt")
            _git(repo, "commit", "-qm", "canary baseline")
            task = TaskSpec.from_mapping(
                {
                    "repo": str(repo),
                    "task_type": "boilerplate",
                    "sensitivity": "public" if model.trust_tier == "web_free" else "internal",
                    "prompt_file": "prompt.txt",
                    "output_contract": "unified_diff",
                    "write_mode": "validated_patch",
                    "allowed_write_paths": ["canary.txt"],
                    "gate_profile": "profile",
                    "max_repairs": 0,
                    "timeout_seconds": timeout_seconds,
                }
            )
            decision = RouteDecision(
                schema_version=1,
                decision_id=str(uuid.uuid4()),
                task_id=task.task_id,
                profile="writer-canary",
                status="READY",
                selected_model=model.id,
                requested_route_model=model.route_model,
                provider=model.provider,
                family=model.family,
                transport=model.transport,
                reviewer_model=None,
                alternatives=(),
                unavailable={},
                auto_execute=True,
                approval_required=False,
                reason="explicit disposable writer canary",
                context_sha256=task.context_sha256(),
                fallback_allowed=False,
                write_mode=task.write_mode,
                trust_tier=model.trust_tier,
                execution_profile=model.execution_profile,
            )
            result, patch = execute(
                task,
                decision,
                gate_commands=(
                    f"python3 -c \"assert open('canary.txt').read().strip() == '{expected}'\"",
                ),
                protected_prefixes=(".git", ".env", ".codex", ".claude", ".kimi-code"),
            )
            passed = (
                result.status == "PATCH_READY"
                and result.gates_passed is True
                and result.provenance_verified
                and expected.encode("utf-8") in patch
            )
            rows.append(
                {
                    "attempt": attempt,
                    "status": "PASSED" if passed else result.status,
                    "latency_ms": (time.monotonic_ns() - started) // 1_000_000,
                    "patch_sha256": hashlib.sha256(patch).hexdigest() if patch else None,
                    "changed_files": result.changed_files,
                    "exit_code": result.exit_code,
                    "error_code": result.error_code,
                    "provenance_verified": result.provenance_verified,
                    "orchestrator_content_retained": False,
                }
            )
    passed_count = sum(row["status"] == "PASSED" for row in rows)
    return {
        "schema_version": 1,
        "kind": "writer_canary",
        "model": model.id,
        "provider": model.provider,
        "trust_tier": model.trust_tier,
        "attempts": attempts,
        "passed": passed_count,
        "status": "PASSED" if passed_count == attempts else "FAILED",
        "promotion_ready": attempts >= 5 and passed_count == attempts,
        "results": rows,
        "orchestrator_content_retained": False,
    }
