"""Metadata-only canaries through already-authenticated official clients."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CanaryRoute:
    provider: str
    model: str
    command: tuple[str, ...]
    response_format: str = "text"


ROUTES = {
    "deepseek": CanaryRoute(
        "deepseek", "deepseek-v4-flash",
        (
            "omniroute", "--quiet", "--no-color", "--output", "json", "chat",
            "--model", "deepseek/deepseek-v4-flash", "__PROMPT__",
        ),
        "openai_json",
    ),
    "minimax": CanaryRoute(
        "minimax", "MiniMax-M3",
        (
            "opencode", "run", "--pure", "--title", "official-provider-canary",
            "--agent", "auditor", "-m", "minimax/MiniMax-M3",
        ),
    ),
    "moonshot": CanaryRoute(
        "moonshot", "kimi-official",
        ("kimi", "-p", "__PROMPT__", "--output-format", "text"),
    ),
}


def _environment() -> dict[str, str]:
    allowed = {
        "HOME", "USER", "LOGNAME", "PATH", "LANG", "LC_ALL", "TERM",
        "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "NO_COLOR",
    }
    result = {key: value for key, value in os.environ.items() if key in allowed}
    result.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    result["NO_COLOR"] = "1"
    return result


def _strip_ansi(value: bytes) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value.decode("utf-8", "replace"))


def run_official_canaries(
    providers: list[str] | None = None, *, timeout_seconds: int = 120
) -> dict[str, Any]:
    selected = providers or list(ROUTES)
    unknown = sorted(set(selected) - set(ROUTES))
    if unknown:
        raise ValueError("unknown canary providers: " + ",".join(unknown))
    if not 10 <= timeout_seconds <= 300:
        raise ValueError("canary timeout must be between 10 and 300 seconds")

    results: list[dict[str, Any]] = []
    for provider in selected:
        route = ROUTES[provider]
        executable = shutil.which(route.command[0])
        if not executable:
            results.append({
                "provider": provider, "model": route.model,
                "status": "CLIENT_UNAVAILABLE", "provenance_verified": False,
            })
            continue
        expected = "CANARY_" + secrets.token_hex(8)
        prompt = f"Reply with exactly {expected}"
        command = [executable, *route.command[1:]]
        use_stdin = "__PROMPT__" not in command
        if not use_stdin:
            command[command.index("__PROMPT__")] = prompt
        started = time.monotonic_ns()
        try:
            completed = subprocess.run(
                command,
                input=prompt.encode("utf-8") if use_stdin else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=_environment(),
                timeout=timeout_seconds,
                check=False,
            )
            latency_ms = (time.monotonic_ns() - started) // 1_000_000
            combined = _strip_ansi(completed.stdout + b"\n" + completed.stderr)
            exact = expected in {line.strip() for line in combined.splitlines()}
            reported_model: str | None = None
            if route.response_format == "openai_json" and completed.returncode == 0:
                try:
                    decoded = completed.stdout.decode("utf-8", "replace")
                    payload = json.loads(decoded[decoded.index("{"):])
                    message = ((payload.get("choices") or [{}])[0].get("message") or {})
                    exact = message.get("content", "").strip() == expected
                    reported_model = payload.get("model")
                except (UnicodeDecodeError, json.JSONDecodeError, AttributeError, IndexError, ValueError):
                    exact = False
            quota = "usage limit" in combined.lower() or "quota" in combined.lower()
            status = "PASSED" if completed.returncode == 0 and exact else (
                "QUOTA_UNAVAILABLE" if quota else "FAILED"
            )
            results.append({
                "provider": provider,
                "model": route.model,
                "reported_model": reported_model,
                "transport": "authenticated_official_client",
                "status": status,
                "latency_ms": int(latency_ms),
                "exit_code": completed.returncode,
                "stdout_bytes": len(completed.stdout),
                "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
                "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
                "provenance_verified": status == "PASSED",
                "orchestrator_content_retained": False,
            })
        except subprocess.TimeoutExpired:
            results.append({
                "provider": provider, "model": route.model,
                "status": "AMBIGUOUS_TIMEOUT", "provenance_verified": False,
                "orchestrator_content_retained": False,
            })

    statuses = {str(row["status"]) for row in results}
    overall = "PASSED" if statuses == {"PASSED"} else (
        "DEGRADED_QUOTA" if statuses <= {"PASSED", "QUOTA_UNAVAILABLE"} else "FAILED"
    )
    return {
        "schema_version": 1,
        "kind": "official_provider_canary",
        "status": overall,
        "providers": results,
        "incremental_spend_allowed": False,
        "third_party_routes_allowed": False,
    }
