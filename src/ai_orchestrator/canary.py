"""Metadata-only canaries through already-authenticated official clients."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_orchestrator.credentials import minimax_runtime_key, omniroute_runtime_key
from ai_orchestrator.supervisor import run_supervised


@dataclass(frozen=True)
class CanaryRoute:
    provider: str
    model: str
    command: tuple[str, ...]
    response_format: str = "text"


ROUTES = {
    "deepseek": CanaryRoute(
        "deepseek", "ds/deepseek-v4-flash",
        (
            "qwen", "--auth-type", "openai", "--bare", "--output-format", "json",
            "--approval-mode", "plan", "--exclude-tools", "run_shell_command,agent,web_fetch,web_search,skill",
            "--max-wall-time", "__TIMEOUT__",
            "--max-session-turns", "2", "--max-tool-calls", "0",
            "--model", "ds/deepseek-v4-flash",
        ),
        "qwen_json",
    ),
    "minimax": CanaryRoute(
        "minimax", "MiniMax-M3",
        (
            "qwen", "--auth-type", "anthropic", "--bare", "--output-format", "json",
            "--approval-mode", "plan", "--exclude-tools", "run_shell_command,agent,web_fetch,web_search,skill",
            "--max-wall-time", "__TIMEOUT__",
            "--max-session-turns", "2", "--max-tool-calls", "0",
            "--model", "MiniMax-M3",
        ),
        "qwen_json",
    ),
    "moonshot": CanaryRoute(
        "moonshot", "kimi-official",
        ("kimi", "-p", "__PROMPT__", "--output-format", "text"),
    ),
}


def _environment(provider: str | None = None) -> dict[str, str]:
    allowed = {
        "HOME", "USER", "LOGNAME", "PATH", "LANG", "LC_ALL", "TERM",
        "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "NO_COLOR",
    }
    if provider == "deepseek":
        allowed.add("OMNIROUTE_API_KEY")
    elif provider == "minimax":
        allowed.add("MINIMAX_API_KEY")
    result = {key: value for key, value in os.environ.items() if key in allowed}
    if provider == "deepseek":
        runtime_key = result.pop("OMNIROUTE_API_KEY", None) or omniroute_runtime_key(result)
        if runtime_key:
            result["OPENAI_API_KEY"] = runtime_key
        result["OPENAI_BASE_URL"] = "http://127.0.0.1:20128/v1"
    elif provider == "minimax":
        runtime_key = minimax_runtime_key(result)
        result.pop("MINIMAX_API_KEY", None)
        if runtime_key:
            result["ANTHROPIC_API_KEY"] = runtime_key
        # SIN sufijo /v1 — ver el comentario en adapters.py. OJO: esta URL esta
        # DUPLICADA en los dos ficheros y pueden derivar; arreglar una sola deja
        # el canario fallando mientras el adaptador funciona, que es justo lo
        # que paso el 2026-08-12.
        result["ANTHROPIC_BASE_URL"] = "https://api.minimax.io/anthropic"
    result.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    result["NO_COLOR"] = "1"
    return result


def _strip_ansi(value: bytes) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value.decode("utf-8", "replace"))


def _qwen_canary(stdout: bytes, expected: str, model: str) -> tuple[bool, str | None]:
    try:
        payload = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False, None
    if not isinstance(payload, list):
        return False, None
    reported_model: str | None = None
    for row in payload:
        if not isinstance(row, dict):
            continue
        if row.get("type") == "system" and isinstance(row.get("model"), str):
            reported_model = str(row["model"])
        if row.get("type") == "result":
            exact = (
                row.get("is_error") is False
                and row.get("subtype") == "success"
                and str(row.get("result", "")).strip() == expected
            )
            valid_model = reported_model in {model, model.split("/", 1)[-1]}
            return exact and valid_model, reported_model
    return False, reported_model


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
        if "__TIMEOUT__" in command:
            command[command.index("__TIMEOUT__")] = f"{timeout_seconds}s"
        use_stdin = "__PROMPT__" not in command
        if not use_stdin:
            command[command.index("__PROMPT__")] = prompt
        started = time.monotonic_ns()
        completed = run_supervised(
            command,
            input_data=prompt.encode("utf-8") if use_stdin else None,
            cwd=Path.cwd(),
            env=_environment(provider),
            timeout_seconds=timeout_seconds,
        )
        latency_ms = (time.monotonic_ns() - started) // 1_000_000
        combined = _strip_ansi(completed.stdout + b"\n" + completed.stderr)
        exact = expected in {line.strip() for line in combined.splitlines()}
        reported_model: str | None = None
        if route.response_format == "qwen_json" and completed.returncode == 0:
            exact, reported_model = _qwen_canary(completed.stdout, expected, route.model)
        lower = combined.lower()
        quota = "usage limit" in lower or "quota" in lower
        missing_credential = (
            "missing api key" in lower
            or "no auth type" in lower
            or "api_key environment variable not found" in lower
        )
        status = "WALL_TIMEOUT" if completed.timed_out else (
            "PASSED" if completed.returncode == 0 and exact else (
                "QUOTA_UNAVAILABLE" if quota else (
                    "CREDENTIAL_UNAVAILABLE" if missing_credential else "FAILED"
                )
            )
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
    statuses = {str(row["status"]) for row in results}
    overall = "PASSED" if statuses == {"PASSED"} else (
        "DEGRADED_QUOTA" if statuses <= {"PASSED", "QUOTA_UNAVAILABLE"} else (
            "DEGRADED_UNAVAILABLE"
            if statuses <= {"PASSED", "QUOTA_UNAVAILABLE", "CREDENTIAL_UNAVAILABLE"}
            else "FAILED"
        )
    )
    return {
        "schema_version": 1,
        "kind": "official_provider_canary",
        "status": overall,
        "providers": results,
        "incremental_spend_allowed": False,
        "third_party_routes_allowed": False,
    }
