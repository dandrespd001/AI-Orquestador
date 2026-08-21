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
    interactive_auth_only: bool = False


ROUTES = {
    "openai": CanaryRoute(
        "openai",
        "gpt-5.6-luna",
        (
            "codex",
            "exec",
            "--json",
            "--ephemeral",
            "--ignore-user-config",
            "--sandbox",
            "read-only",
            "--model",
            "gpt-5.6-luna",
            "-c",
            'model_reasoning_effort="low"',
            "-",
        ),
        "codex_jsonl",
    ),
    "anthropic": CanaryRoute(
        "anthropic",
        "sonnet",
        (
            "claude",
            "--print",
            "--safe-mode",
            "--no-session-persistence",
            "--model",
            "sonnet",
            "--effort",
            "low",
            "--output-format",
            "json",
            "--tools",
            "",
            "--permission-mode",
            "plan",
            "--disallowedTools",
            "Bash,Agent,WebFetch,WebSearch,Edit,Write",
        ),
        "claude_json",
    ),
    "deepseek": CanaryRoute(
        "deepseek",
        "ds/deepseek-v4-flash",
        (
            "qwen",
            "--auth-type",
            "openai",
            "--bare",
            "--output-format",
            "json",
            "--approval-mode",
            "plan",
            "--exclude-tools",
            "run_shell_command,agent,web_fetch,web_search,skill",
            "--max-wall-time",
            "__TIMEOUT__",
            "--max-session-turns",
            "2",
            "--max-tool-calls",
            "0",
            "--model",
            "ds/deepseek-v4-flash",
        ),
        "qwen_json",
    ),
    "omniroute": CanaryRoute(
        "omniroute",
        "ds/deepseek-v4-flash",
        (
            "qwen",
            "--auth-type",
            "openai",
            "--bare",
            "--output-format",
            "json",
            "--approval-mode",
            "plan",
            "--exclude-tools",
            "run_shell_command,agent,web_fetch,web_search,skill",
            "--max-wall-time",
            "__TIMEOUT__",
            "--max-session-turns",
            "2",
            "--max-tool-calls",
            "0",
            "--model",
            "ds/deepseek-v4-flash",
        ),
        "qwen_json",
    ),
    "minimax": CanaryRoute(
        "minimax",
        "MiniMax-M3",
        (
            "qwen",
            "--auth-type",
            "openai",
            "--bare",
            "--output-format",
            "json",
            "--approval-mode",
            "plan",
            "--exclude-tools",
            "run_shell_command,agent,web_fetch,web_search,skill",
            "--max-wall-time",
            "__TIMEOUT__",
            "--max-session-turns",
            "2",
            "--max-tool-calls",
            "0",
            "--model",
            "MiniMax-M3",
        ),
        "qwen_json",
    ),
    "moonshot": CanaryRoute(
        "moonshot",
        "kimi-official",
        ("kimi", "-p", "__PROMPT__", "--output-format", "text"),
    ),
    # The official Abacus CLI documents an interactive login but no stable
    # non-interactive prompt contract. The canary therefore verifies the pinned
    # client and reports that the human login probe is still required instead
    # of inventing an unsafe command or reading its OAuth material.
    "abacus": CanaryRoute(
        "abacus",
        "abacus-pro",
        ("abacusai", "--version"),
        "version",
        interactive_auth_only=True,
    ),
    "fx": CanaryRoute(
        "fx",
        "zai/glm-5.2",
        (
            "fx",
            "--no-additional-dirs",
            "ask",
            "--json",
            "--no-save",
            "--no-color",
        ),
        "fx_json",
    ),
}
DEFAULT_ROUTES = ("deepseek", "minimax", "moonshot")


def _environment(provider: str | None = None) -> dict[str, str]:
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
    if provider in {"deepseek", "omniroute"}:
        allowed.add("OMNIROUTE_API_KEY")
    elif provider == "minimax":
        allowed.add("MINIMAX_API_KEY")
    result = {key: value for key, value in os.environ.items() if key in allowed}
    if provider in {"deepseek", "omniroute"}:
        runtime_key = result.pop("OMNIROUTE_API_KEY", None) or omniroute_runtime_key(
            result
        )
        if runtime_key:
            result["OPENAI_API_KEY"] = runtime_key
        result["OPENAI_BASE_URL"] = "http://127.0.0.1:20128/v1"
    elif provider == "minimax":
        runtime_key = minimax_runtime_key(result)
        result.pop("MINIMAX_API_KEY", None)
        if runtime_key:
            result["OPENAI_API_KEY"] = runtime_key
        result["OPENAI_BASE_URL"] = "https://api.minimax.io/v1"
    elif provider == "fx":
        result["FX_MODEL"] = "zai/glm-5.2"
        result["FX_PERMISSION_MODE"] = "ask"
        result["FX_MAX_AGENT_STEPS"] = "2"
        result["FX_AUTO_UPGRADE"] = "0"
        result["FX_HERDR"] = "0"
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


def _codex_canary(stdout: bytes, expected: str, model: str) -> tuple[bool, str | None]:
    exact = False
    for raw_line in stdout.splitlines():
        try:
            event = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(event, dict):
            continue
        item = event.get("item")
        if isinstance(item, dict) and item.get("type") == "agent_message":
            exact = str(item.get("text", "")).strip() == expected
    # Codex's JSONL contract pins the model in argv but does not currently emit
    # it in every event stream. The official client invocation is the receipt.
    return exact, model if exact else None


def _claude_canary(stdout: bytes, expected: str, model: str) -> tuple[bool, str | None]:
    try:
        payload = json.loads(stdout)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False, None
    if not isinstance(payload, dict):
        return False, None
    exact = str(payload.get("result", "")).strip() == expected
    usage = payload.get("modelUsage")
    reported = next(iter(usage), None) if isinstance(usage, dict) else None
    return exact, str(reported or model) if exact else None


def _fx_canary(stdout: bytes, expected: str, model: str) -> tuple[bool, str | None]:
    try:
        payload = json.loads(stdout)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False, None
    if not isinstance(payload, dict):
        return False, None
    reported = payload.get("model")
    exact = (
        payload.get("exit_code") == 0
        and str(payload.get("output", "")).strip() == expected
        and reported == model
        and payload.get("session_id") == ""
    )
    return exact, str(reported) if isinstance(reported, str) else None


def run_official_canaries(
    providers: list[str] | None = None, *, timeout_seconds: int = 120
) -> dict[str, Any]:
    selected = providers or list(DEFAULT_ROUTES)
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
            results.append(
                {
                    "provider": provider,
                    "model": route.model,
                    "status": "CLIENT_UNAVAILABLE",
                    "provenance_verified": False,
                }
            )
            continue
        if route.interactive_auth_only:
            started = time.monotonic_ns()
            completed = run_supervised(
                [executable, *route.command[1:]],
                input_data=None,
                cwd=Path.cwd(),
                env=_environment(provider),
                timeout_seconds=min(timeout_seconds, 30),
            )
            latency_ms = (time.monotonic_ns() - started) // 1_000_000
            version_matches = "2.6.9" in _strip_ansi(
                completed.stdout + completed.stderr
            )
            results.append(
                {
                    "provider": provider,
                    "model": route.model,
                    "reported_model": None,
                    "transport": "official_interactive_client",
                    "status": (
                        "INTERACTIVE_AUTH_REQUIRED"
                        if completed.returncode == 0 and version_matches
                        else "FAILED"
                    ),
                    "latency_ms": int(latency_ms),
                    "exit_code": completed.returncode,
                    "stdout_bytes": len(completed.stdout),
                    "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
                    "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
                    "provenance_verified": False,
                    "orchestrator_content_retained": False,
                }
            )
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
            exact, reported_model = _qwen_canary(
                completed.stdout, expected, route.model
            )
        elif route.response_format == "codex_jsonl" and completed.returncode == 0:
            exact, reported_model = _codex_canary(
                completed.stdout, expected, route.model
            )
        elif route.response_format == "claude_json" and completed.returncode == 0:
            exact, reported_model = _claude_canary(
                completed.stdout, expected, route.model
            )
        elif route.response_format == "fx_json" and completed.returncode == 0:
            exact, reported_model = _fx_canary(completed.stdout, expected, route.model)
        lower = combined.lower()
        quota = "usage limit" in lower or "quota" in lower
        missing_credential = (
            "missing api key" in lower
            or "no auth type" in lower
            or "api_key environment variable not found" in lower
        )
        status = (
            "WALL_TIMEOUT"
            if completed.timed_out
            else (
                "PASSED"
                if completed.returncode == 0 and exact
                else (
                    "QUOTA_UNAVAILABLE"
                    if quota
                    else ("CREDENTIAL_UNAVAILABLE" if missing_credential else "FAILED")
                )
            )
        )
        results.append(
            {
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
            }
        )
    statuses = {str(row["status"]) for row in results}
    overall = (
        "PASSED"
        if statuses == {"PASSED"}
        else (
            "DEGRADED_QUOTA"
            if statuses <= {"PASSED", "QUOTA_UNAVAILABLE"}
            else (
                "DEGRADED_UNAVAILABLE"
                if statuses
                <= {
                    "PASSED",
                    "QUOTA_UNAVAILABLE",
                    "CREDENTIAL_UNAVAILABLE",
                    "INTERACTIVE_AUTH_REQUIRED",
                }
                else "FAILED"
            )
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
