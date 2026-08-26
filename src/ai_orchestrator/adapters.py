from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_orchestrator.credentials import (
    deepseek_runtime_key,
    minimax_runtime_key,
    omniroute_runtime_key,
)
from ai_orchestrator.kimi_acp import AcpError, run_kimi_acp
from ai_orchestrator.models import (
    NATIVE_WRITER_PROFILES,
    RouteDecision,
    RunResult,
    TaskSpec,
    UsageRecord,
    ensure_private_directory,
)
from ai_orchestrator.output_contracts import validate_output
from ai_orchestrator.supervisor import ProcessOutcome, run_supervised
from ai_orchestrator.writer import (
    WorktreeTransaction,
    WriterError,
    gates_passed,
    repair_prompt,
    run_gates,
)


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
    if transport in {"omniroute", "qwen_omniroute_cli"}:
        allowed.update({"OMNIROUTE_API_KEY", "OMNIROUTE_BASE_URL", "OMNIROUTE_CONTEXT"})
    if transport == "qwen_minimax_cli":
        allowed.add("MINIMAX_API_KEY")
    if transport == "qwen_deepseek_cli":
        allowed.add("DEEPSEEK_API_KEY")
    if transport == "deepseek_harness":
        allowed.update(
            {
                "AI_ORCHESTRATOR_SOCKET",
                "AI_ORCHESTRATOR_HARNESS_CREDENTIAL",
                "DSH_HOME",
            }
        )
    if transport == "kimi_cli":
        allowed.update({"KIMI_CODE_HOME", "KIMI_LOG_LEVEL"})
    result = {key: value for key, value in os.environ.items() if key in allowed}
    if transport == "qwen_omniroute_cli":
        runtime_key = result.pop("OMNIROUTE_API_KEY", None) or omniroute_runtime_key(result)
        if runtime_key:
            result["OPENAI_API_KEY"] = runtime_key
        result["OPENAI_BASE_URL"] = result.pop(
            "OMNIROUTE_BASE_URL", "http://127.0.0.1:20128/v1"
        )
    elif transport == "qwen_deepseek_cli":
        runtime_key = deepseek_runtime_key(result)
        result.pop("DEEPSEEK_API_KEY", None)
        if runtime_key:
            result["OPENAI_API_KEY"] = runtime_key
        result["OPENAI_BASE_URL"] = "https://api.deepseek.com"
    elif transport == "deepseek_harness":
        # Harness receives only a loopback-scoped, revocable broker credential.
        # Provider keys are resolved by the bridge and never enter this process.
        result.pop("DEEPSEEK_API_KEY", None)
        result.pop("OPENAI_API_KEY", None)
        result.pop("OPENAI_BASE_URL", None)
    elif transport == "qwen_minimax_cli":
        runtime_key = minimax_runtime_key(result)
        result.pop("MINIMAX_API_KEY", None)
        # 2026-08-20: se pasa a la superficie COMPATIBLE CON OPENAI. La
        # configuracion Anthropic de abajo se verifico el 2026-08-12 y
        # funcionaba entonces, pero el 2026-08-19 una tarea real devolvio
        # output_bytes=0 en 5,6 s sin error de red (status MALFORMED_PATCH).
        # El Director verifico a mano con `qwen --auth-type openai` sobre
        # https://api.minimax.io/v1 que ese endpoint SI responde. Lo mas
        # probable es que MiniMax haya cambiado su superficie Anthropic.
        # Para revertir: restaurar auth_type "anthropic" aqui y en get_command.
        if runtime_key:
            result["OPENAI_API_KEY"] = runtime_key
        result["OPENAI_BASE_URL"] = "https://api.minimax.io/v1"
        # --- HISTORICO, desactivado el 2026-08-20 (ver arriba) ---
        # SIN sufijo /v1: el cliente Anthropic de Qwen lo anade el mismo. Con
        # ".../anthropic/v1" la peticion sale a /anthropic/v1/v1/messages y el
        # servidor devuelve 404 — que Qwen envuelve como subtype=success,
        # is_error=false y exit 0, con el texto "[API Error: 404 404 page not
        # found]" dentro. Verificado el 2026-08-12: /anthropic/v1/v1/messages
        # da 404 y /anthropic/v1/messages da 401, y con esta base el canario
        # devuelve el token exacto.
        # result["ANTHROPIC_BASE_URL"] = "https://api.minimax.io/anthropic"
    elif transport == "kimi_cli":
        result["KIMI_DISABLE_TELEMETRY"] = "1"
        result["KIMI_CODE_NO_AUTO_UPDATE"] = "1"
        result["KIMI_DISABLE_CRON"] = "1"
        result["KIMI_CODE_BACKGROUND_KEEP_ALIVE_ON_EXIT"] = "0"
        result["KIMI_CODE_BACKGROUND_MAX_RUNNING_TASKS"] = "1"
        result["KIMI_SUBAGENT_TIMEOUT_MS"] = "1"
        result["KIMI_CODE_BUILTIN_PRODUCT_SKILLS"] = "0"
        result["KIMI_LOG_LEVEL"] = "error"
    elif transport == "fx_cli":
        # OAuth remains in ~/.fx/auth.json. API keys are deliberately absent
        # from the allowlist, and noninteractive `ask` stops on every unresolved
        # sensitive tool request instead of invoking the paid auto-reviewer.
        result["FX_MODEL"] = "zai/glm-5.2"
        result["FX_PERMISSION_MODE"] = "ask"
        result["FX_MAX_AGENT_STEPS"] = "12"
        result["FX_AUTO_UPGRADE"] = "0"
        result["FX_HERDR"] = "0"
    result.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    result["PYTHONDONTWRITEBYTECODE"] = "1"
    return result


def _command(task: TaskSpec, decision: RouteDecision, prompt_path: Path) -> list[str]:
    writing = task.write_mode != "read_only"
    if decision.transport == "codex_cli":
        command = [
            "codex",
            "exec",
            "--json",
            "--ephemeral",
            "--ignore-user-config",
            "--sandbox",
            "workspace-write" if writing else "read-only",
            "--model",
            str(decision.requested_route_model),
            "-c",
            f'model_reasoning_effort="{decision.reasoning_effort or "high"}"',
            "-",
        ]
        return command
    if decision.transport == "claude_cli":
        command = [
            "claude",
            "--print",
            "--safe-mode",
            "--no-session-persistence",
            "--model",
            str(decision.requested_route_model),
            "--effort",
            str(decision.reasoning_effort or "high"),
            "--output-format",
            "json",
            "--tools",
            "Read,Edit,Write" if writing else "Read",
            "--permission-mode",
            "acceptEdits" if writing else "plan",
            "--disallowedTools",
            "Bash,Agent,WebFetch,WebSearch",
        ]
        return command
    if decision.transport in {
        "qwen_deepseek_cli",
        "qwen_omniroute_cli",
        "qwen_minimax_cli",
    }:
        # 2026-08-20: MiniMax pasa tambien a "openai" — su superficie Anthropic
        # dejo de responder (output_bytes=0). Antes: "openai" solo para omniroute.
        auth_type = "openai"
        # Guarded writers never receive a writable mount. They return a diff,
        # which the orchestrator validates and applies inside its worktree.
        guarded_patch = writing and decision.execution_profile == "patch_api_writer"
        return [
            "qwen",
            "--auth-type", auth_type,
            "--bare",
            "--output-format", "json",
            "--approval-mode", "plan" if guarded_patch or not writing else "auto-edit",
            "--exclude-tools", "run_shell_command,agent,web_fetch,web_search,skill",
            "--max-wall-time", f"{task.timeout_seconds}s",
            # 30, no 8. Con 8 solo caben los canarios: una tarea real de
            # analisis sobre el repo consumio 23 turnos de asistente y 10 de
            # usuario (medido el 2026-08-12 sobre spatial_hash de CHUNSA001).
            # Con 8 fallaba SIEMPRE con FatalTurnLimitedError codigo 53, que el
            # supervisor traduce a TURN_BUDGET_EXCEEDED — un limite que no deja
            # completar nada no acota el riesgo, solo lo esconde.
            # El techo real sigue siendo --max-wall-time y --max-tool-calls.
            "--max-session-turns", "30",
            "--max-tool-calls", "32",
            "--model",
            str(decision.requested_route_model),
        ]
    if decision.transport == "kimi_cli":
        return ["kimi", "acp"]
    if decision.transport == "deepseek_harness":
        return [
            sys.executable,
            "-m",
            "ai_orchestrator",
            "harness",
            "worker",
            "--model",
            str(decision.requested_route_model),
            "--timeout",
            str(task.timeout_seconds),
            "--orchestrated-worktree",
        ]
    if decision.transport == "fx_cli":
        if writing:
            raise AdapterError(
                "HANDOFF_REQUIRED", "FX promotional route is read-only"
            )
        return [
            "fx",
            "--no-additional-dirs",
            "--context-limit",
            "project_instructions_total_bytes=32768",
            "ask",
            "--json",
            "--no-save",
            "--no-color",
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


# Cada CLI ejecutor escribe su propio estado dentro del workspace. No es
# trabajo del modelo, y en el perfil chunsa cae ademas en rutas protegidas.
TOOL_STATE_DIRECTORIES = {
    "qwen_deepseek_cli": (".qwen",),
    "qwen_omniroute_cli": (".qwen",),
    "qwen_minimax_cli": (".qwen",),
    "codex_cli": (".codex",),
    "claude_cli": (".claude",),
}


def _qwen_sandbox_command(
    command: list[str], worktree: Path, *, writable: bool = True
) -> tuple[list[str], dict[str, str]]:
    """Wrap Qwen in a filesystem sandbox while preserving host networking."""

    bwrap = shutil.which("bwrap")
    qwen = shutil.which("qwen")
    if not bwrap or not qwen:
        raise AdapterError("SANDBOX_UNAVAILABLE", "bwrap and qwen are required")
    wrapper = Path(qwen).resolve()
    qwen_root_candidates: list[Path] = []
    try:
        launcher = wrapper.read_text(encoding="utf-8")[:4096]
        matched = re.search(r"['\"](/[^'\"]+/qwen-code-[^/'\"]+)/bin/qwen['\"]", launcher)
        if matched:
            qwen_root_candidates.append(Path(matched.group(1)))
    except (OSError, UnicodeError):
        pass
    qwen_root_candidates.extend((wrapper.parents[1] / "lib").glob("qwen-code-*"))
    if not qwen_root_candidates:
        # The managed installation keeps the versioned runtime alongside the
        # wrapper under ~/.local/lib.
        qwen_root_candidates = sorted(Path(qwen).resolve().parents[1].glob("lib/qwen-code-*"))
    if not qwen_root_candidates:
        qwen_root_candidates = sorted(Path.home().glob(".local/lib/qwen-code-*"))
    if not qwen_root_candidates:
        raise AdapterError("SANDBOX_UNAVAILABLE", "cannot locate the Qwen runtime")
    qwen_root = qwen_root_candidates[0].resolve()
    sandboxed = [
        bwrap,
        "--unshare-all",
        "--share-net",
        "--die-with-parent",
        "--new-session",
        "--ro-bind", "/", "/",
        "--tmpfs", "/home",
        "--tmpfs", "/tmp",
        "--tmpfs", "/opt",
        "--proc", "/proc",
        "--dev", "/dev",
        "--dir", "/opt/qwen",
        "--ro-bind", str(qwen_root), "/opt/qwen",
        "--dir", "/tmp/workspace",
        "--bind" if writable else "--ro-bind", str(worktree), "/tmp/workspace",
        "--dir", "/tmp/qwen-home",
        "--chdir", "/tmp/workspace",
        "--",
        "/opt/qwen/bin/qwen",
        *command[1:],
    ]
    return sandboxed, {
        "HOME": "/tmp/qwen-home",
        "XDG_CONFIG_HOME": "/tmp/qwen-home/.config",
        "XDG_DATA_HOME": "/tmp/qwen-home/.local/share",
        "XDG_CACHE_HOME": "/tmp/qwen-home/.cache",
    }


def _writer_prompt(task: TaskSpec, prompt: bytes, *, patch_only: bool = False) -> bytes:
    paths = ", ".join(task.allowed_write_paths)
    if patch_only:
        policy = (
            "Produce the requested code change as a git unified diff against the current "
            "worktree. Return only the diff starting with 'diff --git'; do not use Markdown "
            f"fences or explanatory text. The diff may modify only: {paths}. Do not run shell "
            "commands, start agents, use the web, change .git, or modify permission configuration. "
            "The orchestrator—not you—will validate and apply the diff in an isolated worktree, "
            "then run deterministic gates.\n\n"
        )
    else:
        # «Implement the requested change directly» resulto ambiguo: el
        # 2026-08-14 ds/deepseek-v4-pro hizo 13 `read_file`, cero `edit`, y
        # escribio las modificaciones en prosa («### Edit 1: Old: ... New:
        # ...»). Tenia la herramienta y el arbol escribible. Hay que decir que
        # aplique, que la prosa no cuenta, y que no explore: el transcrito de
        # un escritor nativo tiene techo, y leer de mas es lo que lo revienta.
        policy = (
            "You are editing an isolated Git worktree. Apply the change with your file "
            "editing tools (`edit`); do not describe it. Prose, a diff, or a summary of "
            "the change is NOT a deliverable — only the edited files count, and a reply "
            "that explains the edit without performing it counts as no work at all. "
            f"You may modify only these repository-relative paths: {paths}. "
            "Read only the specific ranges the task cites; do not explore the repository "
            "and never reproduce a whole file in your reply — your transcript has a size "
            "limit and exceeding it discards the run. "
            "Do not run shell commands, start agents, use the web, change .git, or modify permission "
            "configuration. The orchestrator will run deterministic gates after you finish.\n\n"
        )
    return policy.encode("utf-8") + prompt


def _run_once(
    task: TaskSpec,
    decision: RouteDecision,
    *,
    prompt: bytes,
    cwd: Path,
) -> tuple[ProcessOutcome, str | None, bool]:
    if task.write_mode != "read_only" and decision.execution_profile == "patch_api_writer":
        worker_environment = _minimal_environment(str(decision.transport))
        if decision.transport == "qwen_omniroute_cli":
            credential = omniroute_runtime_key()
            worker_environment.pop("OPENAI_API_KEY", None)
            if credential:
                worker_environment["OMNIROUTE_API_KEY"] = credential
        elif decision.transport == "qwen_deepseek_cli":
            credential = deepseek_runtime_key()
            worker_environment.pop("OPENAI_API_KEY", None)
            if credential:
                worker_environment["DEEPSEEK_API_KEY"] = credential
        elif decision.transport == "qwen_minimax_cli":
            credential = minimax_runtime_key()
            worker_environment.pop("ANTHROPIC_API_KEY", None)
            if credential:
                worker_environment["MINIMAX_API_KEY"] = credential
        worker_payload = json.dumps(
            {
                "task": task.as_dict(),
                "decision": decision.as_dict(),
                "worktree": str(cwd),
                "prompt_b64": base64.b64encode(prompt).decode("ascii"),
            },
            separators=(",", ":"),
        ).encode("utf-8")
        worker = run_supervised(
            [sys.executable, "-m", "ai_orchestrator.patch_worker"],
            input_data=worker_payload,
            cwd=cwd,
            env=worker_environment,
            timeout_seconds=task.timeout_seconds,
        )
        if worker.returncode != 0:
            return worker, None, False
        try:
            worker_result = json.loads(worker.stdout)
            patch = base64.b64decode(worker_result["patch_b64"], validate=True)
            reported_model = str(worker_result["reported_model"])
        except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            return ProcessOutcome(1, b"", b"malformed patch worker output"), None, False
        envelope = json.dumps(
            [
                {"type": "system", "model": reported_model},
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "result": patch.decode("utf-8"),
                },
            ],
            separators=(",", ":"),
        ).encode("utf-8")
        expected = str(decision.requested_route_model)
        provenance = reported_model in {
            expected,
            expected.split("/", 1)[-1],
            decision.selected_model,
        }
        return ProcessOutcome(0, envelope, b""), reported_model, provenance
    command = _command(task, decision, task.prompt_path() or cwd)
    environment = _minimal_environment(str(decision.transport))
    if decision.transport == "kimi_cli":
        executable = shutil.which("kimi")
        if not executable:
            return ProcessOutcome(127, b"", b"kimi unavailable"), None, False
        try:
            outcome = run_kimi_acp(
                executable=executable,
                worktree=cwd,
                allowed_write_paths=task.allowed_write_paths,
                environment=environment,
                model=str(decision.requested_route_model),
                prompt=prompt,
                timeout_seconds=task.timeout_seconds,
            )
            return ProcessOutcome(0, outcome.stdout, outcome.stderr), outcome.reported_model, True
        except AcpError as exc:
            return ProcessOutcome(1, b"", str(exc).encode("utf-8")), None, False
    if decision.transport in {
        "qwen_deepseek_cli",
        "qwen_omniroute_cli",
        "qwen_minimax_cli",
    }:
        # El montaje sigue al perfil. Estaba cableado a False, lo que convertia
        # el parametro en codigo muerto: un escritor nativo recibia el arbol en
        # solo-lectura y no podia escribir lo que --approval-mode auto-edit le
        # decia que escribiera. La contencion no es el montaje, es
        # collect_patch: arbol desechable, allowlist, .git y symlinks vetados.
        writable = (
            task.write_mode != "read_only"
            and decision.execution_profile in NATIVE_WRITER_PROFILES
        )
        command, sandbox_environment = _qwen_sandbox_command(
            command, cwd, writable=writable
        )
        environment.update(sandbox_environment)
    outcome = run_supervised(
        command,
        input_data=prompt if decision.transport != "omniroute" else None,
        cwd=cwd,
        env=environment,
        timeout_seconds=task.timeout_seconds,
    )
    reported, provenance = _reported_model(outcome.stdout, decision)
    return outcome, reported, provenance


def _process_status(outcome: ProcessOutcome, decision: RouteDecision) -> tuple[str, str | None]:
    if outcome.timed_out:
        return "WALL_TIMEOUT", "WALL_TIMEOUT"
    if outcome.returncode != 0:
        combined = outcome.stdout + b"\n" + outcome.stderr
        if outcome.returncode == 55 and b"wall-clock budget" in combined.lower():
            return "WALL_TIMEOUT", "WALL_TIMEOUT"
        if outcome.returncode == 55:
            return "BUDGET_EXCEEDED", "BUDGET_EXCEEDED"
        if outcome.returncode == 53:
            return "TURN_BUDGET_EXCEEDED", "TURN_BUDGET_EXCEEDED"
        if (
            b"missing api key" in combined.lower()
            or b"no auth type" in combined.lower()
            or b"api_key environment variable not found" in combined.lower()
            or b"authrequired" in combined.lower()
        ):
            return "CREDENTIAL_UNAVAILABLE", "CREDENTIAL_UNAVAILABLE"
        if b"quota" in combined.lower() or b"usage limit" in combined.lower():
            return "QUOTA_UNAVAILABLE", "QUOTA_UNAVAILABLE"
        lowered = combined.lower()
        if b"insufficient balance" in lowered or b"billing limit" in lowered:
            return "QUOTA_UNAVAILABLE", "QUOTA_UNAVAILABLE"
        if any(
            marker in lowered
            for marker in (b"overloaded", b"temporarily unavailable", b"capacity", b"rate limit")
        ):
            return "OVERLOADED", "OVERLOADED"
        if any(
            marker in lowered
            for marker in (b"connection reset", b"connection refused", b"network unreachable", b"temporary failure in name resolution")
        ):
            return "NETWORK_ERROR", "NETWORK_ERROR"
        # patch_worker writes a PatchApiError code on stderr and nothing else.
        # It is a fixed literal from a closed set, so trusting it leaks no
        # provider text, and it is the only thing that separates a 429 from a
        # dead DNS. The substring heuristics above can never reach it: they look
        # for provider prose that deliberately never crosses that boundary.
        code = outcome.stderr.strip().decode("ascii", "ignore")
        if code and code.isupper() and code.replace("_", "").isalnum():
            transient_status = {
                "PROVIDER_SERVER_ERROR": "TRANSIENT_PROVIDER_ERROR",
                "PROVIDER_UNREACHABLE": "NETWORK_ERROR",
                "PROVIDER_HTTP_408": "TRANSIENT_PROVIDER_ERROR",
            }.get(code)
            if transient_status is not None:
                return transient_status, code
            return "FAILED", code
        return "FAILED", "MODEL_PROCESS_FAILED"
    if decision.transport in {
        "qwen_deepseek_cli",
        "qwen_omniroute_cli",
        "qwen_minimax_cli",
    }:
        error = _qwen_error(outcome.stdout, outcome.stderr)
        return ("SUCCESS", None) if error is None else (error, error)
    return "SUCCESS", None


def _find_reported_model(payload: Any) -> str | None:
    if isinstance(payload, dict):
        for key in ("model", "reported_model", "requestedModel", "requested_model"):
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
    if decision.transport == "codex_cli":
        # Codex route IDs are concrete policy IDs, not floating family aliases.
        return decision.selected_model, True
    try:
        payload = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, False
    reported = _find_reported_model(payload)
    if not reported:
        return None, False
    expected = str(decision.requested_route_model)
    if decision.transport == "claude_cli":
        alias = expected.lower()
        concrete = reported.lower()
        verified = (
            concrete != alias
            and concrete != str(decision.selected_model).lower()
            and concrete.startswith(f"claude-{alias}-")
        )
        return reported, verified
    return reported, reported in {expected, expected.split("/", 1)[-1], decision.selected_model}


def _deliverable(stdout: bytes, decision: RouteDecision) -> bytes:
    """Extract assistant content from provider envelopes, never their metadata."""

    if decision.transport == "codex_cli":
        rows: list[Any] = []
        for line in stdout.splitlines():
            try:
                rows.append(json.loads(line))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        payload: Any = rows
    else:
        try:
            payload = json.loads(stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return stdout
    found = _find_deliverable(payload)
    return found.encode("utf-8") if found is not None else stdout


def _find_deliverable(payload: Any) -> str | None:
    if isinstance(payload, dict):
        for key in ("result", "output_text", "text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        item = payload.get("item")
        if isinstance(item, dict) and item.get("type") in {
            "agent_message",
            "assistant_message",
        }:
            value = item.get("text")
            if isinstance(value, str) and value.strip():
                return value
        content = payload.get("content")
        if isinstance(content, list):
            for row in reversed(content):
                found = _find_deliverable(row)
                if found:
                    return found
        for value in reversed(list(payload.values())):
            found = _find_deliverable(value)
            if found:
                return found
    elif isinstance(payload, list):
        for value in reversed(payload):
            found = _find_deliverable(value)
            if found:
                return found
    return None


def _usage_record(
    stdout: bytes,
    decision: RouteDecision,
    *,
    outcome: str,
    latency_ms: int,
    provenance_verified: bool,
) -> UsageRecord:
    try:
        if decision.transport == "codex_cli":
            payload: Any = [json.loads(line) for line in stdout.splitlines() if line.strip()]
        else:
            payload = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = None
    usage = _find_usage(payload) or {}
    details = usage.get("input_tokens_details")
    if not isinstance(details, dict):
        details = {}
    return UsageRecord(
        provider=str(decision.provider),
        model=str(decision.selected_model),
        outcome=outcome,
        latency_ms=latency_ms,
        input_tokens=_optional_int(usage.get("input_tokens")),
        output_tokens=_optional_int(usage.get("output_tokens")),
        cached_tokens=_optional_int(
            usage.get("cache_read_input_tokens", details.get("cached_tokens"))
        ),
        cache_write_tokens=_optional_int(usage.get("cache_creation_input_tokens")),
        credits_used=_optional_float(usage.get("credits_used")),
        balance_before=_optional_float(usage.get("balance_before")),
        balance_after=_optional_float(usage.get("balance_after")),
        provenance_verified=provenance_verified,
        route_plan_id=decision.plan_id,
    )


def _find_usage(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        usage = payload.get("usage")
        if isinstance(usage, dict):
            return usage
        for value in payload.values():
            found = _find_usage(value)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in reversed(payload):
            found = _find_usage(value)
            if found is not None:
                return found
    return None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _qwen_error(stdout: bytes, stderr: bytes) -> str | None:
    """Validate Qwen's buffered JSON result envelope.

    Qwen may emit a result object even for upstream failures, so a zero process
    exit code alone is not a success signal.
    """

    try:
        payload = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        # Un transcrito CORTADO no es una salida malformada, y confundirlos
        # manda a buscar el fallo al sitio equivocado. Medido el 2026-08-14
        # sobre step.hpp: el CLI devolvio 65536 bytes exactos, partidos a mitad
        # de una cadena. La firma es empezar como array y no cerrarlo; no se
        # comprueba el 65536 porque el tope es de la herramienta y puede mover.
        trimmed = stdout.strip()
        if trimmed.startswith(b"[") and not trimmed.endswith(b"]"):
            return "OUTPUT_TRUNCATED"
        return "MALFORMED_OUTPUT"
    if not isinstance(payload, list):
        return "MALFORMED_OUTPUT"
    results = [row for row in payload if isinstance(row, dict) and row.get("type") == "result"]
    if not results:
        return "MALFORMED_OUTPUT"
    result = results[-1]
    if result.get("is_error") is True or result.get("subtype") != "success":
        combined = stdout + b"\n" + stderr
        if b"quota" in combined.lower() or b"usage limit" in combined.lower():
            return "QUOTA_UNAVAILABLE"
        return "PROVIDER_ERROR"
    final_text = result.get("result")
    if not isinstance(final_text, str) or not final_text.strip():
        return "MALFORMED_OUTPUT"
    if re.match(r"^\[(?:API )?Error:", final_text.strip(), re.IGNORECASE):
        return "PROVIDER_ERROR"
    if isinstance(result.get("error"), dict):
        return "PROVIDER_ERROR"
    return None


def _qwen_result_text(stdout: bytes) -> bytes:
    try:
        payload = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WriterError("MALFORMED_PATCH", "Qwen did not return JSON") from exc
    if not isinstance(payload, list):
        raise WriterError("MALFORMED_PATCH", "Qwen result envelope is not a list")
    results = [row for row in payload if isinstance(row, dict) and row.get("type") == "result"]
    final = results[-1].get("result") if results else None
    if not isinstance(final, str):
        raise WriterError("MALFORMED_PATCH", "Qwen result has no textual patch")
    patch = final.strip()
    if not patch.startswith("diff --git ") or "```" in patch:
        raise WriterError("MALFORMED_PATCH", "Qwen result is not a bare git unified diff")
    return (patch + "\n").encode("utf-8")


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


def write_output_artifact(task: TaskSpec, output_path: Path, data: bytes) -> None:
    """Publish an accepted artifact outside the source repository, atomically."""

    resolved_output = output_path.expanduser().resolve()
    if resolved_output.is_relative_to(Path(task.repo).resolve()):
        raise AdapterError(
            "OUTPUT_IN_REPO_FORBIDDEN", "model output must remain in staging"
        )
    _atomic_private_write(resolved_output, data)


def execute(
    task: TaskSpec,
    decision: RouteDecision,
    *,
    prompt_override: bytes | None = None,
    output_path: Path | None = None,
    gate_commands: tuple[str, ...] = (),
    gate_timeout_seconds: int | None = None,
    protected_prefixes: tuple[str, ...] = (),
) -> tuple[RunResult, bytes]:
    if not decision.auto_execute or decision.status != "READY":
        raise AdapterError("HANDOFF_REQUIRED", decision.reason)
    prompt_path = task.prompt_path()
    if prompt_path is None:
        raise AdapterError("PROMPT_REQUIRED", "run requires prompt_file")

    prompt = prompt_override if prompt_override is not None else prompt_path.read_bytes()
    started = datetime.now(UTC)
    start_ns = time.monotonic_ns()
    stdout = b""
    patch_sha256: str | None = None
    changed_files = 0
    repairs = 0
    gate_state: bool | None = None
    error_detail: str | None = None
    reported: str | None = None
    provenance = False
    exit_code: int | None = None
    usage_envelope = b""
    contract_validated: bool | None = None
    gate_records: list[dict[str, Any]] = []

    if task.write_mode == "read_only":
        outcome, reported, provenance = _run_once(
            task, decision, prompt=prompt, cwd=Path(task.repo)
        )
        exit_code = outcome.returncode
        usage_envelope = outcome.stdout
        stdout = outcome.stdout
        status, error_code = _process_status(outcome, decision)
        if status == "SUCCESS":
            stdout = _deliverable(outcome.stdout, decision)
            validation = validate_output(task, stdout)
            contract_validated = validation.valid
            if not validation.valid:
                status, error_code = "CONTRACT_INVALID", validation.error_code
    else:
        try:
            with WorktreeTransaction(task) as transaction:
                assert transaction.worktree is not None
                patch_transport = decision.execution_profile == "patch_api_writer"
                writer_input = _writer_prompt(task, prompt, patch_only=patch_transport)
                while True:
                    # Antes de cada pasada del escritor, no una sola vez: los
                    # gates de la vuelta anterior dejan artefactos ignorados y
                    # no son cosa del modelo.
                    transaction.snapshot_ignored()
                    outcome, reported, provenance = _run_once(
                        task,
                        decision,
                        prompt=writer_input,
                        cwd=transaction.worktree,
                    )
                    exit_code = outcome.returncode
                    usage_envelope = outcome.stdout
                    process_status, process_error = _process_status(outcome, decision)
                    if process_status != "SUCCESS":
                        status, error_code = process_status, process_error
                        stdout = outcome.stdout
                        # Para un escritor NATIVO el arbol es el entregable y la
                        # envoltura es charla: si el CLI se corta pero los
                        # ficheros ya estan editados, tirarlos sin mirarlos
                        # pierde trabajo hecho. Para un escritor por parche no
                        # aplica —su entregable ES la envoltura—, y ahi romper
                        # es lo correcto. El rescate NUNCA se salta los gates ni
                        # la validacion de alcance, y sale con un codigo propio
                        # para que no se confunda con un exito limpio.
                        if decision.execution_profile not in NATIVE_WRITER_PROFILES:
                            break
                        transaction.discard_tool_state(
                            TOOL_STATE_DIRECTORIES.get(str(decision.transport), ())
                        )
                        salvaged = transaction.collect_patch(protected_prefixes)
                        if not salvaged.patch:
                            break
                        stdout = salvaged.patch
                        patch_sha256 = salvaged.sha256
                        changed_files = len(salvaged.paths)
                        validation = validate_output(task, salvaged.patch)
                        contract_validated = validation.valid
                        if not validation.valid:
                            status, error_code = (
                                "CONTRACT_INVALID",
                                validation.error_code,
                            )
                            break
                        gate_outcomes = run_gates(
                            transaction.worktree,
                            gate_commands,
                            gate_timeout_seconds or task.timeout_seconds,
                        )
                        gate_records.extend(_gate_records(gate_outcomes))
                        gate_state = gates_passed(gate_outcomes, len(gate_commands))
                        if gate_state:
                            status, error_code = "PATCH_READY", "WRITER_OUTPUT_UNUSABLE"
                        break
                    if not provenance:
                        status, error_code = "PROVENANCE_UNVERIFIED", "PROVENANCE_UNVERIFIED"
                        stdout = outcome.stdout
                        break

                    if patch_transport:
                        transaction.apply_candidate_patch(_qwen_result_text(outcome.stdout))

                    transaction.discard_tool_state(
                        TOOL_STATE_DIRECTORIES.get(str(decision.transport), ())
                    )
                    patch = transaction.collect_patch(protected_prefixes)
                    stdout = patch.patch
                    patch_sha256 = patch.sha256
                    changed_files = len(patch.paths)
                    if not patch.patch:
                        status, error_code = "NO_CHANGES", "NO_CHANGES"
                        break

                    validation = validate_output(task, patch.patch)
                    contract_validated = validation.valid
                    if not validation.valid:
                        status, error_code = "CONTRACT_INVALID", validation.error_code
                        break

                    gate_outcomes = run_gates(
                        transaction.worktree,
                        gate_commands,
                        gate_timeout_seconds or task.timeout_seconds,
                    )
                    gate_records.extend(_gate_records(gate_outcomes))
                    if gates_passed(gate_outcomes, len(gate_commands)):
                        gate_state = True
                        native_writer = decision.execution_profile == "native_writer"
                        if (
                            task.write_mode == "validated_apply"
                            and native_writer
                            and transaction.can_apply()
                        ):
                            transaction.apply(patch.patch)
                            status, error_code = "APPLIED", None
                        else:
                            if task.write_mode == "validated_apply" and not native_writer:
                                error_code = "EXTERNAL_WRITER_REVIEW_REQUIRED"
                            elif task.write_mode == "validated_apply" and not transaction.can_apply():
                                error_code = "DIRTY_WORKTREE"
                            else:
                                error_code = None
                            status = "PATCH_READY"
                        break
                    if not gate_commands:
                        gate_state = None
                        status, error_code = "PATCH_READY", "GATES_REQUIRED_FOR_APPLY"
                        break
                    gate_state = False
                    if repairs >= task.max_repairs:
                        status, error_code = "GATE_FAILED", "GATE_FAILED"
                        break
                    repairs += 1
                    writer_input = _writer_prompt(
                        task,
                        repair_prompt(prompt, gate_outcomes),
                        patch_only=patch_transport,
                    )
        except WriterError as exc:
            status, error_code = exc.code, exc.code
            # El mensaje lo redacta este codigo, no el modelo: lleva la ruta
            # ofensora o el limite superado, y nada mas.
            error_detail = str(exc)[:200] or None

    finished = datetime.now(UTC)
    latency_ms = (time.monotonic_ns() - start_ns) // 1_000_000
    if status == "SUCCESS" and not provenance:
        status = "PROVENANCE_UNVERIFIED"
        error_code = "PROVENANCE_UNVERIFIED"

    digest = hashlib.sha256(stdout).hexdigest() if stdout else None
    if output_path is not None and stdout:
        write_output_artifact(task, output_path, stdout)

    result = RunResult(
        schema_version=2,
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
        error_detail=error_detail,
        changed_files=changed_files,
        patch_sha256=patch_sha256,
        repairs=repairs,
        gates_passed=gate_state,
        gate_results=tuple(gate_records),
        contract_validated=contract_validated,
        usage=_usage_record(
            usage_envelope,
            decision,
            outcome=status,
            latency_ms=int(latency_ms),
            provenance_verified=provenance,
        ),
    )
    return result, stdout


def _gate_records(outcomes: tuple[Any, ...]) -> tuple[dict[str, Any], ...]:
    """Keep gate status metadata while dropping potentially sensitive output."""

    return tuple(
        {
            "command": outcome.command,
            "returncode": outcome.returncode,
            "timed_out": outcome.timed_out,
        }
        for outcome in outcomes
    )
