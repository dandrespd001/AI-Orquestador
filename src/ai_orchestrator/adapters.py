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

from ai_orchestrator.abacus import AbacusDelegator
from ai_orchestrator.credentials import minimax_runtime_key, omniroute_runtime_key
from ai_orchestrator.kimi_acp import AcpError, run_kimi_acp
from ai_orchestrator.models import (
    NATIVE_WRITER_PROFILES,
    ContractError,
    RouteDecision,
    RunResult,
    TaskSpec,
    ensure_private_directory,
)
from ai_orchestrator.provider_adapters import require_adapter
from ai_orchestrator.supervisor import ProcessOutcome, run_supervised
from ai_orchestrator.writer import (
    WorktreeTransaction,
    WriterError,
    gates_passed,
    patch_repair_prompt,
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
    if transport == "kimi_cli":
        allowed.update({"KIMI_CODE_HOME", "KIMI_LOG_LEVEL"})
    result = {key: value for key, value in os.environ.items() if key in allowed}
    if transport == "qwen_omniroute_cli":
        runtime_key = result.pop("OMNIROUTE_API_KEY", None) or omniroute_runtime_key(
            result
        )
        if runtime_key:
            result["OPENAI_API_KEY"] = runtime_key
        result["OPENAI_BASE_URL"] = result.pop(
            "OMNIROUTE_BASE_URL", "http://127.0.0.1:20128/v1"
        )
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
            "Bash,Agent" if decision.web_access else "Bash,Agent,WebFetch,WebSearch",
        ]
        if decision.web_access:
            # La lectura de la web se declara ademas en --tools: quitarla del
            # bloqueo no basta para que el CLI la ofrezca.
            index = command.index("--tools") + 1
            command[index] = f"{command[index]},WebFetch,WebSearch"
        return command
    if decision.transport in {"qwen_omniroute_cli", "qwen_minimax_cli"}:
        # --auth-type VUELVE, y no era opcional. Se quito creyendo que Qwen Code
        # 0.21.15 lo habia retirado; 0.22.1 lo registra
        # (choices: openai, anthropic, qwen-oauth, gemini, vertex-ai) y sin el
        # muere en no interactivo con "No auth type is selected". Poner
        # OPENAI_API_KEY/OPENAI_BASE_URL no basta: el CLI exige que la seleccion
        # sea explicita. Verificado el 2026-08-25 contra el binario instalado.
        # --safe-mode evita configuracion, hooks, extensiones, skills y MCP del
        # usuario. Los controles headless siguen aceptados aunque la ayuda corta
        # ya no los enumere; se conservan junto al limite externo del supervisor.
        guarded_patch = writing and decision.execution_profile == "patch_api_writer"
        return [
            "qwen",
            "--auth-type",
            "openai",
            "--safe-mode",
            "--output-format",
            "json",
            "--approval-mode",
            _qwen_approval_mode(decision, writing=writing, guarded=guarded_patch),
            "--exclude-tools",
            _qwen_excluded_tools(decision, writing=writing),
            "--max-wall-time",
            f"{task.timeout_seconds}s",
            # 100 y 200, no 30 y 32. Misma leccion que subio los turnos de 8 a
            # 30 el 2026-08-12, ahora medida sobre las llamadas: el 2026-08-25
            # dos encargos reales de CHUNSA murieron con exit 55 y CERO bytes,
            # uno de ellos tras haber escrito ya los ficheros. Navegar un
            # cabecero de 2000 lineas para hacer una edicion cuesta varias
            # llamadas, y 32 no llegan a un encargo de cuatro ficheros. Un
            # limite que no deja completar nada no acota el riesgo: lo esconde,
            # y encima disfraza el fallo de saldo agotado porque con
            # --output-format json morir por limite da 0 bytes.
            # El techo real sigue siendo --max-wall-time, que corta por reloj y
            # no depende de lo grande que sea el arbol.
            "--max-session-turns",
            "100",
            "--max-tool-calls",
            "200",
            "--model",
            str(decision.requested_route_model),
        ]
    if decision.transport == "kimi_cli":
        return ["kimi", "acp"]
    if decision.transport == "fx_cli":
        if writing:
            raise AdapterError("HANDOFF_REQUIRED", "FX promotional route is read-only")
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
    raise AdapterError(
        "HANDOFF_REQUIRED", f"unsupported supervised transport: {decision.transport}"
    )


# Cada CLI ejecutor escribe su propio estado dentro del workspace. No es
# trabajo del modelo, y en el perfil chunsa cae ademas en rutas protegidas.
TOOL_STATE_DIRECTORIES = {
    "qwen_omniroute_cli": (".qwen",),
    "qwen_minimax_cli": (".qwen",),
    "codex_cli": (".codex",),
    "claude_cli": (".claude",),
}


_QWEN_ALWAYS_EXCLUDED = ("run_shell_command", "agent", "skill")
_QWEN_WEB_TOOLS = ("web_fetch", "web_search")
_QWEN_WRITE_TOOLS = ("edit", "write_file", "notebook_edit")


def _qwen_excluded_tools(decision: RouteDecision, *, writing: bool) -> str:
    """La contencion vive aqui, no en el modo de aprobacion.

    `--approval-mode auto` prometia lo mejor de los dos mundos y resulto ser una
    moneda al aire: medido el 2026-08-25 con dos canarios seguidos, la MISMA
    clase de llamada -- `web_fetch` sobre raw.githubusercontent.com -- salio
    `auto_accept: 1` en uno y `auto_accept: 0` en el siguiente. Un clasificador
    LLM no da una contencion reproducible.

    Asi que en lectura con web se usa `yolo`, que es determinista, y lo que se
    quita se quita por nombre: ademas de shell, agente y skills, tambien las
    tres de escritura. Medido: el registro se queda en 21 herramientas sin
    ninguna de escritura, y `web_fetch` se auto-acepta siempre.
    """

    excluded = list(_QWEN_ALWAYS_EXCLUDED)
    if not decision.web_access:
        excluded.extend(_QWEN_WEB_TOOLS)
    elif not writing:
        # `yolo` registra edit/write_file/notebook_edit que `plan` mantenia
        # fuera. Se retiran por nombre para no perder esa capa al ganar la web.
        excluded.extend(_QWEN_WRITE_TOOLS)
    return ",".join(excluded)


def _qwen_approval_mode(
    decision: RouteDecision, *, writing: bool, guarded: bool
) -> str:
    """`plan` deniega la web en silencio: hay que decirlo por su nombre.

    Medido el 2026-08-25 con un canario a MiniMax-M3: `web_fetch` SI aparece en
    el registro de herramientas, pero la llamada vuelve con "Qwen Code requires
    permission to use 'web_fetch', but that permission was declined
    (non-interactive mode cannot prompt for confirmation)". Es una boundary tool
    que pide confirmacion, y en headless la confirmacion se auto-deniega. Abrir
    `web_access` sin tocar esto deja el rol de investigacion leyendo solo el
    repositorio, que es justo lo que no es investigar.

    La apertura se acota a lo minimo: SOLO lectura y SOLO con web concedida. Lo
    que contiene la tarea no es este modo, son las otras tres capas, y ninguna
    se toca aqui: bwrap monta el arbol con --ro-bind, `--exclude-tools` retira
    run_shell_command, agent y skill, y HOME es un tmpfs vacio. Una tarea de
    escritura NUNCA llega a esta rama.
    """

    if writing:
        return "plan" if guarded else "auto-edit"
    return "yolo" if decision.web_access else "plan"


def _qwen_runtime(launcher: Path) -> tuple[Path, str]:
    """Localiza la raiz versionada de Qwen y su punto de entrada real.

    El punto de entrada NO se puede cablear a `bin/qwen`. Qwen Code 0.22.1
    cambio el reparto de la instalacion: ahora `~/.local/bin/qwen` apunta a
    `node_modules/.bin/qwen`, que a su vez apunta a
    `node_modules/@qwen-code/qwen-code/cli-entry.js`, y ya no existe ningun
    `bin/qwen`. El sandbox seguia ejecutando la ruta antigua y bwrap moria con
    `execvp /opt/qwen/bin/qwen: No such file or directory` -- returncode 1, cero
    bytes y dos segundos, que por fuera es indistinguible de un proveedor caido.

    Se deriva del ejecutable ya resuelto, que es la unica fuente que no depende
    de como reparta los ficheros la version de turno.
    """

    target = launcher.resolve()
    for parent in target.parents:
        if parent.name.startswith("qwen-code-"):
            return parent, "/opt/qwen/" + target.relative_to(parent).as_posix()
    candidates = sorted(Path.home().glob(".local/lib/qwen-code-*"))
    if not candidates:
        raise AdapterError("SANDBOX_UNAVAILABLE", "cannot locate the Qwen runtime")
    root = candidates[-1].resolve()
    try:
        return root, "/opt/qwen/" + target.relative_to(root).as_posix()
    except ValueError:
        return root, "/opt/qwen/bin/qwen"


def _qwen_sandbox_command(
    command: list[str], worktree: Path, *, writable: bool = True
) -> tuple[list[str], dict[str, str]]:
    """Wrap Qwen in a filesystem sandbox while preserving host networking."""

    bwrap = shutil.which("bwrap")
    qwen = shutil.which("qwen")
    if not bwrap or not qwen:
        raise AdapterError("SANDBOX_UNAVAILABLE", "bwrap and qwen are required")
    qwen_root, entrypoint = _qwen_runtime(Path(qwen))
    sandboxed = [
        bwrap,
        "--unshare-all",
        "--share-net",
        "--die-with-parent",
        "--new-session",
        "--ro-bind",
        "/",
        "/",
        "--tmpfs",
        "/home",
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        "/opt",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--dir",
        "/opt/qwen",
        "--ro-bind",
        str(qwen_root),
        "/opt/qwen",
        "--dir",
        "/tmp/workspace",
        "--bind" if writable else "--ro-bind",
        str(worktree),
        "/tmp/workspace",
        "--dir",
        "/tmp/qwen-home",
        "--chdir",
        "/tmp/workspace",
        "--",
        entrypoint,
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


# Abacus corre en un directorio vacio desechable con --disallowed-tools "*",
# asi que nunca ve el repositorio. Por eso models.toml lo mantenia fuera de
# roles.toml: enrutarlo sin mas producia HANDOFF_REQUIRED. Para que la
# suscripcion sirva de algo en el enrutador, el contexto viaja dentro del
# prompt en lugar de esperar que el modelo lea ficheros.
MAX_ABACUS_CONTEXT_BYTES = 60_000


def _abacus_context(task: TaskSpec, prompt: bytes) -> str:
    repo = Path(task.repo).resolve()
    parts = [prompt.decode("utf-8", errors="replace")]
    budget = MAX_ABACUS_CONTEXT_BYTES
    for relative in task.input_paths:
        candidate = (repo / relative).resolve()
        if not candidate.is_relative_to(repo) or not candidate.is_file():
            continue
        try:
            body = candidate.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            # Un binario no se puede inlinear; decirlo es mejor que omitirlo.
            parts.append(f"\n\n--- {relative} (binary, not inlined) ---")
            continue
        chunk = body[:budget]
        budget -= len(chunk)
        truncated = " (truncated)" if len(chunk) < len(body) else ""
        parts.append(f"\n\n--- {relative}{truncated} ---\n{chunk}")
        if budget <= 0:
            break
    return "".join(parts)


def _run_abacus(
    task: TaskSpec,
    decision: RouteDecision,
    *,
    prompt: bytes,
    cwd: Path,
) -> tuple[ProcessOutcome, str | None, bool]:
    if task.write_mode != "read_only":
        # Los modelos de Abacus declaran capabilities = ["read"], asi que el
        # enrutador ya los descarta antes de llegar aqui. Esto es el cinturon.
        raise AdapterError(
            "HANDOFF_REQUIRED", "Abacus routes are read-only by contract"
        )
    requested = str(decision.requested_route_model)
    try:
        payload = AbacusDelegator().delegate(
            prompt=_abacus_context(task, prompt),
            task_type=str(task.task_type),
            model=requested,
            sensitivity=str(task.sensitivity),
            timeout_seconds=max(10, min(600, int(task.timeout_seconds))),
        )
    except (ContractError, RuntimeError) as exc:
        return ProcessOutcome(1, b"", str(exc).encode("utf-8")), None, False
    reported = str(payload.get("model") or "")
    return (
        ProcessOutcome(0, str(payload.get("result", "")).encode("utf-8"), b""),
        reported,
        reported == requested,
    )


def _run_once(
    task: TaskSpec,
    decision: RouteDecision,
    *,
    prompt: bytes,
    cwd: Path,
) -> tuple[ProcessOutcome, str | None, bool]:
    require_adapter(str(decision.provider), str(decision.transport))
    if (
        task.write_mode != "read_only"
        and decision.execution_profile == "patch_api_writer"
    ):
        worker_environment = _minimal_environment(str(decision.transport))
        if decision.transport == "qwen_omniroute_cli":
            credential = omniroute_runtime_key()
            worker_environment.pop("OPENAI_API_KEY", None)
            if credential:
                worker_environment["OMNIROUTE_API_KEY"] = credential
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
        except (
            KeyError,
            TypeError,
            ValueError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ):
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
    if decision.transport == "abacus_cli":
        return _run_abacus(task, decision, prompt=prompt, cwd=cwd)
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
                web_access=decision.web_access,
            )
            return (
                ProcessOutcome(0, outcome.stdout, outcome.stderr),
                outcome.reported_model,
                True,
            )
        except AcpError as exc:
            return ProcessOutcome(1, b"", str(exc).encode("utf-8")), None, False
    if decision.transport in {"qwen_omniroute_cli", "qwen_minimax_cli"}:
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


def _process_status(
    outcome: ProcessOutcome, decision: RouteDecision
) -> tuple[str, str | None]:
    if outcome.output_limited:
        return "OUTPUT_LIMIT_EXCEEDED", "OUTPUT_LIMIT_EXCEEDED"
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
        # patch_worker writes a PatchApiError code on stderr and nothing else.
        # It is a fixed literal from a closed set, so trusting it leaks no
        # provider text, and it is the only thing that separates a 429 from a
        # dead DNS. The substring heuristics above can never reach it: they look
        # for provider prose that deliberately never crosses that boundary.
        code = outcome.stderr.strip().decode("ascii", "ignore")
        if code and code.isupper() and code.replace("_", "").isalnum():
            return "FAILED", code
        return "FAILED", "MODEL_PROCESS_FAILED"
    if decision.transport in {"qwen_omniroute_cli", "qwen_minimax_cli"}:
        error = _qwen_error(outcome.stdout, outcome.stderr)
        return ("SUCCESS", None) if error is None else (error, error)
    return "SUCCESS", None


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
    if decision.transport in {"codex_cli", "claude_cli"}:
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
    return reported, reported in {
        expected,
        expected.split("/", 1)[-1],
        decision.selected_model,
    }


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
    results = [
        row for row in payload if isinstance(row, dict) and row.get("type") == "result"
    ]
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
    results = [
        row for row in payload if isinstance(row, dict) and row.get("type") == "result"
    ]
    final = results[-1].get("result") if results else None
    if not isinstance(final, str):
        raise WriterError("MALFORMED_PATCH", "Qwen result has no textual patch")
    patch = final.strip()
    if not patch.startswith("diff --git ") or "```" in patch:
        raise WriterError(
            "MALFORMED_PATCH", "Qwen result is not a bare git unified diff"
        )
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


def _read_bounded_prompt(path: Path, max_context_tokens: int) -> bytes:
    """Read at most the deterministic UTF-8 budget used by routing.

    Four bytes per token is an intentionally simple upper estimate. Provider
    tokenizers remain authoritative for usage accounting, but no transport may
    receive an unbounded prompt while advertising a context budget.
    """

    max_bytes = max_context_tokens * 4
    with path.open("rb") as source:
        prompt = source.read(max_bytes + 1)
    if len(prompt) > max_bytes:
        raise AdapterError(
            "CONTEXT_BUDGET_EXCEEDED",
            f"prompt exceeds the configured {max_context_tokens}-token budget",
        )
    return prompt


def execute(
    task: TaskSpec,
    decision: RouteDecision,
    *,
    output_path: Path | None = None,
    gate_commands: tuple[str, ...] = (),
    gate_timeout_seconds: int | None = None,
    protected_prefixes: tuple[str, ...] = (),
    prompt_bytes: bytes | None = None,
) -> tuple[RunResult, bytes]:
    if not decision.auto_execute or decision.status != "READY":
        raise AdapterError("HANDOFF_REQUIRED", decision.reason)
    prompt_path = task.prompt_path()
    if prompt_bytes is None:
        if prompt_path is None:
            raise AdapterError(
                "PROMPT_REQUIRED", "run requires prompt_file or inline prompt"
            )
        prompt = _read_bounded_prompt(prompt_path, task.max_context_tokens)
    else:
        if not isinstance(prompt_bytes, bytes) or not prompt_bytes:
            raise AdapterError(
                "PROMPT_REQUIRED", "inline prompt must be non-empty bytes"
            )
        if len(prompt_bytes) > task.max_context_tokens * 4:
            raise AdapterError(
                "CONTEXT_BUDGET_EXCEEDED",
                f"prompt exceeds the configured {task.max_context_tokens}-token budget",
            )
        prompt = prompt_bytes
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

    if task.write_mode == "read_only":
        outcome, reported, provenance = _run_once(
            task, decision, prompt=prompt, cwd=Path(task.repo)
        )
        exit_code = outcome.returncode
        stdout = outcome.stdout
        status, error_code = _process_status(outcome, decision)
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
                        gate_outcomes = run_gates(
                            transaction.worktree,
                            gate_commands,
                            gate_timeout_seconds or task.timeout_seconds,
                        )
                        gate_state = gates_passed(gate_outcomes, len(gate_commands))
                        if gate_state:
                            status, error_code = "PATCH_READY", "WRITER_OUTPUT_UNUSABLE"
                        break
                    if not provenance:
                        status, error_code = (
                            "PROVENANCE_UNVERIFIED",
                            "PROVENANCE_UNVERIFIED",
                        )
                        stdout = outcome.stdout
                        break

                    if patch_transport:
                        try:
                            transaction.apply_candidate_patch(
                                _qwen_result_text(outcome.stdout)
                            )
                        except WriterError as exc:
                            # Un fallo de gates se reintenta y uno de sintaxis
                            # del diff no, cuando es el mas facil de corregir:
                            # basta devolverle al modelo la queja de git. Como
                            # apply_candidate_patch hace `apply --check` antes
                            # de aplicar, el worktree sigue limpio y reintentar
                            # es seguro.
                            if (
                                exc.code not in {"GIT_ERROR", "MALFORMED_PATCH"}
                                or repairs >= task.max_repairs
                            ):
                                raise
                            repairs += 1
                            writer_input = _writer_prompt(
                                task,
                                patch_repair_prompt(prompt, str(exc)),
                                patch_only=True,
                            )
                            continue

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

                    gate_outcomes = run_gates(
                        transaction.worktree,
                        gate_commands,
                        gate_timeout_seconds or task.timeout_seconds,
                    )
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
                            if (
                                task.write_mode == "validated_apply"
                                and not native_writer
                            ):
                                error_code = "EXTERNAL_WRITER_REVIEW_REQUIRED"
                            elif (
                                task.write_mode == "validated_apply"
                                and not transaction.can_apply()
                            ):
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
        resolved_output = output_path.expanduser().resolve()
        if resolved_output.is_relative_to(Path(task.repo).resolve()):
            raise AdapterError(
                "OUTPUT_IN_REPO_FORBIDDEN", "model output must remain in staging"
            )
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
        error_detail=error_detail,
        changed_files=changed_files,
        patch_sha256=patch_sha256,
        repairs=repairs,
        gates_passed=gate_state,
    )
    return result, stdout
