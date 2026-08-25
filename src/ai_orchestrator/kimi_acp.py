"""Minimal ACP client that gives Kimi bounded text-file access, never a terminal."""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


class AcpError(RuntimeError):
    pass


@dataclass(frozen=True)
class AcpOutcome:
    stdout: bytes
    stderr: bytes
    reported_model: str


_POLICY_TEMPLATE = """default_permission_mode = "manual"
merge_all_available_skills = false
builtin_product_skills = false
telemetry = false

[background]
max_running_tasks = 1
keep_alive_on_exit = false
bash_auto_background_on_timeout = false

[tools]
disabled = [{disabled}]

[[permission.rules]]
decision = "allow"
pattern = "Read"

[[permission.rules]]
decision = "allow"
pattern = "Write"

[[permission.rules]]
decision = "allow"
pattern = "Edit"

[[permission.rules]]
decision = "deny"
pattern = "Bash"

[[permission.rules]]
decision = "deny"
pattern = "Agent"

[[permission.rules]]
decision = "deny"
pattern = "AgentSwarm"
"""

_ALWAYS_DISABLED = (
    "Bash",
    "Agent",
    "AgentSwarm",
    "CronCreate",
    "CronList",
    "CronDelete",
    "EnterPlanMode",
    "ExitPlanMode",
)
# Leer la web se abre por rol, no globalmente: un rol de investigacion lo
# necesita y uno de implementacion no tiene por que tenerlo.
_WEB_TOOLS = ("WebSearch", "FetchURL")


def _policy_bytes(*, web_access: bool = False) -> bytes:
    disabled = _ALWAYS_DISABLED if web_access else (*_ALWAYS_DISABLED, *_WEB_TOOLS)
    rendered = ", ".join(f'"{name}"' for name in disabled)
    return _POLICY_TEMPLATE.format(disabled=rendered).encode()


@contextmanager
def temporary_kimi_policy(worktree: Path, *, web_access: bool = False):
    """Install a non-persistent project policy while the ACP process starts."""

    directory = worktree / ".kimi-code"
    policy = directory / "local.toml"
    if directory.is_symlink() or policy.is_symlink():
        raise AcpError("unsafe project-local Kimi policy path")
    existed = policy.is_file()
    previous = policy.read_bytes() if existed else None
    previous_mode = policy.stat().st_mode & 0o777 if existed else None
    directory.mkdir(parents=True, exist_ok=True)
    policy.write_bytes(_policy_bytes(web_access=web_access))
    policy.chmod(0o600)
    try:
        yield
    finally:
        if previous is not None:
            policy.write_bytes(previous)
            policy.chmod(previous_mode or 0o600)
        else:
            try:
                policy.unlink()
            except FileNotFoundError:
                pass
            try:
                directory.rmdir()
            except OSError:
                pass


class KimiAcpClient:
    def __init__(
        self,
        *,
        executable: str,
        worktree: Path,
        allowed_write_paths: tuple[str, ...],
        environment: dict[str, str],
        timeout_seconds: int,
    ) -> None:
        self.worktree = worktree.resolve()
        self.allowed = tuple(PurePosixPath(value) for value in allowed_write_paths)
        self.timeout_seconds = timeout_seconds
        self.process = subprocess.Popen(
            [executable, "acp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.worktree,
            env=environment,
            start_new_session=True,
        )
        self.events: queue.Queue[tuple[str, bytes]] = queue.Queue()
        self.stderr_parts: list[bytes] = []
        self.message_parts: list[str] = []
        self.next_id = 1
        threading.Thread(
            target=self._reader, args=("stdout", self.process.stdout), daemon=True
        ).start()
        threading.Thread(
            target=self._reader, args=("stderr", self.process.stderr), daemon=True
        ).start()

    def _reader(self, name: str, stream: Any) -> None:
        while True:
            line = stream.readline()
            if not line:
                return
            self.events.put((name, line))

    def _send(self, payload: dict[str, Any]) -> None:
        if self.process.stdin is None:
            raise AcpError("ACP stdin is closed")
        self.process.stdin.write(
            json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"
        )
        self.process.stdin.flush()

    def _resolve(self, raw: object, *, write: bool) -> Path:
        # `Path.relative_to` is PURELY LEXICAL: it keeps `..` components instead
        # of normalising them. `<worktree>/src/../../escaped.txt` reduced to
        # `src/../../escaped.txt`, whose `parents` literally contain `src`, so
        # the allowed prefix validated its own escape — and these writes never
        # reach writer.py, so the allowlist, the protected prefixes and the
        # gates were all bypassed at once. Reject `..` outright and check
        # containment on the RESOLVED path, the same way writer.py:122-135 does.
        path = Path(str(raw))
        if not path.is_absolute():
            raise AcpError("ACP paths must be absolute")
        if ".." in path.parts:
            raise AcpError("ACP path contains a parent-directory component")
        try:
            relative = path.relative_to(self.worktree)
        except ValueError as exc:
            raise AcpError("ACP path escaped the worktree") from exc
        pure = PurePosixPath(relative.as_posix())
        # The allowlist is checked for reads too. A read that escapes hands the
        # file straight back to the model, which is an exfiltration channel and
        # not merely a scoping mistake.
        if not any(pure == prefix or prefix in pure.parents for prefix in self.allowed):
            raise AcpError(
                "ACP write escaped the task allowlist"
                if write
                else "ACP read escaped the task allowlist"
            )
        current = path
        while current != self.worktree:
            if current.is_symlink():
                raise AcpError("ACP path contains a symlink")
            current = current.parent
        return path

    def _reverse_request(self, payload: dict[str, Any]) -> None:
        request_id = payload.get("id")
        method = payload.get("method")
        params = (
            payload.get("params") if isinstance(payload.get("params"), dict) else {}
        )
        try:
            if method == "fs/read_text_file":
                path = self._resolve(params.get("path"), write=False)
                lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
                line = max(1, int(params.get("line", 1)))
                limit = max(1, min(1000, int(params.get("limit", 1000))))
                result: object = {
                    "content": "".join(lines[line - 1 : line - 1 + limit])
                }
            elif method == "fs/write_text_file":
                path = self._resolve(params.get("path"), write=True)
                content = str(params.get("content", ""))
                if len(content.encode("utf-8")) > 2_097_152:
                    raise AcpError("ACP single-file write limit exceeded")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                result = None
            elif method == "session/request_permission":
                tool = (
                    params.get("toolCall")
                    if isinstance(params.get("toolCall"), dict)
                    else {}
                )
                kind = str(tool.get("kind", "other"))
                allow = kind in {"edit", "delete", "move"}
                options = (
                    params.get("options")
                    if isinstance(params.get("options"), list)
                    else []
                )
                wanted = (
                    {"allow_once", "allow_always"}
                    if allow
                    else {"reject_once", "reject_always"}
                )
                selected = next(
                    (
                        value
                        for value in options
                        if isinstance(value, dict) and value.get("kind") in wanted
                    ),
                    None,
                )
                result = (
                    {
                        "outcome": {
                            "outcome": "selected",
                            "optionId": selected["optionId"],
                        }
                    }
                    if selected
                    else {"outcome": {"outcome": "cancelled"}}
                )
            else:
                raise AcpError(f"unsupported ACP reverse method: {method}")
            self._send({"jsonrpc": "2.0", "id": request_id, "result": result})
        except (AcpError, OSError, UnicodeError, ValueError) as exc:
            self._send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32001, "message": str(exc)},
                }
            )

    def request(self, method: str, params: dict[str, Any]) -> Any:
        request_id = self.next_id
        self.next_id += 1
        self._send(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        )
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AcpError(f"ACP request timed out: {method}")
            try:
                stream, line = self.events.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                if self.process.poll() is not None:
                    raise AcpError(f"Kimi ACP exited during {method}")
                continue
            if stream == "stderr":
                self.stderr_parts.append(line)
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AcpError("Kimi ACP produced malformed JSON-RPC") from exc
            if "method" in payload and "id" in payload:
                self._reverse_request(payload)
                continue
            if payload.get("method") == "session/update":
                update = payload.get("params", {}).get("update", {})
                if update.get("sessionUpdate") == "agent_message_chunk":
                    content = update.get("content", {})
                    if content.get("type") == "text":
                        self.message_parts.append(str(content.get("text", "")))
                continue
            if payload.get("id") != request_id:
                continue
            if "error" in payload:
                error = payload["error"]
                raise AcpError(str(error.get("message", "ACP request failed")))
            return payload.get("result")

    def close(self) -> bytes:
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=3)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait()
        if self.process.stdout is not None:
            self.process.stdout.close()
        if self.process.stderr is not None:
            self.process.stderr.close()
        while True:
            try:
                stream, line = self.events.get_nowait()
            except queue.Empty:
                break
            if stream == "stderr":
                self.stderr_parts.append(line)
        return b"".join(self.stderr_parts)


def run_kimi_acp(
    *,
    executable: str,
    worktree: Path,
    allowed_write_paths: tuple[str, ...],
    environment: dict[str, str],
    model: str,
    prompt: bytes,
    timeout_seconds: int,
    web_access: bool = False,
) -> AcpOutcome:
    with temporary_kimi_policy(worktree, web_access=web_access):
        client = KimiAcpClient(
            executable=executable,
            worktree=worktree,
            allowed_write_paths=allowed_write_paths,
            environment=environment,
            timeout_seconds=timeout_seconds,
        )
        try:
            initialized = client.request(
                "initialize",
                {
                    "protocolVersion": 1,
                    "clientCapabilities": {
                        "fs": {"readTextFile": True, "writeTextFile": True},
                        "terminal": False,
                    },
                    "clientInfo": {
                        "name": "ai-orchestrator",
                        "title": "AI Orchestrator",
                        "version": "0.1.0",
                    },
                },
            )
            if (
                not isinstance(initialized, dict)
                or initialized.get("protocolVersion") != 1
            ):
                raise AcpError("Kimi ACP protocol negotiation failed")
            session = client.request(
                "session/new", {"cwd": str(worktree), "mcpServers": []}
            )
            if not isinstance(session, dict) or not session.get("sessionId"):
                raise AcpError("Kimi ACP did not return a session id")
            session_id = str(session["sessionId"])
            client.request(
                "session/set_config_option",
                {"sessionId": session_id, "configId": "model", "value": model},
            )
            client.request(
                "session/prompt",
                {
                    "sessionId": session_id,
                    "prompt": [
                        {"type": "text", "text": prompt.decode("utf-8", "replace")}
                    ],
                },
            )
            return AcpOutcome(
                stdout="".join(client.message_parts).encode("utf-8"),
                stderr=b"",
                reported_model=model,
            )
        finally:
            client.close()
