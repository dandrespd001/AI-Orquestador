"""Fail-closed Claude Code hook for the CHUNSA orchestration gateway."""

from __future__ import annotations

import json
import os
import re
import shlex
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from ai_orchestrator.mcp_install import repo_mcp_status
from ai_orchestrator.models import ensure_private_directory

_SESSION_ID = re.compile(r"^[A-Za-z0-9._-]{1,200}$")
_READ_TOOLS = frozenset({"Read", "Glob", "Grep", "LS", "NotebookRead"})
_READ_ONLY_GIT = frozenset(
    {"status", "diff", "log", "show", "rev-parse", "ls-files", "grep"}
)
_MODEL_CLIS = frozenset(
    {"claude", "codex", "qwen", "kimi", "abacusai", "fx", "opencode"}
)
_SAFE_ORCHESTRATOR_COMMANDS = frozenset(
    {
        "submit",
        "artifact",
        "doctor",
        "doctor-full",
        "route",
        "run",
        "route-plan",
        "report",
        "control",
    }
)


def _command_line_value(arguments: list[str], option: str) -> str | None:
    for index, argument in enumerate(arguments):
        if argument == option and index + 1 < len(arguments):
            return arguments[index + 1]
        prefix = f"{option}="
        if argument.startswith(prefix):
            return argument[len(prefix) :]
    return None


def _claude_launch_overrides() -> tuple[bool, str | None, str | None]:
    """Inspect the trusted Claude ancestor because some releases omit hook model."""

    process_id = os.getppid()
    visited: set[int] = set()
    for _ in range(8):
        if process_id <= 1 or process_id in visited:
            break
        visited.add(process_id)
        process = Path("/proc") / str(process_id)
        try:
            arguments = [
                value.decode("utf-8", errors="replace")
                for value in process.joinpath("cmdline").read_bytes().split(b"\0")
                if value
            ]
            status = process.joinpath("status").read_text(encoding="utf-8")
        except OSError:
            return False, None, None
        if arguments and (
            Path(arguments[0]).name == "claude"
            or any("claude-code" in value for value in arguments[:2])
        ):
            return (
                True,
                _command_line_value(arguments, "--model"),
                _command_line_value(arguments, "--effort"),
            )
        parent_line = next(
            (line for line in status.splitlines() if line.startswith("PPid:")), None
        )
        if parent_line is None:
            return False, None, None
        try:
            process_id = int(parent_line.split(":", 1)[1].strip())
        except ValueError:
            return False, None, None
    return False, None, None


@dataclass(frozen=True)
class HookCommandResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""


class ClaudeGatewayHook:
    def __init__(
        self,
        *,
        profile: str,
        repo: Path,
        state_root: Path,
        gate_commands: tuple[str, ...],
        readiness: Callable[[], Mapping[str, object]],
        launch_overrides: Callable[[], tuple[bool, str | None, str | None]]
        | None = None,
    ) -> None:
        self.profile = profile
        self.repo = repo.expanduser().resolve()
        self.state_root = state_root.expanduser().resolve()
        self.gate_commands = tuple(command.strip() for command in gate_commands)
        self.readiness = readiness
        self.launch_overrides = launch_overrides or _claude_launch_overrides
        ensure_private_directory(self.state_root)

    def _attestation_path(self, session_id: object) -> Path | None:
        value = str(session_id or "")
        if not _SESSION_ID.fullmatch(value):
            return None
        return self.state_root / f"{value}.json"

    def _mcp_configured(self) -> bool:
        # Comparte comparador con doctor: el registro puede llevar el nombre
        # pelado, el symlink de ~/.local/bin o el script del venv, y exigir una
        # sola grafia daba falso negativo sobre un registro perfectamente valido.
        return bool(repo_mcp_status(self.repo, self.profile)["configured"])

    def _project_model(self) -> str:
        source = self.repo / ".claude" / "settings.json"
        if source.is_symlink() or not source.is_file():
            return ""
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ""
        if not isinstance(payload, dict):
            return ""
        return str(payload.get("model", "")).strip().lower()

    @staticmethod
    def _transcript_model(payload: Mapping[str, object]) -> str:
        raw_path = payload.get("transcript_path")
        if not isinstance(raw_path, str) or not raw_path:
            return ""
        source = Path(raw_path).expanduser()
        if source.is_symlink() or not source.is_file():
            return ""
        try:
            with source.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - 1_000_000))
                content = handle.read(1_000_000)
        except OSError:
            return ""
        for line in reversed(content.splitlines()):
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(row, dict):
                continue
            message = row.get("message")
            candidate = message.get("model") if isinstance(message, dict) else None
            candidate = candidate or row.get("model")
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip().lower()
        return ""

    def session_start(
        self,
        payload: Mapping[str, object],
        *,
        environment: Mapping[str, str] | None = None,
    ) -> HookCommandResult:
        path = self._attestation_path(payload.get("session_id"))
        if path is None:
            return HookCommandResult(2, stderr="invalid Claude session identifier")
        if path.exists() and not path.is_symlink():
            path.unlink()
        source = str(payload.get("source", ""))
        cwd = Path(str(payload.get("cwd", ""))).expanduser().resolve()
        selected_environment = os.environ if environment is None else environment
        launch_inspected, launch_model, launch_effort = self.launch_overrides()
        payload_model = str(payload.get("model", "")).strip().lower()
        environment_model = str(selected_environment.get("ANTHROPIC_MODEL", ""))
        if launch_model:
            model = launch_model.strip().lower()
        elif payload_model:
            model = payload_model
        elif environment_model:
            model = environment_model.strip().lower()
        elif source in {"resume", "clear", "compact"}:
            model = self._transcript_model(payload)
        elif launch_inspected:
            model = self._project_model()
        else:
            model = ""
        effort = str(
            launch_effort or selected_environment.get("CLAUDE_CODE_EFFORT_LEVEL", "")
        ).lower()
        readiness = dict(self.readiness())
        failures: list[str] = []
        if not launch_inspected and not payload_model and not environment_model:
            failures.append("Claude launch could not be verified")
        if "sonnet" not in model:
            failures.append(f"coordinator model must be Sonnet (observed: {model!r})")
        if effort != "high":
            failures.append("CLAUDE_CODE_EFFORT_LEVEL must be high")
        if cwd != self.repo:
            failures.append("session repository does not match the profile")
        if not self._mcp_configured():
            failures.append("project AI-Orchestrator MCP is not configured")
        if readiness.get("ok") is not True:
            failures.append("AI-Orchestrator readiness failed")
        if source == "resume" and "opus" in model:
            failures.append("resumed Opus sessions are non-conformant")
        if failures:
            return HookCommandResult(2, stderr="; ".join(failures))

        attestation = {
            "schema_version": 1,
            "session_id": str(payload["session_id"]),
            "profile": self.profile,
            "repo": str(self.repo),
            "model": model,
            "effort": effort,
            "source": source,
        }
        path.write_text(
            json.dumps(attestation, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.chmod(path, 0o600)
        output = {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": (
                    "CHUNSA gateway attested. All execution, writes and delegation "
                    "must use AI-Orchestrator. Direct edits and model CLIs are denied."
                ),
            }
        }
        return HookCommandResult(0, stdout=json.dumps(output, sort_keys=True))

    @staticmethod
    def _decision(value: str, reason: str) -> dict[str, object]:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": value,
                "permissionDecisionReason": reason,
            }
        }

    def _attested(self, payload: Mapping[str, object]) -> bool:
        path = self._attestation_path(payload.get("session_id"))
        if path is None or path.is_symlink() or not path.is_file():
            return False
        try:
            attestation = json.loads(path.read_text(encoding="utf-8"))
            cwd = Path(str(payload.get("cwd", ""))).expanduser().resolve()
        except (json.JSONDecodeError, OSError):
            return False
        return (
            attestation.get("session_id") == payload.get("session_id")
            and attestation.get("profile") == self.profile
            and attestation.get("repo") == str(self.repo)
            and cwd == self.repo
            and "sonnet" in str(attestation.get("model", "")).lower()
            and attestation.get("effort") == "high"
        )

    def _bash_allowed(self, command: object) -> bool:
        if not isinstance(command, str) or not command.strip():
            return False
        normalized = command.strip()
        if normalized in self.gate_commands:
            return True
        try:
            words = shlex.split(normalized)
        except ValueError:
            return False
        if not words or any(
            value in normalized for value in ("\n", "&&", "||", ";", "`", "$(")
        ):
            return False
        if any(word in {"|", "&", ">", ">>", "<", "<<", "2>", "2>>"} for word in words):
            return False
        executable = Path(words[0]).name
        if executable in _MODEL_CLIS:
            return False
        if executable == "git":
            return len(words) >= 2 and words[1] in _READ_ONLY_GIT
        if executable == "ai-orchestrator":
            return len(words) >= 2 and words[1] in _SAFE_ORCHESTRATOR_COMMANDS
        if executable == "chunsa_ai.sh" or words[0].endswith("/chunsa_ai.sh"):
            return len(words) >= 2 and words[1] in _SAFE_ORCHESTRATOR_COMMANDS
        return False

    def pre_tool_use(self, payload: Mapping[str, object]) -> dict[str, object]:
        if not self._attested(payload):
            return self._decision(
                "deny", "missing or invalid CHUNSA session attestation"
            )
        tool_name = str(payload.get("tool_name", ""))
        tool_input = payload.get("tool_input")
        arguments = tool_input if isinstance(tool_input, dict) else {}
        if tool_name in _READ_TOOLS:
            raw_path = arguments.get("file_path", arguments.get("path"))
            if isinstance(raw_path, str) and raw_path:
                candidate = Path(raw_path).expanduser()
                if not candidate.is_absolute():
                    candidate = self.repo / candidate
                try:
                    candidate.resolve().relative_to(self.repo)
                except ValueError:
                    return self._decision(
                        "deny", "repository reads may not escape the CHUNSA profile"
                    )
            return self._decision("allow", "attested repository read")
        if tool_name == "Bash" and self._bash_allowed(arguments.get("command")):
            return self._decision(
                "allow", "approved read, gate or orchestrator command"
            )
        normalized = tool_name.replace("_", "-")
        if normalized.startswith("mcp--ai-orchestrator--"):
            return self._decision("allow", "profile-bound AI-Orchestrator MCP")
        if tool_name.startswith(
            (
                "mcp__codebase-memory__",
                "mcp__codebase_memory__",
                "mcp__codebase_memory_mcp__",
            )
        ):
            return self._decision("allow", "read-only code graph")
        return self._decision(
            "deny",
            "direct execution, mutation, delegation or unapproved MCP is blocked",
        )
