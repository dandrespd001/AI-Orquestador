"""Native, preserving installers for the strict profile-bound MCP server."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_orchestrator.models import ContractError, ensure_private_directory

SERVER_NAME = "ai-orchestrator-chunsa"
LEGACY_SERVER_NAME = "ai-orchestrator"
SUPPORTED_MCP_CLIENTS = frozenset({"codex", "claude", "qwen", "opencode", "abacus"})
UNSUPPORTED = {"kimi": "Kimi Code is integrated through `kimi acp`, not MCP"}


def _server(command: str, profile: str) -> dict[str, Any]:
    return {
        "type": "stdio",
        "command": command,
        "args": ["mcp", "serve", "--profile", profile],
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"unsafe client configuration: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid client JSON configuration: {path}") from exc
    if not isinstance(payload, dict):
        raise ContractError("client configuration must contain an object")
    return payload


def _backup(path: Path) -> str | None:
    if not path.exists():
        return None
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = path.with_name(f"{path.name}.bak.{stamp}")
    counter = 0
    while target.exists():
        counter += 1
        target = path.with_name(f"{path.name}.bak.{stamp}.{counter}")
    shutil.copy2(path, target, follow_symlinks=False)
    os.chmod(target, 0o600)
    return str(target)


def _atomic(path: Path, data: bytes) -> None:
    ensure_private_directory(path.parent)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _install_json(path: Path, client: str, desired: dict[str, Any]) -> dict[str, Any]:
    payload = _load_json(path)
    if client == "opencode":
        servers = payload.setdefault("mcp", {})
        native = {
            "enabled": True,
            "type": "local",
            "command": [desired["command"], *desired["args"]],
        }
    else:
        servers = payload.setdefault("mcpServers", {})
        native = {**desired, "alwaysLoad": True} if client == "claude" else desired
    if not isinstance(servers, dict):
        raise ContractError("client MCP registry has an invalid native shape")
    legacy = servers.get(LEGACY_SERVER_NAME)
    legacy_command = legacy.get("command") if isinstance(legacy, dict) else None
    if client == "opencode":
        legacy_owned = bool(
            isinstance(legacy_command, list)
            and legacy_command in [
                [desired["command"], "mcp-serve"],
                [desired["command"], "mcp", "serve"],
            ]
        )
    else:
        legacy_owned = bool(
            isinstance(legacy, dict)
            and legacy_command == desired["command"]
            and legacy.get("args") in (["mcp-serve"], ["mcp", "serve"])
        )
    if servers.get(SERVER_NAME) == native and not legacy_owned:
        return {"client": client, "path": str(path), "status": "ALREADY_REGISTERED"}
    backup = _backup(path)
    if legacy_owned:
        servers.pop(LEGACY_SERVER_NAME, None)
    servers[SERVER_NAME] = native
    _atomic(
        path,
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False).encode() + b"\n",
    )
    return {"client": client, "path": str(path), "status": "UPDATED", "backup": backup}


def _install_codex(path: Path, desired: dict[str, Any]) -> dict[str, Any]:
    if path.is_symlink():
        raise ContractError("refusing a symlinked Codex configuration")
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    try:
        parsed = tomllib.loads(text) if text.strip() else {}
    except tomllib.TOMLDecodeError as exc:
        raise ContractError("invalid Codex TOML configuration") from exc
    before = parsed.get("mcp_servers", {}).get(SERVER_NAME)
    legacy = parsed.get("mcp_servers", {}).get(LEGACY_SERVER_NAME)
    expected = {"command": desired["command"], "args": desired["args"]}
    legacy_owned = bool(
        isinstance(legacy, dict)
        and legacy.get("command") == desired["command"]
        and legacy.get("args") in (["mcp-serve"], ["mcp", "serve"])
    )
    if before == expected and not legacy_owned:
        return {"client": "codex", "path": str(path), "status": "ALREADY_REGISTERED"}
    header = re.escape(f"[mcp_servers.{SERVER_NAME}]")
    pattern = re.compile(rf"(?ms)^\s*{header}\s*\n.*?(?=^\s*\[|\Z)")
    block = (
        f"[mcp_servers.{SERVER_NAME}]\n"
        f"command = {json.dumps(desired['command'])}\n"
        f"args = {json.dumps(desired['args'])}\n"
    )
    updated = pattern.sub("", text)
    if legacy_owned:
        legacy_header = re.escape(f"[mcp_servers.{LEGACY_SERVER_NAME}]")
        legacy_pattern = re.compile(
            rf"(?ms)^\s*{legacy_header}\s*\n.*?(?=^\s*\[|\Z)"
        )
        updated = legacy_pattern.sub("", updated)
    updated = updated.rstrip() + "\n\n" + block
    try:
        verified = tomllib.loads(updated)["mcp_servers"][SERVER_NAME]
    except (tomllib.TOMLDecodeError, KeyError, TypeError) as exc:
        raise ContractError("generated Codex MCP table is invalid") from exc
    if verified != expected:
        raise ContractError("generated Codex MCP table changed its native contract")
    backup = _backup(path)
    _atomic(path, updated.encode("utf-8"))
    return {"client": "codex", "path": str(path), "status": "UPDATED", "backup": backup}


def install_native_mcp(
    client: str,
    *,
    profile: str = "chunsa",
    home: Path | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    if profile != "chunsa":
        raise ContractError("strict MCP installation is restricted to profile chunsa")
    if client in UNSUPPORTED:
        return {"client": client, "status": "UNSUPPORTED", "detail": UNSUPPORTED[client]}
    if client not in SUPPORTED_MCP_CLIENTS:
        raise ContractError(f"unsupported MCP client: {client}")
    root = (home or Path.home()).expanduser().resolve()
    executable = command or shutil.which("ai-orchestrator") or str(
        root / ".local" / "bin" / "ai-orchestrator"
    )
    desired = _server(executable, profile)
    paths = {
        "claude": root / ".claude.json",
        "qwen": root / ".qwen" / "settings.json",
        "abacus": root / ".abacusai" / "mcp.json",
        "opencode": root / ".config" / "opencode" / "opencode.json",
        "codex": root / ".codex" / "config.toml",
    }
    if client == "codex":
        return _install_codex(paths[client], desired)
    return _install_json(paths[client], client, desired)
