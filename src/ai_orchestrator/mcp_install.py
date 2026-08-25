"""Register the orchestrator's MCP server in each CLI's native configuration.

``client-config`` emits the orchestrator's own extension schema and only writes
where it is told to. Nothing wrote the entry a CLI actually reads, so the server
had to be registered by hand and drifted: Claude, Codex and Abacus had it, Qwen
and OpenCode did not.

Every client keeps its registry in a different shape, so each one gets its own
writer. Kimi is deliberately absent: its CLI exposes no MCP surface at all (it
speaks the Agent Client Protocol instead), and the orchestrator already drives
it from the other side through ``kimi acp``.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ai_orchestrator.models import ContractError

SERVER_NAME = "ai-orchestrator"
EXECUTABLE = Path.home() / ".local" / "bin" / "ai-orchestrator"
SERVE_ARGS = ["mcp-serve"]

# Clientes que no exponen ninguna superficie MCP, con el motivo. Se reportan
# como omitidos y no como fallos: no hay nada que arreglar en ellos.
UNSUPPORTED = {
    "kimi": "the Kimi CLI speaks ACP and exposes no MCP server registry",
}


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, staging = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(staging, 0o600)
        os.replace(staging, path)
    except BaseException:
        Path(staging).unlink(missing_ok=True)
        raise


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    if path.is_symlink():
        raise ContractError(f"refusing to follow a symlinked config: {path}")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ContractError(f"{path} must contain a JSON object")
    return loaded


def _claude_shape() -> dict[str, Any]:
    return {"type": "stdio", "command": str(EXECUTABLE), "args": list(SERVE_ARGS)}


def _mcp_servers_client(path: Path, key: str = "mcpServers") -> dict[str, Any]:
    """Claude, Qwen and Abacus all key a mapping of servers off one field."""

    config = _load_json(path)
    servers = config.get(key)
    if not isinstance(servers, dict):
        servers = {}
    before = servers.get(SERVER_NAME)
    desired = _claude_shape()
    if before == desired:
        return {"path": str(path), "status": "ALREADY_REGISTERED"}
    servers[SERVER_NAME] = desired
    config[key] = servers
    _write_private_json(path, config)
    return {
        "path": str(path),
        "status": "UPDATED" if before else "REGISTERED",
    }


def install_claude() -> dict[str, Any]:
    return _mcp_servers_client(Path.home() / ".claude.json")


def install_qwen() -> dict[str, Any]:
    return _mcp_servers_client(Path.home() / ".qwen" / "settings.json")


def install_abacus() -> dict[str, Any]:
    return _mcp_servers_client(Path.home() / ".abacusai" / "mcp.json")


def install_opencode() -> dict[str, Any]:
    """OpenCode nests the argv in a list and gates each server with `enabled`."""

    path = Path.home() / ".config" / "opencode" / "opencode.json"
    config = _load_json(path)
    servers = config.get("mcp")
    if not isinstance(servers, dict):
        servers = {}
    desired = {
        "enabled": True,
        "type": "local",
        "command": [str(EXECUTABLE), *SERVE_ARGS],
    }
    before = servers.get(SERVER_NAME)
    if before == desired:
        return {"path": str(path), "status": "ALREADY_REGISTERED"}
    servers[SERVER_NAME] = desired
    config["mcp"] = servers
    _write_private_json(path, config)
    return {"path": str(path), "status": "UPDATED" if before else "REGISTERED"}


def install_codex() -> dict[str, Any]:
    """Codex keeps its registry in TOML, appended as a whole table."""

    path = Path.home() / ".codex" / "config.toml"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    header = f"[mcp_servers.{SERVER_NAME}]"
    if header in text:
        return {"path": str(path), "status": "ALREADY_REGISTERED"}
    args = ", ".join(f'"{value}"' for value in SERVE_ARGS)
    block = f'\n{header}\ncommand = "{EXECUTABLE}"\nargs = [{args}]\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(block)
    os.chmod(path, 0o600)
    return {"path": str(path), "status": "REGISTERED"}


INSTALLERS = {
    "claude": install_claude,
    "codex": install_codex,
    "abacus": install_abacus,
    "qwen": install_qwen,
    "opencode": install_opencode,
}


def install(client: str | None = None) -> dict[str, Any]:
    """Register the server in one client, or in every supported one."""

    if client is not None and client in UNSUPPORTED:
        return {
            "schema_version": 1,
            "clients": [
                {
                    "client": client,
                    "status": "UNSUPPORTED",
                    "detail": UNSUPPORTED[client],
                }
            ],
        }
    if client is not None and client not in INSTALLERS:
        raise ContractError(f"unknown MCP client: {client}")
    targets = [client] if client else sorted(INSTALLERS)
    rows: list[dict[str, Any]] = []
    for name in targets:
        try:
            rows.append({"client": name, **INSTALLERS[name]()})
        except (ContractError, OSError) as exc:
            rows.append({"client": name, "status": "FAILED", "detail": str(exc)})
    if client is None:
        rows.extend(
            {"client": name, "status": "UNSUPPORTED", "detail": detail}
            for name, detail in sorted(UNSUPPORTED.items())
        )
    return {"schema_version": 1, "clients": rows}


def status() -> dict[str, Any]:
    """Report where the server is registered, without changing anything."""

    rows: list[dict[str, Any]] = []
    checks = {
        "claude": Path.home() / ".claude.json",
        "codex": Path.home() / ".codex" / "config.toml",
        "abacus": Path.home() / ".abacusai" / "mcp.json",
        "qwen": Path.home() / ".qwen" / "settings.json",
        "opencode": Path.home() / ".config" / "opencode" / "opencode.json",
    }
    for name, path in sorted(checks.items()):
        if not path.is_file():
            rows.append({"client": name, "registered": False, "detail": "no config"})
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        rows.append(
            {
                "client": name,
                "registered": SERVER_NAME in text,
                "path": str(path),
            }
        )
    rows.extend(
        {"client": name, "registered": False, "detail": detail}
        for name, detail in sorted(UNSUPPORTED.items())
    )
    return {"schema_version": 1, "clients": rows}


def resolves_to_orchestrator(command: object) -> bool:
    """Say whether a registered command string is this orchestrator.

    The registry may hold a bare name resolved through PATH, the symlink in
    ~/.local/bin, or the console script inside the venv. Comparing the literal
    string against one spelling made ``doctor`` report ``mcp: false`` on a
    perfectly good global registration.
    """

    if not isinstance(command, str) or not command:
        return False
    candidate = Path(command)
    if not candidate.is_absolute():
        found = shutil.which(command)
        if found is None:
            return candidate.name == EXECUTABLE.name
        candidate = Path(found)
    try:
        return candidate.resolve() == EXECUTABLE.resolve()
    except OSError:
        return False


def repo_mcp_status(repo: Path, profile: str) -> dict[str, Any]:
    """Check a repository-scoped .mcp.json bound to one profile."""

    source = Path(repo) / ".mcp.json"
    if source.is_symlink() or not source.is_file():
        return {"configured": False, "path": str(source), "reason": "absent"}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        server = payload["mcpServers"][SERVER_NAME]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return {"configured": False, "path": str(source), "reason": "unreadable"}
    arguments = list(server.get("args") or ())
    configured = resolves_to_orchestrator(server.get("command")) and arguments == [
        *SERVE_ARGS,
        "--profile",
        profile,
    ]
    return {"configured": configured, "path": str(source)}


def global_mcp_status() -> dict[str, Any]:
    """Check the profile-less registration every CLI reads by default."""

    path = Path.home() / ".claude.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        server = payload["mcpServers"][SERVER_NAME]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return {"configured": False, "path": str(path), "reason": "absent"}
    return {
        "configured": resolves_to_orchestrator(server.get("command")),
        "path": str(path),
        "scope": "global",
    }
