"""Generate per-profile client configuration from the approved catalog."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ai_orchestrator.extensions import (
    ADMIN_PROFILES,
    COMMUNITY_TRUST,
    ExtensionRegistry,
)
from ai_orchestrator.models import (
    ContractError,
    ExtensionManifest,
    ensure_private_directory,
)

SUPPORTED_CLIENTS = {
    "harness",
    "codex",
    "claude",
    "kimi",
    "qwen",
    "opencode",
    "abacus",
    "fx",
}


def _compatible(manifest: ExtensionManifest, client: str) -> bool:
    if not manifest.compatibility:
        return True
    declared = {value.split(":", 1)[0] for value in manifest.compatibility}
    return client in declared or "all" in declared


def generate_client_config(
    registry: ExtensionRegistry,
    profile: str,
    client: str,
) -> dict[str, Any]:
    if client not in SUPPORTED_CLIENTS:
        raise ContractError(f"unsupported extension client: {client}")
    state = registry._load()
    active = state.get("profiles", {}).get(profile, {})
    if not isinstance(active, dict):
        raise ContractError("extension profile is invalid")
    plugins: list[dict[str, Any]] = []
    skills: list[dict[str, Any]] = []
    mcp_servers: list[dict[str, Any]] = []
    for extension_id, reference in sorted(active.items()):
        manifest, _version, verified_path = registry.verify_approved(
            extension_id, reference, state=state
        )
        if profile in ADMIN_PROFILES and manifest.trust in COMMUNITY_TRUST:
            raise ContractError("administrative client config contains community code")
        if not _compatible(manifest, client):
            continue
        common = {
            "id": manifest.id,
            "sha256": manifest.sha256,
            "trust": manifest.trust,
            "path": str(verified_path),
        }
        if manifest.kind == "plugin":
            if client == "harness":
                plugins.append({**common, "permissions": list(manifest.permissions)})
        elif manifest.kind == "skill":
            skills.append(common)
        else:
            server: dict[str, Any] = {
                **common,
                "transport": "stdio" if manifest.source_type == "mcp_stdio" else "http",
                "tools": list(manifest.tools),
                "allowed_domains": list(manifest.remote_domains),
                "environment": {},
            }
            if manifest.source_type == "mcp_stdio":
                server["command"] = manifest.entrypoint[0]
                server["args"] = list(manifest.entrypoint[1:])
            else:
                server["url"] = manifest.source
            mcp_servers.append(server)
    return {
        "schema_version": 1,
        "client": client,
        "scope": "profile",
        "profile": profile,
        "plugins": plugins,
        "skills": skills,
        "mcp_servers": mcp_servers,
        "global_mcp_installed": False,
        "provider_secrets_embedded": False,
        "tool_schema_count": sum(len(row["tools"]) for row in mcp_servers),
    }


def write_client_config(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a generated, secret-free profile configuration."""

    forbidden = {"api_key", "password", "secret", "token"}

    def validate(value: object) -> None:
        if isinstance(value, dict):
            if forbidden & {str(key).lower() for key in value}:
                raise ContractError("generated client config contains a secret field")
            for nested in value.values():
                validate(nested)
        elif isinstance(value, list):
            for nested in value:
                validate(nested)

    validate(payload)
    ensure_private_directory(path.parent)
    if path.is_symlink():
        raise ContractError("refusing to replace a client-config symlink")
    data = json.dumps(payload, sort_keys=True, indent=2).encode("utf-8") + b"\n"
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
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
