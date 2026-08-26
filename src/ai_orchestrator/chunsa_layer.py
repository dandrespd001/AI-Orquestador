"""Generate client-specific CHUNSA bundles from one policy and one Skill source."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from ai_orchestrator.models import ContractError


def layer_documents(root: Path) -> dict[Path, str]:
    layer_path = root / "integrations" / "chunsa" / "layer.toml"
    with layer_path.open("rb") as handle:
        layer = tomllib.load(handle)
    if layer.get("schema_version") != 1 or layer.get("profile") != "chunsa":
        raise ContractError("CHUNSA layer source is invalid")
    command = str(layer["mcp_command"])
    arguments = [str(value) for value in layer["mcp_args"]]
    server = {"command": command, "args": arguments, "type": "stdio"}
    qwen = {
        "name": "chunsa-orchestration",
        "version": "0.2.0",
        "description": "Strict AI-Orchestrator routing and acceptance for CHUNSA.",
        "skills": "skills",
        "mcpServers": {"ai-orchestrator-chunsa": server},
    }
    kimi = {
        "name": "chunsa-orchestration",
        "version": "0.2.0",
        "description": "CHUNSA orchestration commands for Kimi ACP.",
        "skills": "./skills/",
    }
    claude = {"mcpServers": {"ai-orchestrator-chunsa": server}}
    opencode = {
        "mcp": {
            "ai-orchestrator-chunsa": {
                "command": [command, *arguments],
                "enabled": True,
                "type": "local",
            }
        }
    }
    outputs = {
        root / "integrations" / "chunsa" / "qwen-extension" / "qwen-extension.json": json.dumps(
            qwen, sort_keys=True, indent=2, ensure_ascii=False
        )
        + "\n",
        root / "integrations" / "chunsa" / "kimi-plugin" / "kimi.plugin.json": json.dumps(
            kimi, sort_keys=True, indent=2, ensure_ascii=False
        )
        + "\n",
        root / "integrations" / "chunsa" / "packages" / "claude" / ".mcp.json": json.dumps(
            claude, sort_keys=True, indent=2, ensure_ascii=False
        )
        + "\n",
        root / "integrations" / "chunsa" / "packages" / "opencode" / "opencode.json": json.dumps(
            opencode, sort_keys=True, indent=2, ensure_ascii=False
        )
        + "\n",
    }
    skill_source = root / ".agents" / "skills" / "chunsa-orchestration"
    destinations = (
        root / "integrations" / "chunsa" / "plugins" / "chunsa-orchestration" / "skills" / "chunsa-orchestration",
        root / "integrations" / "chunsa" / "qwen-extension" / "skills" / "chunsa-orchestration",
        root / "integrations" / "chunsa" / "kimi-plugin" / "skills" / "chunsa-orchestration",
        root / "integrations" / "chunsa" / "packages" / "claude" / "skills" / "chunsa-orchestration",
        root / "integrations" / "chunsa" / "packages" / "opencode" / "skills" / "chunsa-orchestration",
        root / "integrations" / "deepseek-harness" / "bundle" / "skills" / "chunsa-orchestration",
    )
    for source in sorted(skill_source.rglob("*")):
        if source.is_symlink() or not source.is_file():
            continue
        relative = source.relative_to(skill_source)
        content = source.read_text(encoding="utf-8")
        for destination in destinations:
            outputs[destination / relative] = content
    return outputs
