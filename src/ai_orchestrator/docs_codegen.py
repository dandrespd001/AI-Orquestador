"""Deterministic public documentation generated from parsers and manifests."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

from ai_orchestrator.chunsa_layer import layer_documents
from ai_orchestrator.config import config_root, load_models
from ai_orchestrator.mcp_tools import MCP_TOOLS
from ai_orchestrator.models import ContractError
from ai_orchestrator.updates import load_component_pins


def _commands(parser: argparse.ArgumentParser, prefix: tuple[str, ...] = ()) -> list[str]:
    rows: list[str] = []
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        for name, child in sorted(action.choices.items()):
            command = (*prefix, name)
            nested = _commands(child, command)
            rows.extend(nested or [" ".join(command)])
    return rows


def repo_root(override: str | Path | None = None) -> Path:
    """Raiz donde se ESCRIBEN los documentos y vive `integrations/`.

    No se puede derivar de `__file__`: con el paquete instalado, `parents[2]` es
    `site-packages/../..` = `lib/python3.12`, que no contiene ni `docs/` ni
    `integrations/` ni `config/`. Ese era el defecto que hacia fallar
    `docs check` fuera del arbol fuente con ContractError.
    """

    if override:
        return Path(override).expanduser().resolve()
    environment = os.environ.get("AI_ORCHESTRATOR_REPO")
    if environment:
        return Path(environment).expanduser().resolve()
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").is_file():
        return candidate
    raise ContractError(
        "documentation root unknown: pass --root or set AI_ORCHESTRATOR_REPO"
    )


def _documents(root: Path, config_dir: Path | None = None) -> dict[Path, str]:
    from ai_orchestrator.cli import build_parser

    # `root` es donde se ESCRIBEN los documentos; la configuracion vive donde
    # diga `config_root()`. Derivarla de `root` solo funcionaba ejecutando desde
    # el arbol fuente: con el paquete instalado, `root` acababa siendo
    # site-packages/../.. y `docs check` moria con ContractError.
    models = load_models(config_dir)
    pins = load_component_pins(config_dir)
    commands = "\n".join(f"- `ai-orchestrator {row}`" for row in _commands(build_parser()))
    model_rows = "\n".join(
        f"| `{model.id}` | `{model.provider}` | `{model.transport}` | `{model.status}` |"
        for model in sorted(models.values(), key=lambda item: item.id)
    )
    tools = "\n".join(
        f"- `{tool['name']}` — {tool['description']}" for tool in MCP_TOOLS
    )
    component_rows = "\n".join(
        f"- `{pin.name}` current `{pin.version}`; channel=`{pin.update_channel}`; "
        f"policy=`{pin.update_policy}`; active={str(pin.active).lower()}"
        for pin in sorted(pins.values(), key=lambda item: item.name)
    )
    generated = root / "docs" / "generated"
    documents = {
        generated / "quickstart.md": (
            "# AI-Orchestrator quickstart\n\n"
            "Run `ai-orchestrator status`, then `ai-orchestrator doctor --check readiness`. "
            "For CHUNSA, simulate an explicit TaskSpec before `task submit`; automated "
            "writers return an artifact that must pass gates and be accepted explicitly.\n\n"
            "Harness is loopback-only at `127.0.0.1:3080` and starts only from a promoted "
            "immutable slot that has passed compatibility and security validation.\n"
        ),
        generated / "commands.md": "# Command reference\n\n" + commands + "\n",
        generated / "models.md": (
            "# Model matrix\n\n| Model | Provider | Transport | Status |\n"
            "|---|---|---|---|\n" + model_rows + "\n\n## Components\n\n" + component_rows + "\n"
        ),
        generated / "mcp.md": (
            "# Strict CHUNSA MCP\n\nThe server is bound to profile `chunsa` and exposes exactly four tools.\n\n"
            + tools
            + "\n"
        ),
        generated / "troubleshooting.md": (
            "# Troubleshooting\n\n"
            "Use `doctor --check readiness` for runtime dependencies and "
            "`doctor --check compliance` for policy drift. `AUTH_*`, `QUOTA_*`, "
            "and `ROUTE_*` do not permit silent fallback. `HARNESS_*` keeps the "
            "service disabled until the active slot and both profile compositions pass.\n"
        ),
    }
    documents.update(layer_documents(root))
    return documents


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def generate_docs(
    root: Path, *, check: bool = False, config_dir: Path | None = None
) -> dict[str, Any]:
    expected = _documents(root.resolve(), config_dir)
    stale = [
        str(path.relative_to(root))
        for path, content in expected.items()
        if not path.is_file() or path.read_text(encoding="utf-8") != content
    ]
    if not check:
        for path, content in expected.items():
            _atomic_write(path, content)
        stale = []
    return {
        "schema_version": 1,
        "ok": not stale,
        "mode": "check" if check else "generate",
        "root": str(root),
        "config_root": str(config_dir or config_root()),
        "stale": stale,
        "files": [str(path.relative_to(root)) for path in expected],
    }
