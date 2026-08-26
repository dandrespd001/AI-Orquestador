"""Task-scoped capability packages that keep tool context bounded."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ai_orchestrator.config import _read_toml, config_root
from ai_orchestrator.models import ContractError


@dataclass(frozen=True)
class CapabilityPackage:
    name: str
    description: str
    tools: tuple[str, ...]
    extensions: tuple[str, ...]
    clients: tuple[str, ...]
    estimated_context_tokens: int
    license_policy: str = "free_open_source"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_capability_packages(root: Path | None = None) -> dict[str, CapabilityPackage]:
    payload = _read_toml((root or config_root()) / "capability-packages.toml")
    rows = payload.get("packages")
    if not isinstance(rows, dict):
        raise ContractError("capability-packages.toml must define [packages.*]")
    result: dict[str, CapabilityPackage] = {}
    for name, raw in rows.items():
        if not isinstance(raw, dict):
            raise ContractError(f"capability package {name} must be a table")
        package = CapabilityPackage(
            name=str(name),
            description=str(raw.get("description", "")),
            tools=tuple(str(value) for value in raw.get("tools", ())),
            extensions=tuple(str(value) for value in raw.get("extensions", ())),
            clients=tuple(str(value) for value in raw.get("clients", ())),
            estimated_context_tokens=int(raw.get("estimated_context_tokens", 0)),
            license_policy=str(raw.get("license_policy", "free_open_source")),
        )
        if not package.description or package.estimated_context_tokens < 1:
            raise ContractError(f"capability package {name} is incomplete")
        if package.license_policy != "free_open_source":
            raise ContractError(
                f"capability package {name} violates the license policy"
            )
        if len(set(package.tools)) != len(package.tools):
            raise ContractError(f"capability package {name} has duplicate tools")
        result[package.name] = package
    return result


def select_capability_package(
    name: str,
    *,
    client: str,
    root: Path | None = None,
) -> dict[str, Any]:
    packages = load_capability_packages(root)
    if name not in packages:
        raise ContractError(f"unknown capability package: {name}")
    package = packages[name]
    if package.clients and client not in package.clients:
        raise ContractError(f"capability package {name} is incompatible with {client}")
    total = sum(value.estimated_context_tokens for value in packages.values())
    selected = package.estimated_context_tokens
    reduction = 0.0 if total == 0 else (total - selected) / total * 100.0
    return {
        "schema_version": 1,
        "client": client,
        "loaded_package_count": 1,
        "package": package.as_dict(),
        "estimated_context_tokens": selected,
        "all_packages_context_tokens": total,
        "context_reduction_percent": round(reduction, 2),
    }
