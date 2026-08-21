"""Secret-free facade shared by the panel, CLI, broker and MCP server."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from ai_orchestrator.capabilities import load_capability_packages
from ai_orchestrator.config import (
    config_root,
    load_models,
    load_profile,
    load_roles,
    validate_configuration,
)
from ai_orchestrator.credentials import CredentialStore, authentication_inventory
from ai_orchestrator.extensions import ExtensionRegistry
from ai_orchestrator.models import ContractError, TaskSpec
from ai_orchestrator.provider_adapters import (
    adapter_inventory,
    validate_adapter_catalog,
)
from ai_orchestrator.providers import provider_inventory, validate_provider_catalog
from ai_orchestrator.routing import plan_route
from ai_orchestrator.sessions import SessionStore
from ai_orchestrator.storage import disable_omniroute_content_logging
from ai_orchestrator.telemetry import TelemetryStore, default_state_root
from ai_orchestrator.updates import UpdateRegistry, load_component_pins


class ControlPlane:
    def __init__(
        self,
        *,
        configuration_root: Path | None = None,
        state_root: Path | None = None,
    ) -> None:
        self.config_root = (configuration_root or config_root()).expanduser().resolve()
        self.state_root = (state_root or default_state_root()).expanduser().resolve()

    def providers(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "providers": provider_inventory(self.config_root),
            "adapters": adapter_inventory()["adapters"],
        }

    def credentials(self, profile: str = "default") -> dict[str, Any]:
        return {
            "schema_version": 1,
            "profiles": authentication_inventory(self.config_root, profile=profile),
            "secret_values_included": False,
            "write_only": True,
        }

    def models(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "models": [
                dataclasses.asdict(model)
                for model in sorted(
                    load_models(self.config_root).values(), key=lambda row: row.id
                )
            ],
        }

    def profiles(self) -> dict[str, Any]:
        directory = self.config_root / "profiles"
        rows: list[dict[str, Any]] = []
        for path in sorted(directory.glob("*.toml")):
            if path.is_symlink():
                continue
            rows.append(dataclasses.asdict(load_profile(path.stem, self.config_root)))
        return {"schema_version": 1, "profiles": rows}

    def routes(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "simulation_endpoint": "/api/routes/simulate",
            "hard_gates": [
                "trust",
                "sensitivity",
                "write_capability",
                "quota",
                "incremental_spend",
                "permission_scope",
            ],
            "score_factors": [
                "observed_quality",
                "included_quota",
                "marginal_cost",
                "context_window",
                "cache",
                "latency",
            ],
            "silent_fallback": False,
        }

    def simulate_route(
        self,
        task: dict[str, Any],
        *,
        profile: str = "default",
        observations: dict[str, dict[str, object]] | None = None,
        pin: str | None = None,
    ) -> dict[str, Any]:
        telemetry = TelemetryStore(self.state_root / "telemetry.sqlite")
        result = plan_route(
            TaskSpec.from_mapping(task),
            load_models(self.config_root),
            load_roles(self.config_root),
            load_profile(profile, self.config_root),
            observations=(
                observations
                if observations is not None
                else telemetry.routing_observations()
            ),
            pin=pin,
        )
        return result.as_dict()

    def usage(self) -> dict[str, Any]:
        return TelemetryStore(self.state_root / "telemetry.sqlite").report()

    def extensions(self, profile: str | None = None) -> dict[str, Any]:
        return ExtensionRegistry(self.state_root / "extensions").inventory(
            profile=profile
        )

    def updates(self) -> dict[str, Any]:
        payload = UpdateRegistry(self.state_root / "updates").inventory()
        payload["pins"] = [
            pin.as_dict()
            for pin in sorted(
                load_component_pins(self.config_root).values(), key=lambda row: row.name
            )
        ]
        return payload

    def sessions(self) -> dict[str, Any]:
        return SessionStore(self.state_root / "harness-sessions.sqlite").inventory()

    def capability_packages(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "packages": [
                package.as_dict()
                for package in sorted(
                    load_capability_packages(self.config_root).values(),
                    key=lambda row: row.name,
                )
            ],
        }

    def diagnostics(self, profile: str = "default") -> dict[str, Any]:
        return {
            "schema_version": 1,
            "configuration": validate_configuration(profile, self.config_root),
            "providers": validate_provider_catalog(self.config_root),
            "adapters": validate_adapter_catalog(self.config_root),
            "omniroute_content_purge_preview": disable_omniroute_content_logging(
                apply=False
            ),
            "secret_read_api_exposed": False,
            "automatic_purchases": False,
        }

    def store_credential(
        self,
        provider: str,
        name: str,
        value: str,
        *,
        profile: str = "default",
        human_approved: bool = False,
    ) -> dict[str, object]:
        return CredentialStore(self.config_root).store(
            name,
            value,
            provider=provider,
            profile=profile,
            human_approved=human_approved,
        )

    def activate_extension(
        self,
        extension_id: str,
        profile: str,
        *,
        human_approved: bool = False,
        host_risk_approved: bool = False,
    ) -> dict[str, Any]:
        return ExtensionRegistry(self.state_root / "extensions").activate(
            extension_id,
            profile,
            human_approved=human_approved,
            host_risk_approved=host_risk_approved,
        )

    def query(self, section: str, **params: object) -> dict[str, Any]:
        methods = {
            "providers": self.providers,
            "credentials": self.credentials,
            "models": self.models,
            "profiles": self.profiles,
            "routes": self.routes,
            "usage": self.usage,
            "extensions": self.extensions,
            "updates": self.updates,
            "sessions": self.sessions,
            "capabilities": self.capability_packages,
            "diagnostics": self.diagnostics,
        }
        if section not in methods:
            raise ContractError(f"unknown control-plane section: {section}")
        return methods[section](**params)  # type: ignore[call-arg]
