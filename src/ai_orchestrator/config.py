from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from ai_orchestrator.models import (
    ContractError,
    ModelConfig,
    Profile,
    RoleConfig,
    sensitivity_rank,
    ALLOWED_EXECUTION_PROFILES,
    ALLOWED_MODEL_CAPABILITIES,
    ALLOWED_REASONING_EFFORTS,
    ALLOWED_TRUST_TIERS,
)


@dataclass(frozen=True)
class GateProfile:
    name: str
    commands: tuple[str, ...]
    timeout_seconds: int = 300


def config_root() -> Path:
    override = os.environ.get("AI_ORCHESTRATOR_CONFIG")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".config" / "ai-orchestrator"


def _read_toml(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ContractError(f"missing or unsafe config: {path}")
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    if payload.get("schema_version") != 1:
        raise ContractError(f"unsupported schema_version in {path}")
    return payload


def load_models(root: Path | None = None) -> dict[str, ModelConfig]:
    source = (root or config_root()) / "models.toml"
    payload = _read_toml(source)
    rows = payload.get("models")
    if not isinstance(rows, list):
        raise ContractError("models.toml must define [[models]]")
    result: dict[str, ModelConfig] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            raise ContractError("model entry must be a table")
        model = ModelConfig(
            id=str(raw["id"]),
            family=str(raw["family"]),
            provider=str(raw["provider"]),
            transport=str(raw["transport"]),
            route_model=str(raw["route_model"]),
            official=bool(raw["official"]),
            status=str(raw["status"]),
            roles=tuple(str(value) for value in raw.get("roles", ())),
            max_sensitivity=str(raw["max_sensitivity"]),
            max_concurrency=int(raw["max_concurrency"]),
            trust_tier=str(raw.get("trust_tier", "official_subscription")),
            capabilities=tuple(str(value) for value in raw.get("capabilities", ("read",))),
            execution_profile=str(raw.get("execution_profile", "read_only")),
            base_url=str(raw["base_url"]) if raw.get("base_url") else None,
        )
        if model.trust_tier not in ALLOWED_TRUST_TIERS:
            raise ContractError(f"invalid trust_tier for {model.id}")
        if model.trust_tier == "official_subscription" and not model.official:
            raise ContractError(f"official subscription must set official=true: {model.id}")
        if not model.capabilities or set(model.capabilities) - ALLOWED_MODEL_CAPABILITIES:
            raise ContractError(f"invalid capabilities for {model.id}")
        if "write" in model.capabilities and model.execution_profile == "read_only":
            raise ContractError(f"write-capable model lacks writer profile: {model.id}")
        if model.execution_profile not in ALLOWED_EXECUTION_PROFILES:
            raise ContractError(f"invalid execution_profile for {model.id}")
        sensitivity_rank(model.max_sensitivity)
        if model.id in result:
            raise ContractError(f"duplicate model id: {model.id}")
        if model.max_concurrency < 1:
            raise ContractError(f"invalid concurrency for {model.id}")
        result[model.id] = model
    return result


def load_gates(root: Path | None = None) -> dict[str, GateProfile]:
    source = (root or config_root()) / "gates.toml"
    payload = _read_toml(source)
    raw_profiles = payload.get("gates")
    if not isinstance(raw_profiles, dict):
        raise ContractError("gates.toml must define [gates.*]")
    result: dict[str, GateProfile] = {}
    for name, raw in raw_profiles.items():
        if not isinstance(raw, dict):
            raise ContractError(f"gate profile {name} must be a table")
        commands = tuple(str(value).strip() for value in raw.get("commands", ()))
        if any(not value for value in commands):
            raise ContractError(f"gate profile {name} contains an empty command")
        timeout_seconds = int(raw.get("timeout_seconds", 300))
        if not 1 <= timeout_seconds <= 1800:
            raise ContractError(
                f"gate profile {name} timeout_seconds must be between 1 and 1800"
            )
        result[str(name)] = GateProfile(str(name), commands, timeout_seconds)
    return result


def load_roles(root: Path | None = None) -> dict[str, RoleConfig]:
    source = (root or config_root()) / "roles.toml"
    payload = _read_toml(source)
    raw_roles = payload.get("roles")
    if not isinstance(raw_roles, dict):
        raise ContractError("roles.toml must define [roles.*]")
    result: dict[str, RoleConfig] = {}
    for name, raw in raw_roles.items():
        if not isinstance(raw, dict):
            raise ContractError(f"role {name} must be a table")
        role = RoleConfig(
            name=str(name),
            models=tuple(str(value) for value in raw.get("models", ())),
            reviewers=tuple(str(value) for value in raw.get("reviewers", ())),
            reasoning_effort=str(raw.get("reasoning_effort", "high")),
            reviewer_reasoning_effort=str(
                raw.get("reviewer_reasoning_effort", "high")
            ),
        )
        if role.reasoning_effort not in ALLOWED_REASONING_EFFORTS:
            raise ContractError(f"invalid reasoning_effort for role {name}")
        if role.reviewer_reasoning_effort not in ALLOWED_REASONING_EFFORTS:
            raise ContractError(f"invalid reviewer_reasoning_effort for role {name}")
        result[str(name)] = role
    return result


def load_profile(name: str, root: Path | None = None) -> Profile:
    source = (root or config_root()) / "profiles" / f"{name}.toml"
    raw = _read_toml(source)
    profile = Profile(
        name=str(raw["name"]),
        repo_root=str(raw["repo_root"]) if raw.get("repo_root") else None,
        auto_execute_sensitivities=tuple(
            str(value) for value in raw.get("auto_execute_sensitivities", ())
        ),
        allowed_models=tuple(str(value) for value in raw.get("allowed_models", ())),
        deny_providers=tuple(str(value) for value in raw.get("deny_providers", ())),
        protected_path_prefixes=tuple(
            str(value).strip("/") for value in raw.get("protected_path_prefixes", ())
        ),
        deterministic_gates=tuple(
            str(value) for value in raw.get("deterministic_gates", ())
        ),
        minimax_mode=str(raw.get("minimax_mode", "direct")),
        preserve_live_trading_false=bool(raw.get("preserve_live_trading_false", False)),
        public_repository=bool(raw.get("public_repository", False)),
    )
    for sensitivity in profile.auto_execute_sensitivities:
        sensitivity_rank(sensitivity)
    if profile.minimax_mode not in {"direct", "delegate-minimax-api", "disabled"}:
        raise ContractError("invalid minimax_mode")
    return profile


def validate_configuration(profile_name: str) -> dict[str, object]:
    models = load_models()
    roles = load_roles()
    profile = load_profile(profile_name)
    gates = load_gates()
    errors: list[str] = []
    for model_id in profile.allowed_models:
        if model_id not in models:
            errors.append(f"profile references unknown model: {model_id}")
    for role in roles.values():
        for model_id in role.models + role.reviewers:
            if model_id not in models:
                errors.append(f"role {role.name} references unknown model: {model_id}")
    if "profile" in gates:
        errors.append("gate profile name 'profile' is reserved")
    return {
        "ok": not errors,
        "profile": profile.name,
        "model_count": len(models),
        "role_count": len(roles),
        "gate_profile_count": len(gates),
        "errors": errors,
    }
