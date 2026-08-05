from __future__ import annotations

import os
import tomllib
from pathlib import Path

from ai_orchestrator.models import (
    ContractError,
    ModelConfig,
    Profile,
    RoleConfig,
    sensitivity_rank,
)


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
        )
        if not model.official:
            raise ContractError(f"non-official model forbidden: {model.id}")
        sensitivity_rank(model.max_sensitivity)
        if model.id in result:
            raise ContractError(f"duplicate model id: {model.id}")
        if model.max_concurrency < 1:
            raise ContractError(f"invalid concurrency for {model.id}")
        result[model.id] = model
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
        result[str(name)] = RoleConfig(
            name=str(name),
            models=tuple(str(value) for value in raw.get("models", ())),
            reviewers=tuple(str(value) for value in raw.get("reviewers", ())),
        )
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
        minimax_mode=str(raw.get("minimax_mode", "omniroute")),
        preserve_live_trading_false=bool(raw.get("preserve_live_trading_false", False)),
    )
    for sensitivity in profile.auto_execute_sensitivities:
        sensitivity_rank(sensitivity)
    if profile.minimax_mode not in {"omniroute", "delegate-minimax-api", "disabled"}:
        raise ContractError("invalid minimax_mode")
    return profile


def validate_configuration(profile_name: str) -> dict[str, object]:
    models = load_models()
    roles = load_roles()
    profile = load_profile(profile_name)
    errors: list[str] = []
    for model_id in profile.allowed_models:
        if model_id not in models:
            errors.append(f"profile references unknown model: {model_id}")
    for role in roles.values():
        for model_id in role.models + role.reviewers:
            if model_id not in models:
                errors.append(f"role {role.name} references unknown model: {model_id}")
    return {
        "ok": not errors,
        "profile": profile.name,
        "model_count": len(models),
        "role_count": len(roles),
        "errors": errors,
    }

