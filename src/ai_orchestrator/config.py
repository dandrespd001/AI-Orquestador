from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from ai_orchestrator.models import (
    ALLOWED_EXECUTION_PROFILES,
    ALLOWED_MODEL_CAPABILITIES,
    ALLOWED_REASONING_EFFORTS,
    ALLOWED_TRUST_TIERS,
    ContractError,
    ModelConfig,
    ModelEconomics,
    Profile,
    RoleConfig,
    sensitivity_rank,
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
        economics = None
        if raw.get("plan") is not None or raw.get("billing_mode") is not None:
            economics = ModelEconomics(
                plan=str(raw.get("plan", "unknown")),
                billing_mode=str(raw.get("billing_mode", "unknown")),
                input_per_mtok_usd=float(raw.get("input_per_mtok_usd", 0.0)),
                cache_read_per_mtok_usd=float(
                    raw.get("cache_read_per_mtok_usd", 0.0)
                ),
                cache_write_per_mtok_usd=float(
                    raw.get("cache_write_per_mtok_usd", 0.0)
                ),
                output_per_mtok_usd=float(raw.get("output_per_mtok_usd", 0.0)),
                source=str(raw.get("economics_source", "unknown")),
                verified_at=(
                    str(raw["economics_verified_at"])
                    if raw.get("economics_verified_at") is not None
                    else None
                ),
                confidence=str(raw.get("economics_confidence", "unknown")),
                orchestrator_quota_fraction=(
                    float(raw["orchestrator_quota_fraction"])
                    if raw.get("orchestrator_quota_fraction") is not None
                    else None
                ),
            )
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
            capabilities=tuple(
                str(value) for value in raw.get("capabilities", ("read",))
            ),
            execution_profile=str(raw.get("execution_profile", "read_only")),
            base_url=str(raw["base_url"]) if raw.get("base_url") else None,
            quality_score=float(raw.get("quality_score", 0.5)),
            included_quota=bool(raw.get("included_quota", True)),
            marginal_cost_per_million=float(raw.get("marginal_cost_per_million", 0.0)),
            context_window=int(raw.get("context_window", 120_000)),
            cache_supported=bool(raw.get("cache_supported", False)),
            observed_latency_ms=(
                int(raw["observed_latency_ms"])
                if raw.get("observed_latency_ms") is not None
                else None
            ),
            available_until=(
                str(raw["available_until"])
                if raw.get("available_until") is not None
                else None
            ),
            economics=economics,
            rollout_stage=str(raw.get("rollout_stage", "promoted")),
            candidate_since=(
                str(raw["candidate_since"])
                if raw.get("candidate_since") is not None
                else None
            ),
        )
        if model.trust_tier not in ALLOWED_TRUST_TIERS:
            raise ContractError(f"invalid trust_tier for {model.id}")
        if model.trust_tier == "official_subscription" and not model.official:
            raise ContractError(
                f"official subscription must set official=true: {model.id}"
            )
        if (
            not model.capabilities
            or set(model.capabilities) - ALLOWED_MODEL_CAPABILITIES
        ):
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
        if not 0.0 <= model.quality_score <= 1.0:
            raise ContractError(f"invalid quality_score for {model.id}")
        if model.marginal_cost_per_million < 0:
            raise ContractError(f"invalid marginal cost for {model.id}")
        if model.context_window < 512:
            raise ContractError(f"invalid context_window for {model.id}")
        if model.observed_latency_ms is not None and model.observed_latency_ms < 0:
            raise ContractError(f"invalid observed_latency_ms for {model.id}")
        if model.available_until is not None:
            try:
                date.fromisoformat(model.available_until)
            except ValueError as exc:
                raise ContractError(f"invalid available_until for {model.id}") from exc
        if model.rollout_stage not in {
            "candidate",
            "shadow",
            "read_only",
            "patch_writer",
            "promoted",
        }:
            raise ContractError(f"invalid rollout_stage for {model.id}")
        if model.candidate_since is not None:
            try:
                date.fromisoformat(model.candidate_since)
            except ValueError as exc:
                raise ContractError(
                    f"invalid candidate_since for {model.id}"
                ) from exc
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
            reviewer_reasoning_effort=str(raw.get("reviewer_reasoning_effort", "high")),
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
        allow_paid_routes=bool(raw.get("allow_paid_routes", False)),
        max_marginal_cost_usd=float(raw.get("max_marginal_cost_usd", 0.0)),
        capability_packages=tuple(
            str(value) for value in raw.get("capability_packages", ())
        ),
        extensions=tuple(str(value) for value in raw.get("extensions", ())),
        architect_model=(
            str(raw["architect_model"]) if raw.get("architect_model") else None
        ),
        assurance_floor=str(raw.get("assurance_floor", "standard")),
        workspace_allowlist=tuple(
            str(value) for value in raw.get("workspace_allowlist", ())
        ),
        session_retention_days=int(raw.get("session_retention_days", 30)),
        codex_daily_limit=int(raw.get("codex_daily_limit", 5)),
        codex_important_only=bool(raw.get("codex_important_only", False)),
        worker_normal_max_turns=int(raw.get("worker_normal_max_turns", 100)),
        worker_normal_max_tools=int(raw.get("worker_normal_max_tools", 200)),
        worker_normal_timeout_seconds=int(
            raw.get("worker_normal_timeout_seconds", 3600)
        ),
        worker_extended_max_turns=int(raw.get("worker_extended_max_turns", 300)),
        worker_extended_max_tools=int(raw.get("worker_extended_max_tools", 600)),
        worker_extended_timeout_seconds=int(
            raw.get("worker_extended_timeout_seconds", 14_400)
        ),
        deepseek_models=tuple(str(value) for value in raw.get("deepseek_models", ())),
        max_writers_per_repo=int(raw.get("max_writers_per_repo", 1)),
    )
    for sensitivity in profile.auto_execute_sensitivities:
        sensitivity_rank(sensitivity)
    if profile.minimax_mode not in {"direct", "delegate-minimax-api", "disabled"}:
        raise ContractError("invalid minimax_mode")
    if profile.max_marginal_cost_usd < 0:
        raise ContractError("max_marginal_cost_usd must be non-negative")
    if profile.assurance_floor not in {"standard", "high", "critical"}:
        raise ContractError("invalid assurance_floor")
    if not 1 <= profile.session_retention_days <= 30:
        raise ContractError("session_retention_days must be between 1 and 30")
    if not 1 <= profile.codex_daily_limit <= 100:
        raise ContractError("codex_daily_limit must be between 1 and 100")
    if profile.max_writers_per_repo != 1:
        raise ContractError("max_writers_per_repo must be exactly 1")
    if set(profile.deepseek_models) - {"deepseek-v4-pro", "deepseek-v4-flash"}:
        raise ContractError("profile contains an unsupported DeepSeek Harness model")
    return profile


def validate_configuration(
    profile_name: str, root: Path | None = None
) -> dict[str, object]:
    models = load_models(root)
    roles = load_roles(root)
    profile = load_profile(profile_name, root)
    gates = load_gates(root)
    package_payload = _read_toml((root or config_root()) / "capability-packages.toml")
    package_rows = package_payload.get("packages")
    package_names = set(package_rows) if isinstance(package_rows, dict) else set()
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
    for package in profile.capability_packages:
        if package not in package_names:
            errors.append(f"profile references unknown capability package: {package}")
    if profile.architect_model is not None:
        architecture = roles.get("architecture")
        if profile.architect_model not in models:
            errors.append(
                f"profile references unknown architect model: {profile.architect_model}"
            )
        elif profile.architect_model not in profile.allowed_models:
            errors.append("architect model is not allowed by the profile")
        if architecture is None or profile.architect_model not in architecture.models:
            errors.append("architect model is not assigned to the architecture role")
    return {
        "ok": not errors,
        "profile": profile.name,
        "model_count": len(models),
        "role_count": len(roles),
        "gate_profile_count": len(gates),
        "errors": errors,
    }
