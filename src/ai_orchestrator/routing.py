from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, date, datetime, timedelta
from pathlib import Path, PurePosixPath

from ai_orchestrator.models import (
    ASSURANCE_ORDER,
    ModelConfig,
    ModelEconomics,
    Profile,
    QuotaSnapshot,
    RoleConfig,
    RouteDecision,
    RoutePlan,
    TaskSpec,
    sensitivity_rank,
)

NATIVE_TRANSPORTS = {"codex_cli", "claude_cli"}
HANDOFF_SENSITIVITIES = {"confidential", "secrets", "financial_control", "deployment"}


def _bounded_score(value: object, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, result))


def score_route_candidate(
    model: ModelConfig,
    task: TaskSpec,
    *,
    role_priority: int,
    role_size: int,
    observation: dict[str, object] | None = None,
) -> float:
    """Score an already-policy-eligible model on quality, quota and efficiency."""

    observed = observation or {}
    quality = _bounded_score(
        observed.get("quality", model.quality_score), model.quality_score
    )
    quota = _bounded_score(
        observed.get("quota_available", 1.0 if model.included_quota else 0.0),
        1.0 if model.included_quota else 0.0,
    )
    marginal_cost = max(
        0.0,
        float(
            observed.get("marginal_cost_per_million", model.marginal_cost_per_million)
        ),
    )
    cost = 1.0 / (1.0 + marginal_cost * 10.0)
    context_window = max(1, int(observed.get("context_window", model.context_window)))
    context = min(1.0, context_window / max(1, task.max_context_tokens))
    cache = _bounded_score(
        observed.get("cache_hit_rate", 1.0 if model.cache_supported else 0.0),
        1.0 if model.cache_supported else 0.0,
    )
    latency_ms = observed.get("latency_ms", model.observed_latency_ms)
    latency = (
        0.5
        if latency_ms is None
        else 1.0 / (1.0 + max(0.0, float(latency_ms)) / 10_000.0)
    )
    # Role order is an explicit operator preference and a deterministic tie
    # breaker, but it cannot outweigh measured quality/quota/cost.
    priority = (role_size - role_priority) / max(1, role_size)
    return round(
        quality * 0.42
        + quota * 0.20
        + cost * 0.16
        + context * 0.10
        + cache * 0.05
        + latency * 0.05
        + priority * 0.02,
        6,
    )


def plan_route(
    task: TaskSpec,
    models: dict[str, ModelConfig],
    roles: dict[str, RoleConfig],
    profile: Profile,
    *,
    observations: dict[str, dict[str, object]] | None = None,
    quotas: dict[str, QuotaSnapshot] | None = None,
    rollout_counts: dict[str, int] | None = None,
    pin: str | None = None,
    at: datetime | None = None,
    codex_uses_today: int = 0,
) -> RoutePlan:
    """Build the immutable plan used by simulation and execution.

    Policy gates run before ranking.  Quality uses a conservative interval;
    overlapping intervals are a tie resolved by cost per accepted delivery,
    then quota consumption, latency and the configured role order.
    """

    clock = at or datetime.now(UTC)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=UTC)
    role = roles.get(task.task_type)
    assurance = max(
        (task.assurance, profile.assurance_floor), key=ASSURANCE_ORDER.__getitem__
    )
    observed = observations or {}
    snapshots = quotas or {}
    shadow_counts = rollout_counts or {}
    snapshots_provided = quotas is not None
    legacy_v1_selection = observations is None and quotas is None
    protected = _is_protected(task, profile)
    unavailable: dict[str, str] = {}
    candidates: list[dict[str, object]] = []

    if profile.repo_root and task.repo != str(Path(profile.repo_root).resolve()):
        return _empty_plan(
            task,
            profile,
            assurance,
            "profile repo_root does not match task repository",
            unavailable,
            observed,
            snapshots,
            shadow_counts,
        )
    if role is None:
        return _empty_plan(
            task,
            profile,
            assurance,
            f"unknown task_type: {task.task_type}",
            unavailable,
            observed,
            snapshots,
            shadow_counts,
        )

    for index, model_id in enumerate(role.models):
        model = models[model_id]
        if pin is not None and model_id != pin:
            unavailable[model.id] = "not_pinned"
            continue
        if model.provider == "abacus" and pin is None:
            unavailable[model.id] = "alternative_or_second_opinion_only"
            continue
        if model.provider == "openai" and profile.codex_important_only:
            if not task.important:
                unavailable[model.id] = "codex_requires_important_task"
                continue
            if codex_uses_today >= profile.codex_daily_limit:
                unavailable[model.id] = "codex_daily_limit"
                continue
        if (
            task.task_type == "architecture"
            and profile.architect_model is not None
            and model_id != profile.architect_model
        ):
            unavailable[model.id] = "profile_architect_pinned"
            continue
        eligible, reason = _eligible(
            model, task, profile, protected=protected, at=clock
        )
        if not eligible:
            unavailable[model.id] = reason
            continue

        economics = _model_economics(model)
        observation = observed.get(model.id, {})
        estimated = _estimated_cost(task, economics, observation)
        quota = snapshots.get(model.provider)
        quota_ok, quota_reason, quota_ratio = _quota_allows(
            model,
            economics,
            quota,
            estimated,
            at=clock,
            snapshots_provided=snapshots_provided,
        )
        if not quota_ok:
            unavailable[model.id] = quota_reason
            continue
        paid = economics.billing_mode in {"payg", "credits"}
        if paid and not profile.allow_paid_routes:
            unavailable[model.id] = "incremental_spend_denied"
            continue
        if paid and estimated > profile.max_marginal_cost_usd:
            unavailable[model.id] = "marginal_cost_exceeded"
            continue

        lower, upper, quality = _quality_interval(model, observation)
        acceptance = max(
            0.000001,
            _bounded_score(observation.get("acceptance_rate", quality), quality),
        )
        latency_value = observation.get("latency_ms", model.observed_latency_ms)
        latency = (
            float("inf")
            if latency_value is None
            else max(0.0, float(latency_value))
        )
        candidates.append(
            {
                "model": model,
                "index": index,
                "quality": quality,
                "lower": lower,
                "upper": upper,
                "estimated": estimated,
                "cost_per_accepted": estimated / acceptance,
                "quota_ratio": quota_ratio,
                "latency": latency,
            }
        )

    if pin is not None and pin not in role.models:
        unavailable[pin] = "pin_outside_role"
    if not candidates:
        return _empty_plan(
            task,
            profile,
            assurance,
            "no model satisfies security, quota and budget policy",
            unavailable,
            observed,
            snapshots,
            shadow_counts,
        )

    if legacy_v1_selection:
        # Compatibility for direct v1 callers.  Every v2 surface supplies a
        # quota snapshot collection (possibly empty), so simulation and run
        # use the measured quality/cost ranking below.
        selected_row = min(candidates, key=lambda row: int(row["index"]))
    else:
        best_lower = max(float(row["lower"]) for row in candidates)
        leader = min(
            (row for row in candidates if float(row["lower"]) == best_lower),
            key=lambda row: int(row["index"]),
        )
        overlapping = [
            row
            for row in candidates
            if float(row["lower"]) <= float(leader["upper"])
            and float(leader["lower"]) <= float(row["upper"])
        ]
        selected_row = min(
            overlapping,
            key=lambda row: (
                float(row["cost_per_accepted"]),
                float(row["quota_ratio"]),
                float(row["latency"]),
                int(row["index"]),
            ),
        )
    selected = selected_row["model"]
    assert isinstance(selected, ModelConfig)
    ranked = sorted(
        candidates,
        key=lambda row: (
            row is not selected_row,
            -float(row["lower"]),
            float(row["cost_per_accepted"]),
            float(row["quota_ratio"]),
            float(row["latency"]),
            int(row["index"]),
        ),
    )
    reviewers = _reviewers_for(
        role,
        selected,
        models,
        profile,
        task=task,
        protected=protected,
        observations=observed,
        quotas=snapshots,
        snapshots_provided=snapshots_provided,
        rollout_counts=shadow_counts,
        strict_v2=not legacy_v1_selection,
        unavailable=unavailable,
        assurance=assurance,
        at=clock,
    )
    independent_reviewer_count = len(reviewers)
    if (
        profile.name == "chunsa"
        and task.task_type == "architecture"
        and assurance in {"high", "critical"}
        and selected.id not in reviewers
    ):
        # Claude proposes, Codex reviews, Grok gives the independent second
        # opinion, and the Claude authority emits the final disposition.
        reviewers = (*reviewers, selected.id)
    transport = selected.transport
    if selected.id == "MiniMax-M3" and profile.minimax_mode == "delegate-minimax-api":
        transport = "delegate-minimax-api"
    bridge_handoff = transport == "delegate-minimax-api"
    codex_implementation = (
        selected.provider == "openai"
        and task.task_type
        in {"implementation", "tests", "docs", "boilerplate", "bulk_refactor", "bounded_cpp"}
    )
    reviewer_missing = independent_reviewer_count < _required_reviewer_count(
        task, profile, assurance
    )
    candidate_read_ready = _candidate_read_ready(
        selected, shadow_counts.get(selected.id, 0), clock
    )
    rollout_handoff = not legacy_v1_selection and (
        (
            selected.rollout_stage in {"candidate", "shadow"}
            and (not candidate_read_ready or task.write_mode != "read_only")
        )
        or (
            selected.rollout_stage == "read_only"
            and task.write_mode != "read_only"
        )
    )
    auto_execute = (
        task.sensitivity in profile.auto_execute_sensitivities
        and not protected
        and not bridge_handoff
        and not codex_implementation
        and not reviewer_missing
        and not rollout_handoff
    )
    confirmation_required = selected.id == "deepseek-v4-pro"
    if auto_execute and confirmation_required:
        status = "CONFIRMATION_REQUIRED"
    else:
        status = "READY" if auto_execute else "HANDOFF_REQUIRED"
    permissions = ["provider_network", "repository_read"]
    if task.write_mode != "read_only":
        permissions.append("reviewed_patch_write")
    if task.sensitivity != "public":
        permissions.append("non_public_context")
    reason_parts = ["selected by routing-v2 quality/cost policy"]
    if pin is not None:
        reason_parts.append(f"pin={pin}")
    if protected:
        reason_parts.append("protected input")
    if codex_implementation:
        reason_parts.append("Codex implementation requires handoff")
    if reviewer_missing:
        reason_parts.append("independent reviewer unavailable")
    if rollout_handoff:
        reason_parts.append("candidate shadow route")
    if confirmation_required:
        reason_parts.append("explicit confirmation required")

    quality_breakdown = {
        str(row["model"].id): {
            "quality": float(row["quality"]),
            "lower": float(row["lower"]),
            "upper": float(row["upper"]),
        }
        for row in candidates
    }
    cost_breakdown = {
        str(row["model"].id): {
            "estimated_usd": round(float(row["estimated"]), 8),
            "cost_per_accepted_usd": round(
                float(row["cost_per_accepted"]), 8
            ),
            "quota_ratio": float(row["quota_ratio"]),
            "latency_ms": (
                None if float(row["latency"]) == float("inf") else int(row["latency"])
            ),
        }
        for row in candidates
    }
    estimated = float(selected_row["estimated"])
    plan_payload = _plan_identity_payload(
        task,
        profile,
        selected.id,
        tuple(str(row["model"].id) for row in ranked[1:]),
        reviewers,
        assurance,
        observed,
        snapshots,
        {
            model.id: _candidate_read_ready(
                model, shadow_counts.get(model.id, 0), clock
            )
            for model in models.values()
            if model.rollout_stage in {"candidate", "shadow"}
        },
    )
    return RoutePlan(
        task_id=task.task_id,
        profile=profile.name,
        selected_model=selected.id,
        provider=selected.provider,
        alternatives=tuple(str(row["model"].id) for row in ranked[1:]),
        scores={model_id: values["lower"] for model_id, values in quality_breakdown.items()},
        quality_price_score=float(selected_row["lower"]),
        budget={
            "max_usd": profile.max_marginal_cost_usd,
            "estimated_usd": round(estimated, 8),
            "incremental_spend_allowed": profile.allow_paid_routes,
        },
        permissions=tuple(permissions),
        approval_required=status != "READY",
        reason="; ".join(reason_parts),
        plan_id=_stable_plan_id(plan_payload),
        fallback_allowed=False,
        status=status,
        unavailable=unavailable,
        assurance=assurance,
        quality_breakdown=quality_breakdown,
        cost_breakdown=cost_breakdown,
        reviewer_models=reviewers,
        predicted_quota=(
            snapshots[selected.provider].as_dict()
            if selected.provider in snapshots
            else {"provider": selected.provider, "known": False}
        ),
        confirmation_required=confirmation_required,
    )


def _empty_plan(
    task: TaskSpec,
    profile: Profile,
    assurance: str,
    reason: str,
    unavailable: dict[str, str],
    observations: dict[str, dict[str, object]],
    quotas: dict[str, QuotaSnapshot],
    rollout_counts: dict[str, int],
) -> RoutePlan:
    identity = _plan_identity_payload(
        task,
        profile,
        None,
        (),
        (),
        assurance,
        observations,
        quotas,
        rollout_counts,
    )
    return RoutePlan(
        task_id=task.task_id,
        profile=profile.name,
        selected_model=None,
        alternatives=(),
        scores={},
        quality_price_score=None,
        budget={
            "max_usd": profile.max_marginal_cost_usd,
            "estimated_usd": None,
            "incremental_spend_allowed": profile.allow_paid_routes,
        },
        permissions=(),
        approval_required=True,
        reason=reason,
        plan_id=_stable_plan_id(identity),
        status="HANDOFF_REQUIRED",
        unavailable=dict(unavailable),
        assurance=assurance,
        confirmation_required=True,
    )


def _stable_plan_id(payload: dict[str, object]) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return str(uuid.UUID(digest[:32]))


def _plan_identity_payload(
    task: TaskSpec,
    profile: Profile,
    selected: str | None,
    alternatives: tuple[str, ...],
    reviewers: tuple[str, ...],
    assurance: str,
    observations: dict[str, dict[str, object]],
    quotas: dict[str, QuotaSnapshot],
    rollout_state: dict[str, object],
) -> dict[str, object]:
    return {
        "policy_version": "routing-v2",
        "task_id": task.task_id,
        "context_sha256": task.context_sha256(),
        "profile": profile.name,
        "selected_model": selected,
        "alternatives": alternatives,
        "reviewers": reviewers,
        "assurance": assurance,
        "observations": observations,
        "quotas": {
            name: _quota_plan_identity(value)
            for name, value in sorted(quotas.items())
        },
        "rollout_state": dict(sorted(rollout_state.items())),
    }


def _quota_plan_identity(snapshot: QuotaSnapshot) -> dict[str, object]:
    """Keep material quota facts in a plan ID, excluding refresh metadata."""

    payload = snapshot.as_dict()
    payload.pop("verified_at", None)
    payload.pop("source", None)
    return payload


def _model_economics(model: ModelConfig) -> ModelEconomics:
    if model.economics is not None:
        return model.economics
    return ModelEconomics(
        plan="legacy_config",
        billing_mode="included_quota" if model.included_quota else "payg",
        input_per_mtok_usd=model.marginal_cost_per_million,
        output_per_mtok_usd=model.marginal_cost_per_million,
        source="legacy models.toml",
        confidence="configured",
    )


def _estimated_cost(
    task: TaskSpec,
    economics: ModelEconomics,
    observation: dict[str, object],
) -> float:
    input_tokens = max(
        0, int(observation.get("expected_input_tokens", task.max_context_tokens))
    )
    output_tokens = max(
        0, int(observation.get("expected_output_tokens", task.max_output_tokens))
    )
    cached_tokens = max(0, int(observation.get("expected_cached_tokens", 0)))
    cache_write = max(0, int(observation.get("expected_cache_write_tokens", 0)))
    return (
        input_tokens * economics.input_per_mtok_usd
        + output_tokens * economics.output_per_mtok_usd
        + cached_tokens * economics.cache_read_per_mtok_usd
        + cache_write * economics.cache_write_per_mtok_usd
    ) / 1_000_000.0


def _quality_interval(
    model: ModelConfig, observation: dict[str, object]
) -> tuple[float, float, float]:
    quality = _bounded_score(observation.get("quality", model.quality_score), model.quality_score)
    measured_interval = "quality_lower" in observation or "quality_upper" in observation
    uncertainty = 0.0 if measured_interval else 0.10
    lower = _bounded_score(
        observation.get("quality_lower", quality - uncertainty), quality
    )
    upper = _bounded_score(
        observation.get("quality_upper", quality + uncertainty), quality
    )
    if lower > upper:
        lower, upper = upper, lower
    return lower, upper, quality


def _verified_is_fresh(value: str | None, at: datetime) -> bool:
    if value is None:
        return False
    try:
        verified = datetime.fromisoformat(value)
    except ValueError:
        return False
    if verified.tzinfo is None:
        verified = verified.replace(tzinfo=UTC)
    age = at - verified.astimezone(UTC)
    return age.total_seconds() >= 0 and age.days <= 30


def _quota_allows(
    model: ModelConfig,
    economics: ModelEconomics,
    quota: QuotaSnapshot | None,
    estimated_cost: float,
    *,
    at: datetime,
    snapshots_provided: bool,
) -> tuple[bool, str, float]:
    if model.economics is not None and not _verified_is_fresh(economics.verified_at, at):
        return False, "economics_stale", 1.0
    if economics.orchestrator_quota_fraction is not None:
        if not snapshots_provided:
            return True, "legacy_snapshot_not_supplied", 0.0
        if quota is None or quota.limit is None or quota.used is None:
            return False, "quota_unknown", 1.0
        if not _verified_is_fresh(quota.verified_at, at):
            return False, "quota_stale", 1.0
        predicted = quota.predicted_usage or 0.0
        ceiling = quota.limit * economics.orchestrator_quota_fraction
        if quota.used + predicted > ceiling + 1e-12:
            return False, "orchestrator_quota_limit", 1.0
        return True, "eligible", (quota.used + predicted) / max(ceiling, 1e-12)
    if economics.billing_mode in {"prepaid_balance", "prepaid_balance_only"}:
        if not snapshots_provided:
            return True, "legacy_snapshot_not_supplied", 0.0
        if quota is None or quota.balance_usd is None:
            return False, "balance_unknown", 1.0
        if not _verified_is_fresh(quota.verified_at, at):
            return False, "quota_stale", 1.0
        if quota.balance_usd <= 0 or quota.balance_usd + 1e-12 < estimated_cost:
            return False, "balance_exhausted", 1.0
        return True, "eligible", estimated_cost / max(quota.balance_usd, 1e-12)
    if economics.billing_mode == "measured_credits":
        if not snapshots_provided:
            return True, "legacy_snapshot_not_supplied", 0.0
        if quota is None or quota.credits_remaining is None:
            return False, "quota_unknown", 1.0
        if not _verified_is_fresh(quota.verified_at, at):
            return False, "quota_stale", 1.0
        if quota.credits_remaining <= 0:
            return False, "quota_exhausted", 1.0
        predicted = quota.predicted_usage or 0.0
        return (
            True,
            "eligible",
            predicted / max(quota.credits_remaining, 1e-12),
        )
    if model.provider == "minimax" and model.economics is not None:
        if not snapshots_provided:
            return True, "legacy_snapshot_not_supplied", 0.0
        if quota is None:
            return False, "quota_unknown", 1.0
        if not _verified_is_fresh(quota.verified_at, at):
            return False, "quota_stale", 1.0
        if quota.billing_mode != "included_quota" or quota.extra_usage_enabled is True:
            return False, "incremental_spend_denied", 1.0
        if quota.extra_usage_enabled is not False:
            return False, "extra_usage_not_disabled", 1.0
    if model.provider == "moonshot" and model.economics is not None:
        if not snapshots_provided:
            return True, "legacy_snapshot_not_supplied", 0.0
        if quota is None:
            return False, "quota_unknown", 1.0
        if not _verified_is_fresh(quota.verified_at, at):
            return False, "quota_stale", 1.0
        if quota.extra_usage_enabled is not False:
            return False, "extra_usage_not_disabled", 1.0
        required_windows = {"monthly", "weekly", "five_hours"}
        if set(quota.windows) != required_windows or any(
            quota.windows[name] is None for name in required_windows
        ):
            return False, "quota_unknown", 1.0
        if any(float(quota.windows[name]) <= 0 for name in required_windows):
            return False, "quota_exhausted", 1.0
    if quota is not None and quota.remaining is not None:
        if quota.remaining <= 0:
            return False, "quota_exhausted", 1.0
        predicted = quota.predicted_usage or 0.0
        return True, "eligible", predicted / max(quota.remaining, 1e-12)
    return True, "eligible", 0.0


def _is_protected(task: TaskSpec, profile: Profile) -> bool:
    selected = (
        task.input_paths
        + task.allowed_write_paths
        + ((task.prompt_file,) if task.prompt_file else ())
    )
    for raw in selected:
        path = PurePosixPath(raw)
        for prefix in profile.protected_path_prefixes:
            protected = PurePosixPath(prefix)
            if path == protected or protected in path.parents:
                return True
    return False


def _eligible(
    model: ModelConfig,
    task: TaskSpec,
    profile: Profile,
    *,
    protected: bool,
    at: datetime | None = None,
    codex_uses_today: int = 0,
) -> tuple[bool, str]:
    if model.id not in profile.allowed_models:
        return False, "profile_denied"
    clock = at or datetime.now(UTC)
    if model.available_until is not None and clock.date() >= date.fromisoformat(
        model.available_until
    ):
        return False, "promotion_expired"
    if task.max_context_tokens > model.context_window:
        return False, "context_window_exceeded"
    if model.trust_tier == "web_free" and (
        task.sensitivity != "public" or not profile.public_repository
    ):
        return False, "web_free_requires_public_repository"
    if task.write_mode != "read_only" and "write" not in model.capabilities:
        return False, "model_not_write_capable"
    if model.provider in profile.deny_providers:
        return False, "provider_denied"
    if sensitivity_rank(task.sensitivity) > sensitivity_rank(model.max_sensitivity):
        return False, "sensitivity_exceeds_model_policy"
    if protected and model.transport not in NATIVE_TRANSPORTS:
        if model.id == "MiniMax-M3" and profile.minimax_mode == "delegate-minimax-api":
            return model.status == "available", "bridge_required"
        return False, "protected_input_requires_native_transport"
    if model.status != "available":
        return False, model.status
    return True, "eligible"


def _reviewer_for(
    role: RoleConfig,
    selected: ModelConfig,
    models: dict[str, ModelConfig],
    profile: Profile,
) -> str | None:
    for model_id in role.reviewers:
        reviewer = models[model_id]
        if reviewer.family == selected.family:
            continue
        if reviewer.id not in profile.allowed_models or reviewer.status != "available":
            continue
        if reviewer.provider in profile.deny_providers:
            continue
        return reviewer.id
    return None


def _reviewers_for(
    role: RoleConfig,
    selected: ModelConfig,
    models: dict[str, ModelConfig],
    profile: Profile,
    *,
    task: TaskSpec,
    protected: bool,
    observations: dict[str, dict[str, object]],
    quotas: dict[str, QuotaSnapshot],
    snapshots_provided: bool,
    rollout_counts: dict[str, int],
    strict_v2: bool,
    unavailable: dict[str, str],
    assurance: str,
    at: datetime,
) -> tuple[str, ...]:
    required = _required_reviewer_count(task, profile, assurance)
    result: list[str] = []
    families = {selected.family}
    review_task = TaskSpec(
        task_id=task.task_id,
        repo=task.repo,
        task_type="review",
        sensitivity=task.sensitivity,
        prompt_file=task.prompt_file,
        input_paths=task.input_paths,
        output_contract="review",
        assurance=assurance,
        max_context_tokens=task.max_context_tokens,
        max_output_tokens=max(task.max_output_tokens, 4_096),
        timeout_seconds=task.timeout_seconds,
        write_mode="read_only",
        gate_profile=task.gate_profile,
        max_repairs=0,
    )
    for model_id in role.reviewers:
        reviewer = models[model_id]
        if reviewer.family in families:
            continue
        if (
            strict_v2
            and reviewer.rollout_stage in {"candidate", "shadow"}
            and not _candidate_read_ready(
                reviewer, rollout_counts.get(reviewer.id, 0), at
            )
        ):
            unavailable.setdefault(reviewer.id, "candidate_shadow")
            continue
        ok, reason = _eligible(
            reviewer, review_task, profile, protected=protected, at=at
        )
        if not ok:
            unavailable.setdefault(reviewer.id, reason)
            continue
        economics = _model_economics(reviewer)
        estimate = _estimated_cost(
            review_task, economics, observations.get(reviewer.id, {})
        )
        quota_ok, quota_reason, _ = _quota_allows(
            reviewer,
            economics,
            quotas.get(reviewer.provider),
            estimate,
            at=at,
            snapshots_provided=snapshots_provided,
        )
        if not quota_ok:
            unavailable.setdefault(reviewer.id, quota_reason)
            continue
        if economics.billing_mode in {"payg", "credits"} and (
            not profile.allow_paid_routes
            or estimate > profile.max_marginal_cost_usd
        ):
            unavailable.setdefault(reviewer.id, "incremental_spend_denied")
            continue
        result.append(reviewer.id)
        families.add(reviewer.family)
        if len(result) >= required:
            break
    return tuple(result)


def _required_reviewer_count(
    task: TaskSpec, profile: Profile, assurance: str
) -> int:
    if assurance == "critical":
        return 2
    if (
        assurance == "high"
        and profile.name == "chunsa"
        and task.task_type == "architecture"
    ):
        return 2
    return 1


def _candidate_read_ready(
    model: ModelConfig, simulation_count: int, at: datetime
) -> bool:
    if model.rollout_stage not in {"candidate", "shadow"}:
        return model.rollout_stage in {"read_only", "patch_writer", "promoted"}
    if simulation_count >= 20:
        return True
    if model.candidate_since is None:
        return False
    return at.date() >= date.fromisoformat(model.candidate_since) + timedelta(days=7)


def select_route(
    task: TaskSpec,
    models: dict[str, ModelConfig],
    roles: dict[str, RoleConfig],
    profile: Profile,
    *,
    observations: dict[str, dict[str, object]] | None = None,
    quotas: dict[str, QuotaSnapshot] | None = None,
    rollout_counts: dict[str, int] | None = None,
    pin: str | None = None,
    at: datetime | None = None,
    codex_uses_today: int = 0,
) -> RouteDecision:
    """Compatibility decision derived exclusively from the immutable v2 plan."""

    plan = plan_route(
        task,
        models,
        roles,
        profile,
        observations=observations,
        quotas=quotas,
        rollout_counts=rollout_counts,
        pin=pin,
        at=at,
        codex_uses_today=codex_uses_today,
    )
    context_hash = task.context_sha256()
    if plan.selected_model is None:
        return RouteDecision(
            schema_version=2,
            decision_id=str(uuid.uuid4()),
            task_id=task.task_id,
            profile=profile.name,
            status="NO_ROUTE",
            selected_model=None,
            requested_route_model=None,
            provider=None,
            family=None,
            transport=None,
            reviewer_model=None,
            alternatives=plan.alternatives,
            unavailable=plan.unavailable,
            auto_execute=False,
            approval_required=True,
            reason=plan.reason,
            context_sha256=context_hash,
            write_mode=task.write_mode,
            plan_id=plan.plan_id,
        )

    selected = models[plan.selected_model]
    transport = selected.transport
    if selected.id == "MiniMax-M3" and profile.minimax_mode == "delegate-minimax-api":
        transport = "delegate-minimax-api"
    return RouteDecision(
        schema_version=2,
        decision_id=str(uuid.uuid4()),
        task_id=task.task_id,
        profile=profile.name,
        status=plan.status,
        selected_model=selected.id,
        requested_route_model=selected.route_model,
        provider=selected.provider,
        family=selected.family,
        transport=transport,
        reviewer_model=plan.reviewer_models[0] if plan.reviewer_models else None,
        alternatives=plan.alternatives,
        unavailable=plan.unavailable,
        auto_execute=plan.status == "READY",
        approval_required=plan.approval_required,
        reason=plan.reason,
        context_sha256=context_hash,
        fallback_allowed=False,
        write_mode=task.write_mode,
        trust_tier=selected.trust_tier,
        execution_profile=selected.execution_profile,
        reasoning_effort=roles[task.task_type].reasoning_effort,
        reviewer_reasoning_effort=(
            roles[task.task_type].reviewer_reasoning_effort
            if plan.reviewer_models
            else None
        ),
        plan_id=plan.plan_id,
    )
