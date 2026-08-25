from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath

from ai_orchestrator.models import (
    ModelConfig,
    Profile,
    RoleConfig,
    RouteDecision,
    RoutePlan,
    TaskSpec,
    sensitivity_rank,
)

NATIVE_TRANSPORTS = {"codex_cli", "claude_cli"}
HANDOFF_SENSITIVITIES = {"confidential", "secrets", "financial_control", "deployment"}


# A handful of runs is not a measurement. Below this many samples the observed
# latency and success rate are ignored and the neutral prior is used instead:
# one lucky or unlucky run must not decide which provider a role goes to.
MIN_OBSERVATION_SAMPLES = 3

# Una ruta gratuita elegible se prefiere a una que gasta cuota, y cualquiera
# de las dos a una de pago.
COST_TIER_SCORES = {"free": 1.0, "included": 0.5, "paid": 0.0}


def _bounded_score(value: object, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, result))


def _has_enough_samples(observed: dict[str, object]) -> bool:
    try:
        return int(observed.get("sample_count", 0)) >= MIN_OBSERVATION_SAMPLES
    except (TypeError, ValueError):
        return False


def _web_free_allowed(task: TaskSpec, profile: Profile) -> bool:
    """Decide whether a free web route may see this task.

    Hasta el 2026-08-25 la unica salida era tarea publica en repositorio
    publico. El Director abrio el trabajo interno para poder aprovechar las
    suscripciones gratuitas, a cambio de que codigo privado de sensibilidad
    interna salga por rutas no oficiales. Lo que nunca sale es lo que
    HANDOFF_SENSITIVITIES protege.
    """

    if profile.web_free_max_sensitivity == "internal":
        # Modo abierto: sale trabajo publico e interno, con independencia de si
        # el repositorio es publico. Lo demas nunca sale.
        return task.sensitivity in {"public", "internal"}
    # Modo estricto, el comportamiento historico: solo tarea publica y ademas
    # repositorio publico.
    return task.sensitivity == "public" and profile.public_repository


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
    trusted = _has_enough_samples(observed)
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
    # Declarar la capacidad no es haberla medido: sin muestras suficientes el
    # prior vale la mitad que un cache_hit_rate real, igual que la latencia.
    cache_prior = 0.5 if model.cache_supported else 0.0
    cache = (
        _bounded_score(observed.get("cache_hit_rate", cache_prior), cache_prior)
        if trusted
        else cache_prior
    )
    latency_ms = (
        observed.get("latency_ms", model.observed_latency_ms)
        if trusted
        else model.observed_latency_ms
    )
    latency = (
        0.5
        if latency_ms is None
        else 1.0 / (1.0 + max(0.0, float(latency_ms)) / 10_000.0)
    )
    # Measured reliability. Without it the scorer rewards a model for failing
    # quickly: a fast error looks exactly like a fast answer to the latency
    # term. Kept at the neutral prior until there are enough samples, so a
    # model that has never run is not punished for the absence of evidence.
    success = _bounded_score(observed.get("success_rate", 1.0), 1.0) if trusted else 1.0
    # Role order is an explicit operator preference and a deterministic tie
    # breaker, but it cannot outweigh measured quality/quota/cost.
    # Preferir lo gratuito cuando ya ha pasado todas las puertas de politica:
    # una ruta gratuita elegible conserva cuota de suscripcion para el trabajo
    # que no puede ir por ahi.
    cost_tier = COST_TIER_SCORES.get(model.cost_tier, 0.5)
    priority = (role_size - role_priority) / max(1, role_size)
    return round(
        quality * 0.34
        + quota * 0.18
        + cost * 0.12
        + context * 0.10
        + success * 0.08
        + cost_tier * 0.06
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
    pin: str | None = None,
) -> RoutePlan:
    """Simulate a route without executing it or silently falling back.

    Trust, sensitivity, write capability and budget are hard gates.  Only the
    survivors are scored.  ``alternatives`` are informational and require a
    new explicit route decision if the selected provider fails.
    """

    role = roles.get(task.task_type)
    if role is None:
        return RoutePlan(
            task_id=task.task_id,
            profile=profile.name,
            selected_model=None,
            alternatives=(),
            scores={},
            quality_price_score=None,
            budget={"max_usd": profile.max_marginal_cost_usd, "estimated_usd": None},
            permissions=(),
            approval_required=True,
            reason=f"unknown task_type: {task.task_type}",
        )
    if profile.repo_root and task.repo != str(Path(profile.repo_root).resolve()):
        return RoutePlan(
            task_id=task.task_id,
            profile=profile.name,
            selected_model=None,
            alternatives=(),
            scores={},
            quality_price_score=None,
            budget={"max_usd": profile.max_marginal_cost_usd, "estimated_usd": None},
            permissions=(),
            approval_required=True,
            reason="profile repo_root does not match task repository",
        )
    scored, _ = _rank_candidates(
        task,
        models,
        role,
        profile,
        observations=observations,
        pin=pin,
    )
    if not scored:
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
            reason="no model satisfies security, quota and budget policy",
        )
    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    selected_score, _, selected, estimated = scored[0]
    decision = select_route(
        task,
        models,
        roles,
        profile,
        observations=observations,
        pin=selected.id,
    )
    permissions = ["provider_network", "repository_read"]
    if task.write_mode != "read_only":
        permissions.append("reviewed_patch_write")
    if task.sensitivity != "public":
        permissions.append("non_public_context")
    score_map = {row[2].id: row[0] for row in scored}
    return RoutePlan(
        task_id=task.task_id,
        profile=profile.name,
        selected_model=selected.id,
        provider=selected.provider,
        alternatives=tuple(row[2].id for row in scored[1:]),
        scores=score_map,
        quality_price_score=selected_score,
        budget={
            "max_usd": profile.max_marginal_cost_usd,
            "estimated_usd": round(estimated, 8),
            "incremental_spend_allowed": profile.allow_paid_routes,
        },
        permissions=tuple(permissions),
        approval_required=decision.approval_required or estimated > 0,
        reason=decision.reason,
        fallback_allowed=False,
    )


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
) -> tuple[bool, str]:
    if model.id not in profile.allowed_models:
        return False, "profile_denied"
    if model.available_until is not None and datetime.now(
        UTC
    ).date() > date.fromisoformat(model.available_until):
        return False, "promotion_expired"
    if model.trust_tier == "web_free" and not _web_free_allowed(task, profile):
        return False, "web_free_sensitivity_not_allowed"
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


def _rank_candidates(
    task: TaskSpec,
    models: dict[str, ModelConfig],
    role: RoleConfig,
    profile: Profile,
    *,
    observations: dict[str, dict[str, object]] | None,
    pin: str | None,
) -> tuple[list[tuple[float, int, ModelConfig, float]], dict[str, str]]:
    """Apply hard gates, then return candidates in deterministic score order."""

    protected = _is_protected(task, profile)
    observed = observations or {}
    ranked: list[tuple[float, int, ModelConfig, float]] = []
    unavailable: dict[str, str] = {}
    for index, model_id in enumerate(role.models):
        model = models[model_id]
        if pin is not None and model_id != pin:
            unavailable[model.id] = "not_pinned"
            continue
        if (
            task.task_type == "architecture"
            and profile.architect_model is not None
            and model_id != profile.architect_model
        ):
            unavailable[model.id] = "profile_architect_pinned"
            continue
        eligible, reason = _eligible(model, task, profile, protected=protected)
        if not eligible:
            unavailable[model.id] = reason
            continue
        model_observation = observed.get(model.id, {})
        quota_available = _bounded_score(
            model_observation.get(
                "quota_available", 1.0 if model.included_quota else 0.0
            ),
            1.0 if model.included_quota else 0.0,
        )
        if quota_available <= 0:
            unavailable[model.id] = "quota_unavailable"
            continue
        marginal_cost = max(
            0.0,
            float(
                model_observation.get(
                    "marginal_cost_per_million", model.marginal_cost_per_million
                )
            ),
        )
        estimated = marginal_cost * task.max_context_tokens / 1_000_000.0
        paid = not model.included_quota and estimated > 0
        if paid and not profile.allow_paid_routes:
            unavailable[model.id] = "paid_route_denied"
            continue
        if paid and estimated > profile.max_marginal_cost_usd:
            unavailable[model.id] = "budget_exceeded"
            continue
        score = score_route_candidate(
            model,
            task,
            role_priority=index,
            role_size=len(role.models),
            observation=model_observation,
        )
        ranked.append((score, -index, model, estimated))
    if pin is not None and pin not in role.models:
        unavailable[pin] = "pin_outside_role"
    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return ranked, unavailable


def select_route(
    task: TaskSpec,
    models: dict[str, ModelConfig],
    roles: dict[str, RoleConfig],
    profile: Profile,
    *,
    observations: dict[str, dict[str, object]] | None = None,
    pin: str | None = None,
) -> RouteDecision:
    """``pin`` ESTRECHA los candidatos a un solo modelo; nunca concede nada.

    El modelo anclado sigue pasando por las mismas comprobaciones de
    elegibilidad —perfil, sensibilidad, capacidad de escritura, estado— y si
    falla alguna, la ruta sale NO_ROUTE con el motivo real. Sirve para medir
    dos rutas sobre la misma tarea sin reordenar roles.toml, que seria promover
    antes de medir.
    """

    role = roles.get(task.task_type)
    context_hash = task.context_sha256()
    if profile.repo_root and task.repo != str(Path(profile.repo_root).resolve()):
        return RouteDecision(
            schema_version=1,
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
            alternatives=(),
            unavailable={},
            auto_execute=False,
            approval_required=True,
            reason="profile repo_root does not match task repository",
            context_sha256=context_hash,
            write_mode=task.write_mode,
        )
    if role is None:
        return RouteDecision(
            schema_version=1,
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
            alternatives=(),
            unavailable={},
            auto_execute=False,
            approval_required=True,
            reason=f"unknown task_type: {task.task_type}",
            context_sha256=context_hash,
            write_mode=task.write_mode,
        )

    ranked, unavailable = _rank_candidates(
        task,
        models,
        role,
        profile,
        observations=observations,
        pin=pin,
    )

    if not ranked:
        return RouteDecision(
            schema_version=1,
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
            alternatives=(),
            unavailable=unavailable,
            auto_execute=False,
            approval_required=True,
            reason="no available model satisfies capability, trust and sensitivity policy",
            context_sha256=context_hash,
            write_mode=task.write_mode,
        )

    selected_score, _, selected, _estimated = ranked[0]
    eligible = [row[2] for row in ranked]
    protected = _is_protected(task, profile)
    transport = selected.transport
    if selected.id == "MiniMax-M3" and profile.minimax_mode == "delegate-minimax-api":
        transport = "delegate-minimax-api"

    sensitive_handoff = task.sensitivity in HANDOFF_SENSITIVITIES
    bridge_handoff = transport == "delegate-minimax-api"
    auto_execute = (
        task.sensitivity in profile.auto_execute_sensitivities
        and not protected
        and not bridge_handoff
    )
    status = "READY" if auto_execute else "HANDOFF_REQUIRED"
    reason_parts = [f"selected highest scored model for role {role.name}"]
    reason_parts.append(f"score={selected_score:.6f}")
    if pin is not None:
        # Queda escrito en la decision y en la telemetria: una ruta anclada no
        # es la que el enrutador habria elegido, y eso hay que poder auditarlo.
        reason_parts.append(f"pin={pin}")
    if protected:
        reason_parts.append("protected input")
    if sensitive_handoff:
        reason_parts.append("sensitive task")
    if bridge_handoff:
        reason_parts.append("MiniMax supervised bridge required")
    reviewer_model = _reviewer_for(role, selected, models, profile)

    return RouteDecision(
        schema_version=1,
        decision_id=str(uuid.uuid4()),
        task_id=task.task_id,
        profile=profile.name,
        status=status,
        selected_model=selected.id,
        requested_route_model=selected.route_model,
        provider=selected.provider,
        family=selected.family,
        transport=transport,
        reviewer_model=reviewer_model,
        alternatives=tuple(model.id for model in eligible[1:]),
        unavailable=unavailable,
        auto_execute=auto_execute,
        approval_required=not auto_execute,
        reason="; ".join(reason_parts),
        context_sha256=context_hash,
        fallback_allowed=False,
        write_mode=task.write_mode,
        trust_tier=selected.trust_tier,
        execution_profile=selected.execution_profile,
        reasoning_effort=role.reasoning_effort,
        reviewer_reasoning_effort=(
            role.reviewer_reasoning_effort if reviewer_model else None
        ),
        max_concurrency=selected.max_concurrency,
    )
