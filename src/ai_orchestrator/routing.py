from __future__ import annotations

import uuid
from pathlib import Path, PurePosixPath

from ai_orchestrator.models import (
    ModelConfig,
    Profile,
    RoleConfig,
    RouteDecision,
    TaskSpec,
    sensitivity_rank,
)


NATIVE_TRANSPORTS = {"codex_cli", "claude_cli"}
HANDOFF_SENSITIVITIES = {"confidential", "secrets", "financial_control", "deployment"}


def _is_protected(task: TaskSpec, profile: Profile) -> bool:
    selected = task.input_paths + task.allowed_write_paths + (
        (task.prompt_file,) if task.prompt_file else ()
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


def select_route(
    task: TaskSpec,
    models: dict[str, ModelConfig],
    roles: dict[str, RoleConfig],
    profile: Profile,
) -> RouteDecision:
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

    protected = _is_protected(task, profile)
    eligible: list[ModelConfig] = []
    unavailable: dict[str, str] = {}
    for model_id in role.models:
        model = models[model_id]
        ok, reason = _eligible(model, task, profile, protected=protected)
        if ok:
            eligible.append(model)
        else:
            unavailable[model.id] = reason

    if not eligible:
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

    selected = eligible[0]
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
    reason_parts = [f"selected first eligible model for role {role.name}"]
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
    )
