"""Common MCP delegation entrypoint over the orchestrator's existing policy."""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from ai_orchestrator.adapters import AdapterError, execute, write_output_artifact
from ai_orchestrator.config import (
    config_root,
    load_gates,
    load_models,
    load_profile,
    load_roles,
)
from ai_orchestrator.models import ContractError, RouteDecision, TaskSpec
from ai_orchestrator.output_contracts import validate_output
from ai_orchestrator.providers import refresh_deepseek_quota
from ai_orchestrator.routing import plan_route, select_route
from ai_orchestrator.telemetry import TelemetryStore

MAX_MCP_TASK_OUTPUT_BYTES = 1_048_576
_PROFILE_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _rescue_unaccepted_output(
    task: TaskSpec,
    output_path: Path | None,
    output: bytes,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    """Persistir lo que el modelo ya genero cuando la entrega no se acepta.

    write_output_artifact solo corre en el camino feliz. Las salidas terminales
    de run() ocurren cuando el modelo YA produjo texto: cuota agotada, proveedor
    sin SUCCESS, contrato invalido y las tres de revision. Las siete corridas del
    2026-08-26 murieron ahi -- dos con el contrato ya validado -- y no dejaron ni
    un byte que leer.

    El nombre lleva el estado pegado para que un rescate no pueda confundirse
    jamas con un artefacto aceptado: eso es lo que hacia preferible perderlo.
    """

    if output_path is None or not output:
        return receipt
    status = str(receipt.get("status", "UNKNOWN"))
    rescue_path = output_path.with_name(f"{output_path.name}.RECHAZADO-{status}")
    try:
        write_output_artifact(task, rescue_path, output)
    except (AdapterError, OSError) as exc:
        return {**receipt, "rescued_output_path": None, "rescue_error": str(exc)}
    return {**receipt, "rescued_output_path": str(rescue_path)}


class OrchestratorTaskRunner:
    """Execute an explicit task spec through every configured core adapter.

    MCP is never an apply surface. Writer models operate in isolated worktrees
    and can return only a reviewed `PATCH_READY` artifact.
    """

    def __init__(
        self,
        *,
        configuration_root: Path | None = None,
        executor: Callable[..., Any] = execute,
        telemetry: Any | None = None,
        allow_validated_apply: bool = False,
    ) -> None:
        self.configuration_root = (configuration_root or config_root()).resolve()
        self.executor = executor
        self.telemetry = telemetry
        self.allow_validated_apply = allow_validated_apply

    def _read_task(self, spec_path: str) -> TaskSpec:
        if not isinstance(spec_path, str) or not spec_path.strip():
            raise ContractError("spec_path must be a non-empty string")
        unresolved = Path(spec_path).expanduser()
        if unresolved.is_symlink():
            raise ContractError("task spec must not be a symlink")
        try:
            source = unresolved.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ContractError("task spec does not exist") from exc
        if not source.is_file() or source.stat().st_size > 1_000_000:
            raise ContractError("task spec must be a regular file of at most 1 MB")
        task = TaskSpec.read(source)
        if task.write_mode == "validated_apply" and not self.allow_validated_apply:
            raise ContractError(
                "MCP delegation requires read_only or validated_patch; "
                "validated_apply is forbidden"
            )
        if task.prompt_path() is None:
            raise ContractError("MCP delegation requires prompt_file in the task spec")
        return task

    def _configuration(self, profile: str):
        if not isinstance(profile, str) or not _PROFILE_NAME.fullmatch(profile):
            raise ContractError("invalid profile name")
        return (
            load_models(self.configuration_root),
            load_roles(self.configuration_root),
            load_profile(profile, self.configuration_root),
        )

    def simulate(
        self,
        *,
        spec_path: str,
        profile: str = "default",
        model: str | None = None,
        observations: dict[str, dict[str, object]] | None = None,
        quotas: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise ContractError("model pin must be a non-empty string")
        task = self._read_task(spec_path)
        models, roles, selected_profile = self._configuration(profile)
        telemetry = self.telemetry or TelemetryStore()
        effective_quotas = quotas
        if effective_quotas is None and hasattr(telemetry, "latest_quotas"):
            effective_quotas = telemetry.latest_quotas()
            effective_quotas["deepseek"] = refresh_deepseek_quota(
                telemetry, configuration_root=self.configuration_root
            )
        rollout_counts = (
            telemetry.rollout_counts()
            if hasattr(telemetry, "rollout_counts")
            else None
        )
        codex_uses_today = (
            telemetry.daily_provider_runs("openai")
            if hasattr(telemetry, "daily_provider_runs")
            else 0
        )
        # Public callers pass QuotaSnapshot objects through the Python API;
        # MCP inventory/simulation uses the same dataclass-backed payload.
        plan = plan_route(
            task,
            models,
            roles,
            selected_profile,
            observations=observations,
            quotas=effective_quotas,  # type: ignore[arg-type]
            rollout_counts=rollout_counts,
            pin=model,
            codex_uses_today=codex_uses_today,
        )
        if hasattr(telemetry, "record_route"):
            decision = select_route(
                task,
                models,
                roles,
                selected_profile,
                observations=observations,
                quotas=effective_quotas,  # type: ignore[arg-type]
                rollout_counts=rollout_counts,
                pin=model,
                codex_uses_today=codex_uses_today,
            )
            telemetry.record_route(task, decision, simulated=True)
        return plan.as_dict()

    def authorize_inline(
        self,
        *,
        task_type: str,
        sensitivity: str,
        model: str,
        profile: str = "default",
        write_mode: str = "read_only",
    ) -> dict[str, Any]:
        """Authorize compatibility delegates without persisting their prompt."""

        if write_mode != "read_only":
            raise ContractError("inline compatibility delegation is read-only")
        normalized = task_type.strip().lower()
        if normalized in {"reasoning"}:
            routing_type = "review"
        elif normalized in {"architecture", "review", "research", "security"}:
            routing_type = normalized
        else:
            routing_type = "implementation"
        task = TaskSpec(
            task_id=str(uuid.uuid4()),
            repo=str(self.configuration_root.parent.resolve()),
            task_type=routing_type,
            sensitivity=sensitivity,
            output_contract="text",
            max_context_tokens=120_000,
            max_output_tokens=4_096,
            write_mode="read_only",
        )
        models, roles, selected_profile = self._configuration(profile)
        # Compatibility delegates created without a telemetry store preserve
        # the v1 explicit-pin behaviour. Production surfaces inject the shared
        # store and therefore enforce the measured-credit snapshot gate.
        telemetry = self.telemetry
        quotas = (
            telemetry.latest_quotas()
            if telemetry is not None and hasattr(telemetry, "latest_quotas")
            else None
        )
        rollout_counts = (
            telemetry.rollout_counts()
            if telemetry is not None and hasattr(telemetry, "rollout_counts")
            else None
        )
        codex_uses_today = (
            telemetry.daily_provider_runs("openai")
            if hasattr(telemetry, "daily_provider_runs")
            else 0
        )
        plan = plan_route(
            task,
            models,
            roles,
            selected_profile,
            quotas=quotas,
            rollout_counts=rollout_counts,
            pin=model,
            codex_uses_today=codex_uses_today,
        )
        if plan.selected_model != model or plan.status != "READY":
            detail = plan.unavailable.get(model, plan.reason)
            raise ContractError(
                f"pinned compatibility route requires handoff: {detail}"
            )
        return plan.as_dict()

    def run(
        self,
        *,
        spec_path: str,
        profile: str = "default",
        model: str | None = None,
        confirmed_plan_id: str | None = None,
        observations: dict[str, dict[str, object]] | None = None,
        quotas: dict[str, object] | None = None,
        output_path: Path | None = None,
    ) -> dict[str, Any]:
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise ContractError("model pin must be a non-empty string")
        task = self._read_task(spec_path)
        models, roles, selected_profile = self._configuration(profile)
        telemetry = self.telemetry or TelemetryStore()
        effective_quotas = quotas
        if effective_quotas is None and hasattr(telemetry, "latest_quotas"):
            effective_quotas = telemetry.latest_quotas()
            effective_quotas["deepseek"] = refresh_deepseek_quota(
                telemetry, configuration_root=self.configuration_root
            )
        rollout_counts = (
            telemetry.rollout_counts()
            if hasattr(telemetry, "rollout_counts")
            else None
        )
        codex_uses_today = (
            telemetry.daily_provider_runs("openai")
            if hasattr(telemetry, "daily_provider_runs")
            else 0
        )
        plan = plan_route(
            task,
            models,
            roles,
            selected_profile,
            observations=observations,
            quotas=effective_quotas,  # type: ignore[arg-type]
            rollout_counts=rollout_counts,
            pin=model,
            codex_uses_today=codex_uses_today,
        )
        if confirmed_plan_id is not None and confirmed_plan_id != plan.plan_id:
            return {
                "schema_version": 2,
                "status": "REPLAN_REQUIRED",
                "plan": plan.as_dict(),
                "final_accepted": False,
                "fallback_allowed": False,
                "reason": "confirmed plan no longer matches current policy inputs",
            }
        decision = select_route(
            task,
            models,
            roles,
            selected_profile,
            observations=observations,
            quotas=effective_quotas,  # type: ignore[arg-type]
            rollout_counts=rollout_counts,
            pin=model,
            codex_uses_today=codex_uses_today,
        )
        if (
            plan.status == "CONFIRMATION_REQUIRED"
            and confirmed_plan_id == plan.plan_id
        ):
            decision = replace(
                decision,
                status="READY",
                auto_execute=True,
                approval_required=False,
                reason=decision.reason + "; exact plan confirmed",
            )
        telemetry.record_route(task, decision)
        if not decision.auto_execute or decision.status != "READY":
            return {
                "schema_version": 2,
                "status": "HANDOFF_REQUIRED",
                "plan": plan.as_dict(),
                "decision": decision.as_dict(),
                "final_accepted": False,
                "fallback_allowed": False,
            }

        if task.gate_profile == "profile":
            gate_commands = selected_profile.deterministic_gates
            gate_timeout_seconds = task.timeout_seconds
        else:
            gate_profiles = load_gates(self.configuration_root)
            if task.gate_profile not in gate_profiles:
                raise ContractError(f"unknown gate_profile: {task.gate_profile}")
            gate = gate_profiles[task.gate_profile]
            gate_commands = gate.commands
            gate_timeout_seconds = gate.timeout_seconds

        selected = models[str(decision.selected_model)]
        supports_v2_runtime = hasattr(telemetry, "acquire_lease") and hasattr(
            telemetry, "release_lease"
        )
        lease = (
            telemetry.acquire_lease(
                selected.id,
                max_concurrency=selected.max_concurrency,
                ttl_seconds=task.timeout_seconds + 60,
            )
            if supports_v2_runtime
            else "legacy-runtime"
        )
        if lease is None:
            return {
                "schema_version": 2,
                "status": "REPLAN_REQUIRED",
                "plan": plan.as_dict(),
                "decision": decision.as_dict(),
                "final_accepted": False,
                "fallback_allowed": False,
                "reason": "model concurrency is exhausted",
            }

        attempts: list[dict[str, Any]] = []

        def terminal(*args: Any, **kwargs: Any) -> dict[str, Any]:
            """El recibo terminal, pero salvando a disco lo que ya se pago."""

            return _rescue_unaccepted_output(
                task, output_path, output, self._terminal_receipt(*args, **kwargs)
            )

        try:
            result, output = self._execute_primary(
                task,
                decision,
                selected,
                telemetry,
                attempts,
                gate_commands=gate_commands,
                gate_timeout_seconds=gate_timeout_seconds,
                protected_prefixes=selected_profile.protected_path_prefixes,
            )
            if result.status == "QUOTA_UNAVAILABLE":
                return terminal(
                    "REPLAN_REQUIRED", plan, decision, result, attempts
                )
            if result.status not in {"SUCCESS", "PATCH_READY", "APPLIED"}:
                return terminal(
                    "HANDOFF_REQUIRED", plan, decision, result, attempts
                )
            validation = validate_output(task, output)
            if not validation.valid:
                return terminal(
                    "CONTRACT_INVALID",
                    plan,
                    decision,
                    result,
                    attempts,
                    contract_validated=False,
                    reason=validation.error_code,
                )

            review_payload: dict[str, Any] | None = None
            reviews: list[dict[str, Any]] = []
            review_repairs = 0
            while plan.reviewer_models and supports_v2_runtime:
                reviews = []
                for reviewer_id in plan.reviewer_models:
                    review_payload = self._review(
                        task,
                        decision,
                        reviewer_id,
                        output,
                        models,
                        telemetry,
                        attempts,
                        prior_reviews=reviews,
                    )
                    if review_payload is None:
                        break
                    reviews.append(review_payload)
                critical = any(
                    finding.get("severity") == "critical"
                    for review in reviews
                    for finding in review.get("findings", ())
                    if isinstance(finding, dict)
                )
                authority_final = (
                    plan.assurance in {"high", "critical"}
                    and selected_profile.name == "chunsa"
                    and task.task_type == "architecture"
                    and plan.reviewer_models[-1] == decision.selected_model
                )
                all_accept = len(reviews) == len(plan.reviewer_models) and all(
                    review.get("verdict") == "accept" for review in reviews
                )
                final_accept = bool(
                    reviews and reviews[-1].get("verdict") == "accept"
                )
                if not critical and (all_accept or (authority_final and final_accept)):
                    break
                if (
                    review_payload is None
                    or critical
                    or not any(review.get("verdict") == "revise" for review in reviews)
                    or review_repairs >= task.max_repairs
                ):
                    return terminal(
                        "HANDOFF_REQUIRED",
                        plan,
                        decision,
                        result,
                        attempts,
                        review=review_payload,
                        review_repairs=review_repairs,
                        reason="review did not accept the delivery",
                    )
                review_repairs += 1
                repair_prompt = _review_repair_prompt(output, reviews)
                result, output = self._execute_primary(
                    task,
                    decision,
                    selected,
                    telemetry,
                    attempts,
                    gate_commands=gate_commands,
                    gate_timeout_seconds=gate_timeout_seconds,
                    protected_prefixes=selected_profile.protected_path_prefixes,
                    prompt_override=repair_prompt,
                )
                if result.status not in {"SUCCESS", "PATCH_READY", "APPLIED"}:
                    return terminal(
                        "HANDOFF_REQUIRED",
                        plan,
                        decision,
                        result,
                        attempts,
                        review=review_payload,
                        review_repairs=review_repairs,
                    )
                validation = validate_output(task, output)
                if not validation.valid:
                    return terminal(
                        "CONTRACT_INVALID",
                        plan,
                        decision,
                        result,
                        attempts,
                        review=review_payload,
                        review_repairs=review_repairs,
                        contract_validated=False,
                        reason=validation.error_code,
                    )

            if plan.reviewer_models and supports_v2_runtime and review_payload is None:
                return terminal(
                    "HANDOFF_REQUIRED",
                    plan,
                    decision,
                    result,
                    attempts,
                    reason="independent reviewer unavailable",
                )
            if len(output) > MAX_MCP_TASK_OUTPUT_BYTES:
                raise ContractError("delegated output exceeds the 1 MB MCP limit")
            if output_path is not None:
                write_output_artifact(task, output_path, output)
            result = replace(
                result,
                attempts=tuple(attempts),
                contract_validated=True,
                reviewer_verdict=review_payload,
                final_accepted=True,
            )
            if hasattr(telemetry, "finalize_run"):
                telemetry.finalize_run(result)
            return {
                "schema_version": 2,
                "status": result.status,
                "plan": plan.as_dict(),
                "decision": decision.as_dict(),
                "run": result.as_dict(),
                "attempts": attempts,
                "contract_validated": True,
                "review": review_payload,
                "reviews": reviews,
                "review_repairs": review_repairs,
                "final_accepted": True,
                "output": output.decode("utf-8"),
                "source_repo_modified": result.status == "APPLIED",
                "review_required": result.status == "PATCH_READY",
                "fallback_allowed": False,
            }
        finally:
            if supports_v2_runtime:
                telemetry.release_lease(lease)

    def _execute_primary(
        self,
        task: TaskSpec,
        decision: RouteDecision,
        model: Any,
        telemetry: Any,
        attempts: list[dict[str, Any]],
        *,
        retry_transient: bool = True,
        **kwargs: object,
    ) -> tuple[Any, bytes]:
        transient = {"OVERLOADED", "TRANSIENT_PROVIDER_ERROR", "NETWORK_ERROR"}
        for attempt in range(2 if retry_transient else 1):
            result, output = self.executor(task, decision, **kwargs)
            result = _with_usage_cost(result, model)
            telemetry.record_run(result)
            if result.usage is not None:
                telemetry.record_usage(result.usage)
            attempts.append({"kind": "primary", **result.as_dict()})
            if result.status not in transient or attempt == 1:
                return result, output
        raise AssertionError("unreachable")

    def _review(
        self,
        task: TaskSpec,
        primary: RouteDecision,
        reviewer_id: str,
        output: bytes,
        models: dict[str, Any],
        telemetry: Any,
        attempts: list[dict[str, Any]],
        *,
        prior_reviews: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        reviewer = models[reviewer_id]
        review_task = replace(
            task,
            task_type="review",
            output_contract="review",
            output_schema_file=None,
            max_output_tokens=max(task.max_output_tokens, 4_096),
            write_mode="read_only",
            allowed_write_paths=(),
            max_repairs=0,
        )
        review_decision = replace(
            primary,
            decision_id=str(uuid.uuid4()),
            status="READY",
            selected_model=reviewer.id,
            requested_route_model=reviewer.route_model,
            provider=reviewer.provider,
            family=reviewer.family,
            transport=reviewer.transport,
            reviewer_model=None,
            alternatives=(),
            auto_execute=True,
            approval_required=False,
            reason="independent cross-family review",
            write_mode="read_only",
            trust_tier=reviewer.trust_tier,
            execution_profile="read_only",
            reasoning_effort=primary.reviewer_reasoning_effort,
            reviewer_reasoning_effort=None,
        )
        telemetry.record_route(review_task, review_decision)
        reuse_primary_lease = reviewer.id == primary.selected_model
        lease = (
            "primary-lease"
            if reuse_primary_lease
            else telemetry.acquire_lease(
                reviewer.id,
                max_concurrency=reviewer.max_concurrency,
                ttl_seconds=review_task.timeout_seconds + 60,
            )
        )
        if lease is None:
            return None
        try:
            prompt = _review_prompt(output, prior_reviews=prior_reviews)
            transient = {
                "OVERLOADED",
                "TRANSIENT_PROVIDER_ERROR",
                "NETWORK_ERROR",
            }
            for review_attempt in range(2):
                result, review_output = self.executor(
                    review_task, review_decision, prompt_override=prompt
                )
                result = _with_usage_cost(result, reviewer)
                telemetry.record_run(result)
                if result.usage is not None:
                    telemetry.record_usage(result.usage)
                attempts.append({"kind": "review", **result.as_dict()})
                if result.status not in transient or review_attempt == 1:
                    break
        finally:
            if not reuse_primary_lease:
                telemetry.release_lease(lease)
        if result.status != "SUCCESS":
            return None
        validation = validate_output(review_task, review_output)
        return validation.parsed if validation.valid else None

    @staticmethod
    def _terminal_receipt(
        status: str,
        plan: Any,
        decision: RouteDecision,
        result: Any,
        attempts: list[dict[str, Any]],
        *,
        review: dict[str, Any] | None = None,
        review_repairs: int = 0,
        contract_validated: bool = True,
        reason: str | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "status": status,
            "plan": plan.as_dict(),
            "decision": decision.as_dict(),
            "run": result.as_dict(),
            "attempts": attempts,
            "contract_validated": contract_validated,
            "review": review,
            "review_repairs": review_repairs,
            "final_accepted": False,
            "fallback_allowed": False,
            "reason": reason or result.error_code or result.status,
        }


def _review_prompt(
    output: bytes, *, prior_reviews: list[dict[str, Any]] | None = None
) -> bytes:
    context = (
        b"\nPRIOR INDEPENDENT REVIEWS:\n" + json_bytes(prior_reviews)
        if prior_reviews
        else b""
    )
    return (
        b"Independently review the deliverable below. Return only UTF-8 JSON with "
        b"verdict accept|revise|escalate and findings containing severity, evidence, "
        b"and message. Never override a failed deterministic gate."
        + context
        + b"\n\nDELIVERABLE:\n"
        + output
    )


def _review_repair_prompt(
    output: bytes, reviews: list[dict[str, Any]]
) -> bytes:
    return (
        b"Repair the bounded deliverable using only the review findings. Do not expand "
        b"scope. The orchestrator will repeat deterministic gates and independent review.\n"
        + json_bytes(reviews)
        + b"\nDELIVERABLE:\n"
        + output
    )


def json_bytes(value: object) -> bytes:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _with_usage_cost(result: Any, model: Any) -> Any:
    """Attach actual marginal/equivalent cost without inventing unknown data."""

    usage = result.usage
    economics = model.economics
    if usage is None or economics is None:
        return result
    if not model.cache_supported:
        usage = replace(
            usage,
            cached_tokens=(0 if usage.cached_tokens is None else usage.cached_tokens),
            cache_write_tokens=(
                0
                if usage.cache_write_tokens is None
                else usage.cache_write_tokens
            ),
        )
    prices_known = any(
        value > 0
        for value in (
            economics.input_per_mtok_usd,
            economics.cache_read_per_mtok_usd,
            economics.cache_write_per_mtok_usd,
            economics.output_per_mtok_usd,
        )
    )
    equivalent: float | None = None
    if prices_known:
        try:
            equivalent = economics.cost_for(usage)
        except ContractError:
            equivalent = None
    if economics.billing_mode in {"payg", "prepaid_balance", "prepaid_balance_only"}:
        marginal = equivalent
    elif economics.billing_mode == "included_quota":
        marginal = 0.0
    else:
        # In particular, Abacus credit debits have no published USD mapping.
        marginal = None
    costed = replace(
        usage,
        marginal_cost_usd=marginal,
        equivalent_cost_usd=equivalent,
        estimated_cost_usd=equivalent,
        cache_hit=(
            usage.cached_tokens > 0
            if usage.cached_tokens is not None
            else None
        ),
    )
    return replace(result, usage=costed)
