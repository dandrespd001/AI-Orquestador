"""Bounded Abacus canaries that consume the same rolling attempt budget."""

from __future__ import annotations

from typing import Any

from ai_orchestrator.abacus import AbacusDelegator
from ai_orchestrator.admission import AdmissionController
from ai_orchestrator.models import ContractError
from ai_orchestrator.workflows import WorkflowStore

CANARY_TOKEN = "AI_ORCHESTRATOR_CANARY_OK"
_MODELS = {
    "abacus-qwen3.8-max": ("Qwen3.8 Max", "agentic_implementation"),
    "abacus-grok-4.6": ("Grok 4.6", "review"),
}


def run_abacus_workflow_canaries(
    model: str,
    *,
    attempts: int = 5,
    delegator: AbacusDelegator | Any | None = None,
    store: WorkflowStore | None = None,
    admission: AdmissionController | Any | None = None,
) -> dict[str, object]:
    if model not in _MODELS:
        raise ContractError("unsupported Abacus workflow canary model")
    if attempts != 5:
        raise ContractError("workflow promotion requires exactly five canaries")
    selected_model, task_type = _MODELS[model]
    selected_delegator = delegator or AbacusDelegator()
    selected_store = store or WorkflowStore()
    selected_admission = admission or AdmissionController(
        selected_store.path.with_name("admission.sqlite")
    )
    results: list[dict[str, object]] = []
    for attempt in range(1, attempts + 1):
        selected_store.reserve_attempt(model)
        passed = False
        error_code: str | None = None
        try:
            with selected_admission.acquire(
                model=model,
                max_concurrency=1,
                repo="abacus-workflow-canary",
                allow_parallel=True,
                lease_seconds=180,
                wait_timeout_seconds=30,
            ):
                receipt = selected_delegator.delegate(
                    prompt=(
                        "Return exactly this token and nothing else: " + CANARY_TOKEN
                    ),
                    task_type=task_type,
                    model=selected_model,
                    sensitivity="internal",
                    timeout_seconds=120,
                )
            passed = str(receipt.get("result", "")).strip() == CANARY_TOKEN
            if not passed:
                error_code = "CANARY_OUTPUT_MISMATCH"
        except (ContractError, RuntimeError, OSError) as exc:
            error_code = type(exc).__name__
        selected_store.record_canary(model, passed=passed)
        results.append({"attempt": attempt, "passed": passed, "error_code": error_code})
    promotion_ready = all(bool(row["passed"]) for row in results)
    return {
        "schema_version": 1,
        "model": model,
        "attempts": results,
        "promotion_ready": promotion_ready,
        "status": "available" if promotion_ready else "candidate",
        "automatic_promotion": False,
        "automatic_purchase": False,
        "fallback_allowed": False,
        "prompt_retained": False,
    }
