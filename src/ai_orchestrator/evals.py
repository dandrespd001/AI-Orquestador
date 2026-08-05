from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_orchestrator.config import load_models, load_profile, load_roles
from ai_orchestrator.models import ContractError, TaskSpec
from ai_orchestrator.routing import select_route


def run_routing_eval(suite: Path, profile_name: str) -> dict[str, Any]:
    models = load_models()
    roles = load_roles()
    profile = load_profile(profile_name)
    seen: set[str] = set()
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(suite.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        case_id = str(row.get("case_id") or "")
        if not case_id or case_id in seen:
            raise ContractError(f"missing or duplicate case_id at line {line_number}")
        seen.add(case_id)
        task = TaskSpec.from_mapping(row["task"])
        decision = select_route(task, models, roles, profile)
        expected = row["expected"]
        passed = (
            decision.status == expected["status"]
            and decision.selected_model == expected.get("selected_model")
            and decision.transport == expected.get("transport")
        )
        cases.append(
            {
                "case_id": case_id,
                "passed": passed,
                "expected": expected,
                "actual": {
                    "status": decision.status,
                    "selected_model": decision.selected_model,
                    "transport": decision.transport,
                },
            }
        )
    if not cases:
        raise ContractError("routing eval suite is empty")
    passed_count = sum(case["passed"] is True for case in cases)
    return {
        "schema_version": 1,
        "suite": str(suite.resolve()),
        "profile": profile_name,
        "case_count": len(cases),
        "passed_count": passed_count,
        "pass_rate": passed_count / len(cases),
        "status": "PASSED" if passed_count == len(cases) else "FAILED",
        "cases": cases,
    }
