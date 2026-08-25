"""Availability/compliance split used by doctor and session attestation."""

from __future__ import annotations

from typing import Any

from ai_orchestrator.models import ContractError


def evaluate_doctor(
    *,
    check: str,
    configuration: dict[str, object],
    commands: dict[str, object],
    mcp: dict[str, object],
    broker: dict[str, object],
    authentication: list[dict[str, object]],
    omniroute: dict[str, object],
    config_sync: dict[str, object],
    storage: dict[str, object],
) -> dict[str, Any]:
    if check not in {"readiness", "full"}:
        raise ContractError("doctor check must be readiness or full")
    authentication_ok = bool(authentication) and all(
        row.get("status") == "available" for row in authentication
    )
    readiness = {
        "configuration": configuration.get("ok") is True,
        "commands": bool(commands) and all(bool(value) for value in commands.values()),
        "mcp": mcp.get("configured") is True,
        "broker": broker.get("reachable") is True,
        "authentication": authentication_ok,
        "omniroute": (
            omniroute.get("configured") is True and omniroute.get("reachable") is True
        ),
        "config_sync": config_sync.get("in_sync") is True,
    }
    ready = all(readiness.values())
    compliance_ok = storage.get("ok") is True
    return {
        "schema_version": 2,
        "check": check,
        "ok": ready and (compliance_ok if check == "full" else True),
        "readiness": readiness,
        "configuration": configuration,
        "commands": commands,
        "mcp": mcp,
        "broker": broker,
        "authentication": authentication,
        "omniroute": omniroute,
        "config_sync": config_sync,
        "compliance": {"storage": storage, "ok": compliance_ok},
    }
