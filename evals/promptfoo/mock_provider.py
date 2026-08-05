"""Deterministic provider used only to verify the local Promptfoo harness."""

from __future__ import annotations

from typing import Any


def call_api(prompt: str, options: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    expected = str(context.get("vars", {}).get("expected", ""))
    return {
        "output": expected,
        "cached": False,
        "latencyMs": 0,
        "tokenUsage": {"total": 0, "prompt": 0, "completion": 0, "numRequests": 0},
    }
