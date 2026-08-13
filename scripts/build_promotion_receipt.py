"""Build the committable promotion receipt FROM machine state.

Why this exists: the previous receipt
(``docs/receipts/chunsa-balanced-profile-2026-08-12.json``) was maintained by
hand and drifted from the machine. It claimed "0 of 5" write canaries for four
candidate models when the real receipts recorded **0 of 1**, and it undercounted
DeepSeek and MiniMax, which are both 5/5. Those two statements point at opposite
conclusions: 0/5 means rejected, 0/1 means untested.

A receipt that a human edits is an assertion about evidence, not evidence. This
script derives it, so the two cannot diverge again.

Source of truth:
  * write canaries -> ``~/.local/share/ai-orchestrator/*-writer-canary.json``
  * model registry -> ``config/models.toml`` in this repository
"""

from __future__ import annotations

import argparse
import json
import time
import tomllib
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1]
STATE_ROOT = Path.home() / ".local" / "share" / "ai-orchestrator"
MODELS_TOML = SOURCE / "config" / "models.toml"
DEFAULT_OUTPUT = SOURCE / "docs" / "receipts" / "model-promotion.json"

# ``writer_canary.py`` emits these per attempt. A receipt missing them predates
# the current code and must not be trusted as a record of today's behaviour.
CURRENT_ATTEMPT_FIELDS = {"exit_code", "error_code"}

PROMOTION_MIN_ATTEMPTS = 5


def load_models() -> dict[str, dict]:
    with MODELS_TOML.open("rb") as handle:
        registry = tomllib.load(handle)
    return {entry["id"]: entry for entry in registry.get("models", [])}


def read_canaries() -> list[dict]:
    canaries: list[dict] = []
    for path in sorted(STATE_ROOT.glob("*-writer-canary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        attempts = payload.get("results") or []
        # A stale receipt is worse than a missing one, because it reads as
        # current. Flag it rather than silently folding it into the totals.
        seen_fields = set().union(*(a.keys() for a in attempts)) if attempts else set()
        canaries.append(
            {
                "model": payload.get("model"),
                "provider": payload.get("provider"),
                "attempts": payload.get("attempts"),
                "passed": payload.get("passed"),
                "promotion_ready": payload.get("promotion_ready"),
                "statuses": sorted({a.get("status") for a in attempts if a.get("status")}),
                "provenance_verified": sorted(
                    {a.get("provenance_verified") for a in attempts
                     if a.get("provenance_verified") is not None}
                ),
                "receipt_predates_current_code": bool(
                    attempts and not (CURRENT_ATTEMPT_FIELDS & seen_fields)
                ),
                # Only the file name. The full path pins the operator's user
                # name and disk layout into the history of a repository that
                # declares itself public, and it adds nothing: STATE_ROOT is
                # right above.
                "source": path.name,
            }
        )
    return canaries


def classify(canary: dict, model: dict | None) -> str:
    """Untested is not the same as rejected. Keep them apart."""
    if canary.get("promotion_ready"):
        return "promoted" if (model or {}).get("status") == "available" else "eligible"
    attempts = canary.get("attempts") or 0
    if attempts < PROMOTION_MIN_ATTEMPTS:
        return "untested"
    return "rejected"


def build() -> dict:
    models = load_models()
    canaries = read_canaries()
    for canary in canaries:
        model = models.get(canary["model"])
        canary["registry_status"] = (model or {}).get("status", "unknown")
        canary["trust_tier"] = (model or {}).get("trust_tier", "unknown")
        canary["verdict"] = classify(canary, model)

    stale = [c["model"] for c in canaries if c["receipt_predates_current_code"]]

    return {
        "schema_version": 1,
        "kind": "model_promotion",
        "source_root": "~/.local/share/ai-orchestrator",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "scripts/build_promotion_receipt.py",
        "evidence_health": {
            "stale_receipts": stale,
            # Every attempt row written by the current writer_canary.py carries
            # exit_code and error_code. A row without them was produced by older
            # code, so the promotion it supports was never re-measured.
            "note": "receipts lacking exit_code/error_code predate the current "
                    "writer_canary.py; re-run the canaries to refresh them",
        },
        "promotion_rule": {
            "min_attempts": PROMOTION_MIN_ATTEMPTS,
            "required": "all attempts pass; promotion_ready is advisory and "
                        "does not change models.toml, which is edited by hand",
        },
        "models_without_canary": sorted(
            model_id for model_id, entry in models.items()
            if entry.get("execution_profile") != "native_writer"
            and model_id not in {c["model"] for c in canaries}
        ),
        "canaries": canaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()

    receipt = build()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {arguments.output}")
    for canary in receipt["canaries"]:
        flag = " STALE" if canary["receipt_predates_current_code"] else ""
        print(
            f"  {canary['model']:<22} {canary['passed']}/{canary['attempts']}"
            f"  {canary['verdict']}{flag}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
