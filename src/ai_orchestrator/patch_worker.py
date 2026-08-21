"""Private subprocess boundary for hard-timeout patch API calls."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

from ai_orchestrator.models import RouteDecision, TaskSpec
from ai_orchestrator.patch_api import PatchApiError, generate_patch


def main() -> int:
    try:
        payload = json.loads(sys.stdin.buffer.read(4_000_001))
        if not isinstance(payload, dict):
            raise TypeError("invalid worker payload")
        task = TaskSpec.from_mapping(payload["task"])
        decision_payload = dict(payload["decision"])
        decision_payload["alternatives"] = tuple(
            decision_payload.get("alternatives", ())
        )
        decision = RouteDecision(**decision_payload)
        prompt = base64.b64decode(payload["prompt_b64"], validate=True)
        outcome = generate_patch(
            task,
            decision,
            worktree=Path(str(payload["worktree"])).resolve(),
            prompt=prompt,
        )
        sys.stdout.write(
            json.dumps(
                {
                    "patch_b64": base64.b64encode(outcome.patch).decode("ascii"),
                    "reported_model": outcome.reported_model,
                },
                separators=(",", ":"),
            )
        )
        return 0
    except PatchApiError as exc:
        # The message may quote provider text, so it stays here. The code is a
        # fixed literal from a closed set (see PatchApiError), which is what the
        # caller needs to tell a quota apart from a dead credential.
        sys.stderr.write(exc.code)
        return 1
    except Exception as exc:  # noqa: BLE001 - sanitize every subprocess failure
        # Do not echo provider bodies, prompts, paths or credentials across the
        # subprocess boundary.
        sys.stderr.write(type(exc).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
