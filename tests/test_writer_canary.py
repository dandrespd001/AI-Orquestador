from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from ai_orchestrator.models import ModelConfig, RunResult
from ai_orchestrator.writer_canary import run_writer_canaries


class WriterCanaryTests(unittest.TestCase):
    @patch("ai_orchestrator.writer_canary.execute")
    @patch("ai_orchestrator.writer_canary.load_models")
    @patch("ai_orchestrator.writer_canary.secrets.token_hex", return_value="fixed")
    def test_five_clean_attempts_are_promotion_ready_without_content_retention(
        self, token_hex, load_models, execute
    ) -> None:
        load_models.return_value = {
            "candidate": ModelConfig(
                id="candidate",
                family="test",
                provider="test",
                transport="qwen_omniroute_cli",
                route_model="test/candidate",
                official=False,
                status="candidate",
                roles=("boilerplate",),
                max_sensitivity="public",
                max_concurrency=1,
                trust_tier="web_free",
                capabilities=("read", "write", "tools"),
                execution_profile="patch_api_writer",
            )
        }
        execute.return_value = (
            RunResult(
                schema_version=1,
                run_id="run",
                decision_id="decision",
                task_id="task",
                requested_model="test/candidate",
                reported_model="test/candidate",
                provider="test",
                transport="qwen_omniroute_cli",
                status="PATCH_READY",
                started_at="2026-08-12T00:00:00Z",
                finished_at="2026-08-12T00:00:01Z",
                latency_ms=1,
                exit_code=0,
                error_code=None,
                output_sha256="hash",
                output_bytes=32,
                provenance_verified=True,
                changed_files=1,
                patch_sha256="patch",
                repairs=0,
                gates_passed=True,
            ),
            b"diff --git a/canary.txt b/canary.txt\n+CANARY_fixed\n",
        )

        receipt = run_writer_canaries("candidate", attempts=5)

        self.assertEqual("PASSED", receipt["status"])
        self.assertTrue(receipt["promotion_ready"])
        self.assertEqual(5, execute.call_count)
        self.assertNotIn("CANARY_fixed", json.dumps(receipt))
        self.assertTrue(all(row["exit_code"] == 0 for row in receipt["results"]))
        self.assertTrue(all(row["error_code"] is None for row in receipt["results"]))
        self.assertTrue(all(not row["orchestrator_content_retained"] for row in receipt["results"]))


if __name__ == "__main__":
    unittest.main()
