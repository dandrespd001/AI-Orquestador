from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ai_orchestrator import delegation
from ai_orchestrator.config import load_models, load_profile, load_roles
from ai_orchestrator.delegation import OrchestratorTaskRunner
from ai_orchestrator.models import QuotaSnapshot, RunResult, UsageRecord
from ai_orchestrator.telemetry import TelemetryStore

ROOT = Path(__file__).resolve().parents[1]


class ExecutionEngineSymbolsTests(unittest.TestCase):
    def test_runner_exposes_the_shared_plan_before_execution(self) -> None:
        self.assertTrue(hasattr(OrchestratorTaskRunner, "simulate"))

    def test_actual_usage_is_costed_with_the_selected_model_economics(self) -> None:
        self.assertTrue(hasattr(delegation, "_with_usage_cost"))
        task = SimpleNamespace(task_id="task")
        decision = SimpleNamespace(
            selected_model="deepseek-v4-flash",
            decision_id="decision",
            provider="deepseek",
            transport="qwen_omniroute_cli",
        )
        raw = ExecutionEngineV2Tests.result(task, decision)
        raw = replace(
            raw,
            usage=UsageRecord(
                provider="deepseek",
                model="deepseek-v4-flash",
                outcome="SUCCESS",
                latency_ms=1,
                input_tokens=1_000_000,
                output_tokens=1_000_000,
                cached_tokens=0,
                cache_write_tokens=0,
            ),
        )
        model = load_models(ROOT / "config")["deepseek-v4-flash"]
        costed = delegation._with_usage_cost(raw, model)
        self.assertAlmostEqual(0.42, costed.usage.marginal_cost_usd)
        self.assertAlmostEqual(0.42, costed.usage.equivalent_cost_usd)


@unittest.skipUnless(hasattr(OrchestratorTaskRunner, "simulate"), "v2 runner missing")
class ExecutionEngineV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.repo = root / "repo"
        self.repo.mkdir()
        (self.repo / "prompt.txt").write_text("review this", encoding="utf-8")
        self.spec = root / "task.json"
        self.spec.write_text(
            json.dumps(
                {
                    "task_id": "00000000-0000-0000-0000-000000000111",
                    "repo": str(self.repo),
                    "task_type": "review",
                    "sensitivity": "internal",
                    "prompt_file": "prompt.txt",
                    "output_contract": "text",
                    "assurance": "high",
                    "max_repairs": 1,
                }
            ),
            encoding="utf-8",
        )
        self.telemetry = TelemetryStore(root / "telemetry.sqlite")
        self.telemetry.record_quota(
            QuotaSnapshot(
                provider="openai",
                plan="pro_5x",
                billing_mode="included_quota",
                remaining=100.0,
                limit=100.0,
                used=0.0,
                predicted_usage=1.0,
                window="weekly",
                source="fixture",
                verified_at="2026-08-20T00:00:00+00:00",
                confidence="verified",
            )
        )
        self.telemetry.record_quota(
            QuotaSnapshot(
                provider="abacus",
                plan="abacus_pro_30000_credits",
                billing_mode="measured_credits",
                remaining=30_000.0,
                limit=30_000.0,
                used=0.0,
                predicted_usage=1.0,
                credits_remaining=30_000.0,
                window="monthly",
                source="fixture",
                verified_at="2026-08-20T00:00:00+00:00",
                confidence="verified",
            )
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def result(task: object, decision: object, status: str = "SUCCESS") -> RunResult:
        return RunResult(
            schema_version=2,
            run_id=f"run-{decision.selected_model}-{status}",
            decision_id=decision.decision_id,
            task_id=task.task_id,
            status=status,
            requested_model=decision.selected_model,
            reported_model=decision.selected_model,
            provider=decision.provider,
            transport=decision.transport,
            started_at="2026-08-20T00:00:00+00:00",
            finished_at="2026-08-20T00:00:01+00:00",
            latency_ms=1,
            output_sha256="0" * 64,
            output_bytes=6,
            exit_code=0 if status == "SUCCESS" else 1,
            provenance_verified=True,
            error_code=None if status == "SUCCESS" else status,
            gates_passed=None,
        )

    def runner(self, responses: list[tuple[str, bytes]]):
        calls: list[tuple[str, str]] = []

        def execute(task: object, decision: object, **_kwargs: object):
            calls.append((decision.selected_model, decision.family))
            status, output = responses.pop(0)
            result = self.result(task, decision, status)
            return replace(result, run_id=str(uuid.uuid4())), output

        return (
            OrchestratorTaskRunner(
                configuration_root=ROOT / "config",
                executor=execute,
                telemetry=self.telemetry,
            ),
            calls,
        )

    def test_simulation_and_execution_return_the_exact_same_immutable_plan(self) -> None:
        runner, calls = self.runner(
            [
                ("SUCCESS", b"bounded answer"),
                ("SUCCESS", b'{"verdict":"accept","findings":[]}'),
            ]
        )
        simulated = runner.simulate(
            spec_path=str(self.spec), profile="default", model="claude-opus"
        )
        self.assertEqual(1, self.telemetry.rollout_counts()["claude-opus"])
        receipt = runner.run(
            spec_path=str(self.spec),
            profile="default",
            model="claude-opus",
            confirmed_plan_id=simulated["plan_id"],
        )
        self.assertEqual(simulated, receipt["plan"])
        self.assertTrue(receipt["final_accepted"])
        self.assertEqual("accept", receipt["review"]["verdict"])
        self.assertEqual(2, len(calls))
        self.assertNotEqual(calls[0][1], calls[1][1])

    def test_changed_confirmation_returns_replan_without_calling_a_model(self) -> None:
        runner, calls = self.runner([])
        receipt = runner.run(
            spec_path=str(self.spec),
            profile="default",
            model="claude-opus",
            confirmed_plan_id="00000000-0000-0000-0000-000000000999",
        )
        self.assertEqual("REPLAN_REQUIRED", receipt["status"])
        self.assertEqual([], calls)
        self.assertFalse(receipt["fallback_allowed"])

    def test_exact_deepseek_pin_fails_closed_when_catalog_quota_is_unavailable(
        self,
    ) -> None:
        deepseek_quota = QuotaSnapshot(
            provider="deepseek",
            plan="api_balance",
            billing_mode="prepaid_balance",
            remaining=5.0,
            balance_usd=5.0,
            source="/user/balance",
            verified_at="2026-08-20T00:00:00+00:00",
            confidence="verified",
        )
        self.telemetry.record_quota(deepseek_quota)
        models = load_models(ROOT / "config")
        models["deepseek-v4-pro"] = replace(
            models["deepseek-v4-pro"], rollout_stage="promoted"
        )
        role = replace(
            load_roles(ROOT / "config")["review"],
            models=("deepseek-v4-pro",),
            reviewers=("claude-opus",),
        )
        profile = replace(
            load_profile("default", ROOT / "config"),
            allowed_models=("deepseek-v4-pro", "claude-opus"),
        )
        runner, calls = self.runner([])
        with (
            patch("ai_orchestrator.delegation.load_models", return_value=models),
            patch(
                "ai_orchestrator.delegation.load_roles",
                return_value={"review": role},
            ),
            patch("ai_orchestrator.delegation.load_profile", return_value=profile),
        ):
            simulated = runner.simulate(
                spec_path=str(self.spec),
                profile="default",
                model="deepseek-v4-pro",
                quotas={"deepseek": deepseek_quota},
            )
            self.assertEqual("HANDOFF_REQUIRED", simulated["status"])
            self.assertIsNone(simulated["selected_model"])
            self.assertEqual(
                "quota_unavailable",
                simulated["unavailable"]["deepseek-v4-pro"],
            )
            self.assertFalse(simulated["fallback_allowed"])
        self.assertEqual([], calls)

    def test_one_transient_retry_uses_the_same_model_then_reviews(self) -> None:
        runner, calls = self.runner(
            [
                ("OVERLOADED", b""),
                ("SUCCESS", b"bounded answer"),
                ("SUCCESS", b'{"verdict":"accept","findings":[]}'),
            ]
        )
        receipt = runner.run(
            spec_path=str(self.spec), profile="default", model="claude-opus"
        )
        self.assertTrue(receipt["final_accepted"])
        self.assertEqual(calls[0][0], calls[1][0])
        self.assertEqual(3, len(receipt["attempts"]))

    def test_reviewer_gets_one_transient_retry_on_the_same_model(self) -> None:
        runner, calls = self.runner(
            [
                ("SUCCESS", b"bounded answer"),
                ("NETWORK_ERROR", b""),
                ("SUCCESS", b'{"verdict":"accept","findings":[]}'),
            ]
        )
        receipt = runner.run(
            spec_path=str(self.spec), profile="default", model="claude-opus"
        )
        self.assertTrue(receipt["final_accepted"])
        self.assertEqual(calls[1][0], calls[2][0])
        self.assertEqual(3, len(receipt["attempts"]))

    def test_quota_contract_and_gate_failures_do_not_fallback_or_retry(self) -> None:
        for status, output in (
            ("QUOTA_UNAVAILABLE", b""),
            ("SUCCESS", b" \n"),
            ("GATE_FAILED", b"diff --git invalid"),
        ):
            with self.subTest(status=status, output=output):
                runner, calls = self.runner([(status, output)])
                receipt = runner.run(
                    spec_path=str(self.spec), profile="default", model="claude-opus"
                )
                self.assertFalse(receipt["final_accepted"])
                self.assertEqual(1, len(calls))
                self.assertFalse(receipt["fallback_allowed"])
                self.assertIn(
                    receipt["status"],
                    {"REPLAN_REQUIRED", "CONTRACT_INVALID", "HANDOFF_REQUIRED"},
                )

    def test_revise_gets_one_bounded_repair_and_repeats_review(self) -> None:
        runner, calls = self.runner(
            [
                ("SUCCESS", b"first answer"),
                (
                    "SUCCESS",
                    (
                        b'{"verdict":"revise","findings":['
                        b'{"severity":"high","evidence":"answer:1","message":"fix"}]}'
                    ),
                ),
                ("SUCCESS", b"repaired answer"),
                ("SUCCESS", b'{"verdict":"accept","findings":[]}'),
            ]
        )
        receipt = runner.run(
            spec_path=str(self.spec), profile="default", model="claude-opus"
        )
        self.assertTrue(receipt["final_accepted"])
        self.assertEqual(1, receipt["review_repairs"])
        self.assertEqual(4, len(calls))

    def test_critical_finding_forces_handoff_even_with_accept_verdict(self) -> None:
        runner, _calls = self.runner(
            [
                ("SUCCESS", b"bounded answer"),
                (
                    "SUCCESS",
                    (
                        b'{"verdict":"accept","findings":['
                        b'{"severity":"critical","evidence":"answer:1","message":"unsafe"}]}'
                    ),
                ),
            ]
        )
        receipt = runner.run(
            spec_path=str(self.spec), profile="default", model="claude-opus"
        )
        self.assertEqual("HANDOFF_REQUIRED", receipt["status"])
        self.assertFalse(receipt["final_accepted"])

    def test_output_artifact_is_written_only_after_final_acceptance(self) -> None:
        accepted_path = Path(self.temp.name) / "accepted.txt"
        runner, _calls = self.runner(
            [
                ("SUCCESS", b"bounded answer"),
                ("SUCCESS", b'{"verdict":"accept","findings":[]}'),
            ]
        )
        accepted = runner.run(
            spec_path=str(self.spec),
            profile="default",
            model="claude-opus",
            output_path=accepted_path,
        )
        self.assertTrue(accepted["final_accepted"])
        self.assertEqual("bounded answer", accepted_path.read_text(encoding="utf-8"))

        rejected_path = Path(self.temp.name) / "rejected.txt"
        runner, _calls = self.runner(
            [
                ("SUCCESS", b"unsafe answer"),
                (
                    "SUCCESS",
                    (
                        b'{"verdict":"escalate","findings":['
                        b'{"severity":"critical","evidence":"x","message":"unsafe"}]}'
                    ),
                ),
            ]
        )
        rejected = runner.run(
            spec_path=str(self.spec),
            profile="default",
            model="claude-opus",
            output_path=rejected_path,
        )
        self.assertFalse(rejected["final_accepted"])
        self.assertFalse(rejected_path.exists())

    def test_critical_assurance_executes_both_independent_reviewers(self) -> None:
        payload = json.loads(self.spec.read_text(encoding="utf-8"))
        payload["assurance"] = "critical"
        self.spec.write_text(json.dumps(payload), encoding="utf-8")
        runner, calls = self.runner(
            [
                ("SUCCESS", b"bounded answer"),
                ("SUCCESS", b'{"verdict":"accept","findings":[]}'),
                ("SUCCESS", b'{"verdict":"accept","findings":[]}'),
            ]
        )
        promoted = load_models(ROOT / "config")
        promoted["abacus-grok-4.6"] = replace(
            promoted["abacus-grok-4.6"], rollout_stage="promoted"
        )
        with patch("ai_orchestrator.delegation.load_models", return_value=promoted):
            receipt = runner.run(
                spec_path=str(self.spec), profile="default", model="claude-opus"
            )
        self.assertTrue(receipt["final_accepted"])
        self.assertEqual(2, len(receipt["reviews"]))
        self.assertEqual(3, len(calls))


if __name__ == "__main__":
    unittest.main()
