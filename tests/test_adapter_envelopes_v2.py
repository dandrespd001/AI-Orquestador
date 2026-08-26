from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

from ai_orchestrator.adapters import _process_status, execute
from ai_orchestrator.models import RouteDecision, RunResult, TaskSpec
from ai_orchestrator.supervisor import ProcessOutcome
from ai_orchestrator.writer import GateOutcome


class AdapterEnvelopeSymbolsTests(unittest.TestCase):
    def test_run_result_carries_usage_and_contract_state(self) -> None:
        names = {field.name for field in fields(RunResult)}
        self.assertIn("usage", names)
        self.assertIn("contract_validated", names)


@unittest.skipUnless("usage" in {field.name for field in fields(RunResult)}, "v2 run result missing")
class AdapterEnvelopeV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name)
        (self.repo / "prompt.txt").write_text("bounded", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def task(self, **overrides: object) -> TaskSpec:
        payload: dict[str, object] = {
            "repo": str(self.repo),
            "task_type": "review",
            "sensitivity": "internal",
            "prompt_file": "prompt.txt",
            "output_contract": "text",
        }
        payload.update(overrides)
        return TaskSpec.from_mapping(payload)

    def decision(self, task: TaskSpec, **overrides: object) -> RouteDecision:
        values: dict[str, object] = {
            "schema_version": 2,
            "decision_id": "d1",
            "task_id": task.task_id,
            "profile": "default",
            "status": "READY",
            "selected_model": "claude-opus",
            "requested_route_model": "opus",
            "provider": "anthropic",
            "family": "anthropic-claude",
            "transport": "claude_cli",
            "reviewer_model": "gpt-5.6-sol",
            "alternatives": (),
            "unavailable": {},
            "auto_execute": True,
            "approval_required": False,
            "reason": "eligible",
            "context_sha256": task.context_sha256(),
        }
        values.update(overrides)
        return RouteDecision(**values)

    @patch("ai_orchestrator.adapters.run_supervised")
    def test_floating_claude_alias_requires_a_concrete_reported_model(self, run) -> None:
        task = self.task()
        decision = self.decision(task)
        run.return_value = ProcessOutcome(0, b'{"result":"answer"}', b"")
        missing, _ = execute(task, decision)
        self.assertEqual("PROVENANCE_UNVERIFIED", missing.status)
        self.assertFalse(missing.provenance_verified)

        run.return_value = ProcessOutcome(
            0,
            json.dumps(
                {
                    "model": "claude-opus-5-20260801",
                    "result": "answer",
                    "usage": {
                        "input_tokens": 120,
                        "output_tokens": 20,
                        "cache_read_input_tokens": 40,
                        "cache_creation_input_tokens": 10,
                    },
                }
            ).encode(),
            b"",
        )
        result, output = execute(task, decision)
        self.assertEqual("SUCCESS", result.status)
        self.assertEqual("claude-opus-5-20260801", result.reported_model)
        self.assertTrue(result.provenance_verified)
        self.assertEqual(b"answer", output)
        self.assertIsNotNone(result.usage)
        self.assertEqual(120, result.usage.input_tokens)
        self.assertEqual(40, result.usage.cached_tokens)
        self.assertEqual(10, result.usage.cache_write_tokens)

    @patch("ai_orchestrator.adapters.run_supervised")
    def test_qwen_envelope_returns_only_the_deliverable_and_usage(self, run) -> None:
        task = self.task()
        decision = self.decision(
            task,
            selected_model="deepseek-v4-flash",
            requested_route_model="ds/deepseek-v4-flash",
            provider="deepseek",
            family="deepseek",
            transport="qwen_omniroute_cli",
        )
        run.return_value = ProcessOutcome(
            0,
            b'[{"type":"system","model":"ds/deepseek-v4-flash"},'
            b'{"type":"result","subtype":"success","is_error":false,'
            b'"result":"answer","usage":{"input_tokens":50,"output_tokens":7}}]',
            b"",
        )
        result, output = execute(task, decision)
        self.assertEqual(b"answer", output)
        self.assertEqual(50, result.usage.input_tokens)
        self.assertEqual(7, result.usage.output_tokens)
        self.assertTrue(result.contract_validated)

    def test_error_classifier_separates_quota_overload_and_network(self) -> None:
        task = self.task()
        decision = self.decision(task)
        self.assertEqual(
            ("QUOTA_UNAVAILABLE", "QUOTA_UNAVAILABLE"),
            _process_status(
                ProcessOutcome(1, b"", b"insufficient balance / quota"), decision
            ),
        )
        self.assertEqual(
            ("TRANSIENT_PROVIDER_ERROR", "PROVIDER_SERVER_ERROR"),
            _process_status(
                ProcessOutcome(1, b"", b"PROVIDER_SERVER_ERROR"), decision
            ),
        )
        self.assertEqual(
            ("NETWORK_ERROR", "PROVIDER_UNREACHABLE"),
            _process_status(
                ProcessOutcome(1, b"", b"PROVIDER_UNREACHABLE"), decision
            ),
        )
        self.assertEqual(
            ("OVERLOADED", "OVERLOADED"),
            _process_status(
                ProcessOutcome(1, b"", b"service overloaded, retry later"), decision
            ),
        )
        self.assertEqual(
            ("NETWORK_ERROR", "NETWORK_ERROR"),
            _process_status(
                ProcessOutcome(1, b"", b"connection reset by peer"), decision
            ),
        )
        self.assertEqual(
            ("QUOTA_UNAVAILABLE", "QUOTA_UNAVAILABLE"),
            _process_status(
                ProcessOutcome(1, b"", b"QUOTA_UNAVAILABLE"), decision
            ),
        )

    @patch("ai_orchestrator.adapters.run_supervised")
    def test_invalid_json_contract_is_rejected_by_adapter(self, run) -> None:
        task = self.task(output_contract="json")
        decision = self.decision(task)
        run.return_value = ProcessOutcome(
            0,
            b'{"model":"claude-opus-5-20260801","result":"not-json"}',
            b"",
        )
        result, _output = execute(task, decision)
        self.assertEqual("CONTRACT_INVALID", result.status)
        self.assertEqual("MALFORMED_JSON", result.error_code)
        self.assertFalse(result.contract_validated)

    @patch("ai_orchestrator.adapters.run_gates")
    @patch("ai_orchestrator.adapters._run_once")
    def test_gate_results_are_structured_without_retaining_gate_output(
        self, run_once, run_gates
    ) -> None:
        (self.repo / "value.txt").write_text("old\n", encoding="utf-8")
        subprocess.run(
            ["git", "init", "-q"], cwd=self.repo, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "add", "prompt.txt", "value.txt"],
            cwd=self.repo,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-qm",
                "baseline",
            ],
            cwd=self.repo,
            check=True,
            capture_output=True,
        )
        task = self.task(
            output_contract="unified_diff",
            write_mode="validated_patch",
            allowed_write_paths=["value.txt"],
        )
        decision = self.decision(
            task,
            write_mode="validated_patch",
            execution_profile="native_patch_writer",
        )

        def edit(_task, _decision, *, prompt, cwd):
            (cwd / "value.txt").write_text("new\n", encoding="utf-8")
            return (
                ProcessOutcome(0, b'{"model":"claude-opus-5-20260801"}', b""),
                "claude-opus-5-20260801",
                True,
            )

        run_once.side_effect = edit
        run_gates.return_value = (
            GateOutcome("python3 -m unittest", 0, "sensitive gate output", False),
        )
        result, _output = execute(
            task, decision, gate_commands=("python3 -m unittest",)
        )
        self.assertEqual(
            (
                {
                    "command": "python3 -m unittest",
                    "returncode": 0,
                    "timed_out": False,
                },
            ),
            result.gate_results,
        )
        self.assertNotIn("output", result.gate_results[0])


if __name__ == "__main__":
    unittest.main()
