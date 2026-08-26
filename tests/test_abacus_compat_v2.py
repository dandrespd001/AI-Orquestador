from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

from ai_orchestrator.abacus import AbacusDelegator
from ai_orchestrator.delegation import OrchestratorTaskRunner
from ai_orchestrator.mcp_tools import MCPQueryServer
from ai_orchestrator.models import ContractError
from ai_orchestrator.supervisor import ProcessOutcome
from ai_orchestrator.telemetry import TelemetryStore

ROOT = Path(__file__).resolve().parents[1]


class AbacusCompatibilityV2Tests(unittest.TestCase):
    def test_abacus_adapter_can_measure_credit_delta(self) -> None:
        self.assertIn("credit_reader", inspect.signature(AbacusDelegator).parameters)
        balances = iter((30_000.0, 29_997.0))
        receipt = AbacusDelegator(
            executable="/opt/abacusai",
            runner=lambda *_args, **_kwargs: ProcessOutcome(0, b"answer", b""),
            credit_reader=lambda: next(balances),
        ).delegate(prompt="review", task_type="review")
        self.assertEqual(3.0, receipt["usage"]["credits_used"])
        self.assertEqual(30_000.0, receipt["usage"]["balance_before"])
        self.assertEqual(29_997.0, receipt["usage"]["balance_after"])

    def test_legacy_abacus_tool_is_absent_from_strict_mcp(self) -> None:
        calls: list[dict[str, object]] = []

        class Runner:
            def authorize_inline(self, **kwargs: object) -> dict[str, object]:
                calls.append(kwargs)
                return {
                    "schema_version": 2,
                    "plan_id": "00000000-0000-0000-0000-000000000123",
                    "selected_model": "abacus-grok-4.6",
                    "fallback_allowed": False,
                }

        class Delegator:
            def delegate(self, **_kwargs: object) -> dict[str, object]:
                return {"model": "Grok 4.6", "result": "answer"}

        server = MCPQueryServer(
            object(), delegator=Delegator(), task_runner=Runner()
        )
        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "orchestrator_delegate_abacus",
                    "arguments": {
                        "prompt": "review",
                        "task_type": "review",
                        "model": "Grok 4.6",
                    },
                },
            }
        )
        self.assertEqual(-32602, response["error"]["code"])
        self.assertEqual([], calls)

    def test_removed_abacus_mcp_tool_cannot_write_telemetry(self) -> None:
        class Runner:
            def authorize_inline(self, **_kwargs: object) -> dict[str, object]:
                return {
                    "schema_version": 2,
                    "plan_id": "00000000-0000-0000-0000-000000000123",
                    "selected_model": "abacus-grok-4.6",
                    "fallback_allowed": False,
                }

        class Delegator:
            def delegate(self, **_kwargs: object) -> dict[str, object]:
                return {
                    "model": "Grok 4.6",
                    "result": "answer that must not reach telemetry",
                    "usage": {
                        "credits_used": 3.0,
                        "balance_before": 30_000.0,
                        "balance_after": 29_997.0,
                        "latency_ms": 12,
                    },
                }

        with tempfile.TemporaryDirectory() as directory:
            telemetry = TelemetryStore(Path(directory) / "telemetry.sqlite")
            server = MCPQueryServer(
                object(),
                delegator=Delegator(),
                task_runner=Runner(),
                telemetry=telemetry,
            )
            response = server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": "orchestrator_delegate_abacus",
                        "arguments": {
                            "prompt": "review",
                            "task_type": "review",
                        },
                    },
                }
            )
            report = telemetry.report()
            database_bytes = telemetry.path.read_bytes()
        self.assertEqual(-32602, response["error"]["code"])
        self.assertIsNone(report["usage"]["credits_used"])
        self.assertIsNone(report["usage"]["marginal_cost_usd"])
        self.assertNotIn(b"answer that must not reach telemetry", database_bytes)

    def test_abacus_pin_requires_a_known_nonzero_credit_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner = OrchestratorTaskRunner(
                configuration_root=ROOT / "config",
                telemetry=TelemetryStore(Path(directory) / "telemetry.sqlite"),
            )
            with self.assertRaisesRegex(ContractError, "quota_unknown"):
                runner.authorize_inline(
                    task_type="review",
                    sensitivity="internal",
                    model="abacus-grok-4.6",
                )


if __name__ == "__main__":
    unittest.main()
