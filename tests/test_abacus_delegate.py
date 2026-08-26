from __future__ import annotations

import os
import unittest
from pathlib import Path

from ai_orchestrator.abacus import AbacusDelegator, select_abacus_model
from ai_orchestrator.config import load_models
from ai_orchestrator.mcp_tools import MCPQueryServer
from ai_orchestrator.models import ContractError
from ai_orchestrator.providers import load_provider_specs
from ai_orchestrator.supervisor import ProcessOutcome

ROOT = Path(__file__).resolve().parents[1]


class AbacusRoutingTests(unittest.TestCase):
    def test_quality_price_routes_are_task_specific(self) -> None:
        self.assertEqual("Grok 4.6", select_abacus_model("review"))
        self.assertEqual("Grok 4.6", select_abacus_model("reasoning"))
        self.assertEqual("Qwen3.8 Max", select_abacus_model("agentic_implementation"))
        self.assertEqual("Qwen3.8 Max", select_abacus_model("bulk_refactor"))

    def test_experimental_glm_53_requires_an_explicit_opt_in(self) -> None:
        with self.assertRaises(ContractError):
            select_abacus_model("review", requested_model="ZAI GLM 5.3")
        self.assertEqual(
            "ZAI GLM 5.3",
            select_abacus_model(
                "review", requested_model="ZAI GLM 5.3", allow_experimental=True
            ),
        )

    def test_provider_catalog_lists_observed_abacus_models(self) -> None:
        abacus = load_provider_specs(ROOT / "config")["abacus"]
        self.assertEqual(
            (
                "abacus-grok-4.6",
                "abacus-qwen3.8-max",
                "abacus-qwen3.8-27b",
                "abacus-glm-5.3",
            ),
            abacus.models,
        )

    def test_common_model_inventory_includes_abacus_promotion_status(self) -> None:
        models = load_models(ROOT / "config")
        self.assertEqual("available", models["abacus-grok-4.6"].status)
        self.assertEqual("available", models["abacus-qwen3.8-max"].status)
        self.assertEqual("candidate", models["abacus-qwen3.8-27b"].status)
        self.assertEqual("candidate", models["abacus-glm-5.3"].status)


class AbacusDelegatorTests(unittest.TestCase):
    def test_prompt_is_only_sent_over_stdin_in_a_private_empty_workspace(self) -> None:
        captured: dict[str, object] = {}

        def fake_runner(command: list[str], **kwargs: object) -> ProcessOutcome:
            captured["command"] = command
            captured.update(kwargs)
            captured["cwd_mode"] = os.stat(str(kwargs["cwd"])).st_mode & 0o777
            return ProcessOutcome(0, b"bounded answer\n", b"")

        os.environ["OPENAI_API_KEY"] = "must-not-cross-boundary"
        try:
            receipt = AbacusDelegator(
                executable="/opt/abacusai-2.6.9", runner=fake_runner
            ).delegate(prompt="private prompt", task_type="review")
        finally:
            os.environ.pop("OPENAI_API_KEY", None)

        command = captured["command"]
        self.assertIsInstance(command, list)
        self.assertNotIn("private prompt", command)
        self.assertEqual(b"private prompt", captured["input_data"])
        self.assertEqual("Grok 4.6", receipt["model"])
        self.assertEqual("bounded answer\n", receipt["result"])
        self.assertFalse(receipt["automatic_purchase"])
        self.assertFalse(receipt["incremental_spend_allowed"])
        self.assertEqual([], receipt["allowed_tools"])
        self.assertNotIn("prompt", receipt)

        environment = captured["env"]
        self.assertIsInstance(environment, dict)
        self.assertNotIn("OPENAI_API_KEY", environment)
        self.assertEqual("0", environment["AI_ORCHESTRATOR_AUTO_PURCHASE"])
        workspace = str(captured["cwd"])
        self.assertNotEqual(os.getcwd(), workspace)
        self.assertEqual(0o700, captured["cwd_mode"])
        self.assertFalse(os.path.exists(workspace))

    def test_rejects_sensitive_or_oversized_requests_and_outputs(self) -> None:
        delegator = AbacusDelegator(
            executable="/opt/abacusai-2.6.9",
            runner=lambda *_args, **_kwargs: ProcessOutcome(
                0, b"x" * (256 * 1024 + 1), b""
            ),
        )
        with self.assertRaises(ContractError):
            delegator.delegate(
                prompt="do this", task_type="review", sensitivity="confidential"
            )
        with self.assertRaises(ContractError):
            delegator.delegate(prompt="x" * 100_001, task_type="review")
        with self.assertRaises(ContractError):
            delegator.delegate(prompt="do this", task_type="review")
        with self.assertRaises(ContractError):
            delegator.delegate(
                prompt="do this", task_type="review", timeout_seconds=True
            )
        with self.assertRaises(ContractError):
            delegator.delegate(
                prompt="do this", task_type="review", allow_experimental="yes"
            )

    def test_failure_does_not_replay_stderr_or_prompt(self) -> None:
        secret = "never-replay-this"
        delegator = AbacusDelegator(
            executable="/opt/abacusai-2.6.9",
            runner=lambda *_args, **_kwargs: ProcessOutcome(
                7, b"", f"provider error: {secret}".encode()
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "Abacus CLI failed") as raised:
            delegator.delegate(prompt=secret, task_type="review")
        self.assertNotIn(secret, str(raised.exception))


class AbacusMCPTests(unittest.TestCase):
    def test_strict_mcp_keeps_abacus_behind_common_task_submission(self) -> None:
        class FakeDelegator:
            def delegate(self, **kwargs: object) -> dict[str, object]:
                return {"model": "Grok 4.6", "received": sorted(kwargs)}

        server = MCPQueryServer(object(), delegator=FakeDelegator())
        listed = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        names = {tool["name"] for tool in listed["result"]["tools"]}
        self.assertNotIn("orchestrator_delegate_abacus", names)
        self.assertIn("orchestrator_submit_task", names)

        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "orchestrator_delegate_abacus",
                    "arguments": {"prompt": "review this", "task_type": "review"},
                },
            }
        )
        self.assertEqual(-32602, response["error"]["code"])


if __name__ == "__main__":
    unittest.main()
