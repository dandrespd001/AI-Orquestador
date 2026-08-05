from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from ai_orchestrator.config import load_models, load_profile, load_roles, validate_configuration
from ai_orchestrator.models import ContractError, TaskSpec
from ai_orchestrator.routing import select_route


ROOT = Path(__file__).resolve().parents[1]


class ContractsAndRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous = os.environ.get("AI_ORCHESTRATOR_CONFIG")
        os.environ["AI_ORCHESTRATOR_CONFIG"] = str(ROOT / "config")
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name)
        (self.repo / "prompt.txt").write_text("Implement the bounded change", encoding="utf-8")
        (self.repo / "tests").mkdir()
        (self.repo / "tests" / "test_example.py").write_text("pass\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()
        if self.previous is None:
            os.environ.pop("AI_ORCHESTRATOR_CONFIG", None)
        else:
            os.environ["AI_ORCHESTRATOR_CONFIG"] = self.previous

    def task(self, **overrides: object) -> TaskSpec:
        payload: dict[str, object] = {
            "repo": str(self.repo),
            "task_type": "tests",
            "sensitivity": "internal",
            "prompt_file": "prompt.txt",
            "input_paths": ["tests/test_example.py"],
            "output_contract": "unified_diff",
        }
        payload.update(overrides)
        return TaskSpec.from_mapping(payload)

    def test_configuration_is_official_only_and_complete(self) -> None:
        report = validate_configuration("default")
        self.assertTrue(report["ok"], report)
        models = load_models()
        self.assertTrue(all(model.official for model in models.values()))
        self.assertNotIn("huggingchat", {model.provider for model in models.values()})

    def test_tests_route_to_deepseek_without_fallback(self) -> None:
        decision = select_route(
            self.task(), load_models(), load_roles(), load_profile("default")
        )
        self.assertEqual("deepseek-v4-flash", decision.selected_model)
        self.assertEqual("omniroute", decision.transport)
        self.assertTrue(decision.auto_execute)
        self.assertFalse(decision.fallback_allowed)
        self.assertEqual("gpt-5.6-sol", decision.reviewer_model)

    def test_protected_algoritmic_input_uses_native_model(self) -> None:
        (self.repo / "configs").mkdir()
        (self.repo / "configs" / "risk.yml").write_text("live: false\n", encoding="utf-8")
        task = self.task(input_paths=["configs/risk.yml"])
        profile = replace(load_profile("algoritmic"), repo_root=str(self.repo))
        decision = select_route(task, load_models(), load_roles(), profile)
        self.assertEqual("gpt-5.6-sol", decision.selected_model)
        self.assertEqual("codex_cli", decision.transport)
        self.assertFalse(decision.auto_execute)
        self.assertEqual("HANDOFF_REQUIRED", decision.status)

    def test_long_context_uses_read_only_official_minimax(self) -> None:
        decision = select_route(
            self.task(task_type="long_context"),
            load_models(),
            load_roles(),
            load_profile("default"),
        )
        self.assertEqual("MiniMax-M3", decision.selected_model)
        self.assertEqual("minimax_official_cli", decision.transport)
        self.assertEqual("quota_unavailable", decision.unavailable["kimi-official"])

    def test_algoritmic_minimax_is_a_supervised_handoff_when_available(self) -> None:
        models = load_models()
        models["MiniMax-M3"] = replace(models["MiniMax-M3"], status="available")
        task = self.task(task_type="long_context", input_paths=[])
        profile = replace(load_profile("algoritmic"), repo_root=str(self.repo))
        decision = select_route(task, models, load_roles(), profile)
        self.assertEqual("MiniMax-M3", decision.selected_model)
        self.assertEqual("delegate-minimax-api", decision.transport)
        self.assertFalse(decision.auto_execute)
        self.assertEqual("HANDOFF_REQUIRED", decision.status)

    def test_path_escape_and_symlink_are_rejected(self) -> None:
        with self.assertRaises(ContractError):
            self.task(prompt_file="../outside.txt")
        target = self.repo / "real.txt"
        target.write_text("x", encoding="utf-8")
        (self.repo / "link.txt").symlink_to(target)
        with self.assertRaises(ContractError):
            self.task(prompt_file="link.txt")

    def test_repository_profile_cannot_be_reused_for_another_repo(self) -> None:
        decision = select_route(
            self.task(), load_models(), load_roles(), load_profile("algoritmic")
        )
        self.assertEqual("NO_ROUTE", decision.status)
        self.assertIn("repo_root", decision.reason)


if __name__ == "__main__":
    unittest.main()
