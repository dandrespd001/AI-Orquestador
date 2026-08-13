from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from ai_orchestrator.config import (
    load_gates,
    load_models,
    load_profile,
    load_roles,
    validate_configuration,
)
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

    def test_configuration_has_explicit_trust_tiers_and_is_complete(self) -> None:
        report = validate_configuration("default")
        self.assertTrue(report["ok"], report)
        models = load_models()
        self.assertTrue(
            all(model.official for model in models.values() if model.trust_tier == "official_subscription")
        )
        self.assertTrue(
            all(model.max_sensitivity == "public" for model in models.values() if model.trust_tier == "web_free")
        )
        self.assertEqual("available", models["gemini-free-fast"].status)
        self.assertTrue(all(
            model.status == "candidate"
            for model in models.values()
            if model.trust_tier != "official_subscription" and model.id != "gemini-free-fast"
        ))

    def test_tests_route_to_guarded_deepseek_writer_without_fallback(self) -> None:
        decision = select_route(
            self.task(
                write_mode="validated_patch",
                allowed_write_paths=["tests"],
                gate_profile="python-unittest",
            ),
            load_models(),
            load_roles(),
            load_profile("default"),
        )
        self.assertEqual("deepseek-v4-flash", decision.selected_model)
        self.assertEqual("qwen_omniroute_cli", decision.transport)
        self.assertEqual("patch_api_writer", decision.execution_profile)
        self.assertTrue(decision.auto_execute)
        self.assertFalse(decision.fallback_allowed)

    def test_public_tests_use_promoted_free_writer(self) -> None:
        profile = replace(load_profile("default"), public_repository=True)
        decision = select_route(
            self.task(
                sensitivity="public",
                write_mode="validated_patch",
                allowed_write_paths=["tests"],
                gate_profile="python-unittest",
            ),
            load_models(),
            load_roles(),
            profile,
        )
        self.assertEqual("gemini-free-fast", decision.selected_model)
        self.assertEqual("web_free", decision.trust_tier)
        self.assertTrue(decision.auto_execute)

    def test_web_free_model_requires_public_repository_and_public_task(self) -> None:
        decision = select_route(
            self.task(
                sensitivity="public",
                write_mode="validated_patch",
                allowed_write_paths=["tests"],
                gate_profile="python-unittest",
            ),
            load_models(),
            load_roles(),
            load_profile("default"),
        )
        self.assertEqual("deepseek-v4-flash", decision.selected_model)
        self.assertEqual(
            "web_free_requires_public_repository",
            decision.unavailable["gemini-free-fast"],
        )

    def test_chunsa_profile_and_long_gate_are_valid(self) -> None:
        report = validate_configuration("chunsa")
        self.assertTrue(report["ok"], report)
        profile = load_profile("chunsa")
        self.assertTrue(profile.public_repository)
        self.assertEqual(
            "/home/adquiod/Imágenes/Project/CHUNSA001",
            profile.repo_root,
        )
        gates = load_gates()
        self.assertEqual(1200, gates["chunsa-determinism"].timeout_seconds)

    def test_chunsa_determinism_uses_sol_with_xhigh_effort(self) -> None:
        profile = replace(load_profile("chunsa"), repo_root=str(self.repo))
        decision = select_route(
            self.task(task_type="determinism"),
            load_models(),
            load_roles(),
            profile,
        )
        self.assertEqual("gpt-5.6-sol", decision.selected_model)
        self.assertEqual("xhigh", decision.reasoning_effort)
        self.assertEqual("xhigh", decision.reviewer_reasoning_effort)

    def test_chunsa_bounded_cpp_uses_deepseek_as_reviewed_patch_writer(self) -> None:
        profile = replace(load_profile("chunsa"), repo_root=str(self.repo))
        decision = select_route(
            self.task(
                task_type="bounded_cpp",
                write_mode="validated_patch",
                allowed_write_paths=["tests"],
                gate_profile="chunsa-standard",
            ),
            load_models(),
            load_roles(),
            profile,
        )
        self.assertEqual("deepseek-v4-flash", decision.selected_model)
        self.assertEqual("patch_api_writer", decision.execution_profile)
        self.assertEqual("gpt-5.6-sol", decision.reviewer_model)
        self.assertEqual("high", decision.reasoning_effort)
        self.assertEqual("xhigh", decision.reviewer_reasoning_effort)

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

    def test_long_context_uses_minimax_and_still_skips_unavailable_kimi(self) -> None:
        # 2026-08-12: MiniMax pasa a `available` tras rotar la credencial,
        # cifrarla con systemd-creds y aprobar el canario, asi que ya NO se
        # salta. Lo que este test sigue protegiendo es que kimi NO se
        # autoejecuta nunca aunque este listado en el rol.
        decision = select_route(
            self.task(task_type="long_context"),
            load_models(),
            load_roles(),
            load_profile("default"),
        )
        self.assertEqual("MiniMax-M3", decision.selected_model)
        self.assertEqual("qwen_minimax_cli", decision.transport)
        self.assertEqual("quota_unavailable", decision.unavailable["kimi-official"])
        self.assertNotIn("MiniMax-M3", decision.unavailable)

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

    def test_kimi_uses_supervised_acp_when_quota_returns(self) -> None:
        models = load_models()
        models["kimi-official"] = replace(models["kimi-official"], status="available")
        decision = select_route(
            self.task(task_type="research", input_paths=[]),
            models,
            load_roles(),
            load_profile("default"),
        )
        self.assertEqual("kimi-official", decision.selected_model)
        self.assertEqual("kimi_cli", decision.transport)
        self.assertTrue(decision.auto_execute)
        self.assertEqual("READY", decision.status)
        self.assertEqual("kimi_acp_writer", decision.execution_profile)

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
