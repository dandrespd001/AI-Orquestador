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
from ai_orchestrator.models import NATIVE_WRITER_PROFILES, ContractError, TaskSpec
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
        # 2026-08-18: Flash pasa a `native_patch_writer` (canario 5/5, decision
        # de coste del Director). El test afirmaba el perfil CONCRETO y se puso
        # rojo al promoverlo — codificaba el estado, no la propiedad. Lo que de
        # verdad protege es que un escritor externo NUNCA auto-aplique: eso lo
        # decide `native_writer` a solas, y Flash no lo es ni debe serlo.
        self.assertIn(decision.execution_profile, NATIVE_WRITER_PROFILES)
        self.assertNotEqual("native_writer", decision.execution_profile)
        self.assertTrue(decision.auto_execute)
        self.assertFalse(decision.fallback_allowed)

    def test_pin_narrows_the_candidates_to_one_model(self) -> None:
        """Fijar el modelo es lo que permite comparar dos rutas en la misma tarea.

        Sin esto, medir a Pro contra Flash exigiria reordenar roles.toml — es
        decir, promover antes de medir, el orden contrario al que pide la
        politica.
        """

        # Pro solo esta permitido en el perfil `chunsa`, cuyo repo_root apunta
        # al repo real: para probar el anclaje en aislamiento hay que ampliar
        # el perfil `default` a proposito.
        base = load_profile("default")
        profile = replace(
            base, allowed_models=base.allowed_models + ("deepseek-v4-pro",)
        )
        decision = select_route(
            self.task(
                task_type="bounded_cpp",
                write_mode="validated_patch",
                allowed_write_paths=["tests"],
                gate_profile="python-unittest",
            ),
            load_models(),
            load_roles(),
            profile,
            pin="deepseek-v4-pro",
        )
        self.assertEqual("deepseek-v4-pro", decision.selected_model)
        self.assertEqual("native_patch_writer", decision.execution_profile)
        self.assertIn("pin", decision.reason)
        self.assertEqual("not_pinned", decision.unavailable["deepseek-v4-flash"])

    def test_pin_cannot_widen_past_policy(self) -> None:
        """El anclaje estrecha, nunca concede: un modelo inelegible sigue fuera."""

        decision = select_route(
            self.task(
                task_type="bounded_cpp",
                sensitivity="internal",
                write_mode="validated_patch",
                allowed_write_paths=["tests"],
                gate_profile="python-unittest",
            ),
            load_models(),
            load_roles(),
            load_profile("default"),
            pin="gemini-free-fast",
        )
        self.assertEqual("NO_ROUTE", decision.status)
        self.assertIsNone(decision.selected_model)

    def test_pin_outside_the_role_is_refused(self) -> None:
        decision = select_route(
            self.task(
                task_type="bounded_cpp",
                write_mode="validated_patch",
                allowed_write_paths=["tests"],
                gate_profile="python-unittest",
            ),
            load_models(),
            load_roles(),
            load_profile("default"),
            pin="claude-opus",
        )
        self.assertEqual("NO_ROUTE", decision.status)

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
        # Ver la nota de arriba: se comprueba la propiedad (escribe en el arbol
        # pero no auto-aplica), no el perfil concreto.
        self.assertIn(decision.execution_profile, NATIVE_WRITER_PROFILES)
        self.assertNotEqual("native_writer", decision.execution_profile)
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

    def test_long_context_uses_minimax_even_con_kimi_disponible(self) -> None:
        # 2026-08-12: MiniMax pasa a `available` tras rotar la credencial,
        # cifrarla con systemd-creds y aprobar el canario, asi que ya NO se
        # salta.
        #
        # 2026-08-18: kimi-official pasa a `available` (cuota del Director,
        # canario 5/5). El test ANTERIOR afirmaba `unavailable["kimi-official"]
        # == "quota_unavailable"` y se puso rojo al promoverlo — codificaba el
        # estado, no la propiedad. Lo que de verdad protege es el ORDEN del rol:
        # MiniMax primero en long_context, y kimi detras aunque este disponible.
        models = load_models()
        self.assertEqual("available", models["kimi-official"].status)
        decision = select_route(
            self.task(task_type="long_context"),
            models,
            load_roles(),
            load_profile("default"),
        )
        self.assertEqual("MiniMax-M3", decision.selected_model)
        self.assertEqual("qwen_minimax_cli", decision.transport)
        self.assertNotIn("MiniMax-M3", decision.unavailable)
        # Kimi ya no aparece como no disponible: es elegible y simplemente
        # pierde por orden.
        self.assertNotIn("kimi-official", decision.unavailable)

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
