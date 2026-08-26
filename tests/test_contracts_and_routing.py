from __future__ import annotations

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
from ai_orchestrator.models import (
    NATIVE_WRITER_PROFILES,
    ContractError,
    ModelConfig,
    Profile,
    RoleConfig,
    TaskSpec,
)
from ai_orchestrator.routing import plan_route, select_route

ROOT = Path(__file__).resolve().parents[1]


class ContractsAndRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous = os.environ.get("AI_ORCHESTRATOR_CONFIG")
        os.environ["AI_ORCHESTRATOR_CONFIG"] = str(ROOT / "config")
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name)
        (self.repo / "prompt.txt").write_text(
            "Implement the bounded change", encoding="utf-8"
        )
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
            all(
                model.official
                for model in models.values()
                if model.trust_tier == "official_subscription"
            )
        )
        # 2026-08-25: el Director abrio el trabajo interno a las rutas
        # gratuitas. El invariante deja de ser "solo publico" y pasa a ser el
        # que de verdad protege: una ruta no oficial no ve nunca nada por
        # encima de interno.
        self.assertTrue(
            all(
                model.max_sensitivity in {"public", "internal"}
                for model in models.values()
                if model.trust_tier == "web_free"
            )
        )
        promoted_free = {
            "gemini-free-fast",
            "openrouter-north-code-free",
            "mistral-devstral-free",
            "cloudflare-qwen25-coder-free",
        }
        self.assertTrue(
            all(models[model_id].status == "available" for model_id in promoted_free)
        )
        self.assertTrue(
            all(
                model.status == "candidate"
                for model in models.values()
                if model.trust_tier != "official_subscription"
                and model.id not in promoted_free
            )
        )

    def test_tests_route_to_guarded_deepseek_writer_without_fallback(self) -> None:
        # 2026-08-25: con las rutas gratuitas abiertas al trabajo interno, el
        # rol tests lo encabeza gemini-free-fast para conservar cuota. DeepSeek
        # sigue siendo el escritor que cubre todo lo que la ruta gratuita no
        # puede tomar, y es esa garantia la que se ancla aqui.
        decision = select_route(
            self.task(
                write_mode="validated_patch",
                allowed_write_paths=["tests"],
                gate_profile="python-unittest",
            ),
            load_models(),
            load_roles(),
            load_profile("default"),
            pin="deepseek-v4-flash",
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

    def test_promoted_openrouter_free_route_keeps_policy_gates(self) -> None:
        task = self.task(
            sensitivity="internal",
            write_mode="validated_patch",
            allowed_write_paths=["tests"],
            gate_profile="python-unittest",
        )
        decision = select_route(
            task,
            load_models(),
            load_roles(),
            load_profile("default"),
            pin="openrouter-north-code-free",
        )
        self.assertEqual("openrouter-north-code-free", decision.selected_model)
        self.assertEqual("openrouter-free", decision.provider)
        self.assertTrue(decision.auto_execute)
        self.assertFalse(decision.fallback_allowed)

        confidential = select_route(
            replace(task, sensitivity="confidential"),
            load_models(),
            load_roles(),
            load_profile("default"),
            pin="openrouter-north-code-free",
        )
        self.assertEqual("NO_ROUTE", confidential.status)
        self.assertEqual(
            "web_free_sensitivity_not_allowed",
            confidential.unavailable["openrouter-north-code-free"],
        )

    def test_web_free_gate_has_two_modes_and_never_leaks_confidential(self) -> None:
        def route(profile_name: str, sensitivity: str):
            return select_route(
                self.task(
                    sensitivity=sensitivity,
                    write_mode="validated_patch",
                    allowed_write_paths=["tests"],
                    gate_profile="python-unittest",
                ),
                load_models(),
                load_roles(),
                replace(load_profile(profile_name), repo_root=None),
            )

        # Modo abierto (default): sale trabajo publico e interno aunque el
        # repositorio sea privado.
        for sensitivity in ("public", "internal"):
            with self.subTest(sensitivity=sensitivity):
                decision = route("default", sensitivity)
                self.assertEqual("gemini-free-fast", decision.selected_model)

        # Modo estricto (algoritmic, el repositorio de trading): alli la ruta
        # gratuita esta fuera de allowed_models, y esa puerta corta antes que
        # la sensibilidad. La denegacion por proveedor la respalda detras.
        strict = route("algoritmic", "internal")
        self.assertEqual("profile_denied", strict.unavailable["gemini-free-fast"])

        # Lo que ningun modo deja salir.
        confidential = route("default", "confidential")
        self.assertEqual(
            "web_free_sensitivity_not_allowed",
            confidential.unavailable["gemini-free-fast"],
        )
        self.assertNotEqual("gemini-free-fast", confidential.selected_model)

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
        (self.repo / "configs" / "risk.yml").write_text(
            "live: false\n", encoding="utf-8"
        )
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

    def test_algoritmic_minimax_runs_direct_since_the_handoff_was_lifted(self) -> None:
        # 2026-08-25: el Director levanto el handoff obligatorio. Antes,
        # minimax_mode = "delegate-minimax-api" forzaba HANDOFF_REQUIRED en
        # cualquier ruta a MiniMax dentro de algoritmic, asi que el modelo no
        # llegaba a ejecutar nunca en ese repositorio. Lo que sigue protegido
        # es que el transporte sea el CLI real y que las puertas normales de
        # sensibilidad y rutas protegidas sigan aplicando.
        models = load_models()
        models["MiniMax-M3"] = replace(models["MiniMax-M3"], status="available")
        task = self.task(task_type="long_context", input_paths=[])
        profile = replace(load_profile("algoritmic"), repo_root=str(self.repo))
        decision = select_route(task, models, load_roles(), profile)
        self.assertEqual("MiniMax-M3", decision.selected_model)
        self.assertEqual("qwen_minimax_cli", decision.transport)
        self.assertNotEqual("delegate-minimax-api", decision.transport)

    def test_kimi_uses_supervised_acp_when_quota_returns(self) -> None:
        # El vehiculo es `godot_ui`, no `research`: el 2026-08-25 kimi-official
        # salio de la lista de research para que su cuota se gaste en
        # desarrollo, interfaces y graficos. Lo que este test prueba -- que Kimi
        # se ejecuta por ACP supervisado y auto_execute -- no cambia; solo se
        # comprueba sobre el rol donde Kimi vive ahora.
        models = load_models()
        models["kimi-official"] = replace(models["kimi-official"], status="available")
        decision = select_route(
            self.task(task_type="godot_ui", input_paths=[]),
            models,
            load_roles(),
            load_profile("default"),
            pin="kimi-official",
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

    def test_real_route_matches_the_scored_route_plan(self) -> None:
        task = self.task(input_paths=[])
        models = {
            "first": ModelConfig(
                "first",
                "family-a",
                "provider-a",
                "codex_cli",
                "first",
                True,
                "available",
                ("tests",),
                "internal",
                1,
                quality_score=0.10,
            ),
            "best": ModelConfig(
                "best",
                "family-b",
                "provider-b",
                "codex_cli",
                "best",
                True,
                "available",
                ("tests",),
                "internal",
                1,
                quality_score=0.95,
            ),
        }
        roles = {"tests": RoleConfig("tests", ("first", "best"), ())}
        profile = Profile(
            "test",
            None,
            ("internal",),
            ("first", "best"),
            (),
            (),
            (),
            "direct",
        )

        plan = plan_route(task, models, roles, profile)
        decision = select_route(task, models, roles, profile)

        self.assertEqual("best", plan.selected_model)
        self.assertEqual(plan.selected_model, decision.selected_model)
        self.assertIn("score=", decision.reason)

    def test_directory_context_hash_tracks_content_and_tree_changes(self) -> None:
        source = self.repo / "src"
        source.mkdir()
        first = source / "first.py"
        first.write_text("VALUE = 1\n", encoding="utf-8")
        task = self.task(input_paths=["src"])

        original = task.context_sha256()
        first.write_text("VALUE = 2\n", encoding="utf-8")
        modified = task.context_sha256()
        first.rename(source / "renamed.py")
        renamed = task.context_sha256()
        (source / "renamed.py").unlink()
        deleted = task.context_sha256()

        self.assertEqual(4, len({original, modified, renamed, deleted}))


if __name__ == "__main__":
    unittest.main()
