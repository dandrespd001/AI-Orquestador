from __future__ import annotations

import json
import subprocess
import unittest
from unittest.mock import patch

from ai_orchestrator.models import ContractError, ModelConfig, RunResult
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
        self.assertTrue(
            all(not row["orchestrator_content_retained"] for row in receipt["results"])
        )


class ProfileProbeTests(unittest.TestCase):
    """Medir un perfil antes de escribirlo en la configuracion.

    La politica es «permitido no es promovido»: un modelo no cambia de perfil
    hasta que cinco pasadas limpias lo justifiquen. Pero el canario leia el
    perfil de models.toml, asi que para medir la ruta nativa habia que
    promoverla antes — exactamente al reves.
    """

    def model(self) -> dict[str, object]:
        return {
            "candidate": ModelConfig(
                id="candidate",
                family="test",
                provider="test",
                transport="qwen_omniroute_cli",
                route_model="test/candidate",
                official=True,
                status="available",
                roles=("implementation",),
                max_sensitivity="internal",
                max_concurrency=1,
                trust_tier="official_subscription",
                capabilities=("read", "write", "tools"),
                execution_profile="patch_api_writer",
            )
        }

    @patch("ai_orchestrator.writer_canary.execute")
    @patch("ai_orchestrator.writer_canary.load_models")
    def test_canary_can_probe_a_profile_the_model_does_not_have_yet(
        self, load_models, execute
    ) -> None:
        load_models.return_value = self.model()
        execute.side_effect = AssertionError("stop after the decision is built")

        with self.assertRaises(AssertionError):
            run_writer_canaries(
                "candidate", attempts=1, execution_profile="native_patch_writer"
            )

        decision = execute.call_args.args[1]
        self.assertEqual("native_patch_writer", decision.execution_profile)

    @patch("ai_orchestrator.writer_canary.execute")
    @patch("ai_orchestrator.writer_canary.load_models")
    def test_probing_an_unknown_profile_is_refused(self, load_models, execute) -> None:
        load_models.return_value = self.model()
        with self.assertRaises(ContractError):
            run_writer_canaries(
                "candidate", attempts=1, execution_profile="root_writer"
            )
        execute.assert_not_called()

    @patch("ai_orchestrator.writer_canary.execute")
    @patch("ai_orchestrator.writer_canary.load_models")
    def test_canary_measures_editing_a_tracked_file_not_creating_one(
        self, load_models, execute
    ) -> None:
        """Medido el 2026-08-14 contra ds/deepseek-v4-pro.

        Qwen 0.21.12 en modo --bare no expone `write_file`: las unicas
        herramientas de escritura son `edit` y `notebook_edit`. Un canario que
        pide crear un fichero devuelve NO_CHANGES aunque la ruta funcione —el
        modelo razono «let me use write_file» y no la encontro—, mientras que
        editar un fichero rastreado sale a la primera. El trabajo real es
        editar step.hpp, no crear ficheros, asi que el canario medía el caso
        que no importa.
        """

        load_models.return_value = self.model()
        seen: dict[str, object] = {}

        def inspect(task, _decision, **_kwargs):
            # El repo desechable solo existe mientras corre el canario.
            seen["tracked"] = subprocess.run(
                ["git", "-C", str(task.repo), "ls-files"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.split()
            seen["allowed"] = task.allowed_write_paths
            raise AssertionError("stop after the fixture is built")

        execute.side_effect = inspect
        with self.assertRaises(AssertionError):
            run_writer_canaries("candidate", attempts=1)

        self.assertIn("canary.txt", seen["tracked"])
        self.assertEqual(("canary.txt",), seen["allowed"])

    @patch("ai_orchestrator.writer_canary.execute")
    @patch("ai_orchestrator.writer_canary.load_models")
    def test_the_configured_profile_is_used_when_none_is_given(
        self, load_models, execute
    ) -> None:
        load_models.return_value = self.model()
        execute.side_effect = AssertionError("stop after the decision is built")

        with self.assertRaises(AssertionError):
            run_writer_canaries("candidate", attempts=1)

        self.assertEqual(
            "patch_api_writer", execute.call_args.args[1].execution_profile
        )


if __name__ == "__main__":
    unittest.main()
