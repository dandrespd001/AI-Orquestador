"""Abacus enrutado: el contexto viaja en el prompt, porque el modelo no ve el repo.

El CLI de Abacus corre en un directorio vacio desechable con
--disallowed-tools "*", asi que nunca ve el arbol de trabajo. Por eso
models.toml lo mantenia fuera de roles.toml, con un comentario explicito:
enrutarlo sin mas producia HANDOFF_REQUIRED. Al meterlo en los roles, siete
tareas de solo lectura empezaron a fallar por ahi.

La via es inlinear el contexto en el prompt. Lo que estos tests fijan es que el
inlineado respeta el alcance del repositorio, acota el tamano y no finge leer
un binario.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_orchestrator.adapters import (
    MAX_ABACUS_CONTEXT_BYTES,
    AdapterError,
    _abacus_context,
    _run_abacus,
)
from ai_orchestrator.models import ContractError, RouteDecision, TaskSpec


class _Repo(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name)
        (self.repo / "src").mkdir()
        (self.repo / "src" / "calc.py").write_text(
            "def suma(): ...\n", encoding="utf-8"
        )
        (self.repo / "src" / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe")
        (self.repo / "prompt.txt").write_text("revisa", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _task(self, **overrides: object) -> TaskSpec:
        payload: dict[str, object] = {
            "repo": str(self.repo),
            "task_type": "review",
            "sensitivity": "internal",
            "prompt_file": "prompt.txt",
            "write_mode": "read_only",
        }
        payload.update(overrides)
        return TaskSpec.from_mapping(payload)

    def _decision(self, route_model: str = "Grok 4.6") -> RouteDecision:
        return RouteDecision(
            schema_version=1,
            decision_id="d",
            task_id="t",
            profile="default",
            status="READY",
            selected_model="abacus-grok-4.6",
            requested_route_model=route_model,
            provider="abacus",
            family="grok",
            transport="abacus_cli",
            reviewer_model=None,
            alternatives=(),
            unavailable={},
            auto_execute=True,
            approval_required=False,
            reason="eligible",
            context_sha256="0" * 64,
            write_mode="read_only",
            execution_profile="read_only",
        )


class ContextInliningTests(_Repo):
    def test_declared_inputs_reach_the_prompt(self) -> None:
        rendered = _abacus_context(self._task(input_paths=["src/calc.py"]), b"revisa")
        self.assertIn("revisa", rendered)
        self.assertIn("def suma()", rendered)
        self.assertIn("src/calc.py", rendered)

    def test_a_binary_is_named_but_never_inlined(self) -> None:
        rendered = _abacus_context(self._task(input_paths=["src/logo.png"]), b"mira")
        self.assertIn("binary, not inlined", rendered)
        self.assertNotIn("PNG", rendered.replace("src/logo.png", ""))

    def test_escaping_the_repository_is_refused_before_the_context_is_built(
        self,
    ) -> None:
        # La guarda esta mas arriba de lo que parece: TaskSpec rechaza la ruta
        # al construirse, asi que el inlineado nunca llega a ver un fichero de
        # fuera. El helper conserva su propia comprobacion como cinturon.
        with self.assertRaises(ContractError):
            self._task(input_paths=["../../etc/passwd"])

    def test_a_declared_input_must_exist(self) -> None:
        with self.assertRaises(ContractError):
            self._task(input_paths=["src/absent.py"])

    def test_the_helper_still_skips_a_file_that_vanished(self) -> None:
        # Entre la construccion de la tarea y la ejecucion puede desaparecer.
        task = self._task(input_paths=["src/calc.py"])
        (self.repo / "src" / "calc.py").unlink()
        self.assertIn("revisa", _abacus_context(task, b"revisa"))

    def test_the_context_is_bounded(self) -> None:
        big = self.repo / "src" / "big.py"
        big.write_text("x = 1\n" * 40_000, encoding="utf-8")
        rendered = _abacus_context(self._task(input_paths=["src/big.py"]), b"revisa")
        self.assertLess(len(rendered), MAX_ABACUS_CONTEXT_BYTES + 4_000)
        self.assertIn("truncated", rendered)


class RoutedExecutionTests(_Repo):
    def test_a_successful_delegation_reports_provenance(self) -> None:
        class _Delegator:
            def delegate(self, **kwargs: object) -> dict[str, object]:
                return {"model": "Grok 4.6", "result": "sin problemas"}

        with patch("ai_orchestrator.adapters.AbacusDelegator", _Delegator):
            outcome, reported, provenance = _run_abacus(
                self._task(input_paths=["src/calc.py"]),
                self._decision(),
                prompt=b"revisa",
                cwd=self.repo,
            )
        self.assertEqual(0, outcome.returncode)
        self.assertEqual(b"sin problemas", outcome.stdout)
        self.assertEqual("Grok 4.6", reported)
        self.assertTrue(provenance)

    def test_a_different_model_answering_breaks_provenance(self) -> None:
        class _Delegator:
            def delegate(self, **kwargs: object) -> dict[str, object]:
                return {"model": "Qwen3.8 Max", "result": "hecho"}

        with patch("ai_orchestrator.adapters.AbacusDelegator", _Delegator):
            _, reported, provenance = _run_abacus(
                self._task(), self._decision(), prompt=b"revisa", cwd=self.repo
            )
        self.assertEqual("Qwen3.8 Max", reported)
        self.assertFalse(provenance)

    def test_a_failing_delegation_becomes_a_failed_outcome(self) -> None:
        class _Delegator:
            def delegate(self, **kwargs: object) -> dict[str, object]:
                raise RuntimeError("Abacus CLI timed out")

        with patch("ai_orchestrator.adapters.AbacusDelegator", _Delegator):
            outcome, reported, provenance = _run_abacus(
                self._task(), self._decision(), prompt=b"revisa", cwd=self.repo
            )
        self.assertEqual(1, outcome.returncode)
        self.assertIn(b"timed out", outcome.stderr)
        self.assertIsNone(reported)
        self.assertFalse(provenance)

    def test_a_write_task_is_refused_even_if_it_reaches_here(self) -> None:
        task = self._task(write_mode="validated_patch", allowed_write_paths=["src"])
        with self.assertRaises(AdapterError):
            _run_abacus(task, self._decision(), prompt=b"x", cwd=self.repo)


if __name__ == "__main__":
    unittest.main()
