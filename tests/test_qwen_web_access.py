from __future__ import annotations

import unittest
from unittest.mock import Mock

from ai_orchestrator.adapters import _qwen_approval_mode, _qwen_excluded_tools


def _decision(web_access: bool) -> Mock:
    decision = Mock()
    decision.web_access = web_access
    return decision


class ExcludedToolsTests(unittest.TestCase):
    """La contencion se declara por nombre, no se delega a un clasificador."""

    def test_without_web_the_web_tools_are_excluded(self) -> None:
        excluded = _qwen_excluded_tools(_decision(False), writing=False).split(",")

        self.assertIn("web_fetch", excluded)
        self.assertIn("web_search", excluded)
        self.assertIn("run_shell_command", excluded)

    def test_reading_with_web_keeps_write_tools_out(self) -> None:
        """`yolo` registra edit/write_file/notebook_edit; se retiran por nombre.

        Sin esto, ganar la web costaba perder una capa de contencion: medido el
        2026-08-25, el registro pasaba de 21 herramientas a 24.
        """

        excluded = _qwen_excluded_tools(_decision(True), writing=False).split(",")

        self.assertNotIn("web_fetch", excluded)
        for tool in ("edit", "write_file", "notebook_edit"):
            self.assertIn(tool, excluded)
        for tool in ("run_shell_command", "agent", "skill"):
            self.assertIn(tool, excluded)

    def test_a_writer_never_loses_its_edit_tools(self) -> None:
        excluded = _qwen_excluded_tools(_decision(True), writing=True).split(",")

        for tool in ("edit", "write_file", "notebook_edit"):
            self.assertNotIn(tool, excluded)
        self.assertIn("run_shell_command", excluded)


class ApprovalModeTests(unittest.TestCase):
    def test_reading_without_web_stays_in_plan(self) -> None:
        self.assertEqual(
            "plan", _qwen_approval_mode(_decision(False), writing=False, guarded=False)
        )

    def test_reading_with_web_is_deterministic_not_classified(self) -> None:
        """Ni `plan` (deniega la web) ni `auto` (moneda al aire): `yolo`.

        Medido con dos canarios seguidos, `auto` dio `auto_accept: 1` en uno y
        `auto_accept: 0` en el siguiente para la MISMA clase de llamada.
        """

        self.assertEqual(
            "yolo", _qwen_approval_mode(_decision(True), writing=False, guarded=False)
        )

    def test_writers_are_unaffected_by_web_access(self) -> None:
        self.assertEqual(
            "auto-edit", _qwen_approval_mode(_decision(True), writing=True, guarded=False)
        )
        self.assertEqual(
            "plan", _qwen_approval_mode(_decision(True), writing=True, guarded=True)
        )


if __name__ == "__main__":
    unittest.main()
