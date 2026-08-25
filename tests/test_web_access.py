"""Mirar la web se abre por rol, y sigue cerrada en todo lo demas.

Los tres transportes de escritura bloqueaban busqueda y fetch: Claude por
--disallowedTools, Qwen por --exclude-tools y Kimi por una politica efimera.
Investigar sin poder mirar fuera del repositorio no es investigar, pero abrirlo
globalmente convertiria cualquier tarea de implementacion en una salida a
internet. El permiso viaja desde RoleConfig hasta el argv, y por defecto esta
apagado.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from ai_orchestrator.adapters import _command
from ai_orchestrator.config import load_roles
from ai_orchestrator.kimi_acp import _policy_bytes
from ai_orchestrator.models import RouteDecision, TaskSpec

ROOT = Path(__file__).resolve().parents[1]

# Los roles que el Director abrio el 2026-08-25. Cualquier ampliacion futura
# tiene que pasar por aqui a proposito.
EXPECTED_WEB_ROLES = {"research", "docs", "design"}


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="t",
        repo=".",
        task_type="research",
        sensitivity="internal",
        write_mode="validated_patch",
        allowed_write_paths=("src",),
    )


def _decision(transport: str, *, web_access: bool) -> RouteDecision:
    return RouteDecision(
        schema_version=1,
        decision_id="d",
        task_id="t",
        profile="default",
        status="READY",
        selected_model="m",
        requested_route_model="route/m",
        provider="p",
        family="f",
        transport=transport,
        reviewer_model=None,
        alternatives=(),
        unavailable={},
        auto_execute=True,
        approval_required=False,
        reason="eligible",
        context_sha256="0" * 64,
        write_mode="validated_patch",
        execution_profile="native_writer",
        web_access=web_access,
    )


class RoleDeclarationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous = os.environ.get("AI_ORCHESTRATOR_CONFIG")
        os.environ["AI_ORCHESTRATOR_CONFIG"] = str(ROOT / "config")

    def tearDown(self) -> None:
        if self.previous is None:
            os.environ.pop("AI_ORCHESTRATOR_CONFIG", None)
        else:
            os.environ["AI_ORCHESTRATOR_CONFIG"] = self.previous

    def test_only_the_intended_roles_may_reach_the_web(self) -> None:
        roles = load_roles()
        opened = {name for name, role in roles.items() if role.web_access}
        self.assertEqual(EXPECTED_WEB_ROLES, opened)

    def test_writing_roles_stay_closed(self) -> None:
        roles = load_roles()
        for name in ("implementation", "tests", "bounded_cpp", "determinism"):
            with self.subTest(role=name):
                self.assertFalse(roles[name].web_access)


class ClaudeArgvTests(unittest.TestCase):
    def test_closed_role_blocks_both_web_tools(self) -> None:
        argv = _command(
            _task(), _decision("claude_cli", web_access=False), Path("prompt.txt")
        )
        blocked = argv[argv.index("--disallowedTools") + 1]
        self.assertIn("WebFetch", blocked)
        self.assertIn("WebSearch", blocked)
        self.assertNotIn("WebFetch", argv[argv.index("--tools") + 1])

    def test_open_role_offers_them_and_still_blocks_the_shell(self) -> None:
        argv = _command(
            _task(), _decision("claude_cli", web_access=True), Path("prompt.txt")
        )
        blocked = argv[argv.index("--disallowedTools") + 1]
        offered = argv[argv.index("--tools") + 1]
        self.assertNotIn("WebFetch", blocked)
        self.assertIn("WebFetch", offered)
        self.assertIn("WebSearch", offered)
        # Abrir la lectura de la web no abre la ejecucion de comandos.
        self.assertIn("Bash", blocked)
        self.assertIn("Agent", blocked)


class QwenArgvTests(unittest.TestCase):
    def test_closed_role_excludes_the_web_tools(self) -> None:
        argv = _command(
            _task(),
            _decision("qwen_omniroute_cli", web_access=False),
            Path("prompt.txt"),
        )
        excluded = argv[argv.index("--exclude-tools") + 1]
        self.assertIn("web_fetch", excluded)
        self.assertIn("web_search", excluded)

    def test_open_role_keeps_the_shell_excluded(self) -> None:
        argv = _command(
            _task(),
            _decision("qwen_omniroute_cli", web_access=True),
            Path("prompt.txt"),
        )
        excluded = argv[argv.index("--exclude-tools") + 1]
        self.assertNotIn("web_fetch", excluded)
        self.assertNotIn("web_search", excluded)
        self.assertIn("run_shell_command", excluded)
        self.assertIn("agent", excluded)


class KimiPolicyTests(unittest.TestCase):
    def test_closed_role_disables_search_and_fetch(self) -> None:
        policy = _policy_bytes().decode()
        self.assertIn("WebSearch", policy)
        self.assertIn("FetchURL", policy)

    def test_open_role_keeps_the_dangerous_tools_disabled(self) -> None:
        policy = _policy_bytes(web_access=True).decode()
        self.assertNotIn("WebSearch", policy)
        self.assertNotIn("FetchURL", policy)
        for name in ("Bash", "Agent", "AgentSwarm", "CronCreate"):
            with self.subTest(tool=name):
                self.assertIn(name, policy)


if __name__ == "__main__":
    unittest.main()
