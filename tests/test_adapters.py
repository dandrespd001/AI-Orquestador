from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_orchestrator.adapters import AdapterError, _command, _minimal_environment, execute
from ai_orchestrator.models import RouteDecision, TaskSpec


class AdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        (self.repo / "prompt.txt").write_text("bounded task", encoding="utf-8")
        self.task = TaskSpec.from_mapping(
            {
                "repo": str(self.repo),
                "task_type": "tests",
                "sensitivity": "internal",
                "prompt_file": "prompt.txt",
            }
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def decision(self, **overrides: object) -> RouteDecision:
        values: dict[str, object] = {
            "schema_version": 1,
            "decision_id": "d1",
            "task_id": self.task.task_id,
            "profile": "default",
            "status": "READY",
            "selected_model": "gpt-5.6-sol",
            "requested_route_model": "gpt-5.6-sol",
            "provider": "openai",
            "family": "openai-gpt",
            "transport": "codex_cli",
            "reviewer_model": "claude-opus",
            "alternatives": (),
            "unavailable": {},
            "auto_execute": True,
            "approval_required": False,
            "reason": "eligible",
            "context_sha256": self.task.context_sha256(),
        }
        values.update(overrides)
        return RouteDecision(**values)

    def test_command_uses_argv_and_stdin_without_shell(self) -> None:
        command = _command(self.decision(), self.task.prompt_path())
        self.assertEqual("codex", command[0])
        self.assertIn("--model", command)
        self.assertNotIn("sh", command)

    def test_minimax_official_transport_is_read_only_auditor(self) -> None:
        decision = self.decision(
            selected_model="MiniMax-M3",
            requested_route_model="minimax/MiniMax-M3",
            provider="minimax",
            family="minimax",
            transport="minimax_official_cli",
        )
        command = _command(decision, self.task.prompt_path())
        self.assertEqual("opencode", command[0])
        self.assertIn("auditor", command)
        self.assertNotIn("delegado", command)

    def test_environment_does_not_forward_unrelated_tokens(self) -> None:
        with patch.dict(
            os.environ,
            {"BROKER_SECRET_KEY": "forbidden", "ANTHROPIC_AUTH_TOKEN": "forbidden", "HOME": str(self.root)},
            clear=True,
        ):
            environment = _minimal_environment("codex_cli")
        self.assertNotIn("BROKER_SECRET_KEY", environment)
        self.assertNotIn("ANTHROPIC_AUTH_TOKEN", environment)
        self.assertEqual(str(self.root), environment["HOME"])

    def test_handoff_is_never_executed(self) -> None:
        decision = self.decision(status="HANDOFF_REQUIRED", auto_execute=False)
        with self.assertRaises(AdapterError) as context:
            execute(self.task, decision)
        self.assertEqual("HANDOFF_REQUIRED", context.exception.code)


if __name__ == "__main__":
    unittest.main()
