"""Un diff que git rechaza es reparable, no terminal.

El bucle reintentaba un fallo de gates y no reintentaba un fallo de sintaxis
del diff, que es el mas facil de corregir: basta devolverle al modelo la queja
de git. Una tarea real contra gemini-free-fast moria en GIT_ERROR con
changed_files=0 y repairs=0 teniendo max_repairs=1.

Reintentar es seguro porque ``apply_candidate_patch`` ejecuta ``git apply
--check`` antes de aplicar: un parche corrupto deja el worktree intacto.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_orchestrator.adapters import execute
from ai_orchestrator.models import RouteDecision, TaskSpec
from ai_orchestrator.supervisor import ProcessOutcome
from ai_orchestrator.writer import patch_repair_prompt

GOOD_PATCH = """diff --git a/src/value.txt b/src/value.txt
--- a/src/value.txt
+++ b/src/value.txt
@@ -1 +1 @@
-old
+new
"""


def _envelope(text: str) -> bytes:
    return json.dumps(
        [
            {"type": "system", "model": "antigravity/gemini-3.6-flash-low"},
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": text,
            },
        ]
    ).encode()


class PatchRepairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name) / "repo"
        self.repo.mkdir()
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        for key, value in (("user.email", "t@e.invalid"), ("user.name", "T")):
            subprocess.run(
                ["git", "-C", str(self.repo), "config", key, value], check=True
            )
        (self.repo / "src").mkdir()
        (self.repo / "src" / "value.txt").write_text("old\n", encoding="utf-8")
        (self.repo / "prompt.txt").write_text("change value", encoding="utf-8")
        subprocess.run(["git", "-C", str(self.repo), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "commit", "-qm", "initial"], check=True
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _task(self, **overrides: object) -> TaskSpec:
        payload: dict[str, object] = {
            "repo": str(self.repo),
            "task_type": "tests",
            "sensitivity": "internal",
            "prompt_file": "prompt.txt",
            "write_mode": "validated_patch",
            "allowed_write_paths": ["src"],
            "gate_profile": "profile",
        }
        payload.update(overrides)
        return TaskSpec.from_mapping(payload)

    def _decision(self, task: TaskSpec) -> RouteDecision:
        return RouteDecision(
            schema_version=1,
            decision_id="d1",
            task_id=task.task_id,
            profile="default",
            status="READY",
            selected_model="gemini-free-fast",
            requested_route_model="antigravity/gemini-3.6-flash-low",
            provider="antigravity",
            family="google-gemini",
            transport="qwen_omniroute_cli",
            reviewer_model="gpt-5.6-sol",
            alternatives=(),
            unavailable={},
            auto_execute=True,
            approval_required=False,
            reason="eligible",
            context_sha256=task.context_sha256(),
            write_mode=task.write_mode,
            trust_tier="web_free",
            execution_profile="patch_api_writer",
        )

    def test_a_corrupt_diff_is_retried_and_the_second_answer_lands(self) -> None:
        task = self._task(max_repairs=2)
        answers = [
            _envelope("Sure! Here is the patch:\n```diff\nnot a real hunk\n```"),
            _envelope(GOOD_PATCH),
        ]

        def fake_run(*args, **kwargs):
            return (
                ProcessOutcome(
                    returncode=0, stdout=answers.pop(0), stderr=b"", timed_out=False
                ),
                "gemini-3.6-flash-low",
                True,
            )

        with patch("ai_orchestrator.adapters._run_once", side_effect=fake_run):
            result, _ = execute(task, self._decision(task))

        self.assertEqual("PATCH_READY", result.status)
        self.assertEqual(1, result.repairs)
        self.assertEqual(1, result.changed_files)
        self.assertEqual([], answers)

    def test_the_repair_budget_is_respected(self) -> None:
        task = self._task(max_repairs=1)
        broken = _envelope("```diff\nnot a real hunk\n```")

        def always_broken(*args, **kwargs):
            return (
                ProcessOutcome(
                    returncode=0, stdout=broken, stderr=b"", timed_out=False
                ),
                "gemini-3.6-flash-low",
                True,
            )

        with patch("ai_orchestrator.adapters._run_once", side_effect=always_broken):
            result, _ = execute(task, self._decision(task))

        # Una reparacion, y a la segunda se rinde con el codigo real de git.
        self.assertEqual(1, result.repairs)
        self.assertIn(result.status, {"GIT_ERROR", "MALFORMED_PATCH"})

    def test_a_zero_budget_still_fails_on_the_first_rejection(self) -> None:
        task = self._task(max_repairs=0)
        broken = _envelope("```diff\nnot a real hunk\n```")

        def always_broken(*args, **kwargs):
            return (
                ProcessOutcome(
                    returncode=0, stdout=broken, stderr=b"", timed_out=False
                ),
                "gemini-3.6-flash-low",
                True,
            )

        with patch("ai_orchestrator.adapters._run_once", side_effect=always_broken):
            result, _ = execute(task, self._decision(task))

        self.assertEqual(0, result.repairs)


class PatchRepairPromptTests(unittest.TestCase):
    def test_the_prompt_carries_gits_complaint_and_asks_for_a_bare_diff(self) -> None:
        rendered = patch_repair_prompt(b"original brief", "corrupt patch at line 23")
        self.assertIn(b"original brief", rendered)
        self.assertIn(b"corrupt patch at line 23", rendered)
        self.assertIn(b"no markdown", rendered)


if __name__ == "__main__":
    unittest.main()
