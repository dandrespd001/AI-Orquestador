from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from ai_orchestrator.adapters import execute
from ai_orchestrator.models import RouteDecision, TaskSpec
from ai_orchestrator.supervisor import ProcessOutcome
from ai_orchestrator.writer import WorktreeTransaction, WriterError, run_gates


class WriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name) / "repo"
        self.repo.mkdir()
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(["git", "-C", str(self.repo), "config", "user.email", "test@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "config", "user.name", "Test"], check=True)
        (self.repo / "src").mkdir()
        (self.repo / "src" / "value.txt").write_text("old\n", encoding="utf-8")
        (self.repo / "prompt.txt").write_text("change value", encoding="utf-8")
        subprocess.run(["git", "-C", str(self.repo), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.repo), "commit", "-qm", "initial"], check=True)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def task(self, **overrides: object) -> TaskSpec:
        payload: dict[str, object] = {
            "repo": str(self.repo),
            "task_type": "implementation",
            "sensitivity": "internal",
            "prompt_file": "prompt.txt",
            "write_mode": "validated_apply",
            "allowed_write_paths": ["src"],
            "gate_profile": "profile",
        }
        payload.update(overrides)
        return TaskSpec.from_mapping(payload)

    def decision(self, task: TaskSpec) -> RouteDecision:
        return RouteDecision(
            schema_version=1,
            decision_id="d1",
            task_id=task.task_id,
            profile="default",
            status="READY",
            selected_model="gpt-5.6-sol",
            requested_route_model="gpt-5.6-sol",
            provider="openai",
            family="openai-gpt",
            transport="codex_cli",
            reviewer_model="claude-opus",
            alternatives=(),
            unavailable={},
            auto_execute=True,
            approval_required=False,
            reason="eligible",
            context_sha256=task.context_sha256(),
            write_mode=task.write_mode,
            trust_tier="official_subscription",
            execution_profile="native_writer",
        )

    @patch("ai_orchestrator.adapters._run_once")
    def test_validated_apply_edits_only_after_gate(self, run_once) -> None:
        task = self.task()

        def edit(_task, _decision, *, prompt, cwd):
            (cwd / "src" / "value.txt").write_text("new\n", encoding="utf-8")
            return ProcessOutcome(0, b"done", b""), "gpt-5.6-sol", True

        run_once.side_effect = edit
        result, patch_bytes = execute(
            task,
            self.decision(task),
            gate_commands=(
                "python3 -c \"assert open('src/value.txt').read() == 'new\\n'\"",
            ),
            protected_prefixes=(".git", ".env"),
        )
        self.assertEqual("APPLIED", result.status)
        self.assertTrue(result.gates_passed)
        self.assertEqual(1, result.changed_files)
        self.assertIn(b"+new", patch_bytes)
        self.assertEqual("new\n", (self.repo / "src" / "value.txt").read_text())
        status = subprocess.run(
            ["git", "-C", str(self.repo), "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        self.assertEqual(" M src/value.txt\n", status)

    def test_scope_escape_is_rejected(self) -> None:
        task = self.task(write_mode="validated_patch")
        with WorktreeTransaction(task) as transaction:
            assert transaction.worktree is not None
            (transaction.worktree / "outside.txt").write_text("no\n", encoding="utf-8")
            with self.assertRaises(WriterError) as context:
                transaction.collect_patch((".git",))
        self.assertEqual("WRITE_SCOPE_VIOLATION", context.exception.code)

    def test_dirty_source_never_auto_applies(self) -> None:
        (self.repo / "unrelated.txt").write_text("dirty\n", encoding="utf-8")
        task = self.task()
        with WorktreeTransaction(task) as transaction:
            assert transaction.worktree is not None
            (transaction.worktree / "src" / "value.txt").write_text("new\n", encoding="utf-8")
            patch_outcome = transaction.collect_patch((".git",))
            self.assertFalse(transaction.can_apply())
            self.assertTrue(patch_outcome.patch)

    @patch("ai_orchestrator.adapters._run_once")
    def test_guarded_qwen_patch_requires_review_after_gate(self, run_once) -> None:
        task = self.task()
        decision = RouteDecision(
            schema_version=1,
            decision_id="d2",
            task_id=task.task_id,
            profile="default",
            status="READY",
            selected_model="deepseek-v4-flash",
            requested_route_model="ds/deepseek-v4-flash",
            provider="deepseek",
            family="deepseek",
            transport="qwen_omniroute_cli",
            reviewer_model="gpt-5.6-sol",
            alternatives=(),
            unavailable={},
            auto_execute=True,
            approval_required=False,
            reason="eligible",
            context_sha256=task.context_sha256(),
            write_mode=task.write_mode,
            trust_tier="official_subscription",
            execution_profile="patch_api_writer",
        )
        patch_text = (
            "diff --git a/src/value.txt b/src/value.txt\n"
            "index 3367afd..3e75765 100644\n"
            "--- a/src/value.txt\n"
            "+++ b/src/value.txt\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
        envelope = json.dumps(
            [
                {"type": "system", "model": "ds/deepseek-v4-flash"},
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "result": patch_text,
                },
            ]
        ).encode()
        run_once.return_value = (
            ProcessOutcome(0, envelope, b""),
            "ds/deepseek-v4-flash",
            True,
        )

        result, patch_bytes = execute(
            task,
            decision,
            gate_commands=(
                "python3 -c \"assert open('src/value.txt').read() == 'new\\n'\"",
            ),
            protected_prefixes=(".git", ".env"),
        )

        self.assertEqual("PATCH_READY", result.status)
        self.assertEqual("EXTERNAL_WRITER_REVIEW_REQUIRED", result.error_code)
        self.assertTrue(result.gates_passed)
        self.assertIn(b"+new", patch_bytes)
        self.assertEqual("old\n", (self.repo / "src" / "value.txt").read_text())

    @patch("ai_orchestrator.adapters._run_once")
    def test_kimi_writer_requires_review_after_gate(self, run_once) -> None:
        task = self.task()
        decision = replace(
            self.decision(task),
            selected_model="kimi-official",
            requested_route_model="kimi-code/k3",
            provider="moonshot",
            family="moonshot-kimi",
            transport="kimi_cli",
            execution_profile="kimi_acp_writer",
        )

        def edit(_task, _decision, *, prompt, cwd):
            (cwd / "src" / "value.txt").write_text("new\n", encoding="utf-8")
            return ProcessOutcome(0, b"done", b""), "kimi-code/k3", True

        run_once.side_effect = edit
        result, patch_bytes = execute(
            task,
            decision,
            gate_commands=(
                "python3 -c \"assert open('src/value.txt').read() == 'new\\n'\"",
            ),
            protected_prefixes=(".git", ".env"),
        )

        self.assertEqual("PATCH_READY", result.status)
        self.assertEqual("EXTERNAL_WRITER_REVIEW_REQUIRED", result.error_code)
        self.assertTrue(result.gates_passed)
        self.assertIn(b"+new", patch_bytes)
        self.assertEqual("old\n", (self.repo / "src" / "value.txt").read_text())

    @patch("ai_orchestrator.writer.run_supervised")
    def test_named_gate_timeout_is_not_silently_capped(self, run) -> None:
        run.return_value = ProcessOutcome(0, b"ok", b"")
        outcomes = run_gates(self.repo, ("true",), 1200)
        self.assertEqual(1, len(outcomes))
        self.assertEqual(1200, run.call_args.kwargs["timeout_seconds"])

    def test_gates_cannot_see_the_operator_home(self) -> None:
        """A gate runs the model's code, so it must not reach the real HOME.

        `cmake`, `ctest` and `pytest` all execute what the model just wrote. Run
        unsandboxed they can read `~/.omniroute/.env`, `~/.ssh` and
        `~/.config/ai-orchestrator/credentials`, and `repair_prompt` hands the
        gate output back to the model, which closes the loop into an
        exfiltration channel.

        This runs for real, not against a mock: mocking `run_supervised` here
        would assert the argv and prove nothing about the isolation.
        """
        home = Path.home()
        # Not "the home directory is absent": `~/.local/lib` is deliberately
        # re-exposed read-only so toolchain gates keep their Python packages.
        # What must be unreachable is the material an exfiltration would want.
        for sensitive in (".ssh", ".omniroute", ".config/ai-orchestrator", ".codex"):
            outcomes = run_gates(self.repo, (f"test -e {home / sensitive}",), 60)
            self.assertEqual(1, len(outcomes))
            self.assertNotEqual(
                0,
                outcomes[0].returncode,
                f"the gate could see {home / sensitive}",
            )

    def test_gates_still_run_and_can_write_in_the_worktree(self) -> None:
        """The isolation must not be bought by breaking the gates.

        A sandbox that makes every gate fail would look secure and be useless,
        so this pins the other half: the worktree stays writable and a normal
        command still succeeds inside it.
        """
        outcomes = run_gates(self.repo, ("touch gate-probe.txt",), 60)
        self.assertEqual(0, outcomes[0].returncode, outcomes[0].output)
        self.assertTrue((self.repo / "gate-probe.txt").exists())


if __name__ == "__main__":
    unittest.main()
