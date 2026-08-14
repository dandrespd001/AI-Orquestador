"""El escritor nativo acotado: puede ejecutar, no puede auto-aplicar.

`native_writer` empaquetaba dos privilegios distintos: editar el arbol de
trabajo directamente (que es lo que permite iterar, hacer fase roja y tocar
ficheros grandes) y auto-aplicar el resultado al repo de origen sin revision.
Un modelo externo necesita el primero y no ha ganado el segundo.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_orchestrator.adapters import _command, _run_once, execute
from ai_orchestrator.models import ALLOWED_EXECUTION_PROFILES, RouteDecision, TaskSpec
from ai_orchestrator.supervisor import ProcessOutcome
from ai_orchestrator.writer import WorktreeTransaction, WriterError


class WorktreeCase(unittest.TestCase):
    """Repo git minimo con un fichero rastreado, uno ignorado y un prompt."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name) / "repo"
        self.repo.mkdir()
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        for key, value in (("user.email", "test@example.invalid"), ("user.name", "Test")):
            subprocess.run(["git", "-C", str(self.repo), "config", key, value], check=True)
        (self.repo / "src").mkdir()
        (self.repo / "src" / "value.txt").write_text("old\n", encoding="utf-8")
        (self.repo / "prompt.txt").write_text("change value", encoding="utf-8")
        (self.repo / ".gitignore").write_text("build/\n__pycache__/\n", encoding="utf-8")
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

    def decision(self, task: TaskSpec, profile: str) -> RouteDecision:
        return RouteDecision(
            schema_version=1,
            decision_id="d1",
            task_id=task.task_id,
            profile="default",
            status="READY",
            selected_model="deepseek-v4-pro",
            requested_route_model="ds/deepseek-v4-pro",
            provider="deepseek",
            family="deepseek",
            transport="qwen_omniroute_cli",
            reviewer_model="claude-opus",
            alternatives=(),
            unavailable={},
            auto_execute=True,
            approval_required=False,
            reason="eligible",
            context_sha256=task.context_sha256(),
            write_mode=task.write_mode,
            trust_tier="official_subscription",
            execution_profile=profile,
        )


class NativePatchWriterTests(WorktreeCase):
    def test_native_patch_writer_is_an_accepted_execution_profile(self) -> None:
        self.assertIn("native_patch_writer", ALLOWED_EXECUTION_PROFILES)

    def test_native_patch_writer_edits_instead_of_returning_a_diff(self) -> None:
        task = self.task(write_mode="validated_patch")
        command = _command(task, self.decision(task, "native_patch_writer"), task.prompt_path())
        self.assertIn("auto-edit", command)
        self.assertNotIn("plan", command)

    @patch("ai_orchestrator.adapters.run_supervised")
    @patch("ai_orchestrator.adapters._qwen_sandbox_command")
    def test_native_patch_writer_receives_a_writable_worktree(self, sandbox, run) -> None:
        sandbox.return_value = (["/bin/true"], {})
        run.return_value = ProcessOutcome(0, b"[]", b"")
        task = self.task(write_mode="validated_patch")
        _run_once(
            task,
            self.decision(task, "native_patch_writer"),
            prompt=b"go",
            cwd=self.repo,
        )
        self.assertTrue(sandbox.call_args.kwargs["writable"])

    @patch("ai_orchestrator.adapters.run_supervised")
    @patch("ai_orchestrator.adapters._qwen_sandbox_command")
    def test_read_only_task_never_receives_a_writable_worktree(self, sandbox, run) -> None:
        sandbox.return_value = (["/bin/true"], {})
        run.return_value = ProcessOutcome(0, b"[]", b"")
        task = self.task(write_mode="read_only", allowed_write_paths=[])
        _run_once(
            task,
            self.decision(task, "native_patch_writer"),
            prompt=b"go",
            cwd=self.repo,
        )
        self.assertFalse(sandbox.call_args.kwargs["writable"])

    @patch("ai_orchestrator.adapters._run_once")
    def test_native_patch_writer_never_auto_applies_to_source(self, run_once) -> None:
        task = self.task()

        # Un escritor nativo edita el arbol y devuelve prosa, no un diff: la
        # envoltura es la que emite el CLI de Qwen en modo auto-edit.
        envelope = json.dumps(
            [
                {"type": "system", "model": "ds/deepseek-v4-pro"},
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "result": "He actualizado src/value.txt.",
                },
            ]
        ).encode()

        def edit(_task, _decision, *, prompt, cwd):
            (cwd / "src" / "value.txt").write_text("new\n", encoding="utf-8")
            return ProcessOutcome(0, envelope, b""), "ds/deepseek-v4-pro", True

        run_once.side_effect = edit
        result, patch_bytes = execute(
            task,
            self.decision(task, "native_patch_writer"),
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


class IgnoredFileScopeTests(WorktreeCase):
    """Un arbol escribible vuelve viva una brecha que antes era teorica.

    `collect_patch` listaba los no rastreados con `--exclude-standard`, asi que
    un fichero cubierto por .gitignore no pasaba por la allowlist. Nunca llega
    al diff, pero si vive en el arbol donde corren los gates, y un gate ejecuta
    el codigo del modelo. La regla no puede ser «no hay ficheros ignorados»
    —los gates crean build/ y __pycache__ y el bucle de reparacion moriria—
    sino «el escritor no creo ficheros ignorados».
    """

    def test_ignored_file_created_by_the_writer_is_rejected(self) -> None:
        task = self.task(write_mode="validated_patch")
        with WorktreeTransaction(task) as transaction:
            assert transaction.worktree is not None
            transaction.snapshot_ignored()
            (transaction.worktree / "src" / "value.txt").write_text("new\n", encoding="utf-8")
            (transaction.worktree / "build").mkdir()
            (transaction.worktree / "build" / "sitecustomize.py").write_text(
                "raise SystemExit(0)\n", encoding="utf-8"
            )
            with self.assertRaises(WriterError) as context:
                transaction.collect_patch((".git",))
        self.assertEqual("WRITE_SCOPE_VIOLATION", context.exception.code)

    def test_ignored_file_left_by_a_gate_does_not_break_the_repair_loop(self) -> None:
        task = self.task(write_mode="validated_patch")
        with WorktreeTransaction(task) as transaction:
            assert transaction.worktree is not None
            # Lo que deja un gate de la vuelta anterior, antes de que el
            # escritor vuelva a correr.
            (transaction.worktree / "__pycache__").mkdir()
            (transaction.worktree / "__pycache__" / "m.pyc").write_bytes(b"\x00")
            transaction.snapshot_ignored()
            (transaction.worktree / "src" / "value.txt").write_text("new\n", encoding="utf-8")
            outcome = transaction.collect_patch((".git",))
        self.assertEqual(("src/value.txt",), outcome.paths)
        self.assertIn(b"+new", outcome.patch)


if __name__ == "__main__":
    unittest.main()
