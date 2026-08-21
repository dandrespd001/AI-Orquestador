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

from ai_orchestrator.adapters import (
    _command,
    _qwen_error,
    _run_once,
    _writer_prompt,
    execute,
)
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
        for key, value in (
            ("user.email", "test@example.invalid"),
            ("user.name", "Test"),
        ):
            subprocess.run(
                ["git", "-C", str(self.repo), "config", key, value], check=True
            )
        (self.repo / "src").mkdir()
        (self.repo / "src" / "value.txt").write_text("old\n", encoding="utf-8")
        (self.repo / "prompt.txt").write_text("change value", encoding="utf-8")
        (self.repo / ".gitignore").write_text(
            "build/\n__pycache__/\n", encoding="utf-8"
        )
        subprocess.run(["git", "-C", str(self.repo), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "commit", "-qm", "initial"], check=True
        )

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

    def envelope(self) -> bytes:
        return json.dumps(
            [
                {"type": "system", "model": "ds/deepseek-v4-pro"},
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "result": "hecho",
                },
            ]
        ).encode()

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
        command = _command(
            task, self.decision(task, "native_patch_writer"), task.prompt_path()
        )
        self.assertIn("auto-edit", command)
        self.assertNotIn("plan", command)

    @patch("ai_orchestrator.adapters.run_supervised")
    @patch("ai_orchestrator.adapters._qwen_sandbox_command")
    def test_native_patch_writer_receives_a_writable_worktree(
        self, sandbox, run
    ) -> None:
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
    def test_read_only_task_never_receives_a_writable_worktree(
        self, sandbox, run
    ) -> None:
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
            (transaction.worktree / "src" / "value.txt").write_text(
                "new\n", encoding="utf-8"
            )
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
            (transaction.worktree / "src" / "value.txt").write_text(
                "new\n", encoding="utf-8"
            )
            outcome = transaction.collect_patch((".git",))
        self.assertEqual(("src/value.txt",), outcome.paths)
        self.assertIn(b"+new", outcome.patch)


class TruncatedOutputTests(unittest.TestCase):
    """Un transcrito cortado no es una salida malformada.

    Medido el 2026-08-14 delegando `step.hpp` a ds/deepseek-v4-pro: el CLI de
    Qwen devolvio **65536 bytes exactos**, cortados a mitad de una cadena JSON.
    `run_supervised` usa `communicate()` sin tope y no hay ningun 65536 en este
    codigo, asi que el corte es del CLI. Se reportaba `MALFORMED_OUTPUT`, que
    manda a buscar el fallo en el sitio equivocado: no lo emitio mal, se quedo
    sin sitio. Diagnosticarlo costo contar los bytes a mano.
    """

    def test_a_cut_transcript_is_reported_as_truncated(self) -> None:
        cut = b'[{"type":"system","model":"ds/deepseek-v4-pro"},{"type":"assistant","text":"### Edit 1: N'
        self.assertEqual("OUTPUT_TRUNCATED", _qwen_error(cut, b""))

    def test_genuine_garbage_is_still_malformed(self) -> None:
        """El diagnostico nuevo no puede tragarse al viejo."""

        self.assertEqual(
            "MALFORMED_OUTPUT", _qwen_error(b"no soy json en absoluto", b"")
        )
        self.assertEqual("MALFORMED_OUTPUT", _qwen_error(b'{"type":"result"}', b""))

    def test_native_writer_is_told_to_use_its_editing_tool(self) -> None:
        """El modelo tenia `edit` y narro las ediciones en prosa.

        13 `read_file`, 2 `grep_search`, **cero** `edit`, y luego «### Edit 1:
        Old: ... New: ...». La politica decia solo «Implement the requested
        change directly» y nunca que la prosa no es entregable.
        """

        task = TaskSpec.from_mapping(
            {
                "repo": str(Path(__file__).resolve().parents[1]),
                "task_type": "implementation",
                "sensitivity": "internal",
                "write_mode": "validated_patch",
                "allowed_write_paths": ["src"],
            }
        )
        policy = _writer_prompt(task, b"cuerpo", patch_only=False).decode("utf-8")
        self.assertIn("edit", policy)
        self.assertIn("prose", policy.lower())
        patch_policy = _writer_prompt(task, b"cuerpo", patch_only=True).decode("utf-8")
        self.assertIn("diff --git", patch_policy)


class ScopeDiagnosisTests(WorktreeCase):
    """Un `WRITE_SCOPE_VIOLATION` sin la ruta ofensora no es diagnosticable.

    `WriterError` ya lleva la ruta en el mensaje; `except WriterError` se
    quedaba con `exc.code` y lo tiraba. Medido el 2026-08-14 delegando
    `step.hpp`: la ejecucion aborto por alcance y averiguar cual fue el fichero
    era imposible sin volver a gastar una pasada entera.
    """

    @patch("ai_orchestrator.adapters._run_once")
    def test_the_offending_path_reaches_the_result(self, run_once) -> None:
        task = self.task(write_mode="validated_patch")

        def edit_outside(_task, _decision, *, prompt, cwd):
            (cwd / "fuera.txt").write_text("no\n", encoding="utf-8")
            return ProcessOutcome(0, self.envelope(), b""), "ds/deepseek-v4-pro", True

        run_once.side_effect = edit_outside
        result, _ = execute(
            task,
            self.decision(task, "native_patch_writer"),
            gate_commands=('python3 -c "pass"',),
            protected_prefixes=(".git", ".env"),
        )
        self.assertEqual("WRITE_SCOPE_VIOLATION", result.status)
        self.assertIsNotNone(result.error_detail)
        self.assertIn("fuera.txt", str(result.error_detail))

    def test_the_file_limit_names_what_was_changed(self) -> None:
        """«changed-file limit exceeded» a secas es media respuesta.

        Saber que se paso del limite no dice CUAL fichero sobra, y averiguarlo
        costaba otra pasada entera contra el proveedor.
        """

        task = self.task(write_mode="validated_patch", max_changed_files=1)
        with WorktreeTransaction(task) as transaction:
            assert transaction.worktree is not None
            transaction.snapshot_ignored()
            (transaction.worktree / "src" / "value.txt").write_text(
                "new\n", encoding="utf-8"
            )
            (transaction.worktree / "src" / "otro.txt").write_text(
                "otro\n", encoding="utf-8"
            )
            with self.assertRaises(WriterError) as context:
                transaction.collect_patch((".git",))
        message = str(context.exception)
        self.assertIn("2", message)
        self.assertIn("src/value.txt", message)
        self.assertIn("src/otro.txt", message)

    @patch("ai_orchestrator.adapters._run_once")
    def test_the_cli_own_state_is_not_counted_as_the_model_work(self, run_once) -> None:
        """`.qwen/settings.json` lo escribe el CLI, no el modelo.

        Medido el 2026-08-14 delegando `step.hpp`: los cambios fueron
        `.qwen/settings.json`, `step.hpp` y `bench_multisemilla.cpp`. Los dos
        ultimos son el encargo; el primero es estado del propio Qwen, que lo
        crea al aprobar una edicion. Contarlo agota el tope de ficheros y,
        peor, choca con `.qwen` como ruta protegida: el trabajo bueno se
        rechazaba por la basura de la herramienta.
        """

        task = self.task(write_mode="validated_patch", max_changed_files=1)

        def edit_and_litter(_task, _decision, *, prompt, cwd):
            (cwd / "src" / "value.txt").write_text("new\n", encoding="utf-8")
            (cwd / ".qwen").mkdir(exist_ok=True)
            (cwd / ".qwen" / "settings.json").write_text("{}\n", encoding="utf-8")
            return ProcessOutcome(0, self.envelope(), b""), "ds/deepseek-v4-pro", True

        run_once.side_effect = edit_and_litter
        result, patch_bytes = execute(
            task,
            self.decision(task, "native_patch_writer"),
            gate_commands=(
                "python3 -c \"assert open('src/value.txt').read() == 'new\\n'\"",
            ),
            protected_prefixes=(".git", ".qwen"),
        )
        self.assertEqual("PATCH_READY", result.status)
        self.assertEqual(1, result.changed_files)
        self.assertNotIn(b".qwen", patch_bytes)

    @patch("ai_orchestrator.adapters._run_once")
    def test_tracked_cli_state_is_restored_not_deleted(self, run_once) -> None:
        """En CHUNSA `.qwen/settings.json` esta RASTREADO: Qwen lo modifica.

        Borrarlo no lo saca del conjunto de cambios, lo convierte en un
        borrado, que git sigue contando. Hay que devolverlo a HEAD.
        """

        qwen_dir = self.repo / ".qwen"
        qwen_dir.mkdir()
        (qwen_dir / "settings.json").write_text(
            '{"original": true}\n', encoding="utf-8"
        )
        subprocess.run(["git", "-C", str(self.repo), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "commit", "-qm", "qwen"], check=True
        )

        task = self.task(write_mode="validated_patch", max_changed_files=1)

        def edit_and_touch_settings(_task, _decision, *, prompt, cwd):
            (cwd / "src" / "value.txt").write_text("new\n", encoding="utf-8")
            (cwd / ".qwen" / "settings.json").write_text(
                '{"tocado": 1}\n', encoding="utf-8"
            )
            return ProcessOutcome(0, self.envelope(), b""), "ds/deepseek-v4-pro", True

        run_once.side_effect = edit_and_touch_settings
        result, patch_bytes = execute(
            task,
            self.decision(task, "native_patch_writer"),
            gate_commands=(
                "python3 -c \"assert open('src/value.txt').read() == 'new\\n'\"",
            ),
            protected_prefixes=(".git", ".qwen"),
        )
        self.assertEqual("PATCH_READY", result.status)
        self.assertEqual(1, result.changed_files)
        self.assertNotIn(b".qwen", patch_bytes)

    def test_a_clean_run_carries_no_detail(self) -> None:
        """El campo es para fallos: en verde no debe aparecer nada."""

        with WorktreeTransaction(
            self.task(write_mode="validated_patch")
        ) as transaction:
            assert transaction.worktree is not None
            transaction.snapshot_ignored()
            (transaction.worktree / "src" / "value.txt").write_text(
                "new\n", encoding="utf-8"
            )
            outcome = transaction.collect_patch((".git",))
        self.assertTrue(outcome.patch)


class SalvageTests(WorktreeCase):
    """Para un escritor nativo el ARBOL es el entregable; la envoltura es charla.

    El bucle rompia antes de `collect_patch` en cuanto el estado del proceso no
    era SUCCESS, y `__exit__` destruia el worktree. Si el modelo habia editado,
    se tiraba sin mirarlo. Para un escritor por parche eso es correcto —el diff
    ES la envoltura—; para uno nativo, no.
    """

    @patch("ai_orchestrator.adapters._run_once")
    def test_edits_survive_an_unusable_envelope(self, run_once) -> None:
        task = self.task(write_mode="validated_patch")

        def edit_then_truncate(_task, _decision, *, prompt, cwd):
            (cwd / "src" / "value.txt").write_text("new\n", encoding="utf-8")
            return (
                ProcessOutcome(0, b'[{"type":"assistant","text":"cort', b""),
                None,
                False,
            )

        run_once.side_effect = edit_then_truncate
        result, patch_bytes = execute(
            task,
            self.decision(task, "native_patch_writer"),
            gate_commands=(
                "python3 -c \"assert open('src/value.txt').read() == 'new\\n'\"",
            ),
            protected_prefixes=(".git", ".env"),
        )
        self.assertEqual("PATCH_READY", result.status)
        self.assertEqual("WRITER_OUTPUT_UNUSABLE", result.error_code)
        self.assertTrue(result.gates_passed)
        self.assertIn(b"+new", patch_bytes)

    @patch("ai_orchestrator.adapters._run_once")
    def test_nothing_to_salvage_keeps_the_original_failure(self, run_once) -> None:
        task = self.task(write_mode="validated_patch")
        run_once.return_value = (
            ProcessOutcome(0, b'[{"type":"assistant","text":"cort', b""),
            None,
            False,
        )
        result, _ = execute(
            task,
            self.decision(task, "native_patch_writer"),
            gate_commands=('python3 -c "pass"',),
            protected_prefixes=(".git", ".env"),
        )
        self.assertEqual("OUTPUT_TRUNCATED", result.status)

    @patch("ai_orchestrator.adapters._run_once")
    def test_a_patch_writer_gets_no_salvage(self, run_once) -> None:
        """Su entregable ES la envoltura: si esta rota, no hay nada que rescatar."""

        task = self.task(write_mode="validated_patch")

        def edit_then_truncate(_task, _decision, *, prompt, cwd):
            (cwd / "src" / "value.txt").write_text("new\n", encoding="utf-8")
            return (
                ProcessOutcome(0, b'[{"type":"assistant","text":"cort', b""),
                None,
                False,
            )

        run_once.side_effect = edit_then_truncate
        result, _ = execute(
            task,
            self.decision(task, "patch_api_writer"),
            gate_commands=('python3 -c "pass"',),
            protected_prefixes=(".git", ".env"),
        )
        self.assertEqual("OUTPUT_TRUNCATED", result.status)


if __name__ == "__main__":
    unittest.main()
