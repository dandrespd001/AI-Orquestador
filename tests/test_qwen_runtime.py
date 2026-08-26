from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from ai_orchestrator.adapters import _qwen_runtime


class QwenRuntimeTests(unittest.TestCase):
    """El punto de entrada se deriva del ejecutable, nunca se cablea.

    Qwen Code 0.22.1 movio el binario de `bin/qwen` a
    `node_modules/@qwen-code/qwen-code/cli-entry.js`. El sandbox seguia
    ejecutando la ruta antigua y bwrap moria con `execvp ...: No such file or
    directory`: returncode 1, cero bytes, dos segundos. Por fuera es identico a
    un proveedor caido, y ningun test lo veia porque todos mockean
    `_qwen_sandbox_command`.
    """

    def _install(self, root: Path, relative: str) -> Path:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("#!/usr/bin/env node\n", encoding="utf-8")
        target.chmod(0o755)
        return target

    def test_layout_0_22_with_node_modules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "qwen-code-0.22.1"
            relative = "node_modules/@qwen-code/qwen-code/cli-entry.js"
            target = self._install(root, relative)
            launcher = Path(directory) / "qwen"
            launcher.symlink_to(target)

            resolved_root, entrypoint = _qwen_runtime(launcher)

            self.assertEqual(root.resolve(), resolved_root)
            self.assertEqual("/opt/qwen/" + relative, entrypoint)

    def test_legacy_layout_still_resolves(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "qwen-code-0.21.15"
            target = self._install(root, "bin/qwen")
            launcher = Path(directory) / "qwen"
            launcher.symlink_to(target)

            resolved_root, entrypoint = _qwen_runtime(launcher)

            self.assertEqual(root.resolve(), resolved_root)
            self.assertEqual("/opt/qwen/bin/qwen", entrypoint)

    def test_entrypoint_exists_inside_the_mounted_root(self) -> None:
        """Lo que se monta y lo que se ejecuta tienen que ser el mismo fichero."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "qwen-code-9.9.9"
            relative = "node_modules/.pnpm/qwen/lib/entry.js"
            self._install(root, relative)
            launcher = Path(directory) / "qwen"
            launcher.symlink_to(root / relative)

            resolved_root, entrypoint = _qwen_runtime(launcher)

            inside = resolved_root / entrypoint.removeprefix("/opt/qwen/")
            self.assertTrue(inside.is_file(), f"{inside} no existe dentro del montaje")

    def test_real_installation_resolves_to_a_real_file(self) -> None:
        """Contra la instalacion de la maquina, no contra un doble."""

        import shutil

        qwen = shutil.which("qwen")
        if not qwen:
            self.skipTest("qwen no esta instalado")
        root, entrypoint = _qwen_runtime(Path(qwen))
        inside = root / entrypoint.removeprefix("/opt/qwen/")
        self.assertTrue(inside.is_file(), f"{inside} no existe")
        self.assertTrue(os.access(inside, os.X_OK), f"{inside} no es ejecutable")


if __name__ == "__main__":
    unittest.main()
