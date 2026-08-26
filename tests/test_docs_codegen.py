from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ai_orchestrator.cli import main
from ai_orchestrator.docs_codegen import generate_docs, repo_root

ROOT = Path(__file__).resolve().parents[1]


class RepoRootTests(unittest.TestCase):
    """La raiz de documentos no se deriva de __file__.

    Con el paquete instalado, `Path(__file__).parents[2]` es
    `site-packages/../..` = `lib/python3.12`, que no contiene ni `docs/` ni
    `config/`. Ese cableado hacia fallar `docs check` con un ContractError que
    culpaba a `config/models.toml` en vez de a la raiz mal resuelta.
    """

    def test_explicit_override_wins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                Path(directory).resolve(), repo_root(directory)
            )

    def test_environment_is_honoured(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            previous = os.environ.get("AI_ORCHESTRATOR_REPO")
            os.environ["AI_ORCHESTRATOR_REPO"] = directory
            try:
                self.assertEqual(Path(directory).resolve(), repo_root())
            finally:
                if previous is None:
                    os.environ.pop("AI_ORCHESTRATOR_REPO", None)
                else:
                    os.environ["AI_ORCHESTRATOR_REPO"] = previous

    def test_source_tree_is_detected_by_manifest(self) -> None:
        previous = os.environ.pop("AI_ORCHESTRATOR_REPO", None)
        try:
            self.assertEqual(ROOT, repo_root())
        finally:
            if previous is not None:
                os.environ["AI_ORCHESTRATOR_REPO"] = previous

    def test_installed_layout_names_the_real_problem(self) -> None:
        """Sin manifiesto, el error dice que falta la raiz, no que falte un TOML."""

        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "site-packages" / "ai_orchestrator"
            package.parent.mkdir(parents=True)
            shutil.copytree(ROOT / "src" / "ai_orchestrator", package)
            previous_repo = os.environ.pop("AI_ORCHESTRATOR_REPO", None)
            try:
                completed = subprocess.run(
                    [sys.executable, "-m", "ai_orchestrator", "docs", "check"],
                    capture_output=True,
                    check=False,
                    cwd=directory,
                    env={**os.environ, "PYTHONPATH": str(package.parent)},
                    text=True,
                )
            finally:
                if previous_repo is not None:
                    os.environ["AI_ORCHESTRATOR_REPO"] = previous_repo
        self.assertNotEqual(0, completed.returncode)
        self.assertIn("documentation root unknown", completed.stdout)
        self.assertNotIn("models.toml", completed.stdout)


class GenerateDocsTests(unittest.TestCase):
    def test_configuration_is_read_from_config_root_not_from_root(self) -> None:
        """`root` y la configuracion son dos sitios distintos.

        Se pasa una raiz de documentos que NO contiene `config/`: si
        `generate_docs` siguiera derivando la configuracion de ella, esto
        levantaria ContractError.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "documentos"
            shutil.copytree(ROOT / "integrations", root / "integrations")
            self.assertFalse((root / "config").exists())

            payload = generate_docs(root, config_dir=ROOT / "config")

            self.assertTrue(payload["ok"])
            self.assertEqual(str(ROOT / "config"), payload["config_root"])
            self.assertTrue((root / "docs" / "generated" / "models.md").is_file())

    def test_check_reports_stale_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "documentos"
            shutil.copytree(ROOT / "integrations", root / "integrations")

            payload = generate_docs(root, check=True, config_dir=ROOT / "config")

            self.assertFalse(payload["ok"])
            self.assertIn("docs/generated/models.md", payload["stale"])
            self.assertFalse((root / "docs").exists())

    def test_cli_check_passes_against_the_versioned_configuration(self) -> None:
        self.assertEqual(0, main(["docs", "check", "--root", str(ROOT)]))


if __name__ == "__main__":
    unittest.main()
