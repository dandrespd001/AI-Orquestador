from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_orchestrator import config_sync
from ai_orchestrator.config_sync import apply_config_sync, config_sync_status

ROOT = Path(__file__).resolve().parents[1]


class ConfigSyncTests(unittest.TestCase):
    def test_installed_runtime_discovers_its_sibling_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            venv = root / "runtime" / "venv"
            installed_module = (
                venv
                / "lib/python3.12/site-packages/ai_orchestrator/config_sync.py"
            )
            runtime_config = root / "runtime" / "config"
            runtime_config.mkdir(parents=True)

            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(config_sync, "__file__", str(installed_module)),
                patch.object(sys, "prefix", str(venv)),
            ):
                discovered = config_sync.canonical_config_root()

            self.assertEqual(runtime_config.resolve(), discovered)

    def test_apply_is_atomic_and_preserves_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "config"
            credentials = target / "credentials"
            credentials.mkdir(parents=True)
            credentials.chmod(0o700)
            encrypted = credentials / "minimax-api-key.cred"
            encrypted.write_bytes(b"encrypted-placeholder")
            encrypted.chmod(0o600)

            receipt = apply_config_sync(ROOT / "config", target)
            self.assertTrue(receipt["in_sync"])
            self.assertTrue((target / ".generation.json").is_file())
            self.assertEqual(b"encrypted-placeholder", encrypted.read_bytes())

            (target / "roles.toml").write_text("drift\n", encoding="utf-8")
            status = config_sync_status(ROOT / "config", target)
            self.assertEqual(["roles.toml"], status["drift"])
            repaired = apply_config_sync(ROOT / "config", target)
            self.assertTrue(repaired["in_sync"])
            self.assertIsNotNone(repaired["backup"])


if __name__ == "__main__":
    unittest.main()
