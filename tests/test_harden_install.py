from __future__ import annotations

import errno
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import harden_install
from scripts.harden_install import harden_modes


class HardenInstallTests(unittest.TestCase):
    def test_search_proxy_credential_is_created_once_without_plaintext(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config"

            def encrypt(command, *, input, check, stdout, stderr):
                self.assertEqual(b"ephemeral-local-secret", input)
                Path(command[-1]).write_bytes(b"encrypted-credential")

            with (
                patch.object(harden_install, "CONFIG_ROOT", config),
                patch.object(
                    harden_install.secrets,
                    "token_urlsafe",
                    return_value="ephemeral-local-secret",
                ),
                patch.object(harden_install.subprocess, "run", side_effect=encrypt) as run,
            ):
                first = harden_install.ensure_local_credential("web_search_proxy_key")
                second = harden_install.ensure_local_credential("web_search_proxy_key")

            self.assertEqual(first, second)
            self.assertEqual(b"encrypted-credential", first.read_bytes())
            self.assertEqual(0o600, first.stat().st_mode & 0o777)
            self.assertEqual(1, run.call_count)

    def test_install_files_points_launchers_at_managed_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            home = root / "home"
            source = root / "source"
            state = home / ".local/share/ai-orchestrator"
            config = home / ".config/ai-orchestrator"
            backup_root = state / "backups/test"
            (source / "config").mkdir(parents=True)
            (source / "config/models.toml").write_text("schema_version=2\n")
            (source / "systemd").mkdir()
            (source / "systemd/example.service").write_text("[Service]\n")
            runtime_bin = state / "runtime/venv/bin"
            runtime_bin.mkdir(parents=True)
            for name in (
                "ai-orchestrator",
                "ai-orchestrator-omniroute-wrapper",
                "omniroute-safe",
            ):
                (runtime_bin / name).write_text("#!/bin/sh\n")

            with (
                patch.object(harden_install, "HOME", home),
                patch.object(harden_install, "SOURCE", source),
                patch.object(harden_install, "STATE_ROOT", state),
                patch.object(harden_install, "CONFIG_ROOT", config),
                patch.object(harden_install, "BACKUP_ROOT", backup_root),
            ):
                harden_install.install_files()

            launcher = home / ".local/bin/ai-orchestrator"
            self.assertTrue(launcher.is_symlink())
            self.assertEqual(
                (runtime_bin / "ai-orchestrator").resolve(), launcher.resolve()
            )
            runtime_config = state / "runtime/config/models.toml"
            self.assertTrue(runtime_config.is_file())
            self.assertEqual(
                "schema_version=2\n",
                runtime_config.read_text(),
            )

    def test_read_only_managed_file_does_not_abort_install(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            managed = root / "managed-id"
            managed.write_text("opaque\n", encoding="utf-8")
            original_chmod = Path.chmod

            def guarded_chmod(path: Path, mode: int, *args, **kwargs) -> None:
                if path == managed:
                    raise OSError(errno.EROFS, "read-only file system", str(path))
                original_chmod(path, mode, *args, **kwargs)

            with patch.object(Path, "chmod", autospec=True, side_effect=guarded_chmod):
                result = harden_modes((root,))

        self.assertEqual(0, result["changed"])
        self.assertEqual(1, result["skipped_read_only"])


if __name__ == "__main__":
    unittest.main()
