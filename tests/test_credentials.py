from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_orchestrator.credentials import (
    omniroute_runtime_key,
    private_dotenv_value,
    resolve_credential,
)


class CredentialTests(unittest.TestCase):
    def test_private_dotenv_reads_only_exact_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "OTHER=forbidden\nexport OMNIROUTE_API_KEY='runtime-secret'\n",
                encoding="utf-8",
            )
            path.chmod(0o600)
            self.assertEqual("runtime-secret", private_dotenv_value(path, "OMNIROUTE_API_KEY"))
            self.assertIsNone(private_dotenv_value(path, "MISSING"))

    def test_permissive_or_symlinked_files_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.write_text("OMNIROUTE_API_KEY=runtime-secret\n", encoding="utf-8")
            source.chmod(0o644)
            self.assertIsNone(private_dotenv_value(source, "OMNIROUTE_API_KEY"))
            link = root / "link"
            link.symlink_to(source)
            self.assertIsNone(private_dotenv_value(link, "OMNIROUTE_API_KEY"))

    def test_explicit_environment_has_priority(self) -> None:
        self.assertEqual(
            "explicit-secret",
            omniroute_runtime_key(
                {"HOME": os.devnull, "OMNIROUTE_API_KEY": "explicit-secret"}
            ),
        )

    def test_runtime_credential_precedes_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            runtime.mkdir()
            credential = runtime / "minimax-api-key"
            credential.write_text("runtime-secret", encoding="utf-8")
            credential.chmod(0o600)
            self.assertEqual(
                "runtime-secret",
                resolve_credential(
                    "minimax-api-key",
                    env_names=("MINIMAX_API_KEY",),
                    environment={
                        "CREDENTIALS_DIRECTORY": str(runtime),
                        "MINIMAX_API_KEY": "environment-secret",
                    },
                    root=root,
                ),
            )

    @patch("ai_orchestrator.credentials.subprocess.run")
    def test_encrypted_credential_is_decrypted_without_shell(self, run) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encrypted = root / "credentials" / "minimax-api-key.cred"
            encrypted.parent.mkdir()
            encrypted.parent.chmod(0o700)
            encrypted.write_bytes(b"encrypted")
            encrypted.chmod(0o600)
            run.return_value.returncode = 0
            run.return_value.stdout = b"decrypted-secret\n"
            self.assertEqual(
                "decrypted-secret",
                resolve_credential("minimax-api-key", environment={}, root=root),
            )
            command = run.call_args.args[0]
            self.assertEqual("systemd-creds", command[0])
            self.assertNotIn("sh", command)


if __name__ == "__main__":
    unittest.main()
