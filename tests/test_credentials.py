from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_orchestrator.credentials import (
    authentication_inventory,
    deepseek_runtime_key,
    minimax_runtime_key,
    omniroute_runtime_key,
    private_dotenv_value,
    resolve_credential,
)


class CredentialTests(unittest.TestCase):
    def _qwen_settings(self, root: Path, *, mode: int = 0o600) -> Path:
        path = root / "settings.json"
        path.write_text(
            json.dumps(
                {
                    "env": {
                        "DEEPSEEK_API_KEY": "deepseek-profile-secret",
                        "MINIMAX_API_KEY": "minimax-profile-secret",
                    },
                    "modelProviders": {
                        "openai": [
                            {
                                "id": "deepseek-v4-flash",
                                "baseUrl": "https://api.deepseek.com",
                                "envKey": "DEEPSEEK_API_KEY",
                            },
                            {
                                "id": "MiniMax-M3",
                                "baseUrl": "https://api.minimax.io/v1",
                                "envKey": "MINIMAX_API_KEY",
                            },
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )
        path.chmod(mode)
        return path

    def test_private_qwen_profiles_are_runtime_credential_fallbacks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = self._qwen_settings(root)
            self.assertEqual(
                "deepseek-profile-secret",
                deepseek_runtime_key(
                    {}, root=root, qwen_settings=settings
                ),
            )
            self.assertEqual(
                "minimax-profile-secret",
                minimax_runtime_key(
                    {}, root=root, qwen_settings=settings
                ),
            )

    def test_qwen_profile_fallback_rejects_permissive_or_unexpected_routes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = self._qwen_settings(root, mode=0o644)
            self.assertIsNone(
                deepseek_runtime_key({}, root=root, qwen_settings=settings)
            )
            settings.chmod(0o600)
            payload = json.loads(settings.read_text(encoding="utf-8"))
            payload["modelProviders"]["openai"][0]["baseUrl"] = (
                "https://example.invalid"
            )
            settings.write_text(json.dumps(payload), encoding="utf-8")
            settings.chmod(0o600)
            self.assertIsNone(
                deepseek_runtime_key({}, root=root, qwen_settings=settings)
            )

    def test_environment_precedes_qwen_profile_and_inventory_reports_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = self._qwen_settings(root)
            self.assertEqual(
                "environment-secret",
                deepseek_runtime_key(
                    {"DEEPSEEK_API_KEY": "environment-secret"},
                    root=root,
                    qwen_settings=settings,
                ),
            )
            inventory = authentication_inventory(root, qwen_settings=settings)
            by_provider = {row["provider"]: row for row in inventory}
            self.assertEqual("available", by_provider["deepseek"]["status"])
            self.assertEqual("qwen_private_profile", by_provider["deepseek"]["source"])
            self.assertEqual("available", by_provider["minimax"]["status"])
            self.assertNotIn("deepseek-profile-secret", str(inventory))

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
