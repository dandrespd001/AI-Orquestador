from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_orchestrator.qwen_safe import (
    EXIT_CONFIG,
    SafeQwenError,
    _environment,
    credential_name,
    main,
    referenced_env_keys,
    resolve_environment,
)

SETTINGS = {
    "modelProviders": {
        "openai": [
            {
                "id": "kmc/k3",
                "baseUrl": "http://localhost:20128/v1",
                "envKey": "QWEN_CUSTOM_API_KEY_OPENAI_HTTP_LOCALHOST_20128_V1_ABC123",
            },
            {"id": "MiniMax-M3", "envKey": "MINIMAX_API_KEY"},
            {"id": "deepseek-v4-pro", "envKey": "DEEPSEEK_API_KEY"},
            {"id": "no-key-provider"},
        ]
    },
    "model": {"name": "deepseek-v4-flash"},
}


def _settings_file(directory: str, payload: dict | None = None) -> Path:
    path = Path(directory) / "settings.json"
    path.write_text(json.dumps(payload or SETTINGS), encoding="utf-8")
    return path


class QwenSafeNamingTests(unittest.TestCase):
    def test_known_variables_map_to_stable_credential_names(self) -> None:
        self.assertEqual("qwen-deepseek-api-key", credential_name("DEEPSEEK_API_KEY"))
        self.assertEqual("qwen-minimax-api-key", credential_name("MINIMAX_API_KEY"))

    def test_omniroute_key_reuses_the_service_credential(self) -> None:
        # A second copy of the same secret would be a second thing to rotate.
        self.assertEqual("omniroute_api_key", credential_name("OMNIROUTE_API_KEY"))

    def test_per_installation_custom_key_collapses_to_one_name(self) -> None:
        # The suffix is an installation hash, so it must not reach the store.
        self.assertEqual(
            "qwen-omniroute-custom-api-key",
            credential_name("QWEN_CUSTOM_API_KEY_OPENAI_HTTP_LOCALHOST_20128_V1_ABC1"),
        )

    def test_unknown_variable_still_gets_a_deterministic_name(self) -> None:
        self.assertEqual(
            "qwen-new-vendor-api-key", credential_name("NEW_VENDOR_API_KEY")
        )


class QwenSafeSettingsTests(unittest.TestCase):
    def test_env_keys_are_collected_from_providers_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            keys = referenced_env_keys(_settings_file(directory))
        self.assertEqual(
            (
                "DEEPSEEK_API_KEY",
                "MINIMAX_API_KEY",
                "QWEN_CUSTOM_API_KEY_OPENAI_HTTP_LOCALHOST_20128_V1_ABC123",
            ),
            keys,
        )

    def test_a_leftover_env_block_is_still_covered(self) -> None:
        payload = dict(SETTINGS, env={"LOCAL_LLM_API_KEY": "unused"})
        with tempfile.TemporaryDirectory() as directory:
            keys = referenced_env_keys(_settings_file(directory, payload))
        self.assertIn("LOCAL_LLM_API_KEY", keys)

    def test_missing_settings_file_fails_closed(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises(SafeQwenError),
        ):
            referenced_env_keys(Path(directory) / "absent.json")


class QwenSafeResolutionTests(unittest.TestCase):
    def test_one_unresolved_key_blocks_the_whole_launch(self) -> None:
        def only_deepseek(name: str, **kwargs: object) -> str | None:
            return "secret" if name == "qwen-deepseek-api-key" else None

        with tempfile.TemporaryDirectory() as directory:
            path = _settings_file(directory)
            with (
                patch(
                    "ai_orchestrator.qwen_safe.resolve_credential",
                    side_effect=only_deepseek,
                ),
                self.assertRaises(SafeQwenError) as caught,
            ):
                resolve_environment(path)
        self.assertIn("MINIMAX_API_KEY", str(caught.exception))

    def test_environment_carries_secrets_and_drops_unrelated_variables(self) -> None:
        with patch.dict(
            os.environ,
            {"HOME": "/tmp/home", "BROKER_SECRET": "forbidden", "TERM": "xterm"},
            clear=True,
        ):
            environment = _environment({"DEEPSEEK_API_KEY": "runtime-secret"})
        self.assertEqual("runtime-secret", environment["DEEPSEEK_API_KEY"])
        self.assertEqual("xterm", environment["TERM"])
        self.assertNotIn("BROKER_SECRET", environment)
        self.assertEqual("env-key", environment["NO_LOG_API_KEY_IDS"])


class QwenSafeLaunchTests(unittest.TestCase):
    def test_exec_uses_argv_without_a_shell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "qwen"
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            executable.chmod(0o700)
            with (
                patch("ai_orchestrator.qwen_safe._executable", return_value=executable),
                patch(
                    "ai_orchestrator.qwen_safe.resolve_environment",
                    return_value={"DEEPSEEK_API_KEY": "runtime-secret"},
                ),
                patch(
                    "ai_orchestrator.qwen_safe.os.execve", side_effect=OSError("stop")
                ) as run,
                self.assertRaises(OSError),
            ):
                main(("--version",))
            argv = run.call_args.args[1]
            environment = run.call_args.args[2]
        self.assertEqual([str(executable), "--version"], argv)
        self.assertNotIn("sh", argv)
        self.assertEqual("runtime-secret", environment["DEEPSEEK_API_KEY"])

    def test_unresolved_credentials_return_config_exit_before_exec(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "qwen"
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            executable.chmod(0o700)
            with (
                patch("ai_orchestrator.qwen_safe._executable", return_value=executable),
                patch(
                    "ai_orchestrator.qwen_safe.resolve_environment",
                    side_effect=SafeQwenError("unavailable"),
                ),
                patch("ai_orchestrator.qwen_safe.os.execve") as run,
            ):
                status = main(("--version",))
        self.assertEqual(EXIT_CONFIG, status)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
