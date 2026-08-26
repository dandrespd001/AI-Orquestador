from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_orchestrator.omniroute_safe import (
    EXIT_CONFIG,
    SafeOmniRouteError,
    _environment,
    _validate_arguments,
    main,
)


class OmniRouteSafeTests(unittest.TestCase):
    def test_destructive_and_auth_export_commands_are_blocked(self) -> None:
        for arguments in (
            ("resilience", "reset"),
            ("usage", "budget", "reset"),
            ("storage", "purge"),
            ("auth", "export"),
        ):
            with self.subTest(arguments=arguments), self.assertRaises(SafeOmniRouteError):
                _validate_arguments(arguments)

    def test_environment_contains_only_runtime_key_and_loopback_base(self) -> None:
        with patch.dict(
            os.environ,
            {"HOME": "/tmp/home", "BROKER_SECRET": "forbidden", "TERM": "xterm"},
            clear=True,
        ):
            environment = _environment("runtime-secret")
        self.assertEqual("runtime-secret", environment["OMNIROUTE_API_KEY"])
        self.assertEqual("http://127.0.0.1:20128/v1", environment["OMNIROUTE_BASE_URL"])
        self.assertNotIn("BROKER_SECRET", environment)

    def test_exec_uses_argv_without_shell_and_never_persists_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "omniroute"
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            executable.chmod(0o700)
            with (
                patch("ai_orchestrator.omniroute_safe._executable", return_value=executable),
                patch(
                    "ai_orchestrator.omniroute_safe.omniroute_runtime_key",
                    return_value="runtime-secret",
                ),
                patch("ai_orchestrator.omniroute_safe.os.execve", side_effect=OSError("stop")) as run,
                self.assertRaises(OSError),
            ):
                main(("doctor",))
            argv = run.call_args.args[1]
            environment = run.call_args.args[2]
            self.assertEqual([str(executable), "doctor"], argv)
            self.assertNotIn("sh", argv)
            self.assertEqual("runtime-secret", environment["OMNIROUTE_API_KEY"])

    @patch("ai_orchestrator.omniroute_safe._executable")
    def test_policy_failure_returns_config_exit_before_exec(self, executable) -> None:
        self.assertEqual(EXIT_CONFIG, main(("resilience", "reset")))
        executable.assert_not_called()


if __name__ == "__main__":
    unittest.main()
