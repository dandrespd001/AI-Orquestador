from __future__ import annotations

import os
import subprocess
import unittest
from unittest.mock import patch

from ai_orchestrator.canary import _environment, run_official_canaries


class CanaryTests(unittest.TestCase):
    def test_environment_never_forwards_provider_credentials(self) -> None:
        with patch.dict(
            os.environ,
            {"HOME": "/tmp/home", "PATH": "/usr/bin", "ANTHROPIC_AUTH_TOKEN": "forbidden"},
            clear=True,
        ):
            environment = _environment()
        self.assertNotIn("ANTHROPIC_AUTH_TOKEN", environment)
        self.assertEqual("/tmp/home", environment["HOME"])

    @patch("ai_orchestrator.canary.secrets.token_hex", return_value="abc123")
    @patch("ai_orchestrator.canary.shutil.which", side_effect=lambda value: "/usr/bin/" + value)
    @patch("ai_orchestrator.canary.subprocess.run")
    def test_quota_is_degraded_without_retaining_content(self, run, _which, _token) -> None:
        run.side_effect = [
            subprocess.CompletedProcess(
                [], 0,
                b'{"model":"deepseek-v4-flash","choices":[{"message":{"content":"CANARY_abc123"}}]}',
                b"",
            ),
            subprocess.CompletedProcess([], 0, b"CANARY_abc123\n", b""),
            subprocess.CompletedProcess([], 1, b"", b"usage limit reached"),
        ]
        receipt = run_official_canaries()
        self.assertEqual("DEGRADED_QUOTA", receipt["status"])
        self.assertEqual(
            ["PASSED", "PASSED", "QUOTA_UNAVAILABLE"],
            [row["status"] for row in receipt["providers"]],
        )
        self.assertTrue(
            all(row.get("orchestrator_content_retained") is False for row in receipt["providers"])
        )


if __name__ == "__main__":
    unittest.main()
