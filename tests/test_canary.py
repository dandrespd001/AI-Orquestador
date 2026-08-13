from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from ai_orchestrator.canary import _environment, run_official_canaries
from ai_orchestrator.supervisor import ProcessOutcome


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
    @patch("ai_orchestrator.canary.run_supervised")
    def test_quota_is_degraded_without_retaining_content(self, run, _which, _token) -> None:
        qwen_success = (
            b'[{"type":"system","model":"ds/deepseek-v4-flash"},'
            b'{"type":"result","subtype":"success","is_error":false,'
            b'"result":"CANARY_abc123"}]'
        )
        minimax_success = (
            b'[{"type":"system","model":"MiniMax-M3"},'
            b'{"type":"result","subtype":"success","is_error":false,'
            b'"result":"CANARY_abc123"}]'
        )
        run.side_effect = [
            ProcessOutcome(0, qwen_success, b""),
            ProcessOutcome(0, minimax_success, b""),
            ProcessOutcome(1, b"", b"usage limit reached"),
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

    @patch("ai_orchestrator.canary.secrets.token_hex", return_value="abc123")
    @patch("ai_orchestrator.canary.shutil.which", return_value="/usr/bin/qwen")
    @patch("ai_orchestrator.canary.run_supervised")
    def test_missing_minimax_key_is_explicit(self, run, _which, _token) -> None:
        run.return_value = ProcessOutcome(
            1,
            b'[{"type":"result","subtype":"error_during_execution","is_error":true,'
            b'"error":{"message":"MINIMAX_API_KEY environment variable not found"}}]',
            b"",
        )
        receipt = run_official_canaries(["minimax"])
        self.assertEqual("DEGRADED_UNAVAILABLE", receipt["status"])
        self.assertEqual("CREDENTIAL_UNAVAILABLE", receipt["providers"][0]["status"])


if __name__ == "__main__":
    unittest.main()
