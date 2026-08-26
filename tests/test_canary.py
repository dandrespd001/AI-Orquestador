from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from ai_orchestrator.canary import (
    ROUTES,
    _abacus_canary,
    _claude_canary,
    _environment,
    _harness_canary,
    _kimi_canary,
    run_official_canaries,
)
from ai_orchestrator.supervisor import ProcessOutcome


class CanaryTests(unittest.TestCase):
    def test_provider_routes_use_the_authenticated_concrete_profiles(self) -> None:
        self.assertEqual("deepseek-v4-flash", ROUTES["deepseek-flash"].model)
        self.assertIn("deepseek-v4-flash", ROUTES["deepseek-flash"].command)
        self.assertEqual("deepseek-v4-pro", ROUTES["deepseek-pro"].model)
        self.assertIn("harness", ROUTES["deepseek-pro"].command)
        self.assertNotIn("qwen", ROUTES["deepseek-pro"].command)
        self.assertIn("--prompt-suggestions", ROUTES["anthropic"].command)
        suggestion_flag = ROUTES["anthropic"].command.index("--prompt-suggestions")
        self.assertEqual("false", ROUTES["anthropic"].command[suggestion_flag + 1])
        self.assertEqual("kimi-code/k3-256k", ROUTES["moonshot"].model)
        self.assertIn("kimi-code/k3-256k", ROUTES["moonshot"].command)
        self.assertFalse(ROUTES["abacus"].interactive_auth_only)
        self.assertIn("Grok 4.6", ROUTES["abacus"].command)
        self.assertIn("*", ROUTES["abacus"].command)

    def test_kimi_accepts_only_its_exact_text_wrapper(self) -> None:
        model = "kimi-code/k3-256k"
        self.assertEqual(
            (True, model), _kimi_canary(b"\xe2\x80\xa2 CANARY\n\n", "CANARY", model)
        )
        self.assertEqual(
            (False, model), _kimi_canary(b"\xe2\x80\xa2 CANARY extra\n", "CANARY", model)
        )

    def test_abacus_canary_requires_an_exact_no_tools_answer(self) -> None:
        self.assertEqual(
            (True, "Grok 4.6"),
            _abacus_canary(b"CANARY\n", "CANARY", "Grok 4.6"),
        )
        self.assertEqual(
            (False, "Grok 4.6"),
            _abacus_canary(b"Here is CANARY\n", "CANARY", "Grok 4.6"),
        )

    def test_deepseek_environment_never_receives_a_provider_key(self) -> None:
        with patch.dict(
            os.environ,
            {"HOME": "/tmp/home", "DEEPSEEK_API_KEY": "forbidden"},
            clear=True,
        ):
            environment = _environment("deepseek-flash")
        self.assertNotIn("OPENAI_API_KEY", environment)
        self.assertNotIn("DEEPSEEK_API_KEY", environment)

    def test_harness_canary_requires_exact_result_and_model(self) -> None:
        payload = (
            b'{"status":"SUCCESS","reported_model":"deepseek-v4-pro",'
            b'"result":"CANARY"}'
        )
        self.assertEqual(
            (True, "deepseek-v4-pro"),
            _harness_canary(payload, "CANARY", "deepseek-v4-pro"),
        )

    def test_codex_canary_can_run_outside_a_git_repository(self) -> None:
        self.assertIn("--skip-git-repo-check", ROUTES["openai"].command)

    def test_claude_alias_requires_matching_reported_family(self) -> None:
        wrong = (
            b'{"result":"CANARY","modelUsage":'
            b'{"claude-haiku-4-5-20251001":{}}}'
        )
        right = (
            b'{"result":"CANARY","modelUsage":'
            b'{"claude-sonnet-5-20260801":{}}}'
        )
        self.assertEqual((False, "claude-haiku-4-5-20251001"), _claude_canary(wrong, "CANARY", "sonnet"))
        self.assertEqual((True, "claude-sonnet-5-20260801"), _claude_canary(right, "CANARY", "sonnet"))

    def test_claude_canary_ignores_auxiliary_model_usage(self) -> None:
        mixed = (
            b'{"result":"CANARY","modelUsage":{'
            b'"claude-haiku-4-5-20251001":{},'
            b'"claude-sonnet-5":{}}}'
        )

        self.assertEqual(
            (True, "claude-sonnet-5"),
            _claude_canary(mixed, "CANARY", "sonnet"),
        )

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
        harness_success = (
            b'{"status":"SUCCESS","reported_model":"deepseek-v4-flash",'
            b'"result":"CANARY_abc123"}'
        )
        minimax_success = (
            b'[{"type":"system","model":"MiniMax-M3"},'
            b'{"type":"result","subtype":"success","is_error":false,'
            b'"result":"CANARY_abc123"}]'
        )
        run.side_effect = [
            ProcessOutcome(0, harness_success, b""),
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
