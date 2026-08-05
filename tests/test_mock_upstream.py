from __future__ import annotations

import unittest

from ai_orchestrator.mock_upstream import response_for


class MockUpstreamTests(unittest.TestCase):
    def test_chat_and_tool_contracts_are_deterministic(self) -> None:
        status, chat = response_for("/v1/chat/completions", {"model": "mock-model"})
        self.assertEqual(200, status)
        self.assertEqual("mock-model", chat["model"])
        self.assertEqual("gateway-ok", chat["choices"][0]["message"]["content"])

        status, tool = response_for(
            "/v1/chat/completions", {"model": "mock-model", "tools": [{"type": "function"}]}
        )
        self.assertEqual(200, status)
        self.assertEqual("sum_values", tool["choices"][0]["message"]["tool_calls"][0]["function"]["name"])

    def test_synthetic_error_is_body_free_and_stable(self) -> None:
        status, payload = response_for(
            "/v1/chat/completions", {"metadata": {"test_case": "force-429"}}
        )
        self.assertEqual(429, status)
        self.assertEqual("rate_limit_error", payload["error"]["type"])


if __name__ == "__main__":
    unittest.main()
