from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_orchestrator.models import RouteDecision, TaskSpec
from ai_orchestrator.patch_api import PatchApiError, generate_patch


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, _limit: int) -> bytes:
        return self.payload


class PatchApiTests(unittest.TestCase):
    @patch("ai_orchestrator.patch_api.deepseek_runtime_key", return_value="key")
    @patch("ai_orchestrator.patch_api.urllib.request.urlopen")
    def test_deepseek_direct_patch_uses_official_endpoint(self, urlopen, _runtime_key) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            (repo / "src").mkdir()
            (repo / "src" / "value.txt").write_text("old\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            task = TaskSpec.from_mapping(
                {
                    "repo": str(repo),
                    "task_type": "tests",
                    "sensitivity": "internal",
                    "write_mode": "validated_patch",
                    "allowed_write_paths": ["src"],
                }
            )
            decision = RouteDecision(
                schema_version=1,
                decision_id="deepseek-direct",
                task_id=task.task_id,
                profile="default",
                status="READY",
                selected_model="deepseek-v4-flash",
                requested_route_model="deepseek-v4-flash",
                provider="deepseek",
                family="deepseek",
                transport="qwen_deepseek_cli",
                reviewer_model=None,
                alternatives=(),
                unavailable={},
                auto_execute=True,
                approval_required=False,
                reason="eligible",
                context_sha256=task.context_sha256(),
                write_mode=task.write_mode,
                trust_tier="official_subscription",
                execution_profile="patch_api_writer",
            )
            urlopen.return_value = _Response(
                {
                    "model": "deepseek-v4-flash",
                    "choices": [
                        {"message": {"content": "diff --git a/src/value.txt b/src/value.txt"}}
                    ],
                }
            )

            outcome = generate_patch(task, decision, worktree=repo, prompt=b"bounded")

            request = urlopen.call_args.args[0]
            payload = json.loads(request.data)
            self.assertEqual(
                "https://api.deepseek.com/chat/completions", request.full_url
            )
            self.assertEqual("Bearer key", request.get_header("Authorization"))
            self.assertEqual("deepseek-v4-flash", payload["model"])
            self.assertNotIn("key", request.full_url)
            self.assertEqual("deepseek-v4-flash", outcome.reported_model)

    @patch("ai_orchestrator.patch_api.omniroute_runtime_key", return_value="key")
    @patch("ai_orchestrator.patch_api.urllib.request.urlopen")
    def test_omniroute_patch_receives_only_bounded_context(self, urlopen, runtime_key) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            (repo / "src").mkdir()
            (repo / "src" / "value.txt").write_text("old\n", encoding="utf-8")
            (repo / "outside.txt").write_text("secret\n", encoding="utf-8")
            (repo / "prompt.txt").write_text("change it", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            task = TaskSpec.from_mapping(
                {
                    "repo": str(repo),
                    "task_type": "tests",
                    "sensitivity": "internal",
                    "prompt_file": "prompt.txt",
                    "write_mode": "validated_patch",
                    "allowed_write_paths": ["src"],
                }
            )
            decision = RouteDecision(
                schema_version=1,
                decision_id="decision",
                task_id=task.task_id,
                profile="default",
                status="READY",
                selected_model="deepseek-v4-flash",
                requested_route_model="ds/deepseek-v4-flash",
                provider="deepseek",
                family="deepseek",
                transport="qwen_omniroute_cli",
                reviewer_model=None,
                alternatives=(),
                unavailable={},
                auto_execute=True,
                approval_required=False,
                reason="eligible",
                context_sha256=task.context_sha256(),
                write_mode=task.write_mode,
                trust_tier="official_subscription",
                execution_profile="patch_api_writer",
            )
            urlopen.return_value = _Response(
                {
                    "model": "ds/deepseek-v4-flash",
                    "choices": [
                        {
                            "message": {
                                "content": "diff --git a/src/value.txt b/src/value.txt\n"
                            }
                        }
                    ],
                }
            )

            outcome = generate_patch(task, decision, worktree=repo, prompt=b"bounded")

            request = urlopen.call_args.args[0]
            payload = json.loads(request.data)
            body = request.data.decode("utf-8")
            self.assertIn("src/value.txt", body)
            self.assertIn("old", body)
            self.assertNotIn("outside.txt", body)
            self.assertNotIn("secret", body)
            self.assertEqual(task.max_output_tokens, payload["max_tokens"])
            self.assertEqual("ds/deepseek-v4-flash", outcome.reported_model)

    @patch("ai_orchestrator.patch_api.minimax_runtime_key", return_value="key")
    @patch("ai_orchestrator.patch_api.urllib.request.urlopen")
    def test_minimax_uses_responses_api_without_retention(self, urlopen, _runtime_key) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            (repo / "src").mkdir()
            (repo / "src" / "value.txt").write_text("old\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            task = TaskSpec.from_mapping(
                {
                    "repo": str(repo),
                    "task_type": "docs",
                    "sensitivity": "internal",
                    "write_mode": "validated_patch",
                    "allowed_write_paths": ["src"],
                }
            )
            decision = RouteDecision(
                schema_version=1,
                decision_id="minimax-decision",
                task_id=task.task_id,
                profile="default",
                status="READY",
                selected_model="MiniMax-M3",
                requested_route_model="MiniMax-M3",
                provider="minimax",
                family="minimax",
                transport="qwen_minimax_cli",
                reviewer_model=None,
                alternatives=(),
                unavailable={},
                auto_execute=True,
                approval_required=False,
                reason="eligible",
                context_sha256=task.context_sha256(),
                write_mode=task.write_mode,
                trust_tier="official_subscription",
                execution_profile="patch_api_writer",
            )
            urlopen.return_value = _Response(
                {
                    "model": "MiniMax-M3",
                    "status": "completed",
                    "output_text": "diff --git a/src/value.txt b/src/value.txt",
                }
            )

            outcome = generate_patch(task, decision, worktree=repo, prompt=b"bounded")

            request = urlopen.call_args.args[0]
            payload = json.loads(request.data)
            self.assertEqual("https://api.minimax.io/v1/responses", request.full_url)
            self.assertEqual("Bearer key", request.get_header("Authorization"))
            self.assertEqual("MiniMax-M3", payload["model"])
            self.assertIn("bounded", payload["input"])
            self.assertEqual(
                task.max_output_tokens, payload["max_output_tokens"]
            )
            self.assertFalse(payload["store"])
            self.assertNotIn("messages", payload)
            self.assertEqual("MiniMax-M3", outcome.reported_model)


if __name__ == "__main__":
    unittest.main()


class PatchApiErrorCodeTests(unittest.TestCase):
    """A failure must name its own cause. Before these codes existed, every
    provider error reached the caller as MODEL_PROCESS_FAILED and three of four
    rejected candidate models had no diagnosable reason at all."""

    def _task_and_decision(self, directory: str):
        repo = Path(directory)
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        (repo / "src").mkdir()
        (repo / "src" / "value.txt").write_text("old\n", encoding="utf-8")
        (repo / "prompt.txt").write_text("change it", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        task = TaskSpec.from_mapping(
            {
                "repo": str(repo),
                "task_type": "tests",
                "sensitivity": "internal",
                "prompt_file": "prompt.txt",
                "write_mode": "validated_patch",
                "allowed_write_paths": ["src"],
            }
        )
        decision = RouteDecision(
            schema_version=1,
            decision_id="decision",
            task_id=task.task_id,
            profile="default",
            status="READY",
            selected_model="deepseek-v4-flash",
            requested_route_model="ds/deepseek-v4-flash",
            provider="deepseek",
            family="deepseek",
            transport="qwen_omniroute_cli",
            reviewer_model=None,
            alternatives=(),
            unavailable={},
            auto_execute=True,
            approval_required=False,
            reason="eligible",
            context_sha256=task.context_sha256(),
            write_mode=task.write_mode,
            trust_tier="official_subscription",
            execution_profile="patch_api_writer",
        )
        return task, decision

    def _run(self, urlopen, payload):
        with tempfile.TemporaryDirectory() as directory:
            task, decision = self._task_and_decision(directory)
            urlopen.return_value = _Response(payload)
            with self.assertRaises(PatchApiError) as raised:
                generate_patch(task, decision, worktree=Path(directory), prompt=b"do it")
            return raised.exception

    @patch("ai_orchestrator.patch_api.omniroute_runtime_key", return_value="key")
    @patch("ai_orchestrator.patch_api.urllib.request.urlopen")
    def test_reasoning_that_eats_the_whole_budget_is_named(self, urlopen, _key) -> None:
        # Measured on 2026-08-13: content empty, reasoning_content 48838 chars,
        # reasoning_tokens 16384 of a 16384 budget, finish_reason 'length'.
        # The fix for this is to shrink the request, not to retry it, so it must
        # not be reported as a generic empty answer.
        error = self._run(
            urlopen,
            {
                "model": "ds/deepseek-v4-flash",
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": "", "reasoning_content": "thinking" * 900},
                    }
                ],
            },
        )
        self.assertEqual(error.code, "REASONING_BUDGET_EXHAUSTED")

    @patch("ai_orchestrator.patch_api.omniroute_runtime_key", return_value="key")
    @patch("ai_orchestrator.patch_api.urllib.request.urlopen")
    def test_a_genuinely_empty_answer_stays_distinct(self, urlopen, _key) -> None:
        error = self._run(
            urlopen,
            {
                "model": "ds/deepseek-v4-flash",
                "choices": [{"finish_reason": "stop", "message": {"content": "   "}}],
            },
        )
        self.assertEqual(error.code, "PROVIDER_EMPTY_PATCH")

    @patch("ai_orchestrator.patch_api.omniroute_runtime_key", return_value=None)
    def test_missing_credential_is_not_a_process_failure(self, _key) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task, decision = self._task_and_decision(directory)
            with self.assertRaises(PatchApiError) as raised:
                generate_patch(task, decision, worktree=Path(directory), prompt=b"do it")
        self.assertEqual(raised.exception.code, "CREDENTIAL_UNAVAILABLE")

    def test_worker_writes_only_the_code_across_the_boundary(self) -> None:
        # The message may quote provider text; the code is a fixed literal. The
        # subprocess boundary must carry the second and never the first.
        error = PatchApiError("provider said something sensitive", "QUOTA_UNAVAILABLE")
        self.assertEqual(error.code, "QUOTA_UNAVAILABLE")
        self.assertNotIn(error.code, str(error))
