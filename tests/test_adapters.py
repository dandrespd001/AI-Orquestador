from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_orchestrator.adapters import (
    AdapterError,
    _command,
    _minimal_environment,
    _qwen_error,
    execute,
)
from ai_orchestrator.models import RouteDecision, TaskSpec
from ai_orchestrator.supervisor import ProcessOutcome


class AdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        (self.repo / "prompt.txt").write_text("bounded task", encoding="utf-8")
        self.task = TaskSpec.from_mapping(
            {
                "repo": str(self.repo),
                "task_type": "tests",
                "sensitivity": "internal",
                "prompt_file": "prompt.txt",
            }
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def decision(self, **overrides: object) -> RouteDecision:
        values: dict[str, object] = {
            "schema_version": 1,
            "decision_id": "d1",
            "task_id": self.task.task_id,
            "profile": "default",
            "status": "READY",
            "selected_model": "gpt-5.6-sol",
            "requested_route_model": "gpt-5.6-sol",
            "provider": "openai",
            "family": "openai-gpt",
            "transport": "codex_cli",
            "reviewer_model": "claude-opus",
            "alternatives": (),
            "unavailable": {},
            "auto_execute": True,
            "approval_required": False,
            "reason": "eligible",
            "context_sha256": self.task.context_sha256(),
        }
        values.update(overrides)
        return RouteDecision(**values)

    def test_command_uses_argv_and_stdin_without_shell(self) -> None:
        command = _command(
            self.task,
            self.decision(reasoning_effort="xhigh"),
            self.task.prompt_path(),
        )
        self.assertEqual("codex", command[0])
        self.assertIn("--model", command)
        self.assertIn('model_reasoning_effort="xhigh"', command)
        self.assertNotIn("sh", command)

    def test_claude_command_receives_role_effort(self) -> None:
        decision = self.decision(
            selected_model="claude-opus",
            requested_route_model="opus",
            provider="anthropic",
            family="anthropic-claude",
            transport="claude_cli",
            reasoning_effort="xhigh",
        )
        command = _command(self.task, decision, self.task.prompt_path())
        self.assertEqual("claude", command[0])
        self.assertEqual("xhigh", command[command.index("--effort") + 1])

    def test_claude_review_uses_native_structured_output(self) -> None:
        task = TaskSpec.from_mapping(
            {
                "repo": str(self.repo),
                "task_type": "review",
                "sensitivity": "internal",
                "prompt_file": "prompt.txt",
                "output_contract": "review",
            }
        )
        decision = self.decision(
            selected_model="claude-opus",
            requested_route_model="opus",
            provider="anthropic",
            family="anthropic-claude",
            transport="claude_cli",
        )

        command = _command(task, decision, task.prompt_path())

        schema = json.loads(command[command.index("--json-schema") + 1])
        self.assertEqual(["verdict", "findings"], schema["required"])
        self.assertEqual(
            ["accept", "revise", "escalate"],
            schema["properties"]["verdict"]["enum"],
        )
        finding = schema["properties"]["findings"]["items"]
        self.assertEqual(["severity", "evidence", "message"], finding["required"])
        self.assertFalse(schema["additionalProperties"])
        self.assertFalse(finding["additionalProperties"])

    def test_minimax_research_uses_isolated_opencode_transport(self) -> None:
        task = TaskSpec.from_mapping(
            {
                "repo": str(self.repo),
                "task_type": "research",
                "sensitivity": "public",
                "prompt_file": "prompt.txt",
            }
        )
        decision = self.decision(
            selected_model="MiniMax-M3",
            requested_route_model="MiniMax-M3",
            provider="minimax",
            family="minimax",
            transport="opencode_minimax_research",
        )
        command = _command(task, decision, task.prompt_path())
        self.assertEqual("opencode", command[0])
        self.assertIn("--pure", command)
        self.assertEqual("minimax-research/MiniMax-M3", command[command.index("--model") + 1])
        self.assertEqual("research", command[command.index("--agent") + 1])

    def test_minimax_qwen_research_fails_closed_without_search_backend(self) -> None:
        task = TaskSpec.from_mapping(
            {
                "repo": str(self.repo),
                "task_type": "research",
                "sensitivity": "public",
                "prompt_file": "prompt.txt",
            }
        )
        decision = self.decision(
            selected_model="MiniMax-M3",
            requested_route_model="MiniMax-M3",
            provider="minimax",
            family="minimax",
            transport="qwen_minimax_cli",
        )
        with self.assertRaises(AdapterError) as context:
            _command(task, decision, task.prompt_path())
        self.assertEqual("WEB_RESEARCH_UNAVAILABLE", context.exception.code)

    @patch("ai_orchestrator.adapters.run_supervised")
    def test_minimax_research_environment_is_ephemeral_and_web_only(self, run) -> None:
        task = TaskSpec.from_mapping(
            {
                "repo": str(self.repo),
                "task_type": "research",
                "sensitivity": "public",
                "prompt_file": "prompt.txt",
            }
        )
        decision = self.decision(
            selected_model="MiniMax-M3",
            requested_route_model="MiniMax-M3",
            provider="minimax",
            family="minimax",
            transport="opencode_minimax_research",
        )
        run.return_value = ProcessOutcome(
            0,
            b'{"type":"step_start","model":"MiniMax-M3"}\n'
            b'{"type":"tool_use","part":{"tool":"websearch"}}\n'
            b'{"type":"text","text":"researched answer"}\n',
            b"",
        )
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "secret"}, clear=False):
            result, output = execute(task, decision)
        self.assertEqual("SUCCESS", result.status)
        self.assertEqual(b"researched answer", output)
        environment = run.call_args.kwargs["env"]
        config = json.loads(environment["OPENCODE_CONFIG_CONTENT"])
        permissions = config["agent"]["research"]["permission"]
        self.assertEqual("deny", permissions["*"])
        self.assertEqual("allow", permissions["websearch"])
        # Un brief que da las URLs hechas necesita leerlas, no buscarlas.
        self.assertEqual("allow", permissions["webfetch"])
        self.assertEqual("deny", permissions["bash"])
        self.assertEqual("deny", permissions["edit"])
        self.assertNotIn("secret", environment["OPENCODE_CONFIG_CONTENT"])
        self.assertEqual("1", environment["OPENCODE_ENABLE_EXA"])
        self.assertNotEqual(os.environ.get("HOME"), environment["HOME"])

    @patch("ai_orchestrator.adapters.run_supervised")
    def test_minimax_research_verifies_model_from_ephemeral_session(self, run) -> None:
        task = TaskSpec.from_mapping(
            {
                "repo": str(self.repo),
                "task_type": "research",
                "sensitivity": "public",
                "prompt_file": "prompt.txt",
            }
        )
        decision = self.decision(
            selected_model="MiniMax-M3",
            requested_route_model="MiniMax-M3",
            provider="minimax",
            family="minimax",
            transport="opencode_minimax_research",
        )

        def opencode_run(*_args, **kwargs):
            database = Path(kwargs["env"]["XDG_DATA_HOME"]) / "opencode" / "opencode.db"
            database.parent.mkdir(parents=True)
            with sqlite3.connect(database) as connection:
                connection.execute(
                    "CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, "
                    "time_created INTEGER, data TEXT)"
                )
                connection.execute(
                    "INSERT INTO message VALUES (?, ?, ?, ?)",
                    (
                        "msg_assistant",
                        "ses_research",
                        1,
                        json.dumps(
                            {
                                "role": "assistant",
                                "modelID": "MiniMax-M3",
                                "providerID": "minimax-research",
                            }
                        ),
                    ),
                )
            return ProcessOutcome(
                0,
                b'{"type":"step_start","sessionID":"ses_research"}\n'
                b'{"type":"tool_use","part":{"tool":"websearch"}}\n'
                b'{"type":"text","text":"researched answer"}\n',
                b"",
            )

        run.side_effect = opencode_run
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "secret"}, clear=False):
            result, output = execute(task, decision)
        self.assertEqual("SUCCESS", result.status)
        self.assertTrue(result.provenance_verified)
        self.assertEqual("MiniMax-M3", result.reported_model)
        self.assertEqual(b"researched answer", output)

    @patch("ai_orchestrator.adapters.run_supervised")
    def test_minimax_research_without_a_real_search_is_rejected(self, run) -> None:
        task = TaskSpec.from_mapping(
            {
                "repo": str(self.repo),
                "task_type": "research",
                "sensitivity": "public",
                "prompt_file": "prompt.txt",
            }
        )
        decision = self.decision(
            selected_model="MiniMax-M3",
            requested_route_model="MiniMax-M3",
            provider="minimax",
            family="minimax",
            transport="opencode_minimax_research",
        )
        run.return_value = ProcessOutcome(
            0,
            b'{"type":"step_start","model":"MiniMax-M3"}\n'
            b'{"type":"text","text":"answer from memory"}\n',
            b"",
        )
        result, _ = execute(task, decision)
        self.assertEqual("WEB_RESEARCH_NOT_PERFORMED", result.status)
        self.assertEqual("WEB_RESEARCH_NOT_PERFORMED", result.error_code)

    def test_minimax_transport_uses_bounded_read_only_qwen(self) -> None:
        decision = self.decision(
            selected_model="MiniMax-M3",
            requested_route_model="MiniMax-M3",
            provider="minimax",
            family="minimax",
            transport="qwen_minimax_cli",
        )
        command = _command(self.task, decision, self.task.prompt_path())
        self.assertEqual("qwen", command[0])
        self.assertIn("plan", command)
        self.assertIn("--max-wall-time", command)
        self.assertNotIn("opencode", command)

    def test_deepseek_transport_uses_direct_qwen_profile_without_secret_argv(self) -> None:
        decision = self.decision(
            selected_model="deepseek-v4-flash",
            requested_route_model="deepseek-v4-flash",
            provider="deepseek",
            family="deepseek",
            transport="qwen_deepseek_cli",
        )
        command = _command(self.task, decision, self.task.prompt_path())
        self.assertEqual("qwen", command[0])
        self.assertEqual("deepseek-v4-flash", command[command.index("--model") + 1])
        self.assertEqual(1, command.count("--max-tool-calls"))
        self.assertNotIn("DEEPSEEK_API_KEY", " ".join(command))

    def test_qwen_writer_uses_read_only_patch_mode_and_real_tool_names(self) -> None:
        task = TaskSpec.from_mapping(
            {
                "repo": str(self.repo),
                "task_type": "tests",
                "sensitivity": "internal",
                "prompt_file": "prompt.txt",
                "write_mode": "validated_patch",
                "allowed_write_paths": ["generated.txt"],
            }
        )
        decision = self.decision(
            selected_model="deepseek-v4-flash",
            requested_route_model="ds/deepseek-v4-flash",
            provider="deepseek",
            family="deepseek",
            transport="qwen_omniroute_cli",
            write_mode="validated_patch",
            trust_tier="official_subscription",
            execution_profile="patch_api_writer",
        )
        command = _command(task, decision, task.prompt_path())
        self.assertIn("plan", command)
        self.assertNotIn("auto-edit", command)
        excluded = command[command.index("--exclude-tools") + 1]
        self.assertIn("run_shell_command", excluded)
        self.assertIn("agent", excluded)
        self.assertNotIn("shell,write,edit,agent", excluded)

    def test_environment_does_not_forward_unrelated_tokens(self) -> None:
        with patch.dict(
            os.environ,
            {"BROKER_SECRET_KEY": "forbidden", "ANTHROPIC_AUTH_TOKEN": "forbidden", "HOME": str(self.root)},
            clear=True,
        ):
            environment = _minimal_environment("codex_cli")
        self.assertNotIn("BROKER_SECRET_KEY", environment)
        self.assertNotIn("ANTHROPIC_AUTH_TOKEN", environment)
        self.assertEqual(str(self.root), environment["HOME"])

    def test_qwen_environment_only_allows_its_transport_credential(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HOME": str(self.root),
                "OMNIROUTE_API_KEY": "allowed",
                "MINIMAX_API_KEY": "forbidden",
            },
            clear=True,
        ):
            environment = _minimal_environment("qwen_omniroute_cli")
        self.assertEqual("allowed", environment["OPENAI_API_KEY"])
        self.assertEqual("http://127.0.0.1:20128/v1", environment["OPENAI_BASE_URL"])
        self.assertNotIn("OMNIROUTE_API_KEY", environment)
        self.assertNotIn("MINIMAX_API_KEY", environment)

    def test_minimax_environment_uses_openai_v1_surface(self) -> None:
        with patch.dict(
            os.environ,
            {"HOME": str(self.root), "MINIMAX_API_KEY": "minimax-secret"},
            clear=True,
        ):
            environment = _minimal_environment("qwen_minimax_cli")
        self.assertEqual("minimax-secret", environment["OPENAI_API_KEY"])
        self.assertEqual("https://api.minimax.io/v1", environment["OPENAI_BASE_URL"])
        self.assertNotIn("MINIMAX_API_KEY", environment)
        self.assertNotIn("ANTHROPIC_API_KEY", environment)

    def test_deepseek_environment_uses_official_api_surface(self) -> None:
        with patch.dict(
            os.environ,
            {"HOME": str(self.root), "DEEPSEEK_API_KEY": "deepseek-secret"},
            clear=True,
        ):
            environment = _minimal_environment("qwen_deepseek_cli")
        self.assertEqual("deepseek-secret", environment["OPENAI_API_KEY"])
        self.assertEqual("https://api.deepseek.com", environment["OPENAI_BASE_URL"])
        self.assertNotIn("DEEPSEEK_API_KEY", environment)

    def test_qwen_zero_exit_error_envelope_is_not_success(self) -> None:
        stdout = b'[{"type":"result","subtype":"error","is_error":true,"result":"upstream"}]'
        self.assertEqual("PROVIDER_ERROR", _qwen_error(stdout, b""))

    def test_qwen_false_success_api_error_is_not_success(self) -> None:
        stdout = (
            b'[{"type":"system","model":"ds/deepseek-v4-flash"},'
            b'{"type":"result","subtype":"success","is_error":false,'
            b'"result":"[API Error: Connection error.]"}]'
        )
        self.assertEqual("PROVIDER_ERROR", _qwen_error(stdout, b""))

    @patch("ai_orchestrator.adapters.run_supervised")
    def test_qwen_success_requires_valid_result_and_provenance(self, run) -> None:
        run.return_value = ProcessOutcome(
            0,
            b'[{"type":"system","model":"ds/deepseek-v4-flash"},'
            b'{"type":"result","subtype":"success","is_error":false,"result":"ok"}]',
            b"",
        )
        decision = self.decision(
            selected_model="deepseek-v4-flash",
            requested_route_model="ds/deepseek-v4-flash",
            provider="deepseek",
            family="deepseek",
            transport="qwen_omniroute_cli",
        )
        result, _ = execute(self.task, decision)
        self.assertEqual("SUCCESS", result.status)
        self.assertTrue(result.provenance_verified)
        self.assertEqual(b"bounded task", run.call_args.kwargs["input_data"])

    @patch("ai_orchestrator.adapters.run_supervised")
    def test_wall_timeout_is_explicit(self, run) -> None:
        run.return_value = ProcessOutcome(124, b"", b"", timed_out=True)
        result, _ = execute(self.task, self.decision())
        self.assertEqual("WALL_TIMEOUT", result.status)
        self.assertEqual("WALL_TIMEOUT", result.error_code)
        self.assertEqual(124, result.exit_code)

    def test_handoff_is_never_executed(self) -> None:
        decision = self.decision(status="HANDOFF_REQUIRED", auto_execute=False)
        with self.assertRaises(AdapterError) as context:
            execute(self.task, decision)
        self.assertEqual("HANDOFF_REQUIRED", context.exception.code)



class ResearchEgressTests(unittest.TestCase):
    """Lo que costo tenerlo al reves: un informe entero escrito de memoria."""

    def test_the_research_prompt_tells_the_model_to_open_named_urls(self) -> None:
        from ai_orchestrator.adapters import _research_prompt

        policy = _research_prompt(b"encargo").decode("utf-8")
        self.assertIn("webfetch", policy)
        self.assertIn("HTTP status you actually got", policy)

    def test_fetching_without_searching_still_counts_as_research(self) -> None:
        from ai_orchestrator.adapters import _contains_tool

        # Un encargo que da las seis URLs y prohibe buscar: solo hay webfetch.
        payload = [{"type": "tool_use", "part": {"tool": "webfetch"}}]
        self.assertTrue(_contains_tool(payload, "webfetch"))
        self.assertFalse(_contains_tool(payload, "websearch"))


if __name__ == "__main__":
    unittest.main()
