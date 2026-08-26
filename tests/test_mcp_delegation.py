from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_orchestrator.delegation import OrchestratorTaskRunner
from ai_orchestrator.mcp_tools import MCPQueryServer
from ai_orchestrator.models import ContractError, RunResult

ROOT = Path(__file__).resolve().parents[1]


class UnifiedMCPDelegationTests(unittest.TestCase):
    def test_codex_and_claude_complete_initialize_and_tools_list(self) -> None:
        server = MCPQueryServer(object())
        for client in ("codex", "claude-code"):
            with self.subTest(client=client):
                initialized = server.handle(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2025-03-26",
                            "capabilities": {},
                            "clientInfo": {"name": client, "version": "test"},
                        },
                    }
                )
                listed = server.handle(
                    {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
                )
                self.assertEqual("ai-orchestrator", initialized["result"]["serverInfo"]["name"])
                self.assertEqual(4, len(listed["result"]["tools"]))

    def test_mcp_exposes_one_policy_entrypoint_for_configured_providers(self) -> None:
        class FakeTaskRunner:
            def run(self, **kwargs: object) -> dict[str, object]:
                return {"status": "SUCCESS", "received": sorted(kwargs)}

        server = MCPQueryServer(
            object(), task_runner=FakeTaskRunner(), fixed_profile="default"
        )
        listed = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        names = {tool["name"] for tool in listed["result"]["tools"]}
        self.assertIn("orchestrator_submit_task", names)

        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            spec = repo / "task.json"
            spec.write_text(
                json.dumps(
                    {
                        "repo": str(repo),
                        "task_type": "review",
                        "sensitivity": "internal",
                        "write_mode": "read_only",
                    }
                )
            )
            response = server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "orchestrator_submit_task",
                        "arguments": {
                            "spec_path": str(spec),
                            "model": "MiniMax-M3",
                        },
                    },
                }
            )
        payload = json.loads(response["result"]["content"][0]["text"])
        self.assertFalse(response["result"]["isError"])
        self.assertEqual("SUCCESS", payload["status"])
        self.assertEqual(["model", "profile", "spec_path"], payload["received"])

    def test_task_runner_routes_a_read_only_task_through_the_common_policy(
        self,
    ) -> None:
        class FakeTelemetry:
            routes = 0
            runs = 0

            def record_route(self, _task: object, _decision: object) -> None:
                self.routes += 1

            def record_run(self, _run: object) -> None:
                self.runs += 1

        telemetry = FakeTelemetry()

        def fake_execute(task: object, decision: object, **_kwargs: object):
            result = RunResult(
                schema_version=1,
                run_id="00000000-0000-0000-0000-000000000002",
                decision_id=decision.decision_id,
                task_id=task.task_id,
                status="SUCCESS",
                requested_model=decision.selected_model,
                reported_model=decision.selected_model,
                provider=decision.provider,
                transport=decision.transport,
                started_at="2026-08-20T00:00:00+00:00",
                finished_at="2026-08-20T00:00:01+00:00",
                latency_ms=1,
                output_sha256="0" * 64,
                output_bytes=6,
                exit_code=0,
                provenance_verified=True,
            )
            return result, b"answer"

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repo"
            repo.mkdir()
            (repo / "prompt.txt").write_text("review", encoding="utf-8")
            spec = Path(directory) / "task.json"
            spec.write_text(
                json.dumps(
                    {
                        "task_id": "00000000-0000-0000-0000-000000000001",
                        "repo": str(repo),
                        "task_type": "review",
                        "sensitivity": "internal",
                        "prompt_file": "prompt.txt",
                        "write_mode": "read_only",
                    }
                ),
                encoding="utf-8",
            )
            receipt = OrchestratorTaskRunner(
                configuration_root=ROOT / "config",
                executor=fake_execute,
                telemetry=telemetry,
            ).run(
                spec_path=str(spec),
                profile="default",
                model="gpt-5.6-terra",
            )

        self.assertEqual("SUCCESS", receipt["status"])
        self.assertEqual("gpt-5.6-terra", receipt["decision"]["selected_model"])
        self.assertEqual("answer", receipt["output"])
        self.assertEqual(1, telemetry.routes)
        self.assertEqual(1, telemetry.runs)

    def test_mcp_task_runner_never_applies_a_patch_to_the_source_repo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repo"
            repo.mkdir()
            (repo / "prompt.txt").write_text("change", encoding="utf-8")
            spec = Path(directory) / "task.json"
            spec.write_text(
                json.dumps(
                    {
                        "repo": str(repo),
                        "task_type": "implementation",
                        "sensitivity": "internal",
                        "prompt_file": "prompt.txt",
                        "write_mode": "validated_apply",
                        "allowed_write_paths": ["src/value.py"],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ContractError, "validated_patch"):
                OrchestratorTaskRunner(configuration_root=ROOT / "config").run(
                    spec_path=str(spec), profile="default"
                )


if __name__ == "__main__":
    unittest.main()
