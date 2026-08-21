from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_orchestrator.admission import AdmissionController
from ai_orchestrator.delegation import OrchestratorTaskRunner
from ai_orchestrator.mcp_tools import MCPQueryServer
from ai_orchestrator.models import ContractError, RunResult

ROOT = Path(__file__).resolve().parents[1]


class UnifiedMCPDelegationTests(unittest.TestCase):
    def test_initialize_negotiates_versions_and_exposes_resources(self) -> None:
        server = MCPQueryServer(object())
        supported = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-06-18"},
            }
        )
        future = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "initialize",
                "params": {"protocolVersion": "2099-01-01"},
            }
        )

        self.assertEqual("2025-06-18", supported["result"]["protocolVersion"])
        self.assertEqual("2025-11-25", future["result"]["protocolVersion"])
        self.assertIn("resources", supported["result"]["capabilities"])

    def test_read_only_inventory_is_available_as_an_mcp_resource(self) -> None:
        class FakeControl:
            def query(self, section: str, **_options: object) -> dict[str, object]:
                return {"section": section, "secret_values_included": False}

        server = MCPQueryServer(FakeControl())
        listed = server.handle(
            {"jsonrpc": "2.0", "id": 1, "method": "resources/list", "params": {}}
        )
        uri = listed["result"]["resources"][0]["uri"]
        read = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "resources/read",
                "params": {"uri": uri},
            }
        )

        self.assertTrue(uri.startswith("orchestrator://inventory/"))
        self.assertEqual(uri, read["result"]["contents"][0]["uri"])
        self.assertIn(
            '"secret_values_included":false', read["result"]["contents"][0]["text"]
        )

    def test_mcp_exposes_one_policy_entrypoint_for_configured_providers(self) -> None:
        class FakeTaskRunner:
            def run(self, **kwargs: object) -> dict[str, object]:
                return {"status": "SUCCESS", "received": sorted(kwargs)}

        server = MCPQueryServer(object(), task_runner=FakeTaskRunner())
        listed = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        names = {tool["name"] for tool in listed["result"]["tools"]}
        self.assertIn("orchestrator_run_task", names)

        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "orchestrator_run_task",
                    "arguments": {
                        "spec_path": "/tmp/task.json",
                        "profile": "default",
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
                admission=AdmissionController(Path(directory) / "admission.sqlite"),
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

    def test_oversized_output_is_recorded_as_failure_not_success(self) -> None:
        class FakeTelemetry:
            last_run: RunResult | None = None

            def record_route(self, _task: object, _decision: object) -> None:
                return None

            def record_run(self, run: RunResult) -> None:
                self.last_run = run

        telemetry = FakeTelemetry()

        def fake_execute(task: object, decision: object, **_kwargs: object):
            output = b"x" * 1_048_577
            return (
                RunResult(
                    schema_version=1,
                    run_id="00000000-0000-0000-0000-000000000004",
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
                    output_bytes=len(output),
                    exit_code=0,
                    provenance_verified=True,
                ),
                output,
            )

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repo"
            repo.mkdir()
            (repo / "prompt.txt").write_text("review", encoding="utf-8")
            spec = Path(directory) / "task.json"
            spec.write_text(
                json.dumps(
                    {
                        "repo": str(repo),
                        "task_type": "review",
                        "sensitivity": "internal",
                        "prompt_file": "prompt.txt",
                        "write_mode": "read_only",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ContractError, "1 MB"):
                OrchestratorTaskRunner(
                    configuration_root=ROOT / "config",
                    executor=fake_execute,
                    telemetry=telemetry,
                    admission=AdmissionController(Path(directory) / "admission.sqlite"),
                ).run(
                    spec_path=str(spec),
                    profile="default",
                    model="gpt-5.6-terra",
                )

        self.assertIsNotNone(telemetry.last_run)
        self.assertEqual("OUTPUT_LIMIT_EXCEEDED", telemetry.last_run.status)
        self.assertEqual("OUTPUT_LIMIT_EXCEEDED", telemetry.last_run.error_code)


if __name__ == "__main__":
    unittest.main()
