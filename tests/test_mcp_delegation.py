from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

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

    # --- output_path: la salida delegada tenia que poder sobrevivir a la corrida ---

    @staticmethod
    def _server_and_spec(temporary: str):
        """Un servidor con runner falso y un spec valido en su propio repo."""

        class FakeTaskRunner:
            seen: ClassVar[dict] = {}

            def run(self, **kwargs: object) -> dict[str, object]:
                FakeTaskRunner.seen = dict(kwargs)
                return {"status": "SUCCESS"}

        repo = Path(temporary) / "repo"
        repo.mkdir()
        spec = repo / "task.json"
        spec.write_text(
            json.dumps(
                {
                    "repo": str(repo),
                    "task_type": "research",
                    "sensitivity": "public",
                    "write_mode": "read_only",
                }
            )
        )
        server = MCPQueryServer(
            object(), task_runner=FakeTaskRunner(), fixed_profile="default"
        )
        return server, repo, spec, FakeTaskRunner

    def _call(self, server, spec, **extra):
        return server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "orchestrator_submit_task",
                    "arguments": {"spec_path": str(spec), **extra},
                },
            }
        )

    def test_output_path_reaches_the_runner_so_the_delegated_report_survives(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            server, _repo, spec, runner = self._server_and_spec(temporary)
            staging = Path(temporary) / "staging"
            staging.mkdir()
            destination = staging / "informe.md"

            response = self._call(server, spec, output_path=str(destination))

        self.assertFalse(response["result"]["isError"])
        self.assertEqual(destination.resolve(), runner.seen["output_path"])

    def test_a_destination_inside_the_repository_fails_before_the_model_runs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            server, repo, spec, runner = self._server_and_spec(temporary)
            runner.seen = {}

            response = self._call(
                server, spec, output_path=str(repo / "docs" / "informe.md")
            )

        payload = json.loads(response["result"]["content"][0]["text"])
        self.assertTrue(response["result"]["isError"])
        self.assertIn("outside the task repository", payload["message"])
        # Lo que se cobra: el runner no llego a arrancar.
        self.assertEqual({}, runner.seen)

    def test_a_symlinked_destination_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            server, _repo, spec, _runner = self._server_and_spec(temporary)
            staging = Path(temporary) / "staging"
            staging.mkdir()
            real = staging / "real.md"
            real.write_text("x")
            link = staging / "enlace.md"
            link.symlink_to(real)

            response = self._call(server, spec, output_path=str(link))

        payload = json.loads(response["result"]["content"][0]["text"])
        self.assertTrue(response["result"]["isError"])
        self.assertIn("symlink", payload["message"])

    def test_omitting_output_path_keeps_the_previous_call_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            server, _repo, spec, runner = self._server_and_spec(temporary)

            response = self._call(server, spec)

        self.assertFalse(response["result"]["isError"])
        self.assertNotIn("output_path", runner.seen)

    # --- rescate: una entrega no aceptada tambien se pago, y hay que poder leerla ---

    @staticmethod
    def _runner_with_status(status: str, body: bytes, directory: str):
        """Un runner cuyo modelo devuelve `status` y `body`, sin red ni cuota."""

        class FakeTelemetry:
            def record_route(self, _task: object, _decision: object) -> None:
                return None

            def record_run(self, _run: object) -> None:
                return None

        def fake_execute(task: object, decision: object, **_kwargs: object):
            result = RunResult(
                schema_version=1,
                run_id="00000000-0000-0000-0000-000000000003",
                decision_id=decision.decision_id,
                task_id=task.task_id,
                status=status,
                requested_model=decision.selected_model,
                reported_model=decision.selected_model,
                provider=decision.provider,
                transport=decision.transport,
                started_at="2026-08-26T00:00:00+00:00",
                finished_at="2026-08-26T00:00:01+00:00",
                latency_ms=1,
                output_sha256="0" * 64,
                output_bytes=len(body),
                exit_code=0,
                provenance_verified=True,
            )
            return result, body

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
        runner = OrchestratorTaskRunner(
            configuration_root=ROOT / "config",
            executor=fake_execute,
            telemetry=FakeTelemetry(),
        )
        return runner, spec

    def test_an_unaccepted_delivery_still_leaves_its_output_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner, spec = self._runner_with_status(
                "PROVIDER_ERROR", b"el informe que se pago", directory
            )
            staging = Path(directory) / "staging"
            staging.mkdir()
            destination = staging / "informe.md"

            receipt = runner.run(
                spec_path=str(spec),
                profile="default",
                model="gpt-5.6-terra",
                output_path=destination,
            )

            self.assertFalse(receipt["final_accepted"])
            rescued = Path(receipt["rescued_output_path"])
            self.assertEqual("el informe que se pago", rescued.read_text())
            # El nombre canta el estado: nunca se confunde con lo aceptado.
            self.assertIn(receipt["status"], rescued.name)
            self.assertFalse(destination.exists())

    def test_nothing_is_rescued_when_the_model_produced_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner, spec = self._runner_with_status("PROVIDER_ERROR", b"", directory)
            staging = Path(directory) / "staging"
            staging.mkdir()

            receipt = runner.run(
                spec_path=str(spec),
                profile="default",
                model="gpt-5.6-terra",
                output_path=staging / "informe.md",
            )

            self.assertNotIn("rescued_output_path", receipt)
            self.assertEqual([], list(staging.iterdir()))

    def test_without_output_path_a_failed_delivery_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner, spec = self._runner_with_status(
                "PROVIDER_ERROR", b"salida", directory
            )

            receipt = runner.run(
                spec_path=str(spec), profile="default", model="gpt-5.6-terra"
            )

        self.assertNotIn("rescued_output_path", receipt)

    def test_the_tool_schema_advertises_output_path(self) -> None:
        server = MCPQueryServer(object(), fixed_profile="default")
        listed = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        submit = next(
            tool
            for tool in listed["result"]["tools"]
            if tool["name"] == "orchestrator_submit_task"
        )
        self.assertIn("output_path", submit["inputSchema"]["properties"])


if __name__ == "__main__":
    unittest.main()
