from __future__ import annotations

import argparse
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

from ai_orchestrator import cli


class _FakeRunner:
    instances: ClassVar[list[_FakeRunner]] = []

    def __init__(self, **kwargs: object) -> None:
        self.init_kwargs = kwargs
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.__class__.instances.append(self)

    def simulate(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("simulate", kwargs))
        return {
            "schema_version": 2,
            "status": "READY",
            "plan_id": "plan-stable",
            "selected_model": "claude-opus",
        }

    def run(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("run", kwargs))
        return {
            "schema_version": 2,
            "status": "SUCCESS",
            "final_accepted": True,
            "output": "bounded answer",
        }


class CliRoutingV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeRunner.instances.clear()

    @patch("ai_orchestrator.cli.OrchestratorTaskRunner", _FakeRunner)
    def test_route_plan_and_run_use_the_shared_runner(self) -> None:
        route_args = argparse.Namespace(
            spec="task.json", profile="default", model="claude-opus"
        )
        run_args = argparse.Namespace(
            spec="task.json",
            profile="default",
            model="claude-opus",
            confirmed_plan_id="plan-stable",
            output=None,
        )

        route_stdout = io.StringIO()
        with redirect_stdout(route_stdout):
            self.assertEqual(0, cli.command_route_plan(route_args))
        run_stdout = io.StringIO()
        with redirect_stdout(run_stdout):
            self.assertEqual(0, cli.command_run(run_args))

        route_runner, run_runner = _FakeRunner.instances
        self.assertEqual("simulate", route_runner.calls[0][0])
        self.assertEqual("run", run_runner.calls[0][0])
        self.assertEqual(
            "plan-stable", run_runner.calls[0][1]["confirmed_plan_id"]
        )
        self.assertTrue(run_runner.init_kwargs["allow_validated_apply"])
        self.assertEqual("plan-stable", json.loads(route_stdout.getvalue())["plan_id"])
        self.assertTrue(json.loads(run_stdout.getvalue())["final_accepted"])

    def test_run_parser_accepts_confirmed_plan_id(self) -> None:
        args = cli.build_parser().parse_args(
            [
                "run",
                "--spec",
                "task.json",
                "--confirmed-plan-id",
                "plan-stable",
            ]
        )
        self.assertEqual("plan-stable", args.confirmed_plan_id)

    def test_promotion_assessment_parser_requires_evidence_files(self) -> None:
        args = cli.build_parser().parse_args(
            [
                "promotion-assess",
                "--model",
                "deepseek-v4-flash",
                "--provider",
                "deepseek",
                "--containment",
                "containment.json",
                "--evaluations",
                "matrix.json",
                "--target-stage",
                "patch_writer",
            ]
        )
        self.assertEqual("patch_writer", args.target_stage)

    def test_harness_status_fails_when_active_slot_is_not_serving(self) -> None:
        registry = {
            "components": {
                "deepseek-harness": {"active": {"version": "0.1.1-rc.2"}}
            }
        }
        stdout = io.StringIO()
        with (
            patch.object(cli.UpdateRegistry, "inventory", return_value=registry),
            patch.object(
                cli,
                "_harness_health",
                return_value={"reachable": False, "status": None},
                create=True,
            ),
            redirect_stdout(stdout),
        ):
            result = cli.command_harness_status(argparse.Namespace(port=3080))

        payload = json.loads(stdout.getvalue())
        self.assertEqual(cli.EXIT_POLICY, result)
        self.assertFalse(payload["ready"])
        self.assertFalse(payload["service"]["reachable"])

    def test_doctor_readiness_requires_live_harness(self) -> None:
        registry = {
            "components": {
                "deepseek-harness": {"active": {"version": "0.1.1-rc.2"}}
            }
        }
        stdout = io.StringIO()
        with (
            patch.object(cli, "validate_configuration", return_value={"ok": True, "errors": []}),
            patch.object(cli.shutil, "which", return_value="/bin/tool"),
            patch.object(cli, "audit_storage", return_value={"ok": True}),
            patch.object(cli, "config_sync_status", return_value={"in_sync": True}),
            patch.object(cli, "_credential_readiness", return_value={"ok": True}),
            patch.object(cli, "validate_provider_catalog", return_value={"errors": []}),
            patch.object(cli.UpdateRegistry, "inventory", return_value=registry),
            patch.object(cli, "_local_health", return_value={"reachable": True, "status": 200}),
            patch.object(cli, "default_socket_path", return_value=Path("/tmp")),
            patch.object(
                cli,
                "_harness_health",
                return_value={"reachable": False, "status": None},
                create=True,
            ),
            redirect_stdout(stdout),
        ):
            result = cli.command_doctor(
                argparse.Namespace(profile="default", check="readiness")
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(cli.EXIT_POLICY, result)
        self.assertFalse(payload["readiness"]["ok"])
        self.assertFalse(payload["readiness"]["harness_service"])


if __name__ == "__main__":
    unittest.main()
