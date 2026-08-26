from __future__ import annotations

import argparse
import io
import json
import unittest
from contextlib import redirect_stdout
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


if __name__ == "__main__":
    unittest.main()
