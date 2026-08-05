from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from ai_orchestrator.models import RouteDecision, RunResult, TaskSpec
from ai_orchestrator.telemetry import TelemetryStore


class TelemetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.previous = os.environ.get("AI_ORCHESTRATOR_STATE")
        os.environ["AI_ORCHESTRATOR_STATE"] = str(self.root / "state")
        self.repo = self.root / "repo"
        self.repo.mkdir()
        (self.repo / "prompt.txt").write_text("secret prompt body", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()
        if self.previous is None:
            os.environ.pop("AI_ORCHESTRATOR_STATE", None)
        else:
            os.environ["AI_ORCHESTRATOR_STATE"] = self.previous

    def test_store_contains_metadata_but_never_prompt_or_response_bodies(self) -> None:
        task = TaskSpec.from_mapping(
            {
                "repo": str(self.repo),
                "task_type": "docs",
                "sensitivity": "internal",
                "prompt_file": "prompt.txt",
            }
        )
        decision = RouteDecision(
            schema_version=1,
            decision_id="d1",
            task_id=task.task_id,
            profile="default",
            status="READY",
            selected_model="deepseek-v4-flash",
            requested_route_model="deepseek/deepseek-v4-flash",
            provider="deepseek",
            family="deepseek",
            transport="omniroute",
            reviewer_model="gpt-5.6-sol",
            alternatives=(),
            unavailable={},
            auto_execute=True,
            approval_required=False,
            reason="eligible",
            context_sha256=task.context_sha256(),
        )
        result = RunResult(
            schema_version=1,
            run_id="r1",
            decision_id="d1",
            task_id=task.task_id,
            status="SUCCESS",
            requested_model="deepseek-v4-flash",
            reported_model="deepseek-v4-flash",
            provider="deepseek",
            transport="omniroute",
            started_at="2026-08-05T00:00:00+00:00",
            finished_at="2026-08-05T00:00:01+00:00",
            latency_ms=1000,
            output_sha256="f" * 64,
            output_bytes=18,
            exit_code=0,
            provenance_verified=True,
        )
        store = TelemetryStore()
        store.record_route(task, decision)
        store.record_run(result)

        raw = store.path.read_bytes()
        self.assertNotIn(b"secret prompt body", raw)
        self.assertNotIn(b"model response body", raw)
        self.assertEqual(0o600, store.path.stat().st_mode & 0o777)
        self.assertEqual(0o700, store.path.parent.stat().st_mode & 0o777)

        with store._connect() as connection:
            self.assertEqual("wal", connection.execute("PRAGMA journal_mode").fetchone()[0])
            self.assertEqual(2, connection.execute("PRAGMA synchronous").fetchone()[0])
            self.assertEqual(1, connection.execute("PRAGMA foreign_keys").fetchone()[0])
            self.assertEqual(1, connection.execute("PRAGMA secure_delete").fetchone()[0])
            columns = {
                row[1]
                for table in ("route_decisions", "runs")
                for row in connection.execute(f"PRAGMA table_info({table})")
            }
        self.assertFalse({"prompt", "response", "reasoning", "content"} & columns)


if __name__ == "__main__":
    unittest.main()
