from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from ai_orchestrator.models import QuotaSnapshot, UsageRecord
from ai_orchestrator.telemetry import TelemetryStore


class TelemetryV2SymbolsTests(unittest.TestCase):
    def test_quota_snapshot_store_exists(self) -> None:
        self.assertTrue(hasattr(TelemetryStore, "record_quota"))
        self.assertTrue(hasattr(TelemetryStore, "latest_quotas"))
        self.assertTrue(hasattr(TelemetryStore, "rollout_counts"))


@unittest.skipUnless(hasattr(TelemetryStore, "record_quota"), "v2 telemetry missing")
class TelemetryV2Tests(unittest.TestCase):
    def test_unknown_usage_stays_null_instead_of_becoming_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TelemetryStore(Path(directory) / "telemetry.sqlite")
            store.record_usage(
                UsageRecord(
                    provider="anthropic",
                    model="claude-opus",
                    outcome="SUCCESS",
                    latency_ms=10,
                )
            )
            report = store.report()
            self.assertIsNone(report["usage"]["input_tokens"])
            self.assertIsNone(report["usage"]["estimated_cost_usd"])
            self.assertEqual(1, report["usage"]["unknown_input_records"])

    def test_v1_usage_table_migrates_without_losing_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.sqlite"
            connection = sqlite3.connect(path)
            connection.executescript(
                """
                CREATE TABLE usage_records (
                    record_id TEXT PRIMARY KEY,
                    route_plan_id TEXT,
                    created_at TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    cached_tokens INTEGER NOT NULL,
                    credits_used REAL NOT NULL,
                    estimated_cost_usd REAL NOT NULL,
                    cache_hit INTEGER NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    outcome TEXT NOT NULL,
                    provenance_verified INTEGER NOT NULL
                );
                INSERT INTO usage_records VALUES (
                    'old', NULL, '2026-08-20T00:00:00+00:00', 'deepseek', 'flash',
                    10, 2, 3, 0, 0.01, 1, 20, 'SUCCESS', 1
                );
                """
            )
            connection.commit()
            connection.close()

            store = TelemetryStore(path)
            with store._connect() as migrated:
                row = migrated.execute(
                    "SELECT input_tokens, cache_write_tokens, marginal_cost_usd "
                    "FROM usage_records WHERE record_id='old'"
                ).fetchone()
            self.assertEqual((10, None, 0.01), row)

    def test_latest_quota_snapshot_is_dated_and_secret_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TelemetryStore(Path(directory) / "telemetry.sqlite")
            store.record_quota(
                QuotaSnapshot(
                    provider="deepseek",
                    plan="api_balance",
                    billing_mode="prepaid_balance",
                    balance_usd=4.25,
                    remaining=4.25,
                    reset_at=None,
                    source="/user/balance",
                    verified_at="2026-08-20T10:00:00+00:00",
                    confidence="verified",
                    windows={"monthly": 9.0, "weekly": 4.0, "five_hours": 1.0},
                )
            )
            snapshots = store.latest_quotas()
            self.assertEqual(4.25, snapshots["deepseek"].balance_usd)
            self.assertEqual("/user/balance", snapshots["deepseek"].source)
            self.assertIn("five_hours", snapshots["deepseek"].windows)
            self.assertEqual(1.0, snapshots["deepseek"].windows["five_hours"])
            self.assertFalse(
                {"token", "api_key", "secret"}
                & set(snapshots["deepseek"].as_dict())
            )


if __name__ == "__main__":
    unittest.main()
