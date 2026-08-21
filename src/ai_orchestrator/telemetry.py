from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ai_orchestrator.models import (
    RouteDecision,
    RunResult,
    TaskSpec,
    UsageRecord,
    ensure_private_directory,
)


def default_state_root() -> Path:
    override = os.environ.get("AI_ORCHESTRATOR_STATE")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".local" / "share" / "ai-orchestrator"


class TelemetryStore:
    """Metadata-only SQLite store. Prompt and response bodies are never accepted."""

    def __init__(self, path: Path | None = None) -> None:
        root = path.parent if path is not None else default_state_root()
        ensure_private_directory(root)
        self.path = path or root / "telemetry.sqlite"
        if self.path.is_symlink():
            raise ValueError("telemetry database must not be a symlink")
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA secure_delete=ON")
        connection.execute("PRAGMA auto_vacuum=INCREMENTAL")
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA wal_autocheckpoint=1000")
        connection.execute("PRAGMA journal_size_limit=67108864")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS route_decisions (
                    decision_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    repo_hash TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    sensitivity TEXT NOT NULL,
                    context_sha256 TEXT NOT NULL,
                    selected_model TEXT,
                    provider TEXT,
                    transport TEXT,
                    reviewer_model TEXT,
                    status TEXT NOT NULL,
                    auto_execute INTEGER NOT NULL CHECK (auto_execute IN (0, 1)),
                    reason_code TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_route_created
                    ON route_decisions(created_at);
                CREATE INDEX IF NOT EXISTS idx_route_role_model
                    ON route_decisions(task_type, selected_model);

                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    decision_id TEXT NOT NULL REFERENCES route_decisions(decision_id),
                    task_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL,
                    requested_model TEXT NOT NULL,
                    reported_model TEXT,
                    provider TEXT NOT NULL,
                    transport TEXT NOT NULL,
                    status TEXT NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    output_sha256 TEXT,
                    output_bytes INTEGER NOT NULL,
                    exit_code INTEGER,
                    provenance_verified INTEGER NOT NULL
                        CHECK (provenance_verified IN (0, 1)),
                    error_code TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_run_finished
                    ON runs(finished_at);
                CREATE INDEX IF NOT EXISTS idx_run_model_status
                    ON runs(requested_model, status);

                CREATE TABLE IF NOT EXISTS usage_records (
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
                    cache_hit INTEGER NOT NULL CHECK (cache_hit IN (0, 1)),
                    latency_ms INTEGER NOT NULL,
                    outcome TEXT NOT NULL,
                    provenance_verified INTEGER NOT NULL
                        CHECK (provenance_verified IN (0, 1))
                );
                CREATE INDEX IF NOT EXISTS idx_usage_created
                    ON usage_records(created_at);
                CREATE INDEX IF NOT EXISTS idx_usage_provider_model
                    ON usage_records(provider, model);
                """
            )
        os.chmod(self.path, 0o600)
        for suffix in ("-wal", "-shm"):
            candidate = Path(f"{self.path}{suffix}")
            if candidate.exists() and not candidate.is_symlink():
                os.chmod(candidate, 0o600)

    def record_route(self, task: TaskSpec, decision: RouteDecision) -> None:
        import hashlib

        repo_hash = hashlib.sha256(task.repo.encode("utf-8")).hexdigest()
        created_at = datetime.now(UTC).isoformat()
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO route_decisions (
                    decision_id, task_id, created_at, repo_hash, profile,
                    task_type, sensitivity, context_sha256, selected_model,
                    provider, transport, reviewer_model, status, auto_execute,
                    reason_code
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision.decision_id,
                    task.task_id,
                    created_at,
                    repo_hash,
                    decision.profile,
                    task.task_type,
                    task.sensitivity,
                    decision.context_sha256,
                    decision.selected_model,
                    decision.provider,
                    decision.transport,
                    decision.reviewer_model,
                    decision.status,
                    int(decision.auto_execute),
                    decision.reason,
                ),
            )

    def record_run(self, result: RunResult) -> None:
        usage = UsageRecord(
            provider=result.provider,
            model=result.requested_model,
            outcome=result.status,
            latency_ms=result.latency_ms,
            provenance_verified=result.provenance_verified,
            route_plan_id=result.decision_id,
        )
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO runs (
                    run_id, decision_id, task_id, started_at, finished_at,
                    requested_model, reported_model, provider, transport,
                    status, latency_ms, output_sha256, output_bytes, exit_code,
                    provenance_verified, error_code
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.run_id,
                    result.decision_id,
                    result.task_id,
                    result.started_at,
                    result.finished_at,
                    result.requested_model,
                    result.reported_model,
                    result.provider,
                    result.transport,
                    result.status,
                    result.latency_ms,
                    result.output_sha256,
                    result.output_bytes,
                    result.exit_code,
                    int(result.provenance_verified),
                    result.error_code,
                ),
            )
            connection.execute(
                """
                INSERT INTO usage_records (
                    record_id, route_plan_id, created_at, provider, model,
                    input_tokens, output_tokens, cached_tokens, credits_used,
                    estimated_cost_usd, cache_hit, latency_ms, outcome,
                    provenance_verified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    usage.record_id,
                    usage.route_plan_id,
                    usage.created_at,
                    usage.provider,
                    usage.model,
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.cached_tokens,
                    usage.credits_used,
                    usage.estimated_cost_usd,
                    int(usage.cache_hit),
                    usage.latency_ms,
                    usage.outcome,
                    int(usage.provenance_verified),
                ),
            )

    def record_usage(self, record: UsageRecord) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO usage_records (
                    record_id, route_plan_id, created_at, provider, model,
                    input_tokens, output_tokens, cached_tokens, credits_used,
                    estimated_cost_usd, cache_hit, latency_ms, outcome,
                    provenance_verified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.record_id,
                    record.route_plan_id,
                    record.created_at,
                    record.provider,
                    record.model,
                    record.input_tokens,
                    record.output_tokens,
                    record.cached_tokens,
                    record.credits_used,
                    record.estimated_cost_usd,
                    int(record.cache_hit),
                    record.latency_ms,
                    record.outcome,
                    int(record.provenance_verified),
                ),
            )

    def routing_observations(
        self, sample_limit: int = 1000
    ) -> dict[str, dict[str, object]]:
        """Return bounded aggregate metadata suitable for deterministic routing."""

        if not 1 <= sample_limit <= 100_000:
            raise ValueError("routing observation sample limit must be 1..100000")
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                """
                WITH recent AS (
                    SELECT model, latency_ms, cache_hit, outcome
                    FROM usage_records
                    ORDER BY created_at DESC
                    LIMIT ?
                )
                SELECT model,
                       ROUND(AVG(latency_ms), 3),
                       ROUND(AVG(cache_hit), 6),
                       COUNT(*),
                       ROUND(AVG(CASE WHEN outcome IN
                           ('SUCCESS', 'APPLIED', 'PATCH_READY')
                           THEN 1.0 ELSE 0.0 END), 6)
                FROM recent
                GROUP BY model
                ORDER BY model
                """,
                (sample_limit,),
            ).fetchall()
        return {
            str(row[0]): {
                "latency_ms": float(row[1]),
                "cache_hit_rate": float(row[2]),
                "sample_count": int(row[3]),
                "success_rate": float(row[4]),
            }
            for row in rows
        }

    def report(self) -> dict[str, Any]:
        with closing(self._connect()) as connection, connection:
            routes = connection.execute(
                "SELECT COUNT(*) FROM route_decisions"
            ).fetchone()[0]
            runs = connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            usage_records = connection.execute(
                "SELECT COUNT(*) FROM usage_records"
            ).fetchone()[0]
            rows = connection.execute(
                """
                SELECT requested_model, COUNT(*) AS total,
                       SUM(CASE WHEN status='SUCCESS' THEN 1 ELSE 0 END) AS ok,
                       ROUND(AVG(latency_ms), 1) AS avg_latency_ms,
                       SUM(output_bytes) AS output_bytes
                FROM runs GROUP BY requested_model ORDER BY requested_model
                """
            ).fetchall()
            usage = connection.execute(
                """
                SELECT COALESCE(SUM(input_tokens), 0),
                       COALESCE(SUM(output_tokens), 0),
                       COALESCE(SUM(cached_tokens), 0),
                       COALESCE(SUM(credits_used), 0),
                       COALESCE(SUM(estimated_cost_usd), 0)
                FROM usage_records
                """
            ).fetchone()
        return {
            "schema_version": 1,
            "route_count": routes,
            "run_count": runs,
            "usage_record_count": usage_records,
            "usage": {
                "input_tokens": int(usage[0]),
                "output_tokens": int(usage[1]),
                "cached_tokens": int(usage[2]),
                "credits_used": float(usage[3]),
                "estimated_cost_usd": float(usage[4]),
            },
            "models": [
                {
                    "model": row[0],
                    "runs": row[1],
                    "successes": row[2],
                    "average_latency_ms": row[3],
                    "output_bytes": row[4],
                }
                for row in rows
            ],
        }

    def purge_metadata(self, retention_days: int = 90) -> dict[str, int]:
        preview = self.preview_metadata_purge(retention_days)
        cutoff = preview["cutoff"]
        with closing(self._connect()) as connection, connection:
            deleted_runs = connection.execute(
                "DELETE FROM runs WHERE finished_at < ?", (cutoff,)
            ).rowcount
            deleted_usage = connection.execute(
                "DELETE FROM usage_records WHERE created_at < ?", (cutoff,)
            ).rowcount
            deleted_routes = connection.execute(
                """
                DELETE FROM route_decisions
                WHERE created_at < ?
                  AND decision_id NOT IN (SELECT decision_id FROM runs)
                """,
                (cutoff,),
            ).rowcount
            connection.execute("PRAGMA incremental_vacuum")
            connection.execute("PRAGMA optimize")
        return {"runs": deleted_runs, "routes": deleted_routes, "usage": deleted_usage}

    def preview_metadata_purge(self, retention_days: int = 90) -> dict[str, Any]:
        if not 30 <= retention_days <= 90:
            raise ValueError("metadata retention must be between 30 and 90 days")
        cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
        with closing(self._connect()) as connection, connection:
            runs = connection.execute(
                "SELECT COUNT(*) FROM runs WHERE finished_at < ?", (cutoff,)
            ).fetchone()[0]
            routes = connection.execute(
                """
                SELECT COUNT(*) FROM route_decisions
                WHERE created_at < ?
                  AND decision_id NOT IN (SELECT decision_id FROM runs)
                """,
                (cutoff,),
            ).fetchone()[0]
            usage = connection.execute(
                "SELECT COUNT(*) FROM usage_records WHERE created_at < ?", (cutoff,)
            ).fetchone()[0]
        return {
            "runs": int(runs),
            "routes": int(routes),
            "usage": int(usage),
            "cutoff": cutoff,
            "dry_run": True,
        }
