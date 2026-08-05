from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ai_orchestrator.models import RouteDecision, RunResult, TaskSpec, ensure_private_directory


def default_state_root() -> Path:
    override = os.environ.get("AI_ORCHESTRATOR_STATE")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".local" / "share" / "ai-orchestrator"


class TelemetryStore:
    """Metadata-only SQLite store. Prompt and response bodies are never accepted."""

    def __init__(self, path: Path | None = None) -> None:
        root = default_state_root()
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
        with self._connect() as connection:
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
        with self._connect() as connection:
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
        with self._connect() as connection:
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

    def report(self) -> dict[str, Any]:
        with self._connect() as connection:
            routes = connection.execute("SELECT COUNT(*) FROM route_decisions").fetchone()[0]
            runs = connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            rows = connection.execute(
                """
                SELECT requested_model, COUNT(*) AS total,
                       SUM(CASE WHEN status='SUCCESS' THEN 1 ELSE 0 END) AS ok,
                       ROUND(AVG(latency_ms), 1) AS avg_latency_ms,
                       SUM(output_bytes) AS output_bytes
                FROM runs GROUP BY requested_model ORDER BY requested_model
                """
            ).fetchall()
        return {
            "schema_version": 1,
            "route_count": routes,
            "run_count": runs,
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

    def purge_metadata(self, retention_days: int = 365) -> dict[str, int]:
        preview = self.preview_metadata_purge(retention_days)
        cutoff = preview["cutoff"]
        with self._connect() as connection:
            deleted_runs = connection.execute(
                "DELETE FROM runs WHERE finished_at < ?", (cutoff,)
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
        return {"runs": deleted_runs, "routes": deleted_routes}

    def preview_metadata_purge(self, retention_days: int = 365) -> dict[str, Any]:
        if retention_days < 30:
            raise ValueError("metadata retention must be at least 30 days")
        cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
        with self._connect() as connection:
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
        return {"runs": int(runs), "routes": int(routes), "cutoff": cutoff, "dry_run": True}
