from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ai_orchestrator.models import (
    QuotaSnapshot,
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

    def __init__(self, path: Path | None = None, *, create: bool = True) -> None:
        root = path.parent if path is not None else default_state_root()
        self.path = path or root / "telemetry.sqlite"
        if self.path.is_symlink():
            raise ValueError("telemetry database must not be a symlink")
        self.create = create
        if create:
            ensure_private_directory(root)
            self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if not self.create:
            connection = sqlite3.connect(
                f"file:{self.path}?mode=ro", uri=True, timeout=5.0
            )
            connection.execute("PRAGMA query_only=ON")
            connection.execute("PRAGMA busy_timeout=5000")
        else:
            connection = sqlite3.connect(self.path, timeout=5.0)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA secure_delete=ON")
            connection.execute("PRAGMA auto_vacuum=INCREMENTAL")
            connection.execute("PRAGMA busy_timeout=5000")
            connection.execute("PRAGMA wal_autocheckpoint=1000")
            connection.execute("PRAGMA journal_size_limit=67108864")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

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

                CREATE TABLE IF NOT EXISTS usage_records (
                    record_id TEXT PRIMARY KEY,
                    route_plan_id TEXT,
                    created_at TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    cached_tokens INTEGER,
                    cache_write_tokens INTEGER,
                    credits_used REAL,
                    balance_before REAL,
                    balance_after REAL,
                    marginal_cost_usd REAL,
                    equivalent_cost_usd REAL,
                    estimated_cost_usd REAL,
                    cache_hit INTEGER CHECK (cache_hit IN (0, 1)),
                    latency_ms INTEGER NOT NULL,
                    outcome TEXT NOT NULL,
                    provenance_verified INTEGER NOT NULL
                        CHECK (provenance_verified IN (0, 1))
                );
                CREATE INDEX IF NOT EXISTS idx_usage_created
                    ON usage_records(created_at);
                CREATE INDEX IF NOT EXISTS idx_usage_provider_model
                    ON usage_records(provider, model);

                CREATE TABLE IF NOT EXISTS quota_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    plan TEXT NOT NULL,
                    billing_mode TEXT NOT NULL,
                    remaining REAL,
                    quota_limit REAL,
                    used REAL,
                    predicted_usage REAL,
                    balance_usd REAL,
                    credits_remaining REAL,
                    window TEXT,
                    reset_at TEXT,
                    source TEXT NOT NULL,
                    verified_at TEXT,
                    confidence TEXT NOT NULL,
                    extra_usage_enabled INTEGER
                        CHECK (extra_usage_enabled IN (0, 1) OR extra_usage_enabled IS NULL),
                    windows_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_quota_provider_verified
                    ON quota_snapshots(provider, verified_at DESC);

                CREATE TABLE IF NOT EXISTS model_leases (
                    lease_id TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    acquired_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_model_leases_active
                    ON model_leases(model, expires_at);
                """
            )
            self._migrate_usage_v2(connection)
            self._ensure_column(connection, "route_decisions", "plan_id", "TEXT")
            self._ensure_column(
                connection,
                "route_decisions",
                "policy_version",
                "TEXT NOT NULL DEFAULT 'routing-v1'",
            )
            self._ensure_column(
                connection,
                "route_decisions",
                "simulated",
                "INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(
                connection,
                "quota_snapshots",
                "windows_json",
                "TEXT NOT NULL DEFAULT '{}'",
            )
            self._ensure_column(connection, "runs", "attempt_count", "INTEGER")
            self._ensure_column(
                connection, "runs", "contract_validated", "INTEGER"
            )
            self._ensure_column(connection, "runs", "gates_passed", "INTEGER")
            self._ensure_column(connection, "runs", "reviewer_verdict", "TEXT")
            self._ensure_column(
                connection,
                "runs",
                "final_accepted",
                "INTEGER NOT NULL DEFAULT 0",
            )
            connection.executescript(
                """
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

    @staticmethod
    def _migrate_usage_v2(connection: sqlite3.Connection) -> None:
        columns = connection.execute("PRAGMA table_info(usage_records)").fetchall()
        by_name = {row[1]: row for row in columns}
        if "cache_write_tokens" in by_name and not by_name["input_tokens"][3]:
            return
        connection.executescript(
            """
            DROP INDEX IF EXISTS idx_usage_created;
            DROP INDEX IF EXISTS idx_usage_provider_model;
            ALTER TABLE usage_records RENAME TO usage_records_v1;
            CREATE TABLE usage_records (
                record_id TEXT PRIMARY KEY,
                route_plan_id TEXT,
                created_at TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cached_tokens INTEGER,
                cache_write_tokens INTEGER,
                credits_used REAL,
                balance_before REAL,
                balance_after REAL,
                marginal_cost_usd REAL,
                equivalent_cost_usd REAL,
                estimated_cost_usd REAL,
                cache_hit INTEGER CHECK (cache_hit IN (0, 1)),
                latency_ms INTEGER NOT NULL,
                outcome TEXT NOT NULL,
                provenance_verified INTEGER NOT NULL
                    CHECK (provenance_verified IN (0, 1))
            );
            INSERT INTO usage_records (
                record_id, route_plan_id, created_at, provider, model,
                input_tokens, output_tokens, cached_tokens, cache_write_tokens,
                credits_used, balance_before, balance_after, marginal_cost_usd,
                equivalent_cost_usd, estimated_cost_usd, cache_hit, latency_ms,
                outcome, provenance_verified
            )
            SELECT record_id, route_plan_id, created_at, provider, model,
                   input_tokens, output_tokens, cached_tokens, NULL,
                   credits_used, NULL, NULL, estimated_cost_usd,
                   NULL, estimated_cost_usd, cache_hit, latency_ms,
                   outcome, provenance_verified
            FROM usage_records_v1;
            DROP TABLE usage_records_v1;
            """
        )

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection, table: str, column: str, declaration: str
    ) -> None:
        names = {
            row[1] for row in connection.execute(f"PRAGMA table_info({table})")
        }
        if column not in names:
            connection.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
            )

    def acquire_lease(
        self,
        model: str,
        *,
        max_concurrency: int,
        ttl_seconds: int,
        now: float | None = None,
    ) -> str | None:
        """Atomically reserve one model slot across orchestrator processes."""

        if not model or max_concurrency < 1 or ttl_seconds < 1:
            raise ValueError("invalid model lease request")
        current = time.time() if now is None else float(now)
        lease_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM model_leases WHERE expires_at <= ?", (current,)
            )
            active = connection.execute(
                "SELECT COUNT(*) FROM model_leases WHERE model=? AND expires_at > ?",
                (model, current),
            ).fetchone()[0]
            if active >= max_concurrency:
                return None
            connection.execute(
                """
                INSERT INTO model_leases (lease_id, model, acquired_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (lease_id, model, current, current + ttl_seconds),
            )
        return lease_id

    def release_lease(self, lease_id: str) -> bool:
        if not lease_id:
            return False
        with self._connect() as connection:
            deleted = connection.execute(
                "DELETE FROM model_leases WHERE lease_id=?", (lease_id,)
            ).rowcount
        return bool(deleted)

    def record_route(
        self,
        task: TaskSpec,
        decision: RouteDecision,
        *,
        simulated: bool = False,
    ) -> None:
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
                    reason_code, plan_id, policy_version, simulated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    decision.plan_id,
                    decision.policy_version,
                    int(simulated),
                ),
            )

    def rollout_counts(self) -> dict[str, int]:
        """Count shadow simulations per selected model without storing content."""

        if not self.path.exists() and not self.create:
            return {}
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT selected_model, COUNT(*)
                FROM route_decisions
                WHERE simulated=1 AND selected_model IS NOT NULL
                GROUP BY selected_model
                """
            ).fetchall()
        return {str(model): int(count) for model, count in rows}

    def daily_provider_runs(self, provider: str, *, at: datetime | None = None) -> int:
        """Count provider executions since local midnight for shared daily limits."""

        if not self.path.exists() and not self.create:
            return 0
        clock = (at or datetime.now().astimezone()).astimezone()
        start = clock.replace(hour=0, minute=0, second=0, microsecond=0)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM runs WHERE provider=? AND finished_at>=?",
                (provider, start.astimezone(UTC).isoformat()),
            ).fetchone()
        return int(row[0])

    def record_run(self, result: RunResult) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    run_id, decision_id, task_id, started_at, finished_at,
                    requested_model, reported_model, provider, transport,
                    status, latency_ms, output_sha256, output_bytes, exit_code,
                    provenance_verified, error_code, attempt_count,
                    contract_validated, gates_passed, reviewer_verdict,
                    final_accepted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    len(result.attempts),
                    (
                        int(result.contract_validated)
                        if result.contract_validated is not None
                        else None
                    ),
                    int(result.gates_passed) if result.gates_passed is not None else None,
                    (
                        str(result.reviewer_verdict.get("verdict"))
                        if isinstance(result.reviewer_verdict, dict)
                        and result.reviewer_verdict.get("verdict") is not None
                        else None
                    ),
                    int(result.final_accepted),
                ),
            )

    def finalize_run(self, result: RunResult) -> None:
        """Attach aggregate acceptance metadata without retaining review content."""

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE runs
                SET attempt_count=?, contract_validated=?, gates_passed=?,
                    reviewer_verdict=?, final_accepted=?
                WHERE run_id=?
                """,
                (
                    len(result.attempts),
                    (
                        int(result.contract_validated)
                        if result.contract_validated is not None
                        else None
                    ),
                    int(result.gates_passed) if result.gates_passed is not None else None,
                    (
                        str(result.reviewer_verdict.get("verdict"))
                        if isinstance(result.reviewer_verdict, dict)
                        and result.reviewer_verdict.get("verdict") is not None
                        else None
                    ),
                    int(result.final_accepted),
                    result.run_id,
                ),
            )

    def record_usage(self, record: UsageRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO usage_records (
                    record_id, route_plan_id, created_at, provider, model,
                    input_tokens, output_tokens, cached_tokens, cache_write_tokens,
                    credits_used, balance_before, balance_after, marginal_cost_usd,
                    equivalent_cost_usd, estimated_cost_usd, cache_hit, latency_ms,
                    outcome, provenance_verified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    record.cache_write_tokens,
                    record.credits_used,
                    record.balance_before,
                    record.balance_after,
                    record.marginal_cost_usd,
                    record.equivalent_cost_usd,
                    record.estimated_cost_usd,
                    int(record.cache_hit) if record.cache_hit is not None else None,
                    record.latency_ms,
                    record.outcome,
                    int(record.provenance_verified),
                ),
            )

    def record_quota(self, snapshot: QuotaSnapshot) -> str:
        snapshot_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO quota_snapshots (
                    snapshot_id, provider, plan, billing_mode, remaining,
                    quota_limit, used, predicted_usage, balance_usd,
                    credits_remaining, window, reset_at, source, verified_at,
                    confidence, extra_usage_enabled
                    , windows_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    snapshot.provider,
                    snapshot.plan,
                    snapshot.billing_mode,
                    snapshot.remaining,
                    snapshot.limit,
                    snapshot.used,
                    snapshot.predicted_usage,
                    snapshot.balance_usd,
                    snapshot.credits_remaining,
                    snapshot.window,
                    snapshot.reset_at,
                    snapshot.source,
                    snapshot.verified_at,
                    snapshot.confidence,
                    (
                        int(snapshot.extra_usage_enabled)
                        if snapshot.extra_usage_enabled is not None
                        else None
                    ),
                    json.dumps(
                        snapshot.windows, sort_keys=True, separators=(",", ":")
                    ),
                ),
            )
        return snapshot_id

    def latest_quotas(self) -> dict[str, QuotaSnapshot]:
        if not self.path.exists() and not self.create:
            return {}
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT provider, plan, billing_mode, remaining, quota_limit,
                       used, predicted_usage, balance_usd, credits_remaining,
                       window, reset_at, source, verified_at, confidence,
                       extra_usage_enabled, windows_json
                FROM quota_snapshots AS snapshots
                WHERE snapshot_id = (
                    SELECT candidate.snapshot_id
                    FROM quota_snapshots AS candidate
                    WHERE candidate.provider = snapshots.provider
                    ORDER BY candidate.verified_at DESC, candidate.rowid DESC
                    LIMIT 1
                )
                ORDER BY provider
                """
            ).fetchall()
        return {
            row[0]: QuotaSnapshot(
                provider=row[0],
                plan=row[1],
                billing_mode=row[2],
                remaining=row[3],
                limit=row[4],
                used=row[5],
                predicted_usage=row[6],
                balance_usd=row[7],
                credits_remaining=row[8],
                window=row[9],
                reset_at=row[10],
                source=row[11],
                verified_at=row[12],
                confidence=row[13],
                extra_usage_enabled=(
                    bool(row[14]) if row[14] is not None else None
                ),
                windows=(
                    json.loads(row[15])
                    if isinstance(row[15], str) and row[15]
                    else {}
                ),
            )
            for row in rows
        }

    def report(self) -> dict[str, Any]:
        if not self.path.exists() and not self.create:
            return {
                "schema_version": 1,
                "route_count": 0,
                "run_count": 0,
                "usage_record_count": 0,
                "usage": {},
                "models": [],
            }
        with self._connect() as connection:
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
                SELECT SUM(input_tokens), SUM(output_tokens), SUM(cached_tokens),
                       SUM(cache_write_tokens), SUM(credits_used),
                       SUM(marginal_cost_usd), SUM(equivalent_cost_usd),
                       SUM(estimated_cost_usd),
                       SUM(CASE WHEN input_tokens IS NULL THEN 1 ELSE 0 END),
                       SUM(CASE WHEN estimated_cost_usd IS NULL THEN 1 ELSE 0 END)
                FROM usage_records
                """
            ).fetchone()
        return {
            "schema_version": 1,
            "route_count": routes,
            "run_count": runs,
            "usage_record_count": usage_records,
            "usage": {
                "input_tokens": int(usage[0]) if usage[0] is not None else None,
                "output_tokens": int(usage[1]) if usage[1] is not None else None,
                "cached_tokens": int(usage[2]) if usage[2] is not None else None,
                "cache_write_tokens": int(usage[3]) if usage[3] is not None else None,
                "credits_used": float(usage[4]) if usage[4] is not None else None,
                "marginal_cost_usd": float(usage[5]) if usage[5] is not None else None,
                "equivalent_cost_usd": float(usage[6]) if usage[6] is not None else None,
                "estimated_cost_usd": float(usage[7]) if usage[7] is not None else None,
                "unknown_input_records": int(usage[8] or 0),
                "unknown_cost_records": int(usage[9] or 0),
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
        with self._connect() as connection:
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
