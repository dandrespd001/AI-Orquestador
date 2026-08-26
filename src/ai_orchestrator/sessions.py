"""Private Harness session retention, separate from metadata telemetry."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ai_orchestrator.models import ContractError, ensure_private_directory
from ai_orchestrator.telemetry import default_state_root

SESSION_RETENTION_DAYS = 30
MAX_SESSION_BYTES = 4 * 1024 * 1024


class SessionStore:
    def __init__(self, path: Path | None = None, *, create: bool = True) -> None:
        root = path.parent if path is not None else default_state_root()
        self.path = path or root / "harness-sessions.sqlite"
        if self.path.is_symlink():
            raise ContractError("session database must not be a symlink")
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
            connection.execute("PRAGMA secure_delete=ON")
            connection.execute("PRAGMA busy_timeout=5000")
            connection.execute("PRAGMA journal_size_limit=33554432")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        ensure_private_directory(self.path.parent)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    pinned INTEGER NOT NULL CHECK (pinned IN (0, 1)),
                    content_json TEXT NOT NULL,
                    content_bytes INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_updated
                    ON sessions(updated_at, pinned);
                """
            )
        os.chmod(self.path, 0o600)
        for suffix in ("-wal", "-shm"):
            candidate = Path(f"{self.path}{suffix}")
            if candidate.exists() and not candidate.is_symlink():
                os.chmod(candidate, 0o600)

    @staticmethod
    def _session_id(value: str) -> str:
        if (
            not value
            or len(value) > 128
            or any(
                ch
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                for ch in value
            )
        ):
            raise ContractError("invalid session id")
        return value

    def put(
        self, session_id: str, content: object, *, pinned: bool = False
    ) -> dict[str, Any]:
        session_id = self._session_id(session_id)
        encoded = json.dumps(
            content, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        size = len(encoded.encode("utf-8"))
        if size > MAX_SESSION_BYTES:
            raise ContractError("session content exceeds the retention limit")
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT created_at, pinned FROM sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
            created_at = str(existing[0]) if existing else now
            effective_pin = bool(existing[1]) if existing else bool(pinned)
            if pinned:
                effective_pin = True
            connection.execute(
                """
                INSERT INTO sessions (
                    session_id, created_at, updated_at, pinned, content_json, content_bytes
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    updated_at=excluded.updated_at,
                    pinned=excluded.pinned,
                    content_json=excluded.content_json,
                    content_bytes=excluded.content_bytes
                """,
                (session_id, created_at, now, int(effective_pin), encoded, size),
            )
        return {
            "session_id": session_id,
            "updated_at": now,
            "pinned": effective_pin,
            "content_bytes": size,
            "retention_days": SESSION_RETENTION_DAYS,
        }

    def get(self, session_id: str) -> object | None:
        session_id = self._session_id(session_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT content_json FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def set_pinned(
        self, session_id: str, pinned: bool, *, human_approved: bool = False
    ) -> dict[str, Any]:
        if not human_approved:
            raise ContractError(
                "pinning or releasing a session requires a human action"
            )
        session_id = self._session_id(session_id)
        with self._connect() as connection:
            changed = connection.execute(
                "UPDATE sessions SET pinned=? WHERE session_id=?",
                (int(pinned), session_id),
            ).rowcount
        if changed != 1:
            raise ContractError("unknown session")
        return {"session_id": session_id, "pinned": bool(pinned)}

    def inventory(self) -> dict[str, Any]:
        if not self.path.exists() and not self.create:
            return {
                "schema_version": 1,
                "retention_days": SESSION_RETENTION_DAYS,
                "sessions": [],
                "content_included": False,
            }
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT session_id, created_at, updated_at, pinned, content_bytes "
                "FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        return {
            "schema_version": 1,
            "retention_days": SESSION_RETENTION_DAYS,
            "sessions": [
                {
                    "session_id": row[0],
                    "created_at": row[1],
                    "updated_at": row[2],
                    "pinned": bool(row[3]),
                    "content_bytes": int(row[4]),
                }
                for row in rows
            ],
            "content_included": False,
        }

    def preview_purge(
        self, retention_days: int = SESSION_RETENTION_DAYS
    ) -> dict[str, Any]:
        if not 1 <= retention_days <= 30:
            raise ContractError("session retention must be between 1 and 30 days")
        cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
        if not self.path.exists() and not self.create:
            return {
                "sessions": 0,
                "cutoff": cutoff,
                "pinned_excluded": True,
                "dry_run": True,
            }
        with self._connect() as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM sessions WHERE pinned=0 AND updated_at < ?",
                (cutoff,),
            ).fetchone()[0]
        return {
            "sessions": int(count),
            "cutoff": cutoff,
            "pinned_excluded": True,
            "dry_run": True,
        }

    def purge(self, retention_days: int = SESSION_RETENTION_DAYS) -> dict[str, Any]:
        preview = self.preview_purge(retention_days)
        with self._connect() as connection:
            deleted = connection.execute(
                "DELETE FROM sessions WHERE pinned=0 AND updated_at < ?",
                (preview["cutoff"],),
            ).rowcount
            connection.execute("PRAGMA incremental_vacuum")
        return {"sessions": int(deleted), "pinned_excluded": True}
