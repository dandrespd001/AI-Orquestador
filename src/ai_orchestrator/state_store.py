"""Small transactional JSON state store backed by private SQLite."""

from __future__ import annotations

import functools
import json
import os
import sqlite3
import time
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Any

from ai_orchestrator.models import ContractError, ensure_private_directory


class StateConflict(RuntimeError):
    """An optimistic state write raced with another process."""


class SQLiteStateStore:
    """Persist one versioned mapping with optimistic cross-process writes."""

    def __init__(
        self,
        path: Path,
        *,
        default_factory: Callable[[], dict[str, Any]],
        legacy_path: Path | None = None,
    ) -> None:
        self.path = path.expanduser().resolve()
        self.default_factory = default_factory
        ensure_private_directory(self.path.parent)
        if self.path.is_symlink():
            raise ContractError("state database must not be a symlink")
        self._initialize(legacy_path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _initialize(self, legacy_path: Path | None) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS state_document (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    generation INTEGER NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            exists = connection.execute(
                "SELECT 1 FROM state_document WHERE singleton = 1"
            ).fetchone()
            if exists is None:
                payload = self._legacy_or_default(legacy_path)
                connection.execute(
                    """
                    INSERT INTO state_document(singleton, generation, payload)
                    VALUES (1, 0, ?)
                    """,
                    (json.dumps(payload, sort_keys=True, separators=(",", ":")),),
                )
        os.chmod(self.path, 0o600)

    def _legacy_or_default(self, legacy_path: Path | None) -> dict[str, Any]:
        if legacy_path is None or not legacy_path.exists():
            return self.default_factory()
        if legacy_path.is_symlink() or not legacy_path.is_file():
            raise ContractError("legacy state path is unsafe")
        try:
            payload = json.loads(legacy_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError("legacy state is invalid") from exc
        if not isinstance(payload, dict):
            raise ContractError("legacy state must contain an object")
        return payload

    def load(self) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT generation, payload FROM state_document WHERE singleton = 1"
            ).fetchone()
        if row is None:
            raise ContractError("state database is uninitialized")
        try:
            payload = json.loads(row[1])
        except json.JSONDecodeError as exc:
            raise ContractError("state database contains invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ContractError("state database must contain an object")
        payload["_state_generation"] = int(row[0])
        return payload

    def save(self, state: dict[str, Any]) -> None:
        payload = dict(state)
        try:
            expected_generation = int(payload.pop("_state_generation"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError("state generation is missing") from exc
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT generation FROM state_document WHERE singleton = 1"
            ).fetchone()
            if current is None or int(current[0]) != expected_generation:
                connection.rollback()
                raise StateConflict("state changed during the operation")
            connection.execute(
                """
                UPDATE state_document
                SET generation = ?, payload = ?
                WHERE singleton = 1
                """,
                (expected_generation + 1, encoded),
            )
            connection.commit()
        finally:
            connection.close()


def retry_state_conflicts[**P, R](
    operation: Callable[P, R],
) -> Callable[P, R]:
    """Retry an idempotent registry mutation after an optimistic conflict."""

    @functools.wraps(operation)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        for attempt in range(8):
            try:
                return operation(*args, **kwargs)
            except StateConflict:
                if attempt == 7:
                    raise ContractError("state remained busy after 8 retries")
                time.sleep(0.002 * (attempt + 1))
        raise AssertionError("unreachable")

    return wrapped
