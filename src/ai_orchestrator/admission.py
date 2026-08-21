"""Cross-process admission control for provider and repository execution."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
import uuid
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

from ai_orchestrator.models import ContractError, ensure_private_directory


class AdmissionUnavailable(ContractError):
    """Raised when configured execution capacity cannot be acquired in time."""


def default_admission_path() -> Path:
    override = os.environ.get("AI_ORCHESTRATOR_STATE")
    root = (
        Path(override).expanduser().resolve()
        if override
        else Path.home() / ".local" / "share" / "ai-orchestrator"
    )
    return root / "admission.sqlite"


@dataclass
class AdmissionLease:
    controller: AdmissionController
    lease_id: str
    lease_seconds: float
    released: bool = False

    def renew(self) -> None:
        if self.released:
            raise AdmissionUnavailable("cannot renew a released admission lease")
        self.controller.renew(self.lease_id, lease_seconds=self.lease_seconds)

    def release(self) -> None:
        if not self.released:
            self.controller.release(self.lease_id)
            self.released = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.release()


class AdmissionController:
    """SQLite-backed leases shared by every local orchestrator process."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = (path or default_admission_path()).expanduser().resolve()
        ensure_private_directory(self.path.parent)
        if self.path.is_symlink():
            raise ValueError("admission database must not be a symlink")
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS admission_leases (
                    lease_id TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    holder_pid INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    PRIMARY KEY (lease_id, resource)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_admission_resource_expiry
                ON admission_leases(resource, expires_at)
                """
            )
        os.chmod(self.path, 0o600)

    @staticmethod
    def _resources(model: str, repo: str, allow_parallel: bool) -> tuple[str, ...]:
        resources = [f"model:{model}"]
        if not allow_parallel:
            repo_hash = hashlib.sha256(repo.encode("utf-8")).hexdigest()
            resources.append(f"repo:{repo_hash}")
        return tuple(resources)

    def acquire(
        self,
        *,
        model: str,
        max_concurrency: int,
        repo: str,
        allow_parallel: bool,
        lease_seconds: float = 1900.0,
        wait_timeout_seconds: float = 30.0,
    ) -> AdmissionLease:
        if not model:
            raise ValueError("admission model is required")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if lease_seconds <= 0 or wait_timeout_seconds < 0:
            raise ValueError("admission timing values are invalid")

        resources = self._resources(model, repo, allow_parallel)
        capacities = {f"model:{model}": max_concurrency}
        capacities.update({resource: 1 for resource in resources[1:]})
        deadline = time.monotonic() + wait_timeout_seconds
        lease_id = str(uuid.uuid4())

        while True:
            now = time.time()
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "DELETE FROM admission_leases WHERE expires_at <= ?", (now,)
                )
                available = True
                for resource in resources:
                    active = connection.execute(
                        "SELECT COUNT(*) FROM admission_leases WHERE resource = ?",
                        (resource,),
                    ).fetchone()[0]
                    if int(active) >= capacities[resource]:
                        available = False
                        break
                if available:
                    connection.executemany(
                        """
                        INSERT INTO admission_leases (
                            lease_id, resource, holder_pid, created_at, expires_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        [
                            (lease_id, resource, os.getpid(), now, now + lease_seconds)
                            for resource in resources
                        ],
                    )
                    connection.commit()
                    return AdmissionLease(self, lease_id, lease_seconds)
                connection.rollback()
            finally:
                connection.close()

            if time.monotonic() >= deadline:
                raise AdmissionUnavailable(
                    f"execution capacity unavailable for model {model}"
                )
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    def renew(self, lease_id: str, *, lease_seconds: float) -> None:
        expires_at = time.time() + lease_seconds
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "UPDATE admission_leases SET expires_at = ? WHERE lease_id = ?",
                (expires_at, lease_id),
            )
        if cursor.rowcount == 0:
            raise AdmissionUnavailable("admission lease has expired")

    def release(self, lease_id: str) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "DELETE FROM admission_leases WHERE lease_id = ?", (lease_id,)
            )

    def inventory(self) -> dict[str, object]:
        now = time.time()
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "DELETE FROM admission_leases WHERE expires_at <= ?", (now,)
            )
            active = connection.execute(
                "SELECT COUNT(DISTINCT lease_id) FROM admission_leases"
            ).fetchone()[0]
            resources = connection.execute(
                """
                SELECT resource, COUNT(*)
                FROM admission_leases
                GROUP BY resource
                ORDER BY resource
                """
            ).fetchall()
        return {
            "schema_version": 1,
            "active_leases": int(active),
            "resources": {str(row[0]): int(row[1]) for row in resources},
        }
