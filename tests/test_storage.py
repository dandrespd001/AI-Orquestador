from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from ai_orchestrator.storage import _sqlite_info


class StorageTests(unittest.TestCase):
    def test_sqlite_audit_reports_integrity_and_reclaimable_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "state.sqlite"
            with sqlite3.connect(path) as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute("PRAGMA secure_delete=ON")
                connection.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, value TEXT)")
                connection.executemany(
                    "INSERT INTO events(value) VALUES (?)",
                    [("x" * 200,) for _ in range(500)],
                )
                connection.execute("DELETE FROM events")
            info = _sqlite_info(path)
            self.assertEqual("ok", info["quick_check"])
            self.assertEqual("wal", info["journal_mode"])
            self.assertGreaterEqual(info["reclaimable_bytes"], 0)


if __name__ == "__main__":
    unittest.main()

