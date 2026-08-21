from __future__ import annotations

import gc
import tempfile
import unittest
import warnings
from pathlib import Path

from ai_orchestrator.admission import AdmissionController
from ai_orchestrator.sessions import SessionStore
from ai_orchestrator.state_store import SQLiteStateStore
from ai_orchestrator.telemetry import TelemetryStore


class SQLiteStateStoreTests(unittest.TestCase):
    def test_initialize_load_and_save_close_every_connection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ResourceWarning)
                store = SQLiteStateStore(
                    Path(directory) / "state.sqlite",
                    default_factory=lambda: {"items": []},
                )
                state = store.load()
                state["items"].append("saved")
                store.save(state)
                del store
                gc.collect()

            leaked = [
                warning
                for warning in caught
                if issubclass(warning.category, ResourceWarning)
                and "database" in str(warning.message)
            ]
            self.assertEqual([], leaked)

    def test_sqlite_services_close_read_and_initialization_connections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ResourceWarning)
                AdmissionController(root / "admission.sqlite").inventory()
                SessionStore(root / "sessions.sqlite").inventory()
                TelemetryStore(root / "telemetry.sqlite").report()
                gc.collect()

            leaked = [
                warning
                for warning in caught
                if issubclass(warning.category, ResourceWarning)
                and "database" in str(warning.message)
            ]
            self.assertEqual([], leaked)


if __name__ == "__main__":
    unittest.main()
