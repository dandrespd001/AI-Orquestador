from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from ai_orchestrator.admission import AdmissionController, AdmissionUnavailable


class AdmissionControllerTests(unittest.TestCase):
    def test_model_capacity_is_shared_between_controller_instances(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "admission.sqlite"
            first = AdmissionController(path)
            second = AdmissionController(path)

            with (
                first.acquire(
                    model="model-a",
                    max_concurrency=1,
                    repo="/repo/a",
                    allow_parallel=True,
                    wait_timeout_seconds=0,
                ),
                self.assertRaises(AdmissionUnavailable),
            ):
                second.acquire(
                    model="model-a",
                    max_concurrency=1,
                    repo="/repo/b",
                    allow_parallel=True,
                    wait_timeout_seconds=0,
                )

            with second.acquire(
                model="model-a",
                max_concurrency=1,
                repo="/repo/b",
                allow_parallel=True,
                wait_timeout_seconds=0,
            ):
                self.assertEqual(1, second.inventory()["active_leases"])

    def test_non_parallel_tasks_serialize_the_same_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = AdmissionController(Path(directory) / "admission.sqlite")
            with (
                controller.acquire(
                    model="model-a",
                    max_concurrency=2,
                    repo="/repo/shared",
                    allow_parallel=False,
                    wait_timeout_seconds=0,
                ),
                self.assertRaises(AdmissionUnavailable),
            ):
                controller.acquire(
                    model="model-b",
                    max_concurrency=2,
                    repo="/repo/shared",
                    allow_parallel=False,
                    wait_timeout_seconds=0,
                )

    def test_expired_lease_is_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = AdmissionController(Path(directory) / "admission.sqlite")
            controller.acquire(
                model="model-a",
                max_concurrency=1,
                repo="/repo/a",
                allow_parallel=True,
                lease_seconds=0.01,
                wait_timeout_seconds=0,
            )
            time.sleep(0.02)

            with controller.acquire(
                model="model-a",
                max_concurrency=1,
                repo="/repo/a",
                allow_parallel=True,
                wait_timeout_seconds=0,
            ):
                self.assertEqual(1, controller.inventory()["active_leases"])


if __name__ == "__main__":
    unittest.main()
