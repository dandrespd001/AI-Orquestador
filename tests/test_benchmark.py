from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_orchestrator.benchmark import (
    Observation,
    atomic_private_json,
    load_benchmark_manifest,
    run_gateway_benchmark,
    select_gateway,
)
from ai_orchestrator.models import ContractError


class GatewayBenchmarkTests(unittest.TestCase):
    def _manifest(self, root: Path, base_url: str) -> Path:
        (root / "payload.json").write_text(
            json.dumps(
                {
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "sentinel"}],
                }
            ),
            encoding="utf-8",
        )
        path = root / "benchmark.toml"
        path.write_text(
            f'''schema_version = 1
repetitions = 4
warmups = 1
timeout_seconds = 2

[[targets]]
name = "mock"
version = "1.0"
base_url = "{base_url}"
official_providers = ["deepseek", "moonshot", "minimax"]
provider_coverage_attested = true
custom_provider_code = false
content_logging = false
immutable_ref = "mock@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

[[scenarios]]
name = "chat"
path = "/v1/chat/completions"
payload_file = "payload.json"
expected_status = 200
expected_model = "mock-model"
concurrency = 4
''',
            encoding="utf-8",
        )
        return path

    def test_metadata_only_benchmark_passes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            manifest = self._manifest(Path(raw), "http://127.0.0.1:21001")
            observation = Observation(
                status=200,
                first_byte_ms=1.0,
                total_ms=2.0,
                response_bytes=64,
                response_sha256="f" * 64,
                reported_model="mock-model",
                error_code=None,
            )
            with patch("ai_orchestrator.benchmark._request", return_value=observation):
                report = run_gateway_benchmark(manifest)
        self.assertEqual("PASSED", report["status"])
        self.assertEqual(1.0, report["results"][0]["correctness_rate"])
        encoded = json.dumps(report)
        self.assertNotIn("sentinel", encoded)
        self.assertIn("request_sha256", encoded)

    def test_non_loopback_target_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            manifest = self._manifest(Path(raw), "https://example.com")
            with self.assertRaises(ContractError):
                load_benchmark_manifest(manifest)

    def test_selection_requires_material_improvement_and_hard_gates(self) -> None:
        report = {
            "kind": "gateway_benchmark",
            "status": "PASSED",
            "target_gates": [
                {"target": "current", "promotion_eligible": True},
                {"target": "fast", "promotion_eligible": True},
                {"target": "partial", "promotion_eligible": False},
            ],
            "results": [
                {"target": "current", "scenario": "chat", "total_ms": {"p95": 10.0}},
                {"target": "fast", "scenario": "chat", "total_ms": {"p95": 7.0}},
                {"target": "partial", "scenario": "chat", "total_ms": {"p95": 1.0}},
            ],
        }
        decision = select_gateway(report, "current")
        self.assertEqual("fast", decision["selected"])
        self.assertTrue(decision["replacement"])

    def test_private_writer_never_changes_public_parent_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            root.chmod(0o755)
            before = root.stat().st_mode & 0o777
            with self.assertRaises(ContractError):
                atomic_private_json(root / "receipt.json", {"status": "PASSED"})
            self.assertEqual(before, root.stat().st_mode & 0o777)


if __name__ == "__main__":
    unittest.main()
