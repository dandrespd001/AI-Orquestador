from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_orchestrator.models import ContractError
from ai_orchestrator.optimization import (
    load_prompt_experiment,
    prompt_optimization_preflight,
)

ROOT = Path(__file__).resolve().parents[1]


class OptimizationTests(unittest.TestCase):
    def test_pinned_experiment_has_disjoint_admissible_dataset(self) -> None:
        experiment, dataset, rows = load_prompt_experiment(ROOT / "evals" / "dspy" / "tests-v1.toml")
        self.assertEqual("dspy", experiment.engine)
        self.assertEqual("BootstrapFewShot", experiment.optimizer)
        self.assertEqual(40, experiment.max_calls)
        self.assertEqual(40, len(rows))
        self.assertEqual(40, len({row["case_id"] for row in rows}))
        self.assertTrue(dataset.is_file())

    def test_preflight_requires_passing_baseline_and_exact_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            baseline = Path(raw) / "baseline.json"
            baseline.write_text(json.dumps({"status": "PASSED"}), encoding="utf-8")
            with patch("ai_orchestrator.optimization.importlib.metadata.version", return_value="3.3.0"):
                report = prompt_optimization_preflight(
                    ROOT / "evals" / "dspy" / "tests-v1.toml", baseline
                )
            self.assertEqual("READY", report["status"])
            self.assertFalse(report["incremental_spend_allowed"])
            self.assertFalse(report["live_or_financial_inputs_allowed"])

            baseline.write_text(json.dumps({"status": "FAILED"}), encoding="utf-8")
            with self.assertRaises(ContractError):
                prompt_optimization_preflight(ROOT / "evals" / "dspy" / "tests-v1.toml", baseline)


if __name__ == "__main__":
    unittest.main()
