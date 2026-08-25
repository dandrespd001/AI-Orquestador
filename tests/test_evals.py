from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from ai_orchestrator.evals import run_routing_eval

ROOT = Path(__file__).resolve().parents[1]


class RoutingEvalTests(unittest.TestCase):
    def test_golden_route_suite(self) -> None:
        old = os.environ.get("AI_ORCHESTRATOR_CONFIG")
        os.environ["AI_ORCHESTRATOR_CONFIG"] = str(ROOT / "config")
        try:
            with tempfile.TemporaryDirectory() as temporary:
                repo = Path(temporary)
                (repo / "prompt.txt").write_text("test", encoding="utf-8")
                (repo / "test.py").write_text("pass\n", encoding="utf-8")
                row = {
                    "case_id": "ordinary-tests",
                    "task": {
                        "repo": str(repo),
                        "task_type": "tests",
                        "sensitivity": "internal",
                        "prompt_file": "prompt.txt",
                        "input_paths": ["test.py"],
                        "output_contract": "unified_diff",
                    },
                    # 2026-08-25: con las rutas gratuitas abiertas al trabajo
                    # interno, el rol tests lo encabeza la ruta verificada de
                    # Gemini y conserva cuota de suscripcion.
                    "expected": {
                        "status": "READY",
                        "selected_model": "gemini-free-fast",
                        "transport": "qwen_omniroute_cli",
                    },
                }
                suite = repo / "suite.jsonl"
                suite.write_text(json.dumps(row) + "\n", encoding="utf-8")
                report = run_routing_eval(suite, "default")
            self.assertEqual("PASSED", report["status"])
            self.assertEqual(1.0, report["pass_rate"])
        finally:
            if old is None:
                os.environ.pop("AI_ORCHESTRATOR_CONFIG", None)
            else:
                os.environ["AI_ORCHESTRATOR_CONFIG"] = old


if __name__ == "__main__":
    unittest.main()
