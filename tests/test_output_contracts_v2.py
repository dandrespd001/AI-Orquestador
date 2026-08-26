from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

try:
    from ai_orchestrator.output_contracts import reviewer_accepts, validate_output
except ImportError:
    reviewer_accepts = None
    validate_output = None

from ai_orchestrator.models import TaskSpec


class OutputContractSymbolsTests(unittest.TestCase):
    def test_output_contract_validator_exists(self) -> None:
        self.assertIsNotNone(validate_output)
        self.assertIsNotNone(reviewer_accepts)


@unittest.skipUnless(validate_output is not None, "output validator missing")
class OutputContractsV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name) / "repo"
        self.repo.mkdir()
        (self.repo / "value.txt").write_text("old\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def task(self, contract: str, **overrides: object) -> TaskSpec:
        payload: dict[str, object] = {
            "repo": str(self.repo),
            "task_type": "review",
            "sensitivity": "internal",
            "output_contract": contract,
        }
        payload.update(overrides)
        return TaskSpec.from_mapping(payload)

    def test_text_must_be_nonempty_utf8_and_within_output_budget(self) -> None:
        self.assertEqual(
            "EMPTY_OUTPUT", validate_output(self.task("text"), b" \n").error_code
        )
        self.assertEqual(
            "INVALID_UTF8", validate_output(self.task("text"), b"\xff").error_code
        )
        oversized = validate_output(
            self.task("text", max_output_tokens=1), b"two words"
        )
        self.assertEqual("OUTPUT_TOKEN_LIMIT", oversized.error_code)
        self.assertTrue(validate_output(self.task("text"), "válido".encode()).valid)

    def test_json_is_parsed_and_checked_against_repository_schema(self) -> None:
        (self.repo / "schema.json").write_text(
            json.dumps(
                {
                    "type": "object",
                    "required": ["answer"],
                    "properties": {"answer": {"type": "string"}},
                    "additionalProperties": False,
                }
            ),
            encoding="utf-8",
        )
        task = self.task(
            "json", schema_version=2, output_schema_file="schema.json"
        )
        self.assertEqual(
            "MALFORMED_JSON", validate_output(task, b"{broken").error_code
        )
        self.assertEqual(
            "SCHEMA_VIOLATION",
            validate_output(task, b'{"answer":3}').error_code,
        )
        valid = validate_output(task, b'{"answer":"yes"}')
        self.assertTrue(valid.valid)
        self.assertEqual({"answer": "yes"}, valid.parsed)

    def test_review_requires_verdict_severity_and_evidence(self) -> None:
        malformed = validate_output(
            self.task("review"), b'{"verdict":"accept","findings":[{}]}'
        )
        self.assertEqual("INVALID_REVIEW", malformed.error_code)

        critical = validate_output(
            self.task("review"),
            json.dumps(
                {
                    "verdict": "accept",
                    "findings": [
                        {
                            "severity": "critical",
                            "evidence": "gate bypass at value.txt:1",
                            "message": "unsafe",
                        }
                    ],
                }
            ).encode(),
        )
        self.assertTrue(critical.valid)
        self.assertFalse(reviewer_accepts(critical))

        accepted = validate_output(
            self.task("review"), b'{"verdict":"accept","findings":[]}'
        )
        self.assertTrue(reviewer_accepts(accepted))

    def test_unified_diff_must_apply_cleanly_without_mutating_repository(self) -> None:
        subprocess.run(
            ["git", "init", "-q"], cwd=self.repo, check=True, capture_output=True
        )
        patch = (
            b"diff --git a/value.txt b/value.txt\n"
            b"--- a/value.txt\n"
            b"+++ b/value.txt\n"
            b"@@ -1 +1 @@\n"
            b"-old\n"
            b"+new\n"
        )
        valid = validate_output(self.task("unified_diff"), patch)
        self.assertTrue(valid.valid, valid.error_code)
        self.assertEqual("old\n", (self.repo / "value.txt").read_text(encoding="utf-8"))

        invalid = validate_output(
            self.task("unified_diff"), patch.replace(b"-old", b"-missing")
        )
        self.assertEqual("DIFF_NOT_APPLICABLE", invalid.error_code)


if __name__ == "__main__":
    unittest.main()
