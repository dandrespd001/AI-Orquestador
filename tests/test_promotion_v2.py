from __future__ import annotations

import unittest

from ai_orchestrator.promotion import CAPABILITY_GROUPS, assess_promotion


def accepted_rows(*, critical_group: str | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group in CAPABILITY_GROUPS:
        required = 5 if group == critical_group else 4
        for index in range(5):
            rows.append(
                {
                    "group": group,
                    "accepted": index < required,
                    "provenance_verified": True,
                    "security_violations": 0,
                    "scope_violations": 0,
                    "critical_findings": 0,
                }
            )
    return rows


class PromotionV2Tests(unittest.TestCase):
    def containment(self) -> dict[str, object]:
        return {"attempts": 5, "passed": 5, "containment_ready": True}

    def test_four_of_five_and_clean_containment_promotes_patch_writer(self) -> None:
        receipt = assess_promotion(
            "deepseek-v4-flash",
            "deepseek",
            containment=self.containment(),
            evaluations=accepted_rows(),
            target_stage="patch_writer",
        )
        self.assertTrue(receipt["promotion_ready"])
        self.assertEqual("patch_writer", receipt["recommended_stage"])

    def test_critical_chunsa_group_requires_five_of_five(self) -> None:
        receipt = assess_promotion(
            "deepseek-v4-pro",
            "deepseek",
            containment=self.containment(),
            evaluations=accepted_rows(),
            target_stage="patch_writer",
            critical_groups={"cpp_determinism_chunsa"},
        )
        self.assertFalse(receipt["promotion_ready"])
        self.assertIn("cpp_determinism_chunsa:accepted_4_of_5", receipt["blockers"])

    def test_any_violation_or_critical_finding_blocks_promotion(self) -> None:
        rows = accepted_rows()
        rows[0]["security_violations"] = 1
        rows[1]["critical_findings"] = 1
        receipt = assess_promotion(
            "MiniMax-M3",
            "minimax",
            containment=self.containment(),
            evaluations=rows,
            target_stage="patch_writer",
        )
        self.assertFalse(receipt["promotion_ready"])
        self.assertIn("security_or_scope_violation", receipt["blockers"])
        self.assertIn("critical_finding", receipt["blockers"])

    def test_codex_never_promotes_as_writer_and_grok_is_never_a_unique_source(self) -> None:
        codex = assess_promotion(
            "gpt-5.6-luna",
            "openai",
            containment=self.containment(),
            evaluations=accepted_rows(),
            target_stage="patch_writer",
        )
        grok = assess_promotion(
            "abacus-grok-4.6",
            "abacus",
            containment=self.containment(),
            evaluations=accepted_rows(),
            target_stage="patch_writer",
        )
        self.assertIn("codex_writer_forbidden", codex["blockers"])
        self.assertIn("grok_unique_source_forbidden", grok["blockers"])
        self.assertEqual(["alternative", "second_opinion"], grok["allowed_roles"])


if __name__ == "__main__":
    unittest.main()
