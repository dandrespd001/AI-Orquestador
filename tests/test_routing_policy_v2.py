from __future__ import annotations

import inspect
import tempfile
import unittest
from dataclasses import fields, replace
from datetime import UTC, datetime
from pathlib import Path

from ai_orchestrator import models as model_contracts
from ai_orchestrator.models import (
    ContractError,
    ModelConfig,
    Profile,
    RoleConfig,
    TaskSpec,
    UsageRecord,
)
from ai_orchestrator.routing import plan_route, select_route
from ai_orchestrator.telemetry import TelemetryStore

ModelEconomics = getattr(model_contracts, "ModelEconomics", None)
QuotaSnapshot = getattr(model_contracts, "QuotaSnapshot", None)


class ContractSymbolTests(unittest.TestCase):
    def test_v2_economics_and_quota_contracts_exist(self) -> None:
        self.assertIsNotNone(ModelEconomics)
        self.assertIsNotNone(QuotaSnapshot)

    def test_transactional_lease_api_exists(self) -> None:
        self.assertTrue(hasattr(TelemetryStore, "acquire_lease"))
        self.assertTrue(hasattr(TelemetryStore, "release_lease"))

    def test_route_planner_accepts_quota_and_clock_snapshots(self) -> None:
        parameters = inspect.signature(plan_route).parameters
        self.assertIn("quotas", parameters)
        self.assertIn("at", parameters)

    def test_quota_snapshot_represents_all_kimi_windows(self) -> None:
        self.assertIn("windows", {field.name for field in fields(QuotaSnapshot)})


@unittest.skipUnless(ModelEconomics is not None and QuotaSnapshot is not None, "v2 contracts missing")
@unittest.skipUnless("quotas" in inspect.signature(plan_route).parameters, "v2 planner API missing")
class RoutingPolicyV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name) / "repo"
        self.repo.mkdir()
        (self.repo / "prompt.txt").write_text("bounded task", encoding="utf-8")
        (self.repo / "schema.json").write_text(
            '{"type":"object","required":["answer"],'
            '"properties":{"answer":{"type":"string"}},'
            '"additionalProperties":false}',
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def task(self, **overrides: object) -> TaskSpec:
        payload: dict[str, object] = {
            "repo": str(self.repo),
            "task_type": "review",
            "sensitivity": "internal",
            "prompt_file": "prompt.txt",
            "max_context_tokens": 10_000,
        }
        payload.update(overrides)
        return TaskSpec.from_mapping(payload)

    @staticmethod
    def model(
        model_id: str,
        family: str,
        provider: str,
        *,
        economics: ModelEconomics,
        context_window: int = 256_000,
        capabilities: tuple[str, ...] = ("read",),
    ) -> ModelConfig:
        return ModelConfig(
            id=model_id,
            family=family,
            provider=provider,
            transport="codex_cli" if provider == "openai" else "claude_cli",
            route_model=model_id,
            official=True,
            status="available",
            roles=("review", "implementation"),
            max_sensitivity="internal",
            max_concurrency=1,
            capabilities=capabilities,
            execution_profile="read_only",
            context_window=context_window,
            economics=economics,
        )

    @staticmethod
    def profile(*model_ids: str, assurance_floor: str = "standard") -> Profile:
        return Profile(
            "default",
            None,
            ("internal",),
            tuple(model_ids),
            (),
            (),
            (),
            "direct",
            assurance_floor=assurance_floor,
        )

    def test_v1_task_defaults_and_v2_schema_path_are_compatible(self) -> None:
        legacy = self.task()
        self.assertEqual("standard", legacy.assurance)
        self.assertEqual(4_096, legacy.max_output_tokens)
        self.assertIsNone(legacy.output_schema_file)

        version_two = self.task(
            schema_version=2,
            output_contract="json",
            output_schema_file="schema.json",
            assurance="high",
            max_output_tokens=8_192,
        )
        self.assertEqual("high", version_two.assurance)
        self.assertEqual(8_192, version_two.max_output_tokens)
        self.assertEqual("schema.json", version_two.output_schema_file)
        with self.assertRaises(ContractError):
            self.task(output_contract="text", output_schema_file="schema.json")

    def test_task_contract_allows_the_minimax_one_million_context_window(self) -> None:
        task = self.task(max_context_tokens=1_000_000)
        self.assertEqual(1_000_000, task.max_context_tokens)

    def test_effective_cost_accounts_for_cache_read_and_cache_write(self) -> None:
        economics = ModelEconomics(
            plan="api_balance",
            billing_mode="prepaid_balance",
            input_per_mtok_usd=2.0,
            cache_read_per_mtok_usd=0.2,
            cache_write_per_mtok_usd=2.5,
            output_per_mtok_usd=12.0,
            source="official pricing",
            verified_at="2026-08-20T00:00:00+00:00",
            confidence="verified",
        )
        usage = UsageRecord(
            provider="provider",
            model="model",
            outcome="SUCCESS",
            latency_ms=10,
            input_tokens=1_000_000,
            output_tokens=100_000,
            cached_tokens=500_000,
            cache_write_tokens=250_000,
        )
        self.assertAlmostEqual(3.925, economics.cost_for(usage), places=6)

    def test_codex_weekly_cap_is_a_hard_joint_fifteen_percent_limit(self) -> None:
        economics = ModelEconomics(
            plan="pro_5x",
            billing_mode="included_quota",
            input_per_mtok_usd=5.0,
            output_per_mtok_usd=30.0,
            source="official pricing",
            verified_at="2026-08-20T00:00:00+00:00",
            confidence="verified",
            orchestrator_quota_fraction=0.15,
        )
        model = self.model("gpt-5.6-sol", "openai-gpt", "openai", economics=economics)
        role = RoleConfig("review", (model.id,), ())
        quota = QuotaSnapshot(
            provider="openai",
            plan="pro_5x",
            billing_mode="included_quota",
            remaining=86.0,
            limit=100.0,
            used=14.0,
            predicted_usage=2.0,
            window="weekly",
            reset_at="2026-08-24T00:00:00+00:00",
            source="codex status",
            verified_at="2026-08-20T00:00:00+00:00",
            confidence="verified",
        )
        blocked = plan_route(
            self.task(),
            {model.id: model},
            {"review": role},
            self.profile(model.id),
            quotas={"openai": quota},
            at=datetime(2026, 8, 20, tzinfo=UTC),
        )
        self.assertIsNone(blocked.selected_model)
        self.assertEqual("HANDOFF_REQUIRED", blocked.status)
        self.assertEqual("orchestrator_quota_limit", blocked.unavailable[model.id])

        allowed = plan_route(
            self.task(),
            {model.id: model},
            {"review": role},
            self.profile(model.id),
            quotas={"openai": replace(quota, used=13.0, remaining=87.0)},
            at=datetime(2026, 8, 20, tzinfo=UTC),
        )
        self.assertEqual(model.id, allowed.selected_model)

    def test_missing_codex_limit_and_empty_deepseek_balance_require_handoff(self) -> None:
        codex_economics = ModelEconomics(
            plan="pro_5x",
            billing_mode="included_quota",
            source="official pricing",
            verified_at="2026-08-20T00:00:00+00:00",
            confidence="verified",
            orchestrator_quota_fraction=0.15,
        )
        deepseek_economics = ModelEconomics(
            plan="api_balance",
            billing_mode="prepaid_balance",
            input_per_mtok_usd=0.14,
            output_per_mtok_usd=0.28,
            source="official pricing",
            verified_at="2026-08-20T00:00:00+00:00",
            confidence="verified",
        )
        codex = self.model("codex", "openai-gpt", "openai", economics=codex_economics)
        flash = self.model("flash", "deepseek", "deepseek", economics=deepseek_economics)
        role = RoleConfig("review", (codex.id, flash.id), ())
        route = plan_route(
            self.task(),
            {codex.id: codex, flash.id: flash},
            {"review": role},
            self.profile(codex.id, flash.id),
            quotas={
                "deepseek": QuotaSnapshot(
                    provider="deepseek",
                    plan="api_balance",
                    billing_mode="prepaid_balance",
                    remaining=0.0,
                    balance_usd=0.0,
                    source="/user/balance",
                    verified_at="2026-08-20T00:00:00+00:00",
                    confidence="verified",
                )
            },
            at=datetime(2026, 8, 20, tzinfo=UTC),
        )
        self.assertIsNone(route.selected_model)
        self.assertEqual("quota_unknown", route.unavailable[codex.id])
        self.assertEqual("balance_exhausted", route.unavailable[flash.id])

    def test_codex_reviewer_also_consumes_the_joint_weekly_cap(self) -> None:
        included = ModelEconomics(
            plan="subscription",
            billing_mode="included_quota",
            source="official",
            verified_at="2026-08-20T00:00:00+00:00",
            confidence="verified",
        )
        codex_economics = ModelEconomics(
            plan="pro_5x",
            billing_mode="included_quota",
            source="official pricing",
            verified_at="2026-08-20T00:00:00+00:00",
            confidence="verified",
            orchestrator_quota_fraction=0.15,
        )
        primary = self.model(
            "claude", "anthropic-claude", "anthropic", economics=included
        )
        codex = self.model(
            "codex", "openai-gpt", "openai", economics=codex_economics
        )
        route = plan_route(
            self.task(assurance="high"),
            {primary.id: primary, codex.id: codex},
            {"review": RoleConfig("review", (primary.id,), (codex.id,))},
            self.profile(primary.id, codex.id),
            quotas={},
            at=datetime(2026, 8, 20, tzinfo=UTC),
        )
        self.assertEqual(primary.id, route.selected_model)
        self.assertEqual((), route.reviewer_models)
        self.assertEqual("HANDOFF_REQUIRED", route.status)
        self.assertEqual("quota_unknown", route.unavailable[codex.id])

    def test_codex_never_autoexecutes_an_implementation_role(self) -> None:
        included = ModelEconomics(
            plan="included",
            billing_mode="included_quota",
            source="official",
            verified_at="2026-08-20T00:00:00+00:00",
            confidence="verified",
        )
        codex = self.model("codex", "openai-gpt", "openai", economics=included)
        reviewer = self.model(
            "claude", "anthropic-claude", "anthropic", economics=included
        )
        route = plan_route(
            self.task(task_type="implementation", write_mode="read_only"),
            {codex.id: codex, reviewer.id: reviewer},
            {
                "implementation": RoleConfig(
                    "implementation", (codex.id,), (reviewer.id,)
                )
            },
            self.profile(codex.id, reviewer.id),
            quotas={},
            at=datetime(2026, 8, 20, tzinfo=UTC),
        )
        self.assertEqual(codex.id, route.selected_model)
        self.assertEqual("HANDOFF_REQUIRED", route.status)
        self.assertIn("Codex implementation requires handoff", route.reason)

    def test_candidate_rollout_is_shadow_only_on_v2_surfaces(self) -> None:
        economics = ModelEconomics(
            plan="included",
            billing_mode="included_quota",
            source="official",
            verified_at="2026-08-20T00:00:00+00:00",
            confidence="verified",
        )
        candidate = replace(
            self.model(
                "candidate", "candidate-family", "anthropic", economics=economics
            ),
            rollout_stage="candidate",
            candidate_since="2026-08-20",
        )
        reviewer = self.model(
            "reviewer", "review-family", "openai", economics=economics
        )
        role = RoleConfig("review", (candidate.id,), (reviewer.id,))
        strict = plan_route(
            self.task(),
            {candidate.id: candidate, reviewer.id: reviewer},
            {"review": role},
            self.profile(candidate.id, reviewer.id),
            quotas={},
            at=datetime(2026, 8, 20, tzinfo=UTC),
        )
        self.assertEqual(candidate.id, strict.selected_model)
        self.assertEqual("HANDOFF_REQUIRED", strict.status)
        self.assertIn("candidate shadow route", strict.reason)

        read_canary = plan_route(
            self.task(),
            {candidate.id: candidate, reviewer.id: reviewer},
            {"review": role},
            self.profile(candidate.id, reviewer.id),
            quotas={},
            rollout_counts={candidate.id: 20},
            at=datetime(2026, 8, 20, tzinfo=UTC),
        )
        self.assertEqual("READY", read_canary.status)

        legacy = select_route(
            self.task(),
            {candidate.id: candidate, reviewer.id: reviewer},
            {"review": role},
            self.profile(candidate.id, reviewer.id),
            at=datetime(2026, 8, 20, tzinfo=UTC),
        )
        self.assertEqual("READY", legacy.status)

    def test_overlapping_quality_intervals_choose_lower_cost_per_acceptance(self) -> None:
        flash_economics = ModelEconomics(
            plan="api_balance",
            billing_mode="prepaid_balance",
            input_per_mtok_usd=0.14,
            output_per_mtok_usd=0.28,
            source="official pricing",
            verified_at="2026-08-20T00:00:00+00:00",
            confidence="verified",
        )
        pro_economics = replace(
            flash_economics,
            input_per_mtok_usd=0.435,
            output_per_mtok_usd=0.87,
        )
        flash = self.model("flash", "deepseek", "deepseek", economics=flash_economics)
        pro = self.model("pro", "deepseek", "deepseek", economics=pro_economics)
        role = RoleConfig("review", (pro.id, flash.id), ())
        quota = QuotaSnapshot(
            provider="deepseek",
            plan="api_balance",
            billing_mode="prepaid_balance",
            remaining=1.0,
            balance_usd=1.0,
            source="/user/balance",
            verified_at="2026-08-20T00:00:00+00:00",
            confidence="verified",
        )
        route = plan_route(
            self.task(max_output_tokens=1_000),
            {flash.id: flash, pro.id: pro},
            {"review": role},
            self.profile(pro.id, flash.id),
            observations={
                "flash": {"quality_lower": 0.70, "quality_upper": 0.90, "acceptance_rate": 0.8},
                "pro": {"quality_lower": 0.72, "quality_upper": 0.88, "acceptance_rate": 0.8},
            },
            quotas={"deepseek": quota},
            at=datetime(2026, 8, 20, tzinfo=UTC),
        )
        self.assertEqual("flash", route.selected_model)
        self.assertLess(
            route.cost_breakdown["flash"]["cost_per_accepted_usd"],
            route.cost_breakdown["pro"]["cost_per_accepted_usd"],
        )

        decisive = plan_route(
            self.task(max_output_tokens=1_000),
            {flash.id: flash, pro.id: pro},
            {"review": role},
            self.profile(pro.id, flash.id),
            observations={
                "flash": {"quality_lower": 0.70, "quality_upper": 0.80, "acceptance_rate": 0.8},
                "pro": {"quality_lower": 0.85, "quality_upper": 0.90, "acceptance_rate": 0.8},
            },
            quotas={"deepseek": quota},
            at=datetime(2026, 8, 20, tzinfo=UTC),
        )
        self.assertEqual("pro", decisive.selected_model)

    def test_deepseek_pro_requires_the_exact_plan_confirmation(self) -> None:
        prepaid = ModelEconomics(
            plan="api_balance",
            billing_mode="prepaid_balance",
            input_per_mtok_usd=0.435,
            output_per_mtok_usd=0.87,
            source="official pricing",
            verified_at="2026-08-20T00:00:00+00:00",
            confidence="verified",
        )
        included = replace(
            prepaid,
            plan="subscription",
            billing_mode="included_quota",
            input_per_mtok_usd=0.0,
            output_per_mtok_usd=0.0,
        )
        pro = self.model(
            "deepseek-v4-pro", "deepseek", "deepseek", economics=prepaid
        )
        reviewer = self.model(
            "claude", "anthropic-claude", "anthropic", economics=included
        )
        plan = plan_route(
            self.task(),
            {pro.id: pro, reviewer.id: reviewer},
            {"review": RoleConfig("review", (pro.id,), (reviewer.id,))},
            self.profile(pro.id, reviewer.id),
            quotas={
                "deepseek": QuotaSnapshot(
                    provider="deepseek",
                    plan="api_balance",
                    billing_mode="prepaid_balance",
                    balance_usd=5.0,
                    source="/user/balance",
                    verified_at="2026-08-20T00:00:00+00:00",
                    confidence="verified",
                )
            },
            at=datetime(2026, 8, 20, tzinfo=UTC),
        )
        self.assertEqual("CONFIRMATION_REQUIRED", plan.status)
        self.assertTrue(plan.confirmation_required)

    def test_profile_can_raise_but_not_lower_assurance_and_plan_id_is_stable(self) -> None:
        economics = ModelEconomics(
            plan="subscription",
            billing_mode="included_quota",
            source="operator",
            verified_at="2026-08-20T00:00:00+00:00",
            confidence="verified",
        )
        primary = self.model("primary", "family-a", "anthropic", economics=economics)
        reviewer = self.model("reviewer", "family-b", "anthropic", economics=economics)
        role = RoleConfig("review", (primary.id,), (reviewer.id,))
        profile = self.profile(primary.id, reviewer.id, assurance_floor="high")
        first = plan_route(
            self.task(assurance="standard"),
            {primary.id: primary, reviewer.id: reviewer},
            {"review": role},
            profile,
            at=datetime(2026, 8, 20, tzinfo=UTC),
        )
        second = plan_route(
            self.task(task_id=first.task_id, assurance="standard"),
            {primary.id: primary, reviewer.id: reviewer},
            {"review": role},
            profile,
            at=datetime(2026, 8, 20, tzinfo=UTC),
        )
        self.assertEqual("high", first.assurance)
        self.assertEqual(first.plan_id, second.plan_id)
        decision = select_route(
            self.task(task_id=first.task_id, assurance="standard"),
            {primary.id: primary, reviewer.id: reviewer},
            {"review": role},
            profile,
            at=datetime(2026, 8, 20, tzinfo=UTC),
        )
        self.assertEqual(first.plan_id, decision.plan_id)

    def test_quota_refresh_timestamp_alone_does_not_invalidate_plan_id(self) -> None:
        economics = ModelEconomics(
            plan="prepaid",
            billing_mode="prepaid_balance",
            input_per_mtok_usd=0.14,
            output_per_mtok_usd=0.28,
            source="official",
            verified_at="2026-08-20T00:00:00+00:00",
            confidence="verified",
        )
        primary = self.model(
            "flash", "deepseek", "deepseek", economics=economics
        )
        reviewer = self.model(
            "reviewer", "anthropic", "anthropic", economics=replace(
                economics, billing_mode="included_quota"
            )
        )
        task = self.task()
        kwargs = {
            "task": task,
            "models": {primary.id: primary, reviewer.id: reviewer},
            "roles": {"review": RoleConfig("review", (primary.id,), (reviewer.id,))},
            "profile": self.profile(primary.id, reviewer.id),
            "at": datetime(2026, 8, 20, 12, tzinfo=UTC),
        }
        first = plan_route(
            **kwargs,
            quotas={
                "deepseek": QuotaSnapshot(
                    provider="deepseek",
                    plan="prepaid",
                    billing_mode="prepaid_balance",
                    balance_usd=5.0,
                    verified_at="2026-08-20T11:59:00+00:00",
                    confidence="verified",
                )
            },
        )
        second = plan_route(
            **kwargs,
            quotas={
                "deepseek": QuotaSnapshot(
                    provider="deepseek",
                    plan="prepaid",
                    billing_mode="prepaid_balance",
                    balance_usd=5.0,
                    verified_at="2026-08-20T12:00:00+00:00",
                    confidence="verified",
                )
            },
        )
        self.assertEqual(first.plan_id, second.plan_id)

    def test_kimi_context_is_256k_and_fx_expires_on_september_first(self) -> None:
        economics = ModelEconomics(
            plan="included",
            billing_mode="included_quota",
            source="official",
            verified_at="2026-08-20T00:00:00+00:00",
            confidence="verified",
        )
        kimi = self.model(
            "kimi", "moonshot-kimi", "moonshot", economics=economics, context_window=256_000
        )
        fx = replace(
            self.model("fx", "zai", "fx", economics=economics),
            available_until="2026-09-01",
        )
        role = RoleConfig("review", (kimi.id, fx.id), ())
        route = plan_route(
            self.task(max_context_tokens=300_000),
            {kimi.id: kimi, fx.id: fx},
            {"review": role},
            self.profile(kimi.id, fx.id),
            at=datetime(2026, 9, 2, tzinfo=UTC),
        )
        self.assertIsNone(route.selected_model)
        self.assertEqual("context_window_exceeded", route.unavailable[kimi.id])
        self.assertEqual("promotion_expired", route.unavailable[fx.id])
        cutoff = plan_route(
            self.task(max_context_tokens=10_000),
            {kimi.id: kimi, fx.id: fx},
            {"review": role},
            self.profile(kimi.id, fx.id),
            pin=fx.id,
            at=datetime(2026, 9, 1, tzinfo=UTC),
        )
        self.assertEqual("promotion_expired", cutoff.unavailable[fx.id])

    def test_any_exhausted_kimi_window_blocks_automatic_routing(self) -> None:
        economics = ModelEconomics(
            plan="moderato_99",
            billing_mode="included_quota",
            source="Kimi membership",
            verified_at="2026-08-20T00:00:00+00:00",
            confidence="verified",
        )
        kimi = self.model(
            "kimi", "moonshot-kimi", "moonshot", economics=economics
        )
        quota = QuotaSnapshot(
            provider="moonshot",
            plan="moderato_99",
            billing_mode="included_quota",
            remaining=10.0,
            windows={"monthly": 10.0, "weekly": 2.0, "five_hours": 0.0},
            source="kimi status",
            verified_at="2026-08-20T00:00:00+00:00",
            confidence="verified",
            extra_usage_enabled=False,
        )
        route = plan_route(
            self.task(),
            {kimi.id: kimi},
            {"review": RoleConfig("review", (kimi.id,), ())},
            self.profile(kimi.id),
            quotas={"moonshot": quota},
            at=datetime(2026, 8, 20, tzinfo=UTC),
        )
        self.assertIsNone(route.selected_model)
        self.assertEqual("quota_exhausted", route.unavailable[kimi.id])

    def test_minimax_payg_or_credit_snapshot_is_never_auto_routable(self) -> None:
        economics = ModelEconomics(
            plan="plus_20",
            billing_mode="included_quota",
            source="MiniMax Token Plan",
            verified_at="2026-08-20T00:00:00+00:00",
            confidence="verified",
        )
        minimax = self.model(
            "MiniMax-M3", "minimax", "minimax", economics=economics
        )
        route = plan_route(
            self.task(),
            {minimax.id: minimax},
            {"review": RoleConfig("review", (minimax.id,), ())},
            self.profile(minimax.id),
            quotas={
                "minimax": QuotaSnapshot(
                    provider="minimax",
                    plan="credits",
                    billing_mode="payg",
                    remaining=100.0,
                    source="account status",
                    verified_at="2026-08-20T00:00:00+00:00",
                    confidence="verified",
                    extra_usage_enabled=True,
                )
            },
            at=datetime(2026, 8, 20, tzinfo=UTC),
        )
        self.assertIsNone(route.selected_model)
        self.assertEqual("incremental_spend_denied", route.unavailable[minimax.id])
        missing = plan_route(
            self.task(),
            {minimax.id: minimax},
            {"review": RoleConfig("review", (minimax.id,), ())},
            self.profile(minimax.id),
            quotas={},
            at=datetime(2026, 8, 20, tzinfo=UTC),
        )
        self.assertIsNone(missing.selected_model)
        self.assertEqual("quota_unknown", missing.unavailable[minimax.id])

    def test_critical_chunsa_architecture_has_codex_grok_and_claude_final(self) -> None:
        included = ModelEconomics(
            plan="included",
            billing_mode="included_quota",
            source="official",
            verified_at="2026-08-20T00:00:00+00:00",
            confidence="verified",
        )
        claude = self.model(
            "claude-opus", "anthropic-claude", "anthropic", economics=included
        )
        codex = self.model(
            "gpt-5.6-sol", "openai-gpt", "openai", economics=included
        )
        grok = self.model(
            "abacus-grok-4.6", "xai-grok", "abacus", economics=included
        )
        profile = replace(
            self.profile(claude.id, codex.id, grok.id, assurance_floor="high"),
            name="chunsa",
            architect_model=claude.id,
        )
        route = plan_route(
            self.task(task_type="architecture", assurance="critical"),
            {claude.id: claude, codex.id: codex, grok.id: grok},
            {
                "architecture": RoleConfig(
                    "architecture",
                    (claude.id, codex.id, grok.id),
                    (codex.id, grok.id, claude.id),
                )
            },
            profile,
            quotas={},
            at=datetime(2026, 8, 20, tzinfo=UTC),
        )
        self.assertEqual(claude.id, route.selected_model)
        self.assertEqual((codex.id, grok.id, claude.id), route.reviewer_models)

    def test_high_chunsa_architecture_also_requires_both_second_opinions(self) -> None:
        included = ModelEconomics(
            plan="included",
            billing_mode="included_quota",
            source="official",
            verified_at="2026-08-20T00:00:00+00:00",
            confidence="verified",
        )
        claude = self.model(
            "claude-opus", "anthropic-claude", "anthropic", economics=included
        )
        codex = self.model(
            "gpt-5.6-sol", "openai-gpt", "openai", economics=included
        )
        unavailable_grok = replace(
            self.model(
                "abacus-grok-4.6", "xai-grok", "abacus", economics=included
            ),
            status="candidate",
        )
        profile = replace(
            self.profile(
                claude.id, codex.id, unavailable_grok.id, assurance_floor="high"
            ),
            name="chunsa",
            architect_model=claude.id,
        )
        route = plan_route(
            self.task(task_type="architecture", assurance="high"),
            {
                claude.id: claude,
                codex.id: codex,
                unavailable_grok.id: unavailable_grok,
            },
            {
                "architecture": RoleConfig(
                    "architecture",
                    (claude.id,),
                    (codex.id, unavailable_grok.id, claude.id),
                )
            },
            profile,
            quotas={},
            at=datetime(2026, 8, 20, tzinfo=UTC),
        )
        self.assertEqual(claude.id, route.selected_model)
        self.assertEqual("HANDOFF_REQUIRED", route.status)
        self.assertIn("independent reviewer unavailable", route.reason)


@unittest.skipUnless(hasattr(TelemetryStore, "acquire_lease"), "lease API missing")
class LeaseTests(unittest.TestCase):
    def test_sqlite_lease_is_cross_process_safe_and_expires(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TelemetryStore(Path(directory) / "telemetry.sqlite")
            first = store.acquire_lease(
                "model", max_concurrency=1, ttl_seconds=30, now=100.0
            )
            self.assertIsNotNone(first)
            self.assertIsNone(
                store.acquire_lease(
                    "model", max_concurrency=1, ttl_seconds=30, now=101.0
                )
            )
            expired = store.acquire_lease(
                "model", max_concurrency=1, ttl_seconds=30, now=131.0
            )
            self.assertIsNotNone(expired)
            self.assertNotEqual(first, expired)
            self.assertTrue(store.release_lease(str(expired)))


if __name__ == "__main__":
    unittest.main()
