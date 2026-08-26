from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_orchestrator import providers
from ai_orchestrator.config import load_models
from ai_orchestrator.control_plane import ControlPlane

ROOT = Path(__file__).resolve().parents[1]


class ProviderEconomicsSymbolsTests(unittest.TestCase):
    def test_deepseek_balance_reader_exists(self) -> None:
        self.assertTrue(hasattr(providers, "fetch_deepseek_balance"))


@unittest.skipUnless(hasattr(providers, "fetch_deepseek_balance"), "balance reader missing")
class ProviderEconomicsV2Tests(unittest.TestCase):
    def test_deepseek_refresh_records_balance_without_exposing_credential(self) -> None:
        recorded: list[object] = []

        class Store:
            def record_quota(self, snapshot: object) -> None:
                recorded.append(snapshot)

        snapshot = providers.refresh_deepseek_quota(
            Store(),
            credential_resolver=lambda **_kwargs: "deepseek-secret",
            balance_reader=lambda _key, **_kwargs: providers.QuotaSnapshot(
                provider="deepseek",
                plan="prepaid_api_balance",
                billing_mode="prepaid_balance",
                balance_usd=3.0,
                remaining=3.0,
                source="/user/balance",
                verified_at="2026-08-20T12:00:00+00:00",
                confidence="verified",
            ),
        )
        self.assertEqual(3.0, snapshot.balance_usd)
        self.assertEqual([snapshot], recorded)
        self.assertNotIn("deepseek-secret", str(snapshot.as_dict()))

    def test_missing_deepseek_credential_returns_unknown_snapshot(self) -> None:
        snapshot = providers.refresh_deepseek_quota(
            object(), credential_resolver=lambda **_kwargs: None
        )
        self.assertIsNone(snapshot.balance_usd)
        self.assertEqual("unavailable", snapshot.confidence)

    @patch("ai_orchestrator.delegation.refresh_deepseek_quota")
    def test_simulation_refreshes_deepseek_before_reading_snapshots(self, refresh) -> None:
        from ai_orchestrator.delegation import OrchestratorTaskRunner
        from ai_orchestrator.models import TaskSpec
        from ai_orchestrator.telemetry import TelemetryStore

        refresh.return_value = providers.QuotaSnapshot(
            provider="deepseek",
            plan="prepaid_api_balance",
            billing_mode="prepaid_balance",
            confidence="unavailable",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            prompt = repo / "prompt.txt"
            prompt.write_text("review", encoding="utf-8")
            spec = root / "task.json"
            task = TaskSpec(
                task_id="00000000-0000-4000-8000-000000000001",
                repo=str(repo),
                task_type="review",
                sensitivity="internal",
                output_contract="text",
                prompt_file="prompt.txt",
            )
            spec.write_text(json.dumps(task.as_dict()), encoding="utf-8")
            runner = OrchestratorTaskRunner(
                configuration_root=ROOT / "config",
                telemetry=TelemetryStore(root / "telemetry.sqlite"),
            )
            runner.simulate(spec_path=str(spec))
        refresh.assert_called_once()

    def test_deepseek_balance_uses_official_endpoint_without_secret_in_url(self) -> None:
        captured: dict[str, object] = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self, _limit: int) -> bytes:
                return json.dumps(
                    {
                        "is_available": True,
                        "balance_infos": [
                            {
                                "currency": "USD",
                                "total_balance": "2.50",
                                "granted_balance": "0.00",
                                "topped_up_balance": "2.50",
                            }
                        ],
                    }
                ).encode()

        def opener(request: object, **kwargs: object) -> Response:
            captured["url"] = request.full_url
            captured["authorization"] = request.headers.get("Authorization")
            captured.update(kwargs)
            return Response()

        secret = "deepseek-secret"
        snapshot = providers.fetch_deepseek_balance(
            secret,
            opener=opener,
            verified_at="2026-08-20T12:00:00+00:00",
        )
        self.assertEqual("https://api.deepseek.com/user/balance", captured["url"])
        self.assertNotIn(secret, captured["url"])
        self.assertEqual("Bearer deepseek-secret", captured["authorization"])
        self.assertEqual(2.5, snapshot.balance_usd)
        self.assertEqual("/user/balance", snapshot.source)

    def test_long_context_declarations_match_paid_plans(self) -> None:
        models = load_models(ROOT / "config")
        self.assertEqual(1_000_000, models["MiniMax-M3"].context_window)
        self.assertEqual(256_000, models["kimi-official"].context_window)
        self.assertEqual("kimi-code/k3-256k", models["kimi-official"].route_model)
        self.assertEqual(
            "deepseek_harness", models["deepseek-v4-flash"].transport
        )
        self.assertEqual(
            "deepseek-v4-flash", models["deepseek-v4-flash"].route_model
        )
        self.assertEqual(
            "included_quota", models["MiniMax-M3"].economics.billing_mode
        )
        for model_id in (
            "deepseek-v4-pro",
            "deepseek-v4-flash",
            "MiniMax-M3",
            "kimi-official",
            "abacus-grok-4.6",
            "gemini-free-fast",
        ):
            with self.subTest(model=model_id):
                self.assertEqual("candidate", models[model_id].rollout_stage)

    def test_inventory_exposes_candidate_rollout_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inventory = ControlPlane(
                configuration_root=ROOT / "config",
                state_root=Path(directory),
            ).models()
        minimax = next(
            row for row in inventory["models"] if row["id"] == "MiniMax-M3"
        )
        self.assertEqual("candidate", minimax["rollout"]["stage"])
        self.assertEqual(0, minimax["rollout"]["simulated_decisions"])
        self.assertEqual(20, minimax["rollout"]["read_activation_after_decisions"])
        self.assertEqual(7, minimax["rollout"]["read_activation_after_days"])


if __name__ == "__main__":
    unittest.main()
