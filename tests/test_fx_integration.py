from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from ai_orchestrator.adapters import _command, _minimal_environment
from ai_orchestrator.canary import ROUTES, _fx_canary
from ai_orchestrator.client_catalog import SUPPORTED_CLIENTS
from ai_orchestrator.config import load_models, load_profile, load_roles
from ai_orchestrator.models import TaskSpec
from ai_orchestrator.providers import load_provider_specs, validate_provider_catalog
from ai_orchestrator.routing import select_route

ROOT = Path(__file__).resolve().parents[1]


class FxIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name) / "repo"
        self.repo.mkdir()
        (self.repo / "prompt.txt").write_text("review", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def task(self, *, write_mode: str = "read_only") -> TaskSpec:
        payload: dict[str, object] = {
            "repo": str(self.repo),
            "task_type": "review",
            "sensitivity": "internal",
            "prompt_file": "prompt.txt",
            "write_mode": write_mode,
        }
        if write_mode != "read_only":
            payload["allowed_write_paths"] = ["src/value.py"]
        return TaskSpec.from_mapping(payload)

    def test_fx_is_first_class_and_temporary_in_the_common_catalog(self) -> None:
        providers = load_provider_specs(ROOT / "config")
        models = load_models(ROOT / "config")
        self.assertTrue(validate_provider_catalog(ROOT / "config")["ok"])
        self.assertIn("fx-glm-5.2-promo", providers["fx"].models)
        model = models["fx-glm-5.2-promo"]
        self.assertEqual("fx_cli", model.transport)
        self.assertEqual("zai/glm-5.2", model.route_model)
        self.assertEqual("2026-09-01", model.available_until)
        self.assertEqual(("read",), model.capabilities)
        self.assertEqual(("review", "research", "docs"), model.roles)
        self.assertIn("fx", SUPPORTED_CLIENTS)

    def test_fx_route_is_read_only_and_expires_without_paid_fallback(self) -> None:
        models = load_models(ROOT / "config")
        roles = load_roles(ROOT / "config")
        profile = replace(load_profile("default", ROOT / "config"), repo_root=None)
        # Anclado a proposito: lo que este test protege es que la ruta de FX
        # sea de solo lectura y caduque sin caer en una de pago, no que FX
        # encabece el rol. Desde que el scorer lee la tasa de exito medida, FX
        # cae por su 0 de 7 y el rol lo gana otro modelo.
        decision = select_route(
            self.task(), models, roles, profile, pin="fx-glm-5.2-promo"
        )
        self.assertEqual("fx-glm-5.2-promo", decision.selected_model)
        self.assertFalse(decision.fallback_allowed)

        expired = dict(models)
        expired["fx-glm-5.2-promo"] = replace(
            models["fx-glm-5.2-promo"], available_until="2020-01-01"
        )
        expired_decision = select_route(
            self.task(), expired, roles, profile, pin="fx-glm-5.2-promo"
        )
        self.assertEqual("NO_ROUTE", expired_decision.status)
        self.assertEqual(
            "promotion_expired",
            expired_decision.unavailable["fx-glm-5.2-promo"],
        )

        writer = select_route(
            self.task(write_mode="validated_patch"), models, roles, profile
        )
        self.assertNotEqual("fx-glm-5.2-promo", writer.selected_model)
        self.assertEqual(
            "model_not_write_capable", writer.unavailable["fx-glm-5.2-promo"]
        )

    def test_fx_uses_stdin_no_session_and_does_not_inherit_api_keys(self) -> None:
        models = load_models(ROOT / "config")
        roles = load_roles(ROOT / "config")
        profile = replace(load_profile("default", ROOT / "config"), repo_root=None)
        task = self.task()
        decision = select_route(task, models, roles, profile, pin="fx-glm-5.2-promo")
        command = _command(task, decision, task.prompt_path())
        self.assertEqual("fx", command[0])
        self.assertIn("--no-additional-dirs", command)
        self.assertIn("--no-save", command)
        self.assertIn("--json", command)
        self.assertNotIn("review", command)

        with patch.dict(
            os.environ,
            {
                "HOME": str(Path.home()),
                "AI_GATEWAY_API_KEY": "do-not-forward",
                "OPENAI_API_KEY": "do-not-forward",
            },
            clear=True,
        ):
            environment = _minimal_environment("fx_cli")
        self.assertNotIn("AI_GATEWAY_API_KEY", environment)
        self.assertNotIn("OPENAI_API_KEY", environment)
        self.assertEqual("zai/glm-5.2", environment["FX_MODEL"])
        self.assertEqual("ask", environment["FX_PERMISSION_MODE"])
        self.assertEqual("0", environment["FX_AUTO_UPGRADE"])

    def test_fx_canary_verifies_exact_output_and_model(self) -> None:
        expected = "CANARY_123"
        valid, reported = _fx_canary(
            json.dumps(
                {
                    "output": expected,
                    "exit_code": 0,
                    "model": "zai/glm-5.2",
                    "session_id": "",
                    "steps": 1,
                    "tool_calls": [],
                }
            ).encode(),
            expected,
            "zai/glm-5.2",
        )
        self.assertTrue(valid)
        self.assertEqual("zai/glm-5.2", reported)
        self.assertEqual("fx_json", ROUTES["fx"].response_format)


if __name__ == "__main__":
    unittest.main()
