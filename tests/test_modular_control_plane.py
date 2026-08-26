from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import time
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from ai_orchestrator.abacus import ABACUS_CLI_VERSION, plan_supercomputer_use
from ai_orchestrator.api import PANEL_HTML, HumanActionGate, PanelApplication
from ai_orchestrator.broker import UnixBrokerServer, broker_request
from ai_orchestrator.canary import ROUTES
from ai_orchestrator.capabilities import select_capability_package
from ai_orchestrator.config import load_models, load_profile, load_roles
from ai_orchestrator.control_plane import ControlPlane
from ai_orchestrator.credentials import CredentialStore
from ai_orchestrator.harness import (
    HARNESS_VERSION,
    harness_launch_spec,
    validate_harness_launch,
)
from ai_orchestrator.mcp_tools import MCPQueryServer
from ai_orchestrator.models import (
    ContractError,
    ModelConfig,
    Profile,
    ProviderSpec,
    RoleConfig,
    TaskSpec,
    UsageRecord,
)
from ai_orchestrator.provider_adapters import validate_adapter_catalog
from ai_orchestrator.providers import (
    discover_kimi_models,
    load_provider_specs,
    validate_kimi_alias,
    validate_provider_catalog,
)
from ai_orchestrator.routing import plan_route
from ai_orchestrator.sessions import SessionStore
from ai_orchestrator.updates import ComponentPin, UpdateRegistry, load_component_pins

ROOT = Path(__file__).resolve().parents[1]


class ContractAndRoutingTests(unittest.TestCase):
    def test_provider_contract_cannot_contain_a_secret(self) -> None:
        with self.assertRaises(ContractError):
            ProviderSpec.from_mapping(
                {
                    "id": "provider",
                    "endpoint": "https://example.invalid/v1",
                    "protocol": "openai_responses",
                    "models": ["model"],
                    "authentication": {"api_key": "forbidden"},
                    "quota": "included",
                    "trust": "official_subscription",
                    "capabilities": ["read"],
                    "canary": "provider",
                }
            )

    def test_provider_contract_rejects_nested_secrets_and_lookalike_loopback(
        self,
    ) -> None:
        base = {
            "id": "provider",
            "protocol": "openai_responses",
            "models": ["model"],
            "quota": "included",
            "trust": "official_subscription",
            "capabilities": ["read"],
            "canary": "provider",
        }
        with self.assertRaises(ContractError):
            ProviderSpec.from_mapping(
                {
                    **base,
                    "endpoint": "https://example.invalid/v1",
                    "authentication": {"oauth": {"refresh_token": "forbidden"}},
                }
            )
        with self.assertRaises(ContractError):
            ProviderSpec.from_mapping(
                {
                    **base,
                    "endpoint": "http://localhost.example.invalid/v1",
                    "authentication": "none",
                }
            )

    def test_usage_contract_has_no_body_fields(self) -> None:
        payload = UsageRecord(
            provider="deepseek",
            model="deepseek-v4-flash",
            outcome="SUCCESS",
            latency_ms=12,
            input_tokens=100,
            output_tokens=20,
        ).as_dict()
        self.assertFalse({"prompt", "response", "body", "content"} & set(payload))

    def test_provider_catalog_is_complete_and_minimax_is_responses_api(self) -> None:
        providers = load_provider_specs(ROOT / "config")
        self.assertTrue(validate_provider_catalog(ROOT / "config")["ok"])
        self.assertTrue(validate_adapter_catalog(ROOT / "config")["ok"])
        self.assertEqual("openai_responses", providers["minimax"].protocol)
        self.assertEqual("https://api.minimax.io/v1", providers["minimax"].endpoint)
        self.assertFalse(providers["minimax"].incremental_spend_allowed)

    def test_verified_free_omniroute_routes_start_as_candidates(self) -> None:
        providers = load_provider_specs(ROOT / "config")
        models = load_models(ROOT / "config")
        expected = {
            "opencode-mimo-free": (
                "opencode-free",
                "oc/mimo-v2.5-free",
                1_048_576,
                "candidate",
            ),
            "openrouter-north-code-free": (
                "openrouter-free",
                "openrouter/cohere/north-mini-code:free",
                256_000,
                "available",
            ),
            "zai-web-glm-5.3": (
                "zai-web",
                "zai-web/glm-5.3",
                200_000,
                "candidate",
            ),
            "t3-free-qwen3-235b": (
                "t3-web",
                "t3-web/qwen3-235b",
                128_000,
                "candidate",
            ),
        }
        for model_id, (
            provider_id,
            route_model,
            context_window,
            status,
        ) in expected.items():
            with self.subTest(model=model_id):
                self.assertIn(provider_id, providers)
                model = models[model_id]
                self.assertEqual(provider_id, model.provider)
                self.assertEqual(route_model, model.route_model)
                self.assertEqual(context_window, model.context_window)
                self.assertEqual(status, model.status)
                self.assertEqual("web_free", model.trust_tier)
                self.assertEqual("free", model.cost_tier)
                self.assertEqual("patch_api_writer", model.execution_profile)

    def test_kimi_models_are_discovered_without_reading_oauth_tokens(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self, _limit: int) -> bytes:
                return b'{"models":[{"id":"kimi-code/k3"}]}'

        aliases = discover_kimi_models(
            "http://127.0.0.1:8123", opener=lambda *_args, **_kwargs: Response()
        )
        receipt = validate_kimi_alias(
            "kimi-code/k3",
            aliases,
            {
                "providers": [
                    {
                        "provider": "moonshot",
                        "model": "kimi-code/k3",
                        "reported_model": "kimi-code/k3",
                        "status": "PASSED",
                        "provenance_verified": True,
                    }
                ]
            },
        )
        self.assertTrue(receipt["valid"])

    def test_security_and_budget_filter_before_quality_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "src").mkdir()
            (repo / "src" / "value.py").write_text("VALUE=1\n", encoding="utf-8")
            task = TaskSpec.from_mapping(
                {
                    "repo": str(repo),
                    "task_type": "implementation",
                    "sensitivity": "internal",
                    "input_paths": ["src"],
                }
            )
            included = ModelConfig(
                "included",
                "a",
                "local",
                "codex_cli",
                "included",
                True,
                "available",
                ("implementation",),
                "internal",
                1,
                quality_score=0.45,
                included_quota=True,
            )
            paid = ModelConfig(
                "paid",
                "b",
                "remote",
                "qwen_minimax_cli",
                "paid",
                True,
                "available",
                ("implementation",),
                "internal",
                1,
                quality_score=1.0,
                included_quota=False,
                marginal_cost_per_million=20.0,
            )
            role = RoleConfig("implementation", ("paid", "included"), ())
            profile = Profile(
                "default",
                None,
                ("internal",),
                ("paid", "included"),
                (),
                (),
                (),
                "direct",
                allow_paid_routes=False,
            )
            result = plan_route(
                task,
                {"included": included, "paid": paid},
                {"implementation": role},
                profile,
            )
            self.assertEqual("included", result.selected_model)
            self.assertFalse(result.fallback_allowed)
            self.assertFalse(result.budget["incremental_spend_allowed"])

    def test_task_scoped_package_reduces_context_by_more_than_25_percent(self) -> None:
        receipt = select_capability_package(
            "games", client="harness", root=ROOT / "config"
        )
        self.assertEqual(1, receipt["loaded_package_count"])
        self.assertGreaterEqual(receipt["context_reduction_percent"], 25.0)

    def test_chunsa_architecture_is_hard_pinned_to_claude(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "src").mkdir()
            task = TaskSpec.from_mapping(
                {
                    "repo": str(repo),
                    "task_type": "architecture",
                    "sensitivity": "internal",
                    "input_paths": ["src"],
                }
            )
            profile = replace(
                load_profile("chunsa", ROOT / "config"), repo_root=str(repo)
            )
            route = plan_route(
                task,
                load_models(ROOT / "config"),
                load_roles(ROOT / "config"),
                profile,
                observations={
                    "gpt-5.6-sol": {"quality": 1.0},
                    "claude-opus": {"quality": 0.1},
                },
            )
        self.assertEqual("claude-opus", route.selected_model)

    def test_every_required_provider_has_a_bounded_canary(self) -> None:
        self.assertTrue(
            {
                "openai",
                "anthropic",
                "deepseek",
                "minimax",
                "moonshot",
                "omniroute",
                "abacus",
            }
            <= set(ROUTES)
        )


class CredentialRotationTests(unittest.TestCase):
    def test_rotation_is_atomic_private_and_write_only(self) -> None:
        captured: list[bytes] = []

        def encryptor(_name: str, value: bytes, output: Path) -> None:
            captured.append(value)
            output.write_bytes(b"encrypted-v1")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "config"
            store = CredentialStore(root, encryptor=encryptor)
            secret = "super-secret-value"
            receipt = store.store(
                "provider-key",
                secret,
                provider="provider",
                human_approved=True,
            )
            target = root / "credentials" / "provider-key.cred"
            self.assertEqual([secret.encode()], captured)
            self.assertEqual(0o700, target.parent.stat().st_mode & 0o777)
            self.assertEqual(0o600, target.stat().st_mode & 0o777)
            self.assertNotIn(secret, json.dumps(receipt))
            self.assertFalse(receipt["secret_returned"])

            def failing(_name: str, _value: bytes, _output: Path) -> None:
                raise ContractError("failed")

            with self.assertRaises(ContractError):
                CredentialStore(root, encryptor=failing).rotate(
                    "provider-key",
                    "replacement-value",
                    provider="provider",
                    human_approved=True,
                )
            self.assertEqual(b"encrypted-v1", target.read_bytes())

    def test_rotation_requires_human_approval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = CredentialStore(Path(directory), encryptor=lambda *_args: None)
            with self.assertRaises(ContractError):
                store.store("key", "12345678", provider="provider")


class SessionAndHarnessTests(unittest.TestCase):
    def test_pinned_session_survives_seven_day_purge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory) / "sessions.sqlite")
            store.put("normal", {"message": "old"})
            store.put("pinned", {"message": "keep"}, pinned=True)
            with closing(store._connect()) as connection, connection:
                connection.execute(
                    "UPDATE sessions SET updated_at='2020-01-01T00:00:00+00:00'"
                )
            preview = store.preview_purge()
            self.assertEqual(1, preview["sessions"])
            store.purge()
            self.assertIsNone(store.get("normal"))
            self.assertEqual({"message": "keep"}, store.get("pinned"))
            self.assertFalse(store.inventory()["content_included"])

    def test_harness_is_pinned_loopback_and_receives_no_provider_keys(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {
                    "HOME": directory,
                    "PATH": "/usr/bin",
                    "DEEPSEEK_API_KEY": "forbidden",
                },
                clear=True,
            ),
        ):
            spec = harness_launch_spec(state_root=Path(directory) / "state")
            report = validate_harness_launch(spec, require_socket=False)
        self.assertTrue(report["ok"])
        self.assertEqual("127.0.0.1", spec.host)
        self.assertIn(HARNESS_VERSION, " ".join(spec.command))
        self.assertIn(
            ("--host", "127.0.0.1"), tuple(zip(spec.command, spec.command[1:]))
        )
        self.assertIn(("--port", "3080"), tuple(zip(spec.command, spec.command[1:])))
        self.assertNotIn("DEEPSEEK_API_KEY", spec.environment)
        self.assertFalse(spec.community_code_loaded)


class UpdateTests(unittest.TestCase):
    def test_updates_validate_in_parallel_slot_and_keep_rollback(self) -> None:
        pins = load_component_pins(ROOT / "config")
        self.assertEqual("0.1.0-rc.7", pins["deepseek-harness"].version)
        self.assertEqual("2.6.9", pins["abacus-cli"].version)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.tgz"
            artifact.write_bytes(b"version-one")
            registry = UpdateRegistry(root / "updates")
            component = ComponentPin(
                "component", "1.0.0", "local", None, ("contracts",), ("security",)
            )
            registry.stage(component, artifact)
            registry.record_validation(
                "component",
                compatibility_passed=True,
                security_passed=True,
                evaluation_passed=True,
            )
            first = registry.promote("component", human_approved=True)
            self.assertFalse(first["rollback_available"])
            artifact.write_bytes(b"version-two")
            registry.stage(component, artifact, version="2.0.0")
            registry.record_validation(
                "component",
                compatibility_passed=True,
                security_passed=True,
                evaluation_passed=True,
            )
            second = registry.promote("component", human_approved=True)
            self.assertTrue(second["rollback_available"])
            rolled_back = registry.rollback("component", human_approved=True)
            self.assertEqual("1.0.0", rolled_back["active"]["version"])

    def test_tampered_update_slot_cannot_be_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.tgz"
            artifact.write_bytes(b"approved-content")
            registry = UpdateRegistry(root / "updates")
            component = ComponentPin(
                "component", "1.0.0", "local", None, ("contracts",), ("security",)
            )
            staged = registry.stage(component, artifact)
            Path(staged["path"]).write_bytes(b"tampered-content")
            registry.record_validation(
                "component",
                compatibility_passed=True,
                security_passed=True,
                evaluation_passed=True,
            )
            with self.assertRaises(ContractError):
                registry.promote("component", human_approved=True)

    def test_parallel_staging_does_not_lose_registry_updates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.tgz"
            artifact.write_bytes(b"shared-artifact")
            first = UpdateRegistry(root / "updates")
            second = UpdateRegistry(root / "updates")
            barrier = threading.Barrier(2)

            def synchronize_first_load(registry: UpdateRegistry) -> None:
                original = registry._load
                pending = True

                def load() -> dict[str, object]:
                    nonlocal pending
                    state = original()
                    if pending:
                        pending = False
                        barrier.wait(timeout=2)
                    return state

                registry._load = load  # type: ignore[method-assign]

            synchronize_first_load(first)
            synchronize_first_load(second)
            errors: list[BaseException] = []

            def stage(registry: UpdateRegistry, name: str) -> None:
                try:
                    registry.stage(
                        ComponentPin(name, "1.0.0", "local", None, (), ()),
                        artifact,
                    )
                except Exception as exc:  # noqa: BLE001 - thread forwards to assertion
                    errors.append(exc)

            threads = [
                threading.Thread(target=stage, args=(first, "first")),
                threading.Thread(target=stage, args=(second, "second")),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=3)

            self.assertFalse(errors)
            self.assertEqual(
                {"first", "second"},
                set(UpdateRegistry(root / "updates").inventory()["components"]),
            )

    def test_abacus_supercomputer_cannot_buy_credits_silently(self) -> None:
        self.assertEqual("2.6.9", ABACUS_CLI_VERSION)
        with self.assertRaises(ContractError):
            plan_supercomputer_use(
                available_included_credits=10,
                estimated_minutes=10,
                human_approved=False,
            )
        receipt = plan_supercomputer_use(
            available_included_credits=10,
            estimated_minutes=10,
            human_approved=True,
        )
        self.assertEqual(2, receipt["estimated_credits"])
        self.assertFalse(receipt["automatic_purchase"])


class BrokerAndPanelTests(unittest.TestCase):
    def test_private_broker_exposes_no_write_method(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broker" / "control.sock"
            server = UnixBrokerServer(
                path, handler=lambda request: {"method": request["method"]}
            )
            try:
                server.open()
            except PermissionError:
                self.skipTest("sandbox forbids AF_UNIX bind")
            try:
                self.assertEqual(0o600, path.stat().st_mode & 0o777)
                thread = threading.Thread(target=server.serve_once)
                thread.start()
                response = broker_request("health", path=path)
                thread.join(timeout=2)
                self.assertTrue(response["ok"])
                self.assertEqual("health", response["result"]["method"])
            finally:
                server.close()

    def test_slow_peer_does_not_block_other_broker_clients(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broker" / "control.sock"
            server = UnixBrokerServer(
                path,
                handler=lambda request: {"method": request["method"]},
                connection_timeout_seconds=0.2,
                max_workers=2,
            )
            try:
                server.open()
            except PermissionError:
                self.skipTest("sandbox forbids AF_UNIX bind")
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            slow = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                slow.connect(str(path))
                slow.sendall(b'{"method":"health"')
                started = time.monotonic()
                response = broker_request("health", path=path, timeout_seconds=1)
                elapsed = time.monotonic() - started
                self.assertTrue(response["ok"])
                self.assertLess(elapsed, 0.8)
            finally:
                slow.close()
                server.close()
                thread.join(timeout=1)

    def test_mcp_tool_inventory_has_no_mutating_tools(self) -> None:
        server = MCPQueryServer(object())
        response = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        names = {tool["name"] for tool in response["result"]["tools"]}
        self.assertFalse(
            any(
                word in name
                for name in names
                for word in ("secret", "install", "activate", "write")
            )
        )

    def test_control_plane_exposes_a_non_fallback_route_panel(self) -> None:
        control = ControlPlane(configuration_root=ROOT / "config")
        payload = control.query("routes")
        self.assertFalse(payload["silent_fallback"])
        self.assertIn("quota", payload["hard_gates"])

    def test_panel_never_replays_a_saved_credential(self) -> None:
        class FakeControl:
            def __init__(self) -> None:
                self.saved: str | None = None

            def store_credential(
                self, _provider: str, _name: str, value: str, **_kwargs: object
            ):
                self.saved = value
                return {"status": "stored", "secret_returned": False}

        control = FakeControl()
        gate = HumanActionGate("human-token")
        app = PanelApplication(control, gate)  # type: ignore[arg-type]
        secret = "never-render-this-value"
        status, _content_type, response = app.dispatch(
            "POST",
            "/api/credentials/minimax",
            headers={"X-Human-Approval": "human-token"},
            body=json.dumps(
                {"credential": "minimax-api-key", "value": secret}
            ).encode(),
        )
        self.assertEqual(200, status)
        self.assertEqual(secret, control.saved)
        self.assertNotIn(secret.encode(), response)
        self.assertNotIn(secret, PANEL_HTML)
        self.assertIn('type="password" value=""', PANEL_HTML)
        with self.assertRaises(ContractError):
            app.dispatch(
                "POST",
                "/api/credentials/minimax",
                headers={"X-Human-Approval": "human-token"},
                body=json.dumps(
                    {"credential": "minimax-api-key", "value": secret}
                ).encode(),
            )


if __name__ == "__main__":
    unittest.main()
