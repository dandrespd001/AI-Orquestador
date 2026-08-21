from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_orchestrator.client_catalog import generate_client_config
from ai_orchestrator.extensions import ExtensionRegistry
from ai_orchestrator.models import ContractError, ExtensionManifest
from ai_orchestrator.workers import build_worker_plan, validate_worker_plan

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "extensions"


class ExtensionLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.registry = ExtensionRegistry(Path(self.temp.name) / "registry")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def manifest(
        self,
        extension_id: str,
        kind: str,
        fixture: str,
        *,
        version: str = "1.0.0",
        source_type: str = "local",
        permissions: tuple[str, ...] = (),
        tools: tuple[str, ...] = (),
        remote_domains: tuple[str, ...] = (),
        entrypoint: tuple[str, ...] = (),
        source: str | None = None,
    ) -> tuple[ExtensionManifest, Path]:
        payload = FIXTURES / fixture
        manifest = ExtensionManifest.from_mapping(
            {
                "id": extension_id,
                "kind": kind,
                "source": source or str(payload),
                "source_type": source_type,
                "version": version,
                "sha256": "",
                "license": "MIT",
                "permissions": list(permissions),
                "compatibility": ["harness:0.1.0-rc.7", "codex:any"],
                "profiles": ["default", "research"],
                "trust": "community_verified",
                "tools": list(tools),
                "remote_domains": list(remote_domains),
                "entrypoint": list(entrypoint),
            }
        )
        return manifest, payload

    def approve_and_activate(
        self, manifest: ExtensionManifest, payload: Path, profile: str = "default"
    ) -> ExtensionManifest:
        pinned = self.registry.quarantine(manifest, payload)
        report = self.registry.inspect(pinned.id)
        self.assertTrue(report.approval_possible, report.as_dict())
        self.registry.mark_tested(pinned.id, passed=True, sandbox="container")
        self.registry.approve(
            pinned.id,
            expected_sha256=pinned.sha256,
            approver="operator",
            human_approved=True,
        )
        self.registry.activate(pinned.id, profile, human_approved=True)
        return pinned

    def test_community_plugin_can_be_pinned_activated_updated_and_rolled_back(
        self,
    ) -> None:
        manifest, payload = self.manifest(
            "community-plugin", "plugin", "community-plugin"
        )
        first = self.approve_and_activate(manifest, payload)
        second_manifest = ExtensionManifest.from_mapping(
            {**manifest.as_dict(), "version": "2.0.0", "sha256": ""}
        )
        second = self.approve_and_activate(second_manifest, payload)
        self.assertEqual(first.sha256, second.sha256)
        receipt = self.registry.rollback(
            "community-plugin", "default", human_approved=True
        )
        self.assertEqual("1.0.0", receipt["to"])

    def test_skill_and_local_and_remote_mcp_are_first_class(self) -> None:
        cases = (
            self.manifest("community-skill", "skill", "community-skill"),
            self.manifest(
                "community-mcp-local",
                "mcp_local",
                "community-mcp-local",
                source_type="mcp_stdio",
                tools=("fixture_status",),
                entrypoint=("python", "server.py"),
            ),
            self.manifest(
                "community-mcp-remote",
                "mcp_remote",
                "community-mcp-remote",
                source_type="mcp_http",
                permissions=("network",),
                tools=("fixture_status",),
                remote_domains=("mcp.example.invalid",),
                source="https://mcp.example.invalid/v1",
            ),
        )
        for manifest, payload in cases:
            with self.subTest(manifest.id):
                self.approve_and_activate(manifest, payload)
        generated = generate_client_config(self.registry, "default", "harness")
        self.assertEqual(1, len(generated["skills"]))
        self.assertEqual(2, len(generated["mcp_servers"]))
        self.assertFalse(generated["provider_secrets_embedded"])
        self.assertTrue(
            all(row["environment"] == {} for row in generated["mcp_servers"])
        )

    def test_malicious_extension_stays_in_quarantine(self) -> None:
        manifest, payload = self.manifest(
            "malicious-plugin", "plugin", "malicious-plugin"
        )
        pinned = self.registry.quarantine(manifest, payload)
        report = self.registry.inspect(pinned.id)
        self.assertEqual("critical", report.risk)
        self.assertFalse(report.approval_possible)
        self.assertIn(
            "COMMUNITY_SECRET_ACCESS", {finding.code for finding in report.findings}
        )
        self.registry.mark_tested(pinned.id, passed=True, sandbox="container")
        with self.assertRaises(ContractError):
            self.registry.approve(
                pinned.id,
                expected_sha256=pinned.sha256,
                approver="operator",
                human_approved=True,
            )

    def test_symlink_payload_is_rejected_before_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory) / "payload"
            payload.mkdir()
            (payload / "safe.txt").write_text("safe", encoding="utf-8")
            (payload / "escape").symlink_to(Path(directory).parent)
            manifest = ExtensionManifest.from_mapping(
                {
                    "id": "symlink-plugin",
                    "kind": "plugin",
                    "source": str(payload),
                    "source_type": "local",
                    "version": "1.0.0",
                    "sha256": "",
                    "license": "MIT",
                    "permissions": [],
                    "compatibility": [],
                    "profiles": ["default"],
                    "trust": "private_local",
                }
            )
            with self.assertRaises(ContractError):
                self.registry.quarantine(manifest, payload)

    def test_symlink_payload_root_is_rejected(self) -> None:
        manifest, payload = self.manifest("symlink-root", "plugin", "community-plugin")
        with tempfile.TemporaryDirectory() as directory:
            link = Path(directory) / "payload"
            link.symlink_to(payload, target_is_directory=True)
            with self.assertRaises(ContractError):
                self.registry.quarantine(manifest, link)

    def test_approved_payload_is_rehashed_before_activation(self) -> None:
        manifest, payload = self.manifest("tamper-check", "plugin", "community-plugin")
        pinned = self.registry.quarantine(manifest, payload)
        self.registry.inspect(pinned.id)
        self.registry.mark_tested(pinned.id, passed=True, sandbox="container")
        self.registry.approve(
            pinned.id,
            expected_sha256=pinned.sha256,
            approver="operator",
            human_approved=True,
        )
        state = self.registry._load()
        approved = Path(
            state["extensions"][pinned.id]["versions"]["1.0.0"]["approved_path"]
        )
        (approved / "index.js").write_text("tampered\n", encoding="utf-8")
        with self.assertRaises(ContractError):
            self.registry.activate(pinned.id, "default", human_approved=True)

    def test_approved_version_cannot_be_replaced_under_the_same_reference(self) -> None:
        manifest, payload = self.manifest(
            "immutable-version", "plugin", "community-plugin"
        )
        self.approve_and_activate(manifest, payload)
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "changed"
            changed.mkdir()
            (changed / "index.js").write_text("changed\n", encoding="utf-8")
            with self.assertRaises(ContractError):
                self.registry.quarantine(manifest, changed)

    def test_administrative_profile_never_loads_community_code(self) -> None:
        manifest, payload = self.manifest("admin-blocked", "plugin", "community-plugin")
        pinned = self.registry.quarantine(manifest, payload)
        self.registry.inspect(pinned.id)
        self.registry.mark_tested(pinned.id, passed=True, sandbox="container")
        self.registry.approve(
            pinned.id,
            expected_sha256=pinned.sha256,
            approver="operator",
            human_approved=True,
        )
        with self.assertRaises(ContractError):
            self.registry.activate(pinned.id, "harness-admin", human_approved=True)

    def test_worker_has_no_home_secrets_socket_or_arbitrary_shell(self) -> None:
        manifest, payload = self.manifest(
            "worker-mcp",
            "mcp_local",
            "community-mcp-local",
            source_type="mcp_stdio",
            tools=("fixture_status",),
            entrypoint=("python", "/extension/server.py"),
        )
        with tempfile.TemporaryDirectory() as directory:
            plan = build_worker_plan(manifest, payload, Path(directory), risk="low")
        self.assertEqual("bubblewrap", plan.runtime)
        self.assertIn("--unshare-net", plan.command)
        self.assertTrue(validate_worker_plan(plan)["ok"])
        self.assertFalse(plan.broker_socket_mounted)
        self.assertFalse(plan.host_home_mounted)
        self.assertFalse(plan.provider_secrets_readable)

        shell_manifest = ExtensionManifest.from_mapping(
            {**manifest.as_dict(), "entrypoint": ["sh", "-c", "env"]}
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises(ContractError),
        ):
            build_worker_plan(shell_manifest, payload, Path(directory), risk="low")

    def test_network_request_remains_behind_a_domain_filtered_proxy(self) -> None:
        manifest, payload = self.manifest(
            "network-worker",
            "mcp_local",
            "community-mcp-local",
            source_type="mcp_stdio",
            permissions=("network",),
            tools=("fixture_status",),
            remote_domains=("api.example.invalid",),
            entrypoint=("python", "/extension/server.py"),
        )
        with tempfile.TemporaryDirectory() as directory:
            plan = build_worker_plan(manifest, payload, Path(directory), risk="low")
        self.assertTrue(plan.network_requested)
        self.assertTrue(plan.egress_proxy_required)
        self.assertFalse(plan.network_enabled)
        self.assertTrue(validate_worker_plan(plan)["ok"])


if __name__ == "__main__":
    unittest.main()
