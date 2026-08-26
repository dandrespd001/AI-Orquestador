from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from ai_orchestrator import harness
from ai_orchestrator.adapters import _command, _minimal_environment
from ai_orchestrator.cli import _credential_readiness, build_parser
from ai_orchestrator.config import load_models, load_profile, load_roles
from ai_orchestrator.harness_bridge import HarnessBridge
from ai_orchestrator.mcp_install import install_native_mcp
from ai_orchestrator.mcp_tools import MCP_TOOLS
from ai_orchestrator.models import ContractError, RouteDecision, TaskSpec
from ai_orchestrator.routing import plan_route
from ai_orchestrator.updates import UpdateRegistry, load_component_pins

ROOT = Path(__file__).resolve().parents[1]


class ComponentManifestV2Tests(unittest.TestCase):
    def test_manifest_v2_records_current_versions_and_update_channels(self) -> None:
        pins = load_component_pins(ROOT / "config")
        harness = pins["deepseek-harness"]

        self.assertEqual("0.1.1-rc.2", harness.version)
        self.assertEqual(
            "sha512-UP1UIh6q3Gme/yXRn/QL2P8IsVlv8Shpg22TRJIZPsCRWLm4CBiA1MUvXmJAfsOEETBMLAl+xWPtFw6ICsN3wg==",
            harness.integrity,
        )
        self.assertEqual("node_modules/.bin/dsh", harness.active_executable)
        self.assertFalse(harness.auto_update)
        self.assertTrue(harness.rollback_eligible)
        self.assertEqual("0.38.0", pins["kimi-cli"].version)
        self.assertEqual("0.22.1", pins["qwen-cli"].version)
        self.assertTrue(all(pin.install_slot for pin in pins.values()))
        self.assertTrue(all(pin.active in {True, False} for pin in pins.values()))
        self.assertTrue(all(pin.update_policy == "manual_promote" for pin in pins.values()))
        self.assertTrue(all(pin.update_channel in {"latest", "stable"} for pin in pins.values()))

    def test_inventory_is_pure_when_state_does_not_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "not-created"
            registry = UpdateRegistry(state, create=False)

            self.assertEqual({}, registry.inventory()["components"])
            self.assertFalse(state.exists())

    def test_promotion_changes_the_executable_consumed_by_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            install = root / "install"
            executable = install / "bin" / "dsh"
            executable.parent.mkdir(parents=True)
            executable.write_text("#!/bin/sh\necho rc2\n", encoding="utf-8")
            executable.chmod(0o755)
            (install / "bundle").mkdir()
            for profile in ("orchestrator-web", "orchestrator-worker"):
                (install / "profiles" / profile).mkdir(parents=True)
            lock = install / "package-lock.json"
            lock.write_text('{"lockfileVersion":3}\n', encoding="utf-8")
            digest = hashlib.sha256(lock.read_bytes()).hexdigest()

            pins = load_component_pins(ROOT / "config")
            component = pins["deepseek-harness"]
            registry = UpdateRegistry(root / "registry")
            staged = registry.stage_directory(
                component,
                install,
                expected_lock_sha256=digest,
                executable="bin/dsh",
            )
            registry.record_validation(
                component.name,
                compatibility_passed=True,
                security_passed=True,
                evaluation_passed=True,
            )
            registry.promote(component.name, human_approved=True)

            active = registry.resolve_active_executable(component.name)
            self.assertEqual("#!/bin/sh\necho rc2\n", active.read_text(encoding="utf-8"))
            self.assertTrue((registry.root / component.name / "current").is_symlink())
            self.assertEqual(staged["slot"], str(active.parents[1]))

    def test_prune_inactive_slots_keeps_only_current_and_clears_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            install = root / "install"
            executable = install / "bin" / "dsh"
            executable.parent.mkdir(parents=True)
            lock = install / "package-lock.json"
            pins = load_component_pins(ROOT / "config")
            component = pins["deepseek-harness"]
            registry = UpdateRegistry(root / "registry")

            for version in ("old", "current"):
                executable.write_text(
                    f"#!/bin/sh\necho {version}\n", encoding="utf-8"
                )
                executable.chmod(0o755)
                lock.write_text(
                    json.dumps({"lockfileVersion": 3, "version": version}) + "\n",
                    encoding="utf-8",
                )
                digest = hashlib.sha256(lock.read_bytes()).hexdigest()
                registry.stage_directory(
                    component,
                    install,
                    expected_lock_sha256=digest,
                    executable="bin/dsh",
                    version=version,
                )
                registry.record_validation(
                    component.name,
                    compatibility_passed=True,
                    security_passed=True,
                    evaluation_passed=True,
                )
                registry.promote(component.name, human_approved=True)

            result = registry.prune_inactive_slots(
                component.name, human_approved=True
            )
            inventory = registry.inventory()["components"][component.name]
            active, _ = registry.resolve_active_slot(component.name)
            installed = [
                path
                for version_root in (registry.slots / component.name).iterdir()
                for path in version_root.iterdir()
            ]

            self.assertEqual("pruned", result["status"])
            self.assertEqual([active], installed)
            self.assertIsNone(inventory["rollback"])
            self.assertIsNone(inventory["staged"])

    def test_prune_inactive_slots_requires_explicit_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = UpdateRegistry(Path(temporary) / "registry")
            with self.assertRaisesRegex(ContractError, "human action"):
                registry.prune_inactive_slots("deepseek-harness")

    def test_pnpm_slot_preserves_internal_bin_symlink_and_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            install = root / "install"
            real_executable = install / "node_modules" / "dsh" / "bin" / "dsh.js"
            real_executable.parent.mkdir(parents=True)
            real_executable.write_text("#!/usr/bin/env node\n", encoding="utf-8")
            real_executable.chmod(0o755)
            bin_dir = install / "node_modules" / ".bin"
            bin_dir.mkdir()
            (bin_dir / "dsh").symlink_to("../dsh/bin/dsh.js")
            lock = install / "pnpm-lock.yaml"
            lock.write_text("lockfileVersion: '9.0'\n", encoding="utf-8")
            component = load_component_pins(ROOT / "config")["deepseek-harness"]
            registry = UpdateRegistry(root / "updates")

            registry.stage_directory(
                component,
                install,
                expected_lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),
            )

            escaped = root / "escaped"
            escaped.mkdir()
            (escaped / "pnpm-lock.yaml").write_text("x", encoding="utf-8")
            (escaped / "dsh").symlink_to("/bin/sh")
            with self.assertRaisesRegex(Exception, "missing or unsafe"):
                registry.stage_directory(
                    component,
                    escaped,
                    expected_lock_sha256=hashlib.sha256(
                        (escaped / "pnpm-lock.yaml").read_bytes()
                    ).hexdigest(),
                    executable="dsh",
                )


class HarnessContractTests(unittest.TestCase):
    def test_harness_v2_contract_symbols_exist(self) -> None:
        self.assertTrue(hasattr(harness, "HarnessReceiptV2"))
        self.assertTrue(hasattr(harness, "WorkspaceRegistry"))
        self.assertTrue(hasattr(harness, "run_harness_worker"))

    def test_worker_profile_has_a_matching_permission_preset_and_mcp_config(self) -> None:
        worker = (
            ROOT
            / "integrations/deepseek-harness/profiles/orchestrator-worker/cordis.patch.yml"
        ).read_text(encoding="utf-8")
        bundle = (
            ROOT / "integrations/deepseek-harness/bundle/cordis.patch.yml"
        ).read_text(encoding="utf-8")

        self.assertIn("defaultPreset: orchestrator-worker", worker)
        self.assertIn("sandbox: workspace-write", worker)
        self.assertIn("approval: never", worker)
        self.assertIn("AI_ORCHESTRATOR_CONFIG:", bundle)
        self.assertIn("AI_ORCHESTRATOR_SOURCE_CONFIG:", bundle)

    def test_ephemeral_profile_from_immutable_slot_is_writable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            slot = root / "slot"
            bundle_file = slot / "bundle" / "package.json"
            profile_file = slot / "profiles" / "orchestrator-worker" / "cordis.yml"
            bundle_file.parent.mkdir(parents=True)
            profile_file.parent.mkdir(parents=True)
            bundle_file.write_text("{}\n", encoding="utf-8")
            profile_file.write_text("plugins: []\n", encoding="utf-8")
            bundle_file.chmod(0o444)
            profile_file.chmod(0o444)

            composed = harness._materialize_profile(
                slot, root / "runtime", "orchestrator-worker", ephemeral=True
            )

            self.assertTrue(composed)
            runtime_profile = root / "runtime/profiles/orchestrator-worker/cordis.yml"
            self.assertTrue(runtime_profile.stat().st_mode & 0o200)

    def test_harness_uses_active_slot_and_clean_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = UpdateRegistry(root / "updates")
            component = load_component_pins(ROOT / "config")["deepseek-harness"]
            install = root / "install"
            executable = install / "node_modules" / ".bin" / "dsh"
            executable.parent.mkdir(parents=True)
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            executable.chmod(0o755)
            (install / "bundle").mkdir()
            (install / "bundle" / "package.json").write_text("{}\n", encoding="utf-8")
            for profile in ("orchestrator-web", "orchestrator-worker"):
                (install / "profiles" / profile).mkdir(parents=True)
                (install / "profiles" / profile / "cordis.yml").write_text(
                    "plugins: []\n", encoding="utf-8"
                )
            lock = install / "package-lock.json"
            lock.write_text("{}\n", encoding="utf-8")
            registry.stage_directory(
                component,
                install,
                expected_lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),
            )
            registry.record_validation(
                component.name,
                compatibility_passed=True,
                security_passed=True,
                evaluation_passed=True,
            )
            registry.promote(component.name, human_approved=True)

            spec = harness.harness_launch_spec(
                state_root=root / "state",
                registry_root=registry.root,
                profile="orchestrator-web",
            )

            self.assertEqual(harness.HARNESS_VERSION, "0.1.1-rc.2")
            self.assertEqual(str(registry.resolve_active_executable(component.name)), spec.command[0])
            self.assertNotIn("npx", spec.command)
            self.assertEqual("orchestrator-web", spec.profile)
            self.assertNotIn("DEEPSEEK_API_KEY", spec.environment)

            for path in (root / "state" / "harness" / "bundle").iterdir():
                path.chmod(0o444)
            for path in (
                root / "state" / "harness" / "profiles" / "orchestrator-web"
            ).iterdir():
                path.chmod(0o444)
            repeated = harness.harness_launch_spec(
                state_root=root / "state",
                registry_root=registry.root,
                profile="orchestrator-web",
            )
            self.assertEqual(spec.command, repeated.command)
            runtime_profile = (
                root
                / "state"
                / "harness"
                / "profiles"
                / "orchestrator-web"
                / "cordis.yml"
            )
            self.assertTrue(runtime_profile.stat().st_mode & 0o200)

    def test_receipt_v2_never_serializes_content(self) -> None:
        receipt = harness.HarnessReceiptV2(
            run_id="a2b8aa0d-cab0-4aa8-9957-21b7ce0ec503",
            task_hash="a" * 64,
            profile="orchestrator-worker",
            provider="deepseek",
            model="deepseek-v4-flash",
            harness_version="0.1.1-rc.2",
            transport="deepseek_harness",
            started_at="2026-08-25T12:00:00+00:00",
            finished_at="2026-08-25T12:01:00+00:00",
            usage={"input_tokens": 10, "output_tokens": 4},
            status="PATCH_READY",
            artifact={"sha256": "b" * 64},
            gates=[{"name": "unit", "status": "PASSED"}],
        ).as_dict()

        serialized = json.dumps(receipt)
        self.assertEqual(2, receipt["schema_version"])
        self.assertNotIn("prompt", serialized.lower())
        self.assertNotIn("response", serialized.lower())
        self.assertNotIn("reasoning", serialized.lower())

    def test_harness_versions_can_advance_but_insecure_rc7_stays_blocked(self) -> None:
        self.assertTrue(harness.harness_version_supported("0.1.2"))
        self.assertTrue(harness.harness_version_supported("0.2.0-rc.1"))
        self.assertFalse(harness.harness_version_supported("0.1.0-rc.7"))

    def test_workspace_allowlist_requires_git_root_and_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            repo.mkdir()
            (repo / ".git").mkdir()
            registry = harness.WorkspaceRegistry(root / "workspaces.json")

            with self.assertRaisesRegex(ValueError, "human approval"):
                registry.add(repo)
            registry.add(repo, human_approved=True)
            self.assertEqual([str(repo.resolve())], registry.list()["workspaces"])
            drop_in = root / "systemd" / "20-workspaces.conf"
            with self.assertRaisesRegex(ValueError, "human approval"):
                registry.write_systemd_dropin(drop_in)
            result = registry.write_systemd_dropin(
                drop_in, human_approved=True
            )
            self.assertEqual(str(drop_in.resolve()), result["path"])
            self.assertEqual(
                f'[Service]\nReadWritePaths="{repo.resolve()}"\n',
                drop_in.read_text(encoding="utf-8"),
            )

            outside = root / "plain"
            outside.mkdir()
            with self.assertRaisesRegex(ValueError, "Git root"):
                registry.add(outside, human_approved=True)


class PublicSurfaceTests(unittest.TestCase):
    def test_strict_mcp_has_exactly_four_policy_tools(self) -> None:
        self.assertEqual(
            [
                "orchestrator_inventory",
                "orchestrator_diagnostics",
                "orchestrator_simulate_route",
                "orchestrator_submit_task",
            ],
            [tool["name"] for tool in MCP_TOOLS],
        )

    def test_chunsa_profile_encodes_worker_retention_and_codex_policy(self) -> None:
        profile = load_profile("chunsa", ROOT / "config")
        self.assertEqual(30, profile.session_retention_days)
        self.assertEqual(5, profile.codex_daily_limit)
        self.assertTrue(profile.codex_important_only)
        self.assertEqual(1, profile.max_writers_per_repo)
        self.assertEqual(("deepseek-v4-pro", "deepseek-v4-flash"), profile.deepseek_models)

    def test_codex_pin_respects_important_flag_and_shared_daily_counter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            models = load_models(ROOT / "config")
            roles = load_roles(ROOT / "config")
            profile = load_profile("chunsa", ROOT / "config")
            profile = dataclasses.replace(profile, repo_root=str(repo))
            base = {
                "repo": str(repo),
                "task_type": "docs",
                "sensitivity": "internal",
                "write_mode": "read_only",
            }
            ordinary = TaskSpec.from_mapping(base)
            denied = plan_route(
                ordinary, models, roles, profile, pin="gpt-5.6-sol"
            )
            self.assertEqual(
                "codex_requires_important_task", denied.unavailable["gpt-5.6-sol"]
            )
            important = TaskSpec.from_mapping({**base, "important": True})
            exhausted = plan_route(
                important,
                models,
                roles,
                profile,
                pin="gpt-5.6-sol",
                codex_uses_today=5,
            )
            self.assertEqual("codex_daily_limit", exhausted.unavailable["gpt-5.6-sol"])

    def test_coherent_cli_families_and_version_are_parseable(self) -> None:
        parser = build_parser()
        for argv in (
            ["status"],
            ["doctor"],
            ["task", "submit", "--spec", "task.json"],
            ["model", "list"],
            ["client", "list"],
            ["mcp", "status"],
            ["harness", "status"],
            ["component", "list"],
            ["extension", "list"],
            ["docs", "check"],
        ):
            with self.subTest(argv=argv):
                self.assertIsNotNone(parser.parse_args(argv).func)

    def test_native_mcp_install_preserves_unrelated_config_and_backs_up(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            qwen = home / ".qwen" / "settings.json"
            qwen.parent.mkdir()
            qwen.write_text(
                '{"theme":"dark","mcpServers":{'
                '"foreign":{"x":1},'
                '"ai-orchestrator":{"command":"/opt/ai-orchestrator",'
                '"args":["mcp-serve"]}}}\n'
            )
            result = install_native_mcp(
                "qwen", home=home, command="/opt/ai-orchestrator"
            )
            payload = json.loads(qwen.read_text())
            self.assertEqual("dark", payload["theme"])
            self.assertEqual({"x": 1}, payload["mcpServers"]["foreign"])
            self.assertNotIn("ai-orchestrator", payload["mcpServers"])
            server = payload["mcpServers"]["ai-orchestrator-chunsa"]
            self.assertEqual("/opt/ai-orchestrator", server["command"])
            self.assertEqual(
                ["mcp", "serve", "--profile", "chunsa"], server["args"]
            )
            self.assertTrue(Path(result["backup"]).is_file())

    def test_read_only_cli_commands_create_no_state_or_home_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            home = root / "home"
            home.mkdir()
            state = root / "state"
            environment = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": str(home),
                "AI_ORCHESTRATOR_CONFIG": str(ROOT / "config"),
                "AI_ORCHESTRATOR_SOURCE_CONFIG": str(ROOT / "config"),
                "AI_ORCHESTRATOR_STATE": str(state),
                "PYTHONPATH": str(ROOT / "src"),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            for arguments in (
                ["--version"],
                ["--help"],
                ["status", "--profile", "chunsa"],
                ["doctor", "--profile", "chunsa", "--check", "compliance"],
                ["component", "list"],
                ["extension", "list"],
            ):
                completed = subprocess.run(
                    ["python3.12", "-m", "ai_orchestrator", *arguments],
                    cwd=ROOT,
                    env=environment,
                    capture_output=True,
                    timeout=20,
                    check=False,
                )
                self.assertEqual(0, completed.returncode, completed.stderr.decode())
            self.assertFalse(state.exists())
            self.assertEqual([], list(home.iterdir()))

    def test_storage_maintenance_preview_does_not_mutate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "audited.sqlite"
            database.touch()
            arguments = build_parser().parse_args(
                [
                    "storage",
                    "maintain",
                    "--apply-permissions",
                    "--compact",
                    str(database),
                ]
            )
            with (
                patch("ai_orchestrator.cli.SQLITE_TARGETS", ((database, "test"),)),
                patch("ai_orchestrator.cli.apply_private_permissions") as permissions,
                patch("ai_orchestrator.cli.compact_sqlite") as compact,
                patch("ai_orchestrator.cli.audit_storage", return_value={"ok": True}),
                patch("ai_orchestrator.cli._print"),
            ):
                self.assertEqual(0, arguments.func(arguments))
            permissions.assert_not_called()
            compact.assert_not_called()


class DeepSeekHarnessTransportTests(unittest.TestCase):
    def test_readiness_requires_the_bridge_not_a_direct_deepseek_key(self) -> None:
        readiness = _credential_readiness(
            {"deepseek_direct": False, "minimax": True, "omniroute": True}
        )

        self.assertTrue(readiness["ok"])
        self.assertEqual("omniroute_local_bridge", readiness["deepseek_transport"])

    def test_deepseek_models_route_only_through_harness(self) -> None:
        models = load_models(ROOT / "config")
        for model_id in ("deepseek-v4-pro", "deepseek-v4-flash"):
            with self.subTest(model=model_id):
                model = models[model_id]
                self.assertEqual("deepseek_harness", model.transport)
                self.assertEqual("prepaid_api_balance_only", model.economics.plan)
                self.assertEqual("prepaid_balance_only", model.economics.billing_mode)
                self.assertEqual("quota_unavailable", model.status)

    def test_headless_command_uses_stdin_and_never_qwen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            prompt = repo / "task.txt"
            prompt.write_text("change it", encoding="utf-8")
            task = TaskSpec.from_mapping(
                {
                    "repo": str(repo),
                    "task_type": "implementation",
                    "sensitivity": "internal",
                    "prompt_file": "task.txt",
                    "write_mode": "read_only",
                }
            )
            decision = RouteDecision(
                schema_version=2,
                decision_id="d",
                task_id=task.task_id,
                profile="chunsa",
                status="READY",
                selected_model="deepseek-v4-flash",
                requested_route_model="deepseek-v4-flash",
                provider="deepseek",
                family="deepseek",
                transport="deepseek_harness",
                reviewer_model="claude-sonnet",
                alternatives=(),
                unavailable={},
                reason="eligible",
                auto_execute=True,
                approval_required=False,
                context_sha256=task.context_sha256(),
                execution_profile="native_patch_writer",
            )

            command = _command(task, decision, prompt)
            environment = _minimal_environment("deepseek_harness")

            self.assertNotIn("qwen", command)
            self.assertEqual(["harness", "worker"], command[3:5])
            self.assertNotIn("change it", " ".join(command))
            self.assertNotIn("DEEPSEEK_API_KEY", environment)
            self.assertNotIn("OPENAI_API_KEY", environment)

    def test_worker_uses_ephemeral_prompt_file_and_cleans_its_session(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "repo"
            workspace.mkdir()
            subprocess.run(["git", "init", "-q", str(workspace)], check=True)
            (workspace / "marker.txt").write_text("canonical\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(workspace), "add", "marker.txt"], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(workspace),
                    "-c",
                    "user.name=Harness Test",
                    "-c",
                    "user.email=harness@example.invalid",
                    "commit",
                    "-qm",
                    "initial",
                ],
                check=True,
            )
            install = root / "install"
            executable = install / "node_modules" / ".bin" / "dsh"
            executable.parent.mkdir(parents=True)
            executable.write_text(
                "#!/bin/sh\n"
                "case \"$*\" in *sensitive*) exit 90;; esac\n"
                "test -f \"$AI_ORCHESTRATOR_TASK_FILE\" || exit 91\n"
                "grep -q sensitive \"$AI_ORCHESTRATOR_TASK_FILE\" || exit 92\n"
                "printf 'worker-result'\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            lock = install / "package-lock.json"
            lock.write_text("{}\n", encoding="utf-8")
            component = load_component_pins(ROOT / "config")["deepseek-harness"]
            updates = UpdateRegistry(root / "updates")
            updates.stage_directory(
                component,
                install,
                expected_lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),
            )
            updates.record_validation(
                component.name,
                compatibility_passed=True,
                security_passed=True,
                evaluation_passed=True,
            )
            updates.promote(component.name, human_approved=True)

            result = harness.run_harness_worker(
                model="deepseek-v4-flash",
                prompt=b"sensitive instructions",
                workspace=workspace,
                registry_root=updates.root,
                state_root=root / "state",
                timeout_seconds=60,
                enforce_catalog=False,
            )

            self.assertEqual("worker-result", result["result"])
            self.assertEqual("deepseek-v4-flash", result["reported_model"])
            self.assertTrue(result["worker_session_deleted"])
            self.assertFalse(result["canonical_workspace_modified"])
            self.assertEqual("canonical\n", (workspace / "marker.txt").read_text())
            self.assertNotIn("sensitive instructions", json.dumps(result))
            self.assertEqual([], list((root / "state").rglob("task.txt")))


class HarnessBridgeTests(unittest.TestCase):
    def test_bridge_restricts_auth_model_and_forwards_without_retention(self) -> None:
        observed: dict[str, object] = {}

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: object) -> None:
                pass

            def do_POST(self) -> None:
                length = int(self.headers["Content-Length"])
                observed["authorization"] = self.headers["Authorization"]
                observed["payload"] = json.loads(self.rfile.read(length))
                response = b'{"id":"ok"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

        try:
            upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        except PermissionError:
            self.skipTest("sandbox forbids loopback sockets")
        thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        thread.start()
        try:
            with HarnessBridge(
                upstream_credential="real-upstream-secret",
                port=0,
                upstream_url=f"http://127.0.0.1:{upstream.server_port}/v1/chat/completions",
            ) as bridge:
                body = json.dumps(
                    {"model": "deepseek-v4-flash", "messages": [{"content": "private"}]}
                ).encode()
                request = urllib.request.Request(
                    bridge.url + "/chat/completions",
                    data=body,
                    headers={
                        "Authorization": f"Bearer {bridge.credential}",
                        "Content-Type": "application/json",
                    },
                )
                with urllib.request.urlopen(request) as response:
                    self.assertEqual(b'{"id":"ok"}', response.read())
                self.assertEqual("Bearer real-upstream-secret", observed["authorization"])
                self.assertEqual("private", observed["payload"]["messages"][0]["content"])
                with self.assertRaises(urllib.error.HTTPError) as denied:
                    urllib.request.urlopen(
                        urllib.request.Request(
                            bridge.url + "/chat/completions",
                            data=json.dumps({"model": "not-allowed"}).encode(),
                            headers={"Authorization": f"Bearer {bridge.credential}"},
                        )
                    )
                self.assertEqual(400, denied.exception.code)
                denied.exception.close()
        finally:
            upstream.shutdown()
            upstream.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
