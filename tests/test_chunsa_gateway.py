from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import subprocess
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Self

from ai_orchestrator.abacus_canary import run_abacus_workflow_canaries
from ai_orchestrator.admission import AdmissionController
from ai_orchestrator.artifacts import ArtifactStore
from ai_orchestrator.claude_gateway import ClaudeGatewayHook
from ai_orchestrator.config import load_profile
from ai_orchestrator.delegation import OrchestratorTaskRunner
from ai_orchestrator.mcp_tools import FallbackControlPlane, MCPQueryServer, serve_stdio
from ai_orchestrator.models import (
    ContractError,
    Profile,
    RunResult,
    TaskSpec,
    UsageRecord,
)
from ai_orchestrator.readiness import evaluate_doctor
from ai_orchestrator.telemetry import TelemetryStore
from ai_orchestrator.workflows import (
    DefaultStageExecutor,
    StageExecution,
    WorkflowRunner,
    WorkflowStage,
    WorkflowStore,
    workflow_route,
)
from ai_orchestrator.writer import GateOutcome

ROOT = Path(__file__).resolve().parents[1]


def _git(repo: Path, *arguments: str) -> str:
    return (
        subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            capture_output=True,
        )
        .stdout.decode()
        .strip()
    )


class ClaudeGatewayHookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        (self.repo / ".mcp.json").write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "ai-orchestrator": {
                            "command": "ai-orchestrator",
                            "args": ["mcp-serve", "--profile", "chunsa"],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        self.hook = ClaudeGatewayHook(
            profile="chunsa",
            repo=self.repo,
            state_root=self.root / "state",
            gate_commands=("git diff --check",),
            readiness=lambda: {"ok": True},
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _attest(self) -> None:
        result = self.hook.session_start(
            {
                "hook_event_name": "SessionStart",
                "session_id": "session-1",
                "source": "startup",
                "cwd": str(self.repo),
                "model": "claude-sonnet-4-6",
            },
            environment={"CLAUDE_CODE_EFFORT_LEVEL": "high"},
        )
        self.assertEqual(0, result.exit_code)

    def test_session_attests_only_sonnet_high_in_the_profile_repository(self) -> None:
        self._attest()
        allowed = self.hook.pre_tool_use(
            {
                "hook_event_name": "PreToolUse",
                "session_id": "session-1",
                "cwd": str(self.repo),
                "tool_name": "Read",
                "tool_input": {"file_path": str(self.repo / "README.md")},
            }
        )
        decision = allowed["hookSpecificOutput"]
        self.assertEqual("allow", decision["permissionDecision"])

        refused = self.hook.session_start(
            {
                "hook_event_name": "SessionStart",
                "session_id": "opus-resume",
                "source": "resume",
                "cwd": str(self.repo),
                "model": "claude-opus-4-6",
            },
            environment={"CLAUDE_CODE_EFFORT_LEVEL": "high"},
        )
        self.assertEqual(2, refused.exit_code)
        self.assertFalse((self.root / "state" / "opus-resume.json").exists())

    def test_pre_tool_hook_blocks_mutation_models_agents_and_other_mcp(self) -> None:
        self._attest()
        blocked = (
            ("Edit", {"file_path": str(self.repo / "value.cpp")}),
            ("Agent", {"prompt": "delegate"}),
            ("Bash", {"command": "git commit -am unsafe"}),
            ("Bash", {"command": "git status > changed.txt"}),
            ("Bash", {"command": "qwen --model anything"}),
            ("mcp__filesystem__write_file", {"path": "value.cpp"}),
            ("Read", {"file_path": "/etc/passwd"}),
        )
        for tool_name, tool_input in blocked:
            with self.subTest(tool_name=tool_name, tool_input=tool_input):
                result = self.hook.pre_tool_use(
                    {
                        "hook_event_name": "PreToolUse",
                        "session_id": "session-1",
                        "cwd": str(self.repo),
                        "tool_name": tool_name,
                        "tool_input": tool_input,
                    }
                )
                self.assertEqual(
                    "deny",
                    result["hookSpecificOutput"]["permissionDecision"],
                )

        for tool_name, tool_input in (
            ("Bash", {"command": "git diff --check"}),
            ("Bash", {"command": "git status --short"}),
            ("Bash", {"command": "tools/chunsa_ai.sh run task.json"}),
            ("Bash", {"command": "tools/chunsa_ai.sh doctor-full"}),
            (
                "mcp__ai-orchestrator__orchestrator_submit_task",
                {"prompt": "bounded", "task": {"task_type": "tests"}},
            ),
        ):
            with self.subTest(tool_name=tool_name, tool_input=tool_input):
                result = self.hook.pre_tool_use(
                    {
                        "hook_event_name": "PreToolUse",
                        "session_id": "session-1",
                        "cwd": str(self.repo),
                        "tool_name": tool_name,
                        "tool_input": tool_input,
                    }
                )
                self.assertEqual(
                    "allow",
                    result["hookSpecificOutput"]["permissionDecision"],
                )

    def test_missing_attestation_fails_closed(self) -> None:
        result = self.hook.pre_tool_use(
            {
                "hook_event_name": "PreToolUse",
                "session_id": "missing",
                "cwd": str(self.repo),
                "tool_name": "Read",
                "tool_input": {},
            }
        )
        self.assertEqual("deny", result["hookSpecificOutput"]["permissionDecision"])

    def test_session_start_resolves_missing_model_only_from_a_verified_launch(
        self,
    ) -> None:
        (self.repo / ".claude").mkdir()
        (self.repo / ".claude" / "settings.json").write_text(
            json.dumps({"model": "sonnet"}), encoding="utf-8"
        )

        fallback = ClaudeGatewayHook(
            profile="chunsa",
            repo=self.repo,
            state_root=self.root / "fallback-state",
            gate_commands=("git diff --check",),
            readiness=lambda: {"ok": True},
            launch_overrides=lambda: (True, None, None),
        ).session_start(
            {
                "session_id": "missing-model",
                "source": "startup",
                "cwd": str(self.repo),
            },
            environment={"CLAUDE_CODE_EFFORT_LEVEL": "high"},
        )
        self.assertEqual(0, fallback.exit_code)
        allowed_after_fallback = ClaudeGatewayHook(
            profile="chunsa",
            repo=self.repo,
            state_root=self.root / "fallback-state",
            gate_commands=("git diff --check",),
            readiness=lambda: {"ok": True},
            launch_overrides=lambda: (True, None, None),
        ).pre_tool_use(
            {
                "session_id": "missing-model",
                "cwd": str(self.repo),
                "tool_name": "Read",
                "tool_input": {"file_path": "README.md"},
            }
        )
        self.assertEqual(
            "allow",
            allowed_after_fallback["hookSpecificOutput"]["permissionDecision"],
        )

        override = ClaudeGatewayHook(
            profile="chunsa",
            repo=self.repo,
            state_root=self.root / "override-state",
            gate_commands=("git diff --check",),
            readiness=lambda: {"ok": True},
            launch_overrides=lambda: (True, "opus", "low"),
        ).session_start(
            {
                "session_id": "overridden-model",
                "source": "startup",
                "cwd": str(self.repo),
            },
            environment={"CLAUDE_CODE_EFFORT_LEVEL": "high"},
        )
        self.assertEqual(2, override.exit_code)
        self.assertIn("Sonnet", override.stderr)
        self.assertIn("effort", override.stderr.lower())

        unverifiable = ClaudeGatewayHook(
            profile="chunsa",
            repo=self.repo,
            state_root=self.root / "unverified-state",
            gate_commands=("git diff --check",),
            readiness=lambda: {"ok": True},
            launch_overrides=lambda: (False, None, None),
        ).session_start(
            {
                "session_id": "unverified-launch",
                "source": "startup",
                "cwd": str(self.repo),
            },
            environment={"CLAUDE_CODE_EFFORT_LEVEL": "high"},
        )
        self.assertEqual(2, unverifiable.exit_code)
        self.assertIn("launch", unverifiable.stderr)


class StrictMCPTests(unittest.TestCase):
    def test_submit_is_inline_and_profile_bound(self) -> None:
        class Runner:
            def submit(self, **arguments: object) -> dict[str, object]:
                self.arguments = arguments
                return {"status": "SUCCESS", "workflow_id": "workflow-1"}

        runner = Runner()
        server = MCPQueryServer(object(), task_runner=runner, required_profile="chunsa")
        listed = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        names = {tool["name"] for tool in listed["result"]["tools"]}
        self.assertIn("orchestrator_submit_task", names)
        self.assertNotIn("orchestrator_run_task", names)
        self.assertNotIn("orchestrator_delegate_abacus", names)

        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "orchestrator_submit_task",
                    "arguments": {
                        "prompt": "do not persist me",
                        "task": {
                            "task_type": "tests",
                            "write_mode": "validated_patch",
                            "allowed_write_paths": ["tests/value.cpp"],
                        },
                    },
                },
            }
        )
        payload = json.loads(response["result"]["content"][0]["text"])
        self.assertFalse(response["result"]["isError"])
        self.assertEqual("workflow-1", payload["workflow_id"])
        self.assertEqual("chunsa", runner.arguments["profile"])
        self.assertNotIn("profile", runner.arguments["metadata"])

    def test_profile_bound_server_rejects_a_different_launch_repository(self) -> None:
        with self.assertRaisesRegex(ContractError, "repository"):
            serve_stdio(
                object(),
                profile="chunsa",
                launched_repo=Path("/tmp"),
                configuration_root=ROOT / "config",
                input_stream=io.BytesIO(),
                output_stream=io.BytesIO(),
            )

    def test_broker_failure_falls_back_to_direct_control_plane(self) -> None:
        class Broker:
            def diagnostics(self, _profile: str) -> dict[str, object]:
                raise OSError("socket unavailable")

        class Direct:
            def diagnostics(self, profile: str) -> dict[str, object]:
                return {"ok": True, "profile": profile, "mode": "direct"}

        control = FallbackControlPlane(Broker(), Direct())
        self.assertEqual(
            {"ok": True, "profile": "chunsa", "mode": "direct"},
            control.diagnostics("chunsa"),
        )


class InlineTaskRunnerTests(unittest.TestCase):
    def test_inline_execution_hashes_prompt_without_creating_a_prompt_file(
        self,
    ) -> None:
        captured: dict[str, object] = {}

        class Telemetry:
            def routing_observations(self) -> dict[str, object]:
                return {}

            def record_route(self, task: TaskSpec, _decision: object) -> None:
                captured["context_sha256"] = task.context_sha256()

            def record_run(self, _result: RunResult) -> None:
                return None

        def executor(task: TaskSpec, decision: object, **kwargs: object):
            captured["task"] = task
            captured["prompt_bytes"] = kwargs.get("prompt_bytes")
            return (
                RunResult(
                    schema_version=1,
                    run_id=str(__import__("uuid").uuid4()),
                    decision_id=decision.decision_id,
                    task_id=task.task_id,
                    status="SUCCESS",
                    requested_model=decision.selected_model,
                    reported_model=decision.selected_model,
                    provider=decision.provider,
                    transport=decision.transport,
                    started_at="2026-08-21T00:00:00+00:00",
                    finished_at="2026-08-21T00:00:01+00:00",
                    latency_ms=1,
                    output_sha256="0" * 64,
                    output_bytes=6,
                    exit_code=0,
                    provenance_verified=True,
                ),
                b"answer",
            )

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repo"
            repo.mkdir()
            task = TaskSpec.from_mapping(
                {
                    "repo": str(repo),
                    "task_type": "review",
                    "sensitivity": "internal",
                    "write_mode": "read_only",
                }
            )
            receipt = OrchestratorTaskRunner(
                configuration_root=ROOT / "config",
                executor=executor,
                telemetry=Telemetry(),
                admission=AdmissionController(Path(directory) / "admission.sqlite"),
            ).run_inline(
                task=task,
                prompt=b"inline-only",
                profile="default",
                model="gpt-5.6-terra",
            )

        self.assertEqual("SUCCESS", receipt["status"])
        self.assertEqual(b"inline-only", captured["prompt_bytes"])
        self.assertIsNone(captured["task"].prompt_path())
        self.assertEqual(
            hashlib.sha256(b"inline-only").hexdigest(),
            captured["context_sha256"],
        )


class WorkflowPolicyTests(unittest.TestCase):
    def test_routes_use_distinct_providers_and_never_offer_fallback(self) -> None:
        expected = {
            "architecture": (
                ("execute", "claude-opus"),
                ("review", "abacus-grok-4.6"),
            ),
            "implementation": (
                ("plan", "abacus-qwen3.8-max"),
                ("execute", "claude-sonnet"),
                ("review", "abacus-grok-4.6"),
            ),
            "bounded_cpp": (
                ("execute", "deepseek-v4-flash"),
                ("review", "gpt-5.6-sol"),
            ),
            "determinism": (
                ("execute", "gpt-5.6-sol"),
                ("review", "claude-opus"),
            ),
            "godot_ui": (
                ("plan", "abacus-qwen3.8-max"),
                ("execute", "kimi-official"),
                ("review", "claude-sonnet"),
            ),
            "security": (
                ("execute", "abacus-grok-4.6"),
                ("review", "gpt-5.6-terra"),
            ),
        }
        for task_type, wanted in expected.items():
            with self.subTest(task_type=task_type):
                route = workflow_route(task_type)
                self.assertEqual(wanted, tuple((row.name, row.model) for row in route))
                self.assertGreaterEqual(len({row.provider for row in route}), 2)
                self.assertTrue(
                    all(
                        left.provider != right.provider
                        for left, right in pairwise(route)
                    )
                )
                self.assertTrue(all(not row.fallback_allowed for row in route))

    def test_grok_and_qwen_attempt_limits_use_a_rolling_24_hour_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = WorkflowStore(Path(directory) / "workflow.sqlite")
            now = datetime(2026, 8, 21, 12, tzinfo=UTC)
            for _ in range(6):
                store.reserve_attempt("abacus-grok-4.6", now=now)
            with self.assertRaisesRegex(ContractError, "24-hour attempt limit"):
                store.reserve_attempt("abacus-grok-4.6", now=now)

            store.record_attempt(
                "abacus-qwen3.8-max",
                outcome="FAILED",
                created_at=now - timedelta(hours=25),
            )
            for _ in range(12):
                store.reserve_attempt("abacus-qwen3.8-max", now=now)
            with self.assertRaisesRegex(ContractError, "24-hour attempt limit"):
                store.reserve_attempt("abacus-qwen3.8-max", now=now)

    def test_qwen_writer_promotion_requires_five_consecutive_canaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = WorkflowStore(Path(directory) / "workflow.sqlite")
            for passed in (True, True, False, True, True, True, True):
                store.record_canary("abacus-qwen3.8-max", passed=passed)
            self.assertFalse(store.qwen_patch_enabled())
            store.record_canary("abacus-qwen3.8-max", passed=True)
            self.assertTrue(store.qwen_patch_enabled())

    def test_abacus_canaries_count_failures_and_never_promote_partial_runs(
        self,
    ) -> None:
        class Delegator:
            def __init__(self, outputs: list[str]) -> None:
                self.outputs = outputs

            def delegate(self, **_kwargs: object) -> dict[str, object]:
                return {"result": self.outputs.pop(0)}

        class Lease:
            def __enter__(self) -> Self:
                return self

            def __exit__(self, *_args: object) -> None:
                return None

        class Admission:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def acquire(self, **kwargs: object) -> Lease:
                self.calls.append(kwargs)
                return Lease()

        with tempfile.TemporaryDirectory() as directory:
            failed_store = WorkflowStore(Path(directory) / "failed.sqlite")
            failed = run_abacus_workflow_canaries(
                "abacus-qwen3.8-max",
                attempts=5,
                delegator=Delegator(["AI_ORCHESTRATOR_CANARY_OK"] * 4 + ["wrong"]),
                store=failed_store,
            )
            self.assertFalse(failed["promotion_ready"])
            self.assertEqual("candidate", failed["status"])
            self.assertFalse(failed_store.qwen_patch_enabled())
            self.assertFalse(failed_store.workflow_model_enabled("abacus-qwen3.8-max"))

            passed_store = WorkflowStore(Path(directory) / "passed.sqlite")
            admission = Admission()
            passed = run_abacus_workflow_canaries(
                "abacus-qwen3.8-max",
                attempts=5,
                delegator=Delegator(["AI_ORCHESTRATOR_CANARY_OK"] * 5),
                store=passed_store,
                admission=admission,
            )
            self.assertTrue(passed["promotion_ready"])
            self.assertTrue(passed_store.qwen_patch_enabled())
            self.assertTrue(passed_store.workflow_model_enabled("abacus-qwen3.8-max"))
            self.assertEqual(5, len(admission.calls))
            self.assertTrue(
                all(call["max_concurrency"] == 1 for call in admission.calls)
            )


class WorkflowRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        _git(self.repo, "init")
        _git(self.repo, "config", "user.email", "test@example.invalid")
        _git(self.repo, "config", "user.name", "Test")
        (self.repo / "value.txt").write_text("before\n", encoding="utf-8")
        _git(self.repo, "add", "value.txt")
        _git(self.repo, "commit", "-m", "initial")
        self.patch = (
            b"diff --git a/value.txt b/value.txt\n"
            b"--- a/value.txt\n"
            b"+++ b/value.txt\n"
            b"@@ -1 +1 @@\n"
            b"-before\n"
            b"+after\n"
        )
        self.profile = Profile(
            name="chunsa",
            repo_root=str(self.repo),
            auto_execute_sensitivities=("public", "internal"),
            allowed_models=(
                "abacus-qwen3.8-max",
                "claude-sonnet",
                "abacus-grok-4.6",
            ),
            deny_providers=(),
            protected_path_prefixes=(".git",),
            deterministic_gates=("git diff --check",),
            minimax_mode="direct",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_inline_prompt_is_never_persisted_and_approved_patch_is_staged(
        self,
    ) -> None:
        secret = "INLINE-SECRET-MUST-NOT-BE-STORED"
        seen: list[tuple[str, str, bytes]] = []

        def execute_stage(
            stage: WorkflowStage, _task: object, prompt: bytes
        ) -> StageExecution:
            seen.append((stage.name, stage.model, prompt))
            if stage.name == "plan":
                return StageExecution(status="SUCCESS", output=b"bounded plan")
            if stage.name == "execute":
                return StageExecution(
                    status="PATCH_READY", output=self.patch, run_id="run-execute"
                )
            return StageExecution(
                status="SUCCESS",
                output=b'{"verdict":"approve"}',
                verdict="approve",
                run_id="run-review",
            )

        telemetry = TelemetryStore(self.root / "state" / "telemetry.sqlite")
        runner = WorkflowRunner(
            profile_loader=lambda _name: self.profile,
            stage_executor=execute_stage,
            workflow_store=WorkflowStore(self.root / "state" / "workflow.sqlite"),
            telemetry=telemetry,
            artifacts=ArtifactStore(self.root / "state" / "artifacts"),
        )
        receipt = runner.submit(
            prompt=secret,
            metadata={
                "task_type": "implementation",
                "write_mode": "validated_patch",
                "allowed_write_paths": ["value.txt"],
            },
            profile="chunsa",
        )

        self.assertEqual("PATCH_READY", receipt["status"])
        self.assertIsNotNone(receipt["artifact_id"])
        self.assertEqual(
            ["abacus", "anthropic", "abacus"],
            [row["provider"] for row in receipt["stages"]],
        )
        self.assertEqual("before\n", (self.repo / "value.txt").read_text())
        self.assertIn(b"bounded plan", seen[1][2])
        self.assertIn(self.patch, seen[2][2])
        for source in (self.root / "state").rglob("*"):
            if source.is_file():
                self.assertNotIn(secret.encode(), source.read_bytes())
        report = telemetry.report()
        self.assertEqual(1, report["workflow_count"])

    def test_reviewer_rejection_is_terminal_and_creates_no_artifact(self) -> None:
        called: list[str] = []

        def execute_stage(
            stage: WorkflowStage, _task: object, _prompt: bytes
        ) -> StageExecution:
            called.append(stage.name)
            if stage.name == "plan":
                return StageExecution(status="SUCCESS", output=b"plan")
            if stage.name == "execute":
                return StageExecution(status="PATCH_READY", output=self.patch)
            return StageExecution(status="SUCCESS", verdict="reject")

        artifact_root = self.root / "state" / "artifacts"
        runner = WorkflowRunner(
            profile_loader=lambda _name: self.profile,
            stage_executor=execute_stage,
            workflow_store=WorkflowStore(self.root / "state" / "workflow.sqlite"),
            telemetry=TelemetryStore(self.root / "state" / "telemetry.sqlite"),
            artifacts=ArtifactStore(artifact_root),
        )
        receipt = runner.submit(
            prompt="change value",
            metadata={
                "task_type": "implementation",
                "write_mode": "validated_patch",
                "allowed_write_paths": ["value.txt"],
            },
            profile="chunsa",
        )
        self.assertEqual("REJECTED", receipt["status"])
        self.assertIsNone(receipt["artifact_id"])
        self.assertEqual(["plan", "execute", "review"], called)
        self.assertEqual([], [path for path in artifact_root.iterdir()])

    def test_provider_exception_is_terminal_in_workflow_telemetry(self) -> None:
        telemetry_path = self.root / "state" / "telemetry.sqlite"

        def fail_stage(*_args: object) -> StageExecution:
            raise RuntimeError("provider failed")

        runner = WorkflowRunner(
            profile_loader=lambda _name: self.profile,
            stage_executor=fail_stage,
            workflow_store=WorkflowStore(self.root / "state" / "workflow.sqlite"),
            telemetry=TelemetryStore(telemetry_path),
            artifacts=ArtifactStore(self.root / "state" / "artifacts"),
        )
        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            runner.submit(
                prompt="inspect",
                metadata={"task_type": "implementation", "write_mode": "read_only"},
                profile="chunsa",
            )
        with sqlite3.connect(telemetry_path) as connection:
            workflow = connection.execute(
                "SELECT status, final_verdict FROM workflows"
            ).fetchone()
            stage = connection.execute(
                "SELECT stage, status, verdict FROM workflow_stages"
            ).fetchone()
        self.assertEqual(("FAILED", "reject"), workflow)
        self.assertEqual(("plan", "FAILED", "reject"), stage)

    def test_grok_execution_is_rejected_for_write_workflows(self) -> None:
        runner = WorkflowRunner(
            profile_loader=lambda _name: self.profile,
            stage_executor=lambda *_args: self.fail("Grok must not execute a write"),
            workflow_store=WorkflowStore(self.root / "state" / "workflow.sqlite"),
            telemetry=TelemetryStore(self.root / "state" / "telemetry.sqlite"),
            artifacts=ArtifactStore(self.root / "state" / "artifacts"),
        )
        with self.assertRaisesRegex(ContractError, "Grok.*read-only"):
            runner.submit(
                prompt="audit and patch",
                metadata={
                    "task_type": "security",
                    "write_mode": "validated_patch",
                    "allowed_write_paths": ["value.txt"],
                },
                profile="chunsa",
            )

    def test_failed_canary_batch_keeps_model_out_of_automatic_workflows(self) -> None:
        workflow_store = WorkflowStore(self.root / "state" / "workflow.sqlite")
        for passed in (True, True, False, True, True):
            workflow_store.record_canary("abacus-qwen3.8-max", passed=passed)
        runner = WorkflowRunner(
            profile_loader=lambda _name: self.profile,
            stage_executor=lambda *_args: self.fail("candidate model must not run"),
            workflow_store=workflow_store,
            telemetry=TelemetryStore(self.root / "state" / "telemetry.sqlite"),
            artifacts=ArtifactStore(self.root / "state" / "artifacts"),
        )
        with self.assertRaisesRegex(ContractError, "candidate"):
            runner.submit(
                prompt="inspect",
                metadata={"task_type": "implementation", "write_mode": "read_only"},
                profile="chunsa",
            )

    def test_qwen_patch_opt_in_requires_promotion_and_uses_independent_reviewer(
        self,
    ) -> None:
        workflow_store = WorkflowStore(self.root / "state" / "workflow.sqlite")
        telemetry = TelemetryStore(self.root / "state" / "telemetry.sqlite")
        artifact_store = ArtifactStore(
            self.root / "state" / "artifacts",
            gate_runner=lambda _worktree, commands, _timeout: tuple(
                GateOutcome(command, 0, "", False) for command in commands
            ),
        )
        seen: list[tuple[str, str]] = []

        def execute_stage(
            stage: WorkflowStage, _task: object, _prompt: bytes
        ) -> StageExecution:
            seen.append((stage.name, stage.model))
            if stage.name == "execute":
                return StageExecution(status="PATCH_READY", output=self.patch)
            return StageExecution(status="SUCCESS", verdict="approve")

        runner = WorkflowRunner(
            profile_loader=lambda _name: self.profile,
            stage_executor=execute_stage,
            workflow_store=workflow_store,
            telemetry=telemetry,
            artifacts=artifact_store,
        )
        metadata = {
            "task_type": "implementation",
            "write_mode": "validated_patch",
            "allowed_write_paths": ["value.txt"],
            "qwen_patch": True,
        }
        with self.assertRaisesRegex(ContractError, "five consecutive"):
            runner.submit(prompt="change", metadata=metadata, profile="chunsa")
        for _ in range(5):
            workflow_store.record_canary("abacus-qwen3.8-max", passed=True)
        receipt = runner.submit(prompt="change", metadata=metadata, profile="chunsa")
        self.assertEqual("PATCH_READY", receipt["status"])
        self.assertEqual(
            [
                ("execute", "abacus-qwen3.8-max"),
                ("review", "gpt-5.6-terra"),
            ],
            seen,
        )

    def test_default_stage_executor_uses_abacus_for_vendor_stages_and_core_runner_for_others(
        self,
    ) -> None:
        calls: list[tuple[str, object]] = []

        class Abacus:
            def delegate(self, **kwargs: object) -> dict[str, object]:
                calls.append(("abacus", kwargs))
                return {
                    "schema_version": 1,
                    "provider": "abacus",
                    "model": kwargs["model"],
                    "task_type": kwargs["task_type"],
                    "result": '{"verdict":"approve"}',
                    "authentication": "human_cli_login",
                    "quota_policy": "pro_included_credits_only",
                    "automatic_purchase": False,
                    "incremental_spend_allowed": False,
                    "allowed_tools": [],
                    "filesystem_access": "disposable_empty_directory",
                    "fallback_allowed": False,
                }

        class Core:
            def run_inline(self, **kwargs: object) -> dict[str, object]:
                calls.append(("core", kwargs))
                return {
                    "schema_version": 1,
                    "status": "PATCH_READY",
                    "run": {"run_id": "core-run"},
                    "output": self_patch.decode(),
                    "source_repo_modified": False,
                    "fallback_allowed": False,
                }

        self_patch = self.patch
        executor = DefaultStageExecutor(
            profile="chunsa",
            task_runner=Core(),
            abacus=Abacus(),
            admission=AdmissionController(self.root / "state" / "admission.sqlite"),
        )
        plan = executor(
            WorkflowStage("plan", "abacus-qwen3.8-max", "abacus"),
            TaskSpec.from_mapping(
                {
                    "repo": str(self.repo),
                    "task_type": "implementation",
                    "write_mode": "read_only",
                }
            ),
            b"plan me",
        )
        execute = executor(
            WorkflowStage("execute", "claude-sonnet", "anthropic"),
            TaskSpec.from_mapping(
                {
                    "repo": str(self.repo),
                    "task_type": "implementation",
                    "write_mode": "validated_patch",
                    "allowed_write_paths": ["value.txt"],
                }
            ),
            b"execute me",
        )
        review = executor(
            WorkflowStage("review", "abacus-grok-4.6", "abacus"),
            TaskSpec.from_mapping(
                {
                    "repo": str(self.repo),
                    "task_type": "implementation",
                    "write_mode": "read_only",
                }
            ),
            b"review me",
        )
        self.assertEqual("SUCCESS", plan.status)
        self.assertEqual("PATCH_READY", execute.status)
        self.assertEqual(self.patch, execute.output)
        self.assertEqual("approve", review.verdict)
        self.assertEqual(["abacus", "core", "abacus"], [row[0] for row in calls])
        self.assertEqual("Qwen3.8 Max", calls[0][1]["model"])
        self.assertEqual("Grok 4.6", calls[2][1]["model"])


class TelemetryMigrationTests(unittest.TestCase):
    def test_old_usage_rows_survive_nullable_counter_migration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.sqlite"
            with sqlite3.connect(path) as connection:
                connection.execute(
                    """
                    CREATE TABLE usage_records (
                        record_id TEXT PRIMARY KEY,
                        route_plan_id TEXT,
                        created_at TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        model TEXT NOT NULL,
                        input_tokens INTEGER NOT NULL,
                        output_tokens INTEGER NOT NULL,
                        cached_tokens INTEGER NOT NULL,
                        credits_used REAL NOT NULL,
                        estimated_cost_usd REAL NOT NULL,
                        cache_hit INTEGER NOT NULL,
                        latency_ms INTEGER NOT NULL,
                        outcome TEXT NOT NULL,
                        provenance_verified INTEGER NOT NULL
                    )
                    """
                )
                connection.execute(
                    "INSERT INTO usage_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        "old",
                        None,
                        "2026-08-20T00:00:00+00:00",
                        "abacus",
                        "Grok 4.6",
                        0,
                        0,
                        0,
                        0.0,
                        0.0,
                        0,
                        1,
                        "SUCCESS",
                        1,
                    ),
                )

            store = TelemetryStore(path)
            store.record_usage(
                UsageRecord(
                    provider="abacus",
                    model="Qwen3.8 Max",
                    outcome="SUCCESS",
                    latency_ms=2,
                )
            )
            with sqlite3.connect(path) as connection:
                old = connection.execute(
                    "SELECT model FROM usage_records WHERE record_id='old'"
                ).fetchone()
                newest = connection.execute(
                    """
                    SELECT input_tokens, output_tokens, credits_used
                    FROM usage_records WHERE model='Qwen3.8 Max'
                    """
                ).fetchone()
                nullable = {
                    row[1]: row[3] == 0
                    for row in connection.execute("PRAGMA table_info(usage_records)")
                }
            self.assertEqual(("Grok 4.6",), old)
            self.assertEqual((None, None, None), newest)
            self.assertTrue(nullable["input_tokens"])
            self.assertTrue(nullable["credits_used"])


class DoctorTests(unittest.TestCase):
    def test_readiness_does_not_turn_historical_storage_findings_into_downtime(
        self,
    ) -> None:
        inputs = {
            "configuration": {"ok": True},
            "commands": {"git": "/usr/bin/git", "claude": "/usr/bin/claude"},
            "mcp": {"configured": True},
            "broker": {"reachable": True, "direct_fallback_available": True},
            "authentication": [{"provider": "anthropic", "status": "available"}],
            "omniroute": {"configured": True, "reachable": True, "status": 200},
            "config_sync": {"in_sync": True},
            "storage": {"ok": False, "historical_findings": 4},
        }
        ready = evaluate_doctor(check="readiness", **inputs)
        full = evaluate_doctor(check="full", **inputs)
        self.assertTrue(ready["ok"])
        self.assertFalse(full["ok"])
        self.assertFalse(ready["compliance"]["storage"]["ok"])

    def test_readiness_requires_mcp_broker_authentication_and_omniroute(self) -> None:
        base = {
            "configuration": {"ok": True},
            "commands": {"git": "/usr/bin/git"},
            "mcp": {"configured": True},
            "broker": {"reachable": True, "direct_fallback_available": True},
            "authentication": [{"provider": "anthropic", "status": "available"}],
            "omniroute": {"configured": True, "reachable": True},
            "config_sync": {"in_sync": True},
            "storage": {"ok": True},
        }
        for section, broken in (
            ("mcp", {"configured": False}),
            ("broker", {"reachable": False, "direct_fallback_available": True}),
            ("authentication", [{"provider": "anthropic", "status": "unavailable"}]),
            ("omniroute", {"configured": True, "reachable": False}),
        ):
            with self.subTest(section=section):
                values = dict(base)
                values[section] = broken
                self.assertFalse(evaluate_doctor(check="readiness", **values)["ok"])


class ArtifactAcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        _git(self.repo, "init")
        _git(self.repo, "config", "user.email", "test@example.invalid")
        _git(self.repo, "config", "user.name", "Test")
        (self.repo / "value.txt").write_text("before\n", encoding="utf-8")
        _git(self.repo, "add", "value.txt")
        _git(self.repo, "commit", "-m", "initial")
        self.head = _git(self.repo, "rev-parse", "HEAD")
        self.patch = (
            b"diff --git a/value.txt b/value.txt\n"
            b"index 90be1b2..e44b4bc 100644\n"
            b"--- a/value.txt\n"
            b"+++ b/value.txt\n"
            b"@@ -1 +1 @@\n"
            b"-before\n"
            b"+after\n"
        )
        self.store = ArtifactStore(
            self.root / "artifacts",
            gate_runner=lambda _worktree, commands, _timeout: tuple(
                GateOutcome(command, 0, "", False) for command in commands
            ),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _stage(self, *, verdict: str = "approve") -> dict[str, object]:
        return self.store.stage_patch(
            workflow_id="workflow-1",
            task_id="task-1",
            repo=self.repo,
            base_head=self.head,
            patch=self.patch,
            allowed_write_paths=("value.txt",),
            protected_prefixes=(".git",),
            gate_commands=("git diff --check",),
            gate_timeout_seconds=30,
            reviewer_provider="abacus",
            reviewer_model="abacus-grok-4.6",
            reviewer_verdict=verdict,
        )

    def test_accept_rechecks_hash_head_cleanliness_and_gates_then_applies(self) -> None:
        manifest = self._stage()
        receipt = self.store.accept(
            str(manifest["artifact_id"]),
            expected_sha256=hashlib.sha256(self.patch).hexdigest(),
            human_approved=True,
        )
        self.assertEqual("accepted", receipt["status"])
        self.assertEqual("after\n", (self.repo / "value.txt").read_text())
        self.assertTrue(receipt["gates_passed"])

    def test_tamper_changed_head_dirty_tree_and_reviewer_rejection_block_apply(
        self,
    ) -> None:
        manifest = self._stage()
        artifact = self.root / "artifacts" / str(manifest["artifact_id"]) / "patch.diff"
        artifact.write_bytes(self.patch + b"# tampered\n")
        with self.assertRaisesRegex(ContractError, "hash"):
            self.store.accept(
                str(manifest["artifact_id"]),
                expected_sha256=hashlib.sha256(self.patch).hexdigest(),
                human_approved=True,
            )

        self.store = ArtifactStore(
            self.root / "artifacts-head",
            gate_runner=lambda _worktree, commands, _timeout: tuple(
                GateOutcome(command, 0, "", False) for command in commands
            ),
        )
        manifest = self._stage()
        (self.repo / "other.txt").write_text("commit\n", encoding="utf-8")
        _git(self.repo, "add", "other.txt")
        _git(self.repo, "commit", "-m", "head changed")
        with self.assertRaisesRegex(ContractError, "HEAD"):
            self.store.accept(
                str(manifest["artifact_id"]),
                expected_sha256=hashlib.sha256(self.patch).hexdigest(),
                human_approved=True,
            )

        _git(self.repo, "reset", "--hard", self.head)
        self.store = ArtifactStore(
            self.root / "artifacts-dirty",
            gate_runner=lambda _worktree, commands, _timeout: tuple(
                GateOutcome(command, 0, "", False) for command in commands
            ),
        )
        manifest = self._stage()
        (self.repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "clean"):
            self.store.accept(
                str(manifest["artifact_id"]),
                expected_sha256=hashlib.sha256(self.patch).hexdigest(),
                human_approved=True,
            )

        (self.repo / "dirty.txt").unlink()
        self.store = ArtifactStore(
            self.root / "artifacts-rejected",
            gate_runner=lambda _worktree, commands, _timeout: tuple(
                GateOutcome(command, 0, "", False) for command in commands
            ),
        )
        manifest = self._stage(verdict="reject")
        with self.assertRaisesRegex(ContractError, "reviewer"):
            self.store.accept(
                str(manifest["artifact_id"]),
                expected_sha256=hashlib.sha256(self.patch).hexdigest(),
                human_approved=True,
            )

    def test_reject_is_human_only_and_terminal(self) -> None:
        manifest = self._stage()
        with self.assertRaisesRegex(ContractError, "human approval"):
            self.store.reject(str(manifest["artifact_id"]), human_approved=False)
        receipt = self.store.reject(
            str(manifest["artifact_id"]),
            human_approved=True,
            reason="not wanted",
        )
        self.assertEqual("rejected", receipt["status"])
        with self.assertRaisesRegex(ContractError, "pending"):
            self.store.accept(
                str(manifest["artifact_id"]),
                expected_sha256=hashlib.sha256(self.patch).hexdigest(),
                human_approved=True,
            )

    def test_expired_artifacts_are_purged_after_the_retention_window(self) -> None:
        manifest = self._stage()
        directory = self.root / "artifacts" / str(manifest["artifact_id"])
        manifest_path = directory / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["expires_at"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")

        removed = self.store.purge_expired()

        self.assertEqual([manifest["artifact_id"]], removed)
        self.assertFalse(directory.exists())


class ProfileContractTests(unittest.TestCase):
    def test_chunsa_includes_abacus_planner_and_read_only_reviewer(self) -> None:
        profile = load_profile("chunsa", ROOT / "config")
        self.assertIn("abacus-qwen3.8-max", profile.allowed_models)
        self.assertIn("abacus-grok-4.6", profile.allowed_models)


if __name__ == "__main__":
    unittest.main()
