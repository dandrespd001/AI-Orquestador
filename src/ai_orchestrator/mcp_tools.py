"""MCP surface for safe inspection and explicit bounded delegation."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, BinaryIO, ClassVar

from ai_orchestrator.broker import broker_request, default_socket_path
from ai_orchestrator.config import load_profile
from ai_orchestrator.control_plane import ControlPlane
from ai_orchestrator.delegation import OrchestratorTaskRunner
from ai_orchestrator.models import ContractError, TaskSpec

MAX_MCP_MESSAGE_BYTES = 1_000_000
MCP_TOOLS = (
    {
        "name": "orchestrator_inventory",
        "description": "Inspect secret-free providers, auth status, models, profiles, usage, extensions, updates, sessions or capability packages.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "enum": [
                        "providers",
                        "credentials",
                        "models",
                        "profiles",
                        "usage",
                        "routes",
                        "extensions",
                        "updates",
                        "sessions",
                        "capabilities",
                    ],
                },
            },
            "required": ["section"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestrator_diagnostics",
        "description": "Run secret-free configuration and policy diagnostics.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestrator_simulate_route",
        "description": "Simulate routing without executing a model or enabling fallback.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "object"},
                "pin": {"type": "string"},
                "observations": {"type": "object"},
            },
            "required": ["task"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestrator_submit_task",
        "description": (
            "Run an explicit AI-Orchestrator task spec through the common policy, "
            "provider adapters, isolation, gates and metadata-only telemetry. "
            "Writer tasks can only return a patch pending review."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "spec_path": {"type": "string", "minLength": 1},
                "model": {"type": "string", "minLength": 1},
                "confirmed_plan_id": {
                    "type": "string",
                    "pattern": "^[0-9a-fA-F-]{36}$",
                },
            },
            "required": ["spec_path"],
            "additionalProperties": False,
        },
        "annotations": {
            "title": "Submit orchestrated task",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    },
)


class BrokerControlPlane:
    """ControlPlane-shaped client that keeps Harness on the private socket."""

    METHODS: ClassVar[dict[str, str]] = {
        "providers": "providers.list",
        "credentials": "auth.status",
        "models": "models.list",
        "profiles": "profiles.list",
        "routes": "routes.policy",
        "usage": "usage.report",
        "extensions": "extensions.list",
        "updates": "updates.list",
        "sessions": "sessions.list",
        "capabilities": "capabilities.list",
        "diagnostics": "diagnostics",
    }

    def __init__(self, path: Path) -> None:
        self.path = path

    def _call(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        response = broker_request(method, params, path=self.path)
        if response.get("ok") is not True or not isinstance(
            response.get("result"), dict
        ):
            raise ContractError(str(response.get("message", "broker request failed")))
        return response["result"]

    def query(self, section: str, **params: object) -> dict[str, Any]:
        if section not in self.METHODS:
            raise ContractError(f"unknown control-plane section: {section}")
        return self._call(self.METHODS[section], dict(params))

    def diagnostics(self, profile: str = "default") -> dict[str, Any]:
        return self._call("diagnostics", {"profile": profile})

    def simulate_route(
        self,
        task: dict[str, Any],
        *,
        profile: str = "default",
        observations: dict[str, dict[str, object]] | None = None,
        pin: str | None = None,
    ) -> dict[str, Any]:
        return self._call(
            "routes.simulate",
            {
                "task": task,
                "profile": profile,
                "observations": observations or {},
                "pin": pin,
            },
        )


class MCPQueryServer:
    def __init__(
        self,
        control: Any,
        *,
        delegator: Any | None = None,
        task_runner: Any | None = None,
        telemetry: Any | None = None,
        fixed_profile: str = "chunsa",
    ) -> None:
        self.control = control
        self.telemetry = telemetry
        bundled_config = Path(__file__).resolve().parents[2] / "config"
        self.configuration_root = bundled_config if bundled_config.is_dir() else None
        self.fixed_profile = fixed_profile
        load_profile(self.fixed_profile, self.configuration_root)
        self.task_runner = task_runner or OrchestratorTaskRunner(
            configuration_root=self.configuration_root,
            telemetry=telemetry,
        )

    @staticmethod
    def _result(request_id: object, payload: object) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": payload}

    @staticmethod
    def _error(request_id: object, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }

    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        request_id = request.get("id")
        method = request.get("method")
        if method == "notifications/initialized":
            return None
        if method == "initialize":
            return self._result(
                request_id,
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "ai-orchestrator", "version": "0.2.0"},
                    "instructions": (
                        f"This server is bound to profile {self.fixed_profile!r}. Use inventory and "
                        "diagnostics for read-only facts, simulate before submitting, and submit "
                        "only an explicit TaskSpec. Routing has no silent fallback; writer tasks "
                        "return reviewable artifacts and never modify the canonical checkout."
                    ),
                },
            )
        if method == "ping":
            return self._result(request_id, {})
        if method == "tools/list":
            return self._result(request_id, {"tools": list(MCP_TOOLS)})
        if method != "tools/call":
            return self._error(request_id, -32601, "method not found")
        params = request.get("params")
        if not isinstance(params, dict):
            return self._error(request_id, -32602, "invalid params")
        name = params.get("name")
        arguments = params.get("arguments", {})
        if not isinstance(arguments, dict):
            return self._error(request_id, -32602, "invalid tool arguments")
        try:
            if name == "orchestrator_inventory":
                section = str(arguments.get("section", ""))
                options: dict[str, object] = {}
                if section in {"credentials", "extensions"}:
                    options["profile"] = self.fixed_profile
                payload = self.control.query(section, **options)
            elif name == "orchestrator_diagnostics":
                payload = self.control.diagnostics(self.fixed_profile)
            elif name == "orchestrator_simulate_route":
                task = arguments.get("task")
                if not isinstance(task, dict):
                    raise ContractError("task must be an object")
                payload = self.control.simulate_route(
                    task,
                    profile=self.fixed_profile,
                    observations=(
                        arguments.get("observations")
                        if isinstance(arguments.get("observations"), dict)
                        else None
                    ),
                    pin=str(arguments["pin"]) if arguments.get("pin") else None,
                )
            elif name == "orchestrator_submit_task":
                spec_path = arguments.get("spec_path")
                if not isinstance(spec_path, str):
                    raise ContractError("spec_path must be a string")
                task = TaskSpec.read(Path(spec_path).expanduser().resolve(strict=True))
                profile = load_profile(self.fixed_profile, self.configuration_root)
                if profile.repo_root and Path(task.repo).resolve() != Path(
                    profile.repo_root
                ).resolve():
                    raise ContractError("task repository is outside the bound MCP profile")
                call: dict[str, Any] = {
                    "spec_path": spec_path,
                    "profile": self.fixed_profile,
                }
                if arguments.get("model") is not None:
                    call["model"] = arguments["model"]
                if arguments.get("confirmed_plan_id") is not None:
                    call["confirmed_plan_id"] = str(arguments["confirmed_plan_id"])
                payload = self.task_runner.run(**call)
            else:
                return self._error(request_id, -32602, "unknown tool")
        except (ContractError, ValueError, RuntimeError) as exc:
            return self._result(
                request_id,
                {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {"error_code": type(exc).__name__, "message": str(exc)},
                                sort_keys=True,
                            ),
                        }
                    ],
                    "isError": True,
                },
            )
        return self._result(
            request_id,
            {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            payload, sort_keys=True, separators=(",", ":")
                        ),
                    }
                ],
                "isError": False,
            },
        )

def serve_stdio(
    control: Any | None = None,
    *,
    input_stream: BinaryIO | None = None,
    output_stream: BinaryIO | None = None,
    profile: str = "chunsa",
) -> None:
    if control is None:
        configured_socket = os.environ.get("AI_ORCHESTRATOR_SOCKET")
        socket_path = (
            Path(configured_socket).expanduser().resolve()
            if configured_socket
            else default_socket_path()
        )
        control = (
            BrokerControlPlane(socket_path)
            if configured_socket or socket_path.exists()
            else ControlPlane()
        )
    server = MCPQueryServer(control, fixed_profile=profile)
    reader = input_stream or sys.stdin.buffer
    writer = output_stream or sys.stdout.buffer
    for raw in reader:
        if len(raw) > MAX_MCP_MESSAGE_BYTES:
            response = server._error(None, -32700, "message too large")
        else:
            try:
                request = json.loads(raw)
                if not isinstance(request, dict):
                    raise TypeError
                response = server.handle(request)
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
                response = server._error(None, -32700, "parse error")
        if response is not None:
            writer.write(
                json.dumps(response, sort_keys=True, separators=(",", ":")).encode()
                + b"\n"
            )
            writer.flush()
