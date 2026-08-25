"""MCP surface for safe inspection and explicit bounded delegation."""

from __future__ import annotations

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, BinaryIO, ClassVar

from ai_orchestrator.abacus import AbacusDelegator
from ai_orchestrator.broker import broker_request, default_socket_path
from ai_orchestrator.config import config_root, load_profile
from ai_orchestrator.control_plane import ControlPlane
from ai_orchestrator.delegation import OrchestratorTaskRunner
from ai_orchestrator.models import ContractError
from ai_orchestrator.workflows import WorkflowRunner

MAX_MCP_MESSAGE_BYTES = 1_000_000

# El bucle era estrictamente secuencial, asi que una tarea larga bloqueaba el
# servidor entero: en los registros hay dos orchestrator_run_task solapadas de
# 17m42s y 23m38s, la segunda esperando en el buffer de stdin. Quien acota de
# verdad la concurrencia es AdmissionController, que ya reparte leases entre
# procesos; el bucle solo le impedia llegar a ejercerse.
MAX_MCP_WORKERS = 4

# Cada cuanto se informa de que una herramienta sigue viva, cuando el cliente
# ha pedido progreso con un progressToken.
PROGRESS_INTERVAL_SECONDS = 15.0
SUPPORTED_MCP_VERSIONS = ("2025-11-25", "2025-06-18")
MCP_RESOURCE_SECTIONS = (
    "providers",
    "models",
    "profiles",
    "routes",
    "usage",
    "extensions",
    "updates",
    "sessions",
    "capabilities",
)
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
                "profile": {"type": "string"},
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
            "properties": {"profile": {"type": "string"}},
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
                "profile": {"type": "string"},
                "pin": {"type": "string"},
                "observations": {"type": "object"},
            },
            "required": ["task"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestrator_delegate_abacus",
        "description": (
            "Delegate a bounded no-tools task through the logged-in Abacus Pro CLI. "
            "This may consume included subscription credits and never purchases more."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "minLength": 1, "maxLength": 100000},
                "task_type": {
                    "type": "string",
                    "enum": [
                        "architecture",
                        "review",
                        "reasoning",
                        "research",
                        "security",
                        "agentic_implementation",
                        "implementation",
                        "bulk_refactor",
                        "tests",
                        "docs",
                        "boilerplate",
                        "long_context",
                    ],
                },
                "model": {
                    "type": "string",
                    "enum": ["Grok 4.6", "Qwen3.8 Max", "Qwen3.8 27B", "ZAI GLM 5.3"],
                },
                "sensitivity": {
                    "type": "string",
                    "enum": ["public", "internal"],
                    "default": "internal",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "minimum": 10,
                    "maximum": 600,
                    "default": 300,
                },
                "allow_experimental": {"type": "boolean", "default": False},
            },
            "required": ["prompt", "task_type"],
            "additionalProperties": False,
        },
        "annotations": {
            "title": "Delegate to Abacus",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    },
    {
        "name": "orchestrator_run_task",
        "description": (
            "Run an explicit AI-Orchestrator task spec through the common policy, "
            "provider adapters, isolation, gates and metadata-only telemetry. "
            "Writer tasks can only return a patch pending review."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "spec_path": {"type": "string", "minLength": 1},
                "profile": {
                    "type": "string",
                    "pattern": "^[A-Za-z0-9_-]{1,64}$",
                    "default": "default",
                },
                "model": {"type": "string", "minLength": 1},
            },
            "required": ["spec_path"],
            "additionalProperties": False,
        },
        "annotations": {
            "title": "Run orchestrated task",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    },
)

MCP_SUBMIT_TOOL = {
    "name": "orchestrator_submit_task",
    "description": (
        "Submit an inline task to the profile-bound plan, execute and review "
        "workflow. The repository is derived from the server profile and the "
        "prompt is retained only in memory."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "minLength": 1, "maxLength": 100000},
            "task": {
                "type": "object",
                "properties": {
                    "task_type": {"type": "string", "minLength": 1},
                    "sensitivity": {
                        "type": "string",
                        "enum": ["public", "internal"],
                        "default": "internal",
                    },
                    "write_mode": {
                        "type": "string",
                        "enum": ["read_only", "validated_patch"],
                        "default": "read_only",
                    },
                    "input_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 200,
                    },
                    "allowed_write_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 500,
                    },
                    "gate_profile": {"type": "string", "default": "profile"},
                    "timeout_seconds": {
                        "type": "integer",
                        "minimum": 10,
                        "maximum": 1800,
                    },
                    "max_context_tokens": {
                        "type": "integer",
                        "minimum": 512,
                        "maximum": 600000,
                    },
                    "short_task": {"type": "boolean", "default": False},
                    "qwen_plan": {"type": "boolean", "default": True},
                    "qwen_patch": {"type": "boolean", "default": False},
                },
                "required": ["task_type"],
                "additionalProperties": False,
            },
        },
        "required": ["prompt", "task"],
        "additionalProperties": False,
    },
    "annotations": {
        "title": "Submit profile-bound orchestrated task",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
}

STRICT_MCP_TOOL_NAMES = frozenset(
    {
        "orchestrator_inventory",
        "orchestrator_diagnostics",
        "orchestrator_simulate_route",
        "orchestrator_submit_task",
    }
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


class FallbackControlPlane:
    """Prefer the private broker, but keep MCP usable during broker recovery."""

    def __init__(self, primary: Any, direct: Any) -> None:
        self.primary = primary
        self.direct = direct

    def _run(self, method: str, *args: object, **kwargs: object) -> dict[str, Any]:
        try:
            return getattr(self.primary, method)(*args, **kwargs)
        except (OSError, TimeoutError, ConnectionError, ContractError):
            return getattr(self.direct, method)(*args, **kwargs)

    def query(self, section: str, **params: object) -> dict[str, Any]:
        return self._run("query", section, **params)

    def diagnostics(self, profile: str = "default") -> dict[str, Any]:
        return self._run("diagnostics", profile)

    def simulate_route(
        self,
        task: dict[str, Any],
        *,
        profile: str = "default",
        observations: dict[str, dict[str, object]] | None = None,
        pin: str | None = None,
    ) -> dict[str, Any]:
        return self._run(
            "simulate_route",
            task,
            profile=profile,
            observations=observations,
            pin=pin,
        )


class MCPQueryServer:
    def __init__(
        self,
        control: Any,
        *,
        delegator: Any | None = None,
        task_runner: Any | None = None,
        required_profile: str | None = None,
        required_repo: Path | None = None,
    ) -> None:
        self.control = control
        self.delegator = delegator or AbacusDelegator()
        self.task_runner = task_runner or OrchestratorTaskRunner()
        self.required_profile = required_profile
        self.required_repo = required_repo.resolve() if required_repo else None
        self._workflow_runner: Any | None = None

    def _profile(self, requested: object = None) -> str:
        profile = str(requested or self.required_profile or "default")
        if self.required_profile is not None and profile != self.required_profile:
            raise ContractError(
                f"MCP server is bound to profile {self.required_profile}"
            )
        return profile

    def _tools(self) -> list[dict[str, Any]]:
        """List the tools this server actually accepts.

        Sin perfil hay una sola superficie con las seis herramientas: el
        registro global se quedaba sin submit_task, que es la via de envio que
        no representa el prompt en disco. Atado a un perfil se conserva el
        conjunto estricto, que es una frontera de seguridad deliberada de
        CHUNSA y no un accidente.
        """

        if self.required_profile is None:
            return [*MCP_TOOLS, MCP_SUBMIT_TOOL]
        tools = [tool for tool in MCP_TOOLS if tool["name"] in STRICT_MCP_TOOL_NAMES]
        tools.append(MCP_SUBMIT_TOOL)
        return tools

    @property
    def workflow_runner(self) -> Any:
        """The runner that submit_task needs, built on demand when unbound."""

        if self.required_profile is not None:
            return self.task_runner
        if self._workflow_runner is None:
            from ai_orchestrator.workflows import WorkflowRunner

            self._workflow_runner = WorkflowRunner()
        return self._workflow_runner

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
        if method in {"notifications/initialized", "notifications/cancelled"}:
            # El bucle intercepta las cancelaciones antes de llegar aqui.
            return None
        if method == "initialize":
            params = request.get("params", {})
            requested_version = (
                params.get("protocolVersion") if isinstance(params, dict) else None
            )
            negotiated = (
                requested_version
                if requested_version in SUPPORTED_MCP_VERSIONS
                else SUPPORTED_MCP_VERSIONS[0]
            )
            return self._result(
                request_id,
                {
                    "protocolVersion": negotiated,
                    "capabilities": {
                        "tools": {"listChanged": False},
                        "resources": {"subscribe": False, "listChanged": False},
                    },
                    "serverInfo": {"name": "ai-orchestrator", "version": "0.2.0"},
                },
            )
        if method == "ping":
            return self._result(request_id, {})
        if method == "resources/list":
            params = request.get("params", {})
            if not isinstance(params, dict):
                return self._error(request_id, -32602, "invalid params")
            raw_cursor = params.get("cursor", "0")
            try:
                offset = int(raw_cursor)
            except (TypeError, ValueError):
                return self._error(request_id, -32602, "invalid resource cursor")
            if offset < 0 or offset > len(MCP_RESOURCE_SECTIONS):
                return self._error(request_id, -32602, "invalid resource cursor")
            end = min(len(MCP_RESOURCE_SECTIONS), offset + 5)
            payload: dict[str, object] = {
                "resources": [
                    {
                        "uri": f"orchestrator://inventory/{section}",
                        "name": f"AI Orchestrator {section}",
                        "description": "Secret-free local control-plane metadata",
                        "mimeType": "application/json",
                    }
                    for section in MCP_RESOURCE_SECTIONS[offset:end]
                ]
            }
            if end < len(MCP_RESOURCE_SECTIONS):
                payload["nextCursor"] = str(end)
            return self._result(request_id, payload)
        if method == "resources/read":
            params = request.get("params")
            uri = params.get("uri") if isinstance(params, dict) else None
            prefix = "orchestrator://inventory/"
            if not isinstance(uri, str) or not uri.startswith(prefix):
                return self._error(request_id, -32602, "invalid resource URI")
            section = uri.removeprefix(prefix)
            if section not in MCP_RESOURCE_SECTIONS:
                return self._error(request_id, -32602, "unknown resource")
            try:
                resource = self.control.query(section)
            except (ContractError, ValueError, RuntimeError) as exc:
                return self._error(request_id, -32002, str(exc))
            return self._result(
                request_id,
                {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": json.dumps(
                                resource, sort_keys=True, separators=(",", ":")
                            ),
                        }
                    ]
                },
            )
        if method == "tools/list":
            return self._result(request_id, {"tools": self._tools()})
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
            if self.required_profile is not None and name not in STRICT_MCP_TOOL_NAMES:
                raise ContractError("tool is unavailable on the profile-bound MCP")
            if name == "orchestrator_inventory":
                section = str(arguments.get("section", ""))
                options: dict[str, object] = {}
                if section in {"credentials", "extensions"} and arguments.get(
                    "profile"
                ):
                    options["profile"] = self._profile(arguments["profile"])
                payload = self.control.query(section, **options)
            elif name == "orchestrator_diagnostics":
                payload = self.control.diagnostics(
                    self._profile(arguments.get("profile"))
                )
            elif name == "orchestrator_simulate_route":
                task = arguments.get("task")
                if not isinstance(task, dict):
                    raise ContractError("task must be an object")
                payload = self.control.simulate_route(
                    task,
                    profile=self._profile(arguments.get("profile")),
                    observations=(
                        arguments.get("observations")
                        if isinstance(arguments.get("observations"), dict)
                        else None
                    ),
                    pin=str(arguments["pin"]) if arguments.get("pin") else None,
                )
            elif name == "orchestrator_submit_task":
                prompt = arguments.get("prompt")
                metadata = arguments.get("task")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ContractError("prompt must be a non-empty string")
                if len(prompt.encode("utf-8")) > 100_000:
                    raise ContractError("prompt exceeds the 100000-byte limit")
                if not isinstance(metadata, dict):
                    raise ContractError("task must be an object")
                if {"repo", "profile", "prompt", "prompt_file"} & metadata.keys():
                    raise ContractError("repository and prompt are derived by the MCP")
                payload = self.workflow_runner.submit(
                    prompt=prompt,
                    metadata=dict(metadata),
                    profile=self._profile(arguments.get("profile")),
                )
            elif name == "orchestrator_delegate_abacus":
                prompt = arguments.get("prompt")
                task_type = arguments.get("task_type")
                if not isinstance(prompt, str) or not isinstance(task_type, str):
                    raise ContractError("prompt and task_type must be strings")
                payload = self.delegator.delegate(
                    prompt=prompt,
                    task_type=task_type,
                    model=(
                        str(arguments["model"])
                        if arguments.get("model") is not None
                        else None
                    ),
                    sensitivity=str(arguments.get("sensitivity", "internal")),
                    timeout_seconds=arguments.get("timeout_seconds", 300),
                    allow_experimental=arguments.get("allow_experimental", False),
                )
            elif name == "orchestrator_run_task":
                spec_path = arguments.get("spec_path")
                if not isinstance(spec_path, str):
                    raise ContractError("spec_path must be a string")
                call: dict[str, Any] = {
                    "spec_path": spec_path,
                    "profile": str(arguments.get("profile", "default")),
                }
                if arguments.get("model") is not None:
                    call["model"] = arguments["model"]
                payload = self.task_runner.run(**call)
            else:
                return self._error(request_id, -32601, "unknown tool")
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
    profile: str | None = None,
    launched_repo: Path | None = None,
    configuration_root: Path | None = None,
    max_workers: int = MAX_MCP_WORKERS,
) -> None:
    selected_profile = None
    selected_root = (configuration_root or config_root()).resolve()
    if profile is not None:
        selected_profile = load_profile(profile, selected_root)
        if not selected_profile.repo_root:
            raise ContractError("profile-bound MCP requires a repository")
        expected_repo = Path(selected_profile.repo_root).expanduser().resolve()
        launch_value = launched_repo or Path(
            os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
        )
        if launch_value.expanduser().resolve() != expected_repo:
            raise ContractError("MCP launch repository does not match its profile")
    if control is None:
        configured_socket = os.environ.get("AI_ORCHESTRATOR_SOCKET")
        socket_path = (
            Path(configured_socket).expanduser().resolve()
            if configured_socket
            else default_socket_path()
        )
        direct = ControlPlane()
        control = (
            FallbackControlPlane(BrokerControlPlane(socket_path), direct)
            if configured_socket or socket_path.exists()
            else direct
        )
    runner = (
        WorkflowRunner(configuration_root=selected_root)
        if selected_profile is not None
        else OrchestratorTaskRunner(configuration_root=selected_root)
    )
    server = MCPQueryServer(
        control,
        task_runner=runner,
        required_profile=profile,
        required_repo=(
            Path(selected_profile.repo_root) if selected_profile is not None else None
        ),
    )
    reader = input_stream or sys.stdin.buffer
    writer = output_stream or sys.stdout.buffer
    _serve_loop(server, reader, writer, max_workers=max_workers)


def _serve_loop(
    server: MCPQueryServer,
    reader: BinaryIO,
    writer: BinaryIO,
    *,
    max_workers: int = MAX_MCP_WORKERS,
) -> None:
    """Read NDJSON requests and answer them on a bounded pool.

    Reading stays on this thread because stdin is a single stream. Everything
    else runs on the pool, so a long tool no longer holds the whole server.
    """

    write_lock = threading.Lock()
    state_lock = threading.Lock()
    cancelled: set[str] = set()
    started: set[str] = set()

    def emit(message: dict[str, Any]) -> None:
        payload = json.dumps(message, sort_keys=True, separators=(",", ":")).encode()
        with write_lock:
            writer.write(payload + b"\n")
            writer.flush()

    def progress_token(request: dict[str, Any]) -> object:
        params = request.get("params")
        meta = params.get("_meta") if isinstance(params, dict) else None
        return meta.get("progressToken") if isinstance(meta, dict) else None

    def run(request: dict[str, Any]) -> None:
        request_id = request.get("id")
        key = json.dumps(request_id, sort_keys=True)
        with state_lock:
            if key in cancelled:
                # Cancelada mientras esperaba turno: es justo el caso que
                # antes se quedaba encolado detras de una tarea de 17 minutos.
                cancelled.discard(key)
                return
            started.add(key)
        token = progress_token(request)
        stop = threading.Event()
        beacon: threading.Thread | None = None
        if token is not None:

            def announce() -> None:
                sent = 0
                while not stop.wait(PROGRESS_INTERVAL_SECONDS):
                    sent += 1
                    emit(
                        {
                            "jsonrpc": "2.0",
                            "method": "notifications/progress",
                            "params": {"progressToken": token, "progress": sent},
                        }
                    )

            beacon = threading.Thread(target=announce, daemon=True)
            beacon.start()
        try:
            response = server.handle(request)
        except Exception as exc:  # noqa: BLE001
            response = server._error(request_id, -32603, f"internal error: {exc}")
        finally:
            stop.set()
            if beacon is not None:
                beacon.join(timeout=1.0)
            with state_lock:
                started.discard(key)
        if response is not None:
            emit(response)

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        for raw in reader:
            if len(raw) > MAX_MCP_MESSAGE_BYTES:
                emit(server._error(None, -32700, "message too large"))
                continue
            try:
                request = json.loads(raw)
                if not isinstance(request, dict):
                    raise TypeError
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
                emit(server._error(None, -32700, "parse error"))
                continue
            if request.get("method") == "notifications/cancelled":
                params = request.get("params")
                target = params.get("requestId") if isinstance(params, dict) else None
                key = json.dumps(target, sort_keys=True)
                with state_lock:
                    if key not in started:
                        cancelled.add(key)
                continue
            pool.submit(run, request)
