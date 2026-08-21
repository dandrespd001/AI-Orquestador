"""Private Unix-socket broker used by the Harness administrative profile."""

from __future__ import annotations

import json
import os
import socket
import stat
import struct
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Self

from ai_orchestrator.capabilities import load_capability_packages
from ai_orchestrator.config import (
    config_root,
    load_models,
    load_profile,
    load_roles,
    validate_configuration,
)
from ai_orchestrator.credentials import authentication_inventory
from ai_orchestrator.extensions import ExtensionRegistry
from ai_orchestrator.models import ContractError, TaskSpec, ensure_private_directory
from ai_orchestrator.providers import provider_inventory, validate_provider_catalog
from ai_orchestrator.routing import plan_route
from ai_orchestrator.sessions import SessionStore
from ai_orchestrator.telemetry import TelemetryStore, default_state_root
from ai_orchestrator.updates import UpdateRegistry

MAX_BROKER_MESSAGE_BYTES = 65_536
READ_ONLY_METHODS = {
    "health",
    "providers.list",
    "models.list",
    "profiles.list",
    "routes.policy",
    "auth.status",
    "routes.simulate",
    "diagnostics",
    "usage.report",
    "extensions.list",
    "updates.list",
    "sessions.list",
    "capabilities.list",
}
FORBIDDEN_RESPONSE_KEYS = {
    "api_key",
    "authorization",
    "body",
    "content",
    "password",
    "prompt",
    "response",
    "secret",
    "token",
}


def default_socket_path() -> Path:
    override = os.environ.get("AI_ORCHESTRATOR_SOCKET")
    if override:
        return Path(override).expanduser().resolve()
    return default_state_root() / "broker" / "control.sock"


def _sanitize(value: object) -> object:
    if isinstance(value, dict):
        result: dict[str, object] = {}
        for key, nested in value.items():
            lowered = str(key).lower()
            if (
                lowered in FORBIDDEN_RESPONSE_KEYS
                or any(
                    marker in lowered
                    for marker in ("secret", "password", "api_key", "authorization")
                )
                or lowered.endswith(("_token", "_authorization"))
            ):
                continue
            result[str(key)] = _sanitize(nested)
        return result
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return value


def handle_broker_request(
    request: dict[str, Any],
    *,
    config: Path | None = None,
    telemetry: TelemetryStore | None = None,
) -> dict[str, Any]:
    method = str(request.get("method", ""))
    if method not in READ_ONLY_METHODS:
        raise ContractError("broker method is not available to Harness")
    params = request.get("params", {})
    if not isinstance(params, dict):
        raise ContractError("broker params must be an object")
    root = config or config_root()
    if method == "health":
        result: object = {
            "schema_version": 1,
            "ok": True,
            "transport": "unix",
            "write_methods_exposed": False,
        }
    elif method == "providers.list":
        result = {"schema_version": 1, "providers": provider_inventory(root)}
    elif method == "models.list":
        result = {
            "schema_version": 1,
            "models": [
                model.__dict__
                for model in sorted(load_models(root).values(), key=lambda row: row.id)
            ],
        }
    elif method == "profiles.list":
        directory = root / "profiles"
        result = {
            "schema_version": 1,
            "profiles": [
                load_profile(path.stem, root).__dict__
                for path in sorted(directory.glob("*.toml"))
                if not path.is_symlink()
            ],
        }
    elif method == "routes.policy":
        result = {
            "schema_version": 1,
            "hard_gates_before_scoring": True,
            "silent_fallback": False,
            "incremental_spend_requires_human_approval": True,
        }
    elif method == "auth.status":
        result = {
            "schema_version": 1,
            "profiles": authentication_inventory(
                root, profile=str(params.get("profile", "default"))
            ),
        }
    elif method == "routes.simulate":
        raw_task = params.get("task")
        if not isinstance(raw_task, dict):
            raise ContractError("routes.simulate requires a task object")
        profile_name = str(params.get("profile", "default"))
        observations = params.get("observations")
        if observations is not None and not isinstance(observations, dict):
            raise ContractError("route observations must be an object")
        route = plan_route(
            TaskSpec.from_mapping(raw_task),
            load_models(root),
            load_roles(root),
            load_profile(profile_name, root),
            observations=observations,
            pin=str(params["pin"]) if params.get("pin") else None,
        )
        result = route.as_dict()
    elif method == "diagnostics":
        profile_name = str(params.get("profile", "default"))
        result = {
            "schema_version": 1,
            "configuration": validate_configuration(profile_name, root),
            "providers": validate_provider_catalog(root),
            "secret_read_methods_exposed": False,
        }
    elif method == "usage.report":
        result = (telemetry or TelemetryStore()).report()
    elif method == "extensions.list":
        result = ExtensionRegistry().inventory(
            profile=str(params["profile"]) if params.get("profile") else None
        )
    elif method == "updates.list":
        result = UpdateRegistry().inventory()
    elif method == "sessions.list":
        result = SessionStore().inventory()
    else:
        result = {
            "schema_version": 1,
            "packages": [
                value.as_dict()
                for value in sorted(
                    load_capability_packages(root).values(), key=lambda row: row.name
                )
            ],
        }
    sanitized = _sanitize(result)
    if not isinstance(sanitized, dict):
        raise ContractError("broker handler produced an invalid response")
    return sanitized


class UnixBrokerServer:
    def __init__(
        self,
        path: Path | None = None,
        *,
        handler: Callable[[dict[str, Any]], dict[str, Any]] = handle_broker_request,
        connection_timeout_seconds: float = 2.0,
        max_workers: int = 8,
    ) -> None:
        if connection_timeout_seconds <= 0 or max_workers < 1:
            raise ValueError("broker limits must be positive")
        self.path = (path or default_socket_path()).expanduser().resolve()
        self.handler = handler
        self.connection_timeout_seconds = connection_timeout_seconds
        self.max_workers = max_workers
        self.socket: socket.socket | None = None
        self._stop = threading.Event()
        self._capacity = threading.BoundedSemaphore(max_workers)
        self._worker_lock = threading.Lock()
        self._workers: set[threading.Thread] = set()

    def open(self) -> None:
        ensure_private_directory(self.path.parent)
        if self.path.exists() or self.path.is_symlink():
            metadata = self.path.lstat()
            if not stat.S_ISSOCK(metadata.st_mode) or metadata.st_uid != os.getuid():
                raise ContractError("refusing to replace an unsafe broker path")
            self.path.unlink()
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(str(self.path))
            os.chmod(self.path, 0o600)
            server.listen(max(8, self.max_workers * 2))
        except Exception:
            server.close()
            raise
        self.socket = server
        self._stop.clear()

    def close(self) -> None:
        self._stop.set()
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        with self._worker_lock:
            workers = tuple(self._workers)
        for worker in workers:
            if worker is not threading.current_thread():
                worker.join(timeout=self.connection_timeout_seconds + 0.5)
        try:
            if self.path.is_socket() and self.path.lstat().st_uid == os.getuid():
                self.path.unlink()
        except FileNotFoundError:
            pass

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @staticmethod
    def _peer_uid(connection: socket.socket) -> int:
        if not hasattr(socket, "SO_PEERCRED"):
            return os.getuid()
        raw = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        _pid, uid, _gid = struct.unpack("3i", raw)
        return uid

    @staticmethod
    def _encode_response(response: dict[str, object]) -> bytes:
        encoded = json.dumps(response, sort_keys=True, separators=(",", ":")).encode()
        if len(encoded) > MAX_BROKER_MESSAGE_BYTES:
            encoded = json.dumps(
                {
                    "ok": False,
                    "error_code": "RESPONSE_TOO_LARGE",
                    "message": "broker response exceeds its metadata limit",
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        return encoded + b"\n"

    def _handle_connection(self, connection: socket.socket) -> None:
        with connection:
            connection.settimeout(self.connection_timeout_seconds)
            if self._peer_uid(connection) != os.getuid():
                return
            raw = bytearray()
            try:
                while b"\n" not in raw and len(raw) <= MAX_BROKER_MESSAGE_BYTES:
                    chunk = connection.recv(
                        min(4096, MAX_BROKER_MESSAGE_BYTES + 1 - len(raw))
                    )
                    if not chunk:
                        break
                    raw.extend(chunk)
                if len(raw) > MAX_BROKER_MESSAGE_BYTES:
                    raise ContractError("broker request is too large")
                payload = json.loads(bytes(raw).split(b"\n", 1)[0])
                if not isinstance(payload, dict):
                    raise ContractError("broker request must be an object")
                response = {"ok": True, "result": self.handler(payload)}
            except (
                ContractError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                TimeoutError,
            ) as exc:
                response = {
                    "ok": False,
                    "error_code": type(exc).__name__,
                    "message": str(exc),
                }
            try:
                connection.sendall(self._encode_response(response))
            except OSError:
                pass

    def serve_once(self) -> None:
        if self.socket is None:
            raise RuntimeError("broker socket is not open")
        connection, _ = self.socket.accept()
        self._handle_connection(connection)

    def _worker(self, connection: socket.socket) -> None:
        try:
            self._handle_connection(connection)
        finally:
            self._capacity.release()
            with self._worker_lock:
                self._workers.discard(threading.current_thread())

    def serve_forever(self) -> None:
        if self.socket is None:
            self.open()
        while not self._stop.is_set():
            assert self.socket is not None
            try:
                connection, _ = self.socket.accept()
            except OSError:
                if self._stop.is_set():
                    return
                raise
            if not self._capacity.acquire(blocking=False):
                try:
                    connection.sendall(
                        self._encode_response(
                            {
                                "ok": False,
                                "error_code": "BROKER_BUSY",
                                "message": "broker concurrency limit reached",
                            }
                        )
                    )
                finally:
                    connection.close()
                continue
            worker = threading.Thread(
                target=self._worker, args=(connection,), daemon=True
            )
            with self._worker_lock:
                self._workers.add(worker)
            worker.start()


def broker_request(
    method: str,
    params: dict[str, Any] | None = None,
    *,
    path: Path | None = None,
    timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    target = (path or default_socket_path()).expanduser().resolve()
    metadata = target.lstat()
    if not stat.S_ISSOCK(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise ContractError("unsafe broker socket")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise ContractError("broker socket must have mode 0600")
    payload = json.dumps(
        {"method": method, "params": params or {}},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    if len(payload) > MAX_BROKER_MESSAGE_BYTES:
        raise ContractError("broker request is too large")
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(timeout_seconds)
    try:
        client.connect(str(target))
        client.sendall(payload + b"\n")
        raw = bytearray()
        while b"\n" not in raw and len(raw) <= MAX_BROKER_MESSAGE_BYTES:
            chunk = client.recv(4096)
            if not chunk:
                break
            raw.extend(chunk)
    finally:
        client.close()
    if len(raw) > MAX_BROKER_MESSAGE_BYTES:
        raise ContractError("broker response is too large")
    response = json.loads(bytes(raw).split(b"\n", 1)[0])
    if not isinstance(response, dict):
        raise ContractError("broker returned an invalid response")
    return response
