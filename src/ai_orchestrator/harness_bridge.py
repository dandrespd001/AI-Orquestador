"""Content-blind loopback bridge from DeepSeek Harness to OmniRoute."""

from __future__ import annotations

import json
import secrets
import threading
import urllib.error
import urllib.request
from contextlib import AbstractContextManager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Self

from ai_orchestrator.models import ContractError

BRIDGE_HOST = "127.0.0.1"
BRIDGE_PORT = 20129
OMNIROUTE_CHAT_URL = "http://127.0.0.1:20128/v1/chat/completions"
MINIMAX_CHAT_URL = "https://api.minimax.io/v1/chat/completions"
DEEPSEEK_MODELS = frozenset({"deepseek-v4-pro", "deepseek-v4-flash"})
MINIMAX_MODELS = frozenset({"MiniMax-M3"})
ALLOWED_MODELS = DEEPSEEK_MODELS | MINIMAX_MODELS
MAX_REQUEST_BYTES = 8 * 1024 * 1024


class _BridgeServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        *,
        local_credential: str,
        upstream_credential: str,
        upstream_url: str,
        minimax_credential: str | None,
        minimax_upstream_url: str,
        allowed_models: frozenset[str],
    ) -> None:
        super().__init__(address, _BridgeHandler)
        self.local_credential = local_credential
        self.upstream_credential = upstream_credential
        self.upstream_url = upstream_url
        self.minimax_credential = minimax_credential
        self.minimax_upstream_url = minimax_upstream_url
        self.allowed_models = allowed_models


class _BridgeHandler(BaseHTTPRequestHandler):
    server: _BridgeServer
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: Any) -> None:
        """Never place request paths, prompts, or responses in a log."""

    def _error(self, status: int, code: str) -> None:
        payload = json.dumps({"error": {"code": code}}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._error(404, "HARNESS_BRIDGE_PATH")
            return
        payload = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        if self.path.rstrip("/") not in {"/chat/completions", "/v1/chat/completions"}:
            self._error(404, "HARNESS_BRIDGE_PATH")
            return
        authorization = self.headers.get("Authorization", "")
        if not secrets.compare_digest(
            authorization, f"Bearer {self.server.local_credential}"
        ):
            self._error(401, "HARNESS_BRIDGE_AUTH")
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._error(411, "HARNESS_BRIDGE_LENGTH")
            return
        if not 1 <= length <= MAX_REQUEST_BYTES:
            self._error(413, "HARNESS_BRIDGE_SIZE")
            return
        body = self.rfile.read(length)
        try:
            request_payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._error(400, "HARNESS_BRIDGE_JSON")
            return
        if (
            not isinstance(request_payload, dict)
            or request_payload.get("model") not in self.server.allowed_models
        ):
            self._error(400, "HARNESS_BRIDGE_MODEL")
            return
        model = request_payload["model"]
        if model in MINIMAX_MODELS:
            if not self.server.minimax_credential:
                self._error(503, "HARNESS_BRIDGE_MINIMAX_CREDENTIAL")
                return
            upstream_url = self.server.minimax_upstream_url
            upstream_credential = self.server.minimax_credential
        else:
            upstream_url = self.server.upstream_url
            upstream_credential = self.server.upstream_credential
        request = urllib.request.Request(
            upstream_url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {upstream_credential}",
                "Content-Type": "application/json",
                "Accept": self.headers.get("Accept", "application/json"),
            },
        )
        try:
            upstream = urllib.request.urlopen(request, timeout=330)
        except urllib.error.HTTPError as exc:
            upstream = exc
        except (urllib.error.URLError, TimeoutError, OSError):
            self._error(502, "HARNESS_BRIDGE_UPSTREAM")
            return
        with upstream:
            self.send_response(upstream.status)
            has_length = upstream.headers.get("Content-Length") is not None
            for name in ("Content-Type", "Content-Length"):
                value = upstream.headers.get(name)
                if value:
                    self.send_header(name, value)
            self.send_header("Cache-Control", "no-store")
            if not has_length:
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            while chunk := upstream.read(64 * 1024):
                self.wfile.write(chunk)
                self.wfile.flush()


class HarnessBridge(AbstractContextManager["HarnessBridge"]):
    """A private per-service bridge carrying the real key outside DSH."""

    def __init__(
        self,
        *,
        upstream_credential: str,
        port: int = BRIDGE_PORT,
        upstream_url: str = OMNIROUTE_CHAT_URL,
        minimax_credential: str | None = None,
        minimax_upstream_url: str = MINIMAX_CHAT_URL,
        allowed_models: frozenset[str] = DEEPSEEK_MODELS,
    ) -> None:
        if not upstream_credential:
            raise ContractError("Harness bridge requires the broker-held OmniRoute key")
        if not 0 <= port <= 65_535:
            raise ContractError("Harness bridge port is invalid")
        if not allowed_models <= ALLOWED_MODELS:
            raise ContractError("Harness bridge model allowlist is invalid")
        if allowed_models & MINIMAX_MODELS and not minimax_credential:
            raise ContractError("Harness bridge requires the broker-held MiniMax key")
        self.credential = secrets.token_urlsafe(32)
        self._server = _BridgeServer(
            (BRIDGE_HOST, port),
            local_credential=self.credential,
            upstream_credential=upstream_credential,
            upstream_url=upstream_url,
            minimax_credential=minimax_credential,
            minimax_upstream_url=minimax_upstream_url,
            allowed_models=allowed_models,
        )
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{BRIDGE_HOST}:{self._server.server_port}"

    def __enter__(self) -> Self:
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="ai-orchestrator-harness-bridge",
            daemon=True,
        )
        self._thread.start()
        return self

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def __exit__(self, *_exc: object) -> None:
        self.close()
