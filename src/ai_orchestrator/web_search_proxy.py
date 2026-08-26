"""Strict loopback MCP egress proxy for bounded public-web research."""

from __future__ import annotations

import ipaddress
import json
import os
import secrets
import threading
import urllib.error
import urllib.parse
import urllib.request
from contextlib import AbstractContextManager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Self

from ai_orchestrator.credentials import resolve_credential
from ai_orchestrator.models import ContractError

SEARCH_PROXY_HOST = "127.0.0.1"
SEARCH_PROXY_PORT = 20130
EXA_RESEARCH_MCP_URL = "https://mcp.exa.ai/mcp?tools=web_search_exa,web_fetch_exa"
MAX_REQUEST_BYTES = 256 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_QUERY_CHARS = 4096
MAX_RESULTS = 20
MAX_FETCH_URLS = 5
MAX_FETCH_CHARACTERS = 50_000
_SENSITIVE_QUERY_MARKERS = frozenset(
    {
        "api_key",
        "apikey",
        "auth",
        "credential",
        "key",
        "password",
        "secret",
        "signature",
        "token",
    }
)
_ALLOWED_METHODS = frozenset(
    {
        "initialize",
        "notifications/initialized",
        "ping",
        "tools/list",
        "tools/call",
    }
)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _SearchProxyServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        *,
        credential: str,
        upstream_url: str,
    ) -> None:
        super().__init__(address, _SearchProxyHandler)
        self.credential = credential
        self.upstream_url = upstream_url
        self.opener = urllib.request.build_opener(_NoRedirect)


class _SearchProxyHandler(BaseHTTPRequestHandler):
    server: _SearchProxyServer
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: Any) -> None:
        """Do not retain search queries, results, paths, or credentials."""

    def _error(self, status: int, code: str) -> None:
        payload = json.dumps({"error": {"code": code}}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _authenticated(self) -> bool:
        authorization = self.headers.get("Authorization", "")
        if secrets.compare_digest(authorization, f"Bearer {self.server.credential}"):
            return True
        self._error(401, "WEB_SEARCH_PROXY_AUTH")
        return False

    def do_GET(self) -> None:
        if self.path == "/health":
            payload = b'{"ok":true,"public_research_only":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path.rstrip("/") != "/mcp" or not self._authenticated():
            if self.path.rstrip("/") != "/mcp":
                self._error(404, "WEB_SEARCH_PROXY_PATH")
            return
        self._forward(method="GET")

    def do_DELETE(self) -> None:
        if self.path.rstrip("/") != "/mcp":
            self._error(404, "WEB_SEARCH_PROXY_PATH")
            return
        if self._authenticated():
            self._forward(method="DELETE")

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/mcp":
            self._error(404, "WEB_SEARCH_PROXY_PATH")
            return
        if not self._authenticated():
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._error(411, "WEB_SEARCH_PROXY_LENGTH")
            return
        if not 1 <= length <= MAX_REQUEST_BYTES:
            self._error(413, "WEB_SEARCH_PROXY_SIZE")
            return
        body = self.rfile.read(length)
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._error(400, "WEB_SEARCH_PROXY_JSON")
            return
        error = _validate_mcp_request(payload)
        if error is not None:
            status, code = error
            self._error(status, code)
            return
        self._forward(method="POST", body=body)

    def _forward(self, *, method: str, body: bytes | None = None) -> None:
        headers = {
            "Accept": self.headers.get("Accept", "application/json, text/event-stream"),
            "Content-Type": "application/json",
            "User-Agent": "ai-orchestrator-public-research-proxy/1",
        }
        for name in ("MCP-Protocol-Version", "Mcp-Session-Id", "Last-Event-ID"):
            value = self.headers.get(name)
            if value:
                headers[name] = value
        request = urllib.request.Request(
            self.server.upstream_url,
            data=body,
            method=method,
            headers=headers,
        )
        try:
            upstream = self.server.opener.open(request, timeout=60)
        except urllib.error.HTTPError as exc:
            upstream = exc
        except (urllib.error.URLError, TimeoutError, OSError):
            self._error(502, "WEB_SEARCH_PROXY_UPSTREAM")
            return
        with upstream:
            response = upstream.read(MAX_RESPONSE_BYTES + 1)
            if len(response) > MAX_RESPONSE_BYTES:
                self._error(502, "WEB_SEARCH_PROXY_RESPONSE_SIZE")
                return
            self.send_response(upstream.status)
            for name in (
                "Content-Type",
                "Content-Length",
                "Mcp-Session-Id",
                "MCP-Protocol-Version",
            ):
                value = upstream.headers.get(name)
                if value and name.lower() != "content-length":
                    self.send_header(name, value)
            self.send_header("Content-Length", str(len(response)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(response)


def _validate_mcp_request(payload: object) -> tuple[int, str] | None:
    if not isinstance(payload, dict) or payload.get("jsonrpc") != "2.0":
        return 400, "WEB_SEARCH_PROXY_PROTOCOL"
    method = payload.get("method")
    if method not in _ALLOWED_METHODS:
        return 403, "WEB_SEARCH_PROXY_METHOD"
    if method != "tools/call":
        return None
    params = payload.get("params")
    if not isinstance(params, dict) or params.get("name") not in {
        "web_search_exa",
        "web_fetch_exa",
    }:
        return 403, "WEB_SEARCH_PROXY_TOOL"
    arguments = params.get("arguments")
    if not isinstance(arguments, dict):
        return 400, "WEB_SEARCH_PROXY_ARGUMENTS"
    if params["name"] == "web_fetch_exa":
        return _validate_fetch_arguments(arguments)
    if set(arguments) - {"query", "numResults"}:
        return 400, "WEB_SEARCH_PROXY_ARGUMENTS"
    query = arguments.get("query")
    if not isinstance(query, str) or not 1 <= len(query.strip()) <= MAX_QUERY_CHARS:
        return 400, "WEB_SEARCH_PROXY_QUERY"
    results = arguments.get("numResults", 10)
    if isinstance(results, bool) or not isinstance(results, (int, float)):
        return 400, "WEB_SEARCH_PROXY_RESULTS"
    if int(results) != results or not 1 <= int(results) <= MAX_RESULTS:
        return 400, "WEB_SEARCH_PROXY_RESULTS"
    return None


def _validate_fetch_arguments(arguments: dict[str, Any]) -> tuple[int, str] | None:
    if set(arguments) - {"urls", "maxCharacters"}:
        return 400, "WEB_SEARCH_PROXY_ARGUMENTS"
    urls = arguments.get("urls")
    if not isinstance(urls, list) or not 1 <= len(urls) <= MAX_FETCH_URLS:
        return 400, "WEB_SEARCH_PROXY_URLS"
    if any(not _is_public_https_url(value) for value in urls):
        return 403, "WEB_SEARCH_PROXY_PRIVATE_URL"
    max_characters = arguments.get("maxCharacters", 3000)
    if isinstance(max_characters, bool) or not isinstance(max_characters, (int, float)):
        return 400, "WEB_SEARCH_PROXY_FETCH_SIZE"
    if (
        int(max_characters) != max_characters
        or not 1 <= int(max_characters) <= MAX_FETCH_CHARACTERS
    ):
        return 400, "WEB_SEARCH_PROXY_FETCH_SIZE"
    return None


def _is_public_https_url(value: object) -> bool:
    if not isinstance(value, str) or not 1 <= len(value) <= 4096:
        return False
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
    except ValueError:
        return False
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
    ):
        return False
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith(
        (".localhost", ".local", ".internal")
    ):
        return False
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        if not address.is_global:
            return False
    for name, _value in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True):
        lowered = name.lower()
        if any(marker in lowered for marker in _SENSITIVE_QUERY_MARKERS):
            return False
    return True


class WebSearchProxy(AbstractContextManager["WebSearchProxy"]):
    """Fixed-destination MCP proxy exposing bounded Exa search and extraction."""

    def __init__(
        self,
        *,
        credential: str,
        port: int = SEARCH_PROXY_PORT,
        upstream_url: str = EXA_RESEARCH_MCP_URL,
    ) -> None:
        if not credential or "\n" in credential or "\x00" in credential:
            raise ContractError("Web search proxy credential is invalid")
        if not 0 <= port <= 65_535:
            raise ContractError("Web search proxy port is invalid")
        if upstream_url != EXA_RESEARCH_MCP_URL and not upstream_url.startswith(
            "http://127.0.0.1:"
        ):
            raise ContractError("Web search proxy upstream is not allowed")
        self._server = _SearchProxyServer(
            (SEARCH_PROXY_HOST, port),
            credential=credential,
            upstream_url=upstream_url,
        )
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{SEARCH_PROXY_HOST}:{self._server.server_port}/mcp"

    def __enter__(self) -> Self:
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="ai-orchestrator-web-search-proxy",
            daemon=True,
        )
        self._thread.start()
        return self

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def serve_forever(self) -> None:
        try:
            self._server.serve_forever()
        finally:
            self._server.server_close()

    def __exit__(self, *_exc: object) -> None:
        self.close()


def main() -> int:
    credential = resolve_credential("web_search_proxy_key")
    if not credential:
        raise ContractError("Web search proxy credential is unavailable")
    raw_port = os.environ.get("AI_ORCHESTRATOR_WEB_SEARCH_PORT", str(SEARCH_PROXY_PORT))
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise ContractError("Web search proxy port is invalid") from exc
    WebSearchProxy(credential=credential, port=port).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
