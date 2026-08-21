"""Loopback-only operational panel and JSON API."""

from __future__ import annotations

import hmac
import json
import secrets
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ai_orchestrator.control_plane import ControlPlane
from ai_orchestrator.models import ContractError
from ai_orchestrator.sessions import SessionStore

MAX_API_BODY_BYTES = 1_000_000
PANEL_HTML = """<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>AI Orchestrator</title><link rel="stylesheet" href="/app.css"></head>
<body><header><h1>AI Orchestrator</h1><p>Control local · sin secretos en pantalla</p></header>
<nav id="tabs"></nav><main><pre id="output">Cargando…</pre>
<section><h2>Credencial de sólo escritura</h2>
<form id="credential-form" autocomplete="off">
<input name="provider" placeholder="proveedor" required>
<input name="credential" placeholder="nombre" required>
<input name="value" type="password" value="" autocomplete="new-password" required>
<input name="approval" type="password" value="" autocomplete="off" placeholder="aprobación humana" required>
<button type="submit">Guardar o rotar</button></form></section></main>
<script src="/app.js" defer></script></body></html>"""

PANEL_CSS = """body{font:15px system-ui;margin:0;background:#10141a;color:#e8eef5}header,main,nav{padding:1rem 1.5rem}nav button,button,input{margin:.25rem;padding:.55rem}pre{white-space:pre-wrap;background:#171e27;padding:1rem;border-radius:.5rem;min-height:12rem}input{background:#fff;color:#111}h1{margin-bottom:.2rem}"""

PANEL_JS = """const sections=['providers','credentials','models','profiles','routes','usage','extensions','updates','sessions','capabilities','diagnostics'];
const out=document.querySelector('#output'),tabs=document.querySelector('#tabs');
async function load(name){const r=await fetch('/api/'+name,{cache:'no-store'});out.textContent=JSON.stringify(await r.json(),null,2)}
for(const name of sections){const b=document.createElement('button');b.textContent=name;b.onclick=()=>load(name);tabs.appendChild(b)}
document.querySelector('#credential-form').onsubmit=async(e)=>{e.preventDefault();const f=new FormData(e.target),provider=f.get('provider');const body={credential:f.get('credential'),value:f.get('value')};const r=await fetch('/api/credentials/'+encodeURIComponent(provider),{method:'POST',headers:{'Content-Type':'application/json','X-Human-Approval':f.get('approval')},body:JSON.stringify(body)});e.target.elements.value.value='';e.target.elements.approval.value='';out.textContent=JSON.stringify(await r.json(),null,2)};
load('providers');"""


class HumanActionGate:
    """One-time capability supplied out-of-band to a local human operator."""

    def __init__(self, token: str | None = None) -> None:
        self._token = token or secrets.token_urlsafe(32)

    @property
    def token(self) -> str:
        return self._token

    def consume(self, candidate: str | None) -> None:
        if not candidate or not hmac.compare_digest(candidate, self._token):
            raise ContractError("a valid one-time human approval is required")
        self._token = secrets.token_urlsafe(32)


def _json_body(raw: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("request body must be a JSON object") from exc
    if not isinstance(payload, dict):
        raise ContractError("request body must be a JSON object")
    return payload


class PanelApplication:
    def __init__(
        self, control: ControlPlane, gate: HumanActionGate | None = None
    ) -> None:
        self.control = control
        self.gate = gate or HumanActionGate()

    def dispatch(
        self,
        method: str,
        target: str,
        *,
        headers: dict[str, str] | None = None,
        body: bytes = b"",
    ) -> tuple[int, str, bytes]:
        parsed = urllib.parse.urlsplit(target)
        path = parsed.path
        request_headers = {key.lower(): value for key, value in (headers or {}).items()}
        if method == "GET" and path == "/":
            return 200, "text/html; charset=utf-8", PANEL_HTML.encode()
        if method == "GET" and path == "/app.js":
            return 200, "text/javascript; charset=utf-8", PANEL_JS.encode()
        if method == "GET" and path == "/app.css":
            return 200, "text/css; charset=utf-8", PANEL_CSS.encode()
        if method == "GET" and path.startswith("/api/"):
            section = path.removeprefix("/api/")
            query = urllib.parse.parse_qs(parsed.query)
            params: dict[str, object] = {}
            if (
                section in {"credentials", "diagnostics"}
                and query.get("profile")
                or section == "extensions"
                and query.get("profile")
            ):
                params["profile"] = query["profile"][0]
            payload = self.control.query(section, **params)
            return 200, "application/json", json.dumps(payload, sort_keys=True).encode()
        if method == "POST" and path == "/api/routes/simulate":
            payload = _json_body(body)
            task = payload.get("task")
            if not isinstance(task, dict):
                raise ContractError("route simulation requires a task object")
            result = self.control.simulate_route(
                task,
                profile=str(payload.get("profile", "default")),
                observations=payload.get("observations")
                if isinstance(payload.get("observations"), dict)
                else None,
                pin=str(payload["pin"]) if payload.get("pin") else None,
            )
            return 200, "application/json", json.dumps(result, sort_keys=True).encode()
        if method == "POST" and path.startswith("/api/credentials/"):
            provider = urllib.parse.unquote(path.removeprefix("/api/credentials/"))
            payload = _json_body(body)
            name = str(payload.get("credential", ""))
            value = payload.get("value")
            if not isinstance(value, str):
                raise ContractError("credential value is required")
            self.gate.consume(request_headers.get("x-human-approval"))
            result = self.control.store_credential(
                provider,
                name,
                value,
                profile=str(payload.get("profile", "default")),
                human_approved=True,
            )
            return 200, "application/json", json.dumps(result, sort_keys=True).encode()
        if (
            method == "POST"
            and path.startswith("/api/extensions/")
            and path.endswith("/activate")
        ):
            extension_id = urllib.parse.unquote(
                path.removeprefix("/api/extensions/")
                .removesuffix("/activate")
                .rstrip("/")
            )
            payload = _json_body(body)
            self.gate.consume(request_headers.get("x-human-approval"))
            result = self.control.activate_extension(
                extension_id,
                str(payload.get("profile", "default")),
                human_approved=True,
                host_risk_approved=bool(payload.get("host_risk_approved", False)),
            )
            return 200, "application/json", json.dumps(result, sort_keys=True).encode()
        if (
            method == "POST"
            and path.startswith("/api/sessions/")
            and path.endswith("/pin")
        ):
            session_id = urllib.parse.unquote(
                path.removeprefix("/api/sessions/").removesuffix("/pin").rstrip("/")
            )
            payload = _json_body(body)
            self.gate.consume(request_headers.get("x-human-approval"))
            result = SessionStore(
                self.control.state_root / "harness-sessions.sqlite"
            ).set_pinned(
                session_id, bool(payload.get("pinned", True)), human_approved=True
            )
            return 200, "application/json", json.dumps(result, sort_keys=True).encode()
        return 404, "application/json", b'{"error_code":"NOT_FOUND","ok":false}'


def make_handler(application: PanelApplication) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "AIOrchestrator/1"
        sys_version = ""

        def log_message(self, _format: str, *_args: object) -> None:
            return

        def _serve(self) -> None:
            self.connection.settimeout(5.0)
            authority = self.headers.get("Host", "").split(":", 1)[0].strip("[]")
            if authority not in {"127.0.0.1", "localhost", "::1"}:
                self.send_error(403)
                return
            length_text = self.headers.get("Content-Length", "0")
            try:
                length = int(length_text)
            except ValueError:
                length = MAX_API_BODY_BYTES + 1
            if length < 0 or length > MAX_API_BODY_BYTES:
                self.send_error(413)
                return
            body = self.rfile.read(length) if length else b""
            try:
                status, content_type, response = application.dispatch(
                    self.command,
                    self.path,
                    headers=dict(self.headers.items()),
                    body=body,
                )
            except (ContractError, ValueError, RuntimeError) as exc:
                status, content_type = 400, "application/json"
                response = json.dumps(
                    {
                        "ok": False,
                        "error_code": type(exc).__name__,
                        "message": str(exc),
                    },
                    sort_keys=True,
                ).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(response)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; object-src 'none'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(response)

        do_GET = _serve
        do_POST = _serve

    return Handler


class BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """Threaded loopback server with explicit backpressure."""

    daemon_threads = True
    request_queue_size = 16

    def __init__(self, *args: object, max_workers: int = 16, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._capacity = threading.BoundedSemaphore(max_workers)

    def process_request(self, request: object, client_address: object) -> None:
        if not self._capacity.acquire(blocking=False):
            self.close_request(request)  # type: ignore[arg-type]
            return
        super().process_request(request, client_address)  # type: ignore[arg-type]

    def process_request_thread(self, request: object, client_address: object) -> None:
        try:
            super().process_request_thread(
                request,
                client_address,  # type: ignore[arg-type]
            )
        finally:
            self._capacity.release()


def serve_panel(
    control: ControlPlane,
    *,
    gate: HumanActionGate | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    if host != "127.0.0.1":
        raise ContractError("the control panel may bind only to 127.0.0.1")
    if not 1024 <= port <= 65_535:
        raise ContractError("invalid control-panel port")
    server = BoundedThreadingHTTPServer(
        (host, port), make_handler(PanelApplication(control, gate))
    )
    server.serve_forever()
