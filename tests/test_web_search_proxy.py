from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ai_orchestrator.web_search_proxy import EXA_RESEARCH_MCP_URL, WebSearchProxy

ROOT = Path(__file__).resolve().parents[1]


class WebSearchProxyTests(unittest.TestCase):
    def test_proxy_forwards_only_authenticated_bounded_research_calls(self) -> None:
        observed: list[dict[str, object]] = []

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: object) -> None:
                pass

            def do_POST(self) -> None:
                length = int(self.headers["Content-Length"])
                payload = json.loads(self.rfile.read(length))
                observed.append(
                    {
                        "path": self.path,
                        "authorization": self.headers.get("Authorization"),
                        "user_agent": self.headers.get("User-Agent"),
                        "payload": payload,
                    }
                )
                response = json.dumps(
                    {"jsonrpc": "2.0", "id": payload.get("id"), "result": {}}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.send_header("Mcp-Session-Id", "upstream-session")
                self.end_headers()
                self.wfile.write(response)

        try:
            upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        except PermissionError:
            self.skipTest("sandbox forbids loopback sockets")
        thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        thread.start()
        try:
            upstream_url = (
                f"http://127.0.0.1:{upstream.server_port}/mcp?tools=web_search_exa"
            )
            with WebSearchProxy(
                credential="local-search-credential",
                port=0,
                upstream_url=upstream_url,
            ) as proxy:
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": "web_search_exa",
                        "arguments": {
                            "query": "official MiniMax documentation",
                            "numResults": 8,
                        },
                    },
                }
                request = urllib.request.Request(
                    proxy.url,
                    data=json.dumps(payload).encode(),
                    headers={
                        "Authorization": "Bearer local-search-credential",
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream",
                    },
                )
                with urllib.request.urlopen(request) as response:
                    self.assertEqual(
                        "upstream-session", response.headers["Mcp-Session-Id"]
                    )
                    self.assertEqual({}, json.loads(response.read())["result"])

                self.assertEqual(1, len(observed))
                self.assertEqual("/mcp?tools=web_search_exa", observed[0]["path"])
                self.assertIsNone(observed[0]["authorization"])
                self.assertEqual(
                    "ai-orchestrator-public-research-proxy/1",
                    observed[0]["user_agent"],
                )

                fetch = dict(payload)
                fetch["id"] = 2
                fetch["params"] = {
                    "name": "web_fetch_exa",
                    "arguments": {
                        "urls": ["https://example.com/public-report.pdf"],
                        "maxCharacters": 5000,
                    },
                }
                with urllib.request.urlopen(
                    urllib.request.Request(
                        proxy.url,
                        data=json.dumps(fetch).encode(),
                        headers={
                            "Authorization": "Bearer local-search-credential",
                            "Content-Type": "application/json",
                        },
                    )
                ) as response:
                    self.assertEqual({}, json.loads(response.read())["result"])
                self.assertEqual(2, len(observed))

                denied = dict(fetch)
                denied["params"] = {
                    "name": "web_fetch_exa",
                    "arguments": {"urls": ["https://127.0.0.1/private"]},
                }
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(
                        urllib.request.Request(
                            proxy.url,
                            data=json.dumps(denied).encode(),
                            headers={
                                "Authorization": "Bearer local-search-credential",
                                "Content-Type": "application/json",
                            },
                        )
                    )
                self.assertEqual(403, error.exception.code)
                error.exception.close()
                self.assertEqual(2, len(observed))
        finally:
            upstream.shutdown()
            upstream.server_close()
            thread.join(timeout=5)

    def test_proxy_rejects_missing_auth_and_unbounded_searches(self) -> None:
        try:
            proxy_context = WebSearchProxy(
                credential="local-search-credential",
                port=0,
                upstream_url="http://127.0.0.1:9/mcp?tools=web_search_exa",
            )
        except PermissionError:
            self.skipTest("sandbox forbids loopback sockets")
        with proxy_context as proxy:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "web_search_exa",
                    "arguments": {"query": "x", "numResults": 100},
                },
            }
            for authorization, expected in (
                (None, 401),
                ("Bearer local-search-credential", 400),
            ):
                headers = {"Content-Type": "application/json"}
                if authorization:
                    headers["Authorization"] = authorization
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(
                        urllib.request.Request(
                            proxy.url,
                            data=json.dumps(payload).encode(),
                            headers=headers,
                        )
                    )
                self.assertEqual(expected, error.exception.code)
                error.exception.close()


class HarnessWebSearchConfigurationTests(unittest.TestCase):
    def test_harness_exposes_search_only_through_loopback_proxy(self) -> None:
        bundle = (
            ROOT
            / "integrations/deepseek-harness/profiles/orchestrator-web/cordis.patch.yml"
        ).read_text(encoding="utf-8")
        harness_unit = (ROOT / "systemd/ai-orchestrator-harness.service").read_text(
            encoding="utf-8"
        )
        proxy_unit = (ROOT / "systemd/ai-orchestrator-web-search.service").read_text(
            encoding="utf-8"
        )

        self.assertIn("serverName: web-search", bundle)
        self.assertIn("transport: streamable-http", bundle)
        self.assertIn("http://127.0.0.1:20130/mcp", bundle)
        self.assertIn("AI_ORCHESTRATOR_WEB_SEARCH_CREDENTIAL", bundle)
        self.assertEqual(
            "https://mcp.exa.ai/mcp?tools=web_search_exa,web_fetch_exa",
            EXA_RESEARCH_MCP_URL,
        )
        shared_bundle = (
            ROOT / "integrations/deepseek-harness/bundle/cordis.patch.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("id: llm-pi-ai", shared_bundle)
        self.assertIn("minimax-orchestrator:", shared_bundle)
        self.assertIn("MiniMax-M3", shared_bundle)
        self.assertIn("AI_ORCHESTRATOR_HARNESS_CREDENTIAL", shared_bundle)
        self.assertIn("IPAddressDeny=any", harness_unit)
        self.assertIn("ai-orchestrator-web-search.service", harness_unit)
        self.assertNotIn("LoadCredentialEncrypted=", proxy_unit)
        self.assertNotIn("LoadCredentialEncrypted=", harness_unit)
        self.assertIn("AI_ORCHESTRATOR_CONFIG=%h/.config/ai-orchestrator", proxy_unit)
        self.assertIn("PYTHONPATH=%h/.local/share/ai-orchestrator/runtime", proxy_unit)
        self.assertIn("ExecStart=/usr/bin/python3 -m ai_orchestrator.web_search_proxy", proxy_unit)
        self.assertIn("ProtectHome=read-only", proxy_unit)
        self.assertNotIn("BindReadOnlyPaths=", proxy_unit)
        self.assertIn("web_search_proxy", proxy_unit)
        self.assertNotIn("ExecStartPost=", proxy_unit)


if __name__ == "__main__":
    unittest.main()
