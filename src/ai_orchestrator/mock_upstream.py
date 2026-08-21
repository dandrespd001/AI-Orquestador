"""Deterministic OpenAI-compatible upstream for isolated gateway benchmarks."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def response_for(path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    marker = (
        str(payload.get("metadata", {}).get("test_case", ""))
        if isinstance(payload.get("metadata"), dict)
        else ""
    )
    if marker == "force-429":
        return 429, {
            "error": {"type": "rate_limit_error", "message": "synthetic rate limit"}
        }
    if marker == "force-500":
        return 500, {
            "error": {"type": "server_error", "message": "synthetic upstream failure"}
        }
    model = str(payload.get("model", "mock-model"))
    if path.endswith("/responses"):
        return 200, {
            "id": "resp_mock",
            "object": "response",
            "model": model,
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "gateway-ok"}],
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }
    message: dict[str, Any] = {"role": "assistant", "content": "gateway-ok"}
    if payload.get("tools"):
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_mock",
                    "type": "function",
                    "function": {"name": "sum_values", "arguments": '{"a":2,"b":3}'},
                }
            ],
        }
    return 200, {
        "id": "chatcmpl_mock",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if payload.get("tools") else "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


class MockHandler(BaseHTTPRequestHandler):
    server_version = "AIOrchestratorMock/1"

    def do_GET(self) -> None:
        if self.path in {"/health", "/healthz"}:
            self._json(200, {"status": "ok"})
            return
        self._json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        if self.path not in {"/v1/chat/completions", "/v1/responses"}:
            self._json(404, {"error": {"message": "not found"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 1 <= length <= 1_048_576:
                raise ValueError("invalid length")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise TypeError("request must be an object")
        except (TypeError, ValueError, json.JSONDecodeError):
            self._json(400, {"error": {"message": "invalid JSON"}})
            return
        status, response = response_for(self.path, payload)
        if payload.get("stream") is True and status == 200:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            chunk = json.dumps(response, separators=(",", ":")).encode()
            self.wfile.write(b"data: " + chunk + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        self._json(status, response)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), MockHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
