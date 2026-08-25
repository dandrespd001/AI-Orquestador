"""The stdio loop must not let one long tool hold the whole server.

The loop used to read and answer on the same thread, so a second request sat in
the stdin buffer until the first finished. The logs show two overlapping
orchestrator_run_task calls taking 17m42s and 23m38s, the second waiting on the
first for most of that time.
"""

from __future__ import annotations

import io
import json
import threading
import time
import unittest

from ai_orchestrator.mcp_tools import _serve_loop


class _RecordingWriter(io.RawIOBase):
    """Collect whole NDJSON messages, proving writes are not interleaved."""

    def __init__(self) -> None:
        self.lines: list[dict] = []
        self._lock = threading.Lock()

    def write(self, payload) -> int:
        text = bytes(payload).decode("utf-8")
        with self._lock:
            for line in text.splitlines():
                if line.strip():
                    self.lines.append(json.loads(line))
        return len(payload)

    def flush(self) -> None:
        return None


class _StubServer:
    """Minimal stand-in that lets one tool block until released."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.slow_started = threading.Event()

    def handle(self, request: dict) -> dict | None:
        name = (request.get("params") or {}).get("name")
        if name == "slow":
            self.slow_started.set()
            self.release.wait(timeout=10.0)
        return {"jsonrpc": "2.0", "id": request.get("id"), "result": {"tool": name}}

    @staticmethod
    def _error(request_id, code, message):
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }


def _request(request_id: int, name: str, **extra) -> bytes:
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {"name": name, **extra},
    }
    return (json.dumps(payload) + "\n").encode()


class ConcurrencyTests(unittest.TestCase):
    def test_a_fast_call_answers_while_a_slow_one_is_still_running(self) -> None:
        server = _StubServer()
        writer = _RecordingWriter()
        reader = io.BytesIO(_request(1, "slow") + _request(2, "fast"))

        thread = threading.Thread(
            target=_serve_loop, args=(server, reader, writer), kwargs={"max_workers": 4}
        )
        thread.start()
        try:
            self.assertTrue(server.slow_started.wait(timeout=5.0))
            deadline = time.time() + 5.0
            while time.time() < deadline and not writer.lines:
                time.sleep(0.01)
            # La respuesta rapida llega con la lenta todavia bloqueada; antes
            # esto era imposible.
            self.assertEqual([2], [line["id"] for line in writer.lines])
        finally:
            server.release.set()
            thread.join(timeout=10.0)

        self.assertEqual({1, 2}, {line["id"] for line in writer.lines})


class CancellationTests(unittest.TestCase):
    def test_a_queued_request_cancelled_before_it_starts_never_runs(self) -> None:
        server = _StubServer()
        server.release.set()
        writer = _RecordingWriter()
        cancel = (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/cancelled",
                    "params": {"requestId": 7},
                }
            ).encode()
            + b"\n"
        )
        reader = io.BytesIO(cancel + _request(7, "fast") + _request(8, "fast"))

        _serve_loop(server, reader, writer, max_workers=1)

        answered = [line["id"] for line in writer.lines]
        self.assertNotIn(7, answered)
        self.assertIn(8, answered)


class ProgressTests(unittest.TestCase):
    def test_no_progress_notifications_without_a_token(self) -> None:
        server = _StubServer()
        server.release.set()
        writer = _RecordingWriter()
        _serve_loop(server, io.BytesIO(_request(1, "fast")), writer, max_workers=1)
        methods = [line.get("method") for line in writer.lines]
        self.assertNotIn("notifications/progress", methods)


class MalformedInputTests(unittest.TestCase):
    def test_garbage_gets_a_parse_error_and_the_loop_keeps_going(self) -> None:
        server = _StubServer()
        server.release.set()
        writer = _RecordingWriter()
        reader = io.BytesIO(b"not json\n" + _request(3, "fast"))

        _serve_loop(server, reader, writer, max_workers=1)

        codes = [
            line.get("error", {}).get("code")
            for line in writer.lines
            if "error" in line
        ]
        self.assertIn(-32700, codes)
        self.assertIn(3, [line["id"] for line in writer.lines if "result" in line])


if __name__ == "__main__":
    unittest.main()
