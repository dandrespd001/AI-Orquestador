from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from ai_orchestrator.kimi_acp import run_kimi_acp


FAKE_AGENT = r'''
import json, sys

def send(value):
    sys.stdout.write(json.dumps(value, separators=(",", ":")) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    if method == "initialize":
        send({"jsonrpc":"2.0","id":request["id"],"result":{"protocolVersion":1,"agentCapabilities":{},"authMethods":[]}})
    elif method == "session/new":
        send({"jsonrpc":"2.0","id":request["id"],"result":{"sessionId":"session-1"}})
    elif method == "session/set_config_option":
        send({"jsonrpc":"2.0","id":request["id"],"result":None})
    elif method == "session/prompt":
        path = request["params"]["prompt"][0]["text"].split("WRITE:", 1)[1]
        send({"jsonrpc":"2.0","id":100,"method":"fs/write_text_file","params":{"sessionId":"session-1","path":path,"content":"written\n"}})
        response = json.loads(sys.stdin.readline())
        if "result" in response:
            send({"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"session-1","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"done"}}}})
        send({"jsonrpc":"2.0","id":request["id"],"result":{"stopReason":"end_turn"}})
'''


class KimiAcpTests(unittest.TestCase):
    def test_reverse_file_write_is_scoped_and_prompt_stays_off_argv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worktree = root / "worktree"
            (worktree / "src").mkdir(parents=True)
            executable = root / "fake-kimi"
            executable.write_text(
                f"#!{sys.executable}\n" + textwrap.dedent(FAKE_AGENT), encoding="utf-8"
            )
            executable.chmod(0o700)
            target = worktree / "src" / "value.txt"
            outcome = run_kimi_acp(
                executable=str(executable),
                worktree=worktree,
                allowed_write_paths=("src",),
                environment={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                model="kimi-code/k3",
                prompt=f"WRITE:{target}".encode(),
                timeout_seconds=10,
            )
            self.assertEqual("written\n", target.read_text())
            self.assertEqual(b"done", outcome.stdout)
            self.assertEqual("kimi-code/k3", outcome.reported_model)

    def test_dotdot_traversal_cannot_escape_the_worktree(self) -> None:
        """A `..` inside an allowed prefix must not become a write outside.

        `Path.relative_to` is purely lexical: it keeps `..` components instead
        of normalising them, so `<wt>/src/../../escaped.txt` reduces to
        `src/../../escaped.txt`, whose `parents` literally contain `src` — the
        allowed prefix validates its own escape. These writes never reach
        `writer.py`, so the allowlist, the protected prefixes and the gates are
        all bypassed at once.

        The target stays inside the temporary directory on purpose: a red phase
        must not be able to damage the machine it runs on.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worktree = root / "worktree"
            (worktree / "src").mkdir(parents=True)
            executable = root / "fake-kimi"
            executable.write_text(
                f"#!{sys.executable}\n" + textwrap.dedent(FAKE_AGENT), encoding="utf-8"
            )
            executable.chmod(0o700)
            escaped = root / "escaped.txt"
            traversal = worktree / "src" / ".." / ".." / "escaped.txt"
            run_kimi_acp(
                executable=str(executable),
                worktree=worktree,
                allowed_write_paths=("src",),
                environment={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                model="kimi-code/k3",
                prompt=f"WRITE:{traversal}".encode(),
                timeout_seconds=10,
            )
            self.assertFalse(
                escaped.exists(),
                "the ACP write escaped the worktree through a `..` component",
            )


if __name__ == "__main__":
    unittest.main()
