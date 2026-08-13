from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

from ai_orchestrator.supervisor import run_supervised


class SupervisorTests(unittest.TestCase):
    def test_stdin_is_written_and_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outcome = run_supervised(
                [
                    sys.executable,
                    "-c",
                    "import sys; data=sys.stdin.buffer.read(); sys.stdout.buffer.write(data)",
                ],
                input_data=b"bounded prompt",
                cwd=directory,
                env={"PATH": os.environ.get("PATH", "")},
                timeout_seconds=2,
            )
        self.assertEqual(0, outcome.returncode)
        self.assertEqual(b"bounded prompt", outcome.stdout)
        self.assertFalse(outcome.timed_out)

    @unittest.skipUnless(Path("/proc").is_dir(), "Linux process assertions require /proc")
    def test_timeout_kills_the_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outcome = run_supervised(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    (
                        "import subprocess,time; "
                        "child=subprocess.Popen(['sleep','60']); "
                        "print(child.pid, flush=True); time.sleep(60)"
                    ),
                ],
                input_data=None,
                cwd=directory,
                env={"PATH": os.environ.get("PATH", "")},
                timeout_seconds=1,
                terminate_grace_seconds=0.2,
            )
        self.assertTrue(outcome.timed_out)
        self.assertEqual(124, outcome.returncode)
        child_pid = int(outcome.stdout.strip())
        for _ in range(20):
            if not Path(f"/proc/{child_pid}").exists():
                break
            time.sleep(0.05)
        self.assertFalse(Path(f"/proc/{child_pid}").exists(), "child process survived timeout")


if __name__ == "__main__":
    unittest.main()
