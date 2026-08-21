#!/usr/bin/env python3
"""Repeatable local reliability check for the headless process boundary."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ai_orchestrator.supervisor import run_supervised


def _run_echo(root: Path, value: bytes) -> bool:
    outcome = run_supervised(
        [
            sys.executable,
            "-c",
            "import sys; data=sys.stdin.buffer.read(); sys.stdout.buffer.write(data)",
        ],
        input_data=value,
        cwd=root,
        env={"PATH": os.environ.get("PATH", "")},
        timeout_seconds=2,
    )
    return outcome.returncode == 0 and outcome.stdout == value and not outcome.timed_out


def _run_eof(root: Path) -> bool:
    outcome = run_supervised(
        [
            sys.executable,
            "-c",
            "import sys; raise SystemExit(0 if not sys.stdin.buffer.read(1) else 1)",
        ],
        input_data=None,
        cwd=root,
        env={"PATH": os.environ.get("PATH", "")},
        timeout_seconds=2,
    )
    return outcome.returncode == 0 and not outcome.timed_out


def _qwen_open_stdin_check(root: Path) -> dict[str, object]:
    command = [
        "qwen",
        "--auth-type",
        "openai",
        "--bare",
        "--output-format",
        "json",
        "--approval-mode",
        "plan",
        "--max-wall-time",
        "3s",
        "--max-session-turns",
        "1",
        "--max-tool-calls",
        "0",
        "--model",
        "ds/deepseek-v4-flash",
    ]
    environment = {
        "HOME": os.environ.get("HOME", str(root)),
        "PATH": os.environ.get("PATH", ""),
        "OPENAI_API_KEY": "reliability-check-not-a-secret",
        "OPENAI_BASE_URL": "http://127.0.0.1:9/v1",
    }
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=root,
        env=environment,
        start_new_session=True,
    )
    try:
        returncode = process.wait(timeout=3)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        return {"passed": True, "exit_code": returncode, "elapsed_ms": elapsed_ms}
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        return {"passed": False, "exit_code": 124, "elapsed_ms": 3000}
    finally:
        if process.stdin is not None:
            process.stdin.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-open-stdin", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        sequential = [_run_echo(root, f"case-{index}".encode()) for index in range(50)]
        eof = [_run_eof(root) for _ in range(25)]
        with ThreadPoolExecutor(max_workers=5) as pool:
            concurrent = list(
                pool.map(
                    lambda index: _run_echo(root, f"parallel-{index}".encode()),
                    range(25),
                )
            )
        qwen = _qwen_open_stdin_check(root) if args.qwen_open_stdin else None
    checks = sequential + eof + concurrent
    payload = {
        "schema_version": 1,
        "cases": len(checks),
        "passed": sum(checks),
        "failed": len(checks) - sum(checks),
        "elapsed_ms": int((time.monotonic() - started) * 1000),
        "qwen_open_stdin": qwen,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if all(checks) and (qwen is None or qwen["passed"] is True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
