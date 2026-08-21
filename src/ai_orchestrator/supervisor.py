"""Bounded subprocess execution with process-group cleanup."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProcessOutcome:
    returncode: int
    stdout: bytes
    stderr: bytes
    timed_out: bool = False
    output_limited: bool = False


def _signal_process_group(
    process: subprocess.Popen[bytes], sig: signal.Signals
) -> None:
    """Signal only the session created for ``process``.

    ``start_new_session=True`` makes the direct child the process-group leader,
    so using its pid here cannot accidentally target the caller's group.
    """

    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def run_supervised(
    command: list[str],
    *,
    input_data: bytes | None,
    cwd: str | Path,
    env: dict[str, str],
    timeout_seconds: int,
    terminate_grace_seconds: float = 3.0,
    max_stdout_bytes: int = 32 * 1024 * 1024,
    max_stderr_bytes: int = 1024 * 1024,
) -> ProcessOutcome:
    """Run a process with bounded output, EOF-bounded stdin and a wall timeout."""

    if max_stdout_bytes < 0 or max_stderr_bytes < 0:
        raise ValueError("process output limits must be non-negative")

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE if input_data is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=env,
        start_new_session=True,
    )
    stdout = bytearray()
    stderr = bytearray()
    output_limited = threading.Event()

    def read_bounded(stream: object, target: bytearray, limit: int) -> None:
        try:
            while chunk := stream.read(64 * 1024):  # type: ignore[attr-defined]
                remaining = limit - len(target)
                if remaining > 0:
                    target.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    output_limited.set()
        finally:
            stream.close()  # type: ignore[attr-defined]

    def write_input() -> None:
        if process.stdin is None:
            return
        try:
            if input_data is not None:
                process.stdin.write(input_data)
                process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        finally:
            process.stdin.close()

    readers = [
        threading.Thread(
            target=read_bounded,
            args=(process.stdout, stdout, max_stdout_bytes),
            daemon=True,
        ),
        threading.Thread(
            target=read_bounded,
            args=(process.stderr, stderr, max_stderr_bytes),
            daemon=True,
        ),
    ]
    for reader in readers:
        reader.start()
    writer = threading.Thread(target=write_input, daemon=True)
    writer.start()

    deadline = time.monotonic() + timeout_seconds
    timed_out = False
    while process.poll() is None:
        if output_limited.is_set():
            break
        if time.monotonic() >= deadline:
            timed_out = True
            break
        time.sleep(0.01)

    if timed_out or output_limited.is_set():
        _signal_process_group(process, signal.SIGTERM)
        try:
            process.wait(timeout=terminate_grace_seconds)
        except subprocess.TimeoutExpired:
            _signal_process_group(process, signal.SIGKILL)
            process.wait()

    writer.join(timeout=terminate_grace_seconds)
    for reader in readers:
        reader.join(timeout=terminate_grace_seconds)
    if output_limited.is_set():
        return ProcessOutcome(
            125,
            bytes(stdout),
            bytes(stderr),
            output_limited=True,
        )
    if timed_out:
        return ProcessOutcome(124, bytes(stdout), bytes(stderr), timed_out=True)
    return ProcessOutcome(process.returncode, bytes(stdout), bytes(stderr))
