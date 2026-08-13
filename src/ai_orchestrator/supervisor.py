"""Bounded subprocess execution with process-group cleanup."""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProcessOutcome:
    returncode: int
    stdout: bytes
    stderr: bytes
    timed_out: bool = False


def _signal_process_group(process: subprocess.Popen[bytes], sig: signal.Signals) -> None:
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
) -> ProcessOutcome:
    """Run ``command`` with an EOF-bounded stdin and a hard wall timeout."""

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE if input_data is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(input=input_data, timeout=timeout_seconds)
        return ProcessOutcome(process.returncode, stdout, stderr)
    except subprocess.TimeoutExpired:
        _signal_process_group(process, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=terminate_grace_seconds)
        except subprocess.TimeoutExpired:
            _signal_process_group(process, signal.SIGKILL)
            stdout, stderr = process.communicate()
        return ProcessOutcome(124, stdout, stderr, timed_out=True)
