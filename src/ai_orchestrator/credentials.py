"""Read exact runtime credentials without copying them into client config."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path
from typing import Iterable

from ai_orchestrator.config import config_root


MAX_CREDENTIAL_BYTES = 16_384


def _private_regular_file(path: Path, *, require_owner: bool = True) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        return False
    if require_owner and metadata.st_uid != os.getuid():
        return False
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        return False
    return metadata.st_size <= MAX_CREDENTIAL_BYTES


def _valid_value(value: str | None) -> str | None:
    if value is None:
        return None
    candidate = value.strip("\r\n")
    if 8 <= len(candidate) <= MAX_CREDENTIAL_BYTES and "\x00" not in candidate:
        return candidate
    return None


def resolve_credential(
    name: str,
    *,
    env_names: Iterable[str] = (),
    environment: dict[str, str] | None = None,
    root: Path | None = None,
    timeout_seconds: float = 3.0,
) -> str | None:
    """Resolve one credential without a shell or plaintext persistence.

    Precedence is systemd's runtime credential directory, explicit environment
    variables, then a user-bound ``systemd-creds`` encrypted file.
    """

    if not name or any(value in name for value in ("/", "\\", "..")):
        raise ValueError("credential name must be a simple name")
    source = os.environ if environment is None else environment
    credential_directory = source.get("CREDENTIALS_DIRECTORY")
    if credential_directory:
        runtime_path = Path(credential_directory) / name
        if _private_regular_file(runtime_path, require_owner=False):
            value = _valid_value(runtime_path.read_text(encoding="utf-8"))
            if value:
                return value
    for env_name in env_names:
        value = _valid_value(source.get(env_name))
        if value:
            return value

    encrypted = (root or config_root()) / "credentials" / f"{name}.cred"
    if not _private_regular_file(encrypted):
        return None
    try:
        completed = subprocess.run(
            [
                "systemd-creds",
                "decrypt",
                "--user",
                f"--name={name}",
                str(encrypted),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout_seconds,
            check=False,
            env={"PATH": source.get("PATH", "/usr/bin:/bin")},
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if completed.returncode != 0 or len(completed.stdout) > MAX_CREDENTIAL_BYTES:
        return None
    try:
        return _valid_value(completed.stdout.decode("utf-8"))
    except UnicodeDecodeError:
        return None


def minimax_runtime_key(environment: dict[str, str] | None = None) -> str | None:
    return resolve_credential(
        "minimax-api-key",
        env_names=("MINIMAX_API_KEY",),
        environment=environment,
    )


def private_dotenv_value(path: Path, key: str) -> str | None:
    """Return one literal dotenv value from a private, regular file."""

    if not _private_regular_file(path):
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, raw_value = line.partition("=")
        if separator and name.strip() == key:
            value = raw_value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            if 8 <= len(value) <= 16_384 and "\x00" not in value:
                return value
            return None
    return None


def omniroute_runtime_key(environment: dict[str, str] | None = None) -> str | None:
    source = os.environ if environment is None else environment
    resolved = resolve_credential(
        "omniroute_api_key",
        env_names=("OMNIROUTE_API_KEY",),
        environment=dict(source),
    )
    if resolved:
        return resolved
    home = source.get("HOME")
    if not home:
        return None
    return private_dotenv_value(Path(home) / ".omniroute" / ".env", "OMNIROUTE_API_KEY")
