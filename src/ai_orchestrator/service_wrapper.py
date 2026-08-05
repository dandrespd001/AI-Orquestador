from __future__ import annotations

import os
import stat
from pathlib import Path


def _read_credential(directory: Path, name: str) -> str:
    path = directory / name
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"unsafe systemd credential: {name}")
    if not 8 <= metadata.st_size <= 16_384:
        raise RuntimeError(f"invalid systemd credential size: {name}")
    value = path.read_text(encoding="utf-8").strip()
    if not value or "\x00" in value or "\n" in value:
        raise RuntimeError(f"invalid systemd credential content: {name}")
    return value


def main() -> int:
    raw_directory = os.environ.get("CREDENTIALS_DIRECTORY")
    if not raw_directory:
        raise RuntimeError("CREDENTIALS_DIRECTORY is required")
    directory = Path(raw_directory).resolve()
    api_key = _read_credential(directory, "omniroute_api_key")
    storage_key = _read_credential(directory, "omniroute_storage_encryption_key")

    executable = Path.home() / ".local" / "bin" / "omniroute"
    if executable.is_symlink():
        resolved_executable = executable.resolve()
    else:
        resolved_executable = executable
    if not resolved_executable.is_file() or not os.access(resolved_executable, os.X_OK):
        raise RuntimeError("audited OmniRoute executable is unavailable")

    environment = {
        "HOME": str(Path.home()),
        "USER": os.environ.get("USER", ""),
        "LOGNAME": os.environ.get("LOGNAME", ""),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "PATH": f"{Path.home()}/.local/bin:/usr/local/bin:/usr/bin:/bin",
        "OMNIROUTE_API_KEY": api_key,
        "STORAGE_ENCRYPTION_KEY": storage_key,
        "OMNIROUTE_HOST": "127.0.0.1",
    }
    argv = [
        str(resolved_executable),
        "serve",
        "--port",
        "20128",
        "--no-open",
        "--no-tray",
        "--max-restarts",
        "2",
    ]
    os.execve(resolved_executable, argv, environment)
    return 127

