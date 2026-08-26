"""Credential-in-memory, fail-closed launcher for the OmniRoute CLI."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from ai_orchestrator.credentials import omniroute_runtime_key

EXIT_CONFIG = 78
_DESTRUCTIVE_TOKENS = {"delete", "purge", "remove", "reset", "wipe"}


class SafeOmniRouteError(RuntimeError):
    pass


def _validate_arguments(arguments: Sequence[str]) -> None:
    lowered = tuple(value.casefold() for value in arguments)
    destructive = sorted({value for value in lowered if value in _DESTRUCTIVE_TOKENS})
    if destructive:
        raise SafeOmniRouteError(
            f"destructive OmniRoute command blocked: {', '.join(destructive)}"
        )
    if "auth" in lowered and "export" in lowered:
        raise SafeOmniRouteError("OmniRoute authentication export is blocked")


def _executable() -> Path:
    candidate = Path.home() / ".local" / "bin" / "omniroute"
    resolved = candidate.resolve() if candidate.is_symlink() else candidate
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise SafeOmniRouteError("audited OmniRoute executable is unavailable")
    return resolved


def _environment(key: str) -> dict[str, str]:
    source = os.environ
    home = str(Path.home())
    environment = {
        "HOME": home,
        "USER": source.get("USER", ""),
        "LOGNAME": source.get("LOGNAME", ""),
        "LANG": source.get("LANG", "C.UTF-8"),
        "LC_ALL": source.get("LC_ALL", "C.UTF-8"),
        "PATH": f"{home}/.local/bin:/usr/local/bin:/usr/bin:/bin",
        "OMNIROUTE_API_KEY": key,
        "OMNIROUTE_BASE_URL": "http://127.0.0.1:20128/v1",
        "OPENAI_BASE_URL": "http://127.0.0.1:20128/v1",
        "NO_LOG_API_KEY_IDS": "env-key",
    }
    for name in ("TERM", "COLORTERM", "NO_COLOR"):
        if source.get(name):
            environment[name] = source[name]
    return environment


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    try:
        _validate_arguments(arguments)
        executable = _executable()
        key = omniroute_runtime_key()
        if key is None:
            raise SafeOmniRouteError("encrypted OmniRoute credential is unavailable")
        os.execve(executable, [str(executable), *arguments], _environment(key))
    except SafeOmniRouteError as exc:
        print(
            json.dumps(
                {"ok": False, "error_code": "OMNIROUTE_SAFE_POLICY", "message": str(exc)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return EXIT_CONFIG
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
