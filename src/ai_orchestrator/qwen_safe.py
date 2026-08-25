"""Credential-in-memory, fail-closed launcher for the Qwen CLI.

The Qwen CLI resolves each configured provider's key through an ``envKey``
reference in ``~/.qwen/settings.json``. Historically those values were kept in
the settings file's ``env`` block, which duplicated secret material on disk.
This launcher restores the same environment at run time from the encrypted
credential store instead, so the settings file can stay free of secrets while
interactive use keeps working exactly as before.

The variable names are read back from the settings file rather than hardcoded,
because the OmniRoute provider's key name embeds a per-installation hash.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from ai_orchestrator.credentials import resolve_credential

EXIT_CONFIG = 78

SETTINGS_PATH = Path.home() / ".qwen" / "settings.json"

# Stable names for the keys we already know about. Anything else is derived
# from the variable name so a newly added provider still resolves.
KNOWN_CREDENTIAL_NAMES = {
    "DEEPSEEK_API_KEY": "qwen-deepseek-api-key",
    "MINIMAX_API_KEY": "qwen-minimax-api-key",
    "LOCAL_LLM_API_KEY": "qwen-local-llm-api-key",
    # Reuse the credential the OmniRoute service unit already installs instead
    # of storing a second copy of the same key under a Qwen-specific name.
    "OMNIROUTE_API_KEY": "omniroute_api_key",
}

_PASSTHROUGH = (
    "TERM",
    "COLORTERM",
    "NO_COLOR",
    "EDITOR",
    "VISUAL",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_CACHE_HOME",
    "SSH_AUTH_SOCK",
)


class SafeQwenError(RuntimeError):
    pass


def credential_name(variable: str) -> str:
    """Map an environment variable name to its credential-store name."""

    if variable in KNOWN_CREDENTIAL_NAMES:
        return KNOWN_CREDENTIAL_NAMES[variable]
    if variable.startswith("QWEN_CUSTOM_API_KEY_"):
        return "qwen-omniroute-custom-api-key"
    return "qwen-" + variable.lower().replace("_", "-")


def referenced_env_keys(settings_path: Path = SETTINGS_PATH) -> tuple[str, ...]:
    """Collect every ``envKey`` the settings file points a provider at.

    Any leftover ``env`` block is included too, so a partially migrated
    installation still launches with a complete environment.
    """

    if not settings_path.is_file() or settings_path.is_symlink():
        raise SafeQwenError("Qwen settings file is unavailable")
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SafeQwenError(f"Qwen settings file is not valid JSON: {exc}") from exc

    names: set[str] = set(settings.get("env") or {})
    providers = settings.get("modelProviders") or {}
    for entries in providers.values():
        for entry in entries or ():
            if isinstance(entry, dict):
                key = entry.get("envKey")
                if isinstance(key, str) and key:
                    names.add(key)
    return tuple(sorted(names))


def _executable() -> Path:
    candidate = Path.home() / ".local" / "bin" / "qwen"
    resolved = candidate.resolve() if candidate.is_symlink() else candidate
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise SafeQwenError("audited Qwen executable is unavailable")
    return resolved


def _environment(secrets: dict[str, str]) -> dict[str, str]:
    source = os.environ
    home = str(Path.home())
    environment = {
        "HOME": home,
        "USER": source.get("USER", ""),
        "LOGNAME": source.get("LOGNAME", ""),
        "LANG": source.get("LANG", "C.UTF-8"),
        "PATH": f"{home}/.local/bin:/usr/local/bin:/usr/bin:/bin",
        # Keep prompts and responses out of the OmniRoute call log, matching
        # the guarantee the orchestrator's own adapters rely on.
        "NO_LOG_API_KEY_IDS": "env-key",
    }
    for name in _PASSTHROUGH:
        if source.get(name):
            environment[name] = source[name]
    environment.update(secrets)
    return environment


def resolve_environment(settings_path: Path = SETTINGS_PATH) -> dict[str, str]:
    """Resolve every referenced key, failing closed if any is missing."""

    secrets: dict[str, str] = {}
    missing: list[str] = []
    for variable in referenced_env_keys(settings_path):
        value = resolve_credential(credential_name(variable), env_names=(variable,))
        if value:
            secrets[variable] = value
        else:
            missing.append(variable)
    if missing:
        raise SafeQwenError(
            "encrypted Qwen credentials are unavailable: " + ", ".join(missing)
        )
    return secrets


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    try:
        executable = _executable()
        secrets = resolve_environment()
        os.execve(executable, [str(executable), *arguments], _environment(secrets))
    except SafeQwenError as exc:
        print(
            json.dumps(
                {"ok": False, "error_code": "QWEN_SAFE_POLICY", "message": str(exc)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return EXIT_CONFIG
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
