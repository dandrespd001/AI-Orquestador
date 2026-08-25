#!/usr/bin/env python3
"""Move the Qwen CLI's plaintext API keys into the encrypted credential store.

The Qwen CLI resolves each provider's key through an ``envKey`` reference, so
the keys only need to exist in the process environment. Keeping them in
``~/.qwen/settings.json`` duplicates secret material on disk for no benefit and
contradicts the store-everything-in-systemd-creds design the rest of the
orchestrator follows.

This script reads the settings file, hands each value straight to
``CredentialStore`` inside this process, and then rewrites the file without the
``env`` block. The secret never reaches a shell, a pipeline, argv or stdout.
Provider entries keep their ``envKey`` references untouched, so ``qwen-safe``
can repopulate the environment at launch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_orchestrator.credentials import CredentialStore
from ai_orchestrator.qwen_safe import credential_name

SETTINGS_PATH = Path.home() / ".qwen" / "settings.json"


def write_private_json(path: Path, payload: dict) -> None:
    """Replace the settings file atomically, never widening its permissions."""

    handle, staging = tempfile.mkstemp(dir=str(path.parent), prefix=".settings.")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(staging, 0o600)
        os.replace(staging, path)
    except BaseException:
        Path(staging).unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="store the credentials and strip the env block; otherwise report only",
    )
    parser.add_argument("--settings", type=Path, default=SETTINGS_PATH)
    args = parser.parse_args()

    if not args.settings.is_file() or args.settings.is_symlink():
        print(json.dumps({"error": "settings file is not a regular file"}))
        return 2

    settings = json.loads(args.settings.read_text(encoding="utf-8"))
    env = settings.get("env") or {}
    if not env:
        print(json.dumps({"status": "ALREADY_MIGRATED", "stored": []}, indent=2))
        return 0

    store = CredentialStore()
    planned = [
        {"variable": variable, "credential": credential_name(variable)}
        for variable in sorted(env)
    ]

    if not args.apply:
        print(
            json.dumps(
                {"status": "DRY_RUN", "would_store": planned, "apply": False}, indent=2
            )
        )
        return 0

    stored = []
    for entry in planned:
        value = env[entry["variable"]]
        receipt = store.store(
            entry["credential"],
            value,
            provider="qwen",
            human_approved=True,
        )
        stored.append(
            {
                "variable": entry["variable"],
                "credential": entry["credential"],
                "secret_returned": receipt.get("secret_returned", False),
            }
        )

    # Only drop the plaintext once every value is safely encrypted on disk.
    settings.pop("env", None)
    write_private_json(args.settings, settings)

    print(json.dumps({"status": "MIGRATED", "stored": stored}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
