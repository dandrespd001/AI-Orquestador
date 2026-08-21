from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import tempfile
import time
from pathlib import Path

HOME = Path.home()
ENV_PATH = HOME / ".omniroute" / ".env"
CREDENTIAL_ROOT = HOME / ".config" / "ai-orchestrator" / "credentials"
BACKUP_ROOT = (
    HOME
    / ".local"
    / "share"
    / "ai-orchestrator"
    / "backups"
    / time.strftime("omniroute-credential-cutover-%Y%m%dT%H%M%SZ", time.gmtime())
)
NAMES = {
    "OMNIROUTE_API_KEY": "omniroute_api_key",
    "STORAGE_ENCRYPTION_KEY": "omniroute_storage_encryption_key",
}


def _private_file(path: Path) -> None:
    metadata = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
    ):
        raise RuntimeError(f"unsafe private file: {path}")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise RuntimeError(f"private file has permissive mode: {path}")


def _environment_values(lines: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, value = line.partition("=")
        name = name.strip()
        if separator and name in NAMES:
            parsed = value.strip()
            if len(parsed) >= 2 and parsed[0] == parsed[-1] and parsed[0] in {"'", '"'}:
                parsed = parsed[1:-1]
            values[name] = parsed
    if set(values) != set(NAMES) or any(not value for value in values.values()):
        raise RuntimeError("plaintext OmniRoute credentials are incomplete")
    return values


def _decrypted(name: str) -> str:
    path = CREDENTIAL_ROOT / f"{name}.cred"
    _private_file(path)
    completed = subprocess.run(
        ["systemd-creds", "decrypt", "--user", f"--name={name}", str(path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=5,
        check=False,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
    )
    if completed.returncode != 0:
        raise RuntimeError(f"encrypted credential cannot be verified: {name}")
    return completed.stdout.decode("utf-8").strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    _private_file(ENV_PATH)
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines(keepends=True)
    plaintext = _environment_values(lines)
    for source_name, credential_name in NAMES.items():
        if _decrypted(credential_name) != plaintext[source_name]:
            raise RuntimeError(f"encrypted credential does not match: {source_name}")
    if not args.apply:
        print(
            json.dumps(
                {"status": "READY", "verified": sorted(NAMES), "values_exposed": False}
            )
        )
        return 0

    BACKUP_ROOT.mkdir(parents=True, mode=0o700)
    os.chmod(BACKUP_ROOT, 0o700)
    backup = BACKUP_ROOT / "omniroute.env"
    shutil.copy2(ENV_PATH, backup)
    os.chmod(backup, 0o600)
    retained: list[str] = []
    for raw in lines:
        probe = raw.strip()
        if probe.startswith("export "):
            probe = probe[7:].lstrip()
        if probe.partition("=")[0].strip() not in NAMES:
            retained.append(raw)
    fd, temporary = tempfile.mkstemp(prefix=".env.", dir=ENV_PATH.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.writelines(retained)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, ENV_PATH)
        os.chmod(ENV_PATH, 0o600)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    print(
        json.dumps(
            {
                "status": "APPLIED",
                "backup": str(backup),
                "removed": sorted(NAMES),
                "values_exposed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
