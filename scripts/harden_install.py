from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import sqlite3
import stat
import subprocess
import time
from pathlib import Path


HOME = Path.home()
SOURCE = Path(__file__).resolve().parents[1]
STATE_ROOT = HOME / ".local" / "share" / "ai-orchestrator"
CONFIG_ROOT = HOME / ".config" / "ai-orchestrator"
BACKUP_ROOT = STATE_ROOT / "backups" / time.strftime("pre-hardening-%Y%m%dT%H%M%SZ", time.gmtime())
OFFICIAL_OMNIROUTE_PROVIDERS = {"deepseek", "kimi-coding", "moonshot", "minimax"}


def private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)


def backup(path: Path) -> None:
    if not path.exists() or path.is_symlink():
        return
    destination = BACKUP_ROOT / str(path).lstrip("/")
    private_dir(destination.parent)
    shutil.copy2(path, destination)
    destination.chmod(0o600)


def parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        values[name.strip()] = value.strip().strip('"').strip("'")
    required = {"OMNIROUTE_API_KEY", "STORAGE_ENCRYPTION_KEY"}
    if not required <= values.keys() or any(not values[name] for name in required):
        raise RuntimeError("OmniRoute environment does not contain both required credentials")
    return {name: values[name] for name in required}


def encrypt_credentials(values: dict[str, str]) -> list[str]:
    credentials = CONFIG_ROOT / "credentials"
    private_dir(credentials)
    names = {
        "OMNIROUTE_API_KEY": "omniroute_api_key",
        "STORAGE_ENCRYPTION_KEY": "omniroute_storage_encryption_key",
    }
    installed: list[str] = []
    for source_name, credential_name in names.items():
        destination = credentials / f"{credential_name}.cred"
        subprocess.run(
            [
                "systemd-creds", "encrypt", "--user", "--with-key=host",
                f"--name={credential_name}", "-", str(destination),
            ],
            input=values[source_name].encode("utf-8"),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        destination.chmod(0o600)
        installed.append(credential_name)
    return installed


def harden_omniroute_database(path: Path, *, apply_provider_policy: bool) -> dict[str, object]:
    backup(path)
    backup(path.with_name(path.name + "-wal"))
    backup(path.with_name(path.name + "-shm"))
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("BEGIN IMMEDIATE")
        for namespace, key, value in (
            ("databaseSettings", "detailedLogsEnabled", "false"),
            ("databaseSettings", "rawDataRetentionDays", "1"),
            ("databaseSettings", "callLogPipelineEnabled", "false"),
            ("settings", "call_log_pipeline_enabled", "false"),
        ):
            connection.execute(
                "INSERT INTO key_value(namespace,key,value) VALUES(?,?,?) "
                "ON CONFLICT(namespace,key) DO UPDATE SET value=excluded.value",
                (namespace, key, value),
            )
        disabled = 0
        if apply_provider_policy:
            placeholders = ",".join("?" for _ in OFFICIAL_OMNIROUTE_PROVIDERS)
            cursor = connection.execute(
                f"UPDATE provider_connections SET is_active=0 WHERE provider NOT IN ({placeholders}) AND is_active<>0",
                tuple(sorted(OFFICIAL_OMNIROUTE_PROVIDERS)),
            )
            disabled = cursor.rowcount
        connection.commit()
        integrity = connection.execute("PRAGMA quick_check").fetchone()[0]
    if integrity != "ok":
        raise RuntimeError("OmniRoute database failed integrity check after migration")
    return {"disabled_non_official_connections": disabled, "quick_check": integrity}


def install_files() -> None:
    private_dir(CONFIG_ROOT)
    for source in (SOURCE / "config").rglob("*"):
        relative = source.relative_to(SOURCE / "config")
        target = CONFIG_ROOT / relative
        if source.is_dir():
            private_dir(target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            target.chmod(0o600)
    bin_dir = HOME / ".local" / "bin"
    private_dir(bin_dir)
    links = {
        "ai-orchestrator": SOURCE / ".venv" / "bin" / "ai-orchestrator",
        "ai-orchestrator-omniroute-wrapper": SOURCE / ".venv" / "bin" / "ai-orchestrator-omniroute-wrapper",
        "omniroute-safe": SOURCE / ".venv" / "bin" / "omniroute-safe",
    }
    for name, source in links.items():
        target = bin_dir / name
        if target.exists() or target.is_symlink():
            if not target.is_symlink() or target.resolve() != source.resolve():
                backup(target)
                target.unlink()
            else:
                continue
        target.symlink_to(source)
    unit_dir = HOME / ".config" / "systemd" / "user"
    private_dir(unit_dir)
    for source in (SOURCE / "systemd").glob("*.service"):
        target = unit_dir / source.name
        if target.exists() and target.samefile(source):
            continue
        shutil.copy2(source, target)
        target.chmod(0o600)
    for source in (SOURCE / "systemd").glob("*.timer"):
        target = unit_dir / source.name
        if target.exists() and target.samefile(source):
            continue
        shutil.copy2(source, target)
        target.chmod(0o600)


def harden_modes(roots: tuple[Path, ...] | None = None) -> dict[str, int]:
    selected_roots = roots or (
        HOME / ".omniroute", HOME / ".local" / "share" / "opencode",
        HOME / ".codex", HOME / ".claude", HOME / ".kimi-code",
        HOME / ".minimax_mcp", STATE_ROOT, CONFIG_ROOT,
    )
    changed = 0
    skipped_read_only = 0
    for root in selected_roots:
        if not root.exists() or root.is_symlink():
            continue
        for current, directories, files in os.walk(root, followlinks=False):
            paths = [Path(current)] + [Path(current) / name for name in directories + files]
            for path in paths:
                if path.is_symlink():
                    continue
                metadata = path.lstat()
                if metadata.st_uid != os.getuid() or metadata.st_nlink > 1:
                    continue
                desired = 0o700 if stat.S_ISDIR(metadata.st_mode) or metadata.st_mode & 0o111 else 0o600
                if stat.S_IMODE(metadata.st_mode) != desired:
                    try:
                        path.chmod(desired)
                    except OSError as exc:
                        if exc.errno not in {errno.EACCES, errno.EPERM, errno.EROFS}:
                            raise
                        # Files owned by a managed runtime can be intentionally
                        # immutable/read-only. They are outside this installer's
                        # authority and must not block the OmniRoute cutover.
                        skipped_read_only += 1
                    else:
                        changed += 1
    return {"changed": changed, "skipped_read_only": skipped_read_only}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--apply-provider-policy", action="store_true")
    parser.add_argument(
        "--install-only",
        action="store_true",
        help="repair config, launchers and user units without reading or rotating credentials",
    )
    args = parser.parse_args(argv)
    if args.install_only and (args.apply or args.apply_provider_policy):
        raise RuntimeError("--install-only cannot be combined with credential migration flags")
    if args.apply_provider_policy and not args.apply:
        raise RuntimeError("--apply-provider-policy requires --apply")
    if args.install_only:
        private_dir(STATE_ROOT)
        install_files()
        modes = harden_modes()
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "INSTALL_REPAIRED",
                    "credentials_rotated": False,
                    "plaintext_credentials_read": False,
                    "permissions": modes,
                },
                sort_keys=True,
            )
        )
        return 0
    environment_path = HOME / ".omniroute" / ".env"
    values = parse_env(environment_path)
    if not args.apply:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "DRY_RUN",
                    "credential_names": sorted(values),
                    "provider_policy_requested": args.apply_provider_policy,
                    "plaintext_values_exposed": False,
                },
                sort_keys=True,
            )
        )
        return 0
    private_dir(STATE_ROOT)
    private_dir(BACKUP_ROOT)
    init_comm = Path("/proc/1/comm").read_text(encoding="utf-8").strip().lower()
    if init_comm not in {"systemd", "init"}:
        raise RuntimeError("process visibility is restricted; run hardening from the host user session")
    backup(environment_path)
    install_files()
    credentials = encrypt_credentials(values)
    database = harden_omniroute_database(
        HOME / ".omniroute" / "storage.sqlite",
        apply_provider_policy=args.apply_provider_policy,
    )
    modes = harden_modes()
    receipt = {
        "schema_version": 1,
        "status": "STAGED_FOR_SERVICE_CUTOVER",
        "backup_root": str(BACKUP_ROOT),
        "credential_names": credentials,
        "database": database,
        "permissions": modes,
        "plaintext_environment_retained_until_service_verification": True,
    }
    receipt_path = STATE_ROOT / "hardening-receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2), encoding="utf-8")
    receipt_path.chmod(0o600)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
