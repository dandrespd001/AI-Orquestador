from __future__ import annotations

import json
import os
import pwd
import sqlite3
import stat
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from ai_orchestrator.models import ensure_private_directory


@dataclass(frozen=True)
class StorageFinding:
    severity: str
    code: str
    path: str
    detail: str


SENSITIVE_ROOTS = (
    Path.home() / ".omniroute",
    Path.home() / ".local" / "share" / "opencode",
    Path.home() / ".codex",
    Path.home() / ".claude",
    Path.home() / ".kimi-code",
    Path.home() / ".minimax_mcp",
    Path.home() / ".local" / "share" / "ai-orchestrator",
    Path.home() / ".config" / "ai-orchestrator",
)

SQLITE_TARGETS = (
    (Path.home() / ".omniroute" / "storage.sqlite", "omniroute"),
    (Path.home() / ".local" / "share" / "opencode" / "opencode.db", "opencode"),
    (Path.home() / ".codex" / "logs_2.sqlite", "codex"),
    (Path.home() / ".codex" / "state_5.sqlite", "codex"),
    (Path.home() / ".minimax_mcp" / "tasks.db", "minimax_bridge"),
)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


def _size_tree(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, directories, files in os.walk(path, followlinks=False):
        directories[:] = [name for name in directories if not (Path(root) / name).is_symlink()]
        for name in files:
            candidate = Path(root) / name
            try:
                metadata = candidate.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISREG(metadata.st_mode):
                total += metadata.st_size
    return total


def _sqlite_info(path: Path) -> dict[str, Any]:
    uri = f"file:{path}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=1.0) as connection:
        values = {
            "journal_mode": connection.execute("PRAGMA journal_mode").fetchone()[0],
            "synchronous": connection.execute("PRAGMA synchronous").fetchone()[0],
            "foreign_keys": connection.execute("PRAGMA foreign_keys").fetchone()[0],
            "auto_vacuum": connection.execute("PRAGMA auto_vacuum").fetchone()[0],
            "secure_delete": connection.execute("PRAGMA secure_delete").fetchone()[0],
            "wal_autocheckpoint": connection.execute("PRAGMA wal_autocheckpoint").fetchone()[0],
            "journal_size_limit": connection.execute("PRAGMA journal_size_limit").fetchone()[0],
            "page_size": connection.execute("PRAGMA page_size").fetchone()[0],
            "page_count": connection.execute("PRAGMA page_count").fetchone()[0],
            "freelist_count": connection.execute("PRAGMA freelist_count").fetchone()[0],
            "quick_check": connection.execute("PRAGMA quick_check").fetchone()[0],
        }
    values["reclaimable_bytes"] = values["page_size"] * values["freelist_count"]
    return values


def _age_counts(path: Path) -> dict[str, int]:
    result = {"files": 0, "older_7d": 0, "older_90d": 0}
    if not path.is_dir():
        return result
    now = time.time()
    for root, _, files in os.walk(path, followlinks=False):
        for name in files:
            candidate = Path(root) / name
            try:
                metadata = candidate.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(metadata.st_mode):
                continue
            result["files"] += 1
            age = now - metadata.st_mtime
            if age > 7 * 86400:
                result["older_7d"] += 1
            if age > 90 * 86400:
                result["older_90d"] += 1
    return result


def audit_storage() -> dict[str, Any]:
    findings: list[StorageFinding] = []
    roots: list[dict[str, Any]] = []
    for root in SENSITIVE_ROOTS:
        if not root.exists():
            continue
        root_mode = _mode(root)
        roots.append({"path": str(root), "mode": f"{root_mode:04o}", "bytes": _size_tree(root)})
        if root_mode & 0o077:
            findings.append(
                StorageFinding("high", "ROOT_TOO_OPEN", str(root), f"mode={root_mode:04o}, expected=0700")
            )

    sensitive_files = (
        Path.home() / ".omniroute" / ".env",
        Path.home() / ".omniroute" / "storage.sqlite",
        Path.home() / ".local" / "share" / "opencode" / "auth.json",
        Path.home() / ".local" / "share" / "opencode" / "opencode.db",
        Path.home() / ".codex" / "auth.json",
        Path.home() / ".config" / "fish" / "config.fish",
        Path.home() / ".kimi-code" / "credentials" / "kimi-code.json",
        Path.home() / ".minimax_mcp" / "tasks.db",
    )
    for path in sensitive_files:
        if not path.exists() or path.is_symlink():
            continue
        file_mode = _mode(path)
        if file_mode & 0o077:
            findings.append(
                StorageFinding("critical", "SENSITIVE_FILE_TOO_OPEN", str(path), f"mode={file_mode:04o}")
            )

    databases: list[dict[str, Any]] = []
    for path, client in SQLITE_TARGETS:
        if not path.is_file() or path.is_symlink():
            continue
        try:
            info = _sqlite_info(path)
            databases.append({"path": str(path), "client": client, **info})
            if info["quick_check"] != "ok":
                findings.append(StorageFinding("critical", "SQLITE_INTEGRITY", str(path), str(info["quick_check"])))
            if info["journal_size_limit"] < 0:
                findings.append(StorageFinding("medium", "UNBOUNDED_JOURNAL", str(path), "journal_size_limit=-1"))
        except sqlite3.Error as exc:
            findings.append(StorageFinding("high", "SQLITE_UNREADABLE", str(path), type(exc).__name__))

    omniroute_env = Path.home() / ".omniroute" / ".env"
    if omniroute_env.is_file():
        names: list[str] = []
        for line in omniroute_env.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                names.append(stripped.split("=", 1)[0].strip())
        if names:
            findings.append(
                StorageFinding(
                    "critical",
                    "PLAINTEXT_SERVICE_CREDENTIALS",
                    str(omniroute_env),
                    "variables=" + ",".join(sorted(names)),
                )
            )

    age = {
        "codex_sessions": _age_counts(Path.home() / ".codex" / "sessions"),
        "claude_projects": _age_counts(Path.home() / ".claude" / "projects"),
        "omniroute_call_logs": _age_counts(Path.home() / ".omniroute" / "call_logs"),
    }
    return {
        "schema_version": 1,
        "ok": not any(item.severity in {"critical", "high"} for item in findings),
        "roots": roots,
        "databases": databases,
        "age": age,
        "findings": [asdict(item) for item in findings],
    }


def _iter_private_tree(root: Path) -> Iterable[Path]:
    yield root
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        safe_directories: list[str] = []
        for name in directories:
            path = Path(current) / name
            if path.is_symlink():
                continue
            safe_directories.append(name)
            yield path
        directories[:] = safe_directories
        for name in files:
            path = Path(current) / name
            if not path.is_symlink():
                yield path


def apply_private_permissions() -> dict[str, int]:
    uid = os.getuid()
    changed = 0
    skipped = 0
    for root in SENSITIVE_ROOTS:
        if not root.exists() or root.is_symlink():
            continue
        for path in _iter_private_tree(root):
            metadata = path.lstat()
            if metadata.st_uid != uid or metadata.st_nlink > 1:
                skipped += 1
                continue
            if stat.S_ISDIR(metadata.st_mode):
                desired = 0o700
            elif stat.S_ISREG(metadata.st_mode):
                desired = 0o700 if metadata.st_mode & 0o111 else 0o600
            else:
                skipped += 1
                continue
            if stat.S_IMODE(metadata.st_mode) != desired:
                os.chmod(path, desired, follow_symlinks=False)
                changed += 1
    return {"changed": changed, "skipped": skipped}


def _client_running(name: str) -> bool:
    own_pid = os.getpid()
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            command = (entry / "comm").read_text(encoding="utf-8").strip().lower()
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").lower()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if name.lower() == command or name.lower() in cmdline:
            return True
    return False


def _private_backup(path: Path) -> Path:
    backup_root = Path.home() / ".local" / "share" / "ai-orchestrator" / "backups" / "sqlite"
    ensure_private_directory(backup_root.parent)
    ensure_private_directory(backup_root)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    destination = backup_root / f"{path.name}.{timestamp}.sqlite"
    if destination.exists():
        raise RuntimeError("backup destination already exists")
    source_uri = f"file:{path}?mode=ro"
    with sqlite3.connect(source_uri, uri=True) as source, sqlite3.connect(destination) as target:
        source.backup(target)
        if target.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError("backup quick_check failed")
    os.chmod(destination, 0o600)
    return destination


def compact_sqlite(path: Path, client: str) -> dict[str, Any]:
    allowed = {candidate.resolve(): owner for candidate, owner in SQLITE_TARGETS if candidate.exists()}
    resolved = path.expanduser().resolve()
    if resolved not in allowed or allowed[resolved] != client:
        raise ValueError("database is not in the audited compaction allowlist")
    if path.is_symlink() or not path.is_file():
        raise ValueError("database must be a regular file")
    if _client_running(client):
        raise RuntimeError(f"client is running: {client}")

    before = _sqlite_info(path)
    backup = _private_backup(path)
    with sqlite3.connect(path, timeout=5.0) as connection:
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("PRAGMA journal_size_limit=67108864")
        connection.execute("PRAGMA incremental_vacuum")
        connection.execute("PRAGMA optimize")
    after = _sqlite_info(path)
    if after["quick_check"] != "ok":
        raise RuntimeError(f"post-maintenance quick_check failed; restore {backup}")
    os.chmod(path, 0o600)
    return {
        "path": str(path),
        "backup": str(backup),
        "reclaimable_before": before["reclaimable_bytes"],
        "reclaimable_after": after["reclaimable_bytes"],
        "quick_check": after["quick_check"],
    }


def storage_receipt(action: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "action": action,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "actor_uid": os.getuid(),
        "actor": pwd.getpwuid(os.getuid()).pw_name,
        "payload": payload,
    }

