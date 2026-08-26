from __future__ import annotations

import os
import pwd
import sqlite3
import stat
import time
from collections.abc import Iterable
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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

OFFICIAL_OMNIROUTE_PROVIDERS = frozenset({"deepseek", "kimi-coding", "moonshot", "minimax"})
OMNIROUTE_PRIVATE_LOG_SETTINGS = {
    ("databaseSettings", "detailedLogsEnabled"): "false",
    ("databaseSettings", "callLogPipelineEnabled"): "false",
    ("settings", "call_log_pipeline_enabled"): "0",
}


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
    with closing(sqlite3.connect(uri, uri=True, timeout=1.0)) as connection:
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

    provider_inventory: list[dict[str, Any]] = []
    content_inventory: dict[str, int] = {}
    content_logging_disabled = False
    omniroute_db = Path.home() / ".omniroute" / "storage.sqlite"
    if omniroute_db.is_file() and not omniroute_db.is_symlink():
        try:
            uri = f"file:{omniroute_db}?mode=ro"
            with closing(sqlite3.connect(uri, uri=True, timeout=1.0)) as connection:
                provider_inventory = [
                    {"provider": str(row[0]), "active": bool(row[1]), "connections": int(row[2])}
                    for row in connection.execute(
                        "SELECT provider, is_active, COUNT(*) FROM provider_connections "
                        "GROUP BY provider, is_active ORDER BY provider, is_active"
                    )
                ]
                call_row = connection.execute(
                    "SELECT COUNT(*), COALESCE(SUM(has_request_body), 0), "
                    "COALESCE(SUM(has_response_body), 0) FROM call_logs"
                ).fetchone()
                detail_row = connection.execute(
                    "SELECT COUNT(*), COALESCE(SUM(CASE WHEN client_request IS NOT NULL "
                    "OR translated_request IS NOT NULL OR provider_response IS NOT NULL "
                    "OR client_response IS NOT NULL THEN 1 ELSE 0 END), 0) "
                    "FROM request_detail_logs"
                ).fetchone()
                observed_logging = {}
                for namespace, key in OMNIROUTE_PRIVATE_LOG_SETTINGS:
                    setting = connection.execute(
                        "SELECT value FROM key_value WHERE namespace=? AND key=?",
                        (namespace, key),
                    ).fetchone()
                    observed_logging[f"{namespace}.{key}"] = (
                        str(setting[0]) if setting else None
                    )
                desired_logging = {
                    f"{namespace}.{key}": value
                    for (namespace, key), value in OMNIROUTE_PRIVATE_LOG_SETTINGS.items()
                }
                content_logging_disabled = observed_logging == desired_logging
            content_inventory = {
                "call_logs": int(call_row[0]),
                "request_body_references": int(call_row[1]),
                "response_body_references": int(call_row[2]),
                "detail_logs": int(detail_row[0]),
                "detail_logs_with_content": int(detail_row[1]),
            }
            non_official = sorted(
                row["provider"]
                for row in provider_inventory
                if row["active"] and row["provider"] not in OFFICIAL_OMNIROUTE_PROVIDERS
            )
            if non_official:
                findings.append(
                    StorageFinding(
                        "info",
                        "THIRD_PARTY_PROVIDER_INVENTORY",
                        str(omniroute_db),
                        "trust_tiers_enforced_by_orchestrator;providers="
                        + ",".join(non_official),
                    )
                )
            if content_inventory["request_body_references"] or content_inventory["response_body_references"]:
                findings.append(
                    StorageFinding(
                        "info" if content_logging_disabled else "high",
                        "MODEL_CONTENT_RETENTION",
                        str(omniroute_db),
                        "historical=true;future_logging_disabled="
                        + str(content_logging_disabled).lower()
                        + ";request_refs="
                        + str(content_inventory["request_body_references"])
                        + ",response_refs=" + str(content_inventory["response_body_references"]),
                    )
                )
        except sqlite3.Error as exc:
            findings.append(StorageFinding("high", "OMNIROUTE_POLICY_UNREADABLE", str(omniroute_db), type(exc).__name__))

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
        "omniroute_providers": provider_inventory,
        "omniroute_content": content_inventory,
        "omniroute_content_logging_disabled": content_logging_disabled,
        "age": age,
        "findings": [asdict(item) for item in findings],
    }


def enforce_official_omniroute_providers(*, apply: bool = False) -> dict[str, Any]:
    """Disable, but never delete, non-official provider connections."""

    path = Path.home() / ".omniroute" / "storage.sqlite"
    if path.is_symlink() or not path.is_file():
        raise ValueError("OmniRoute database must be a regular file")
    placeholders = ",".join("?" for _ in OFFICIAL_OMNIROUTE_PROVIDERS)
    query = (
        "SELECT id, provider FROM provider_connections WHERE is_active=1 "
        f"AND provider NOT IN ({placeholders}) ORDER BY provider, id"
    )
    with closing(
        sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1.0)
    ) as connection:
        rows = connection.execute(query, tuple(sorted(OFFICIAL_OMNIROUTE_PROVIDERS))).fetchall()
    providers = sorted({str(row[1]) for row in rows})
    result: dict[str, Any] = {
        "dry_run": not apply,
        "connections": len(rows),
        "providers": providers,
        "allowed_providers": sorted(OFFICIAL_OMNIROUTE_PROVIDERS),
    }
    if not apply or not rows:
        return result
    if _client_running("omniroute"):
        raise RuntimeError("client is running: omniroute")
    backup = _private_backup(path)
    with closing(sqlite3.connect(path, timeout=5.0)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.executemany(
            "UPDATE provider_connections SET is_active=0, updated_at=datetime('now') "
            "WHERE id=? AND provider=? AND is_active=1",
            rows,
        )
        connection.commit()
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError(f"post-policy quick_check failed; restore {backup}")
    result.update({"backup": str(backup), "applied": len(rows)})
    return result


def disable_omniroute_content_logging(
    *, apply: bool = False, path: Path | None = None
) -> dict[str, Any]:
    """Disable future body-bearing call logs without deleting historical evidence."""

    database = path or (Path.home() / ".omniroute" / "storage.sqlite")
    if database.is_symlink() or not database.is_file():
        raise ValueError("OmniRoute database must be a regular file")

    def read_settings(connection: sqlite3.Connection) -> dict[str, str | None]:
        result: dict[str, str | None] = {}
        for namespace, key in OMNIROUTE_PRIVATE_LOG_SETTINGS:
            row = connection.execute(
                "SELECT value FROM key_value WHERE namespace=? AND key=?", (namespace, key)
            ).fetchone()
            result[f"{namespace}.{key}"] = str(row[0]) if row else None
        return result

    with closing(
        sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=2.0)
    ) as connection:
        before = read_settings(connection)
    desired = {
        f"{namespace}.{key}": value
        for (namespace, key), value in OMNIROUTE_PRIVATE_LOG_SETTINGS.items()
    }
    result: dict[str, Any] = {"dry_run": not apply, "before": before, "desired": desired}
    if not apply or before == desired:
        return result

    backup = _private_backup(database)
    with closing(sqlite3.connect(database, timeout=5.0)) as connection, connection:
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("BEGIN IMMEDIATE")
        for (namespace, key), value in OMNIROUTE_PRIVATE_LOG_SETTINGS.items():
            connection.execute(
                "INSERT INTO key_value(namespace,key,value) VALUES(?,?,?) "
                "ON CONFLICT(namespace,key) DO UPDATE SET value=excluded.value",
                (namespace, key, value),
            )
        connection.commit()
        after = read_settings(connection)
        integrity = connection.execute("PRAGMA quick_check").fetchone()[0]
    if after != desired or integrity != "ok":
        raise RuntimeError(f"post-logging-policy verification failed; restore {backup}")
    result.update({"backup": str(backup), "after": after, "quick_check": integrity})
    return result


def bound_sqlite_journals(*, apply: bool = False) -> dict[str, Any]:
    """Set the 64 MiB journal bound only while each owning client is stopped."""

    rows: list[dict[str, Any]] = []
    for path, client in SQLITE_TARGETS:
        if path.is_symlink() or not path.is_file():
            continue
        before = _sqlite_info(path)
        row: dict[str, Any] = {
            "path": str(path),
            "client": client,
            "before": before["journal_size_limit"],
            "status": "BOUNDED" if before["journal_size_limit"] == 67_108_864 else "WOULD_BOUND",
        }
        if apply and before["journal_size_limit"] != 67_108_864:
            try:
                running = _client_running(client)
            except RuntimeError:
                running = True
                row["status"] = "SKIPPED_PROCESS_VISIBILITY_RESTRICTED"
            if running:
                if row["status"] != "SKIPPED_PROCESS_VISIBILITY_RESTRICTED":
                    row["status"] = "SKIPPED_CLIENT_RUNNING"
            else:
                backup = _private_backup(path)
                with closing(sqlite3.connect(path, timeout=5.0)) as connection, connection:
                    connection.execute("PRAGMA journal_size_limit=67108864")
                after = _sqlite_info(path)
                row.update({"after": after["journal_size_limit"], "backup": str(backup), "status": "APPLIED"})
        rows.append(row)
    return {"dry_run": not apply, "databases": rows}


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
    try:
        init_comm = Path("/proc/1/comm").read_text(encoding="utf-8").strip().lower()
    except (FileNotFoundError, PermissionError) as exc:
        raise RuntimeError("process visibility cannot be established") from exc
    if init_comm not in {"systemd", "init"}:
        raise RuntimeError("process visibility is restricted; refusing client-state inference")
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
    with (
        closing(sqlite3.connect(source_uri, uri=True)) as source,
        closing(sqlite3.connect(destination)) as target,
        target,
    ):
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
    with closing(sqlite3.connect(path, timeout=5.0)) as connection, connection:
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
