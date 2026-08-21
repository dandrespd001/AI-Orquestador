"""Atomic installation and drift checks for the public orchestrator config."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_orchestrator.capabilities import load_capability_packages
from ai_orchestrator.config import load_gates, load_models, load_profile, load_roles
from ai_orchestrator.extensions import load_extension_policy
from ai_orchestrator.models import ContractError, ensure_private_directory
from ai_orchestrator.providers import load_provider_specs, validate_provider_catalog
from ai_orchestrator.updates import load_component_pins

MANIFEST_NAME = ".generation.json"


def canonical_config_root() -> Path:
    override = os.environ.get("AI_ORCHESTRATOR_SOURCE_CONFIG")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "config"


def _public_files(root: Path) -> tuple[Path, ...]:
    candidates = [
        root / "models.toml",
        root / "roles.toml",
        root / "gates.toml",
        root / "providers.toml",
        root / "capability-packages.toml",
        root / "components.toml",
        root / "extension-policy.toml",
    ]
    profiles = root / "profiles"
    if profiles.is_dir() and not profiles.is_symlink():
        candidates.extend(sorted(profiles.glob("*.toml")))
    files: list[Path] = []
    for path in candidates:
        if (
            not path.is_file()
            or path.is_symlink()
            or not path.resolve().is_relative_to(root)
        ):
            raise ContractError(f"missing or unsafe canonical config: {path}")
        files.append(path)
    return tuple(files)


def _hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in _public_files(root)
    }


def _validate_source(root: Path) -> None:
    load_models(root)
    load_roles(root)
    load_gates(root)
    load_provider_specs(root)
    provider_status = validate_provider_catalog(root)
    if not provider_status["ok"]:
        raise ContractError(
            "invalid provider catalog: " + "; ".join(provider_status["errors"])
        )
    load_capability_packages(root)
    load_component_pins(root)
    load_extension_policy(root)
    profiles = root / "profiles"
    for path in profiles.glob("*.toml"):
        load_profile(path.stem, root)


def config_sync_status(source: Path, target: Path) -> dict[str, Any]:
    source = source.expanduser().resolve()
    target = target.expanduser().resolve()
    _validate_source(source)
    expected = _hashes(source)
    actual: dict[str, str | None] = {}
    for relative in expected:
        destination = target / relative
        actual[relative] = (
            hashlib.sha256(destination.read_bytes()).hexdigest()
            if destination.is_file() and not destination.is_symlink()
            else None
        )
    drift = sorted(name for name, digest in expected.items() if actual[name] != digest)
    return {
        "schema_version": 1,
        "source": str(source),
        "target": str(target),
        "in_sync": not drift,
        "drift": drift,
        "expected_hashes": expected,
    }


def _atomic_write(path: Path, data: bytes, mode: int = 0o600) -> None:
    ensure_private_directory(path.parent)
    if path.exists() and path.is_symlink():
        raise ContractError(f"refusing to replace symlink: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb", closefd=True) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def apply_config_sync(source: Path, target: Path) -> dict[str, Any]:
    source = source.expanduser().resolve()
    target = target.expanduser().resolve()
    before = config_sync_status(source, target)
    ensure_private_directory(target)
    backup_root: Path | None = None
    if before["drift"]:
        generation = (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        )
        backup_root = target / "backups" / generation
        ensure_private_directory(backup_root)
        for relative in before["drift"]:
            current = target / str(relative)
            if current.is_file() and not current.is_symlink():
                backup = backup_root / str(relative)
                ensure_private_directory(backup.parent)
                shutil.copyfile(current, backup)
                os.chmod(backup, 0o600)

    for source_path in _public_files(source):
        relative = source_path.relative_to(source)
        _atomic_write(target / relative, source_path.read_bytes())

    hashes = _hashes(source)
    manifest = {
        "schema_version": 1,
        "generation_id": str(uuid.uuid4()),
        "installed_at": datetime.now(UTC).isoformat(),
        "source": str(source),
        "hashes": hashes,
    }
    _atomic_write(
        target / MANIFEST_NAME,
        json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8") + b"\n",
    )
    after = config_sync_status(source, target)
    after["applied"] = True
    after["backup"] = str(backup_root) if backup_root else None
    return after
