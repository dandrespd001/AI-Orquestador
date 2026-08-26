"""Parallel-slot update staging with contract tests and rollback metadata."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_orchestrator.config import config_root
from ai_orchestrator.models import ContractError, ensure_private_directory
from ai_orchestrator.telemetry import default_state_root

MAX_UPDATE_ARTIFACT_BYTES = 100 * 1024 * 1024
MAX_UPDATE_STATE_BYTES = 1_000_000


def _safe_segment(value: str, field: str) -> str:
    candidate = value.strip()
    if not candidate or any(
        marker in candidate for marker in ("/", "\\", "..", "\x00")
    ):
        raise ContractError(f"{field} must be a simple path segment")
    return candidate


def _hash_artifact(path: Path) -> str:
    metadata = path.stat()
    if metadata.st_size > MAX_UPDATE_ARTIFACT_BYTES:
        raise ContractError("update artifact exceeds the staging limit")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(128 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ComponentPin:
    name: str
    version: str
    source: str
    sha256: str | None
    compatibility_tests: tuple[str, ...]
    security_tests: tuple[str, ...]
    auto_update: bool = False
    integrity: str = "legacy:sha256"
    install_slot: str = "legacy"
    active: bool = False
    active_executable: str = "artifact"
    rollback_eligible: bool = True
    update_policy: str = "manual_promote"
    update_channel: str = "latest"

    def __post_init__(self) -> None:
        _safe_segment(self.name, "component name")
        _safe_segment(self.version, "component version")
        if not self.integrity.strip():
            raise ContractError("component integrity is required")
        if self.update_policy not in {"manual_promote", "disabled"}:
            raise ContractError("component update policy is invalid")
        _safe_segment(self.update_channel, "component update channel")
        install_slot = Path(self.install_slot)
        executable = Path(self.active_executable)
        if (
            install_slot.is_absolute()
            or executable.is_absolute()
            or ".." in install_slot.parts
            or ".." in executable.parts
            or not self.install_slot
            or not self.active_executable
        ):
            raise ContractError("component slot paths must be relative and bounded")
        if self.sha256 and (
            len(self.sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in self.sha256.lower()
            )
        ):
            raise ContractError("component SHA-256 is invalid")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_component_pins(root: Path | None = None) -> dict[str, ComponentPin]:
    source_path = (root or config_root()) / "components.toml"
    if not source_path.is_file() or source_path.is_symlink():
        raise ContractError(f"missing or unsafe config: {source_path}")
    with source_path.open("rb") as handle:
        payload = tomllib.load(handle)
    if payload.get("schema_version") != 2:
        raise ContractError("components.toml must use schema_version=2")
    rows = payload.get("components")
    if not isinstance(rows, dict):
        raise ContractError("components.toml must define [components.*]")
    result: dict[str, ComponentPin] = {}
    for name, raw in rows.items():
        if not isinstance(raw, dict):
            raise ContractError(f"component {name} must be a table")
        pin = ComponentPin(
            name=str(name),
            version=str(raw.get("version", "")),
            source=str(raw.get("source", "")),
            sha256=str(raw["sha256"]) if raw.get("sha256") else None,
            integrity=str(raw.get("integrity", "")),
            install_slot=str(raw.get("install_slot", "")),
            active=bool(raw.get("active", False)),
            active_executable=str(raw.get("active_executable", "")),
            compatibility_tests=tuple(
                str(value) for value in raw.get("compatibility_tests", ())
            ),
            security_tests=tuple(str(value) for value in raw.get("security_tests", ())),
            auto_update=bool(raw.get("auto_update", False)),
            rollback_eligible=bool(raw.get("rollback_eligible", False)),
            update_policy=str(raw.get("update_policy", "manual_promote")),
            update_channel=str(raw.get("update_channel", "latest")),
        )
        if not pin.version or not pin.source:
            raise ContractError(f"component pin is incomplete: {name}")
        if pin.auto_update:
            raise ContractError(f"automatic component updates are forbidden: {name}")
        result[pin.name] = pin
    return result


class UpdateRegistry:
    def __init__(self, root: Path | None = None, *, create: bool = True) -> None:
        self.root = (root or (default_state_root() / "updates")).expanduser().resolve()
        self.state_path = self.root / "state.json"
        self.slots = self.root / "slots"
        if create:
            ensure_private_directory(self.root)
            ensure_private_directory(self.slots)

    def _load(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"schema_version": 2, "components": {}}
        if (
            self.state_path.is_symlink()
            or self.state_path.stat().st_size > MAX_UPDATE_STATE_BYTES
        ):
            raise ContractError("unsafe update registry")
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") not in {1, 2}:
            raise ContractError("invalid update registry")
        payload["schema_version"] = 2
        return payload

    def _save(self, payload: dict[str, Any]) -> None:
        ensure_private_directory(self.root)
        ensure_private_directory(self.slots)
        payload["schema_version"] = 2
        data = json.dumps(payload, sort_keys=True, indent=2).encode() + b"\n"
        if len(data) > MAX_UPDATE_STATE_BYTES:
            raise ContractError("update registry is too large")
        descriptor, temporary = tempfile.mkstemp(prefix=".state.", dir=self.root)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.state_path)
            os.chmod(self.state_path, 0o600)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    def stage(
        self,
        component: ComponentPin,
        artifact: Path,
        *,
        version: str | None = None,
        expected_sha256: str | None = None,
    ) -> dict[str, Any]:
        artifact_input = artifact.expanduser()
        if artifact_input.is_symlink() or not artifact_input.is_file():
            raise ContractError("update artifact must be a regular file")
        artifact = artifact_input.resolve()
        digest = _hash_artifact(artifact)
        expected = expected_sha256 or component.sha256
        if expected and digest != expected:
            raise ContractError("update artifact hash mismatch")
        target_version = _safe_segment(
            version or component.version, "component version"
        )
        slot = self.slots / component.name / target_version / digest[:16]
        ensure_private_directory(slot)
        target = slot / "artifact"
        if not target.exists():
            with artifact.open("rb") as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=128 * 1024)
                writer.flush()
                os.fsync(writer.fileno())
            os.chmod(target, 0o600)
        elif _hash_artifact(target) != digest:
            raise ContractError("parallel update slot failed its integrity check")
        state = self._load()
        row = state.setdefault("components", {}).setdefault(component.name, {})
        row["staged"] = {
            "version": target_version,
            "sha256": digest,
            "path": str(target),
            "staged_at": datetime.now(UTC).isoformat(),
            "compatibility_passed": False,
            "security_passed": False,
        }
        self._save(state)
        return {"component": component.name, **row["staged"], "status": "staged"}

    @staticmethod
    def _hash_tree(root: Path) -> str:
        root = root.resolve()
        digest = hashlib.sha256()
        for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                target = path.resolve()
                if not target.is_relative_to(root):
                    raise ContractError(
                        f"installed component symlink escapes its slot: {relative}"
                    )
                digest.update(relative.encode("utf-8"))
                digest.update(b"\0symlink\0")
                digest.update(os.readlink(path).encode("utf-8"))
                digest.update(b"\0")
                continue
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            if path.is_file():
                digest.update(_hash_artifact(path).encode("ascii"))
            elif not path.is_dir():
                raise ContractError(f"installed component contains a special file: {relative}")
            digest.update(b"\0")
        return digest.hexdigest()

    @staticmethod
    def _freeze_tree(root: Path) -> None:
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_symlink():
                continue
            if path.is_file():
                executable = bool(path.stat().st_mode & 0o111)
                path.chmod(0o555 if executable else 0o444)
            elif path.is_dir():
                path.chmod(0o555)
        root.chmod(0o555)

    @staticmethod
    def _rebase_internal_symlinks(source: Path, target: Path) -> None:
        """Keep pnpm links internal after moving a tree to a deeper slot path."""

        for source_link in source.rglob("*"):
            if not source_link.is_symlink():
                continue
            relative = source_link.relative_to(source)
            source_target = source_link.resolve()
            if not source_target.is_relative_to(source):
                raise ContractError(
                    f"installed component symlink escapes its tree: {relative}"
                )
            target_link = target / relative
            target_destination = target / source_target.relative_to(source)
            target_link.unlink()
            target_link.symlink_to(
                os.path.relpath(target_destination, start=target_link.parent),
                target_is_directory=source_target.is_dir(),
            )

    def stage_directory(
        self,
        component: ComponentPin,
        installed_tree: Path,
        *,
        expected_lock_sha256: str,
        executable: str | None = None,
        lockfile_name: str | None = None,
        version: str | None = None,
    ) -> dict[str, Any]:
        """Copy a verified installation into a content-addressed immutable slot."""

        source = installed_tree.expanduser()
        if source.is_symlink() or not source.is_dir():
            raise ContractError("installed component must be a regular directory")
        source = source.resolve()
        if lockfile_name is not None:
            lockfile_segment = _safe_segment(lockfile_name, "component lockfile")
            lockfile = source / lockfile_segment
        else:
            candidates = [
                source / "pnpm-lock.yaml",
                source / "package-lock.json",
                source / "npm-shrinkwrap.json",
            ]
            lockfile = next((path for path in candidates if path.is_file()), candidates[0])
        if lockfile.is_symlink() or not lockfile.is_file():
            raise ContractError("installed component requires a pinned package lockfile")
        if _hash_artifact(lockfile) != expected_lock_sha256:
            raise ContractError("component lockfile hash mismatch")
        relative_executable = executable or component.active_executable
        executable_path = source / relative_executable
        if (
            Path(relative_executable).is_absolute()
            or ".." in Path(relative_executable).parts
            or not executable_path.resolve().is_relative_to(source)
            or not executable_path.resolve().is_file()
            or not os.access(executable_path.resolve(), os.X_OK)
        ):
            raise ContractError("component active executable is missing or unsafe")
        source_digest = self._hash_tree(source)
        ensure_private_directory(self.slots)
        target_version = _safe_segment(version or component.version, "component version")
        version_root = self.slots / component.name / target_version
        ensure_private_directory(version_root)
        temporary = Path(tempfile.mkdtemp(prefix=".slot.", dir=version_root))
        try:
            shutil.copytree(source, temporary, dirs_exist_ok=True, symlinks=True)
            self._rebase_internal_symlinks(source, temporary)
            if self._hash_tree(source) != source_digest:
                raise ContractError("installed component changed while staging")
            digest = self._hash_tree(temporary)
            slot = version_root / digest[:16]
            if slot.exists():
                if self._hash_tree(slot) != digest:
                    raise ContractError("immutable component slot was modified")
            else:
                self._freeze_tree(temporary)
                os.replace(temporary, slot)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        state = self._load()
        row = state.setdefault("components", {}).setdefault(component.name, {})
        row["staged"] = {
            "version": target_version,
            "integrity": component.integrity,
            "sha256": digest,
            "lock_sha256": expected_lock_sha256,
            "lockfile": lockfile.relative_to(source).as_posix(),
            "path": str(slot),
            "slot": str(slot),
            "executable": relative_executable,
            "staged_at": datetime.now(UTC).isoformat(),
            "compatibility_passed": False,
            "security_passed": False,
            "evaluation_passed": False,
            "rollback_eligible": component.rollback_eligible,
        }
        self._save(state)
        return {"component": component.name, **row["staged"], "status": "staged"}

    def _verify_slot(self, record: dict[str, Any]) -> None:
        raw_path = record.get("path")
        expected = record.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(expected, str):
            raise ContractError("update slot has no integrity metadata")
        path = Path(raw_path)
        if not path.resolve().is_relative_to(self.slots.resolve()):
            raise ContractError("update slot escapes the registry")
        if path.is_symlink():
            raise ContractError("update slot failed its integrity check")
        if path.is_file():
            verified = _hash_artifact(path) == expected
        elif path.is_dir():
            verified = self._hash_tree(path) == expected
        else:
            verified = False
        if not verified:
            raise ContractError("update slot failed its integrity check")

    def _switch_current(self, component: str, record: dict[str, Any]) -> None:
        slot = Path(str(record.get("slot") or record.get("path"))).resolve()
        if not slot.is_dir() or not slot.is_relative_to(self.slots.resolve()):
            raise ContractError("active component slot is not an installed tree")
        component_root = self.root / _safe_segment(component, "component name")
        ensure_private_directory(component_root)
        temporary = component_root / f".current.{os.getpid()}"
        try:
            temporary.symlink_to(slot, target_is_directory=True)
            os.replace(temporary, component_root / "current")
        finally:
            temporary.unlink(missing_ok=True)

    def record_validation(
        self,
        component: str,
        *,
        compatibility_passed: bool,
        security_passed: bool,
        evaluation_passed: bool,
    ) -> dict[str, Any]:
        state = self._load()
        row = state.get("components", {}).get(component)
        if not isinstance(row, dict) or not isinstance(row.get("staged"), dict):
            raise ContractError("component has no staged update")
        row["staged"].update(
            {
                "compatibility_passed": bool(compatibility_passed),
                "security_passed": bool(security_passed),
                "evaluation_passed": bool(evaluation_passed),
                "validated_at": datetime.now(UTC).isoformat(),
            }
        )
        self._save(state)
        return dict(row["staged"])

    def promote(
        self, component: str, *, human_approved: bool = False
    ) -> dict[str, Any]:
        if not human_approved:
            raise ContractError("component promotion requires a human action")
        state = self._load()
        row = state.get("components", {}).get(component)
        if not isinstance(row, dict) or not isinstance(row.get("staged"), dict):
            raise ContractError("component has no staged update")
        staged = row["staged"]
        self._verify_slot(staged)
        required = (
            staged.get("compatibility_passed"),
            staged.get("security_passed"),
            staged.get("evaluation_passed"),
        )
        if required != (True, True, True):
            raise ContractError("component update has not passed all validation gates")
        previous = row.get("active")
        if Path(str(staged.get("slot") or staged.get("path"))).is_dir():
            self._switch_current(component, staged)
        row["rollback"] = previous
        row["active"] = staged
        row["staged"] = None
        self._save(state)
        return {
            "component": component,
            "status": "active",
            "active": row["active"],
            "rollback_available": previous is not None,
        }

    def rollback(
        self, component: str, *, human_approved: bool = False
    ) -> dict[str, Any]:
        if not human_approved:
            raise ContractError("component rollback requires a human action")
        state = self._load()
        row = state.get("components", {}).get(component)
        if not isinstance(row, dict) or not isinstance(row.get("rollback"), dict):
            raise ContractError("component has no rollback slot")
        self._verify_slot(row["rollback"])
        current = row.get("active")
        if Path(str(row["rollback"].get("slot") or row["rollback"].get("path"))).is_dir():
            self._switch_current(component, row["rollback"])
        row["active"] = row["rollback"]
        row["rollback"] = current
        self._save(state)
        return {
            "component": component,
            "status": "rolled_back",
            "active": row["active"],
        }

    def prune_inactive_slots(
        self, component: str, *, human_approved: bool = False
    ) -> dict[str, Any]:
        """Remove every installed slot except the verified active one."""

        if not human_approved:
            raise ContractError("component pruning requires a human action")
        safe_component = _safe_segment(component, "component name")
        state = self._load()
        row = state.get("components", {}).get(component)
        active = row.get("active") if isinstance(row, dict) else None
        if not isinstance(row, dict) or not isinstance(active, dict):
            raise ContractError("component has no active slot")
        self._verify_slot(active)

        component_slots = self.slots / safe_component
        if component_slots.is_symlink() or not component_slots.is_dir():
            raise ContractError("component slot root is missing or unsafe")
        component_slots = component_slots.resolve()
        raw_active = Path(str(active.get("slot") or active.get("path"))).resolve()
        active_slot = raw_active if raw_active.is_dir() else raw_active.parent
        if not active_slot.is_relative_to(component_slots):
            raise ContractError("active component slot escapes its component root")

        targets: list[Path] = []
        version_roots: list[Path] = []
        for version_root in sorted(component_slots.iterdir()):
            if version_root.is_symlink() or not version_root.is_dir():
                raise ContractError("component version root is unsafe")
            version_roots.append(version_root)
            for slot in sorted(version_root.iterdir()):
                if slot.is_symlink() or not slot.is_dir():
                    raise ContractError("component slot is unsafe")
                if slot.resolve() != active_slot:
                    targets.append(slot)

        # Clear references first so an interrupted deletion can never leave the
        # registry advertising a rollback or staged slot that no longer exists.
        row["rollback"] = None
        row["staged"] = None
        self._save(state)

        removed: list[str] = []
        for target in targets:
            for directory, directory_names, file_names in os.walk(
                target, topdown=False, followlinks=False
            ):
                root = Path(directory)
                for name in file_names:
                    path = root / name
                    if not path.is_symlink():
                        path.chmod(0o600)
                for name in directory_names:
                    path = root / name
                    if not path.is_symlink():
                        path.chmod(0o700)
                root.chmod(0o700)
            shutil.rmtree(target)
            removed.append(str(target))
        for version_root in version_roots:
            try:
                version_root.rmdir()
            except OSError:
                pass
        return {
            "component": component,
            "status": "pruned",
            "active": str(active_slot),
            "removed": removed,
        }

    def inventory(self) -> dict[str, Any]:
        state = self._load()
        return {
            "schema_version": 2,
            "components": state.get("components", {}),
            "automatic_updates": False,
        }

    def resolve_active_executable(self, component: str) -> Path:
        slot, executable = self.resolve_active_slot(component)
        candidate = (slot / executable).resolve()
        if (
            executable.is_absolute()
            or ".." in executable.parts
            or not candidate.is_relative_to(slot)
            or not candidate.is_file()
            or not os.access(candidate, os.X_OK)
        ):
            raise ContractError("active component executable is missing or unsafe")
        return candidate

    def resolve_active_slot(self, component: str) -> tuple[Path, Path]:
        """Return the verified immutable slot and its relative executable."""

        state = self._load()
        row = state.get("components", {}).get(component)
        active = row.get("active") if isinstance(row, dict) else None
        if not isinstance(active, dict):
            raise ContractError("component has no active slot")
        self._verify_slot(active)
        slot = Path(str(active.get("slot") or active.get("path"))).resolve()
        executable = Path(str(active.get("executable", "")))
        if (
            executable.is_absolute()
            or ".." in executable.parts
            or not str(executable)
        ):
            raise ContractError("active component executable is missing or unsafe")
        return slot, executable
