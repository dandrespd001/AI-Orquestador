"""Parallel-slot update staging with contract tests and rollback metadata."""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_orchestrator.config import _read_toml, config_root
from ai_orchestrator.models import ContractError, ensure_private_directory
from ai_orchestrator.state_store import SQLiteStateStore, retry_state_conflicts
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

    def __post_init__(self) -> None:
        _safe_segment(self.name, "component name")
        _safe_segment(self.version, "component version")
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
    payload = _read_toml((root or config_root()) / "components.toml")
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
            compatibility_tests=tuple(
                str(value) for value in raw.get("compatibility_tests", ())
            ),
            security_tests=tuple(str(value) for value in raw.get("security_tests", ())),
            auto_update=bool(raw.get("auto_update", False)),
        )
        if not pin.version or not pin.source:
            raise ContractError(f"component pin is incomplete: {name}")
        if pin.auto_update:
            raise ContractError(f"automatic component updates are forbidden: {name}")
        result[pin.name] = pin
    return result


class UpdateRegistry:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or (default_state_root() / "updates")).expanduser().resolve()
        self.state_path = self.root / "state.sqlite"
        self.slots = self.root / "slots"
        ensure_private_directory(self.root)
        ensure_private_directory(self.slots)
        self._state = SQLiteStateStore(
            self.state_path,
            default_factory=lambda: {"schema_version": 1, "components": {}},
            legacy_path=self.root / "state.json",
        )

    def _load(self) -> dict[str, Any]:
        payload = self._state.load()
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ContractError("invalid update registry")
        return payload

    def _save(self, payload: dict[str, Any]) -> None:
        self._state.save(payload)

    @retry_state_conflicts
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

    def _verify_slot(self, record: dict[str, Any]) -> None:
        raw_path = record.get("path")
        expected = record.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(expected, str):
            raise ContractError("update slot has no integrity metadata")
        path = Path(raw_path)
        if not path.resolve().is_relative_to(self.slots.resolve()):
            raise ContractError("update slot escapes the registry")
        if path.is_symlink() or not path.is_file() or _hash_artifact(path) != expected:
            raise ContractError("update slot failed its integrity check")

    @retry_state_conflicts
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

    @retry_state_conflicts
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

    @retry_state_conflicts
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
        row["active"] = row["rollback"]
        row["rollback"] = current
        self._save(state)
        return {
            "component": component,
            "status": "rolled_back",
            "active": row["active"],
        }

    def inventory(self) -> dict[str, Any]:
        state = self._load()
        return {
            "schema_version": 1,
            "components": state.get("components", {}),
            "automatic_updates": False,
        }
