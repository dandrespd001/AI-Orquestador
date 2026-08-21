"""Quarantine-first lifecycle for plugins, skills and MCP servers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import urllib.parse
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_orchestrator.config import _read_toml, config_root
from ai_orchestrator.models import (
    EXTENSION_TRUST_LEVELS,
    ContractError,
    ExtensionManifest,
    ensure_private_directory,
)
from ai_orchestrator.state_store import SQLiteStateStore, retry_state_conflicts
from ai_orchestrator.telemetry import default_state_root

MAX_EXTENSION_FILES = 2_000
MAX_EXTENSION_BYTES = 50 * 1024 * 1024
MAX_INSPECT_TEXT_BYTES = 256 * 1024
MAX_STATE_BYTES = 2 * 1024 * 1024

KNOWN_PERMISSIONS = {
    "filesystem_read",
    "filesystem_write",
    "network",
    "process_spawn",
    "shell",
    "host_access",
    "task_credentials",
    "secrets",
}
ADMIN_PROFILES = {"admin", "administrative", "harness-admin"}
COMMUNITY_TRUST = {"community_verified", "community_experimental"}
INSTALL_SCRIPT_NAMES = {"preinstall", "install", "postinstall", "prepare"}
SHELL_PATTERNS = (
    re.compile(r"\bos\.system\s*\("),
    re.compile(r"\bsubprocess\.(?:run|call|Popen)\s*\("),
    re.compile(r"\bchild_process\b"),
    re.compile(r"\bexecSync\s*\("),
    re.compile(r"\bspawnSync\s*\("),
)
SECRET_PATTERNS = (
    re.compile(r"(?:API_KEY|AUTH_TOKEN|PASSWORD|SECRET)", re.IGNORECASE),
    re.compile(r"(?:os\.environ|os\.getenv|process\.env|CREDENTIALS_DIRECTORY)"),
    re.compile(
        r"(?:\.codex/auth\.json|\.kimi-code/credentials|/credentials/)", re.IGNORECASE
    ),
)
NETWORK_PATTERNS = (
    re.compile(r"https?://", re.IGNORECASE),
    re.compile(r"\b(?:requests|urllib|fetch|axios|socket)\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class InspectionFinding:
    severity: str
    code: str
    detail: str
    path: str | None = None


@dataclass(frozen=True)
class ExtensionPolicy:
    allowed_sources: tuple[str, ...]
    trust_levels: tuple[str, ...]
    clients: tuple[str, ...]
    administrative_profiles: tuple[str, ...]
    high_risk_isolation: str
    community_secret_access: bool
    automatic_activation: bool
    automatic_permission_expansion: bool
    arbitrary_shell_default: bool
    require_version_or_commit: bool
    require_sha256: bool
    require_human_approval: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_extension_policy(root: Path | None = None) -> ExtensionPolicy:
    payload = _read_toml((root or config_root()) / "extension-policy.toml")
    raw = payload.get("policy")
    if not isinstance(raw, dict):
        raise ContractError("extension-policy.toml must define [policy]")
    policy = ExtensionPolicy(
        allowed_sources=tuple(str(value) for value in raw.get("allowed_sources", ())),
        trust_levels=tuple(str(value) for value in raw.get("trust_levels", ())),
        clients=tuple(str(value) for value in raw.get("clients", ())),
        administrative_profiles=tuple(
            str(value) for value in raw.get("administrative_profiles", ())
        ),
        high_risk_isolation=str(raw.get("high_risk_isolation", "")),
        community_secret_access=bool(raw.get("community_secret_access", True)),
        automatic_activation=bool(raw.get("automatic_activation", True)),
        automatic_permission_expansion=bool(
            raw.get("automatic_permission_expansion", True)
        ),
        arbitrary_shell_default=bool(raw.get("arbitrary_shell_default", True)),
        require_version_or_commit=bool(raw.get("require_version_or_commit", False)),
        require_sha256=bool(raw.get("require_sha256", False)),
        require_human_approval=bool(raw.get("require_human_approval", False)),
    )
    if set(policy.allowed_sources) != {
        "github",
        "private_git",
        "local",
        "npm",
        "pypi",
        "mcp_stdio",
        "mcp_http",
    }:
        raise ContractError("extension policy does not cover every supported source")
    if policy.community_secret_access or policy.automatic_activation:
        raise ContractError("extension policy enables an unsafe default")
    if policy.automatic_permission_expansion or policy.arbitrary_shell_default:
        raise ContractError("extension policy enables silent privilege expansion")
    if policy.high_risk_isolation != "container":
        raise ContractError("high-risk extensions must use container isolation")
    if not (
        policy.require_version_or_commit
        and policy.require_sha256
        and policy.require_human_approval
    ):
        raise ContractError(
            "extension policy must require pins, hashes and human approval"
        )
    return policy


@dataclass(frozen=True)
class InspectionReport:
    extension_id: str
    sha256: str
    risk: str
    isolation: str
    compatible: bool
    approval_possible: bool
    files: int
    bytes: int
    dependencies: tuple[str, ...]
    tools: tuple[str, ...]
    findings: tuple[InspectionFinding, ...]
    inspected_at: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_source(source: str) -> None:
    parsed = urllib.parse.urlsplit(source)
    if parsed.username or parsed.password:
        raise ContractError("extension source URL must not contain credentials")
    if parsed.scheme == "http" and parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise ContractError("remote MCP and package sources must use HTTPS")


def tree_sha256(root: Path) -> tuple[str, int, int]:
    """Hash a regular-file tree without following links or special files."""

    if root.is_symlink():
        raise ContractError("extension payload root must not be a symlink")
    root = root.resolve()
    if not root.is_dir():
        raise ContractError("extension payload must be a regular directory")
    digest = hashlib.sha256()
    count = 0
    total = 0
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        base = Path(directory)
        safe_directories: list[str] = []
        for name in sorted(directory_names):
            candidate = base / name
            if candidate.is_symlink():
                raise ContractError(
                    f"extension payload contains a symlink: {candidate.relative_to(root)}"
                )
            metadata = candidate.lstat()
            if not stat.S_ISDIR(metadata.st_mode):
                raise ContractError(
                    "extension payload contains a special directory entry"
                )
            safe_directories.append(name)
        directory_names[:] = safe_directories
        for name in sorted(file_names):
            candidate = base / name
            metadata = candidate.lstat()
            relative = candidate.relative_to(root).as_posix()
            if candidate.is_symlink() or not stat.S_ISREG(metadata.st_mode):
                raise ContractError(
                    f"extension payload contains a special file: {relative}"
                )
            count += 1
            total += metadata.st_size
            if count > MAX_EXTENSION_FILES or total > MAX_EXTENSION_BYTES:
                raise ContractError("extension payload exceeds quarantine limits")
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            with candidate.open("rb") as handle:
                while chunk := handle.read(128 * 1024):
                    digest.update(chunk)
            digest.update(b"\0")
    return digest.hexdigest(), count, total


def _copy_regular_tree(source: Path, destination: Path) -> None:
    digest, _, _ = tree_sha256(source)
    del digest
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    for directory, directory_names, file_names in os.walk(source, followlinks=False):
        relative_directory = Path(directory).relative_to(source)
        target_directory = destination / relative_directory
        os.chmod(target_directory, 0o700)
        for name in sorted(directory_names):
            candidate = Path(directory) / name
            if candidate.is_symlink():
                raise ContractError("extension tree changed during quarantine")
            (target_directory / name).mkdir(mode=0o700)
        for name in sorted(file_names):
            candidate = Path(directory) / name
            if candidate.is_symlink() or not candidate.is_file():
                raise ContractError("extension tree changed during quarantine")
            target = target_directory / name
            with candidate.open("rb") as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=128 * 1024)
                writer.flush()
                os.fsync(writer.fileno())
            os.chmod(target, 0o600)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_private_directory(path.parent)
    if path.is_symlink():
        raise ContractError("refusing to replace a registry symlink")
    data = json.dumps(payload, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    if len(data) > MAX_STATE_BYTES:
        raise ContractError("extension registry is too large")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _manifest_from_state(row: dict[str, Any]) -> ExtensionManifest:
    manifest = row.get("manifest")
    if not isinstance(manifest, dict):
        raise ContractError("extension registry has an invalid manifest")
    return ExtensionManifest.from_mapping(manifest)


class ExtensionRegistry:
    """State machine: discover -> quarantine -> inspect -> test -> approve -> activate."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (
            (root or (default_state_root() / "extensions")).expanduser().resolve()
        )
        self.quarantine_root = self.root / "quarantine"
        self.approved_root = self.root / "approved"
        self.state_path = self.root / "registry.sqlite"
        ensure_private_directory(self.root)
        ensure_private_directory(self.quarantine_root)
        ensure_private_directory(self.approved_root)
        self._state = SQLiteStateStore(
            self.state_path,
            default_factory=lambda: {
                "schema_version": 1,
                "extensions": {},
                "profiles": {},
                "activation_history": [],
            },
            legacy_path=self.root / "registry.json",
        )

    def _load(self) -> dict[str, Any]:
        payload = self._state.load()
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ContractError("unsupported extension registry")
        return payload

    def _save(self, state: dict[str, Any]) -> None:
        self._state.save(state)

    @retry_state_conflicts
    def discover(self, manifest: ExtensionManifest) -> dict[str, Any]:
        _safe_source(manifest.source)
        if manifest.trust not in EXTENSION_TRUST_LEVELS:
            raise ContractError("invalid extension trust level")
        state = self._load()
        extensions = state.setdefault("extensions", {})
        if not isinstance(extensions, dict):
            raise ContractError("invalid extension registry")
        row = extensions.setdefault(
            manifest.id, {"versions": {}, "status": "discovered"}
        )
        versions = row.setdefault("versions", {})
        reference = manifest.version or manifest.commit or "unversioned"
        existing = versions.get(reference)
        if isinstance(existing, dict) and existing.get("status") in {
            "approved",
            "active",
        }:
            previous = _manifest_from_state(existing)
            if previous.as_dict() != manifest.as_dict():
                raise ContractError("an approved extension reference is immutable")
        else:
            versions[reference] = {
                "manifest": manifest.as_dict(),
                "status": "discovered",
                "discovered_at": _utc_now(),
            }
        row["current_candidate"] = reference
        row["status"] = versions[reference].get("status", "discovered")
        self._save(state)
        return {"id": manifest.id, "reference": reference, "status": "discovered"}

    @retry_state_conflicts
    def quarantine(
        self, manifest: ExtensionManifest, payload: Path
    ) -> ExtensionManifest:
        _safe_source(manifest.source)
        payload_input = payload.expanduser()
        if payload_input.is_symlink():
            raise ContractError("extension payload root must not be a symlink")
        payload = payload_input.resolve()
        source_digest, _, _ = tree_sha256(payload)
        if manifest.sha256 and manifest.sha256 != source_digest:
            raise ContractError("extension payload does not match its pinned SHA-256")
        pinned = (
            manifest if manifest.sha256 else replace(manifest, sha256=source_digest)
        )
        reference = pinned.version or pinned.commit or source_digest[:16]
        target_parent = self.quarantine_root / pinned.id
        ensure_private_directory(target_parent)
        target = target_parent / f"{reference}-{source_digest[:16]}"
        if target.exists() or target.is_symlink():
            existing_digest, _, _ = tree_sha256(target)
            if existing_digest != source_digest:
                raise ContractError("quarantine target collision")
        else:
            staging = target_parent / f".staging-{os.getpid()}-{source_digest[:12]}"
            if staging.exists() or staging.is_symlink():
                raise ContractError("unsafe quarantine staging path")
            _copy_regular_tree(payload, staging)
            copied_digest, _, _ = tree_sha256(staging)
            if copied_digest != source_digest:
                raise ContractError("extension changed while entering quarantine")
            os.replace(staging, target)
        state = self._load()
        extensions = state.setdefault("extensions", {})
        row = extensions.setdefault(pinned.id, {"versions": {}})
        versions = row.setdefault("versions", {})
        existing = versions.get(reference)
        if isinstance(existing, dict):
            previous = _manifest_from_state(existing)
            if previous.sha256 and previous.sha256 != pinned.sha256:
                raise ContractError(
                    "extension version/commit already has a different hash"
                )
            if existing.get("status") in {"approved", "active"}:
                if previous.as_dict() != pinned.as_dict():
                    raise ContractError("an approved extension reference is immutable")
                self.verify_approved(pinned.id, reference, state=state)
                row["current_candidate"] = reference
                self._save(state)
                return previous
        versions[reference] = {
            "manifest": pinned.as_dict(),
            "status": "quarantined",
            "quarantine_path": str(target),
            "quarantined_at": _utc_now(),
        }
        row["current_candidate"] = reference
        row["status"] = "quarantined"
        self._save(state)
        return pinned

    def _candidate(
        self, extension_id: str
    ) -> tuple[dict[str, Any], dict[str, Any], str]:
        state = self._load()
        extensions = state.get("extensions")
        if not isinstance(extensions, dict) or extension_id not in extensions:
            raise ContractError(f"unknown extension: {extension_id}")
        row = extensions[extension_id]
        reference = row.get("current_candidate")
        versions = row.get("versions")
        if not isinstance(reference, str) or not isinstance(versions, dict):
            raise ContractError("extension has no current candidate")
        version = versions.get(reference)
        if not isinstance(version, dict):
            raise ContractError("extension candidate is missing")
        return state, version, reference

    def verify_approved(
        self,
        extension_id: str,
        reference: str,
        *,
        state: dict[str, Any] | None = None,
    ) -> tuple[ExtensionManifest, dict[str, Any], Path]:
        """Recheck an approved payload immediately before it is consumed."""

        current = state or self._load()
        row = current.get("extensions", {}).get(extension_id)
        if not isinstance(row, dict):
            raise ContractError("extension is missing from the approved registry")
        versions = row.get("versions")
        version = versions.get(reference) if isinstance(versions, dict) else None
        if not isinstance(version, dict):
            raise ContractError("approved extension version is missing")
        manifest = _manifest_from_state(version)
        approval = version.get("approval")
        approved_path = version.get("approved_path")
        if (
            not isinstance(approval, dict)
            or approval.get("sha256") != manifest.sha256
            or not isinstance(approved_path, str)
        ):
            raise ContractError("extension version has no valid approval receipt")
        path = Path(approved_path)
        expected_parent = (self.approved_root / manifest.id).resolve()
        if not path.resolve().is_relative_to(expected_parent):
            raise ContractError("approved extension path escapes the registry")
        digest, _, _ = tree_sha256(path)
        if digest != manifest.sha256:
            raise ContractError("approved extension payload failed its integrity check")
        return manifest, version, path

    @retry_state_conflicts
    def inspect(self, extension_id: str) -> InspectionReport:
        state, version, reference = self._candidate(extension_id)
        manifest = _manifest_from_state(version)
        raw_path = version.get("quarantine_path")
        if not isinstance(raw_path, str):
            raise ContractError("extension must enter quarantine before inspection")
        path = Path(raw_path)
        expected_parent = (self.quarantine_root / manifest.id).resolve()
        if not path.resolve().is_relative_to(expected_parent):
            raise ContractError("quarantine path escapes the registry")
        digest, file_count, byte_count = tree_sha256(path)
        if digest != manifest.sha256:
            raise ContractError("quarantined extension hash changed")
        findings: list[InspectionFinding] = []
        dependencies: set[str] = set()
        discovered_shell = False
        discovered_network = False
        discovered_secrets = False
        for candidate in sorted(path.rglob("*")):
            if not candidate.is_file() or candidate.is_symlink():
                continue
            relative = candidate.relative_to(path).as_posix()
            if candidate.name == "package.json":
                try:
                    package = json.loads(candidate.read_text(encoding="utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    findings.append(
                        InspectionFinding(
                            "high",
                            "MALFORMED_PACKAGE_JSON",
                            "package.json is not valid JSON",
                            relative,
                        )
                    )
                    continue
                if isinstance(package, dict):
                    for group in (
                        "dependencies",
                        "optionalDependencies",
                        "peerDependencies",
                    ):
                        values = package.get(group, {})
                        if isinstance(values, dict):
                            dependencies.update(str(name) for name in values)
                    scripts = package.get("scripts", {})
                    if isinstance(scripts, dict):
                        active = sorted(INSTALL_SCRIPT_NAMES & set(scripts))
                        for name in active:
                            findings.append(
                                InspectionFinding(
                                    "high",
                                    "INSTALL_SCRIPT",
                                    f"package declares {name}; it must not run on the host",
                                    relative,
                                )
                            )
            if candidate.stat().st_size > MAX_INSPECT_TEXT_BYTES:
                findings.append(
                    InspectionFinding(
                        "medium",
                        "LARGE_CONTEXT_FILE",
                        "file exceeds the per-file inspection/context limit",
                        relative,
                    )
                )
                continue
            try:
                text = candidate.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if any(pattern.search(text) for pattern in SHELL_PATTERNS):
                discovered_shell = True
                findings.append(
                    InspectionFinding(
                        "high",
                        "PROCESS_EXECUTION",
                        "source can spawn a process",
                        relative,
                    )
                )
            if any(pattern.search(text) for pattern in NETWORK_PATTERNS):
                discovered_network = True
            if any(pattern.search(text) for pattern in SECRET_PATTERNS):
                discovered_secrets = True
            if manifest.kind == "skill" and candidate.name == "SKILL.md":
                lowered = text.lower()
                if "description:" not in lowered:
                    findings.append(
                        InspectionFinding(
                            "medium",
                            "SKILL_TRIGGER_UNSCOPED",
                            "skill has no trigger description",
                            relative,
                        )
                    )
                if "always use" in lowered or "every task" in lowered:
                    findings.append(
                        InspectionFinding(
                            "high",
                            "SKILL_TRIGGER_GLOBAL",
                            "skill attempts to trigger for every task",
                            relative,
                        )
                    )
        unknown_permissions = sorted(set(manifest.permissions) - KNOWN_PERMISSIONS)
        for permission in unknown_permissions:
            findings.append(InspectionFinding("high", "UNKNOWN_PERMISSION", permission))
        if (
            discovered_shell
            and "process_spawn" not in manifest.permissions
            and "shell" not in manifest.permissions
        ):
            findings.append(
                InspectionFinding(
                    "critical",
                    "UNDECLARED_PROCESS_EXECUTION",
                    "process execution was not declared",
                )
            )
        if discovered_network and "network" not in manifest.permissions:
            findings.append(
                InspectionFinding(
                    "high", "UNDECLARED_NETWORK", "network access was not declared"
                )
            )
        if discovered_secrets and manifest.trust in COMMUNITY_TRUST:
            findings.append(
                InspectionFinding(
                    "critical",
                    "COMMUNITY_SECRET_ACCESS",
                    "community code contains credential-access patterns",
                )
            )
        if "secrets" in manifest.permissions and manifest.trust in COMMUNITY_TRUST:
            findings.append(
                InspectionFinding(
                    "critical",
                    "COMMUNITY_SECRET_PERMISSION",
                    "community extensions cannot read provider secrets",
                )
            )
        if manifest.install_scripts:
            findings.append(
                InspectionFinding(
                    "high",
                    "DECLARED_INSTALL_SCRIPT",
                    ",".join(manifest.install_scripts),
                )
            )
        if manifest.kind == "mcp":
            if manifest.source_type == "mcp_stdio" and not manifest.entrypoint:
                findings.append(
                    InspectionFinding(
                        "critical",
                        "MCP_ENTRYPOINT_MISSING",
                        "stdio MCP requires a fixed argv entrypoint",
                    )
                )
            if manifest.source_type == "mcp_http":
                _safe_source(manifest.source)
                if not manifest.remote_domains:
                    findings.append(
                        InspectionFinding(
                            "high",
                            "MCP_DOMAIN_UNDECLARED",
                            "remote MCP must declare its allowed domains",
                        )
                    )
            if not manifest.tools:
                findings.append(
                    InspectionFinding(
                        "medium",
                        "MCP_TOOL_ALLOWLIST_EMPTY",
                        "MCP exposes no explicit tool allowlist",
                    )
                )
        severities = {finding.severity for finding in findings}
        risk = (
            "critical"
            if "critical" in severities
            else "high"
            if "high" in severities
            else "medium"
            if "medium" in severities
            else "low"
        )
        compatible = not any(item.startswith("!") for item in manifest.compatibility)
        approval_possible = (
            risk != "critical"
            and bool(manifest.version or manifest.commit)
            and bool(manifest.sha256)
        )
        isolation = (
            "reject"
            if risk == "critical"
            else "container"
            if risk == "high" or manifest.requires_host_access
            else "worker"
        )
        report = InspectionReport(
            extension_id=manifest.id,
            sha256=digest,
            risk=risk,
            isolation=isolation,
            compatible=compatible,
            approval_possible=approval_possible,
            files=file_count,
            bytes=byte_count,
            dependencies=tuple(sorted(dependencies)),
            tools=manifest.tools,
            findings=tuple(findings),
            inspected_at=_utc_now(),
        )
        extensions = state["extensions"]
        row = extensions[manifest.id]
        candidate = row["versions"][reference]
        candidate["inspection"] = report.as_dict()
        candidate["status"] = "inspected" if approval_possible else "quarantined"
        row["status"] = candidate["status"]
        self._save(state)
        return report

    @retry_state_conflicts
    def mark_tested(
        self,
        extension_id: str,
        *,
        passed: bool,
        sandbox: str = "container",
    ) -> dict[str, Any]:
        if sandbox not in {"container", "bubblewrap"}:
            raise ContractError("extensions must be tested in an isolated sandbox")
        state, version, _ = self._candidate(extension_id)
        inspection = version.get("inspection")
        if not isinstance(inspection, dict):
            raise ContractError("extension must be inspected before testing")
        version["test"] = {
            "passed": bool(passed),
            "sandbox": sandbox,
            "tested_at": _utc_now(),
            "host_credentials_mounted": False,
            "broker_socket_mounted": False,
        }
        version["status"] = "tested" if passed else "quarantined"
        state["extensions"][extension_id]["status"] = version["status"]
        self._save(state)
        return dict(version["test"])

    @retry_state_conflicts
    def approve(
        self,
        extension_id: str,
        *,
        expected_sha256: str,
        approver: str,
        human_approved: bool = False,
    ) -> dict[str, Any]:
        if not human_approved or not approver.strip():
            raise ContractError(
                "extension approval requires an identified human action"
            )
        state, version, reference = self._candidate(extension_id)
        manifest = _manifest_from_state(version)
        inspection = version.get("inspection")
        test = version.get("test")
        if (
            not isinstance(inspection, dict)
            or inspection.get("approval_possible") is not True
        ):
            raise ContractError("extension inspection does not permit approval")
        if not isinstance(test, dict) or test.get("passed") is not True:
            raise ContractError("extension must pass isolated tests before approval")
        if inspection.get("risk") == "high" and test.get("sandbox") != "container":
            raise ContractError(
                "high-risk extensions must pass tests in a disposable container"
            )
        if expected_sha256 != manifest.sha256:
            raise ContractError(
                "human approval hash does not match the quarantined payload"
            )
        source = Path(str(version["quarantine_path"]))
        expected_source_parent = (self.quarantine_root / manifest.id).resolve()
        if not source.resolve().is_relative_to(expected_source_parent):
            raise ContractError("quarantined extension path escapes the registry")
        source_digest, _, _ = tree_sha256(source)
        if source_digest != manifest.sha256:
            raise ContractError("quarantined extension changed after inspection")
        approved_parent = self.approved_root / manifest.id
        ensure_private_directory(approved_parent)
        destination = approved_parent / f"{reference}-{manifest.sha256[:16]}"
        if not destination.exists():
            staging = approved_parent / f".staging-{os.getpid()}-{manifest.sha256[:12]}"
            _copy_regular_tree(source, staging)
            copied, _, _ = tree_sha256(staging)
            if copied != manifest.sha256:
                raise ContractError(
                    "approved copy does not match the inspected payload"
                )
            os.replace(staging, destination)
        else:
            copied, _, _ = tree_sha256(destination)
            if copied != manifest.sha256:
                raise ContractError("existing approved extension payload is corrupted")
        version["approved_path"] = str(destination)
        version["approval"] = {
            "approved_by": approver,
            "approved_at": _utc_now(),
            "sha256": manifest.sha256,
            "permissions": list(manifest.permissions),
        }
        version["status"] = "approved"
        row = state["extensions"][extension_id]
        row["approved"] = reference
        row["status"] = "approved"
        self._save(state)
        return {
            "id": extension_id,
            "reference": reference,
            "sha256": manifest.sha256,
            "status": "approved",
            "activated": False,
        }

    @retry_state_conflicts
    def activate(
        self,
        extension_id: str,
        profile: str,
        *,
        human_approved: bool = False,
        host_risk_approved: bool = False,
    ) -> dict[str, Any]:
        if not human_approved:
            raise ContractError("extension activation requires a human action")
        state = self._load()
        row = state.get("extensions", {}).get(extension_id)
        if not isinstance(row, dict) or not isinstance(row.get("approved"), str):
            raise ContractError("extension has no approved pinned version")
        reference = row["approved"]
        manifest, version, _ = self.verify_approved(
            extension_id, reference, state=state
        )
        if profile in ADMIN_PROFILES and manifest.trust in COMMUNITY_TRUST:
            raise ContractError(
                "the Harness administrative profile cannot load community code"
            )
        if manifest.profiles and profile not in manifest.profiles:
            raise ContractError("extension is not approved for this profile")
        inspection = version.get("inspection", {})
        if (
            inspection.get("risk") == "high"
            and inspection.get("isolation") != "container"
        ):
            raise ContractError(
                "high-risk extension lacks disposable-container isolation"
            )
        if manifest.requires_host_access and not host_risk_approved:
            raise ContractError(
                "direct host access requires a separate human risk approval"
            )
        profiles = state.setdefault("profiles", {})
        active = profiles.setdefault(profile, {})
        previous = active.get(extension_id)
        history = state.setdefault("activation_history", [])
        history.append(
            {
                "extension_id": extension_id,
                "profile": profile,
                "previous": previous,
                "current": reference,
                "changed_at": _utc_now(),
            }
        )
        active[extension_id] = reference
        row["status"] = "active"
        self._save(state)
        return {
            "id": extension_id,
            "profile": profile,
            "reference": reference,
            "status": "active",
            "previous": previous,
            "isolation": inspection.get("isolation", "worker"),
        }

    @retry_state_conflicts
    def rollback(
        self,
        extension_id: str,
        profile: str,
        *,
        human_approved: bool = False,
    ) -> dict[str, Any]:
        if not human_approved:
            raise ContractError("extension rollback requires a human action")
        state = self._load()
        profiles = state.get("profiles", {})
        active = profiles.get(profile, {}) if isinstance(profiles, dict) else {}
        if not isinstance(active, dict) or extension_id not in active:
            raise ContractError("extension is not active in this profile")
        current = active[extension_id]
        history = state.get("activation_history", [])
        previous: str | None = None
        if isinstance(history, list):
            for event in reversed(history):
                if (
                    isinstance(event, dict)
                    and event.get("extension_id") == extension_id
                    and event.get("profile") == profile
                    and event.get("current") == current
                ):
                    value = event.get("previous")
                    previous = value if isinstance(value, str) else None
                    break
        if previous is None:
            del active[extension_id]
            status = "disabled"
        else:
            self.verify_approved(extension_id, previous, state=state)
            active[extension_id] = previous
            status = "rolled_back"
        state.setdefault("activation_history", []).append(
            {
                "extension_id": extension_id,
                "profile": profile,
                "previous": current,
                "current": previous,
                "changed_at": _utc_now(),
                "rollback": True,
            }
        )
        self._save(state)
        return {
            "id": extension_id,
            "profile": profile,
            "from": current,
            "to": previous,
            "status": status,
        }

    def inventory(self, *, profile: str | None = None) -> dict[str, Any]:
        state = self._load()
        extensions = state.get("extensions", {})
        rows: list[dict[str, Any]] = []
        if isinstance(extensions, dict):
            for extension_id, row in sorted(extensions.items()):
                if not isinstance(row, dict):
                    continue
                reference = row.get("approved") or row.get("current_candidate")
                version = (
                    row.get("versions", {}).get(reference, {}) if reference else {}
                )
                manifest = (
                    version.get("manifest", {}) if isinstance(version, dict) else {}
                )
                rows.append(
                    {
                        "id": extension_id,
                        "status": row.get("status"),
                        "reference": reference,
                        "manifest": manifest,
                        "inspection": version.get("inspection")
                        if isinstance(version, dict)
                        else None,
                    }
                )
        active = state.get("profiles", {}).get(profile, {}) if profile else {}
        return {
            "schema_version": 1,
            "extensions": rows,
            "profile": profile,
            "active": active,
        }

    def worker_policy(self, extension_id: str, profile: str) -> dict[str, Any]:
        state = self._load()
        reference = state.get("profiles", {}).get(profile, {}).get(extension_id)
        if not isinstance(reference, str):
            raise ContractError("extension is not active for the requested profile")
        version = state["extensions"][extension_id]["versions"][reference]
        manifest = _manifest_from_state(version)
        inspection = version.get("inspection", {})
        return {
            "schema_version": 1,
            "extension_id": extension_id,
            "profile": profile,
            "permissions": list(manifest.permissions),
            "tools": list(manifest.tools),
            "remote_domains": list(manifest.remote_domains),
            "isolation": inspection.get("isolation", "worker"),
            "credential_mode": "ephemeral_task_scoped",
            "provider_secrets_readable": False,
            "administrative_socket_mounted": False,
            "host_home_mounted": False,
        }


def extension_manifest_from_file(path: Path) -> ExtensionManifest:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 256 * 1024:
        raise ContractError("unsafe extension manifest")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("extension manifest must be JSON") from exc
    if not isinstance(payload, dict):
        raise ContractError("extension manifest must contain an object")
    return ExtensionManifest.from_mapping(payload)
