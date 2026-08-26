"""Pinned, loopback-only DeepSeek Harness launch policy."""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import stat
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ai_orchestrator.broker import default_socket_path
from ai_orchestrator.credentials import omniroute_runtime_key
from ai_orchestrator.harness_bridge import HarnessBridge
from ai_orchestrator.models import ContractError, TaskSpec, ensure_private_directory
from ai_orchestrator.supervisor import run_supervised
from ai_orchestrator.telemetry import default_state_root

HARNESS_PACKAGE = "@deepseek-ai/dsh"
HARNESS_VERSION = "0.1.1-rc.2"
MINIMUM_SECURE_HARNESS_VERSION = "0.1.1-rc.1"
HARNESS_HOST = "127.0.0.1"
HARNESS_PORT = 3080
HARNESS_ADMIN_PROFILE = "orchestrator-web"
HARNESS_PROFILES = frozenset({"orchestrator-web", "orchestrator-worker"})
HARNESS_MODELS = frozenset({"deepseek-v4-pro", "deepseek-v4-flash"})
HARNESS_UI_MODELS = HARNESS_MODELS | frozenset({"MiniMax-M3"})


def harness_version_supported(version: str) -> bool:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:-rc\.(\d+))?", version.strip())
    if match is None:
        return False
    major, minor, patch = (int(match.group(index)) for index in (1, 2, 3))
    release_candidate = int(match.group(4)) if match.group(4) is not None else None
    if (major, minor, patch) > (0, 1, 1):
        return True
    if (major, minor, patch) < (0, 1, 1):
        return False
    return release_candidate is None or release_candidate >= 1


def _make_tree_owner_writable(path: Path) -> None:
    for root, directories, files in os.walk(path, followlinks=False):
        os.chmod(root, 0o700)
        for name in directories:
            candidate = Path(root) / name
            if not candidate.is_symlink():
                os.chmod(candidate, 0o700)
        for name in files:
            candidate = Path(root) / name
            if not candidate.is_symlink():
                os.chmod(candidate, 0o600)


def _replace_template_tree(source: Path, target: Path) -> None:
    """Atomically replace audited config while leaving DSH sessions untouched."""

    ensure_private_directory(target.parent)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent)
    )
    candidate = staging_root / target.name
    backup = target.parent / f".{target.name}.previous-{uuid.uuid4().hex}"
    moved_previous = False
    try:
        shutil.copytree(source, candidate, symlinks=True)
        # Immutable slots are mode 0555/0444. DSH derives cordis.yml while
        # composing a profile, so its private runtime copy must be writable.
        _make_tree_owner_writable(candidate)
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_dir():
                raise ContractError("Harness profile destination is unsafe")
            os.replace(target, backup)
            moved_previous = True
        os.replace(candidate, target)
    except Exception:
        if moved_previous and not target.exists():
            os.replace(backup, target)
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
        if moved_previous and backup.exists():
            _make_tree_owner_writable(backup)
            shutil.rmtree(backup)


def _materialize_profile(
    slot: Path, dsh_home: Path, profile: str, *, ephemeral: bool
) -> bool:
    """Install only the audited bundle/profile files into a DSH home."""

    source_profile = slot / "profiles" / profile
    source_bundle = slot / "bundle"
    if not source_profile.is_dir() or not source_bundle.is_dir():
        return False
    if source_profile.is_symlink() or source_bundle.is_symlink():
        raise ContractError("Harness profile templates are unsafe")
    ensure_private_directory(dsh_home)
    bundle_target = dsh_home / "bundle"
    profile_target = dsh_home / "profiles" / profile
    ensure_private_directory(profile_target.parent)
    if ephemeral:
        shutil.copytree(source_bundle, bundle_target, symlinks=True)
        shutil.copytree(source_profile, profile_target, symlinks=True)
        _make_tree_owner_writable(bundle_target)
        _make_tree_owner_writable(profile_target)
    else:
        _replace_template_tree(source_bundle, bundle_target)
        _replace_template_tree(source_profile, profile_target)
    return True


@dataclass(frozen=True)
class HarnessReceiptV2:
    """Metadata-only receipt for interactive and headless Harness runs."""

    run_id: str
    task_hash: str
    profile: str
    provider: str
    model: str
    harness_version: str
    transport: str
    started_at: str
    finished_at: str
    usage: dict[str, Any]
    status: str
    artifact: dict[str, Any] | None
    gates: list[dict[str, Any]]
    schema_version: int = 2

    def __post_init__(self) -> None:
        try:
            uuid.UUID(self.run_id)
        except ValueError as exc:
            raise ContractError("Harness receipt run_id must be a UUID") from exc
        if len(self.task_hash) != 64 or any(
            character not in "0123456789abcdef" for character in self.task_hash.lower()
        ):
            raise ContractError("Harness receipt task hash is invalid")
        if self.profile not in HARNESS_PROFILES:
            raise ContractError("Harness receipt profile is invalid")
        if self.provider != "deepseek" or self.model not in HARNESS_MODELS:
            raise ContractError("Harness receipt provider/model is not allowed")
        if not harness_version_supported(self.harness_version):
            raise ContractError("Harness receipt version is below the secure baseline")
        if self.transport != "deepseek_harness":
            raise ContractError("Harness receipt transport is invalid")
        forbidden = {"prompt", "response", "reasoning", "messages", "content"}

        def inspect(value: object) -> None:
            if isinstance(value, dict):
                if forbidden & {str(key).lower() for key in value}:
                    raise ContractError("Harness receipt contains retained content")
                for nested in value.values():
                    inspect(nested)
            elif isinstance(value, list):
                for nested in value:
                    inspect(nested)

        inspect(self.usage)
        inspect(self.artifact)
        inspect(self.gates)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class WorkspaceRegistry:
    """Atomic, approval-gated allowlist of canonical Git workspaces."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()

    def _load(self) -> list[str]:
        if not self.path.exists():
            return []
        if self.path.is_symlink() or not self.path.is_file():
            raise ContractError("workspace registry is unsafe")
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ContractError("workspace registry is invalid")
        rows = payload.get("workspaces", [])
        if not isinstance(rows, list) or not all(isinstance(row, str) for row in rows):
            raise ContractError("workspace registry is invalid")
        return sorted(set(rows))

    def _save(self, rows: list[str]) -> None:
        ensure_private_directory(self.path.parent)
        data = (
            json.dumps(
                {"schema_version": 1, "workspaces": sorted(set(rows))},
                sort_keys=True,
                indent=2,
            ).encode("utf-8")
            + b"\n"
        )
        descriptor, temporary = tempfile.mkstemp(
            prefix=".workspaces.", dir=self.path.parent
        )
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _git_root(path: Path) -> Path:
        unresolved = path.expanduser()
        if unresolved.is_symlink() or ".." in unresolved.parts:
            raise ContractError("workspace path traversal and symlinks are forbidden")
        resolved = unresolved.resolve(strict=True)
        if not resolved.is_dir() or not (resolved / ".git").exists():
            raise ContractError("workspace must be a Git root")
        return resolved

    def add(self, path: Path, *, human_approved: bool = False) -> dict[str, Any]:
        if not human_approved:
            raise ContractError("workspace change requires human approval")
        workspace = str(self._git_root(path))
        rows = self._load()
        if workspace not in rows:
            rows.append(workspace)
            self._save(rows)
        return self.list()

    def remove(self, path: Path, *, human_approved: bool = False) -> dict[str, Any]:
        if not human_approved:
            raise ContractError("workspace change requires human approval")
        workspace = str(path.expanduser().resolve(strict=True))
        rows = self._load()
        if workspace not in rows:
            raise ContractError("workspace is not allowed")
        rows.remove(workspace)
        self._save(rows)
        return self.list()

    def list(self) -> dict[str, Any]:
        return {"schema_version": 1, "workspaces": self._load()}

    def systemd_dropin(self) -> str:
        rows = self._load()
        directives = (
            " ".join(
                '"' + row.replace("\\", "\\\\").replace('"', '\\"') + '"'
                for row in rows
            )
            if rows
            else "/run/ai-orchestrator-no-workspaces"
        )
        return "[Service]\nReadWritePaths=" + directives + "\n"

    def write_systemd_dropin(
        self, path: Path, *, human_approved: bool = False
    ) -> dict[str, Any]:
        if not human_approved:
            raise ContractError("workspace drop-in update requires human approval")
        destination = path.expanduser().resolve()
        if destination.is_symlink():
            raise ContractError("workspace drop-in destination is unsafe")
        ensure_private_directory(destination.parent)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(self.systemd_dropin())
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            os.chmod(destination, 0o600)
        except Exception:
            Path(temporary).unlink(missing_ok=True)
            raise
        return {
            "schema_version": 1,
            "path": str(destination),
            "workspaces": self._load(),
        }


def run_harness_worker(
    *,
    model: str,
    prompt: bytes,
    workspace: Path,
    registry_root: Path | None = None,
    state_root: Path | None = None,
    timeout_seconds: int = 3600,
    extended: bool = False,
    workspace_isolated: bool = False,
    enforce_catalog: bool = True,
) -> dict[str, Any]:
    """Run one headless Harness task without retaining prompt or conversation."""

    if model not in HARNESS_MODELS:
        raise ContractError("Harness worker model is not allowed")
    if enforce_catalog:
        from ai_orchestrator.config import load_models

        catalog_model = load_models().get(model)
        if catalog_model is None or catalog_model.status != "available":
            status = (
                catalog_model.status if catalog_model is not None else "not_discovered"
            )
            raise ContractError(f"Harness worker model is unavailable: {status}")
    limit = 14_400 if extended else 3_600
    if not 1 <= timeout_seconds <= limit:
        raise ContractError(f"Harness worker timeout must be between 1 and {limit}")
    if len(prompt) > 4 * 1024 * 1024:
        raise ContractError("Harness worker prompt exceeds 4 MiB")
    unresolved_workspace = workspace.expanduser()
    if unresolved_workspace.is_symlink() or not unresolved_workspace.is_dir():
        raise ContractError("Harness worker workspace is unsafe")
    resolved_workspace = unresolved_workspace.resolve()
    root = (state_root or default_state_root()).expanduser().resolve()
    if not workspace_isolated:
        from ai_orchestrator.writer import WorktreeTransaction

        isolation_task = TaskSpec(
            task_id=str(uuid.uuid4()),
            repo=str(resolved_workspace),
            task_type="harness_worker",
            sensitivity="internal",
            timeout_seconds=min(timeout_seconds, 1800),
            write_mode="guarded_patch",
            allowed_write_paths=(".",),
            max_changed_files=500,
            max_patch_bytes=20_971_520,
        )
        with WorktreeTransaction(isolation_task) as transaction:
            transaction.snapshot_ignored()
            if transaction.worktree is None:
                raise ContractError("Harness worker worktree was not created")
            payload = run_harness_worker(
                model=model,
                prompt=prompt,
                workspace=transaction.worktree,
                registry_root=registry_root,
                state_root=root,
                timeout_seconds=timeout_seconds,
                extended=extended,
                workspace_isolated=True,
                enforce_catalog=enforce_catalog,
            )
            patch = transaction.collect_patch(
                (".agents", ".codex", ".qwen", "AGENTS.md")
            )
        payload["artifact"] = {
            "sha256": patch.sha256,
            "paths": list(patch.paths),
            "patch_bytes": len(patch.patch),
        }
        payload["canonical_workspace_modified"] = False
        return payload

    workers_root = root / "harness" / "workers"
    ensure_private_directory(workers_root)
    from ai_orchestrator.updates import UpdateRegistry

    updates = UpdateRegistry(registry_root or (root / "updates"), create=False)
    slot, _ = updates.resolve_active_slot("deepseek-harness")
    executable = updates.resolve_active_executable("deepseek-harness")
    output: bytes
    with tempfile.TemporaryDirectory(prefix="run-", dir=workers_root) as temporary:
        session_root = Path(temporary)
        task_file = session_root / "task.txt"
        task_file.write_bytes(prompt)
        task_file.chmod(0o600)
        dsh_home = session_root / "dsh-home"
        ensure_private_directory(dsh_home)
        composed_profile = _materialize_profile(
            slot, dsh_home, "orchestrator-worker", ephemeral=True
        )
        allowed = {"PATH", "LANG", "LC_ALL", "TERM", "NO_COLOR"}
        environment = {
            key: value for key, value in os.environ.items() if key in allowed
        }
        upstream_credential = omniroute_runtime_key()
        if composed_profile and not upstream_credential:
            raise ContractError("Harness worker cannot authenticate to local OmniRoute")
        environment.update(
            {
                "HOME": str(session_root),
                "DSH_HOME": str(dsh_home),
                "DSH_MODEL": model,
                "AI_ORCHESTRATOR_TASK_FILE": str(task_file),
                "AI_ORCHESTRATOR_SOCKET": str(default_socket_path()),
                "AI_ORCHESTRATOR_PYTHON": os.environ.get(
                    "AI_ORCHESTRATOR_PYTHON", "/usr/bin/python3"
                ),
                "AI_ORCHESTRATOR_ROOT": str(resolved_workspace),
                "AI_ORCHESTRATOR_SKILLS_DIR": os.environ.get(
                    "AI_ORCHESTRATOR_SKILLS_DIR", str(root / "skills")
                ),
                "AI_ORCHESTRATOR_CONFIG": os.environ.get(
                    "AI_ORCHESTRATOR_CONFIG",
                    str(Path.home() / ".config" / "ai-orchestrator"),
                ),
                "AI_ORCHESTRATOR_SOURCE_CONFIG": os.environ.get(
                    "AI_ORCHESTRATOR_SOURCE_CONFIG",
                    str(Path.home() / ".config" / "ai-orchestrator"),
                ),
                "NO_COLOR": "1",
            }
        )
        environment.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
        instruction = (
            "Read the task from AI_ORCHESTRATOR_TASK_FILE, complete it in the current "
            "workspace, and return only the final deliverable."
        )
        if upstream_credential:
            with HarnessBridge(
                upstream_credential=upstream_credential, port=0
            ) as bridge:
                environment["AI_ORCHESTRATOR_HARNESS_CREDENTIAL"] = bridge.credential
                environment["AI_ORCHESTRATOR_DEEPSEEK_BRIDGE_URL"] = bridge.url
                outcome = run_supervised(
                    [str(executable), "--profile", "orchestrator-worker", instruction],
                    input_data=None,
                    cwd=resolved_workspace,
                    env=environment,
                    timeout_seconds=timeout_seconds,
                )
        else:
            environment["AI_ORCHESTRATOR_HARNESS_CREDENTIAL"] = secrets.token_urlsafe(
                32
            )
            outcome = run_supervised(
                [str(executable), "--profile", "orchestrator-worker", instruction],
                input_data=None,
                cwd=resolved_workspace,
                env=environment,
                timeout_seconds=timeout_seconds,
            )
        if outcome.returncode != 0:
            if outcome.timed_out:
                raise ContractError("Harness worker exceeded its wall-clock budget")
            raise ContractError(f"Harness worker failed with exit {outcome.returncode}")
        output = outcome.stdout
    return {
        "schema_version": 2,
        "status": "SUCCESS",
        "reported_model": model,
        "result": output.decode("utf-8", errors="strict"),
        "usage": {
            "input_tokens": None,
            "output_tokens": None,
            "tool_calls": None,
        },
        "worker_session_deleted": True,
        "content_retained": False,
    }


@dataclass(frozen=True)
class HarnessLaunchSpec:
    command: tuple[str, ...]
    environment: dict[str, str]
    host: str
    port: int
    profile: str
    broker_socket: str
    community_code_loaded: bool = False
    harness_version: str = HARNESS_VERSION

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def harness_launch_spec(
    *,
    socket_path: Path | None = None,
    state_root: Path | None = None,
    registry_root: Path | None = None,
    profile: str = HARNESS_ADMIN_PROFILE,
    port: int = HARNESS_PORT,
) -> HarnessLaunchSpec:
    if not 1024 <= port <= 65_535:
        raise ContractError("Harness port must be between 1024 and 65535")
    if profile not in HARNESS_PROFILES:
        raise ContractError("unknown Harness profile")
    broker = (socket_path or default_socket_path()).expanduser().resolve()
    root = (state_root or default_state_root()).expanduser().resolve()
    dsh_home = root / "harness"
    ensure_private_directory(dsh_home)
    from ai_orchestrator.updates import UpdateRegistry

    update_root = registry_root or (root / "updates")
    updates = UpdateRegistry(update_root, create=False)
    slot, _ = updates.resolve_active_slot("deepseek-harness")
    active = updates.inventory()["components"]["deepseek-harness"]["active"]
    executable = updates.resolve_active_executable("deepseek-harness")
    if not _materialize_profile(slot, dsh_home, profile, ephemeral=False):
        raise ContractError("active Harness slot has no audited profile bundle")
    allowed = {"HOME", "USER", "LOGNAME", "PATH", "LANG", "LC_ALL", "TERM", "NO_COLOR"}
    environment = {name: value for name, value in os.environ.items() if name in allowed}
    environment.update(
        {
            "HOST": HARNESS_HOST,
            "DSH_HOST": HARNESS_HOST,
            "DSH_PORT": str(port),
            "DSH_HOME": str(dsh_home),
            "AI_ORCHESTRATOR_SOCKET": str(broker),
            "AI_ORCHESTRATOR_HARNESS_PROFILE": profile,
            "AI_ORCHESTRATOR_PYTHON": os.environ.get(
                "AI_ORCHESTRATOR_PYTHON", "/usr/bin/python3"
            ),
            "AI_ORCHESTRATOR_ROOT": os.environ.get(
                "AI_ORCHESTRATOR_ROOT", str(Path.cwd().resolve())
            ),
            "AI_ORCHESTRATOR_SKILLS_DIR": os.environ.get(
                "AI_ORCHESTRATOR_SKILLS_DIR", str(root / "skills")
            ),
            "AI_ORCHESTRATOR_CONFIG": os.environ.get(
                "AI_ORCHESTRATOR_CONFIG",
                str(Path.home() / ".config" / "ai-orchestrator"),
            ),
            "AI_ORCHESTRATOR_SOURCE_CONFIG": os.environ.get(
                "AI_ORCHESTRATOR_SOURCE_CONFIG",
                str(Path.home() / ".config" / "ai-orchestrator"),
            ),
            "NO_COLOR": "1",
        }
    )
    environment.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    return HarnessLaunchSpec(
        command=(
            str(executable),
            "--profile",
            profile,
            "--host",
            HARNESS_HOST,
            "--port",
            str(port),
            "--no-open",
        ),
        environment=environment,
        host=HARNESS_HOST,
        port=port,
        profile=profile,
        broker_socket=str(broker),
        community_code_loaded=False,
        harness_version=str(active["version"]),
    )


def validate_harness_launch(
    spec: HarnessLaunchSpec, *, require_socket: bool = True
) -> dict[str, Any]:
    errors: list[str] = []
    command = list(spec.command)
    host_is_fixed = any(
        command[index : index + 2] == ["--host", HARNESS_HOST]
        for index in range(max(0, len(command) - 1))
    )
    port_is_fixed = any(
        command[index : index + 2] == ["--port", str(spec.port)]
        for index in range(max(0, len(command) - 1))
    )
    if (
        spec.host != HARNESS_HOST
        or spec.environment.get("HOST") != HARNESS_HOST
        or not host_is_fixed
    ):
        errors.append("Harness must bind only to 127.0.0.1")
    if spec.environment.get("DSH_PORT") != str(spec.port) or not port_is_fixed:
        errors.append("Harness port must be explicit and consistent")
    executable = Path(command[0]) if command else Path()
    if not command or command[0] in {"npx", "npm", "pnpm"}:
        errors.append("Harness must execute the promoted immutable slot")
    elif executable.is_symlink() or not executable.is_file():
        errors.append("Harness active executable is unavailable")
    if spec.profile not in HARNESS_PROFILES:
        errors.append("Harness profile is not approved")
    if "--no-open" not in spec.command:
        errors.append("Harness must not launch a browser automatically")
    if spec.community_code_loaded:
        errors.append("administrative Harness profile contains community code")
    forbidden_environment = {
        key
        for key in spec.environment
        if any(
            marker in key.upper()
            for marker in ("API_KEY", "AUTH_TOKEN", "PASSWORD", "SECRET")
        )
    }
    if forbidden_environment:
        errors.append("Harness environment contains provider credentials")
    broker = Path(spec.broker_socket)
    if require_socket:
        try:
            metadata = broker.lstat()
            if (
                not stat.S_ISSOCK(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o600
            ):
                errors.append("broker socket is not a private 0600 Unix socket")
        except FileNotFoundError:
            errors.append("broker socket is unavailable")
    return {
        "schema_version": 1,
        "ok": not errors,
        "host": spec.host,
        "port": spec.port,
        "package": HARNESS_PACKAGE,
        "version": spec.harness_version,
        "errors": errors,
    }
