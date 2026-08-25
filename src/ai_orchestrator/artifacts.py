"""Private, expiring patch artifacts with explicit human acceptance."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ai_orchestrator.models import ContractError, TaskSpec, ensure_private_directory
from ai_orchestrator.telemetry import default_state_root
from ai_orchestrator.writer import (
    GateOutcome,
    WorktreeTransaction,
    gates_passed,
    run_gates,
)


def _git(repo: Path, arguments: list[str]) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise ContractError("artifact repository Git check failed")
    return completed.stdout


class ArtifactStore:
    def __init__(
        self,
        root: Path | None = None,
        *,
        telemetry: Any | None = None,
        gate_runner: Callable[
            [Path, tuple[str, ...], int], tuple[GateOutcome, ...]
        ] = run_gates,
    ) -> None:
        self.root = (root or default_state_root() / "artifacts").resolve()
        ensure_private_directory(self.root)
        self.telemetry = telemetry
        self.gate_runner = gate_runner
        self.purge_expired()

    def _directory(self, artifact_id: str) -> Path:
        try:
            uuid.UUID(artifact_id)
        except ValueError as exc:
            raise ContractError("invalid artifact identifier") from exc
        return self.root / artifact_id

    def purge_expired(self, *, now: datetime | None = None) -> list[str]:
        """Remove only well-formed artifact directories past their retention date."""

        observed = now or datetime.now(UTC)
        if observed.tzinfo is None:
            raise ValueError("artifact purge timestamp must be timezone-aware")
        removed: list[str] = []
        for directory in self.root.iterdir():
            try:
                uuid.UUID(directory.name)
            except ValueError:
                continue
            if directory.is_symlink() or not directory.is_dir():
                continue
            manifest_path = directory / "manifest.json"
            if manifest_path.is_symlink() or not manifest_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                expires_at = datetime.fromisoformat(str(manifest["expires_at"]))
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if (
                not isinstance(manifest, dict)
                or manifest.get("artifact_id") != directory.name
                or expires_at.tzinfo is None
                or observed < expires_at
            ):
                continue
            shutil.rmtree(directory)
            removed.append(directory.name)
        return sorted(removed)

    @staticmethod
    def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
        )
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)

    def stage_patch(
        self,
        *,
        workflow_id: str,
        task_id: str,
        repo: Path,
        base_head: str,
        patch: bytes,
        allowed_write_paths: tuple[str, ...],
        protected_prefixes: tuple[str, ...],
        gate_commands: tuple[str, ...],
        gate_timeout_seconds: int,
        reviewer_provider: str,
        reviewer_model: str,
        reviewer_verdict: str,
    ) -> dict[str, object]:
        if not patch or len(patch) > 20_971_520:
            raise ContractError("patch artifact must contain at most 20 MiB")
        verdict = str(reviewer_verdict).strip().lower()
        if verdict not in {"approve", "reject"}:
            raise ContractError("reviewer verdict must be approve or reject")
        repository = repo.expanduser().resolve()
        if not repository.is_dir():
            raise ContractError("artifact repository does not exist")
        artifact_id = str(uuid.uuid4())
        directory = self._directory(artifact_id)
        directory.mkdir(mode=0o700)
        patch_path = directory / "patch.diff"
        patch_path.write_bytes(patch)
        os.chmod(patch_path, 0o600)
        now = datetime.now(UTC)
        manifest: dict[str, object] = {
            "schema_version": 1,
            "artifact_id": artifact_id,
            "workflow_id": workflow_id,
            "task_id": task_id,
            "repo": str(repository),
            "base_head": base_head,
            "patch_sha256": hashlib.sha256(patch).hexdigest(),
            "patch_bytes": len(patch),
            "allowed_write_paths": list(allowed_write_paths),
            "protected_prefixes": list(protected_prefixes),
            "gate_commands": list(gate_commands),
            "gate_timeout_seconds": int(gate_timeout_seconds),
            "reviewer_provider": reviewer_provider,
            "reviewer_model": reviewer_model,
            "reviewer_verdict": verdict,
            "status": "pending",
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(days=7)).isoformat(),
            "human_acceptance": None,
        }
        self._write_manifest(directory / "manifest.json", manifest)
        return dict(manifest)

    def validate_candidate(
        self,
        *,
        task: TaskSpec,
        patch: bytes,
        protected_prefixes: tuple[str, ...],
        gate_commands: tuple[str, ...],
        gate_timeout_seconds: int,
    ) -> dict[str, object]:
        """Validate scope and gates without changing or persisting the source."""

        with WorktreeTransaction(task) as transaction:
            transaction.snapshot_ignored()
            transaction.apply_candidate_patch(patch)
            collected = transaction.collect_patch(protected_prefixes)
            gate_results = self.gate_runner(
                transaction.worktree, gate_commands, gate_timeout_seconds
            )
            passed = gates_passed(gate_results, len(gate_commands))
        if not passed:
            raise ContractError("candidate patch gates failed")
        return {
            "patch_sha256": collected.sha256,
            "changed_files": len(collected.paths),
            "gates_passed": True,
        }

    def inspect(self, artifact_id: str) -> dict[str, object]:
        directory = self._directory(artifact_id)
        source = directory / "manifest.json"
        if source.is_symlink() or not source.is_file():
            raise ContractError("artifact does not exist")
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError("artifact manifest is invalid") from exc
        if not isinstance(payload, dict) or payload.get("artifact_id") != artifact_id:
            raise ContractError("artifact manifest does not match its identifier")
        return payload

    def _pending(self, artifact_id: str) -> tuple[Path, dict[str, object]]:
        directory = self._directory(artifact_id)
        manifest = self.inspect(artifact_id)
        if manifest.get("status") != "pending":
            raise ContractError("artifact is not pending")
        expires = datetime.fromisoformat(str(manifest["expires_at"]))
        if datetime.now(UTC) >= expires:
            raise ContractError("artifact has expired")
        return directory, manifest

    def accept(
        self,
        artifact_id: str,
        *,
        expected_sha256: str,
        human_approved: bool,
    ) -> dict[str, object]:
        if not human_approved:
            raise ContractError("artifact acceptance requires human approval")
        directory, manifest = self._pending(artifact_id)
        if manifest.get("reviewer_verdict") != "approve":
            raise ContractError("artifact reviewer did not approve the patch")
        patch_path = directory / "patch.diff"
        if patch_path.is_symlink() or not patch_path.is_file():
            raise ContractError("patch artifact is unavailable")
        patch = patch_path.read_bytes()
        actual_sha256 = hashlib.sha256(patch).hexdigest()
        if (
            actual_sha256 != manifest.get("patch_sha256")
            or actual_sha256 != expected_sha256
        ):
            raise ContractError("patch artifact hash mismatch")
        repo = Path(str(manifest["repo"])).resolve()
        current_head = _git(repo, ["rev-parse", "HEAD"]).decode().strip()
        if current_head != manifest.get("base_head"):
            raise ContractError("repository HEAD changed after artifact creation")
        if _git(repo, ["status", "--porcelain=v1", "--untracked-files=all"]):
            raise ContractError("repository must be clean before artifact acceptance")
        task = TaskSpec.from_mapping(
            {
                "task_id": str(uuid.uuid4()),
                "repo": str(repo),
                "task_type": "implementation",
                "sensitivity": "internal",
                "write_mode": "validated_patch",
                "allowed_write_paths": manifest["allowed_write_paths"],
                "max_patch_bytes": max(1024, int(manifest["patch_bytes"])),
            }
        )
        commands = tuple(str(value) for value in manifest["gate_commands"])
        timeout = int(manifest["gate_timeout_seconds"])
        with WorktreeTransaction(task) as transaction:
            transaction.snapshot_ignored()
            transaction.apply_candidate_patch(patch)
            transaction.collect_patch(
                tuple(str(value) for value in manifest["protected_prefixes"])
            )
            gate_results = self.gate_runner(transaction.worktree, commands, timeout)
            if not gates_passed(gate_results, len(commands)):
                raise ContractError("artifact gates failed")
            transaction.apply(patch)
        accepted_at = datetime.now(UTC).isoformat()
        manifest["status"] = "accepted"
        manifest["human_acceptance"] = "accepted"
        manifest["accepted_at"] = accepted_at
        manifest["gates_passed"] = True
        self._write_manifest(directory / "manifest.json", manifest)
        if self.telemetry is not None:
            self.telemetry.record_acceptance(
                workflow_id=str(manifest["workflow_id"]),
                artifact_id=artifact_id,
                verdict="accepted",
                accepted_at=accepted_at,
            )
        return dict(manifest)

    def reject(
        self,
        artifact_id: str,
        *,
        human_approved: bool,
        reason: str | None = None,
    ) -> dict[str, object]:
        if not human_approved:
            raise ContractError("artifact rejection requires human approval")
        directory, manifest = self._pending(artifact_id)
        rejected_at = datetime.now(UTC).isoformat()
        manifest["status"] = "rejected"
        manifest["human_acceptance"] = "rejected"
        manifest["rejected_at"] = rejected_at
        manifest["rejection_reason"] = str(reason or "")[:200] or None
        self._write_manifest(directory / "manifest.json", manifest)
        if self.telemetry is not None:
            self.telemetry.record_acceptance(
                workflow_id=str(manifest["workflow_id"]),
                artifact_id=artifact_id,
                verdict="rejected",
                accepted_at=rejected_at,
            )
        return dict(manifest)
