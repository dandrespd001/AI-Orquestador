from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SENSITIVITY_ORDER = {
    "public": 0,
    "internal": 1,
    "confidential": 2,
    "secrets": 3,
    "financial_control": 4,
    "deployment": 5,
}

ALLOWED_OUTPUT_CONTRACTS = {"text", "json", "unified_diff", "review"}


class ContractError(ValueError):
    """Raised when an orchestration contract is invalid."""


def sensitivity_rank(value: str) -> int:
    try:
        return SENSITIVITY_ORDER[value]
    except KeyError as exc:
        raise ContractError(f"unknown sensitivity: {value!r}") from exc


def _clean_relative_path(value: object, field_name: str) -> str:
    text = str(value).strip()
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise ContractError(f"{field_name} must be a repository-relative path")
    return path.as_posix()


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    repo: str
    task_type: str
    sensitivity: str
    prompt_file: str | None = None
    input_paths: tuple[str, ...] = ()
    output_contract: str = "text"
    max_context_tokens: int = 120_000
    timeout_seconds: int = 900
    allow_parallel: bool = False

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "TaskSpec":
        task_id = str(payload.get("task_id") or uuid.uuid4())
        try:
            uuid.UUID(task_id)
        except ValueError as exc:
            raise ContractError("task_id must be a UUID") from exc

        repo = Path(str(payload.get("repo", ""))).expanduser().resolve()
        if not repo.is_dir():
            raise ContractError(f"repo does not exist: {repo}")

        task_type = str(payload.get("task_type", "")).strip()
        if not task_type:
            raise ContractError("task_type is required")

        sensitivity = str(payload.get("sensitivity", "internal")).strip()
        sensitivity_rank(sensitivity)

        output_contract = str(payload.get("output_contract", "text"))
        if output_contract not in ALLOWED_OUTPUT_CONTRACTS:
            raise ContractError(f"unknown output_contract: {output_contract!r}")

        prompt_value = payload.get("prompt_file")
        prompt_file = None
        if prompt_value is not None:
            prompt_file = _clean_relative_path(prompt_value, "prompt_file")
            prompt_candidate = repo / prompt_file
            if prompt_candidate.is_symlink():
                raise ContractError("prompt_file must not be a symlink")
            resolved_prompt = prompt_candidate.resolve()
            if not resolved_prompt.is_file() or not resolved_prompt.is_relative_to(repo):
                raise ContractError("prompt_file must be a regular file inside repo")

        input_paths = tuple(
            _clean_relative_path(value, "input_paths")
            for value in payload.get("input_paths", ())
        )
        for relative in input_paths:
            candidate = repo / relative
            if candidate.is_symlink():
                raise ContractError(f"input path must not be a symlink: {relative}")
            resolved = candidate.resolve()
            if not resolved.is_relative_to(repo):
                raise ContractError(f"input path escapes repo: {relative}")
            if not resolved.exists():
                raise ContractError(f"input path missing: {relative}")

        max_context_tokens = int(payload.get("max_context_tokens", 120_000))
        if not 512 <= max_context_tokens <= 600_000:
            raise ContractError("max_context_tokens must be between 512 and 600000")

        timeout_seconds = int(payload.get("timeout_seconds", 900))
        if not 1 <= timeout_seconds <= 1800:
            raise ContractError("timeout_seconds must be between 1 and 1800")

        return cls(
            task_id=task_id,
            repo=str(repo),
            task_type=task_type,
            sensitivity=sensitivity,
            prompt_file=prompt_file,
            input_paths=input_paths,
            output_contract=output_contract,
            max_context_tokens=max_context_tokens,
            timeout_seconds=timeout_seconds,
            allow_parallel=bool(payload.get("allow_parallel", False)),
        )

    @classmethod
    def read(cls, path: str | Path) -> "TaskSpec":
        source = Path(path)
        if not source.is_file() or source.is_symlink():
            raise ContractError("spec must be a regular JSON file")
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ContractError("spec must contain a JSON object")
        return cls.from_mapping(payload)

    def prompt_path(self) -> Path | None:
        return Path(self.repo, self.prompt_file) if self.prompt_file else None

    def context_sha256(self) -> str:
        digest = hashlib.sha256()
        root = Path(self.repo)
        selected = ([self.prompt_file] if self.prompt_file else []) + list(self.input_paths)
        for relative in sorted(set(selected)):
            if relative is None:
                continue
            path = (root / relative).resolve()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            if path.is_file():
                digest.update(path.read_bytes())
            else:
                digest.update(b"<directory>")
            digest.update(b"\0")
        return digest.hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelConfig:
    id: str
    family: str
    provider: str
    transport: str
    route_model: str
    official: bool
    status: str
    roles: tuple[str, ...]
    max_sensitivity: str
    max_concurrency: int


@dataclass(frozen=True)
class RoleConfig:
    name: str
    models: tuple[str, ...]
    reviewers: tuple[str, ...]


@dataclass(frozen=True)
class Profile:
    name: str
    repo_root: str | None
    auto_execute_sensitivities: tuple[str, ...]
    allowed_models: tuple[str, ...]
    deny_providers: tuple[str, ...]
    protected_path_prefixes: tuple[str, ...]
    deterministic_gates: tuple[str, ...]
    minimax_mode: str
    preserve_live_trading_false: bool = False


@dataclass(frozen=True)
class RouteDecision:
    schema_version: int
    decision_id: str
    task_id: str
    profile: str
    status: str
    selected_model: str | None
    requested_route_model: str | None
    provider: str | None
    family: str | None
    transport: str | None
    reviewer_model: str | None
    alternatives: tuple[str, ...]
    unavailable: dict[str, str]
    auto_execute: bool
    approval_required: bool
    reason: str
    context_sha256: str
    fallback_allowed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunResult:
    schema_version: int
    run_id: str
    decision_id: str
    task_id: str
    status: str
    requested_model: str
    reported_model: str | None
    provider: str
    transport: str
    started_at: str
    finished_at: str
    latency_ms: int
    output_sha256: str | None
    output_bytes: int
    exit_code: int | None
    provenance_verified: bool
    error_code: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink() or not path.is_dir():
        raise ContractError(f"unsafe directory: {path}")
    os.chmod(path, 0o700)
