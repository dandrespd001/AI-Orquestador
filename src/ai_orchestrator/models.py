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
ALLOWED_WRITE_MODES = {"read_only", "validated_patch", "validated_apply"}
ALLOWED_TRUST_TIERS = {"official_subscription", "local", "web_free"}
ALLOWED_EXECUTION_PROFILES = {
    "native_writer",
    "patch_api_writer",
    "kimi_acp_writer",
    "read_only",
}
ALLOWED_MODEL_CAPABILITIES = {"read", "write", "tools"}
ALLOWED_REASONING_EFFORTS = {"low", "medium", "high", "xhigh", "max"}


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
    write_mode: str = "read_only"
    allowed_write_paths: tuple[str, ...] = ()
    gate_profile: str = "profile"
    max_repairs: int = 2
    max_changed_files: int = 40
    max_patch_bytes: int = 1_048_576

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

        write_mode = str(payload.get("write_mode", "read_only")).strip()
        if write_mode not in ALLOWED_WRITE_MODES:
            raise ContractError(f"unknown write_mode: {write_mode!r}")
        allowed_write_paths = tuple(
            _clean_relative_path(value, "allowed_write_paths")
            for value in payload.get("allowed_write_paths", ())
        )
        if write_mode != "read_only" and not allowed_write_paths:
            raise ContractError("write tasks require allowed_write_paths")
        if write_mode == "read_only" and allowed_write_paths:
            raise ContractError("read_only tasks must not declare allowed_write_paths")

        gate_profile = str(payload.get("gate_profile", "profile")).strip()
        if not gate_profile or any(value in gate_profile for value in ("/", "\\", "..")):
            raise ContractError("gate_profile must be a simple profile name")
        max_repairs = int(payload.get("max_repairs", 2))
        if not 0 <= max_repairs <= 3:
            raise ContractError("max_repairs must be between 0 and 3")
        max_changed_files = int(payload.get("max_changed_files", 40))
        if not 1 <= max_changed_files <= 500:
            raise ContractError("max_changed_files must be between 1 and 500")
        max_patch_bytes = int(payload.get("max_patch_bytes", 1_048_576))
        if not 1_024 <= max_patch_bytes <= 20_971_520:
            raise ContractError("max_patch_bytes must be between 1024 and 20971520")

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
            write_mode=write_mode,
            allowed_write_paths=allowed_write_paths,
            gate_profile=gate_profile,
            max_repairs=max_repairs,
            max_changed_files=max_changed_files,
            max_patch_bytes=max_patch_bytes,
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
    trust_tier: str = "official_subscription"
    capabilities: tuple[str, ...] = ("read",)
    execution_profile: str = "read_only"
    base_url: str | None = None


@dataclass(frozen=True)
class RoleConfig:
    name: str
    models: tuple[str, ...]
    reviewers: tuple[str, ...]
    reasoning_effort: str = "high"
    reviewer_reasoning_effort: str = "high"


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
    public_repository: bool = False


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
    write_mode: str = "read_only"
    trust_tier: str | None = None
    execution_profile: str = "read_only"
    reasoning_effort: str | None = None
    reviewer_reasoning_effort: str | None = None

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
    changed_files: int = 0
    patch_sha256: str | None = None
    repairs: int = 0
    gates_passed: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink() or not path.is_dir():
        raise ContractError(f"unsafe directory: {path}")
    os.chmod(path, 0o700)
