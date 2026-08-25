from __future__ import annotations

import hashlib
import json
import os
import urllib.parse
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
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
    # Edita el arbol de trabajo como native_writer —y por eso puede iterar,
    # hacer fase roja y tocar ficheros grandes, cosa que un escritor por parche
    # no puede— pero NUNCA auto-aplica al repo de origen: termina siempre en
    # PATCH_READY para que lo revise el Arquitecto. Los dos privilegios venian
    # empaquetados en native_writer; esto los separa.
    "native_patch_writer",
    "patch_api_writer",
    "kimi_acp_writer",
    "read_only",
}
# Perfiles que editan el arbol de trabajo directamente. Solo estos reciben un
# montaje escribible. Que un perfil este aqui no implica que pueda auto-aplicar
# al repo de origen: eso lo decide `native_writer` a solas, en adapters.py.
NATIVE_WRITER_PROFILES = frozenset({"native_writer", "native_patch_writer"})
ALLOWED_MODEL_CAPABILITIES = {"read", "write", "tools"}
ALLOWED_REASONING_EFFORTS = {"low", "medium", "high", "xhigh", "max"}

PROVIDER_PROTOCOLS = {
    "official_cli",
    "oauth_cli",
    "openai_chat",
    "openai_responses",
    "anthropic_messages",
    "local_gateway",
}
EXTENSION_KINDS = {"plugin", "skill", "mcp"}
EXTENSION_TRUST_LEVELS = {
    "official",
    "provider_maintained",
    "community_verified",
    "community_experimental",
    "private_local",
}
EXTENSION_SOURCE_TYPES = {
    "github",
    "private_git",
    "local",
    "npm",
    "pypi",
    "mcp_stdio",
    "mcp_http",
}


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
    # Digest-only representation for MCP/CLI submissions whose prompt must
    # never be materialized in the repository or state directory.
    inline_prompt_sha256: str | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> TaskSpec:
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
            if not resolved_prompt.is_file() or not resolved_prompt.is_relative_to(
                repo
            ):
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
        if not gate_profile or any(
            value in gate_profile for value in ("/", "\\", "..")
        ):
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
    def read(cls, path: str | Path) -> TaskSpec:
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
        if self.inline_prompt_sha256 is not None:
            value = self.inline_prompt_sha256.lower()
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ContractError("inline prompt hash must be a hexadecimal SHA-256")
            if not self.input_paths and self.prompt_file is None:
                return value
        digest = hashlib.sha256()
        if self.inline_prompt_sha256 is not None:
            digest.update(b"inline-prompt-sha256\0")
            digest.update(self.inline_prompt_sha256.encode("ascii"))
            digest.update(b"\0")
        root = Path(self.repo).resolve()
        selected = ([self.prompt_file] if self.prompt_file else []) + list(
            self.input_paths
        )

        def update_path(path: Path) -> None:
            relative_path = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise ContractError(
                    f"context path must not be a symlink: {relative_path}"
                )
            metadata = path.stat()
            kind = "directory" if path.is_dir() else "file"
            digest.update(relative_path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(kind.encode("ascii"))
            digest.update(b"\0")
            digest.update(str(metadata.st_mode & 0o7777).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(metadata.st_size).encode("ascii"))
            digest.update(b"\0")
            if path.is_file():
                with path.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        digest.update(chunk)
            digest.update(b"\0")

        for relative in sorted(set(selected)):
            if relative is None:
                continue
            unresolved = root / relative
            if unresolved.is_symlink():
                raise ContractError(f"context path must not be a symlink: {relative}")
            path = unresolved.resolve()
            if not path.exists() or not path.is_relative_to(root):
                raise ContractError(f"context path is unavailable: {relative}")
            update_path(path)
            if path.is_dir():
                for child in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
                    update_path(child)
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
    quality_score: float = 0.5
    included_quota: bool = True
    marginal_cost_per_million: float = 0.0
    context_window: int = 120_000
    cache_supported: bool = False
    observed_latency_ms: int | None = None
    available_until: str | None = None
    # Que la ruta consuma o no cuota es independiente de si el proveedor es
    # oficial: "free" no gasta nada, "included" gasta cuota de suscripcion,
    # "paid" cobra por uso. Separarlo de trust_tier permite preferir lo
    # gratuito sin bajar el liston de confianza.
    cost_tier: str = "included"


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
    allow_paid_routes: bool = False
    max_marginal_cost_usd: float = 0.0
    capability_packages: tuple[str, ...] = ()
    extensions: tuple[str, ...] = ()
    architect_model: str | None = None


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
    max_concurrency: int = 1

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
    # Detalle acotado del fallo, SOLO de errores que produce el orquestador
    # (WriterError): la ruta que violo el alcance, el limite que se paso. Nunca
    # lleva texto del modelo, asi que no retiene contenido. Sin esto un
    # WRITE_SCOPE_VIOLATION obliga a gastar otra pasada entera para saber que
    # fichero fue.
    error_detail: str | None = None
    changed_files: int = 0
    patch_sha256: str | None = None
    repairs: int = 0
    gates_passed: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _simple_identifier(value: object, field_name: str) -> str:
    text = str(value).strip()
    if not text or any(marker in text for marker in ("/", "\\", "..", "\x00")):
        raise ContractError(f"{field_name} must be a simple identifier")
    return text


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ContractError(f"{field_name} must be an array")
    result = tuple(str(item).strip() for item in value)
    if any(not item or "\x00" in item for item in result):
        raise ContractError(f"{field_name} contains an invalid value")
    return result


def _contains_secret_field(value: object) -> bool:
    """Reject credential-shaped keys at any depth in public contracts."""

    exact = {
        "api_key",
        "apikey",
        "authorization",
        "password",
        "secret",
        "token",
        "access_token",
        "auth_token",
        "client_secret",
        "refresh_token",
    }
    if isinstance(value, dict):
        for raw_key, nested in value.items():
            key = str(raw_key).strip().lower()
            if key in exact or key.endswith(
                ("_api_key", "_password", "_secret", "_token")
            ):
                return True
            if _contains_secret_field(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_secret_field(item) for item in value)
    return False


@dataclass(frozen=True)
class ProviderSpec:
    """Secret-free provider contract shared by the UI, router and MCP tools."""

    id: str
    endpoint: str
    protocol: str
    models: tuple[str, ...]
    authentication: str | dict[str, Any]
    quota: str | dict[str, Any]
    trust: str
    capabilities: tuple[str, ...]
    canary: str | dict[str, Any] | None
    incremental_spend_allowed: bool = False
    content_retention_allowed: bool = False
    enabled: bool = True

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> ProviderSpec:
        provider_id = _simple_identifier(payload.get("id", ""), "provider id")
        protocol = str(payload.get("protocol", "")).strip()
        if protocol not in PROVIDER_PROTOCOLS:
            raise ContractError(f"unsupported provider protocol: {protocol!r}")
        endpoint = str(payload.get("endpoint", "")).strip()
        if not endpoint or "\x00" in endpoint:
            raise ContractError("provider endpoint is required")
        parsed_endpoint = urllib.parse.urlsplit(endpoint)
        if parsed_endpoint.scheme not in {"http", "https", "cli"}:
            raise ContractError("provider endpoint uses an unsupported scheme")
        if parsed_endpoint.scheme in {"http", "https"}:
            if (
                not parsed_endpoint.hostname
                or parsed_endpoint.username
                or parsed_endpoint.password
            ):
                raise ContractError("provider endpoint must not contain credentials")
            if parsed_endpoint.fragment:
                raise ContractError("provider endpoint must not contain a fragment")
            if parsed_endpoint.scheme == "http" and parsed_endpoint.hostname not in {
                "127.0.0.1",
                "localhost",
                "::1",
            }:
                raise ContractError("plaintext provider endpoints must be loopback")
        models = _string_tuple(payload.get("models", ()), "provider models")
        if not models:
            raise ContractError("provider must declare at least one model")
        capabilities = _string_tuple(
            payload.get("capabilities", ("read",)), "provider capabilities"
        )
        authentication = payload.get("authentication", "none")
        quota = payload.get("quota", "unknown")
        if isinstance(authentication, dict):
            authentication = dict(authentication)
        else:
            authentication = str(authentication)
        quota = dict(quota) if isinstance(quota, dict) else str(quota)
        canary = payload.get("canary")
        if isinstance(canary, dict):
            canary = dict(canary)
        elif canary is not None:
            canary = str(canary)
        if _contains_secret_field(
            {"authentication": authentication, "quota": quota, "canary": canary}
        ):
            raise ContractError("ProviderSpec must not contain secrets")
        return cls(
            id=provider_id,
            endpoint=endpoint,
            protocol=protocol,
            models=models,
            authentication=authentication,
            quota=quota,
            trust=str(payload.get("trust", "official_subscription")),
            capabilities=capabilities,
            canary=canary,
            incremental_spend_allowed=bool(
                payload.get("incremental_spend_allowed", False)
            ),
            content_retention_allowed=bool(
                payload.get("content_retention_allowed", False)
            ),
            enabled=bool(payload.get("enabled", True)),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AuthProfileStatus:
    """Authentication metadata. It intentionally has no secret-valued field."""

    provider: str
    profile: str
    status: str
    source: str
    managed_externally: bool = False
    human_action_required: bool = False
    expires_at: str | None = None
    last_rotated_at: str | None = None
    detail_code: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoutePlan:
    """Auditable route simulation. Alternatives are never automatic fallback."""

    task_id: str
    profile: str
    selected_model: str | None
    alternatives: tuple[str, ...]
    scores: dict[str, float]
    quality_price_score: float | None
    budget: dict[str, Any]
    permissions: tuple[str, ...]
    approval_required: bool
    reason: str
    provider: str | None = None
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: int = 1
    fallback_allowed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UsageRecord:
    """Metadata-only usage receipt; request/response bodies are not representable."""

    provider: str
    model: str
    outcome: str
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    credits_used: float | None = None
    estimated_cost_usd: float | None = None
    cache_hit: bool = False
    provenance_verified: bool = False
    route_plan_id: str | None = None
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    schema_version: int = 1

    def __post_init__(self) -> None:
        numeric = (
            self.latency_ms,
            self.input_tokens,
            self.output_tokens,
            self.cached_tokens,
            self.credits_used,
            self.estimated_cost_usd,
        )
        if any(value is not None and value < 0 for value in numeric):
            raise ContractError("usage counters must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExtensionManifest:
    """Pinned extension identity and its declared least-privilege envelope."""

    id: str
    kind: str
    source: str
    source_type: str
    version: str | None
    commit: str | None
    sha256: str
    license: str
    permissions: tuple[str, ...]
    compatibility: tuple[str, ...]
    profiles: tuple[str, ...]
    trust: str
    entrypoint: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    remote_domains: tuple[str, ...] = ()
    install_scripts: tuple[str, ...] = ()
    requires_host_access: bool = False
    enabled: bool = True

    @property
    def extension_class(self) -> str:
        return self.kind

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> ExtensionManifest:
        extension_id = _simple_identifier(payload.get("id", ""), "extension id")
        kind = str(payload.get("kind", payload.get("class", ""))).strip().lower()
        source_hint: str | None = None
        if kind == "mcp_local":
            kind, source_hint = "mcp", "mcp_stdio"
        elif kind == "mcp_remote":
            kind, source_hint = "mcp", "mcp_http"
        if kind not in EXTENSION_KINDS:
            raise ContractError(f"unsupported extension class: {kind!r}")
        source_type = (
            str(payload.get("source_type", source_hint or "local")).strip().lower()
        )
        source_type = {
            "git": "private_git",
            "git_private": "private_git",
            "path": "local",
            "stdio": "mcp_stdio",
            "http": "mcp_http",
        }.get(source_type, source_type)
        if source_type not in EXTENSION_SOURCE_TYPES:
            raise ContractError(f"unsupported extension source: {source_type!r}")
        trust = str(payload.get("trust", "community_experimental")).strip().lower()
        trust = {
            "community": "community_experimental",
            "private": "private_local",
            "local": "private_local",
            "provider": "provider_maintained",
        }.get(trust, trust)
        if trust not in EXTENSION_TRUST_LEVELS:
            raise ContractError(f"unsupported extension trust level: {trust!r}")
        digest = str(payload.get("sha256", "")).strip().lower()
        if digest and (
            len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest)
        ):
            raise ContractError("extension sha256 must be a hexadecimal SHA-256")
        source = str(payload.get("source", "")).strip()
        if not source or "\x00" in source:
            raise ContractError("extension source is required")
        entrypoint_value = payload.get("entrypoint", ())
        if isinstance(entrypoint_value, str):
            entrypoint = (entrypoint_value,)
        else:
            entrypoint = _string_tuple(entrypoint_value, "extension entrypoint")
        raw_compatibility = payload.get("compatibility", ())
        if isinstance(raw_compatibility, dict):
            compatibility = tuple(
                f"{key}:{value}" for key, value in sorted(raw_compatibility.items())
            )
        else:
            compatibility = _string_tuple(raw_compatibility, "extension compatibility")
        return cls(
            id=extension_id,
            kind=kind,
            source=source,
            source_type=source_type,
            version=(str(payload["version"]) if payload.get("version") else None),
            commit=(str(payload["commit"]) if payload.get("commit") else None),
            sha256=digest,
            license=str(payload.get("license", "UNKNOWN")),
            permissions=_string_tuple(
                payload.get("permissions", ()), "extension permissions"
            ),
            compatibility=compatibility,
            profiles=_string_tuple(payload.get("profiles", ()), "extension profiles"),
            trust=trust,
            entrypoint=entrypoint,
            tools=_string_tuple(payload.get("tools", ()), "extension tools"),
            remote_domains=_string_tuple(
                payload.get("remote_domains", ()), "extension remote domains"
            ),
            install_scripts=_string_tuple(
                payload.get("install_scripts", ()), "extension install scripts"
            ),
            requires_host_access=bool(payload.get("requires_host_access", False)),
            enabled=bool(payload.get("enabled", True)),
        )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["class"] = self.kind
        return payload


def ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink() or not path.is_dir():
        raise ContractError(f"unsafe directory: {path}")
    os.chmod(path, 0o700)
