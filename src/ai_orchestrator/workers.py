"""Least-privilege execution plans for approved extension workers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ai_orchestrator.models import ContractError, ExtensionManifest

SHELL_EXECUTABLES = {"sh", "bash", "zsh", "fish", "cmd", "powershell", "pwsh"}


@dataclass(frozen=True)
class WorkerLimits:
    timeout_seconds: int = 120
    max_output_bytes: int = 1_000_000
    max_files: int = 1_000
    max_processes: int = 32
    memory_mb: int = 512
    cpus: float = 1.0
    concurrency: int = 1

    def __post_init__(self) -> None:
        if not 1 <= self.timeout_seconds <= 900:
            raise ContractError("worker timeout must be between 1 and 900 seconds")
        if not 1_024 <= self.max_output_bytes <= 10_000_000:
            raise ContractError("worker output limit is invalid")
        if not 1 <= self.max_files <= 10_000 or not 1 <= self.max_processes <= 256:
            raise ContractError("worker file/process limit is invalid")
        if not 64 <= self.memory_mb <= 4096 or not 0.1 <= self.cpus <= 4.0:
            raise ContractError("worker resource limit is invalid")
        if not 1 <= self.concurrency <= 4:
            raise ContractError("worker concurrency limit is invalid")


@dataclass(frozen=True)
class WorkerPlan:
    extension_id: str
    runtime: str
    command: tuple[str, ...]
    permissions: tuple[str, ...]
    allowed_domains: tuple[str, ...]
    limits: WorkerLimits
    network_requested: bool
    network_enabled: bool
    egress_proxy_required: bool
    broker_socket_mounted: bool = False
    host_home_mounted: bool = False
    provider_secrets_readable: bool = False
    shell_exception: bool = False
    ephemeral_credentials_only: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_worker_plan(
    manifest: ExtensionManifest,
    payload: Path,
    task_directory: Path,
    *,
    risk: str,
    limits: WorkerLimits | None = None,
    shell_exception_approved: bool = False,
    worker_image: str | None = None,
) -> WorkerPlan:
    payload_input = payload.expanduser()
    task_input = task_directory.expanduser()
    if payload_input.is_symlink() or not payload_input.is_dir():
        raise ContractError("worker payload must be an approved regular directory")
    if task_input.is_symlink() or not task_input.is_dir():
        raise ContractError("worker task directory must be a regular directory")
    payload = payload_input.resolve()
    task_directory = task_input.resolve()
    if not manifest.entrypoint:
        raise ContractError("extension worker requires a fixed argv entrypoint")
    executable = Path(manifest.entrypoint[0]).name.lower()
    requests_shell = executable in SHELL_EXECUTABLES or "shell" in manifest.permissions
    if requests_shell and not shell_exception_approved:
        raise ContractError(
            "arbitrary shell commands require a specific human exception"
        )
    runtime = (
        "docker" if risk == "high" or manifest.requires_host_access else "bubblewrap"
    )
    network_enabled = "network" in manifest.permissions
    if network_enabled and not manifest.remote_domains:
        raise ContractError("networked workers require an explicit domain allowlist")
    worker_limits = limits or WorkerLimits()
    if runtime == "docker":
        if not worker_image or "@sha256:" not in worker_image:
            raise ContractError(
                "high-risk worker requires an immutable container image digest"
            )
        image_digest = worker_image.rsplit("@sha256:", 1)[-1]
        if len(image_digest) != 64 or any(
            ch not in "0123456789abcdef" for ch in image_digest.lower()
        ):
            raise ContractError("worker container image has an invalid digest")
        command = (
            "docker",
            "run",
            "--rm",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            f"--pids-limit={worker_limits.max_processes}",
            f"--memory={worker_limits.memory_mb}m",
            f"--cpus={worker_limits.cpus}",
            "--network=none",
            "--mount",
            f"type=bind,src={payload},dst=/extension,readonly",
            "--mount",
            f"type=bind,src={task_directory},dst=/work",
            "--workdir=/work",
            worker_image,
            *manifest.entrypoint,
        )
    else:
        command = (
            "bwrap",
            "--die-with-parent",
            "--new-session",
            "--unshare-all",
            "--unshare-net",
            "--ro-bind",
            str(payload),
            "/extension",
            "--bind",
            str(task_directory),
            "/work",
            "--tmpfs",
            "/tmp",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--chdir",
            "/work",
            "--clearenv",
            "--setenv",
            "PATH",
            "/usr/bin:/bin",
            *manifest.entrypoint,
        )
    # Domain-restricted networking is provided by a separate egress proxy.  A
    # raw network namespace is never granted by this execution plan.
    return WorkerPlan(
        extension_id=manifest.id,
        runtime=runtime,
        command=command,
        permissions=manifest.permissions,
        allowed_domains=manifest.remote_domains,
        limits=worker_limits,
        network_requested=network_enabled,
        # Raw host networking is never enabled here. Networked extensions are
        # held until a domain-filtering egress proxy is attached by the runner.
        network_enabled=False,
        egress_proxy_required=network_enabled,
        shell_exception=requests_shell and shell_exception_approved,
    )


def validate_worker_plan(plan: WorkerPlan) -> dict[str, Any]:
    errors: list[str] = []
    if plan.broker_socket_mounted:
        errors.append("administrative broker socket is mounted")
    if plan.host_home_mounted:
        errors.append("operator home is mounted")
    if plan.provider_secrets_readable:
        errors.append("provider secrets are readable")
    if plan.network_requested and not plan.allowed_domains:
        errors.append("network has no domain allowlist")
    if plan.network_enabled:
        errors.append("worker has raw host networking")
    if plan.network_requested != plan.egress_proxy_required:
        errors.append("network request is not held behind the egress proxy")
    if plan.command[0] not in {"bwrap", "docker"}:
        errors.append("worker has no approved isolation runtime")
    return {"schema_version": 1, "ok": not errors, "errors": errors}
