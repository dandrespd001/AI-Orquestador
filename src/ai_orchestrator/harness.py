"""Pinned, loopback-only DeepSeek Harness launch policy."""

from __future__ import annotations

import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ai_orchestrator.broker import default_socket_path
from ai_orchestrator.models import ContractError, ensure_private_directory
from ai_orchestrator.telemetry import default_state_root

HARNESS_PACKAGE = "@deepseek-ai/dsh"
HARNESS_VERSION = "0.1.0-rc.7"
HARNESS_HOST = "127.0.0.1"
HARNESS_PORT = 3080
HARNESS_ADMIN_PROFILE = "harness-admin"


@dataclass(frozen=True)
class HarnessLaunchSpec:
    command: tuple[str, ...]
    environment: dict[str, str]
    host: str
    port: int
    profile: str
    broker_socket: str
    community_code_loaded: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def harness_launch_spec(
    *,
    socket_path: Path | None = None,
    state_root: Path | None = None,
    port: int = HARNESS_PORT,
) -> HarnessLaunchSpec:
    if not 1024 <= port <= 65_535:
        raise ContractError("Harness port must be between 1024 and 65535")
    broker = (socket_path or default_socket_path()).expanduser().resolve()
    root = (state_root or default_state_root()).expanduser().resolve()
    dsh_home = root / "harness" / "admin"
    ensure_private_directory(dsh_home)
    allowed = {"HOME", "USER", "LOGNAME", "PATH", "LANG", "LC_ALL", "TERM", "NO_COLOR"}
    environment = {name: value for name, value in os.environ.items() if name in allowed}
    environment.update(
        {
            "HOST": HARNESS_HOST,
            "DSH_HOST": HARNESS_HOST,
            "DSH_PORT": str(port),
            "DSH_HOME": str(dsh_home),
            "AI_ORCHESTRATOR_SOCKET": str(broker),
            "NO_COLOR": "1",
        }
    )
    environment.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    return HarnessLaunchSpec(
        command=(
            "npx",
            "--offline",
            f"{HARNESS_PACKAGE}@{HARNESS_VERSION}",
            "web",
            "--host",
            HARNESS_HOST,
            "--port",
            str(port),
            "--no-open",
        ),
        environment=environment,
        host=HARNESS_HOST,
        port=port,
        profile=HARNESS_ADMIN_PROFILE,
        broker_socket=str(broker),
        community_code_loaded=False,
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
    if f"{HARNESS_PACKAGE}@{HARNESS_VERSION}" not in spec.command:
        errors.append("Harness package is not pinned to the approved version")
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
        "version": HARNESS_VERSION,
        "errors": errors,
    }
