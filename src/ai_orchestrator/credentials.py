"""Read exact runtime credentials without copying them into client config."""

from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import uuid
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path

from ai_orchestrator.config import config_root
from ai_orchestrator.models import (
    AuthProfileStatus,
    ContractError,
    ensure_private_directory,
)

MAX_CREDENTIAL_BYTES = 16_384
MAX_ENCRYPTED_CREDENTIAL_BYTES = 65_536

OFFICIAL_OAUTH_STORES = {
    "openai": Path.home() / ".codex" / "auth.json",
    "anthropic": Path.home() / ".claude",
    "moonshot": Path.home() / ".kimi-code" / "credentials",
    "abacus": Path.home() / ".abacusai",
}


def _private_regular_file(
    path: Path,
    *,
    require_owner: bool = True,
    max_bytes: int = MAX_CREDENTIAL_BYTES,
) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        return False
    if require_owner and metadata.st_uid != os.getuid():
        return False
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        return False
    return metadata.st_size <= max_bytes


def _valid_value(value: str | None) -> str | None:
    if value is None:
        return None
    candidate = value.strip("\r\n")
    if 8 <= len(candidate) <= MAX_CREDENTIAL_BYTES and "\x00" not in candidate:
        return candidate
    return None


def resolve_credential(
    name: str,
    *,
    env_names: Iterable[str] = (),
    environment: dict[str, str] | None = None,
    root: Path | None = None,
    timeout_seconds: float = 3.0,
) -> str | None:
    """Resolve one credential without a shell or plaintext persistence.

    Precedence is systemd's runtime credential directory, explicit environment
    variables, then a user-bound ``systemd-creds`` encrypted file.
    """

    _credential_name(name)
    source = os.environ if environment is None else environment
    credential_directory = source.get("CREDENTIALS_DIRECTORY")
    if credential_directory:
        runtime_path = Path(credential_directory) / name
        if _private_regular_file(runtime_path, require_owner=False):
            value = _valid_value(runtime_path.read_text(encoding="utf-8"))
            if value:
                return value
    for env_name in env_names:
        value = _valid_value(source.get(env_name))
        if value:
            return value

    encrypted = (root or config_root()) / "credentials" / f"{name}.cred"
    if not _private_regular_file(encrypted):
        return None
    try:
        completed = subprocess.run(
            [
                "systemd-creds",
                "decrypt",
                "--user",
                f"--name={name}",
                str(encrypted),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout_seconds,
            check=False,
            env={"PATH": "/usr/bin:/bin"},
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if completed.returncode != 0 or len(completed.stdout) > MAX_CREDENTIAL_BYTES:
        return None
    try:
        return _valid_value(completed.stdout.decode("utf-8"))
    except UnicodeDecodeError:
        return None


def minimax_runtime_key(environment: dict[str, str] | None = None) -> str | None:
    return resolve_credential(
        "minimax-api-key",
        env_names=("MINIMAX_API_KEY",),
        environment=environment,
    )


def private_dotenv_value(path: Path, key: str) -> str | None:
    """Return one literal dotenv value from a private, regular file."""

    if not _private_regular_file(path):
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, raw_value = line.partition("=")
        if separator and name.strip() == key:
            value = raw_value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            if 8 <= len(value) <= 16_384 and "\x00" not in value:
                return value
            return None
    return None


def omniroute_runtime_key(environment: dict[str, str] | None = None) -> str | None:
    source = os.environ if environment is None else environment
    resolved = resolve_credential(
        "omniroute_api_key",
        env_names=("OMNIROUTE_API_KEY",),
        environment=dict(source),
    )
    if resolved:
        return resolved
    home = source.get("HOME")
    if not home:
        return None
    return private_dotenv_value(Path(home) / ".omniroute" / ".env", "OMNIROUTE_API_KEY")


def _credential_name(value: str) -> str:
    if (
        not 1 <= len(value) <= 128
        or any(marker in value for marker in ("/", "\\", "..", "\x00"))
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for character in value
        )
    ):
        raise ContractError("credential name must be a simple identifier")
    return value


def _provider_name(value: str) -> str:
    if not 1 <= len(value) <= 64 or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for character in value
    ):
        raise ContractError("provider must be a simple identifier")
    return value


def _default_encryptor(name: str, value: bytes, output: Path) -> None:
    try:
        completed = subprocess.run(
            [
                "systemd-creds",
                "encrypt",
                "--user",
                f"--name={name}",
                "-",
                str(output),
            ],
            input=value,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
            env={"PATH": "/usr/bin:/bin"},
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        raise ContractError("systemd-creds encryption failed") from exc
    if completed.returncode != 0:
        raise ContractError("systemd-creds encryption failed")


class CredentialStore:
    """Write-only encrypted credential rotation with atomic replacement."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        encryptor: Callable[[str, bytes, Path], None] = _default_encryptor,
    ) -> None:
        self.root = (root or config_root()).expanduser().resolve()
        self.directory = self.root / "credentials"
        self._encryptor = encryptor

    def status(
        self,
        provider: str,
        *,
        profile: str = "default",
        credential_name: str | None = None,
        env_names: Iterable[str] = (),
        environment: dict[str, str] | None = None,
        official_store: Path | None = None,
    ) -> AuthProfileStatus:
        source = os.environ if environment is None else environment
        name = _credential_name(credential_name) if credential_name else None
        runtime_directory = source.get("CREDENTIALS_DIRECTORY")
        if name and runtime_directory:
            runtime = Path(runtime_directory) / name
            if _private_regular_file(runtime, require_owner=False):
                return AuthProfileStatus(
                    provider, profile, "available", "systemd_runtime"
                )
        for env_name in env_names:
            if _valid_value(source.get(env_name)):
                return AuthProfileStatus(
                    provider, profile, "available", "process_environment"
                )
        if name:
            encrypted = self.directory / f"{name}.cred"
            if _private_regular_file(
                encrypted, max_bytes=MAX_ENCRYPTED_CREDENTIAL_BYTES
            ):
                rotated = datetime.fromtimestamp(
                    encrypted.stat().st_mtime, tz=UTC
                ).isoformat()
                return AuthProfileStatus(
                    provider,
                    profile,
                    "available",
                    "systemd_encrypted",
                    last_rotated_at=rotated,
                )
        store = official_store or OFFICIAL_OAUTH_STORES.get(provider)
        if store is not None and store.exists() and not store.is_symlink():
            return AuthProfileStatus(
                provider,
                profile,
                "available",
                "official_oauth_store",
                managed_externally=True,
            )
        return AuthProfileStatus(
            provider,
            profile,
            "unavailable",
            "none",
            managed_externally=store is not None,
            human_action_required=True,
            detail_code="AUTHENTICATION_REQUIRED",
        )

    def store(
        self,
        name: str,
        value: str | bytes,
        *,
        provider: str,
        profile: str = "default",
        human_approved: bool = False,
    ) -> dict[str, object]:
        """Encrypt and atomically install a credential; never return its value."""

        name = _credential_name(name)
        provider = _provider_name(provider)
        if not human_approved:
            raise ContractError("credential writes require an explicit human approval")
        raw = value.encode("utf-8") if isinstance(value, str) else bytes(value)
        if not 8 <= len(raw) <= MAX_CREDENTIAL_BYTES or b"\x00" in raw:
            raise ContractError("credential value has an invalid size or encoding")
        ensure_private_directory(self.root)
        ensure_private_directory(self.directory)
        destination = self.directory / f"{name}.cred"
        if destination.is_symlink():
            raise ContractError("refusing to replace a credential symlink")
        generation = str(uuid.uuid4())
        staging = Path(
            tempfile.mkdtemp(prefix=f".{name}.{generation}.", dir=self.directory)
        )
        os.chmod(staging, 0o700)
        encrypted = staging / "encrypted.cred"
        try:
            self._encryptor(name, raw, encrypted)
            if encrypted.is_symlink() or not encrypted.is_file():
                raise ContractError("credential encryptor produced no regular file")
            metadata = encrypted.stat()
            if not 1 <= metadata.st_size <= MAX_ENCRYPTED_CREDENTIAL_BYTES:
                raise ContractError("encrypted credential has an invalid size")
            os.chmod(encrypted, 0o600)
            with encrypted.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(encrypted, destination)
            os.chmod(destination, 0o600)
            directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            try:
                encrypted.unlink()
            except FileNotFoundError:
                pass
            try:
                staging.rmdir()
            except OSError:
                pass
            raw = b""  # Drop our reference promptly; no plaintext file exists.
        rotated_at = datetime.now(UTC).isoformat()
        return {
            "schema_version": 1,
            "provider": provider,
            "profile": profile,
            "credential": name,
            "status": "stored",
            "source": "systemd_encrypted",
            "generation": generation,
            "rotated_at": rotated_at,
            "secret_returned": False,
        }

    def rotate(self, *args: object, **kwargs: object) -> dict[str, object]:
        return self.store(*args, **kwargs)  # type: ignore[arg-type]


def authentication_inventory(
    root: Path | None = None,
    *,
    profile: str = "default",
) -> list[dict[str, object]]:
    store = CredentialStore(root)
    statuses = (
        store.status("openai", profile=profile),
        store.status("anthropic", profile=profile),
        store.status("moonshot", profile=profile),
        store.status("abacus", profile=profile),
        store.status(
            "minimax",
            profile=profile,
            credential_name="minimax-api-key",
            env_names=("MINIMAX_API_KEY",),
        ),
        store.status(
            "omniroute",
            profile=profile,
            credential_name="omniroute_api_key",
            env_names=("OMNIROUTE_API_KEY",),
        ),
    )
    return [status.as_dict() for status in statuses]
