from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ai_orchestrator.models import ensure_private_directory


class ArchiveError(RuntimeError):
    pass


def _regular_files(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    for raw in paths:
        path = raw.expanduser().resolve()
        metadata = path.lstat()
        if path.is_symlink() or not path.is_file() or metadata.st_nlink > 1:
            raise ArchiveError(f"archive input must be a private regular file: {path}")
        if metadata.st_uid != os.getuid():
            raise ArchiveError(f"archive input is not user-owned: {path}")
        result.append(path)
    if not result:
        raise ArchiveError("at least one file is required")
    return sorted(set(result))


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_encrypted_archive(
    paths: Iterable[Path],
    *,
    recipient: str,
    output_root: Path | None = None,
) -> dict[str, Any]:
    files = _regular_files(paths)
    recipient_check = subprocess.run(
        ["gpg", "--batch", "--list-keys", "--", recipient],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if recipient_check.returncode != 0:
        raise ArchiveError("GPG recipient is not available in the local public keyring")

    root = output_root or Path.home() / ".local" / "share" / "ai-orchestrator" / "archive"
    ensure_private_directory(root.parent)
    ensure_private_directory(root)
    archive_id = uuid.uuid4().hex
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    final_path = root / f"sessions-{timestamp}-{archive_id}.tar.zst.gpg"
    external_manifest = root / f"sessions-{timestamp}-{archive_id}.manifest.json"

    manifest = {
        "schema_version": 1,
        "archive_id": archive_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "retention_days": 90,
        "files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _digest(path)}
            for path in files
        ],
    }

    temp_root = Path(tempfile.mkdtemp(prefix="ai-orchestrator-archive-", dir="/tmp"))
    os.chmod(temp_root, 0o700)
    list_path = temp_root / "files.list"
    manifest_path = temp_root / "manifest.json"
    temporary_output = temp_root / "archive.gpg"
    try:
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
        os.chmod(manifest_path, 0o600)
        archive_members = [str(path).lstrip("/").encode("utf-8") for path in files]
        archive_members.append(str(manifest_path).lstrip("/").encode("utf-8"))
        list_path.write_bytes(b"\0".join(archive_members) + b"\0")
        os.chmod(list_path, 0o600)

        tar = subprocess.Popen(
            [
                "tar",
                "--create",
                "--file=-",
                "--directory=/",
                "--null",
                "--verbatim-files-from",
                f"--files-from={list_path}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"PATH": "/usr/bin:/bin", "LANG": "C"},
        )
        assert tar.stdout is not None
        zstd = subprocess.Popen(
            ["zstd", "-q", "-6", "-c"],
            stdin=tar.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"PATH": "/usr/bin:/bin", "LANG": "C"},
        )
        tar.stdout.close()
        assert zstd.stdout is not None
        gpg = subprocess.run(
            [
                "gpg",
                "--batch",
                "--yes",
                "--trust-model",
                "always",
                "--recipient",
                recipient,
                "--encrypt",
                "--output",
                str(temporary_output),
            ],
            stdin=zstd.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
        zstd.stdout.close()
        tar_stderr = tar.communicate()[1]
        zstd_stderr = zstd.communicate()[1]
        if tar.returncode != 0 or zstd.returncode != 0 or gpg.returncode != 0:
            raise ArchiveError(
                "archive pipeline failed: "
                + ",".join(
                    code
                    for code, failed in (
                        ("tar", tar.returncode != 0),
                        ("zstd", zstd.returncode != 0),
                        ("gpg", gpg.returncode != 0),
                    )
                    if failed
                )
            )
        # Diagnostics may contain paths or provider details. Successful tools
        # are allowed to emit them, but they are deliberately never returned.
        del tar_stderr, zstd_stderr
        if not temporary_output.is_file() or temporary_output.stat().st_size == 0:
            raise ArchiveError("encrypted archive was not produced")

        packet_check = subprocess.run(
            ["gpg", "--batch", "--list-packets", str(temporary_output)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if packet_check.returncode != 0:
            raise ArchiveError("encrypted archive failed packet validation")

        os.chmod(temporary_output, 0o600)
        os.replace(temporary_output, final_path)
        os.chmod(final_path, 0o600)
        external_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "archive_id": archive_id,
                    "encrypted_path": str(final_path),
                    "encrypted_bytes": final_path.stat().st_size,
                    "encrypted_sha256": _digest(final_path),
                    "source_count": len(files),
                    "retention_days": 90,
                    "bodies_in_manifest": False,
                },
                sort_keys=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        os.chmod(external_manifest, 0o600)
        return json.loads(external_manifest.read_text(encoding="utf-8"))
    finally:
        for child in (list_path, manifest_path, temporary_output):
            try:
                child.unlink()
            except FileNotFoundError:
                pass
        try:
            temp_root.rmdir()
        except OSError:
            pass
