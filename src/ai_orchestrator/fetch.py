"""Traer un recurso externo a una zona de staging, con procedencia.

El orquestador no dejaba descargar nada: los tres adaptadores bloquean la web y
ningun camino escribe fuera del worktree. Eso es correcto para un modelo, que no
debe elegir por su cuenta que entra en el disco, pero deja al Director sin forma
de bajar un dataset, una referencia o un asset por el camino auditado.

Este modulo es esa via, y es deliberadamente aburrida: la descarga la pide una
persona, nunca un modelo; el fichero aterriza en una zona de staging privada y
jamas dentro de un repositorio; y cada descarga deja un recibo con origen,
tamano, tipo declarado y SHA-256, para que decidir despues que se promueve al
repositorio sea una decision informada y no un acto de fe.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from ai_orchestrator.models import ContractError, ensure_private_directory

# Un recurso mas grande que esto casi nunca es lo que se queria, y llenar el
# disco en silencio es peor que fallar.
MAX_FETCH_BYTES = 256 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 120.0
ALLOWED_SCHEMES = frozenset({"http", "https"})
CHUNK_BYTES = 64 * 1024


def default_staging_root() -> Path:
    return Path.home() / ".local" / "share" / "ai-orchestrator" / "staging"


def _safe_name(url: str) -> str:
    """Derive a filename from the URL without ever trusting it as a path."""

    parsed = urllib.parse.urlparse(url)
    candidate = Path(urllib.parse.unquote(parsed.path)).name
    cleaned = "".join(
        character
        for character in candidate
        if character.isalnum() or character in {".", "-", "_"}
    ).lstrip(".")
    return cleaned[:120] or "download"


def _inside_a_repository(path: Path) -> bool:
    """Say whether a directory sits in a git working tree."""

    for parent in [path, *path.parents]:
        if (parent / ".git").exists():
            return True
    return False


def fetch_resource(
    url: str,
    *,
    staging_root: Path | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = MAX_FETCH_BYTES,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Download one resource into staging and return its receipt.

    Nunca sobrescribe: cada descarga va a su propio directorio sellado con la
    hora, de modo que dos intentos del mismo recurso quedan uno al lado del
    otro y se pueden comparar por hash.
    """

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ContractError(f"unsupported download scheme: {parsed.scheme or 'none'}")
    if not parsed.netloc:
        raise ContractError("download URL has no host")

    root = (staging_root or default_staging_root()).expanduser()
    if _inside_a_repository(root):
        # Un recurso sin revisar no entra a un arbol de trabajo por la puerta
        # de atras: que se promueva es una decision aparte y explicita.
        raise ContractError("staging root must not live inside a git repository")
    ensure_private_directory(root.parent)
    ensure_private_directory(root)

    # El sello lleva un sufijo aleatorio ademas de la hora: dos descargas del
    # mismo recurso dentro del mismo segundo colisionaban, y reintentar una
    # descarga es justo lo que uno hace cuando la primera sale mal.
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    fingerprint = hashlib.sha256(url.encode()).hexdigest()[:12]
    destination_dir = root / f"{stamp}-{fingerprint}-{uuid.uuid4().hex[:8]}"
    if destination_dir.exists():
        raise ContractError("staging destination already exists")
    ensure_private_directory(destination_dir)
    destination = destination_dir / _safe_name(url)

    digest = hashlib.sha256()
    written = 0
    started = time.monotonic()
    request = urllib.request.Request(url, headers={"User-Agent": "ai-orchestrator"})
    handle, staging_path = tempfile.mkstemp(dir=str(destination_dir), prefix=".part.")
    try:
        with (
            urllib.request.urlopen(request, timeout=timeout_seconds) as response,
            os.fdopen(handle, "wb") as stream,
        ):
            final_url = response.geturl()
            if urllib.parse.urlparse(final_url).scheme not in ALLOWED_SCHEMES:
                raise ContractError("download redirected to an unsupported scheme")
            declared = response.headers.get("Content-Type")
            while True:
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise ContractError(f"download exceeds the {max_bytes}-byte limit")
                digest.update(chunk)
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(staging_path, 0o600)
        os.replace(staging_path, destination)
    except urllib.error.HTTPError as exc:
        Path(staging_path).unlink(missing_ok=True)
        raise ContractError(f"download failed with HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        Path(staging_path).unlink(missing_ok=True)
        raise ContractError(f"download failed: {exc}") from exc
    except BaseException:
        Path(staging_path).unlink(missing_ok=True)
        raise

    sha256 = digest.hexdigest()
    if expected_sha256 and expected_sha256.lower() != sha256:
        # El fichero se conserva, pero el recibo dice que no es lo pedido.
        raise ContractError(
            f"sha256 mismatch: expected {expected_sha256.lower()}, got {sha256}"
        )

    receipt = {
        "schema_version": 1,
        "kind": "resource_fetch",
        "requested_url": url,
        "final_url": final_url,
        "path": str(destination),
        "bytes": written,
        "sha256": sha256,
        "declared_content_type": declared,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "latency_ms": int((time.monotonic() - started) * 1000),
        "promoted_to_repository": False,
    }
    receipt_path = destination_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    receipt_path.chmod(0o600)
    return receipt


def list_staged(staging_root: Path | None = None) -> list[dict[str, Any]]:
    """Report every staged download that still has a readable receipt."""

    root = (staging_root or default_staging_root()).expanduser()
    if not root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for entry in sorted(root.iterdir()):
        receipt = entry / "receipt.json"
        if not receipt.is_file():
            continue
        try:
            rows.append(json.loads(receipt.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            rows.append({"path": str(entry), "error": "unreadable receipt"})
    return rows
