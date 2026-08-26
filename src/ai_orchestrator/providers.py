"""Provider catalog and secret-free model discovery.

Provider definitions are policy, not client configuration.  In particular,
OAuth tokens and API keys remain in their official stores or in
``systemd-creds`` and are never accepted by this module.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_orchestrator.config import _read_toml, config_root, load_models
from ai_orchestrator.credentials import deepseek_runtime_key
from ai_orchestrator.models import ContractError, ProviderSpec, QuotaSnapshot

FORBIDDEN_SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "token",
}


def fetch_deepseek_balance(
    api_key: str,
    *,
    timeout_seconds: float = 5.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
    verified_at: str | None = None,
) -> QuotaSnapshot:
    """Read prepaid balance from DeepSeek's official secret-safe endpoint."""

    if not isinstance(api_key, str) or not api_key.strip():
        raise ContractError("DeepSeek balance requires a credential")
    if not 0.1 <= timeout_seconds <= 10.0:
        raise ContractError("DeepSeek balance timeout must be between 0.1 and 10 seconds")
    endpoint = "https://api.deepseek.com/user/balance"
    request = urllib.request.Request(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with opener(request, timeout=timeout_seconds) as response:
            raw = response.read(1_000_001)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ContractError("DeepSeek balance is unavailable") from exc
    if len(raw) > 1_000_000:
        raise ContractError("DeepSeek balance response is too large")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("DeepSeek balance returned malformed JSON") from exc
    if not isinstance(payload, dict) or _contains_secret_fields(payload):
        raise ContractError("DeepSeek balance returned an unsafe envelope")
    rows = payload.get("balance_infos")
    if not isinstance(rows, list):
        raise ContractError("DeepSeek balance response has no balance_infos")
    balance: float | None = None
    for row in rows:
        if not isinstance(row, dict) or row.get("currency") != "USD":
            continue
        try:
            balance = float(row["total_balance"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError("DeepSeek USD balance is malformed") from exc
        break
    if balance is None or balance < 0:
        raise ContractError("DeepSeek USD balance is unavailable")
    if payload.get("is_available") is False:
        balance = 0.0
    return QuotaSnapshot(
        provider="deepseek",
        plan="prepaid_api_balance",
        billing_mode="prepaid_balance",
        remaining=balance,
        balance_usd=balance,
        source="/user/balance",
        verified_at=verified_at or datetime.now(UTC).isoformat(),
        confidence="verified",
    )


def refresh_deepseek_quota(
    telemetry: Any,
    *,
    configuration_root: Path | None = None,
    credential_resolver: Callable[..., str | None] = deepseek_runtime_key,
    balance_reader: Callable[..., QuotaSnapshot] = fetch_deepseek_balance,
) -> QuotaSnapshot:
    """Refresh DeepSeek's prepaid balance or return an explicit unknown value."""

    verified_at = datetime.now(UTC).isoformat()
    key = credential_resolver(root=configuration_root)
    if key is None:
        return _unavailable_deepseek_quota(verified_at, "credential unavailable")
    try:
        snapshot = balance_reader(key, verified_at=verified_at)
    except ContractError:
        return _unavailable_deepseek_quota(verified_at, "balance endpoint unavailable")
    if hasattr(telemetry, "record_quota"):
        telemetry.record_quota(snapshot)
    return snapshot


def _unavailable_deepseek_quota(verified_at: str, source: str) -> QuotaSnapshot:
    return QuotaSnapshot(
        provider="deepseek",
        plan="prepaid_api_balance",
        billing_mode="prepaid_balance",
        source=source,
        verified_at=verified_at,
        confidence="unavailable",
    )


def load_provider_specs(root: Path | None = None) -> dict[str, ProviderSpec]:
    source = (root or config_root()) / "providers.toml"
    payload = _read_toml(source)
    rows = payload.get("providers")
    if not isinstance(rows, list):
        raise ContractError("providers.toml must define [[providers]]")
    result: dict[str, ProviderSpec] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            raise ContractError("provider entry must be a table")
        if _contains_secret_fields(raw):
            raise ContractError("providers.toml must not contain provider secrets")
        provider = ProviderSpec.from_mapping(dict(raw))
        if provider.id in result:
            raise ContractError(f"duplicate provider id: {provider.id}")
        result[provider.id] = provider
    return result


def validate_provider_catalog(root: Path | None = None) -> dict[str, Any]:
    providers = load_provider_specs(root)
    models = load_models(root)
    errors: list[str] = []
    for model in models.values():
        provider = providers.get(model.provider)
        if provider is None:
            errors.append(
                f"model references unknown provider: {model.id}:{model.provider}"
            )
            continue
        if model.id not in provider.models:
            errors.append(f"provider catalog omits model: {provider.id}:{model.id}")
    # Provider catalogs may advertise dynamic or not-yet-promoted aliases
    # (notably Kimi and Abacus).  Only routable ModelConfig rows are required
    # to have a matching provider declaration.
    return {
        "schema_version": 1,
        "ok": not errors,
        "provider_count": len(providers),
        "errors": errors,
    }


def provider_inventory(root: Path | None = None) -> list[dict[str, Any]]:
    return [
        provider.as_dict()
        for provider in sorted(
            load_provider_specs(root).values(), key=lambda row: row.id
        )
    ]


def _loopback_http_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme != "http" or parsed.hostname not in {
        "127.0.0.1",
        "::1",
        "localhost",
    }:
        raise ContractError("provider discovery must use a loopback HTTP endpoint")
    if parsed.username or parsed.password or parsed.fragment:
        raise ContractError("unsafe provider discovery URL")
    return value.rstrip("/")


def _contains_secret_fields(value: object) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = str(key).strip().lower()
            if lowered in FORBIDDEN_SECRET_KEYS or lowered.endswith(
                ("_api_key", "_password", "_secret", "_token")
            ):
                return True
            if _contains_secret_fields(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_secret_fields(item) for item in value)
    return False


def discover_kimi_models(
    server_url: str,
    *,
    timeout_seconds: float = 3.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> tuple[str, ...]:
    """Read model aliases from Kimi's local server API.

    The endpoint is loopback-only and the response is rejected if it contains
    credential-shaped fields.  OAuth remains owned by Kimi's official store.
    """

    if not 0.1 <= timeout_seconds <= 10.0:
        raise ContractError("Kimi discovery timeout must be between 0.1 and 10 seconds")
    endpoint = _loopback_http_url(server_url) + "/api/v1/models"
    request = urllib.request.Request(endpoint, method="GET")
    try:
        with opener(request, timeout=timeout_seconds) as response:
            raw = response.read(1_000_001)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ContractError("Kimi model discovery is unavailable") from exc
    if len(raw) > 1_000_000:
        raise ContractError("Kimi model discovery response is too large")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("Kimi model discovery returned malformed JSON") from exc
    if _contains_secret_fields(payload):
        raise ContractError("Kimi discovery response unexpectedly exposed credentials")
    rows: object
    if isinstance(payload, dict):
        rows = payload.get("models", payload.get("data", ()))
    else:
        rows = payload
    if not isinstance(rows, list):
        raise ContractError("Kimi discovery response has no model list")
    aliases: set[str] = set()
    for row in rows:
        if isinstance(row, str):
            alias = row
        elif isinstance(row, dict):
            alias = str(row.get("id", row.get("alias", row.get("name", ""))))
        else:
            continue
        alias = alias.strip()
        if alias and "\x00" not in alias:
            aliases.add(alias)
    if not aliases:
        raise ContractError("Kimi discovery returned no usable model aliases")
    return tuple(sorted(aliases))


def validate_kimi_alias(
    alias: str,
    discovered_aliases: tuple[str, ...],
    canary_receipt: dict[str, Any],
) -> dict[str, Any]:
    rows = canary_receipt.get("providers", ())
    matching = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("provider") == "moonshot"
    ]
    passed = bool(
        matching
        and matching[0].get("status") == "PASSED"
        and matching[0].get("provenance_verified") is True
    )
    reported = (
        str(matching[0].get("reported_model") or matching[0].get("model") or "")
        if matching
        else ""
    )
    valid = (
        alias in discovered_aliases
        and passed
        and reported in {alias, alias.split("/", 1)[-1]}
    )
    return {
        "schema_version": 1,
        "alias": alias,
        "discovered": alias in discovered_aliases,
        "canary_passed": passed,
        "reported_model": reported or None,
        "valid": valid,
    }
