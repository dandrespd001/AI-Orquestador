"""Isolated adapter metadata for every supported provider/client boundary."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ai_orchestrator.config import load_models
from ai_orchestrator.models import ContractError
from ai_orchestrator.providers import load_provider_specs


@dataclass(frozen=True)
class AdapterDefinition:
    provider: str
    transport: str
    executable: str
    authentication: str
    endpoint: str
    version_pin: str | None = None
    human_login: bool = False
    dynamic_models: bool = False
    included_quota_only: bool = True
    supports_receipts: bool = True
    isolated_process: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


ADAPTERS = {
    row.provider: row
    for row in (
        AdapterDefinition(
            "openai",
            "codex_cli",
            "codex",
            "oauth_official_store",
            "cli://codex",
            human_login=True,
        ),
        AdapterDefinition(
            "anthropic",
            "claude_cli",
            "claude",
            "oauth_official_store",
            "cli://claude",
            human_login=True,
        ),
        AdapterDefinition(
            "deepseek",
            "qwen_omniroute_cli",
            "qwen",
            "systemd_credential_broker",
            "http://127.0.0.1:20128/v1",
        ),
        AdapterDefinition(
            "minimax",
            "qwen_minimax_cli",
            "qwen",
            "systemd_credential_broker",
            "https://api.minimax.io/v1",
        ),
        AdapterDefinition(
            "moonshot",
            "kimi_cli",
            "kimi",
            "oauth_official_store",
            "cli://kimi",
            human_login=True,
            dynamic_models=True,
        ),
        AdapterDefinition(
            "omniroute",
            "omniroute",
            "omniroute",
            "systemd_credential_broker",
            "http://127.0.0.1:20128/v1",
        ),
        AdapterDefinition(
            "qwen", "qwen_cli", "qwen", "delegated_provider_route", "cli://qwen"
        ),
        AdapterDefinition(
            "opencode",
            "opencode_cli",
            "opencode",
            "oauth_official_store",
            "cli://opencode",
            human_login=True,
        ),
        AdapterDefinition(
            "abacus",
            "abacus_cli",
            "abacusai",
            "human_cli_login",
            "cli://abacusai",
            version_pin="2.6.9",
            human_login=True,
        ),
        AdapterDefinition(
            "fx",
            "fx_cli",
            "fx",
            "oauth_official_store",
            "cli://fx@0.0.4",
            version_pin="0.0.4",
            human_login=True,
        ),
        AdapterDefinition(
            "local-ai",
            "qwen_omniroute_cli",
            "qwen",
            "local_gateway",
            "http://127.0.0.1:20128/v1",
        ),
        AdapterDefinition(
            "antigravity",
            "qwen_omniroute_cli",
            "qwen",
            "oauth_official_store",
            "http://127.0.0.1:20128/v1",
        ),
        AdapterDefinition(
            "qwen-web",
            "qwen_omniroute_cli",
            "qwen",
            "oauth_official_store",
            "http://127.0.0.1:20128/v1",
        ),
        AdapterDefinition(
            "huggingchat",
            "qwen_omniroute_cli",
            "qwen",
            "oauth_official_store",
            "http://127.0.0.1:20128/v1",
        ),
        AdapterDefinition(
            "zenmux-free",
            "qwen_omniroute_cli",
            "qwen",
            "oauth_official_store",
            "http://127.0.0.1:20128/v1",
        ),
    )
}


def require_adapter(provider: str, transport: str) -> AdapterDefinition:
    try:
        adapter = ADAPTERS[provider]
    except KeyError as exc:
        raise ContractError(f"unknown provider adapter: {provider}") from exc
    if adapter.transport != transport:
        raise ContractError(
            f"provider adapter transport mismatch: {provider}:{transport}"
        )
    return adapter


def validate_adapter_catalog(root: Path | None = None) -> dict[str, Any]:
    providers = load_provider_specs(root)
    models = load_models(root)
    errors = [
        f"provider has no adapter: {provider}"
        for provider in sorted(set(providers) - set(ADAPTERS))
    ]
    errors.extend(
        f"adapter has no provider: {provider}"
        for provider in sorted(set(ADAPTERS) - set(providers))
    )
    for model in models.values():
        adapter = ADAPTERS.get(model.provider)
        if adapter is not None and adapter.transport != model.transport:
            errors.append(
                "model transport does not match adapter: "
                f"{model.id}:{model.transport}!={adapter.transport}"
            )
    return {"schema_version": 1, "ok": not errors, "errors": errors}


def adapter_inventory() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "adapters": [ADAPTERS[name].as_dict() for name in sorted(ADAPTERS)],
        "incremental_spend_allowed": False,
        "provenance_receipts_required": True,
    }


def adapter_environment(provider: str) -> dict[str, str]:
    """Return a process envelope without copying provider credentials."""

    if provider not in ADAPTERS:
        raise ContractError(f"unknown provider adapter: {provider}")
    allowed = {
        "HOME",
        "USER",
        "LOGNAME",
        "PATH",
        "LANG",
        "LC_ALL",
        "TERM",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
        "NO_COLOR",
    }
    result = {key: value for key, value in os.environ.items() if key in allowed}
    result.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    result["NO_COLOR"] = "1"
    result["AI_ORCHESTRATOR_INCREMENTAL_SPEND"] = "0"
    return result


def provenance_receipt(
    provider: str,
    *,
    requested_model: str,
    reported_model: str | None,
    verified: bool,
) -> dict[str, Any]:
    if provider not in ADAPTERS:
        raise ContractError(f"unknown provider adapter: {provider}")
    return {
        "schema_version": 1,
        "provider": provider,
        "transport": ADAPTERS[provider].transport,
        "requested_model": requested_model,
        "reported_model": reported_model,
        "provenance_verified": bool(verified),
        "incremental_spend_allowed": False,
        "content_retained": False,
    }
