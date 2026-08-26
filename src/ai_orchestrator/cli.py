from __future__ import annotations

import argparse
import dataclasses
import getpass
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ai_orchestrator import __version__
from ai_orchestrator.adapters import AdapterError
from ai_orchestrator.api import HumanActionGate, serve_panel
from ai_orchestrator.archive import ArchiveError, create_encrypted_archive
from ai_orchestrator.benchmark import (
    atomic_private_json,
    run_gateway_benchmark,
    select_gateway,
)
from ai_orchestrator.broker import UnixBrokerServer, default_socket_path
from ai_orchestrator.canary import ROUTES, run_official_canaries
from ai_orchestrator.capabilities import select_capability_package
from ai_orchestrator.client_catalog import (
    SUPPORTED_CLIENTS,
    generate_client_config,
    write_client_config,
)
from ai_orchestrator.config import (
    config_root,
    load_models,
    load_profile,
    load_roles,
    validate_configuration,
)
from ai_orchestrator.config_sync import (
    apply_config_sync,
    canonical_config_root,
    config_sync_status,
)
from ai_orchestrator.control_plane import ControlPlane
from ai_orchestrator.credentials import (
    CredentialStore,
    authentication_inventory,
    deepseek_runtime_key,
    minimax_runtime_key,
    omniroute_runtime_key,
    resolve_credential,
)
from ai_orchestrator.delegation import OrchestratorTaskRunner
from ai_orchestrator.evals import run_routing_eval
from ai_orchestrator.extensions import ExtensionRegistry, extension_manifest_from_file
from ai_orchestrator.harness import (
    HARNESS_UI_MODELS,
    WorkspaceRegistry,
    harness_launch_spec,
    run_harness_worker,
    validate_harness_launch,
)
from ai_orchestrator.harness_bridge import BRIDGE_PORT, HarnessBridge
from ai_orchestrator.harness_install import prepare_and_stage_harness
from ai_orchestrator.mcp_install import SUPPORTED_MCP_CLIENTS, install_native_mcp
from ai_orchestrator.mcp_tools import MCP_TOOLS, serve_stdio
from ai_orchestrator.models import ContractError, TaskSpec
from ai_orchestrator.optimization import prompt_optimization_preflight
from ai_orchestrator.promotion import CAPABILITY_GROUPS, assess_promotion
from ai_orchestrator.providers import refresh_deepseek_quota, validate_provider_catalog
from ai_orchestrator.quality import promptfoo_preflight, run_quality_eval
from ai_orchestrator.routing import select_route
from ai_orchestrator.sessions import SessionStore
from ai_orchestrator.storage import (
    SQLITE_TARGETS,
    apply_private_permissions,
    audit_storage,
    bound_sqlite_journals,
    compact_sqlite,
    disable_omniroute_content_logging,
    enforce_official_omniroute_providers,
    storage_receipt,
)
from ai_orchestrator.telemetry import TelemetryStore, default_state_root
from ai_orchestrator.updates import UpdateRegistry, load_component_pins
from ai_orchestrator.writer_canary import run_writer_canaries

EXIT_POLICY = 4
EXIT_RUNTIME = 5


def _print(payload: object) -> None:
    print(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False))


def _route(
    spec_path: str, profile_name: str, *, record: bool, pin: str | None = None
) -> tuple[TaskSpec, Any]:
    task = TaskSpec.read(spec_path)
    models = load_models()
    roles = load_roles()
    profile = load_profile(profile_name)
    telemetry = TelemetryStore()
    quotas = telemetry.latest_quotas()
    quotas["deepseek"] = refresh_deepseek_quota(telemetry)
    decision = select_route(
        task,
        models,
        roles,
        profile,
        quotas=quotas,
        rollout_counts=telemetry.rollout_counts(),
        pin=pin,
        codex_uses_today=telemetry.daily_provider_runs("openai"),
    )
    if record:
        telemetry.record_route(task, decision, simulated=True)
    return task, decision


def _local_health() -> dict[str, object]:
    key = omniroute_runtime_key()
    if not key:
        return {"reachable": False, "status": None}
    request = urllib.request.Request(
        "http://127.0.0.1:20128/v1/models",
        headers={"Authorization": "Bearer " + key},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=10.0) as response:
            return {"reachable": True, "status": response.status}
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return {"reachable": False, "status": None}


def _harness_health(port: int = 3080) -> dict[str, object]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/",
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            return {"reachable": True, "status": response.status}
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return {"reachable": False, "status": None}


def _credential_readiness(credentials: dict[str, bool]) -> dict[str, object]:
    return {
        "deepseek_transport": "omniroute_local_bridge",
        "ok": credentials["minimax"] and credentials["omniroute"],
    }


def command_doctor(args: argparse.Namespace) -> int:
    configuration = validate_configuration(args.profile)
    commands = {
        name: shutil.which(name)
        for name in (
            "codex",
            "claude",
            "qwen",
            "kimi",
            "omniroute",
            "opencode",
            "abacusai",
            "bwrap",
            "git",
            "gpg",
            "zstd",
            "tar",
            "systemd-creds",
        )
    }
    storage = audit_storage()
    sync = config_sync_status(canonical_config_root(), config_root())
    credentials = {
        "deepseek_direct": deepseek_runtime_key() is not None,
        "minimax": minimax_runtime_key() is not None,
        "omniroute": omniroute_runtime_key() is not None,
    }
    credential_requirements = _credential_readiness(credentials)
    providers = validate_provider_catalog()
    update_inventory = UpdateRegistry(
        default_state_root() / "updates", create=False
    ).inventory()
    harness_active = bool(
        update_inventory.get("components", {}).get("deepseek-harness", {}).get("active")
    )
    harness_service = _harness_health()
    omniroute_health = _local_health()
    readiness_ok = bool(
        configuration["ok"]
        and all(commands.values())
        and credential_requirements["ok"]
        and default_socket_path().exists()
        and harness_active
        and harness_service["reachable"]
        and omniroute_health["reachable"]
    )
    compliance_errors = [
        *configuration.get("errors", []),
        *providers.get("errors", []),
    ]
    compliance_ok = bool(storage["ok"] and sync["in_sync"] and not compliance_errors)
    selected_ok = readiness_ok if args.check == "readiness" else compliance_ok
    if args.check == "all":
        selected_ok = readiness_ok and compliance_ok
    payload = {
        "schema_version": 2,
        "check": args.check,
        "ok": selected_ok,
        "readiness": {
            "ok": readiness_ok,
            "broker": default_socket_path().exists(),
            "harness": harness_active,
            "harness_service": harness_service["reachable"],
        },
        "compliance": {
            "ok": compliance_ok,
            "errors": compliance_errors,
            "warnings": [],
        },
        "configuration": configuration,
        "providers": providers,
        "commands": commands,
        "omniroute": omniroute_health,
        "harness_service": harness_service,
        "credentials": credentials,
        "credential_requirements": credential_requirements,
        "config_sync": sync,
        "storage": storage,
    }
    _print(payload)
    return 0 if payload["ok"] else EXIT_POLICY


def command_status(args: argparse.Namespace) -> int:
    """Return a versioned, read-only snapshot without initializing state stores."""

    configuration = validate_configuration(args.profile)
    update_root = default_state_root() / "updates"
    updates = UpdateRegistry(update_root, create=False).inventory()
    payload = {
        "schema_version": 2,
        "ok": bool(configuration["ok"]),
        "profile": args.profile,
        "configuration": configuration,
        "broker_socket_present": default_socket_path().exists(),
        "harness_active": bool(
            updates.get("components", {}).get("deepseek-harness", {}).get("active")
        ),
        "automatic_updates": False,
    }
    _print(payload)
    return 0 if payload["ok"] else EXIT_POLICY


def command_model_list(args: argparse.Namespace) -> int:
    _print(
        {
            "schema_version": 2,
            "models": [
                dataclasses.asdict(model)
                for model in sorted(load_models().values(), key=lambda item: item.id)
            ],
        }
    )
    return 0


def command_client_list(args: argparse.Namespace) -> int:
    pins = load_component_pins()
    _print(
        {
            "schema_version": 2,
            "clients": [
                pin.as_dict()
                for pin in sorted(pins.values(), key=lambda item: item.name)
                if pin.name.endswith("-cli") or pin.name == "deepseek-harness"
            ],
        }
    )
    return 0


def command_mcp_status(args: argparse.Namespace) -> int:
    encoded = json.dumps(MCP_TOOLS, separators=(",", ":")).encode("utf-8")
    _print(
        {
            "schema_version": 2,
            "profile": args.profile,
            "tools": [tool["name"] for tool in MCP_TOOLS],
            "tool_count": len(MCP_TOOLS),
            "schema_bytes": len(encoded),
            "strict": True,
        }
    )
    return 0


def command_mcp_install(args: argparse.Namespace) -> int:
    _print(install_native_mcp(args.client, profile=args.profile))
    return 0


def command_harness_status(args: argparse.Namespace) -> int:
    registry = UpdateRegistry(default_state_root() / "updates", create=False)
    inventory = registry.inventory()
    active = inventory.get("components", {}).get("deepseek-harness", {}).get("active")
    service = _harness_health(args.port)
    ready = bool(active) and bool(service["reachable"])
    _print(
        {
            "schema_version": 2,
            "version": active.get("version") if isinstance(active, dict) else None,
            "update_channel": "latest",
            "update_policy": "manual_promote",
            "active": active,
            "ready": ready,
            "service": service,
            "host": "127.0.0.1",
            "port": args.port,
        }
    )
    return 0 if ready else EXIT_POLICY


def command_harness_serve(args: argparse.Namespace) -> int:
    upstream_credential = omniroute_runtime_key()
    if not upstream_credential:
        raise ContractError("Harness cannot authenticate to local OmniRoute")
    search_credential = resolve_credential("web_search_proxy_key")
    if args.profile == "orchestrator-web" and not search_credential:
        raise ContractError("Harness web search credential is unavailable")
    minimax_credential = minimax_runtime_key()
    enabled_models = frozenset(
        model.id
        for model in load_models().values()
        if model.id in HARNESS_UI_MODELS and model.status == "available"
    )
    if "MiniMax-M3" in enabled_models and not minimax_credential:
        raise ContractError(
            "Harness cannot authenticate MiniMax through its local bridge"
        )
    with HarnessBridge(
        upstream_credential=upstream_credential,
        port=BRIDGE_PORT,
        allowed_models=enabled_models,
        minimax_credential=minimax_credential,
    ) as bridge:
        spec = harness_launch_spec(profile=args.profile, port=args.port)
        spec.environment["AI_ORCHESTRATOR_HARNESS_CREDENTIAL"] = bridge.credential
        spec.environment["AI_ORCHESTRATOR_DEEPSEEK_BRIDGE_URL"] = bridge.url
        if search_credential:
            spec.environment["AI_ORCHESTRATOR_WEB_SEARCH_CREDENTIAL"] = (
                search_credential
            )
            spec.environment["AI_ORCHESTRATOR_WEB_SEARCH_URL"] = (
                "http://127.0.0.1:20130/mcp"
            )
        validation = validate_harness_launch(spec)
        if not validation["ok"]:
            raise ContractError(
                "Harness launch validation failed: " + "; ".join(validation["errors"])
            )
        dump = [spec.command[0], "--profile", spec.profile, "--dump-config"]
        checked = subprocess.run(
            dump,
            cwd=args.workspace,
            env=spec.environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=20,
            check=False,
        )
        if checked.returncode != 0:
            raise ContractError("Harness composed configuration failed validation")
        completed = subprocess.run(
            list(spec.command), cwd=args.workspace, env=spec.environment, check=False
        )
    return int(completed.returncode)


def command_harness_update(args: argparse.Namespace) -> int:
    source = Path(args.source).expanduser().resolve()
    registry = UpdateRegistry(default_state_root() / "updates")
    _print(
        prepare_and_stage_harness(
            source=source, registry=registry, target_version=args.version
        )
    )
    return 0


def command_harness_worker(args: argparse.Namespace) -> int:
    prompt = sys.stdin.buffer.read(4 * 1024 * 1024 + 1)
    if len(prompt) > 4 * 1024 * 1024:
        raise ContractError("Harness worker prompt exceeds 4 MiB")
    if not prompt.strip():
        raise ContractError("Harness worker requires a prompt on stdin")
    payload = run_harness_worker(
        model=args.model,
        prompt=prompt,
        workspace=Path.cwd(),
        timeout_seconds=args.timeout,
        extended=args.extended,
        workspace_isolated=args.orchestrated_worktree,
    )
    _print(payload)
    return 0


def command_harness_workspace(args: argparse.Namespace) -> int:
    registry = WorkspaceRegistry(default_state_root() / "harness-workspaces.json")
    if args.workspace_command == "list":
        payload = registry.list()
    elif args.workspace_command == "add":
        payload = registry.add(Path(args.path), human_approved=args.approve)
    else:
        payload = registry.remove(Path(args.path), human_approved=args.approve)
    if args.workspace_command != "list":
        drop_in = (
            Path(args.drop_in)
            if args.drop_in
            else (
                Path.home()
                / ".config"
                / "systemd"
                / "user"
                / "ai-orchestrator-harness.service.d"
                / "20-workspaces.conf"
            )
        )
        payload["systemd_dropin"] = registry.write_systemd_dropin(
            drop_in, human_approved=args.approve
        )
    _print(payload)
    return 0


def command_docs(args: argparse.Namespace) -> int:
    from ai_orchestrator.docs_codegen import generate_docs, repo_root

    payload = generate_docs(
        repo_root(getattr(args, "root", None)),
        check=args.docs_command == "check",
    )
    _print(payload)
    return 0 if payload["ok"] else EXIT_POLICY


def command_route(args: argparse.Namespace) -> int:
    _, decision = _route(
        args.spec, args.profile, record=not args.no_record, pin=args.model
    )
    _print(decision.as_dict())
    return 0 if decision.status != "NO_ROUTE" else EXIT_POLICY


def command_run(args: argparse.Namespace) -> int:
    receipt = OrchestratorTaskRunner(allow_validated_apply=True).run(
        spec_path=args.spec,
        profile=args.profile,
        model=args.model,
        confirmed_plan_id=args.confirmed_plan_id,
        output_path=Path(args.output).expanduser() if args.output else None,
    )
    _print(receipt)
    if receipt.get("final_accepted") and receipt.get("status") in {
        "SUCCESS",
        "APPLIED",
        "PATCH_READY",
    }:
        return 0
    if receipt.get("status") in {"HANDOFF_REQUIRED", "REPLAN_REQUIRED"}:
        return EXIT_POLICY
    return EXIT_RUNTIME


def command_config_sync(args: argparse.Namespace) -> int:
    source = (
        Path(args.source).expanduser().resolve()
        if args.source
        else canonical_config_root()
    )
    target = Path(args.target).expanduser().resolve() if args.target else config_root()
    payload = (
        apply_config_sync(source, target)
        if args.apply
        else config_sync_status(source, target)
    )
    _print(payload)
    return 0 if payload["in_sync"] else EXIT_POLICY


def command_quota(args: argparse.Namespace) -> int:
    models = load_models()
    telemetry = TelemetryStore()
    quotas = telemetry.latest_quotas()
    quotas["deepseek"] = refresh_deepseek_quota(telemetry)
    _print(
        {
            "schema_version": 2,
            "quota_snapshots": {
                provider: snapshot.as_dict()
                for provider, snapshot in sorted(quotas.items())
            },
            "models": [
                {
                    "id": model.id,
                    "provider": model.provider,
                    "transport": model.transport,
                    "status": model.status,
                    "incremental_spend_allowed": False,
                    "quota": (
                        quotas[model.provider].as_dict()
                        if model.provider in quotas
                        else None
                    ),
                }
                for model in models.values()
            ],
        }
    )
    return 0


def command_promotion_assess(args: argparse.Namespace) -> int:
    containment = _bounded_json(Path(args.containment), expected=dict)
    evaluations = _bounded_json(Path(args.evaluations), expected=list)
    payload = assess_promotion(
        args.model,
        args.provider,
        containment=containment,
        evaluations=evaluations,
        target_stage=args.target_stage,
        critical_groups=set(args.critical_group or ()),
    )
    _print(payload)
    return 0 if payload["promotion_ready"] else EXIT_POLICY


def _bounded_json(path: Path, *, expected: type) -> Any:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise ContractError("evaluation receipt must not be a symlink")
    resolved = unresolved.resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size > 4_000_000:
        raise ContractError("evaluation receipt must be a file of at most 4 MB")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("evaluation receipt must be valid UTF-8 JSON") from exc
    if not isinstance(payload, expected):
        raise ContractError(f"evaluation receipt must be a JSON {expected.__name__}")
    return payload


def command_storage_audit(args: argparse.Namespace) -> int:
    payload = audit_storage()
    _print(payload)
    return 0 if payload["ok"] else EXIT_POLICY


def command_storage_maintain(args: argparse.Namespace) -> int:
    actions: dict[str, object] = {}
    if args.apply_permissions:
        actions["permissions"] = (
            apply_private_permissions()
            if args.apply
            else {"dry_run": True, "selected": True}
        )
    else:
        actions["permissions"] = {"dry_run": True}

    if args.compact:
        matched = [
            (path, client)
            for path, client in SQLITE_TARGETS
            if str(path) == args.compact
        ]
        if not matched:
            raise ContractError("--compact must name an audited SQLite target exactly")
        path, client = matched[0]
        actions["compaction"] = (
            compact_sqlite(path, client)
            if args.apply
            else {"dry_run": True, "path": str(path), "client": client}
        )

    if args.purge_metadata:
        store = TelemetryStore()
        actions["metadata_purge"] = (
            store.purge_metadata(args.metadata_retention_days)
            if args.apply
            else store.preview_metadata_purge(args.metadata_retention_days)
        )

    if args.bound_journals:
        actions["journal_bounds"] = bound_sqlite_journals(apply=args.apply)

    if args.official_providers:
        actions["official_providers"] = enforce_official_omniroute_providers(
            apply=args.apply
        )

    if args.disable_content_logs:
        actions["content_logging"] = disable_omniroute_content_logging(apply=args.apply)

    receipt = storage_receipt("storage_maintain", actions)
    receipt["post_audit"] = audit_storage()
    _print(receipt)
    return 0


def command_storage_archive(args: argparse.Namespace) -> int:
    paths = [Path(value) for value in args.path]
    receipt = create_encrypted_archive(paths, recipient=args.recipient)
    _print(receipt)
    return 0


def command_report(args: argparse.Namespace) -> int:
    _print(TelemetryStore().report())
    return 0


def command_eval(args: argparse.Namespace) -> int:
    payload = run_routing_eval(Path(args.suite), args.profile)
    _print(payload)
    return 0 if payload["status"] == "PASSED" else EXIT_POLICY


def command_gateway_benchmark(args: argparse.Namespace) -> int:
    payload = run_gateway_benchmark(Path(args.manifest).resolve())
    if args.output:
        atomic_private_json(Path(args.output).expanduser().resolve(), payload)
    _print(payload)
    return 0 if payload["status"] == "PASSED" else EXIT_POLICY


def command_provider_canary(args: argparse.Namespace) -> int:
    payload = run_official_canaries(args.provider, timeout_seconds=args.timeout)
    if args.output:
        atomic_private_json(Path(args.output).expanduser().resolve(), payload)
    _print(payload)
    return 0 if payload["status"] in {"PASSED", "DEGRADED_QUOTA"} else EXIT_RUNTIME


def command_writer_canary(args: argparse.Namespace) -> int:
    payload = run_writer_canaries(
        args.model,
        attempts=args.attempts,
        timeout_seconds=args.timeout,
        execution_profile=args.execution_profile,
    )
    if args.output:
        atomic_private_json(Path(args.output).expanduser().resolve(), payload)
    _print(payload)
    return 0 if payload["status"] == "PASSED" else EXIT_RUNTIME


def command_gateway_select(args: argparse.Namespace) -> int:
    source = Path(args.receipt).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise ContractError("gateway receipt must be a regular JSON file")
    payload = select_gateway(
        json.loads(source.read_text(encoding="utf-8")), args.incumbent
    )
    if args.output:
        atomic_private_json(Path(args.output).expanduser().resolve(), payload)
    _print(payload)
    return 0


def command_quality_preflight(args: argparse.Namespace) -> int:
    payload = promptfoo_preflight(Path(args.project_root).expanduser().resolve())
    _print(payload)
    return 0 if payload["available"] and payload["version_matches"] else EXIT_POLICY


def command_quality_eval(args: argparse.Namespace) -> int:
    payload = run_quality_eval(
        project_root=Path(args.project_root).expanduser().resolve(),
        config=Path(args.config).expanduser().resolve(),
        output=Path(args.output).expanduser().resolve(),
        timeout_seconds=args.timeout,
    )
    _print(payload)
    return 0 if payload["status"] == "PASSED" else EXIT_POLICY


def command_prompt_optimize(args: argparse.Namespace) -> int:
    payload = prompt_optimization_preflight(
        Path(args.experiment).expanduser().resolve(),
        Path(args.baseline).expanduser().resolve(),
    )
    atomic_private_json(Path(args.output).expanduser().resolve(), payload)
    _print(payload)
    return 0 if payload["status"] == "READY" else EXIT_POLICY


def command_control(args: argparse.Namespace) -> int:
    parameters: dict[str, object] = {}
    if args.section in {"credentials", "extensions", "diagnostics"}:
        parameters["profile"] = args.profile
    _print(ControlPlane().query(args.section, **parameters))
    return 0


def command_route_plan(args: argparse.Namespace) -> int:
    result = OrchestratorTaskRunner(allow_validated_apply=True).simulate(
        spec_path=args.spec,
        profile=args.profile,
        model=args.model,
    )
    _print(result)
    return 0 if result.get("status") == "READY" else EXIT_POLICY


def command_credential_status(args: argparse.Namespace) -> int:
    _print(
        {
            "schema_version": 1,
            "profiles": authentication_inventory(profile=args.profile),
            "secret_values_included": False,
        }
    )
    return 0


def command_credential_store(args: argparse.Namespace) -> int:
    if not args.approve:
        raise ContractError("credential rotation requires --approve")
    if sys.stdin.isatty():
        value = getpass.getpass("Credential (write-only): ")
    else:
        raw = sys.stdin.buffer.read(16_385)
        if len(raw) > 16_384:
            raise ContractError("credential input is too large")
        value = raw.decode("utf-8").rstrip("\r\n")
    receipt = CredentialStore().store(
        args.name,
        value,
        provider=args.provider,
        profile=args.profile,
        human_approved=True,
    )
    _print(receipt)
    return 0


def command_extension_inventory(args: argparse.Namespace) -> int:
    _print(ExtensionRegistry(create=False).inventory(profile=args.profile))
    return 0


def command_extension_discover(args: argparse.Namespace) -> int:
    manifest = extension_manifest_from_file(Path(args.manifest).expanduser().resolve())
    _print(ExtensionRegistry().discover(manifest))
    return 0


def command_extension_quarantine(args: argparse.Namespace) -> int:
    manifest = extension_manifest_from_file(Path(args.manifest).expanduser().resolve())
    pinned = ExtensionRegistry().quarantine(
        manifest, Path(args.payload).expanduser().resolve()
    )
    _print({"status": "quarantined", "manifest": pinned.as_dict()})
    return 0


def command_extension_inspect(args: argparse.Namespace) -> int:
    report = ExtensionRegistry().inspect(args.id)
    _print(report.as_dict())
    return 0 if report.approval_possible else EXIT_POLICY


def command_extension_tested(args: argparse.Namespace) -> int:
    _print(
        ExtensionRegistry().mark_tested(
            args.id, passed=args.passed, sandbox=args.sandbox
        )
    )
    return 0 if args.passed else EXIT_POLICY


def command_extension_approve(args: argparse.Namespace) -> int:
    _print(
        ExtensionRegistry().approve(
            args.id,
            expected_sha256=args.sha256,
            approver=args.approver,
            human_approved=args.approve,
        )
    )
    return 0


def command_extension_activate(args: argparse.Namespace) -> int:
    _print(
        ExtensionRegistry().activate(
            args.id,
            args.profile,
            human_approved=args.approve,
            host_risk_approved=args.approve_host_risk,
        )
    )
    return 0


def command_extension_rollback(args: argparse.Namespace) -> int:
    _print(
        ExtensionRegistry().rollback(args.id, args.profile, human_approved=args.approve)
    )
    return 0


def command_broker_serve(args: argparse.Namespace) -> int:
    path = (
        Path(args.socket).expanduser().resolve()
        if args.socket
        else default_socket_path()
    )
    with UnixBrokerServer(path) as server:
        server.serve_forever()
    return 0


def command_mcp_serve(args: argparse.Namespace) -> int:
    serve_stdio(profile=args.profile)
    return 0


def command_panel(args: argparse.Namespace) -> int:
    gate = HumanActionGate()
    print(
        "One-time human approval token (restart the panel for another write): "
        + gate.token,
        file=sys.stderr,
    )
    serve_panel(ControlPlane(), gate=gate, host="127.0.0.1", port=args.port)
    return 0


def command_harness_spec(args: argparse.Namespace) -> int:
    spec = harness_launch_spec(port=args.port)
    _print(
        {
            "launch": spec.as_dict(),
            "validation": validate_harness_launch(spec, require_socket=False),
        }
    )
    return 0


def command_session_maintain(args: argparse.Namespace) -> int:
    store = SessionStore()
    payload = (
        store.purge(args.retention_days)
        if args.apply
        else store.preview_purge(args.retention_days)
    )
    _print(payload)
    return 0


def command_component_inventory(args: argparse.Namespace) -> int:
    _print(ControlPlane().updates())
    return 0


def command_component_stage(args: argparse.Namespace) -> int:
    pins = load_component_pins()
    if args.name not in pins:
        raise ContractError(f"unknown pinned component: {args.name}")
    receipt = UpdateRegistry().stage(
        pins[args.name],
        Path(args.artifact).expanduser(),
        version=args.version,
        expected_sha256=args.sha256,
    )
    _print(receipt)
    return 0


def command_component_validate(args: argparse.Namespace) -> int:
    receipt = UpdateRegistry().record_validation(
        args.name,
        compatibility_passed=args.compatibility_passed,
        security_passed=args.security_passed,
        evaluation_passed=args.evaluation_passed,
    )
    _print(receipt)
    return (
        0
        if all(
            (args.compatibility_passed, args.security_passed, args.evaluation_passed)
        )
        else EXIT_POLICY
    )


def command_component_promote(args: argparse.Namespace) -> int:
    _print(UpdateRegistry().promote(args.name, human_approved=args.approve))
    return 0


def command_component_rollback(args: argparse.Namespace) -> int:
    _print(UpdateRegistry().rollback(args.name, human_approved=args.approve))
    return 0


def command_component_prune(args: argparse.Namespace) -> int:
    _print(
        UpdateRegistry().prune_inactive_slots(args.name, human_approved=args.approve)
    )
    return 0


def command_client_config(args: argparse.Namespace) -> int:
    payload = generate_client_config(ExtensionRegistry(), args.profile, args.client)
    if args.output:
        destination = Path(args.output).expanduser().resolve()
        write_client_config(destination, payload)
        _print(
            {
                "schema_version": 1,
                "status": "written",
                "path": str(destination),
                "client": args.client,
                "profile": args.profile,
                "provider_secrets_embedded": False,
            }
        )
    else:
        _print(payload)
    return 0


def command_capability_select(args: argparse.Namespace) -> int:
    _print(select_capability_package(args.package, client=args.client))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ai-orchestrator")
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--profile", default="default")
    status.set_defaults(func=command_status)

    doctor = subparsers.add_parser("doctor")
    doctor.add_argument("--profile", default="default")
    doctor.add_argument(
        "--check", choices=("readiness", "compliance", "all"), default="all"
    )
    doctor.set_defaults(func=command_doctor)

    config_parser = subparsers.add_parser("config")
    config_subparsers = config_parser.add_subparsers(
        dest="config_command", required=True
    )
    config_sync_parser = config_subparsers.add_parser("sync")
    config_sync_parser.add_argument("--source")
    config_sync_parser.add_argument("--target")
    config_sync_parser.add_argument("--apply", action="store_true")
    config_sync_parser.set_defaults(func=command_config_sync)

    route = subparsers.add_parser("route")
    route.add_argument("--spec", required=True)
    route.add_argument("--profile", default="default")
    route.add_argument("--no-record", action="store_true")
    route.add_argument("--model", default=None)
    route.set_defaults(func=command_route)

    run = subparsers.add_parser("run")
    run.add_argument("--spec", required=True)
    run.add_argument("--profile", default="default")
    # Ancla la ruta a un modelo concreto. Solo ESTRECHA: el modelo sigue
    # pasando por elegibilidad, y la decision registra pin=<id>.
    run.add_argument("--model", default=None)
    run.add_argument("--confirmed-plan-id", default=None)
    run.add_argument("--output")
    run.set_defaults(func=command_run)

    quota = subparsers.add_parser("quota")
    quota.set_defaults(func=command_quota)

    storage = subparsers.add_parser("storage")
    storage_subparsers = storage.add_subparsers(dest="storage_command", required=True)
    storage_audit = storage_subparsers.add_parser("audit")
    storage_audit.set_defaults(func=command_storage_audit)
    storage_maintain = storage_subparsers.add_parser("maintain")
    storage_maintain.add_argument("--apply-permissions", action="store_true")
    storage_maintain.add_argument("--compact")
    storage_maintain.add_argument("--purge-metadata", action="store_true")
    storage_maintain.add_argument("--metadata-retention-days", type=int, default=90)
    storage_maintain.add_argument("--bound-journals", action="store_true")
    storage_maintain.add_argument("--official-providers", action="store_true")
    storage_maintain.add_argument("--disable-content-logs", action="store_true")
    storage_maintain.add_argument("--apply", action="store_true")
    storage_maintain.set_defaults(func=command_storage_maintain)
    storage_archive = storage_subparsers.add_parser("archive")
    storage_archive.add_argument("--recipient", required=True)
    storage_archive.add_argument("--path", action="append", required=True)
    storage_archive.set_defaults(func=command_storage_archive)

    report = subparsers.add_parser("report")
    report.set_defaults(func=command_report)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--suite", required=True)
    eval_parser.add_argument("--profile", default="default")
    eval_parser.set_defaults(func=command_eval)

    provider_canary = subparsers.add_parser("provider-canary")
    provider_canary.add_argument("--provider", action="append", choices=sorted(ROUTES))
    provider_canary.add_argument("--timeout", type=int, default=120)
    provider_canary.add_argument("--output")
    provider_canary.set_defaults(func=command_provider_canary)

    writer_canary = subparsers.add_parser("writer-canary")
    writer_canary.add_argument("--model", required=True)
    writer_canary.add_argument("--attempts", type=int, default=5)
    writer_canary.add_argument("--timeout", type=int, default=120)
    # Mide un perfil que el modelo aun no tiene en models.toml. Promover antes
    # de medir seria el orden contrario al que exige la politica.
    writer_canary.add_argument("--execution-profile", default=None)
    writer_canary.add_argument("--output")
    writer_canary.set_defaults(func=command_writer_canary)

    promotion = subparsers.add_parser("promotion-assess")
    promotion.add_argument("--model", required=True)
    promotion.add_argument("--provider", required=True)
    promotion.add_argument("--containment", required=True)
    promotion.add_argument("--evaluations", required=True)
    promotion.add_argument(
        "--target-stage", choices=("read_only", "patch_writer"), required=True
    )
    promotion.add_argument(
        "--critical-group", action="append", choices=CAPABILITY_GROUPS
    )
    promotion.set_defaults(func=command_promotion_assess)

    benchmark = subparsers.add_parser("gateway-benchmark")
    benchmark.add_argument("--manifest", required=True)
    benchmark.add_argument("--output")
    benchmark.set_defaults(func=command_gateway_benchmark)

    gateway_select = subparsers.add_parser("gateway-select")
    gateway_select.add_argument("--receipt", required=True)
    gateway_select.add_argument("--incumbent", required=True)
    gateway_select.add_argument("--output")
    gateway_select.set_defaults(func=command_gateway_select)

    quality_preflight = subparsers.add_parser("quality-preflight")
    quality_preflight.add_argument(
        "--project-root", default=str(Path(__file__).resolve().parents[2])
    )
    quality_preflight.set_defaults(func=command_quality_preflight)

    quality_eval = subparsers.add_parser("quality-eval")
    quality_eval.add_argument(
        "--project-root", default=str(Path(__file__).resolve().parents[2])
    )
    quality_eval.add_argument("--config", required=True)
    quality_eval.add_argument("--output", required=True)
    quality_eval.add_argument("--timeout", type=int, default=900)
    quality_eval.set_defaults(func=command_quality_eval)

    optimize = subparsers.add_parser("prompt-optimize")
    optimize.add_argument("--experiment", required=True)
    optimize.add_argument("--baseline", required=True)
    optimize.add_argument("--output", required=True)
    optimize.set_defaults(func=command_prompt_optimize)

    control = subparsers.add_parser("control")
    control.add_argument(
        "section",
        choices=(
            "providers",
            "credentials",
            "models",
            "profiles",
            "routes",
            "usage",
            "extensions",
            "updates",
            "sessions",
            "capabilities",
            "diagnostics",
        ),
    )
    control.add_argument("--profile", default="default")
    control.set_defaults(func=command_control)

    route_plan_parser = subparsers.add_parser("route-plan")
    route_plan_parser.add_argument("--spec", required=True)
    route_plan_parser.add_argument("--profile", default="default")
    route_plan_parser.add_argument("--model", default=None)
    route_plan_parser.set_defaults(func=command_route_plan)

    credential = subparsers.add_parser("credential")
    credential_subparsers = credential.add_subparsers(
        dest="credential_command", required=True
    )
    credential_status = credential_subparsers.add_parser("status")
    credential_status.add_argument("--profile", default="default")
    credential_status.set_defaults(func=command_credential_status)
    credential_store = credential_subparsers.add_parser("store")
    credential_store.add_argument("--provider", required=True)
    credential_store.add_argument("--name", required=True)
    credential_store.add_argument("--profile", default="default")
    credential_store.add_argument("--approve", action="store_true")
    credential_store.set_defaults(func=command_credential_store)

    extension = subparsers.add_parser("extension")
    extension_subparsers = extension.add_subparsers(
        dest="extension_command", required=True
    )
    extension_inventory = extension_subparsers.add_parser("inventory")
    extension_inventory.add_argument("--profile", default=None)
    extension_inventory.set_defaults(func=command_extension_inventory)
    extension_list = extension_subparsers.add_parser("list")
    extension_list.add_argument("--profile", default=None)
    extension_list.set_defaults(func=command_extension_inventory)
    extension_discover = extension_subparsers.add_parser("discover")
    extension_discover.add_argument("--manifest", required=True)
    extension_discover.set_defaults(func=command_extension_discover)
    extension_quarantine = extension_subparsers.add_parser("quarantine")
    extension_quarantine.add_argument("--manifest", required=True)
    extension_quarantine.add_argument("--payload", required=True)
    extension_quarantine.set_defaults(func=command_extension_quarantine)
    extension_inspect = extension_subparsers.add_parser("inspect")
    extension_inspect.add_argument("--id", required=True)
    extension_inspect.set_defaults(func=command_extension_inspect)
    extension_tested = extension_subparsers.add_parser("mark-tested")
    extension_tested.add_argument("--id", required=True)
    extension_tested.add_argument(
        "--sandbox", choices=("container", "bubblewrap"), default="container"
    )
    extension_tested.add_argument("--passed", action="store_true")
    extension_tested.set_defaults(func=command_extension_tested)
    extension_approve = extension_subparsers.add_parser("approve")
    extension_approve.add_argument("--id", required=True)
    extension_approve.add_argument("--sha256", required=True)
    extension_approve.add_argument("--approver", required=True)
    extension_approve.add_argument("--approve", action="store_true")
    extension_approve.set_defaults(func=command_extension_approve)
    extension_activate = extension_subparsers.add_parser("activate")
    extension_activate.add_argument("--id", required=True)
    extension_activate.add_argument("--profile", required=True)
    extension_activate.add_argument("--approve", action="store_true")
    extension_activate.add_argument("--approve-host-risk", action="store_true")
    extension_activate.set_defaults(func=command_extension_activate)
    extension_rollback = extension_subparsers.add_parser("rollback")
    extension_rollback.add_argument("--id", required=True)
    extension_rollback.add_argument("--profile", required=True)
    extension_rollback.add_argument("--approve", action="store_true")
    extension_rollback.set_defaults(func=command_extension_rollback)

    broker = subparsers.add_parser("broker-serve")
    broker.add_argument("--socket")
    broker.set_defaults(func=command_broker_serve)

    mcp = subparsers.add_parser("mcp-serve")
    mcp.add_argument("--profile", default="chunsa")
    mcp.set_defaults(func=command_mcp_serve)

    panel = subparsers.add_parser("panel")
    panel.add_argument("--port", type=int, default=8765)
    panel.set_defaults(func=command_panel)

    harness = subparsers.add_parser("harness-spec")
    harness.add_argument("--port", type=int, default=3080)
    harness.set_defaults(func=command_harness_spec)

    sessions = subparsers.add_parser("sessions-maintain")
    sessions.add_argument("--retention-days", type=int, default=7)
    sessions.add_argument("--apply", action="store_true")
    sessions.set_defaults(func=command_session_maintain)

    component = subparsers.add_parser("component")
    component_subparsers = component.add_subparsers(
        dest="component_command", required=True
    )
    component_inventory = component_subparsers.add_parser("inventory")
    component_inventory.set_defaults(func=command_component_inventory)
    component_list = component_subparsers.add_parser("list")
    component_list.set_defaults(func=command_component_inventory)
    component_stage = component_subparsers.add_parser("stage")
    component_stage.add_argument("--name", required=True)
    component_stage.add_argument("--artifact", required=True)
    component_stage.add_argument("--version")
    component_stage.add_argument("--sha256")
    component_stage.set_defaults(func=command_component_stage)
    component_validate = component_subparsers.add_parser("validate")
    component_validate.add_argument("--name", required=True)
    component_validate.add_argument("--compatibility-passed", action="store_true")
    component_validate.add_argument("--security-passed", action="store_true")
    component_validate.add_argument("--evaluation-passed", action="store_true")
    component_validate.set_defaults(func=command_component_validate)
    component_promote = component_subparsers.add_parser("promote")
    component_promote.add_argument("--name", required=True)
    component_promote.add_argument("--approve", action="store_true")
    component_promote.set_defaults(func=command_component_promote)
    component_rollback = component_subparsers.add_parser("rollback")
    component_rollback.add_argument("--name", required=True)
    component_rollback.add_argument("--approve", action="store_true")
    component_rollback.set_defaults(func=command_component_rollback)
    component_prune = component_subparsers.add_parser("prune")
    component_prune.add_argument("--name", required=True)
    component_prune.add_argument("--approve", action="store_true")
    component_prune.set_defaults(func=command_component_prune)

    client_config = subparsers.add_parser("client-config")
    client_config.add_argument(
        "--client",
        required=True,
        choices=(
            "harness",
            "codex",
            "claude",
            "kimi",
            "qwen",
            "opencode",
            "abacus",
            "fx",
        ),
    )
    client_config.add_argument("--profile", default="default")
    client_config.add_argument("--output")
    client_config.set_defaults(func=command_client_config)

    capability = subparsers.add_parser("capability")
    capability.add_argument("--package", required=True)
    capability.add_argument(
        "--client",
        required=True,
        choices=(
            "harness",
            "codex",
            "claude",
            "kimi",
            "qwen",
            "opencode",
            "abacus",
            "fx",
        ),
    )
    capability.set_defaults(func=command_capability_select)

    task = subparsers.add_parser("task")
    task_subparsers = task.add_subparsers(dest="task_command", required=True)
    for name in ("submit", "run"):
        task_run = task_subparsers.add_parser(name)
        task_run.add_argument("--spec", required=True)
        task_run.add_argument("--profile", default="default")
        task_run.add_argument("--model", default=None)
        task_run.add_argument("--confirmed-plan-id", default=None)
        task_run.add_argument("--output")
        task_run.set_defaults(func=command_run)

    model = subparsers.add_parser("model")
    model_subparsers = model.add_subparsers(dest="model_command", required=True)
    model_list = model_subparsers.add_parser("list")
    model_list.set_defaults(func=command_model_list)
    model_route = model_subparsers.add_parser("route")
    model_route.add_argument("--spec", required=True)
    model_route.add_argument("--profile", default="default")
    model_route.add_argument("--model", default=None)
    model_route.set_defaults(func=command_route_plan)
    model_canary = model_subparsers.add_parser("canary")
    model_canary.add_argument("--provider", action="append", choices=sorted(ROUTES))
    model_canary.add_argument("--timeout", type=int, default=120)
    model_canary.add_argument("--output")
    model_canary.set_defaults(func=command_provider_canary)

    client = subparsers.add_parser("client")
    client_subparsers = client.add_subparsers(dest="client_command", required=True)
    client_list = client_subparsers.add_parser("list")
    client_list.set_defaults(func=command_client_list)
    client_install = client_subparsers.add_parser("install")
    client_install.add_argument(
        "--client", required=True, choices=sorted(SUPPORTED_CLIENTS)
    )
    client_install.add_argument("--profile", default="default")
    client_install.add_argument("--output", required=True)
    client_install.set_defaults(func=command_client_config)
    client_doctor = client_subparsers.add_parser("doctor")
    client_doctor.add_argument("--profile", default="default")
    client_doctor.add_argument(
        "--check", choices=("readiness", "compliance", "all"), default="all"
    )
    client_doctor.set_defaults(func=command_doctor)
    client_canary = client_subparsers.add_parser("canary")
    client_canary.add_argument("--provider", action="append", choices=sorted(ROUTES))
    client_canary.add_argument("--timeout", type=int, default=120)
    client_canary.add_argument("--output")
    client_canary.set_defaults(func=command_provider_canary)

    mcp_family = subparsers.add_parser("mcp")
    mcp_subparsers = mcp_family.add_subparsers(dest="mcp_command", required=True)
    mcp_status = mcp_subparsers.add_parser("status")
    mcp_status.add_argument("--profile", default="chunsa")
    mcp_status.set_defaults(func=command_mcp_status)
    mcp_install = mcp_subparsers.add_parser("install")
    mcp_install.add_argument(
        "--client", required=True, choices=sorted(SUPPORTED_MCP_CLIENTS | {"kimi"})
    )
    mcp_install.add_argument("--profile", default="chunsa")
    mcp_install.set_defaults(func=command_mcp_install)
    mcp_serve = mcp_subparsers.add_parser("serve")
    mcp_serve.add_argument("--profile", default="chunsa")
    mcp_serve.set_defaults(func=command_mcp_serve)

    harness_family = subparsers.add_parser("harness")
    harness_subparsers = harness_family.add_subparsers(
        dest="harness_command", required=True
    )
    harness_status = harness_subparsers.add_parser("status")
    harness_status.add_argument("--port", type=int, default=3080)
    harness_status.set_defaults(func=command_harness_status)
    harness_serve = harness_subparsers.add_parser("serve")
    harness_serve.add_argument(
        "--profile",
        choices=("orchestrator-web", "orchestrator-worker"),
        default="orchestrator-web",
    )
    harness_serve.add_argument("--port", type=int, default=3080)
    harness_serve.add_argument("--workspace", default=str(Path.cwd()))
    harness_serve.set_defaults(func=command_harness_serve)
    harness_worker = harness_subparsers.add_parser("worker")
    harness_worker.add_argument(
        "--model", choices=("deepseek-v4-pro", "deepseek-v4-flash"), required=True
    )
    harness_worker.add_argument("--timeout", type=int, default=3600)
    harness_worker.add_argument("--extended", action="store_true")
    harness_worker.add_argument(
        "--orchestrated-worktree", action="store_true", help=argparse.SUPPRESS
    )
    harness_worker.set_defaults(func=command_harness_worker)
    harness_update = harness_subparsers.add_parser("update")
    harness_update.add_argument(
        "--source",
        default=str(
            Path(__file__).resolve().parents[2] / "integrations" / "deepseek-harness"
        ),
    )
    harness_update.add_argument("--version", default="latest")
    harness_update.set_defaults(func=command_harness_update)
    harness_rollback = harness_subparsers.add_parser("rollback")
    harness_rollback.add_argument("--approve", action="store_true")
    harness_rollback.set_defaults(
        func=command_component_rollback, name="deepseek-harness"
    )
    harness_workspace = harness_subparsers.add_parser("workspace")
    workspace_subparsers = harness_workspace.add_subparsers(
        dest="workspace_command", required=True
    )
    workspace_list = workspace_subparsers.add_parser("list")
    workspace_list.set_defaults(func=command_harness_workspace)
    for action in ("add", "remove"):
        workspace_action = workspace_subparsers.add_parser(action)
        workspace_action.add_argument("path")
        workspace_action.add_argument("--approve", action="store_true")
        workspace_action.add_argument("--drop-in")
        workspace_action.set_defaults(func=command_harness_workspace)

    docs = subparsers.add_parser("docs")
    docs_subparsers = docs.add_subparsers(dest="docs_command", required=True)
    for docs_action in ("generate", "check"):
        docs_command = docs_subparsers.add_parser(docs_action)
        docs_command.add_argument("--root")
        docs_command.set_defaults(func=command_docs)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ContractError, AdapterError, ArchiveError, ValueError, RuntimeError) as exc:
        code = getattr(exc, "code", type(exc).__name__)
        _print({"ok": False, "error_code": code, "message": str(exc)})
        return EXIT_POLICY


if __name__ == "__main__":
    raise SystemExit(main())
