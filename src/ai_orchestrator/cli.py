from __future__ import annotations

import argparse
import getpass
import json
import shutil
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ai_orchestrator.adapters import AdapterError, execute
from ai_orchestrator.admission import AdmissionController
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
from ai_orchestrator.client_catalog import generate_client_config, write_client_config
from ai_orchestrator.config import (
    config_root,
    load_gates,
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
    minimax_runtime_key,
    omniroute_runtime_key,
)
from ai_orchestrator.evals import run_routing_eval
from ai_orchestrator.extensions import ExtensionRegistry, extension_manifest_from_file
from ai_orchestrator.harness import harness_launch_spec, validate_harness_launch
from ai_orchestrator.mcp_tools import serve_stdio
from ai_orchestrator.models import ContractError, TaskSpec
from ai_orchestrator.optimization import prompt_optimization_preflight
from ai_orchestrator.quality import promptfoo_preflight, run_quality_eval
from ai_orchestrator.routing import plan_route, select_route
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
from ai_orchestrator.telemetry import TelemetryStore
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
    decision = select_route(
        task,
        models,
        roles,
        profile,
        observations=telemetry.routing_observations(),
        pin=pin,
    )
    if record:
        telemetry.record_route(task, decision)
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
        "minimax": minimax_runtime_key() is not None,
        "omniroute": omniroute_runtime_key() is not None,
    }
    payload = {
        "schema_version": 1,
        "ok": (
            configuration["ok"]
            and all(commands.values())
            and storage["ok"]
            and sync["in_sync"]
            and all(credentials.values())
        ),
        "configuration": configuration,
        "commands": commands,
        "omniroute": _local_health(),
        "credentials": credentials,
        "config_sync": sync,
        "storage": storage,
    }
    _print(payload)
    return 0 if payload["ok"] else EXIT_POLICY


def command_route(args: argparse.Namespace) -> int:
    _, decision = _route(
        args.spec, args.profile, record=not args.no_record, pin=args.model
    )
    _print(decision.as_dict())
    return 0 if decision.status != "NO_ROUTE" else EXIT_POLICY


def command_run(args: argparse.Namespace) -> int:
    task, decision = _route(args.spec, args.profile, record=True, pin=args.model)
    if not decision.auto_execute:
        _print({"decision": decision.as_dict(), "status": "HANDOFF_REQUIRED"})
        return EXIT_POLICY
    output = Path(args.output).expanduser() if args.output else None
    profile = load_profile(args.profile)
    if task.gate_profile == "profile":
        gate_commands = profile.deterministic_gates
        gate_timeout_seconds = task.timeout_seconds
    else:
        gate_profiles = load_gates()
        if task.gate_profile not in gate_profiles:
            raise ContractError(f"unknown gate_profile: {task.gate_profile}")
        gate_profile = gate_profiles[task.gate_profile]
        gate_commands = gate_profile.commands
        gate_timeout_seconds = gate_profile.timeout_seconds
    admission = AdmissionController()
    with admission.acquire(
        model=str(decision.selected_model),
        max_concurrency=decision.max_concurrency,
        repo=task.repo,
        allow_parallel=task.allow_parallel,
        lease_seconds=task.timeout_seconds + 60,
        wait_timeout_seconds=min(30, task.timeout_seconds),
    ):
        result, stdout = execute(
            task,
            decision,
            output_path=output,
            gate_commands=gate_commands,
            gate_timeout_seconds=gate_timeout_seconds,
            protected_prefixes=profile.protected_path_prefixes,
        )
    TelemetryStore().record_run(result)
    payload: dict[str, object] = {
        "decision": decision.as_dict(),
        "run": result.as_dict(),
    }
    if not args.output and stdout:
        payload["output"] = stdout.decode("utf-8", "replace")
    _print(payload)
    return 0 if result.status in {"SUCCESS", "APPLIED", "PATCH_READY"} else EXIT_RUNTIME


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
    _print(
        {
            "schema_version": 1,
            "models": [
                {
                    "id": model.id,
                    "provider": model.provider,
                    "transport": model.transport,
                    "status": model.status,
                    "incremental_spend_allowed": False,
                }
                for model in models.values()
            ],
        }
    )
    return 0


def command_storage_audit(args: argparse.Namespace) -> int:
    payload = audit_storage()
    _print(payload)
    return 0 if payload["ok"] else EXIT_POLICY


def command_storage_maintain(args: argparse.Namespace) -> int:
    actions: dict[str, object] = {}
    if args.apply_permissions:
        actions["permissions"] = apply_private_permissions()
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
        actions["compaction"] = compact_sqlite(path, client)

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
    task = TaskSpec.read(args.spec)
    result = plan_route(
        task,
        load_models(),
        load_roles(),
        load_profile(args.profile),
        pin=args.model,
    )
    _print(result.as_dict())
    return 0 if result.selected_model else EXIT_POLICY


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
    _print(ExtensionRegistry().inventory(profile=args.profile))
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
    serve_stdio()
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
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor")
    doctor.add_argument("--profile", default="default")
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
