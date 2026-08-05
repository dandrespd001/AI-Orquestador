from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from ai_orchestrator.adapters import AdapterError, execute
from ai_orchestrator.archive import ArchiveError, create_encrypted_archive
from ai_orchestrator.benchmark import atomic_private_json, run_gateway_benchmark, select_gateway
from ai_orchestrator.config import (
    load_models,
    load_profile,
    load_roles,
    validate_configuration,
)
from ai_orchestrator.evals import run_routing_eval
from ai_orchestrator.models import ContractError, TaskSpec
from ai_orchestrator.optimization import prompt_optimization_preflight
from ai_orchestrator.quality import promptfoo_preflight, run_quality_eval
from ai_orchestrator.routing import select_route
from ai_orchestrator.storage import (
    SQLITE_TARGETS,
    apply_private_permissions,
    audit_storage,
    bound_sqlite_journals,
    compact_sqlite,
    enforce_official_omniroute_providers,
    storage_receipt,
)
from ai_orchestrator.telemetry import TelemetryStore


EXIT_POLICY = 4
EXIT_RUNTIME = 5


def _print(payload: object) -> None:
    print(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False))


def _route(spec_path: str, profile_name: str, *, record: bool) -> tuple[TaskSpec, Any]:
    task = TaskSpec.read(spec_path)
    models = load_models()
    roles = load_roles()
    profile = load_profile(profile_name)
    decision = select_route(task, models, roles, profile)
    if record:
        TelemetryStore().record_route(task, decision)
    return task, decision


def _local_health() -> dict[str, object]:
    request = urllib.request.Request("http://127.0.0.1:20128/health", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=1.5) as response:
            return {"reachable": True, "status": response.status}
    except (urllib.error.URLError, TimeoutError, ConnectionError, socket.timeout):
        return {"reachable": False, "status": None}


def command_doctor(args: argparse.Namespace) -> int:
    configuration = validate_configuration(args.profile)
    commands = {
        name: shutil.which(name)
        for name in ("codex", "claude", "omniroute", "gpg", "zstd", "tar", "systemd-creds")
    }
    storage = audit_storage()
    payload = {
        "schema_version": 1,
        "ok": configuration["ok"] and all(commands.values()) and storage["ok"],
        "configuration": configuration,
        "commands": commands,
        "omniroute": _local_health(),
        "storage": storage,
    }
    _print(payload)
    return 0 if payload["ok"] else EXIT_POLICY


def command_route(args: argparse.Namespace) -> int:
    _, decision = _route(args.spec, args.profile, record=not args.no_record)
    _print(decision.as_dict())
    return 0 if decision.status != "NO_ROUTE" else EXIT_POLICY


def command_run(args: argparse.Namespace) -> int:
    task, decision = _route(args.spec, args.profile, record=True)
    if not decision.auto_execute:
        _print({"decision": decision.as_dict(), "status": "HANDOFF_REQUIRED"})
        return EXIT_POLICY
    output = Path(args.output).expanduser() if args.output else None
    result, stdout = execute(task, decision, output_path=output)
    TelemetryStore().record_run(result)
    payload: dict[str, object] = {"decision": decision.as_dict(), "run": result.as_dict()}
    if not args.output and stdout:
        payload["output"] = stdout.decode("utf-8", "replace")
    _print(payload)
    return 0 if result.status == "SUCCESS" else EXIT_RUNTIME


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
        matched = [(path, client) for path, client in SQLITE_TARGETS if str(path) == args.compact]
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
        actions["official_providers"] = enforce_official_omniroute_providers(apply=args.apply)

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


def command_gateway_select(args: argparse.Namespace) -> int:
    source = Path(args.receipt).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise ContractError("gateway receipt must be a regular JSON file")
    payload = select_gateway(json.loads(source.read_text(encoding="utf-8")), args.incumbent)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ai-orchestrator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor")
    doctor.add_argument("--profile", default="default")
    doctor.set_defaults(func=command_doctor)

    route = subparsers.add_parser("route")
    route.add_argument("--spec", required=True)
    route.add_argument("--profile", default="default")
    route.add_argument("--no-record", action="store_true")
    route.set_defaults(func=command_route)

    run = subparsers.add_parser("run")
    run.add_argument("--spec", required=True)
    run.add_argument("--profile", default="default")
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
    quality_preflight.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    quality_preflight.set_defaults(func=command_quality_preflight)

    quality_eval = subparsers.add_parser("quality-eval")
    quality_eval.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    quality_eval.add_argument("--config", required=True)
    quality_eval.add_argument("--output", required=True)
    quality_eval.add_argument("--timeout", type=int, default=900)
    quality_eval.set_defaults(func=command_quality_eval)

    optimize = subparsers.add_parser("prompt-optimize")
    optimize.add_argument("--experiment", required=True)
    optimize.add_argument("--baseline", required=True)
    optimize.add_argument("--output", required=True)
    optimize.set_defaults(func=command_prompt_optimize)
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
