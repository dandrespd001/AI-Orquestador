"""Deterministic, metadata-only HTTP gateway benchmark harness."""

from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import stat
import statistics
import tempfile
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_orchestrator.models import ContractError, ensure_private_directory


@dataclass(frozen=True)
class GatewayTarget:
    name: str
    version: str
    base_url: str
    official_providers: tuple[str, ...]
    provider_coverage_attested: bool
    custom_provider_code: bool
    content_logging: bool
    immutable_ref: str


@dataclass(frozen=True)
class BenchmarkScenario:
    name: str
    path: str
    payload: bytes
    expected_status: int
    expected_model: str | None
    concurrency: int


@dataclass(frozen=True)
class Observation:
    status: int | None
    first_byte_ms: float
    total_ms: float
    response_bytes: int
    response_sha256: str | None
    reported_model: str | None
    error_code: str | None


def _safe_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return round(ordered[index], 3)


def _reported_model(value: Any) -> str | None:
    if isinstance(value, dict):
        candidate = value.get("model")
        if isinstance(candidate, str) and candidate:
            return candidate
        for nested in value.values():
            found = _reported_model(nested)
            if found:
                return found
    if isinstance(value, list):
        for nested in value:
            found = _reported_model(nested)
            if found:
                return found
    return None


def _require_loopback(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if (
        parsed.scheme != "http"
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ContractError("gateway base_url must be an unauthenticated http URL")
    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(parsed.hostname, parsed.port or 80)
        }
    except socket.gaierror as exc:
        raise ContractError("gateway base_url host cannot be resolved") from exc
    if not addresses or not all(
        address.startswith("127.") or address == "::1" for address in addresses
    ):
        raise ContractError("gateway benchmark only permits loopback targets")
    return url.rstrip("/")


def _read_payload(manifest: Path, value: object) -> bytes:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ContractError("payload_file must be manifest-relative")
    path = manifest.parent / relative
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"missing or unsafe payload_file: {relative}")
    payload = path.read_bytes()
    if not payload or len(payload) > 1_048_576:
        raise ContractError("payload_file must contain 1..1048576 bytes")
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("payload_file must contain valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ContractError("payload_file must contain a JSON object")
    return payload


def load_benchmark_manifest(
    path: Path,
) -> tuple[list[GatewayTarget], list[BenchmarkScenario], int, int, float]:
    if path.is_symlink() or not path.is_file():
        raise ContractError("benchmark manifest must be a regular TOML file")
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if raw.get("schema_version") != 1:
        raise ContractError("unsupported benchmark manifest schema_version")
    repetitions = int(raw.get("repetitions", 30))
    warmups = int(raw.get("warmups", 5))
    timeout = float(raw.get("timeout_seconds", 10.0))
    if (
        not 1 <= repetitions <= 100
        or not 0 <= warmups <= 20
        or not 0.1 <= timeout <= 60
    ):
        raise ContractError("benchmark limits are outside the permitted range")

    targets: list[GatewayTarget] = []
    seen_targets: set[str] = set()
    for row in raw.get("targets", []):
        if not isinstance(row, dict):
            raise ContractError("target must be a TOML table")
        name = str(row.get("name", "")).strip()
        version = str(row.get("version", "")).strip()
        if not name or name in seen_targets or not version:
            raise ContractError("target names must be unique and versioned")
        seen_targets.add(name)
        providers = tuple(
            sorted({str(value) for value in row.get("official_providers", ())})
        )
        immutable_ref = str(row.get("immutable_ref", "")).strip()
        if not immutable_ref:
            raise ContractError("target immutable_ref is required")
        targets.append(
            GatewayTarget(
                name=name,
                version=version,
                base_url=_require_loopback(str(row["base_url"])),
                official_providers=providers,
                provider_coverage_attested=bool(
                    row.get("provider_coverage_attested", False)
                ),
                custom_provider_code=bool(row.get("custom_provider_code", False)),
                content_logging=bool(row.get("content_logging", True)),
                immutable_ref=immutable_ref,
            )
        )

    scenarios: list[BenchmarkScenario] = []
    seen_scenarios: set[str] = set()
    for row in raw.get("scenarios", []):
        if not isinstance(row, dict):
            raise ContractError("scenario must be a TOML table")
        name = str(row.get("name", "")).strip()
        route = str(row.get("path", "")).strip()
        concurrency = int(row.get("concurrency", 1))
        if (
            not name
            or name in seen_scenarios
            or not route.startswith("/")
            or route.startswith("//")
        ):
            raise ContractError("scenario names and paths must be unique and safe")
        if concurrency not in {1, 4, 16}:
            raise ContractError("scenario concurrency must be 1, 4, or 16")
        seen_scenarios.add(name)
        scenarios.append(
            BenchmarkScenario(
                name=name,
                path=route,
                payload=_read_payload(path, row["payload_file"]),
                expected_status=int(row.get("expected_status", 200)),
                expected_model=str(row["expected_model"])
                if row.get("expected_model")
                else None,
                concurrency=concurrency,
            )
        )
    if not targets or not scenarios:
        raise ContractError("benchmark requires at least one target and one scenario")
    return targets, scenarios, repetitions, warmups, timeout


def _request(
    target: GatewayTarget, scenario: BenchmarkScenario, timeout: float
) -> Observation:
    request = urllib.request.Request(
        target.base_url + scenario.path,
        data=scenario.payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-AI-Orchestrator-Benchmark": "1",
        },
    )
    start = time.perf_counter_ns()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            first = response.read(1)
            first_byte = (time.perf_counter_ns() - start) / 1_000_000
            body = first + response.read()
            total = (time.perf_counter_ns() - start) / 1_000_000
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        first_byte = (time.perf_counter_ns() - start) / 1_000_000
        body = exc.read()
        total = (time.perf_counter_ns() - start) / 1_000_000
        status = int(exc.code)
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        elapsed = (time.perf_counter_ns() - start) / 1_000_000
        return Observation(None, elapsed, elapsed, 0, None, None, type(exc).__name__)

    model = None
    try:
        model = _reported_model(json.loads(body))
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    return Observation(
        status=status,
        first_byte_ms=round(first_byte, 3),
        total_ms=round(total, 3),
        response_bytes=len(body),
        response_sha256=hashlib.sha256(body).hexdigest(),
        reported_model=model,
        error_code=None,
    )


def _summarize(
    target: GatewayTarget, scenario: BenchmarkScenario, rows: list[Observation]
) -> dict[str, Any]:
    correct = sum(
        row.status == scenario.expected_status
        and row.error_code is None
        and (
            scenario.expected_model is None
            or row.reported_model == scenario.expected_model
        )
        for row in rows
    )
    first_bytes = [row.first_byte_ms for row in rows]
    totals = [row.total_ms for row in rows]
    return {
        "target": target.name,
        "version": target.version,
        "scenario": scenario.name,
        "concurrency": scenario.concurrency,
        "request_sha256": hashlib.sha256(scenario.payload).hexdigest(),
        "observations": len(rows),
        "correct": correct,
        "correctness_rate": correct / len(rows),
        "first_byte_ms": {
            "p50": round(statistics.median(first_bytes), 3),
            "p95": _safe_percentile(first_bytes, 0.95),
        },
        "total_ms": {
            "p50": round(statistics.median(totals), 3),
            "p95": _safe_percentile(totals, 0.95),
        },
        "response_bytes": sorted({row.response_bytes for row in rows}),
        "response_sha256": sorted(
            {row.response_sha256 for row in rows if row.response_sha256}
        ),
        "errors": sorted({row.error_code for row in rows if row.error_code}),
        "status": "PASSED" if correct == len(rows) else "FAILED",
    }


def run_gateway_benchmark(manifest: Path) -> dict[str, Any]:
    targets, scenarios, repetitions, warmups, timeout = load_benchmark_manifest(
        manifest
    )
    results: list[dict[str, Any]] = []
    for target in targets:
        for scenario in scenarios:
            for _ in range(warmups):
                _request(target, scenario, timeout)
            with ThreadPoolExecutor(max_workers=scenario.concurrency) as pool:
                rows = list(
                    pool.map(
                        lambda _, current_target=target, current_scenario=scenario: (
                            _request(current_target, current_scenario, timeout)
                        ),
                        range(repetitions),
                    )
                )
            results.append(_summarize(target, scenario, rows))
    target_gates: list[dict[str, Any]] = []
    required_providers = {"deepseek", "moonshot", "minimax"}
    for target in targets:
        target_results = [row for row in results if row["target"] == target.name]
        reasons: list[str] = []
        if not all(row["status"] == "PASSED" for row in target_results):
            reasons.append("CONTRACT_FAILURE")
        if not required_providers.issubset(target.official_providers):
            reasons.append("PROVIDER_COVERAGE")
        if not target.provider_coverage_attested:
            reasons.append("PROVIDER_COVERAGE_UNATTESTED")
        if target.custom_provider_code:
            reasons.append("CUSTOM_PROVIDER_CODE")
        if target.content_logging:
            reasons.append("CONTENT_LOGGING")
        target_gates.append(
            {
                "target": target.name,
                "version": target.version,
                "immutable_ref": target.immutable_ref,
                "official_providers": list(target.official_providers),
                "provider_coverage_attested": target.provider_coverage_attested,
                "promotion_eligible": not reasons,
                "rejection_reasons": reasons,
            }
        )
    passed = all(row["status"] == "PASSED" for row in results)
    return {
        "schema_version": 1,
        "kind": "gateway_benchmark",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "repetitions": repetitions,
        "warmups": warmups,
        "results": results,
        "target_gates": target_gates,
        "status": "PASSED" if passed else "FAILED",
    }


def select_gateway(report: dict[str, Any], incumbent: str) -> dict[str, Any]:
    """Select only a materially faster fully eligible replacement."""

    if report.get("kind") != "gateway_benchmark" or report.get("status") != "PASSED":
        raise ContractError("gateway selection requires a passing benchmark receipt")
    gates = {str(row["target"]): row for row in report.get("target_gates", [])}
    if incumbent not in gates:
        raise ContractError("incumbent is absent from benchmark receipt")
    rows = report.get("results", [])
    incumbent_rows = {
        str(row["scenario"]): row for row in rows if row.get("target") == incumbent
    }
    if not incumbent_rows:
        raise ContractError("incumbent has no benchmark observations")
    candidates: list[dict[str, Any]] = []
    for name, gate in gates.items():
        if name == incumbent or not gate.get("promotion_eligible"):
            continue
        candidate_rows = {
            str(row["scenario"]): row for row in rows if row.get("target") == name
        }
        if set(candidate_rows) != set(incumbent_rows):
            continue
        ratios = [
            float(candidate_rows[key]["total_ms"]["p95"])
            / max(float(incumbent_rows[key]["total_ms"]["p95"]), 0.001)
            for key in sorted(incumbent_rows)
        ]
        candidates.append(
            {
                "target": name,
                "median_p95_ratio": round(statistics.median(ratios), 4),
                "worst_p95_ratio": round(max(ratios), 4),
            }
        )
    qualified = [
        row
        for row in candidates
        if row["median_p95_ratio"] <= 0.8 and row["worst_p95_ratio"] <= 1.05
    ]
    winner = min(qualified, key=lambda row: row["median_p95_ratio"], default=None)
    return {
        "schema_version": 1,
        "kind": "gateway_selection",
        "incumbent": incumbent,
        "selected": winner["target"] if winner else incumbent,
        "replacement": winner is not None,
        "candidates": candidates,
        "reason": "MATERIAL_P95_IMPROVEMENT"
        if winner
        else "NO_MATERIAL_ELIGIBLE_IMPROVEMENT",
    }


def atomic_private_json(path: Path, payload: dict[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    parent = resolved.parent
    if not parent.exists():
        ensure_private_directory(parent)
    elif parent.is_symlink() or not parent.is_dir():
        raise ContractError("output parent must be a regular directory")
    elif stat.S_IMODE(parent.stat().st_mode) & 0o077:
        raise ContractError("output parent must not be group/world accessible")
    if resolved.exists() and resolved.is_symlink():
        raise ContractError("output must not be a symlink")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{resolved.name}.", dir=parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
        os.chmod(resolved, 0o600)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
