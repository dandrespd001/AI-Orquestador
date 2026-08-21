"""Fail-closed admission gate for optional DSPy prompt experiments."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ai_orchestrator.models import ContractError

DSPY_VERSION = "3.3.0"
ALLOWED_OPTIMIZERS = {"BootstrapFewShot", "MIPROv2"}
ALLOWED_ROLES = {"tests", "docs", "review"}


@dataclass(frozen=True)
class PromptExperiment:
    engine: str
    engine_version: str
    role: str
    optimizer: str
    dataset: str
    max_calls: int
    max_labeled_demos: int
    max_bootstrapped_demos: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float


def load_prompt_experiment(
    path: Path,
) -> tuple[PromptExperiment, Path, list[dict[str, Any]]]:
    if path.is_symlink() or not path.is_file():
        raise ContractError("prompt experiment must be a regular TOML file")
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if raw.get("schema_version") != 1:
        raise ContractError("unsupported prompt experiment schema_version")
    experiment = PromptExperiment(
        engine=str(raw.get("engine", "")),
        engine_version=str(raw.get("engine_version", "")),
        role=str(raw.get("role", "")),
        optimizer=str(raw.get("optimizer", "")),
        dataset=str(raw.get("dataset", "")),
        max_calls=int(raw.get("max_calls", 0)),
        max_labeled_demos=int(raw.get("max_labeled_demos", 0)),
        max_bootstrapped_demos=int(raw.get("max_bootstrapped_demos", 0)),
        train_fraction=float(raw.get("train_fraction", 0)),
        validation_fraction=float(raw.get("validation_fraction", 0)),
        test_fraction=float(raw.get("test_fraction", 0)),
    )
    if experiment.engine != "dspy" or experiment.engine_version != DSPY_VERSION:
        raise ContractError("experiment must pin the approved DSPy version")
    if (
        experiment.role not in ALLOWED_ROLES
        or experiment.optimizer not in ALLOWED_OPTIMIZERS
    ):
        raise ContractError("experiment role or optimizer is not approved")
    if not 1 <= experiment.max_calls <= 40:
        raise ContractError("experiment max_calls must be between 1 and 40")
    if (
        not 0 <= experiment.max_labeled_demos <= 4
        or not 0 <= experiment.max_bootstrapped_demos <= 4
    ):
        raise ContractError("experiment demo limits must not exceed four")
    fractions = (
        experiment.train_fraction
        + experiment.validation_fraction
        + experiment.test_fraction
    )
    if (
        abs(fractions - 1.0) > 1e-9
        or min(
            experiment.train_fraction,
            experiment.validation_fraction,
            experiment.test_fraction,
        )
        <= 0
    ):
        raise ContractError(
            "experiment split fractions must be positive and sum to one"
        )

    relative = Path(experiment.dataset)
    if relative.is_absolute() or ".." in relative.parts:
        raise ContractError("experiment dataset must be manifest-relative")
    dataset = path.parent / relative
    if dataset.is_symlink() or not dataset.is_file():
        raise ContractError("experiment dataset is missing or unsafe")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        dataset.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ContractError(f"dataset line {line_number} must be an object")
        case_id = str(row.get("case_id", ""))
        if not case_id or case_id in seen:
            raise ContractError(
                f"dataset case_id missing or duplicate at line {line_number}"
            )
        if not isinstance(row.get("input"), str) or not isinstance(
            row.get("expected"), str
        ):
            raise ContractError(
                f"dataset line {line_number} requires input and expected strings"
            )
        seen.add(case_id)
        rows.append(row)
    if len(rows) < 40:
        raise ContractError("DSPy experiment requires at least 40 cases")
    return experiment, dataset, rows


def prompt_optimization_preflight(
    experiment_path: Path, baseline_path: Path
) -> dict[str, Any]:
    experiment, dataset, rows = load_prompt_experiment(experiment_path)
    if baseline_path.is_symlink() or not baseline_path.is_file():
        raise ContractError("baseline receipt must be a regular JSON file")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if not isinstance(baseline, dict) or baseline.get("status") != "PASSED":
        raise ContractError("DSPy optimization requires a passing quality baseline")
    try:
        installed = importlib.metadata.version("dspy")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    admitted = installed == DSPY_VERSION
    return {
        "schema_version": 1,
        "kind": "prompt_optimization_preflight",
        "experiment": asdict(experiment),
        "experiment_sha256": hashlib.sha256(experiment_path.read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "case_count": len(rows),
        "installed_dspy": installed,
        "incremental_spend_allowed": False,
        "live_or_financial_inputs_allowed": False,
        "status": "READY" if admitted else "DEPENDENCY_UNAVAILABLE",
    }
