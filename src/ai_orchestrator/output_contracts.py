"""Validation for model deliverables before gates or reviewer acceptance."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from ai_orchestrator.models import TaskSpec

REVIEW_VERDICTS = {"accept", "revise", "escalate"}
FINDING_SEVERITIES = {"info", "low", "medium", "high", "critical"}


@dataclass(frozen=True)
class OutputValidation:
    valid: bool
    contract: str
    error_code: str | None = None
    parsed: Any = None


def validate_output(task: TaskSpec, output: bytes) -> OutputValidation:
    """Validate bytes against the declared contract without retaining content."""

    try:
        text = output.decode("utf-8")
    except UnicodeDecodeError:
        return OutputValidation(False, task.output_contract, "INVALID_UTF8")
    if not text.strip():
        return OutputValidation(False, task.output_contract, "EMPTY_OUTPUT")
    if len(re.findall(r"\S+", text)) > task.max_output_tokens:
        return OutputValidation(False, task.output_contract, "OUTPUT_TOKEN_LIMIT")

    if task.output_contract == "text":
        return OutputValidation(True, "text", parsed=text)
    if task.output_contract in {"json", "review"}:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return OutputValidation(False, task.output_contract, "MALFORMED_JSON")
        if task.output_contract == "review":
            return _validate_review(payload)
        if task.output_schema_file is not None:
            try:
                schema = json.loads(
                    Path(task.repo, task.output_schema_file).read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, json.JSONDecodeError):
                return OutputValidation(False, "json", "INVALID_OUTPUT_SCHEMA")
            if not isinstance(schema, dict) or not _matches_schema(payload, schema):
                return OutputValidation(False, "json", "SCHEMA_VIOLATION")
        return OutputValidation(True, "json", parsed=payload)
    if task.output_contract == "unified_diff":
        return _validate_diff(task, output, text)
    return OutputValidation(False, task.output_contract, "UNKNOWN_OUTPUT_CONTRACT")


def _validate_review(payload: Any) -> OutputValidation:
    if not isinstance(payload, dict):
        return OutputValidation(False, "review", "INVALID_REVIEW")
    verdict = payload.get("verdict")
    findings = payload.get("findings")
    if verdict not in REVIEW_VERDICTS or not isinstance(findings, list):
        return OutputValidation(False, "review", "INVALID_REVIEW")
    for finding in findings:
        if not isinstance(finding, dict):
            return OutputValidation(False, "review", "INVALID_REVIEW")
        if finding.get("severity") not in FINDING_SEVERITIES:
            return OutputValidation(False, "review", "INVALID_REVIEW")
        if not isinstance(finding.get("evidence"), str) or not finding[
            "evidence"
        ].strip():
            return OutputValidation(False, "review", "INVALID_REVIEW")
        if not isinstance(finding.get("message"), str) or not finding[
            "message"
        ].strip():
            return OutputValidation(False, "review", "INVALID_REVIEW")
    return OutputValidation(True, "review", parsed=payload)


def reviewer_accepts(validation: OutputValidation) -> bool:
    if not validation.valid or validation.contract != "review":
        return False
    payload = validation.parsed
    if not isinstance(payload, dict) or payload.get("verdict") != "accept":
        return False
    findings = payload.get("findings", ())
    return not any(
        isinstance(finding, dict) and finding.get("severity") == "critical"
        for finding in findings
    )


def _validate_diff(task: TaskSpec, output: bytes, text: str) -> OutputValidation:
    if len(output) > task.max_patch_bytes or not text.startswith("diff --git "):
        return OutputValidation(False, "unified_diff", "INVALID_DIFF")
    for line in text.splitlines():
        if not line.startswith("diff --git a/"):
            continue
        parts = line.split()
        if len(parts) != 4 or not parts[2].startswith("a/") or not parts[3].startswith(
            "b/"
        ):
            return OutputValidation(False, "unified_diff", "INVALID_DIFF")
        for raw in (parts[2][2:], parts[3][2:]):
            path = PurePosixPath(raw)
            if path.is_absolute() or ".." in path.parts or not raw:
                return OutputValidation(False, "unified_diff", "INVALID_DIFF")
    try:
        outcome = subprocess.run(
            ["git", "apply", "--check", "--whitespace=nowarn", "-"],
            input=output,
            cwd=task.repo,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=min(30, task.timeout_seconds),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return OutputValidation(False, "unified_diff", "DIFF_CHECK_FAILED")
    if outcome.returncode != 0:
        return OutputValidation(False, "unified_diff", "DIFF_NOT_APPLICABLE")
    return OutputValidation(True, "unified_diff", parsed={"applicable": True})


def _matches_schema(value: Any, schema: dict[str, Any]) -> bool:
    if "enum" in schema and value not in schema["enum"]:
        return False
    expected_type = schema.get("type")
    if expected_type is not None and not _has_json_type(value, expected_type):
        return False
    if isinstance(value, dict):
        required = schema.get("required", ())
        if not isinstance(required, list) or any(key not in value for key in required):
            return False
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            return False
        if schema.get("additionalProperties") is False and any(
            key not in properties for key in value
        ):
            return False
        for key, nested in properties.items():
            if key in value and (
                not isinstance(nested, dict) or not _matches_schema(value[key], nested)
            ):
                return False
    if isinstance(value, list) and "items" in schema:
        nested = schema["items"]
        if not isinstance(nested, dict) or any(
            not _matches_schema(item, nested) for item in value
        ):
            return False
    if isinstance(value, str):
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            return False
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            return False
        if "pattern" in schema and re.search(str(schema["pattern"]), value) is None:
            return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return False
        if "maximum" in schema and value > schema["maximum"]:
            return False
    return True


def _has_json_type(value: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return any(_has_json_type(value, item) for item in expected)
    checks = {
        "object": lambda: isinstance(value, dict),
        "array": lambda: isinstance(value, list),
        "string": lambda: isinstance(value, str),
        "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": lambda: isinstance(value, bool),
        "null": lambda: value is None,
    }
    check = checks.get(str(expected))
    return check is not None and check()


__all__ = ["OutputValidation", "reviewer_accepts", "validate_output"]
