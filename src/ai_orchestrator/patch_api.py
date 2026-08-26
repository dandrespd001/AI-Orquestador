"""One-shot, tool-free patch generation through authenticated model APIs."""

from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ai_orchestrator.credentials import (
    deepseek_runtime_key,
    minimax_runtime_key,
    omniroute_runtime_key,
)
from ai_orchestrator.models import RouteDecision, TaskSpec


class PatchApiError(RuntimeError):
    """A patch-API failure carrying a fixed, non-secret classification code.

    The message may quote provider text and therefore never crosses the
    subprocess boundary. ``code`` comes from the closed set used at the raise
    sites below, so propagating it leaks nothing while still telling a quota
    apart from a bad credential or a dead connection.

    Before this existed, every failure reached the caller as
    ``MODEL_PROCESS_FAILED``: on 2026-08-13 a real delegation failed after 93 s
    with no diagnosable reason, and three of the four rejected candidate models
    had never had one either.
    """

    def __init__(self, message: str, code: str = "PATCH_API_ERROR") -> None:
        super().__init__(message)
        self.code = code


def _finish_reason(response: dict[str, object]) -> str | None:
    """Best-effort read of choices[0].finish_reason. Never raises: it is only
    used to classify a failure that has already happened."""
    try:
        return str(response["choices"][0]["finish_reason"])  # type: ignore[index]
    except (KeyError, IndexError, TypeError, ValueError):
        details = response.get("incomplete_details")
        if response.get("status") == "incomplete" and isinstance(details, dict):
            reason = details.get("reason")
            return str(reason) if reason is not None else "length"
        return None


def _http_code(status: int) -> str:
    """Map an HTTP status to a fixed code. The status itself is not a secret,
    and it is exactly what separates a quota (429) from a bad key (401)."""
    if status in (401, 403):
        return "CREDENTIAL_REJECTED"
    if status == 429:
        return "QUOTA_UNAVAILABLE"
    if 500 <= status < 600:
        return "PROVIDER_SERVER_ERROR"
    return f"PROVIDER_HTTP_{status}"


@dataclass(frozen=True)
class PatchApiOutcome:
    patch: bytes
    reported_model: str


def _tracked_context_paths(task: TaskSpec, worktree: Path) -> tuple[str, ...]:
    selected: set[str] = set(task.input_paths)
    for allowed in task.allowed_write_paths:
        candidate = worktree / allowed
        if candidate.is_file():
            selected.add(allowed)
        elif candidate.is_dir():
            completed = subprocess.run(
                ["git", "-C", str(worktree), "ls-files", "-z", "--", allowed],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=20,
                check=False,
            )
            if completed.returncode != 0:
                raise PatchApiError(
                    "could not enumerate bounded worktree context", "CONTEXT_UNREADABLE"
                )
            selected.update(
                value.decode("utf-8", "strict")
                for value in completed.stdout.split(b"\0")
                if value
            )
    return tuple(sorted(selected))


def _bounded_context(task: TaskSpec, worktree: Path) -> str:
    # A conservative byte approximation keeps requests under the task's
    # declared context budget without requiring a provider-specific tokenizer.
    remaining = min(task.max_context_tokens * 3, 2_000_000)
    chunks: list[str] = []
    for relative in _tracked_context_paths(task, worktree):
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts:
            raise PatchApiError("unsafe context path", "CONTEXT_UNSAFE_PATH")
        path = worktree / relative
        # Walk the ANCESTORS, not just the leaf. A symlinked directory inside
        # the worktree (`docs/ext -> ~/.config`) would otherwise be followed and
        # its target shipped to the provider inside the prompt. The lexical
        # check above cannot see it. Same walk as writer.py:130-135.
        current = path
        escapes = False
        while current != worktree:
            if current.is_symlink():
                escapes = True
                break
            current = current.parent
        if escapes or not path.is_file():
            continue
        raw = path.read_bytes()
        if b"\0" in raw or len(raw) > remaining:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        chunks.append(
            f"--- BEGIN FILE: {relative} ---\n{text}\n--- END FILE: {relative} ---"
        )
        remaining -= len(raw)
    return "\n\n".join(chunks)


def _request_json(
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, object],
    timeout_seconds: int,
) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read(4_000_001)
    except urllib.error.HTTPError as exc:
        exc.close()
        raise PatchApiError(
            f"provider HTTP error {exc.code}", _http_code(exc.code)
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise PatchApiError(
            "provider connection failed", "PROVIDER_UNREACHABLE"
        ) from exc
    if len(raw) > 4_000_000:
        raise PatchApiError(
            "provider response exceeded size limit", "PROVIDER_RESPONSE_TOO_LARGE"
        )
    try:
        decoded = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PatchApiError(
            "provider returned malformed JSON", "PROVIDER_MALFORMED_JSON"
        ) from exc
    if not isinstance(decoded, dict):
        raise PatchApiError(
            "provider returned an invalid response envelope", "PROVIDER_BAD_ENVELOPE"
        )
    return decoded


def generate_patch(
    task: TaskSpec,
    decision: RouteDecision,
    *,
    worktree: Path,
    prompt: bytes,
) -> PatchApiOutcome:
    context = _bounded_context(task, worktree)
    instruction = prompt.decode("utf-8", "strict")
    user_message = instruction + (
        "\n\nBOUNDED REPOSITORY CONTEXT\n" + context if context else ""
    )
    model = str(decision.requested_route_model)
    if decision.transport == "qwen_deepseek_cli":
        key = deepseek_runtime_key()
        if not key:
            raise PatchApiError(
                "DeepSeek credential unavailable", "CREDENTIAL_UNAVAILABLE"
            )
        response = _request_json(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": "Bearer " + key},
            payload={
                "model": model,
                "messages": [{"role": "user", "content": user_message}],
                "temperature": 0,
                "max_tokens": task.max_output_tokens,
                "stream": False,
            },
            timeout_seconds=task.timeout_seconds,
        )
        try:
            content = response["choices"][0]["message"]["content"]  # type: ignore[index]
        except (KeyError, IndexError, TypeError) as exc:
            raise PatchApiError(
                "DeepSeek response contained no patch", "PROVIDER_NO_CHOICES"
            ) from exc
        reported = response.get("model")
    elif decision.transport == "qwen_omniroute_cli":
        key = omniroute_runtime_key()
        if not key:
            raise PatchApiError(
                "OmniRoute credential unavailable", "CREDENTIAL_UNAVAILABLE"
            )
        response = _request_json(
            "http://127.0.0.1:20128/v1/chat/completions",
            headers={"Authorization": "Bearer " + key},
            payload={
                "model": model,
                "messages": [{"role": "user", "content": user_message}],
                "temperature": 0,
                "max_tokens": task.max_output_tokens,
                "stream": False,
            },
            timeout_seconds=task.timeout_seconds,
        )
        try:
            content = response["choices"][0]["message"]["content"]  # type: ignore[index]
        except (KeyError, IndexError, TypeError) as exc:
            raise PatchApiError(
                "OmniRoute response contained no patch", "PROVIDER_NO_CHOICES"
            ) from exc
        reported = response.get("model")
    elif decision.transport == "qwen_minimax_cli":
        key = minimax_runtime_key()
        if not key:
            raise PatchApiError(
                "MiniMax credential unavailable", "CREDENTIAL_UNAVAILABLE"
            )
        response = _request_json(
            "https://api.minimax.io/v1/responses",
            headers={"Authorization": "Bearer " + key},
            payload={
                "model": model,
                "input": user_message,
                "max_output_tokens": task.max_output_tokens,
                "reasoning": {"effort": "none"},
                "store": False,
            },
            timeout_seconds=task.timeout_seconds,
        )
        content = response.get("output_text")
        if not isinstance(content, str):
            output = response.get("output")
            parts: list[str] = []
            if isinstance(output, list):
                for item in output:
                    if not isinstance(item, dict) or item.get("type") != "message":
                        continue
                    blocks = item.get("content")
                    if not isinstance(blocks, list):
                        continue
                    parts.extend(
                        str(block.get("text", ""))
                        for block in blocks
                        if isinstance(block, dict)
                        and block.get("type") == "output_text"
                    )
            content = "".join(parts)
        reported = response.get("model")
    else:
        raise PatchApiError(
            "transport does not support the guarded patch API", "TRANSPORT_UNSUPPORTED"
        )
    if not isinstance(content, str) or not content.strip():
        # A reasoning model can spend the ENTIRE completion budget thinking and
        # emit zero answer tokens. Measured on 2026-08-13 with a 79 KB request:
        # finish_reason='length', content=0 chars, reasoning_content=48838,
        # reasoning_tokens=16384 of a 16384 budget. Raising max_tokens to 32768
        # did not help and `reasoning_effort` was ignored by the gateway, so the
        # cap is upstream. Reported apart from a genuinely empty answer, because
        # the fix is different: shrink the request, not retry it.
        if _finish_reason(response) in {"length", "max_output_tokens"}:
            raise PatchApiError(
                "provider spent its whole completion budget on reasoning",
                "REASONING_BUDGET_EXHAUSTED",
            )
        raise PatchApiError(
            "provider returned no textual patch", "PROVIDER_EMPTY_PATCH"
        )
    if not isinstance(reported, str) or not reported:
        raise PatchApiError("provider did not report its model", "PROVENANCE_MISSING")
    return PatchApiOutcome(content.strip().encode("utf-8") + b"\n", reported)
