"""Git-worktree transaction and deterministic validation for model edits."""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ai_orchestrator.models import ContractError, TaskSpec, ensure_private_directory
from ai_orchestrator.supervisor import run_supervised


class WriterError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class GateOutcome:
    command: str
    returncode: int
    output: str
    timed_out: bool


@dataclass(frozen=True)
class PatchOutcome:
    patch: bytes
    paths: tuple[str, ...]
    sha256: str


def _git(repo: Path, args: list[str], *, input_data: bytes | None = None) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        input=input_data,
        stdin=None if input_data is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
        env={
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
        },
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", "replace").strip()[:500]
        raise WriterError("GIT_ERROR", message or f"git {' '.join(args)} failed")
    return completed.stdout


def _temporary_root() -> Path:
    root = Path(tempfile.gettempdir()) / f"ai-orchestrator-{os.getuid()}"
    ensure_private_directory(root)
    return root


class WorktreeTransaction:
    def __init__(self, task: TaskSpec) -> None:
        self.task = task
        self.source = Path(task.repo).resolve()
        self.base_commit = ""
        self.source_was_clean = False
        self.root: Path | None = None
        self.worktree: Path | None = None
        self.ignored_baseline: frozenset[str] = frozenset()

    def __enter__(self) -> "WorktreeTransaction":
        top = Path(_git(self.source, ["rev-parse", "--show-toplevel"]).decode().strip()).resolve()
        if top != self.source:
            raise WriterError("GIT_ROOT_REQUIRED", "task repo must be the Git worktree root")
        self.base_commit = _git(self.source, ["rev-parse", "HEAD"]).decode().strip()
        self.source_was_clean = not bool(
            _git(self.source, ["status", "--porcelain=v1", "--untracked-files=all"])
        )
        self.root = Path(tempfile.mkdtemp(prefix=f"{self.task.task_id}-", dir=_temporary_root()))
        os.chmod(self.root, 0o700)
        self.worktree = self.root / "worktree"
        _git(
            self.source,
            ["worktree", "add", "--detach", str(self.worktree), self.base_commit],
        )
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        if self.worktree is not None:
            subprocess.run(
                ["git", "-C", str(self.source), "worktree", "remove", "--force", str(self.worktree)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=False,
            )
        if self.root is not None:
            shutil.rmtree(self.root, ignore_errors=True)

    def _require_worktree(self) -> Path:
        if self.worktree is None:
            raise RuntimeError("worktree transaction has not started")
        return self.worktree

    def _ignored_paths(self) -> frozenset[str]:
        worktree = self._require_worktree()
        raw = _git(worktree, ["ls-files", "--others", "--ignored", "--exclude-standard", "-z"])
        return frozenset(value.decode("utf-8", "replace") for value in raw.split(b"\0") if value)

    def snapshot_ignored(self) -> None:
        """Censar los ficheros ignorados antes de que corra el escritor.

        Un gate compila y deja `build/` o `__pycache__`, y en la vuelta
        siguiente del bucle de reparacion siguen ahi. Sin esta foto no se puede
        distinguir lo que ensucio el gate de lo que escribio el modelo, y la
        unica regla posible seria prohibirlos todos, lo que rompe cualquier
        gate que compile.
        """

        self.ignored_baseline = self._ignored_paths()

    def collect_patch(self, protected_prefixes: tuple[str, ...]) -> PatchOutcome:
        worktree = self._require_worktree()
        # Un fichero ignorado nunca entra en el diff, asi que no puede llegar al
        # repo de origen — pero si vive en el arbol donde corren los gates, y un
        # gate ejecuta el codigo del modelo. Lo que se veta es que el escritor
        # los cree, no que existan.
        new_ignored = sorted(self._ignored_paths() - self.ignored_baseline)
        if new_ignored:
            raise WriterError(
                "WRITE_SCOPE_VIOLATION", f"writer created ignored path: {new_ignored[0]}"
            )
        tracked = _git(worktree, ["diff", "--name-only", "-z", "HEAD"]).split(b"\0")
        untracked = _git(worktree, ["ls-files", "--others", "--exclude-standard", "-z"]).split(b"\0")
        raw_paths = sorted({value.decode("utf-8", "strict") for value in tracked + untracked if value})
        if len(raw_paths) > self.task.max_changed_files:
            raise WriterError("WRITE_SCOPE_VIOLATION", "changed-file limit exceeded")
        allowed = tuple(PurePosixPath(value) for value in self.task.allowed_write_paths)
        protected = tuple(PurePosixPath(value.strip("/")) for value in protected_prefixes)
        for raw in raw_paths:
            path = PurePosixPath(raw)
            if path.is_absolute() or ".." in path.parts or path.parts[0] == ".git":
                raise WriterError("WRITE_SCOPE_VIOLATION", f"unsafe changed path: {raw}")
            if not any(path == prefix or prefix in path.parents for prefix in allowed):
                raise WriterError("WRITE_SCOPE_VIOLATION", f"path outside write allowlist: {raw}")
            if any(path == prefix or prefix in path.parents for prefix in protected):
                raise WriterError("WRITE_SCOPE_VIOLATION", f"protected path changed: {raw}")
            candidate = worktree / raw
            current = candidate
            while current != worktree:
                if current.is_symlink():
                    raise WriterError("WRITE_SCOPE_VIOLATION", f"symlink path changed: {raw}")
                current = current.parent

        new_paths = [value.decode("utf-8", "strict") for value in untracked if value]
        if new_paths:
            _git(worktree, ["add", "--intent-to-add", "--", *new_paths])
        patch = _git(worktree, ["diff", "--binary", "--no-ext-diff", "HEAD", "--"])
        if len(patch) > self.task.max_patch_bytes:
            raise WriterError("WRITE_SCOPE_VIOLATION", "patch-size limit exceeded")
        numstat = _git(worktree, ["diff", "--numstat", "HEAD", "--"]).decode("utf-8", "replace")
        if any(line.startswith("-\t-\t") for line in numstat.splitlines()):
            raise WriterError("WRITE_SCOPE_VIOLATION", "binary changes are not allowed")
        return PatchOutcome(
            patch=patch,
            paths=tuple(raw_paths),
            sha256=hashlib.sha256(patch).hexdigest(),
        )

    def apply_candidate_patch(self, patch: bytes) -> None:
        """Apply a model-produced patch only inside the disposable worktree.

        Scope and protected-path validation still happens through
        ``collect_patch`` before gates or source application. A malformed patch
        therefore cannot reach the caller's worktree.
        """

        worktree = self._require_worktree()
        if not patch.strip() or len(patch) > self.task.max_patch_bytes:
            raise WriterError("MALFORMED_PATCH", "candidate patch is empty or too large")
        _git(worktree, ["apply", "--check", "--binary", "-"], input_data=patch)
        _git(worktree, ["apply", "--binary", "-"], input_data=patch)

    def can_apply(self) -> bool:
        if not self.source_was_clean:
            return False
        head = _git(self.source, ["rev-parse", "HEAD"]).decode().strip()
        clean = not bool(_git(self.source, ["status", "--porcelain=v1", "--untracked-files=all"]))
        return head == self.base_commit and clean

    def apply(self, patch: bytes) -> None:
        if not self.can_apply():
            raise WriterError("DIRTY_WORKTREE", "source worktree changed during the model run")
        _git(self.source, ["apply", "--check", "--binary", "-"], input_data=patch)
        _git(self.source, ["apply", "--binary", "-"], input_data=patch)


_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|authorization)\b\s*[:=]\s*\S+"
)
_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=(.*)$", re.DOTALL)


def _sanitized(data: bytes) -> str:
    text = data.decode("utf-8", "replace")[-12_000:]
    return _SECRET_ASSIGNMENT.sub(lambda match: match.group(1) + "=[REDACTED]", text)


def _gate_argv(command: str) -> tuple[dict[str, str], list[str]]:
    # `"$(`"` was a typo for `"$("` — it spelled the three characters `$`, `(`
    # and a backtick, so command substitution was never actually detected. The
    # backtick entry already covers one half of it; this covers the other.
    #
    # `>` and `<` stay out of the list on purpose: gates legitimately contain
    # them (`python3 -c "assert a < b"`), and no shell is ever involved here —
    # run_supervised receives argv, so this guard is defence in depth against a
    # mis-specified gate, not the thing that stops injection.
    if any(value in command for value in ("\n", "\r", "|", ";", "&", "`", "$(")):
        raise ContractError(f"gate command contains a shell operator: {command!r}")
    tokens = shlex.split(command, posix=True)
    environment: dict[str, str] = {}
    while tokens and _ENV_ASSIGNMENT.match(tokens[0]):
        key, value = tokens.pop(0).split("=", 1)
        environment[key] = value
    if not tokens:
        raise ContractError("gate command has no executable")
    return environment, tokens


GATE_HOME = "/tmp/gate-home"


def _sandboxed_gate(argv: list[str], worktree: Path) -> list[str]:
    """Run a gate with the model's code but WITHOUT the operator's machine.

    A gate is `cmake`, `ctest` or `pytest` over a worktree the model just
    edited, so running one executes the model's code. Until this existed it ran
    with the real HOME and full networking, and its output was handed back to
    the model through `repair_prompt` — a read of `~/.omniroute/.env` plus a
    return channel, in one step. The writer was already sandboxed; its
    validator was not.

    The worktree is bound at ITS OWN PATH rather than a fixed mount point: the
    suite compiles absolute paths in (the golden CHDB, for one), so relocating
    it would break the gates rather than secure them. `--tmpfs /home` hides
    everything else under the home directory, and the bind re-exposes only the
    worktree because later binds win.

    Verified 2026-08-13: `.omniroute`, `.ssh` and `.config/ai-orchestrator` are
    absent inside, there is no outbound network (loopback only), the worktree is
    writable and the toolchain resolves.
    """
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise ContractError(
            "bwrap is required to run gates; refusing to execute model-written "
            "code with the operator's home directory and network"
        )
    sandboxed = [
        bwrap,
        "--unshare-all",          # no network: closes the exfiltration channel
        "--die-with-parent",
        "--new-session",
        "--ro-bind", "/", "/",
        "--tmpfs", "/home",
        "--tmpfs", "/tmp",
        "--proc", "/proc",
        "--dev", "/dev",
        "--bind", str(worktree), str(worktree),
        "--dir", GATE_HOME,
    ]
    # Python derives its user site-packages from HOME, so a disposable HOME
    # loses them and toolchain gates that import jsonschema/PyYAML break. Mount
    # the real ones READ-ONLY *inside* the disposable home, where that
    # derivation finds them. `~/.local/lib` holds packages; the credentials and
    # receipts live in `~/.local/share` and `~/.config`, which stay hidden.
    user_lib = Path.home() / ".local" / "lib"
    if user_lib.is_dir():
        sandboxed += ["--ro-bind", str(user_lib), f"{GATE_HOME}/.local/lib"]
    sandboxed += ["--chdir", str(worktree), "--", *argv]
    return sandboxed


def run_gates(worktree: Path, commands: tuple[str, ...], timeout_seconds: int) -> tuple[GateOutcome, ...]:
    if not 1 <= timeout_seconds <= 1800:
        raise ContractError("gate timeout_seconds must be between 1 and 1800")
    outcomes: list[GateOutcome] = []
    for command in commands:
        additions, argv = _gate_argv(command)
        environment = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": GATE_HOME,
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
            "CI": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
        environment.update(additions)
        result = run_supervised(
            _sandboxed_gate(argv, worktree),
            input_data=None,
            cwd=worktree,
            env=environment,
            timeout_seconds=timeout_seconds,
        )
        output = _sanitized(result.stdout + b"\n" + result.stderr)
        outcomes.append(GateOutcome(command, result.returncode, output, result.timed_out))
        if result.returncode != 0 or result.timed_out:
            break
    return tuple(outcomes)


def gates_passed(outcomes: tuple[GateOutcome, ...], expected_count: int) -> bool:
    return expected_count > 0 and len(outcomes) == expected_count and all(
        value.returncode == 0 and not value.timed_out for value in outcomes
    )


def repair_prompt(task_prompt: bytes, outcomes: tuple[GateOutcome, ...]) -> bytes:
    failure = outcomes[-1] if outcomes else None
    detail = failure.output if failure else "No deterministic gate was configured."
    command = failure.command if failure else "<none>"
    return (
        task_prompt
        + b"\n\nThe previous edit did not pass the orchestrator gate. "
        + f"Repair only the allowed files. Failed gate: {command}\n{detail}".encode("utf-8")
    )
