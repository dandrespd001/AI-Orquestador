"""Reproducible installation and local validation of DeepSeek Harness."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ai_orchestrator.harness import HARNESS_PROFILES, harness_version_supported
from ai_orchestrator.models import ContractError, ensure_private_directory
from ai_orchestrator.updates import UpdateRegistry, load_component_pins


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str], *, cwd: Path, environment: dict[str, str]) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ContractError(f"Harness installation command failed: {command[0]}") from exc
    if completed.returncode != 0:
        detail = (
            completed.stderr.strip().splitlines()[-1:]
            or completed.stdout.strip().splitlines()[-1:]
            or ["no diagnostic"]
        )
        raise ContractError(f"Harness installation failed: {detail[0]}")
    return completed.stdout


def validate_prepared_harness(tree: Path) -> dict[str, Any]:
    """Run the installed parser against both profiles before staging."""

    executable = tree / "node_modules" / ".bin" / "dsh"
    if not executable.resolve().is_relative_to(tree.resolve()) or not executable.is_file():
        raise ContractError("prepared Harness executable is missing or escapes its tree")
    base_environment = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(tree),
        "DSH_HOME": str(tree),
        "AI_ORCHESTRATOR_HARNESS_CREDENTIAL": "validation-only",
        "AI_ORCHESTRATOR_DEEPSEEK_BRIDGE_URL": "http://127.0.0.1:20129",
        "AI_ORCHESTRATOR_WEB_SEARCH_CREDENTIAL": "validation-only",
        "AI_ORCHESTRATOR_WEB_SEARCH_URL": "http://127.0.0.1:20130/mcp",
        "AI_ORCHESTRATOR_PYTHON": "/usr/bin/python3",
        "AI_ORCHESTRATOR_ROOT": str(tree),
        "AI_ORCHESTRATOR_SOCKET": str(tree / "broker.sock"),
        "AI_ORCHESTRATOR_SKILLS_DIR": str(tree / "skills"),
        "NO_COLOR": "1",
    }
    version_output = _run(
        [str(executable), "--version"], cwd=tree, environment=base_environment
    )
    version = version_output.strip().split()[-1]
    if not harness_version_supported(version):
        raise ContractError("prepared Harness is below the secure version baseline")
    profiles: dict[str, dict[str, bool]] = {}
    for profile in sorted(HARNESS_PROFILES):
        output = _run(
            [str(executable), "--profile", profile, "--dump-config"],
            cwd=tree,
            environment={**base_environment, "DSH_MODEL": "deepseek-v4-flash"},
        )
        required = {
            "deepseek-v4-pro": "deepseek-v4-pro" in output,
            "deepseek-v4-flash": "deepseek-v4-flash" in output,
            "strict_mcp": "mcp-ai-orchestrator" in output and "chunsa" in output,
            "local_bridge": "127.0.0.1:20129" in output,
            "minimax": "minimax-orchestrator" in output and "MiniMax-M3" in output,
        }
        if profile == "orchestrator-web":
            required["web_research_proxy"] = (
                "mcp-web-search" in output
                and "127.0.0.1:20130/mcp" in output
            )
        else:
            required["search_disabled"] = "mcp-web-search" not in output
        if not all(required.values()) or "0.1.0-rc.7" in output:
            raise ContractError(f"Harness profile failed composition validation: {profile}")
        profiles[profile] = required
    return {"version": version, "profiles": profiles, "ok": True}


def prepare_and_stage_harness(
    *,
    source: Path,
    registry: UpdateRegistry,
    pnpm: str = "pnpm",
    target_version: str = "latest",
) -> dict[str, Any]:
    """Resolve an update channel, lock it exactly, validate it, and stage it."""

    source = source.expanduser().resolve()
    package = source / "package.json"
    lockfile = source / "pnpm-lock.yaml"
    for path in (package, lockfile, source / "bundle", source / "profiles"):
        if path.is_symlink() or not path.exists():
            raise ContractError("Harness integration source is incomplete or unsafe")
    repository_root = source.parents[1]
    pins = load_component_pins(repository_root / "config")["deepseek-harness"]
    environment = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", str(Path.cwd())),
        "NO_COLOR": "1",
    }
    ensure_private_directory(registry.root)
    pnpm_store = registry.root / "pnpm-store"
    ensure_private_directory(pnpm_store)
    with tempfile.TemporaryDirectory(prefix="harness-install-", dir=registry.root) as temp:
        tree = Path(temp)
        shutil.copy2(package, tree / package.name)
        shutil.copy2(lockfile, tree / lockfile.name)
        shutil.copytree(source / "bundle", tree / "bundle", symlinks=True)
        shutil.copytree(source / "profiles", tree / "profiles", symlinks=True)
        _run(
            [
                pnpm,
                "add",
                "--save-exact",
                f"@deepseek-ai/dsh@{target_version}",
                "--ignore-scripts",
                "--store-dir",
                str(pnpm_store),
            ],
            cwd=tree,
            environment=environment,
        )
        for profile in sorted(HARNESS_PROFILES):
            _run(
                [
                    pnpm,
                    "install",
                    "--prod",
                    "--offline",
                    "--ignore-scripts",
                    "--store-dir",
                    str(pnpm_store),
                ],
                cwd=tree / "profiles" / profile,
                environment=environment,
            )
        validation = validate_prepared_harness(tree)
        resolved_lockfile = tree / "pnpm-lock.yaml"
        staged = registry.stage_directory(
            pins,
            tree,
            expected_lock_sha256=_sha256(resolved_lockfile),
            lockfile_name="pnpm-lock.yaml",
            version=str(validation["version"]),
        )
    return {**staged, "validation": validation}
