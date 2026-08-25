"""Registering the MCP server, and recognising it once registered.

Each CLI keeps its server registry in a different shape, and the orchestrator
had been registered by hand, so the entry drifted: Claude, Codex and Abacus had
it while Qwen and OpenCode did not. Worse, ``doctor`` compared the stored
command against the bare name ``ai-orchestrator`` while every real registration
used the absolute path of a symlink, so a healthy install reported mcp: false.
"""

from __future__ import annotations

import json
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_orchestrator.mcp_install import (
    SERVER_NAME,
    UNSUPPORTED,
    install,
    repo_mcp_status,
    resolves_to_orchestrator,
)


class CommandRecognitionTests(unittest.TestCase):
    def test_the_absolute_path_of_the_symlink_is_recognised(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            real = Path(directory) / "real-ai-orchestrator"
            real.write_text("#!/bin/sh\n", encoding="utf-8")
            real.chmod(0o700)
            link = Path(directory) / "ai-orchestrator"
            link.symlink_to(real)
            with patch("ai_orchestrator.mcp_install.EXECUTABLE", link):
                self.assertTrue(resolves_to_orchestrator(str(link)))
                self.assertTrue(resolves_to_orchestrator(str(real)))

    def test_a_different_executable_is_not_recognised(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mine = Path(directory) / "ai-orchestrator"
            mine.write_text("#!/bin/sh\n", encoding="utf-8")
            other = Path(directory) / "something-else"
            other.write_text("#!/bin/sh\n", encoding="utf-8")
            with patch("ai_orchestrator.mcp_install.EXECUTABLE", mine):
                self.assertFalse(resolves_to_orchestrator(str(other)))

    def test_junk_values_never_pass(self) -> None:
        for value in (None, "", 42, [], {}):
            with self.subTest(value=value):
                self.assertFalse(resolves_to_orchestrator(value))


class RepositoryStatusTests(unittest.TestCase):
    def _repo(self, directory: str, command: str, args: list[str]) -> Path:
        repo = Path(directory)
        (repo / ".mcp.json").write_text(
            json.dumps(
                {"mcpServers": {SERVER_NAME: {"command": command, "args": args}}}
            ),
            encoding="utf-8",
        )
        return repo

    def test_an_absolute_command_is_accepted_for_the_bound_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "ai-orchestrator"
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            repo = self._repo(
                directory, str(executable), ["mcp-serve", "--profile", "chunsa"]
            )
            with patch("ai_orchestrator.mcp_install.EXECUTABLE", executable):
                self.assertTrue(repo_mcp_status(repo, "chunsa")["configured"])

    def test_a_mismatched_profile_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "ai-orchestrator"
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            repo = self._repo(
                directory, str(executable), ["mcp-serve", "--profile", "other"]
            )
            with patch("ai_orchestrator.mcp_install.EXECUTABLE", executable):
                self.assertFalse(repo_mcp_status(repo, "chunsa")["configured"])

    def test_a_missing_file_reports_absent_rather_than_raising(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            status = repo_mcp_status(Path(directory), "chunsa")
        self.assertFalse(status["configured"])
        self.assertEqual("absent", status["reason"])


class InstallTests(unittest.TestCase):
    def test_each_client_gets_its_own_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            (home / ".codex").mkdir()
            (home / ".config" / "opencode").mkdir(parents=True)
            (home / ".qwen").mkdir()
            with patch("ai_orchestrator.mcp_install.Path.home", return_value=home):
                install()

            claude = json.loads((home / ".claude.json").read_text(encoding="utf-8"))
            self.assertIn(SERVER_NAME, claude["mcpServers"])
            self.assertEqual(["mcp-serve"], claude["mcpServers"][SERVER_NAME]["args"])

            # OpenCode nests argv in a list and gates the server with `enabled`.
            opencode = json.loads(
                (home / ".config" / "opencode" / "opencode.json").read_text(
                    encoding="utf-8"
                )
            )
            entry = opencode["mcp"][SERVER_NAME]
            self.assertTrue(entry["enabled"])
            self.assertEqual("mcp-serve", entry["command"][-1])

            codex = tomllib.loads(
                (home / ".codex" / "config.toml").read_text(encoding="utf-8")
            )
            self.assertEqual(["mcp-serve"], codex["mcp_servers"][SERVER_NAME]["args"])

    def test_running_twice_does_not_duplicate_anything(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            (home / ".codex").mkdir()
            with patch("ai_orchestrator.mcp_install.Path.home", return_value=home):
                install()
                second = install()
            codex_text = (home / ".codex" / "config.toml").read_text(encoding="utf-8")
        self.assertEqual(1, codex_text.count(f"[mcp_servers.{SERVER_NAME}]"))
        statuses = {row["client"]: row["status"] for row in second["clients"]}
        self.assertEqual("ALREADY_REGISTERED", statuses["codex"])
        self.assertEqual("ALREADY_REGISTERED", statuses["claude"])

    def test_an_existing_config_keeps_its_other_keys(self) -> None:
        # ~/.claude.json holds the whole Claude Code state; losing it would be
        # far worse than not registering the server.
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            (home / ".claude.json").write_text(
                json.dumps({"projects": {"a": 1}, "numStartups": 72}), encoding="utf-8"
            )
            with patch("ai_orchestrator.mcp_install.Path.home", return_value=home):
                install("claude")
            payload = json.loads((home / ".claude.json").read_text(encoding="utf-8"))
        self.assertEqual({"a": 1}, payload["projects"])
        self.assertEqual(72, payload["numStartups"])
        self.assertIn(SERVER_NAME, payload["mcpServers"])

    def test_kimi_is_reported_as_unsupported_rather_than_failed(self) -> None:
        report = install("kimi")
        row = report["clients"][0]
        self.assertEqual("UNSUPPORTED", row["status"])
        self.assertEqual(UNSUPPORTED["kimi"], row["detail"])


if __name__ == "__main__":
    unittest.main()
