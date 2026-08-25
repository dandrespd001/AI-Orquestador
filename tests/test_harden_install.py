from __future__ import annotations

import errno
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.harden_install import harden_modes, install_chunsa_integration

ROOT = Path(__file__).resolve().parents[1]


class HardenInstallTests(unittest.TestCase):
    def test_read_only_managed_file_does_not_abort_install(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            managed = root / "managed-id"
            managed.write_text("opaque\n", encoding="utf-8")
            original_chmod = Path.chmod

            def guarded_chmod(path: Path, mode: int, *args, **kwargs) -> None:
                if path == managed:
                    raise OSError(errno.EROFS, "read-only file system", str(path))
                original_chmod(path, mode, *args, **kwargs)

            with patch.object(Path, "chmod", autospec=True, side_effect=guarded_chmod):
                result = harden_modes((root,))

        self.assertEqual(0, result["changed"])
        self.assertEqual(1, result["skipped_read_only"])

    def test_chunsa_integration_installs_only_gateway_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary) / "repo"
            repo.mkdir()
            unrelated = repo / "local-change.txt"
            unrelated.write_text("preserve me\n", encoding="utf-8")
            installed = install_chunsa_integration(repo)

            self.assertEqual("preserve me\n", unrelated.read_text())
            self.assertEqual(
                [
                    ".claude/settings.json",
                    ".mcp.json",
                    "tools/chunsa_ai.sh",
                ],
                installed,
            )
            mcp = json.loads((repo / ".mcp.json").read_text())
            self.assertEqual(
                ["mcp-serve", "--profile", "chunsa"],
                mcp["mcpServers"]["ai-orchestrator"]["args"],
            )
            settings = json.loads((repo / ".claude/settings.json").read_text())
            self.assertEqual("sonnet", settings["model"])
            self.assertEqual("high", settings["effortLevel"])
            self.assertEqual("high", settings["env"]["CLAUDE_CODE_EFFORT_LEVEL"])

    def test_chunsa_integration_updates_tracked_clean_gateway_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary) / "repo"
            (repo / ".claude").mkdir(parents=True)
            (repo / "tools").mkdir()
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "config",
                    "user.email",
                    "test@example.invalid",
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repo), "config", "user.name", "Test"],
                check=True,
            )
            (repo / ".claude/settings.json").write_text("{}\n", encoding="utf-8")
            (repo / "tools/chunsa_ai.sh").write_text("#!/bin/sh\n", encoding="utf-8")
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "add",
                    ".claude/settings.json",
                    "tools/chunsa_ai.sh",
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repo), "commit", "-m", "gateway baseline"],
                check=True,
                capture_output=True,
            )
            installed = install_chunsa_integration(repo)
        self.assertEqual(
            [".claude/settings.json", ".mcp.json", "tools/chunsa_ai.sh"],
            installed,
        )

    def test_chunsa_wrapper_maps_readiness_and_full_doctor_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "ai-orchestrator"
            executable.write_text(
                "#!/bin/sh\nprintf '%s\\n' \"$@\"\n", encoding="utf-8"
            )
            executable.chmod(0o700)
            environment = dict(os.environ)
            environment["PATH"] = f"{root}:{environment.get('PATH', '')}"
            wrapper = ROOT / "tools" / "chunsa_ai.sh"
            readiness = subprocess.run(
                [str(wrapper), "doctor"],
                env=environment,
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout.splitlines()
            full = subprocess.run(
                [str(wrapper), "doctor-full"],
                env=environment,
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout.splitlines()
        self.assertEqual(
            ["doctor", "--profile", "chunsa", "--check", "readiness"],
            readiness,
        )
        self.assertEqual(["doctor", "--profile", "chunsa", "--check", "full"], full)


if __name__ == "__main__":
    unittest.main()
