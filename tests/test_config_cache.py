from __future__ import annotations

import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_orchestrator import __version__
from ai_orchestrator.config import _read_toml

ROOT = Path(__file__).resolve().parents[1]


class ConfigSnapshotTests(unittest.TestCase):
    def test_package_version_has_one_release_value(self) -> None:
        with (ROOT / "pyproject.toml").open("rb") as source:
            project_version = tomllib.load(source)["project"]["version"]
        self.assertEqual(project_version, __version__)

    def test_toml_snapshot_is_reused_and_invalidated_by_file_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "config.toml"
            source.write_text("schema_version = 1\nvalue = 1\n", encoding="utf-8")
            with patch(
                "ai_orchestrator.config.tomllib.load", wraps=tomllib.load
            ) as load:
                first = _read_toml(source)
                second = _read_toml(source)
                source.write_text("schema_version = 1\nvalue = 200\n", encoding="utf-8")
                third = _read_toml(source)

        self.assertEqual(first, second)
        self.assertEqual(200, third["value"])
        self.assertEqual(2, load.call_count)


if __name__ == "__main__":
    unittest.main()
