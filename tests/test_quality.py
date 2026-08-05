from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_orchestrator.models import ContractError
from ai_orchestrator.quality import (
    PROMPTFOO_SECURITY_ENV,
    _network_denied_command,
    _validate_config,
    promptfoo_environment,
)


class QualityTests(unittest.TestCase):
    @patch("ai_orchestrator.quality.shutil.which", return_value="/usr/bin/unshare")
    def test_network_isolation_uses_an_empty_user_namespace(self, _which) -> None:
        self.assertEqual(
            _network_denied_command(["promptfoo", "--version"]),
            ["/usr/bin/unshare", "--user", "--map-root-user", "--net", "promptfoo", "--version"],
        )

    def test_promptfoo_environment_is_local_and_drops_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            environment = promptfoo_environment(
                Path(raw),
                {
                    "PATH": "/usr/bin",
                    "OPENAI_API_KEY": "forbidden",
                    "ANTHROPIC_AUTH_TOKEN": "forbidden",
                    "BROKER_SECRET": "forbidden",
                },
            )
        self.assertNotIn("OPENAI_API_KEY", environment)
        self.assertNotIn("ANTHROPIC_AUTH_TOKEN", environment)
        self.assertNotIn("BROKER_SECRET", environment)
        for name, value in PROMPTFOO_SECURITY_ENV.items():
            self.assertEqual(value, environment[name])

    def test_config_must_disable_sharing_and_remote_hosts(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            safe = root / "safe.yaml"
            safe.write_text("sharing: false\nproviders: []\n", encoding="utf-8")
            _validate_config(safe)

            unsafe = root / "unsafe.yaml"
            unsafe.write_text("sharing: true\n", encoding="utf-8")
            with self.assertRaises(ContractError):
                _validate_config(unsafe)


if __name__ == "__main__":
    unittest.main()
