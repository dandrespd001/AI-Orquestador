"""Keyless loopback fixture for the Playwright credential-field smoke test."""

from __future__ import annotations

import os
import tempfile
from http.server import ThreadingHTTPServer
from pathlib import Path

from ai_orchestrator.api import HumanActionGate, PanelApplication, make_handler


class KeylessControlPlane:
    def __init__(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(
            prefix="ai-orchestrator-panel-e2e-"
        )
        self.state_root = Path(self._temporary.name)

    def query(self, section: str, **_params: object) -> dict[str, object]:
        return {
            "schema_version": 1,
            "section": section,
            "secret_values_included": False,
        }

    def store_credential(
        self,
        provider: str,
        name: str,
        value: str,
        **_params: object,
    ) -> dict[str, object]:
        if not provider or not name or not value:
            raise ValueError("fixture received an incomplete credential request")
        return {
            "schema_version": 1,
            "provider": provider,
            "credential": name,
            "status": "stored",
            "secret_returned": False,
        }


def main() -> None:
    port = int(os.environ.get("PANEL_E2E_PORT", "8766"))
    application = PanelApplication(
        KeylessControlPlane(),  # type: ignore[arg-type]
        HumanActionGate("browser-test-approval"),
    )
    ThreadingHTTPServer(("127.0.0.1", port), make_handler(application)).serve_forever()


if __name__ == "__main__":
    main()
