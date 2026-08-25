"""Descargar por el camino auditado, o no descargar.

El orquestador no dejaba traer nada de fuera, lo que es correcto para un modelo
pero deja al Director sin via para un dataset o una referencia. Esta si es una
via, y su valor esta en lo que rechaza: esquemas que no son HTTP, destinos
dentro de un repositorio y respuestas que no paran de crecer.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import tempfile
import threading
import unittest
from pathlib import Path

from ai_orchestrator.fetch import fetch_resource, list_staged
from ai_orchestrator.models import ContractError

PAYLOAD = b"contenido de prueba\n" * 16


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = PAYLOAD * (4096 if self.path == "/huge" else 1)
        self.send_response(404 if self.path == "/missing" else 200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return None


class FetchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.thread.join(timeout=5)
        cls.server.server_close()

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "staging"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _url(self, path: str = "/file.txt") -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def test_the_receipt_matches_the_bytes_on_disk(self) -> None:
        receipt = fetch_resource(self._url(), staging_root=self.root)
        landed = Path(receipt["path"])
        self.assertTrue(landed.is_file())
        self.assertEqual(PAYLOAD, landed.read_bytes())
        self.assertEqual(hashlib.sha256(PAYLOAD).hexdigest(), receipt["sha256"])
        self.assertEqual(len(PAYLOAD), receipt["bytes"])
        self.assertFalse(receipt["promoted_to_repository"])

    def test_the_download_is_private_on_disk(self) -> None:
        receipt = fetch_resource(self._url(), staging_root=self.root)
        landed = Path(receipt["path"])
        self.assertEqual(0o600, landed.stat().st_mode & 0o777)
        self.assertEqual(0o600, (landed.parent / "receipt.json").stat().st_mode & 0o777)

    def test_two_fetches_of_the_same_url_do_not_overwrite(self) -> None:
        first = fetch_resource(self._url(), staging_root=self.root)
        second = fetch_resource(self._url(), staging_root=self.root)
        self.assertNotEqual(first["path"], second["path"])
        self.assertEqual(2, len(list_staged(self.root)))

    def test_a_declared_digest_that_does_not_match_is_refused(self) -> None:
        with self.assertRaises(ContractError) as caught:
            fetch_resource(
                self._url(), staging_root=self.root, expected_sha256="00" * 32
            )
        self.assertIn("sha256 mismatch", str(caught.exception))

    def test_a_matching_digest_passes(self) -> None:
        receipt = fetch_resource(
            self._url(),
            staging_root=self.root,
            expected_sha256=hashlib.sha256(PAYLOAD).hexdigest().upper(),
        )
        self.assertEqual(hashlib.sha256(PAYLOAD).hexdigest(), receipt["sha256"])

    def test_an_oversized_body_stops_instead_of_filling_the_disk(self) -> None:
        with self.assertRaises(ContractError) as caught:
            fetch_resource(self._url("/huge"), staging_root=self.root, max_bytes=4096)
        self.assertIn("limit", str(caught.exception))

    def test_an_http_error_is_reported_as_a_contract_error(self) -> None:
        with self.assertRaises(ContractError) as caught:
            fetch_resource(self._url("/missing"), staging_root=self.root)
        self.assertIn("404", str(caught.exception))


class RefusalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_non_http_schemes_are_refused(self) -> None:
        for url in ("file:///etc/passwd", "ftp://example.invalid/x", "data:,x"):
            with self.subTest(url=url), self.assertRaises(ContractError):
                fetch_resource(url, staging_root=Path(self.temp.name) / "s")

    def test_a_url_without_a_host_is_refused(self) -> None:
        with self.assertRaises(ContractError):
            fetch_resource("https://", staging_root=Path(self.temp.name) / "s")

    def test_staging_inside_a_repository_is_refused(self) -> None:
        # Un recurso sin revisar no entra a un arbol de trabajo por la puerta
        # de atras: promoverlo es una decision aparte.
        repo = Path(self.temp.name) / "repo"
        (repo / ".git").mkdir(parents=True)
        with self.assertRaises(ContractError) as caught:
            fetch_resource("https://example.invalid/x", staging_root=repo / "staging")
        self.assertIn("git repository", str(caught.exception))


class ListingTests(unittest.TestCase):
    def test_an_absent_root_lists_nothing_instead_of_raising(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual([], list_staged(Path(directory) / "absent"))

    def test_an_unreadable_receipt_is_reported_not_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "20260101T000000Z-deadbeef"
            entry.mkdir()
            (entry / "receipt.json").write_text("{not json", encoding="utf-8")
            rows = list_staged(root)
        self.assertEqual(1, len(rows))
        self.assertIn("error", rows[0])

    def test_a_good_receipt_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "20260101T000000Z-deadbeef"
            entry.mkdir()
            (entry / "receipt.json").write_text(
                json.dumps({"sha256": "abc", "bytes": 1}), encoding="utf-8"
            )
            rows = list_staged(root)
        self.assertEqual("abc", rows[0]["sha256"])


if __name__ == "__main__":
    unittest.main()
