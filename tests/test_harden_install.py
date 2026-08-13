from __future__ import annotations

import errno
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.harden_install import harden_modes


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


if __name__ == "__main__":
    unittest.main()
