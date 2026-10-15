import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from download_models import unpack_verified


class DownloadTests(unittest.TestCase):
    def archive(self, folder, member_name):
        path = folder / "models.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            item = tarfile.TarInfo(member_name)
            item.size = 4
            archive.addfile(item, io.BytesIO(b"test"))
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_bad_checksum_does_not_extract_any_files(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            path, _ = self.archive(folder, "fine_tuned_model/config.json")
            with self.assertRaises(ValueError):
                unpack_verified(path, "0" * 64, folder / "output")
            self.assertFalse((folder / "output").exists())

    def test_path_traversal_is_rejected_even_with_valid_checksum(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            path, digest = self.archive(folder, "fine_tuned_model/../../escape")
            with self.assertRaises(ValueError):
                unpack_verified(path, digest, folder / "output")
            self.assertFalse((folder / "output").exists())

    def test_valid_model_files_are_extracted(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            path, digest = self.archive(folder, "fine_tuned_model/config.json")
            unpack_verified(path, digest, folder / "output")
            self.assertEqual(
                (folder / "output/fine_tuned_model/config.json").read_bytes(), b"test"
            )
