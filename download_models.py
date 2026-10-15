"""Получение готовых моделей с проверкой контрольной суммы архива."""

import argparse
import hashlib
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

from config import ROOT, load_json

ALLOWED = {"fine_tuned_model", "baseline_model.pkl", "vectorizer.pkl"}


def unpack_verified(archive_path, expected_sha256, destination):
    with Path(archive_path).open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != expected_sha256:
        raise ValueError("Контрольная сумма моделей не совпала")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or path.parts[0] not in ALLOWED
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError("Архив содержит недопустимые пути")
        archive.extractall(destination, members=members, filter="data")


def main():
    parser = argparse.ArgumentParser(description="Загрузка обученных моделей")
    parser.add_argument("--destination", type=Path, default=ROOT)
    args = parser.parse_args()
    if any((args.destination / name).exists() for name in ALLOWED):
        parser.error("В папке уже есть модели. Укажите другую --destination")
    manifest = load_json(ROOT / "models.json")
    with tempfile.TemporaryDirectory() as name:
        folder = Path(name)
        archive_path = folder / "models.tar.gz"
        request = urllib.request.Request(
            manifest["url"], headers={"User-Agent": "transformers-sentiment"}
        )
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            archive_path.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
        unpack_verified(archive_path, manifest["sha256"], folder / "unpacked")
        args.destination.mkdir(parents=True, exist_ok=True)
        for item in (folder / "unpacked").iterdir():
            shutil.move(str(item), str(args.destination / item.name))
    print(f"Модели загружены в {args.destination}")


if __name__ == "__main__":
    main()
