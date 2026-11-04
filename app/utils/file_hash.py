import hashlib
from pathlib import Path


def calculate_file_hash(file_path: str | Path) -> str:
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        while data := file.read(1024 * 1024):
            sha256.update(data)

    return sha256.hexdigest()