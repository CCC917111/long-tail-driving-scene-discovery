"""Small review/provenance backups for moving between CPU and GPU notebooks.

Camera images, archives, model weights and feature caches are deliberately not
included. They must be retained separately or reacquired and checked.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

ROOT_FILES = {"reviewed.csv", "requested_protocol.json"}
OPTIONAL_FILES = {"model_revision.json", "gpu_package_versions.txt",
                  "cpu_package_versions.txt", "smoke_extraction_summary.json", "acquisition.json"}
DIRECTORY_FILES = {
    "prepared": {"raw_manifest.json", "selected_samples.json", "split_scenes.json",
                 "excluded_rows.json", "manual_review.csv", "review.html", "rubric.md",
                 "prepare_metadata.json"},
    "annotations": {"labels.json", "manifest.json", "excluded_rows.json", "import_metadata.json"},
}


def allowed(name):
    parts = PurePosixPath(name).parts
    return (bool(parts) and not name.startswith("/") and ".." not in parts
            and "\\" not in name and
            (name in ROOT_FILES | OPTIONAL_FILES
             or (len(parts) == 2 and parts[1] in DIRECTORY_FILES.get(parts[0], set()))))


def pack_review(work: Path, destination: Path, source_sha256: str) -> dict:
    work, destination = Path(work), Path(destination)
    files = {name: work / name for name in ROOT_FILES}
    files.update({name: work / name for name in OPTIONAL_FILES if (work / name).is_file()})
    for directory in ("prepared", "annotations"):
        root = work / directory
        if not root.is_dir():
            if directory == "annotations" and not root.exists():
                continue  # Partial manual reviews can be saved before label import.
            raise ValueError(f"Review backup requires {directory}")
        for filename in DIRECTORY_FILES[directory]:
            path = root / filename
            if path.is_symlink():
                raise ValueError("Review backup refuses symlinks")
            if path.is_file():
                files[path.relative_to(work).as_posix()] = path
    if any(not path.is_file() or path.is_symlink() for path in files.values()):
        raise ValueError("Review CSV/protocol files are missing or symlinks")
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    metadata = {"schema": "review-handoff-v1", "source_sha256": source_sha256,
                "files_sha256": hashes, "contains_images_or_weights": False}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for name, path in sorted(files.items()):
                bundle.write(path, name)
            bundle.writestr("handoff.json", json.dumps(metadata, indent=2))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return metadata


def restore_review(archive: Path, work: Path, source_sha256: str) -> dict:
    work = Path(work)
    with zipfile.ZipFile(archive) as bundle:
        inventory = bundle.infolist()
        if (sum(row.file_size for row in inventory) > 256 * 2**20
                or len({row.filename for row in inventory}) != len(inventory)):
            raise ValueError("Duplicate or oversized review backup")
        metadata = json.loads(bundle.read("handoff.json"))
        if (metadata.get("schema") != "review-handoff-v1"
                or metadata.get("source_sha256") != source_sha256):
            raise ValueError("Review backup belongs to a different embedded source snapshot")
        hashes = metadata.get("files_sha256", {})
        if (not isinstance(hashes, dict) or not ROOT_FILES <= hashes.keys()
                or set(bundle.namelist()) != set(hashes) | {"handoff.json"}):
            raise ValueError("Review backup inventory is incomplete or inconsistent")
        contents = {}
        for name, digest in hashes.items():
            if not allowed(name):
                raise ValueError("Unsafe or unexpected review backup path")
            data = bundle.read(name)
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError("Review backup hash mismatch")
            path = work / name
            if (path.is_symlink() or not path.resolve().is_relative_to(work.resolve())
                    or (path.exists() and path.read_bytes() != data)):
                raise ValueError("Existing review files differ; use a fresh WORK directory")
            contents[name] = data
    for name, data in contents.items():
        path = work / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            continue
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return metadata
