"""Review backup fixtures only; no dataset, real review or model weights."""

import hashlib
import json
from pathlib import Path
import sys
import shutil
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from portable_handoff import pack_review, restore_review  # noqa: E402


@pytest.fixture
def backup(tmp_path):
    work = tmp_path / "work"
    for name in ("prepared/rubric.md", "annotations/labels.json", "reviewed.csv",
                 "requested_protocol.json"):
        path = work / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic backup fixture: " + name)
    (work / "model-weights.bin").write_bytes(b"must not be included")
    archive = tmp_path / "review.zip"
    pack_review(work, archive, "a" * 64)
    return work, archive


def test_roundtrip_keeps_only_review_and_protocol_and_preserves_source(backup, tmp_path):
    work, archive = backup
    destination = tmp_path / "restored"
    metadata = restore_review(archive, destination, "a" * 64)
    for name, digest in metadata["files_sha256"].items():
        assert (destination / name).read_bytes() == (work / name).read_bytes()
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == digest
    assert not (destination / "model-weights.bin").exists()
    restore_review(archive, destination, "a" * 64)  # Idempotent resume.


def test_partial_review_roundtrip_before_labels_exist(backup, tmp_path):
    work, _ = backup
    shutil.rmtree(work / "annotations")
    archive = tmp_path / "partial-review.zip"
    packed = pack_review(work, archive, "a" * 64)
    assert not any(name.startswith("annotations/") for name in packed["files_sha256"])
    restored = tmp_path / "restored-partial"
    restore_review(archive, restored, "a" * 64)
    assert (restored / "reviewed.csv").read_bytes() == (work / "reviewed.csv").read_bytes()
    assert (restored / "prepared/rubric.md").read_bytes() == (
        work / "prepared/rubric.md").read_bytes()
    assert not (restored / "annotations").exists()


def test_wrong_snapshot_and_existing_reviews_are_preserved(backup, tmp_path):
    _, archive = backup
    destination = tmp_path / "restored"
    with pytest.raises(ValueError, match="different embedded source"):
        restore_review(archive, destination, "b" * 64)
    destination.mkdir()
    (destination / "reviewed.csv").write_text("keep this user's review")
    with pytest.raises(ValueError, match="Existing review files differ"):
        restore_review(archive, destination, "a" * 64)
    assert (destination / "reviewed.csv").read_text() == "keep this user's review"
    assert not (destination / "prepared").exists()


def test_backup_tampering_or_extra_paths_are_rejected(backup, tmp_path):
    _, archive = backup
    with zipfile.ZipFile(archive) as original:
        metadata = json.loads(original.read("handoff.json"))
        data = {name: original.read(name) for name in original.namelist()}
    data["reviewed.csv"] = b"altered decisions"
    altered = tmp_path / "altered.zip"
    with zipfile.ZipFile(altered, "w") as output:
        for name, value in data.items():
            output.writestr(name, value)
    with pytest.raises(ValueError, match="hash mismatch"):
        restore_review(altered, tmp_path / "no-output", "a" * 64)
    data["reviewed.csv"] = b"restored bytes"
    metadata["files_sha256"]["../outside"] = hashlib.sha256(b"escape").hexdigest()
    data["../outside"] = b"escape"
    data["handoff.json"] = json.dumps(metadata).encode()
    metadata["files_sha256"]["reviewed.csv"] = hashlib.sha256(data["reviewed.csv"]).hexdigest()
    data["handoff.json"] = json.dumps(metadata).encode()
    with zipfile.ZipFile(altered, "w") as output:
        for name, value in data.items():
            output.writestr(name, value)
    with pytest.raises(ValueError, match="Unsafe"):
        restore_review(altered, tmp_path / "no-output", "a" * 64)
    assert not (tmp_path / "outside").exists()
