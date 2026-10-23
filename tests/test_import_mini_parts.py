"""Synthetic original-byte transport checks; no real nuScenes upload represented."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import stat
import sys
import uuid
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from import_mini_parts import SCHEMA, import_parts, main  # noqa: E402
from prepare_driving import CAMERAS, TABLES, prepare  # noqa: E402
from test_prepare_driving import synthetic_nuscenes  # noqa: E402


def sha(body):
    return hashlib.sha256(body).hexdigest()


@pytest.fixture
def handoff(tmp_path):
    source = synthetic_nuscenes(tmp_path / "SYNTHETIC-original", frames=2)
    prepared = tmp_path / "SYNTHETIC-prepared"
    prepare(source, "v1.0-mini", prepared, frames_per_scene=2)
    rows = json.loads((prepared / "selected_samples.json").read_text())
    records, contents = [], {}
    originals = [f"v1.0-mini/{name}.json" for name in TABLES]
    originals += [f"samples/{row['images'][camera]}" for row in rows for camera in CAMERAS]
    for output_path in originals:
        original = (source / output_path).read_bytes()
        metadata = output_path.startswith("v1.0-mini/")
        path = output_path + ".gz" if metadata else output_path
        body = gzip.compress(original, mtime=0) if metadata else original
        records.append(dict(path=path, sha256=sha(body), bytes=len(body), output_path=output_path,
                            output_sha256=sha(original), output_bytes=len(original)))
        contents[path] = body
    manifest = dict(schema=SCHEMA, transfer_id=str(uuid.uuid4()), part_count=3,
                    frames_per_scene=2, version="v1.0-mini", camera_order=list(CAMERAS),
                    source_archive=dict(name="v1.0-mini.tgz", bytes=3_900_000_000,
                                        last_modified=0, sha256=None),
                    selected_tokens=[{key: row[key] for key in
                                      ("sample_token", "scene_token", "timestamp")}
                                     for row in rows],
                    files=records)
    groups = [list(contents)[index::3] for index in range(3)]
    parts = []
    for index, members in enumerate(groups, 1):
        path = tmp_path / f"synthetic-part-{index}.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
            part_manifest = dict(manifest, part_index=index, members=members)
            archive.writestr("HANDOFF.json", json.dumps(part_manifest))
            for name in members:
                archive.writestr(name, contents[name])
        parts.append(path)
    return dict(source=source, parts=parts, manifest=manifest, tmp=tmp_path)


def rewrite(path, *, manifest_change=None, body_change=None, symlink=False):
    with zipfile.ZipFile(path) as original:
        data = {name: original.read(name) for name in original.namelist()}
    manifest = json.loads(data["HANDOFF.json"])
    if manifest_change:
        manifest_change(manifest)
    data["HANDOFF.json"] = json.dumps(manifest).encode()
    if body_change:
        name = manifest["members"][0]
        data[name] = body_change(data[name])
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        for name, body in data.items():
            if symlink and name != "HANDOFF.json":
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, body)
            else:
                archive.writestr(name, body)


def assert_unpublished(root):
    assert not root.exists()
    assert not list(root.parent.glob(".mini-upload-*"))
    assert not list(root.parent.glob(".mini-verify-*"))


def test_import_original_bytes_and_prepare_exact_frozen_selection(handoff):
    root = handoff["tmp"] / "imported"
    report = import_parts(list(reversed(handoff["parts"])), root)
    assert report["status"] == "completed" and report["selected_samples"] == 20
    assert report["camera_files"] == 120 and report["scenes"] == 10
    assert report["frames_per_scene"] == 2 and not report["labels_created"]
    assert not report["official_archive_checksum_verified"]
    for item in handoff["manifest"]["files"]:
        assert (root / item["output_path"]).read_bytes() == (
            handoff["source"] / item["output_path"]).read_bytes()
    prepared = handoff["tmp"] / "reprepared"
    result = prepare(root, "v1.0-mini", prepared, frames_per_scene=2)
    assert result["included_frames"] == 20 and result["excluded_frames"] == 0
    assert not result["labels_created"]
    original_scenes = json.loads(
        (handoff["tmp"] / "SYNTHETIC-prepared/split_scenes.json").read_text())
    assert json.loads((prepared / "split_scenes.json").read_text()) == original_scenes


def test_missing_and_duplicate_parts_never_publish(handoff):
    root = handoff["tmp"] / "missing"
    with pytest.raises(ValueError, match="incomplete"):
        import_parts(handoff["parts"][:-1], root)
    assert_unpublished(root)
    with pytest.raises(ValueError, match="Duplicate handoff part index"):
        import_parts([*handoff["parts"], handoff["parts"][0]], root)
    assert_unpublished(root)


def test_transport_tampering_preserves_clean_destination(handoff):
    root = handoff["tmp"] / "bad-hash"
    rewrite(handoff["parts"][0], body_change=lambda body: bytes([body[0] ^ 1]) + body[1:])
    with pytest.raises(ValueError, match="Transport checksum mismatch"):
        import_parts(handoff["parts"], root)
    assert_unpublished(root)


@pytest.mark.parametrize("failure", ["traversal", "version", "archive-checksum", "mixed",
                                     "wrong-selection", "image-transform"])
def test_untrusted_manifest_changes_are_rejected(handoff, failure):
    root = handoff["tmp"] / failure

    def change(record):
        if failure == "traversal":
            record["files"][0]["output_path"] = "../outside.json"
        elif failure == "version":
            record["version"] = "v1.0-trainval"
        elif failure == "archive-checksum":
            record["source_archive"]["sha256"] = "a" * 64
        elif failure == "mixed":
            if record["part_index"] == 2:
                record["transfer_id"] = str(uuid.uuid4())
        elif failure == "wrong-selection":
            record["selected_tokens"][0]["timestamp"] += 1
        else:
            image = next(item for item in record["files"] if item["path"].startswith("samples/"))
            image["output_sha256"] = "b" * 64

    for part in handoff["parts"]:
        rewrite(part, manifest_change=change)
    with pytest.raises(ValueError):
        import_parts(handoff["parts"], root)
    assert_unpublished(root)
    assert not (root.parent / "outside.json").exists()


def test_zip_symlinks_and_undeclared_entries_are_rejected(handoff):
    root = handoff["tmp"] / "unsafe"
    rewrite(handoff["parts"][0], symlink=True)
    with pytest.raises(ValueError, match="symlink ZIP entry"):
        import_parts(handoff["parts"], root)
    assert_unpublished(root)


def test_existing_data_and_cli_failure_report_are_preserved(handoff):
    root = handoff["tmp"] / "existing"
    root.mkdir()
    (root / "keep.txt").write_text("existing data")
    report = handoff["tmp"] / "report.json"
    code = main(["--parts", *map(str, handoff["parts"]), "--data-root", str(root),
                 "--report", str(report)])
    assert code == 2 and (root / "keep.txt").read_text() == "existing data"
    recorded = json.loads(report.read_text())
    assert recorded["status"] == "failed" and not recorded["data_extracted"]
    assert not recorded["labels_created"]
