#!/usr/bin/env python3
"""Import small offline browser handoff ZIPs without downloading or creating labels.

The original five metadata tables and selected six-view image bytes are checked
before an atomic publish. Per-file checks establish consistency with the upload
manifest, not an official archive checksum or data/label authenticity.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import uuid
import zipfile

from prepare_driving import CAMERAS, TABLES, prepare

SCHEMA = "nuscenes-mini-browser-handoff-v1"
PART_LIMIT = 24 * 2**20
MANIFEST_LIMIT = 2 * 2**20
METADATA_LIMIT = 64 * 2**20
IMAGE_LIMIT = 12 * 2**20
TOTAL_LIMIT = 2 * 2**30
HASH = re.compile(r"[0-9a-f]{64}")
METADATA_PATHS = {f"v1.0-mini/{name}.json.gz" for name in TABLES}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def json_object(value):
    record = json.loads(value, object_pairs_hook=unique_object)
    require(isinstance(record, dict), "Handoff manifest must be a JSON object")
    return record


def safe_path(value):
    require(isinstance(value, str) and value and "\\" not in value and "\0" not in value,
            "Unsafe handoff path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts
            and path.as_posix() == value, f"Unsafe handoff path: {value}")
    return path


def validate_manifest(record):
    require(record.get("schema") == SCHEMA and record.get("version") == "v1.0-mini",
            "Unsupported handoff schema or dataset version")
    transfer_id = record.get("transfer_id")
    require(isinstance(transfer_id, str), "Invalid handoff transfer_id")
    try:
        require(str(uuid.UUID(transfer_id)) == transfer_id.lower(), "Invalid transfer_id")
    except (ValueError, AttributeError) as exc:
        raise ValueError("Invalid handoff transfer_id") from exc
    for key in ("part_index", "part_count"):
        require(type(record.get(key)) is int and 1 <= record[key] <= 128,
                f"Invalid handoff {key}")
    require(record["part_index"] <= record["part_count"], "Part index exceeds part count")
    require(record.get("frames_per_scene") in (2, 8)
            and type(record.get("frames_per_scene")) is int,
            "Unsupported frames_per_scene; expected 2 or 8")
    require(record.get("camera_order") == list(CAMERAS), "Handoff camera order differs")
    source = record.get("source_archive")
    require(isinstance(source, dict) and isinstance(source.get("name"), str)
            and source["name"] and len(source["name"]) <= 255
            and Path(source["name"]).name == source["name"]
            and "\\" not in source["name"]
            and type(source.get("bytes")) is int and source["bytes"] > 0
            and source.get("sha256") is None,
            "Invalid source archive record or unsupported original checksum claim")
    selected = record.get("selected_tokens")
    require(isinstance(selected, list) and 1 <= len(selected) <= 80,
            "Handoff must select between 1 and 80 samples")
    tokens = set()
    for row in selected:
        require(isinstance(row, dict)
                and set(row) == {"sample_token", "scene_token", "timestamp"}
                and all(isinstance(row[key], str) and row[key].strip()
                        for key in ("sample_token", "scene_token"))
                and type(row["timestamp"]) in (int, float) and math.isfinite(row["timestamp"]),
                "Invalid selected sample identity")
        require(row["sample_token"] not in tokens, "Duplicate selected sample token")
        tokens.add(row["sample_token"])
    files = record.get("files")
    require(isinstance(files, list) and len(files) == 5 + 6 * len(selected),
            "Handoff file inventory must contain five tables and six views per sample")
    indexed, destinations, total, metadata_paths = {}, set(), 0, set()
    for item in files:
        require(isinstance(item, dict), "Invalid handoff file record")
        path, output = safe_path(item.get("path")), safe_path(item.get("output_path"))
        require(item["path"] not in indexed and item["output_path"] not in destinations,
                "Duplicate handoff file path")
        metadata = item["path"] in METADATA_PATHS
        image = (len(path.parts) == 3 and path.parts[0] == "samples"
                 and path.parts[1] in CAMERAS)
        require(metadata or image, f"Unsupported handoff file: {path}")
        require(output.as_posix() == (item["path"][:-3] if metadata else item["path"]),
                "Handoff output path differs from source transport path")
        for key in ("sha256", "output_sha256"):
            require(isinstance(item.get(key), str) and HASH.fullmatch(item[key]),
                    f"Invalid {key} for {path}")
        for key, bound in (("bytes", PART_LIMIT),
                           ("output_bytes", METADATA_LIMIT if metadata else IMAGE_LIMIT)):
            require(type(item.get(key)) is int and 0 < item[key] <= bound,
                    f"Invalid or oversized {key} for {path}")
        if image:
            require(item["bytes"] == item["output_bytes"]
                    and item["sha256"] == item["output_sha256"],
                    "Image transport must preserve original image bytes")
        if metadata:
            metadata_paths.add(item["path"])
        total += item["output_bytes"]
        indexed[item["path"]] = item
        destinations.add(item["output_path"])
    require(metadata_paths == METADATA_PATHS and total <= TOTAL_LIMIT,
            "Missing metadata tables or excessive extracted data size")
    members = record.get("members")
    require(isinstance(members, list) and members
            and all(isinstance(name, str) and name in indexed for name in members)
            and len(set(members)) == len(members), "Invalid part member inventory")
    return indexed


def scan_parts(parts):
    require(parts, "Provide at least one handoff ZIP part")
    common, files, indices, seen, validated = None, None, set(), set(), []
    for path in map(Path, parts):
        require(path.is_file() and not path.is_symlink() and path.stat().st_size <= PART_LIMIT,
                f"Part must be a regular ZIP no larger than 24 MiB: {path}")
        with zipfile.ZipFile(path) as archive:
            inventory = archive.infolist()
            require(len({entry.filename for entry in inventory}) == len(inventory),
                    "Duplicate ZIP member name")
            require(sum(entry.file_size for entry in inventory) <= METADATA_LIMIT,
                    "ZIP part expanded inventory exceeds limit")
            for entry in inventory:
                safe_path(entry.filename)
                mode = entry.external_attr >> 16
                require(not entry.is_dir() and not stat.S_ISLNK(mode)
                        and not (mode and stat.S_IFMT(mode) not in (0, stat.S_IFREG))
                        and not entry.flag_bits & 1 and entry.compress_type == zipfile.ZIP_STORED,
                        "Unsupported, encrypted, compressed or symlink ZIP entry")
            require("HANDOFF.json" in archive.namelist(), "ZIP part has no HANDOFF.json")
            require(archive.getinfo("HANDOFF.json").file_size <= MANIFEST_LIMIT,
                    "Handoff manifest is too large")
            record = json_object(archive.read("HANDOFF.json"))
            current_files = validate_manifest(record)
            global_record = {key: value for key, value in record.items()
                             if key not in ("part_index", "members")}
            if common is None:
                common, files = global_record, current_files
            require(common == global_record, "ZIP parts belong to different handoffs")
            require(record["part_index"] not in indices, "Duplicate handoff part index")
            indices.add(record["part_index"])
            require(set(archive.namelist()) == {"HANDOFF.json", *record["members"]},
                    "ZIP inventory differs from declared part members")
            for name in record["members"]:
                require(name not in seen, "Handoff file occurs in more than one part")
                require(archive.getinfo(name).file_size == files[name]["bytes"],
                        f"Transport size differs for {name}")
                seen.add(name)
            validated.append((path, record))
    require(indices == set(range(1, common["part_count"] + 1)), "Handoff parts are incomplete")
    require(seen == set(files), "Handoff file inventory is incomplete")
    return common, files, sorted(validated, key=lambda item: item[1]["part_index"])


def import_parts(parts, data_root):
    """Read/check each part, then publish only after frozen preparation succeeds."""
    data_root = Path(data_root).absolute()
    require(not data_root.is_symlink(), "Data root may not be a symlink")
    require(not data_root.exists() or (data_root.is_dir() and not any(data_root.iterdir())),
            "Data root must be absent or empty; existing data is preserved")
    common, files, validated = scan_parts(parts)
    data_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".mini-upload-", dir=data_root.parent))
    try:
        for path, record in validated:
            with zipfile.ZipFile(path) as archive:
                for name in record["members"]:
                    item = files[name]
                    body = archive.read(name)
                    require(len(body) == item["bytes"] and digest(body) == item["sha256"],
                            f"Transport checksum mismatch for {name}")
                    if name in METADATA_PATHS:
                        with gzip.GzipFile(fileobj=io.BytesIO(body)) as content:
                            body = content.read(METADATA_LIMIT + 1)
                    require(len(body) == item["output_bytes"]
                            and digest(body) == item["output_sha256"],
                            f"Output checksum/size mismatch for {name}")
                    destination = stage / item["output_path"]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("xb") as handle:
                        handle.write(body)
        with tempfile.TemporaryDirectory(prefix=".mini-verify-", dir=data_root.parent) as temporary:
            prepared = Path(temporary) / "prepared"
            result = prepare(stage, "v1.0-mini", prepared,
                             frames_per_scene=common["frames_per_scene"])
            rows = json.loads((prepared / "selected_samples.json").read_text())
            selected = [{key: row[key] for key in ("sample_token", "scene_token", "timestamp")}
                        for row in rows]
            require(not result["excluded_frames"] and selected == common["selected_tokens"],
                    "Prepared selection differs from handoff or has incomplete six-view samples")
            expected = {f"v1.0-mini/{name}.json" for name in TABLES}
            for row in rows:
                expected.update(f"samples/{path}" for path in row["images"].values())
            require(len(expected) == 5 + 6 * len(rows)
                    and expected == {item["output_path"] for item in files.values()},
                    "Handoff contains unexpected or reused camera images")
        provenance = dict(common, provenance_limit=(
            "Per-file hashes and source metadata preserved; original archive checksum and "
            "official source authenticity are not independently verified. No labels created."))
        (stage / "browser_handoff.json").write_text(json.dumps(provenance, indent=2) + "\n")
        os.replace(stage, data_root)
        return {"status": "completed", "schema": SCHEMA, "data_root": str(data_root),
                "transfer_id": common["transfer_id"], "parts": len(validated),
                "frames_per_scene": common["frames_per_scene"], "selected_samples": len(rows),
                "scenes": len({row["scene_token"] for row in rows}), "camera_files": len(rows) * 6,
                "labels_created": False, "source_archive": common["source_archive"],
                "official_archive_checksum_verified": False,
                "original_metadata_bytes_preserved": True,
                "provenance_limit": provenance["provenance_limit"]}
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=Path, nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report = {"recorded_at": datetime.now(timezone.utc).isoformat(), "status": "started",
              "labels_created": False, "data_extracted": False}
    try:
        report.update(import_parts(args.parts, args.data_root), data_extracted=True)
        code = 0
    except (ValueError, OSError, EOFError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        code = 2
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
