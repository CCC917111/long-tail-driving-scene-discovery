#!/usr/bin/env python3
"""Import explicit AI visual annotations for a separate silver-label pilot.

This does not convert AI judgments into human reviews. The frozen preparation,
scene allocation and rubric are reused, while every output retains pending
human review and the AI-provisional origin. Downstream use requires an explicit
``--label-protocol ai-visual-pilot`` opt-in in run_real_driving.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from data_manifest import validate_tokens
from prepare_driving import CAMERAS, SPLITS, fresh_directory, sha256, write_json

SCHEMA = "ai-visual-pilot-label-import-v1"
PROTOCOL = "ai-visual-pilot"
LABEL_SCOPE = "AI-silver-label pilot, not human-ground-truth accuracy"
ORIGIN = "ai_visual_provisional"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _preparation(prepared_dir):
    metadata = _read(prepared_dir / "prepare_metadata.json")
    require(isinstance(metadata, dict)
            and metadata.get("schema_version") == "new-nuscenes-preparation-v1"
            and metadata.get("split_before_labels") is True,
            "AI pilot requires a frozen split-before-labels preparation")
    for name in ("raw_manifest.json", "selected_samples.json", "split_scenes.json", "rubric.md"):
        require(sha256(prepared_dir / name) == metadata.get("prepared_files", {}).get(name),
                f"Prepared file changed after split freeze: {name}")
    raw, selected, scenes = (_read(prepared_dir / name) for name in
                             ("raw_manifest.json", "selected_samples.json", "split_scenes.json"))
    require(isinstance(raw, list) and all(isinstance(row, dict) for row in raw),
            "Invalid raw manifest")
    require(isinstance(selected, list) and selected
            and all(isinstance(row, dict) for row in selected), "Invalid prepared selection")
    require(isinstance(scenes, list) and all(isinstance(row, dict) for row in scenes),
            "Invalid frozen scene allocation")
    validate_tokens(raw, source="raw manifest")
    validate_tokens(selected, source="prepared selection")
    scene_splits = {row["scene_token"]: row["split"] for row in scenes}
    require(len(scene_splits) == len(scenes), "Duplicate frozen scene assignments")
    eligible = {row["sample_token"]: row for row in raw if not row.get("exclusion_reason")}
    require(set(eligible) == {row["sample_token"] for row in selected},
            "Prepared selection differs from eligible raw manifest")
    for row in selected:
        source = eligible[row["sample_token"]]
        require(all(source.get(key) == value for key, value in row.items()),
                "Prepared selection metadata differs from raw manifest")
        require(source.get("split") in SPLITS
                and source.get("split") == scene_splits.get(source.get("scene_token")),
                "Prepared scene allocation mismatch")
        require(set(source.get("image_files", {})) == set(CAMERAS)
                and all(isinstance(source["image_files"][camera].get("sha256"), str)
                        and len(source["image_files"][camera]["sha256"]) == 64
                        for camera in CAMERAS), "Prepared camera hashes missing")
    return metadata, raw, selected, eligible


def _annotation(row, source, position, rubric_sha256):
    require(isinstance(row, dict), "AI annotation must be a JSON object")
    require(row.get("rubric_sha256") == rubric_sha256,
            "AI annotation rubric_sha256 must match the frozen preparation rubric")
    require(type(row.get("index")) is int and row["index"] == position,
            "AI annotation index differs from frozen selected order")
    require(row.get("label") in ("normal_core", "not_normal_core", "uncertain"),
            "Unknown AI label; use normal_core/not_normal_core/uncertain")
    require(row.get("label_origin") == ORIGIN
            and row.get("human_review_status") == "pending",
            "AI annotation must retain AI origin and pending human review")
    require(isinstance(row.get("reviewer"), str) and row["reviewer"].startswith("AI ")
            and row["reviewer"].strip(), "AI reviewer identity must start with 'AI '")
    require(row.get("confidence") in ("high", "medium", "low"), "Unknown AI confidence")
    require(row["confidence"] != "low" or row["label"] == "uncertain",
            "Low-confidence AI judgments must remain uncertain")
    evidence = row.get("visible_evidence")
    require(isinstance(evidence, list) and evidence
            and all(isinstance(item, str) and item.strip() for item in evidence),
            "AI annotation requires visible evidence")
    require(isinstance(row.get("uncertainty_reason"), str),
            "AI annotation requires an uncertainty_reason string")
    require(row["label"] != "uncertain" or row["uncertainty_reason"].strip(),
            "Uncertain AI judgments require an explanation")
    cameras = row.get("cameras_inspected")
    require(isinstance(cameras, list) and len(cameras) == len(CAMERAS)
            and set(cameras) == set(CAMERAS), "AI annotation must record all six cameras")
    require(all(key not in row or row[key] == source[key] for key in ("scene_token", "split")),
            "AI annotation attempted to change frozen scene/split")
    return {key: row[key] for key in
            ("index", "label", "confidence", "visible_evidence", "uncertainty_reason",
             "cameras_inspected", "reviewer")}


def import_ai_labels(prepared_dir: Path, annotation_paths: list[Path], output_dir: Path) -> dict:
    """Write a new AI-pilot bundle, preserving raw judgments and frozen allocation."""
    prepared_dir = Path(prepared_dir).resolve()
    preparation, raw, selected, eligible = _preparation(prepared_dir)
    positions = {row["sample_token"]: index for index, row in enumerate(selected)}
    require(annotation_paths, "At least one AI annotation JSON is required")
    annotations, source_hashes = {}, {}
    for source_path in map(Path, annotation_paths):
        path = source_path.resolve()
        require(str(path) not in source_hashes, "Duplicate AI annotation input path")
        source_hashes[str(path)] = sha256(path)
        rows = _read(path)
        require(isinstance(rows, list) and rows, "AI annotation input must be a nonempty array")
        for line, row in enumerate(rows, 1):
            token = row.get("sample_token") if isinstance(row, dict) else None
            require(isinstance(token, str) and token in eligible,
                    f"Unknown or excluded AI sample_token {token!r}")
            require(token not in annotations, f"Duplicate AI annotation sample_token {token}")
            validated = _annotation(row, eligible[token], positions[token],
                                    preparation["rubric_sha256"])
            annotations[token] = {**validated, "annotation_source_path": str(path),
                                  "annotation_source_row": line}
    require(set(annotations) == set(eligible),
            "AI annotations must cover the exact complete prepared selection")

    labels, manifest, excluded = [], [], []
    for source in raw:
        token = source["sample_token"]
        if source.get("exclusion_reason"):
            excluded.append({**source, "original_split": source["split"], "split": "excluded"})
            continue
        annotation = annotations[token]
        binary = {"normal_core": 0, "not_normal_core": 1, "uncertain": None}[annotation["label"]]
        record = {**annotation, "sample_token": token, "scene_token": source["scene_token"],
                  "label": ("uncertain" if binary is None else
                            "long_tail" if binary else "normal_core"),
                  "submitted_label": annotation["label"], "label_source": ORIGIN,
                  "label_origin": ORIGIN, "review_status": "ai_provisional",
                  "human_review_status": "pending", "label_protocol": PROTOCOL,
                  "label_scope": LABEL_SCOPE, "source_version": source["source_version"],
                  "evaluation_scope": source["evaluation_scope"],
                  "rubric_version": preparation["rubric_version"],
                  "rubric_sha256": preparation["rubric_sha256"],
                  "image_sha256": {camera: source["image_files"][camera]["sha256"]
                                   for camera in CAMERAS}}
        labels.append(record)
        if binary is None:
            excluded.append({**source, **record, "original_split": source["split"],
                             "split": "excluded",
                             "exclusion_reason": "uncertain_ai_visual_provisional"})
        else:
            manifest.append({**{key: record[key] for key in (
                                "sample_token", "scene_token", "label_source", "label_origin",
                                "review_status", "human_review_status", "reviewer",
                                "label_protocol", "label_scope", "source_version",
                                "evaluation_scope", "rubric_version", "rubric_sha256")},
                             "label": binary, "split": source["split"], "exclusion_reason": ""})
    support = {split: {"normal": sum(row["split"] == split and row["label"] == 0
                                     for row in manifest),
                       "tail": sum(row["split"] == split and row["label"] == 1
                                   for row in manifest)} for split in SPLITS}
    image_manifest = {row["sample_token"]: row["image_sha256"] for row in labels}
    metadata = {"schema_version": SCHEMA, "label_protocol": PROTOCOL,
                "label_scope": LABEL_SCOPE, "human_review_complete": False,
                "scope": preparation["scope"], "split_recomputed": False,
                "annotation_inputs_sha256": source_hashes,
                "prepare_metadata_sha256": sha256(prepared_dir / "prepare_metadata.json"),
                "raw_manifest_sha256": sha256(prepared_dir / "raw_manifest.json"),
                "rubric_version": preparation["rubric_version"],
                "rubric_sha256": preparation["rubric_sha256"],
                "image_manifest_sha256": hashlib.sha256(json.dumps(
                    image_manifest, sort_keys=True).encode()).hexdigest(),
                "source_annotations": len(annotations), "included_labels": len(manifest),
                "excluded_rows": len(excluded), "support": support,
                "ready_for_current_training_contract": all(all(value.values())
                                                           for value in support.values()),
                "provenance_limit": "AI-generated silver labels, not human ground truth; "
                                    "file hashes do not establish judgment correctness."}
    out = fresh_directory(output_dir)
    for name, value in (("labels.json", labels), ("manifest.json", manifest),
                        ("excluded_rows.json", excluded), ("import_metadata.json", metadata)):
        write_json(out / name, value)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = import_ai_labels(args.prepared_dir, args.annotations, args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
