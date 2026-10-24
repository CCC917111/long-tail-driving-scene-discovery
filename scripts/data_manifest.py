"""Frozen, scene-disjoint sample assignments shared by all controlled runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")


def validate_tokens(rows: list[dict], *, source: str) -> None:
    seen = set()
    for i, row in enumerate(rows):
        token = row.get("sample_token")
        if not isinstance(token, str) or not token.strip():
            raise ValueError(f"{source} row {i}: missing/invalid sample_token")
        if token in seen:
            raise ValueError(f"{source}: duplicate sample_token {token!r}")
        seen.add(token)


def create_manifest(meta: list[dict], labels: np.ndarray,
                    splits: dict[str, np.ndarray]) -> list[dict]:
    validate_tokens(meta, source="metadata")
    if len(labels) != len(meta):
        raise ValueError("Labels and metadata must be row-aligned")
    assignment = {}
    for name, indices in splits.items():
        if name not in SPLITS:
            raise ValueError(f"Invalid split {name!r}")
        for index in indices:
            index = int(index)
            if index in assignment or not 0 <= index < len(meta):
                raise ValueError("Split indices overlap or are out of range")
            assignment[index] = name
    if len(assignment) != len(meta):
        raise ValueError("Split indices do not cover every labelled sample")
    rows = [{"sample_token": m["sample_token"], "scene_token": m.get("scene_token"),
             "label": int(labels[i]), "split": assignment[i],
             "label_source": m.get("label_source", "unverified"),
             "review_status": m.get("review_status", "unverified"),
             "exclusion_reason": ""} for i, m in enumerate(meta)]
    _validate_assignment(rows, meta, labels)
    return rows


def _validate_assignment(rows, meta, labels):
    validate_tokens(rows, source="manifest")
    validate_tokens(meta, source="metadata")
    if len(meta) != len(labels):
        raise ValueError("Labels and metadata must be row-aligned")
    by_token = {m["sample_token"]: (i, m) for i, m in enumerate(meta)}
    used, scene_split = set(), {}
    result = {name: [] for name in SPLITS}
    for row in rows:
        token = row["sample_token"]
        if row.get("exclusion_reason"):
            if row.get("split") not in (None, "", "excluded"):
                raise ValueError(f"Excluded token {token} has an active split")
            if token in by_token:
                raise ValueError(f"Excluded token {token} still appears in loaded data")
            continue
        if token not in by_token:
            raise ValueError(f"Manifest token {token} missing from labelled data")
        index, entry = by_token[token]
        label = row.get("label")
        if type(label) is not int or label not in (0, 1) or label != labels[index]:
            raise ValueError(f"Label conflict for {token}")
        scene = row.get("scene_token")
        if not isinstance(scene, str) or not scene.strip() or scene != entry.get("scene_token"):
            raise ValueError(f"Missing or conflicting scene_token for {token}")
        split = row.get("split")
        if split not in SPLITS:
            raise ValueError(f"Invalid split for {token}: {split!r}")
        if scene_split.setdefault(scene, split) != split:
            raise ValueError(f"Scene {scene} leaks across splits")
        used.add(token)
        result[split].append(index)
    if used != set(by_token):
        raise ValueError("Manifest does not cover exactly the loaded labelled samples")
    if any(not result[name] for name in SPLITS):
        raise ValueError("Manifest must contain nonempty train, val and test splits")
    return {name: np.array(indices, dtype=np.int64) for name, indices in result.items()}


def load_frozen_manifest(path: Path, meta: list[dict], labels: np.ndarray) -> dict:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Manifest must be a JSON array")
    return _validate_assignment(rows, meta, labels)
