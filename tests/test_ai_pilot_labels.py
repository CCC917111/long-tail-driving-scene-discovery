"""AI silver-label provenance checks on synthetic data only, never human labels."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ai_pilot_labels import LABEL_SCOPE, SCHEMA, import_ai_labels  # noqa: E402
from prepare_driving import CAMERAS, prepare  # noqa: E402
from run_real_driving import validate_label_import, validate_workflow  # noqa: E402
from test_prepare_driving import synthetic_nuscenes  # noqa: E402
from test_real_driving_workflow import handoff  # noqa: E402,F401


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def annotations_for(prepared):
    selected = json.loads((prepared / "selected_samples.json").read_text())
    rubric_sha256 = json.loads((prepared / "prepare_metadata.json").read_text())["rubric_sha256"]
    return [{"index": index, "sample_token": row["sample_token"],
             "label": "normal_core" if row["sample_token"].endswith("-00")
             else "not_normal_core", "confidence": "high",
             "visible_evidence": ["SYNTHETIC software fixture; not an image judgment"],
             "uncertainty_reason": "", "cameras_inspected": list(CAMERAS),
             "reviewer": "AI synthetic test", "label_origin": "ai_visual_provisional",
             "human_review_status": "pending", "rubric_sha256": rubric_sha256}
            for index, row in enumerate(selected)]


@pytest.fixture
def ai_input(tmp_path):
    data = synthetic_nuscenes(tmp_path / "synthetic-data", frames=2)
    prepared = tmp_path / "prepared"
    prepare(data, "v1.0-mini", prepared, frames_per_scene=2)
    rows = annotations_for(prepared)
    source = tmp_path / "ai-annotations.json"
    write(source, rows)
    return SimpleNamespace(prepared=prepared, rows=rows, source=source,
                           output=tmp_path / "ai-annotations")


def test_ai_import_keeps_origin_frozen_splits_and_uncertain_exclusion(ai_input):
    ai_input.rows[0].update(label="uncertain", confidence="low",
                            uncertainty_reason="Synthetic uncertain test")
    write(ai_input.source, ai_input.rows)
    metadata = import_ai_labels(ai_input.prepared, [ai_input.source], ai_input.output)
    assert metadata["schema_version"] == SCHEMA
    assert metadata["human_review_complete"] is False
    assert metadata["label_scope"] == LABEL_SCOPE
    assert metadata["included_labels"] == 19 and metadata["excluded_rows"] == 1
    assert metadata["split_recomputed"] is False
    labels = json.loads((ai_input.output / "labels.json").read_text())
    manifest = json.loads((ai_input.output / "manifest.json").read_text())
    raw = {row["sample_token"]: row for row in
           json.loads((ai_input.prepared / "raw_manifest.json").read_text())}
    assert all(row["review_status"] == "ai_provisional"
               and row["human_review_status"] == "pending" for row in labels)
    assert all(row["label_source"] == "ai_visual_provisional"
               and row["label_origin"] == "ai_visual_provisional" for row in labels)
    assert labels[0]["label"] == "uncertain"
    assert ai_input.rows[0]["sample_token"] not in {row["sample_token"] for row in manifest}
    assert all(row["split"] == raw[row["sample_token"]]["split"] for row in manifest)
    excluded = json.loads((ai_input.output / "excluded_rows.json").read_text())
    assert excluded[0]["exclusion_reason"] == "uncertain_ai_visual_provisional"
    assert all(set(row["image_sha256"]) == set(CAMERAS) for row in labels)


@pytest.mark.parametrize("case,message", [
    ("unknown", "Unknown or excluded AI sample_token"),
    ("duplicate", "Duplicate AI annotation sample_token"),
    ("missing", "exact complete prepared selection"),
    ("label", "Unknown AI label"),
    ("index", "index differs"),
    ("camera", "all six cameras"),
    ("human", "pending human review"),
    ("origin", "AI origin"),
    ("reviewer", "AI reviewer identity"),
    ("confidence", "Low-confidence"),
    ("evidence", "visible evidence"),
    ("uncertain", "require an explanation"),
    ("split", "change frozen scene/split"),
    ("wrong_rubric", "rubric_sha256 must match the frozen"),
    ("missing_rubric", "rubric_sha256 must match the frozen"),
    ("root", "nonempty array"),
])
def test_invalid_ai_annotations_fail_before_output(ai_input, case, message):
    rows = ai_input.rows
    if case == "unknown":
        rows[0]["sample_token"] = "synthetic-unknown-token"
    elif case == "duplicate":
        rows.append(rows[0])
    elif case == "missing":
        rows.pop()
    elif case == "label":
        rows[0]["label"] = "anything-is-tail"
    elif case == "index":
        rows[0]["index"] = 999
    elif case == "camera":
        rows[0]["cameras_inspected"] = list(CAMERAS[:-1])
    elif case == "human":
        rows[0]["human_review_status"] = "reviewed"
    elif case == "origin":
        rows[0]["label_origin"] = "human"
    elif case == "reviewer":
        rows[0]["reviewer"] = "Yike"
    elif case == "confidence":
        rows[0]["confidence"] = "low"
    elif case == "evidence":
        rows[0]["visible_evidence"] = []
    elif case == "uncertain":
        rows[0]["label"] = "uncertain"
    elif case == "split":
        rows[0]["split"] = "changed"
    elif case == "wrong_rubric":
        rows[0]["rubric_sha256"] = "0" * 64
    elif case == "missing_rubric":
        del rows[0]["rubric_sha256"]
    elif case == "root":
        rows = {"samples": rows}
    write(ai_input.source, rows)
    with pytest.raises(ValueError, match=message):
        import_ai_labels(ai_input.prepared, [ai_input.source], ai_input.output)
    assert not ai_input.output.exists()


def test_ai_replay_is_opt_in_and_rejects_modified_inputs_or_outputs(ai_input):
    metadata = import_ai_labels(ai_input.prepared, [ai_input.source], ai_input.output)
    with pytest.raises(ValueError, match="Manual label protocol rejects AI"):
        validate_label_import(SimpleNamespace(), ai_input.prepared, ai_input.output, metadata)
    args = SimpleNamespace(label_protocol="ai-visual-pilot")
    evidence, identity = validate_label_import(args, ai_input.prepared, ai_input.output, metadata)
    assert len(evidence) == 1 and identity["human_reviewed"] is False
    assert identity["human_ground_truth_verified"] is False
    labels_path = ai_input.output / "labels.json"
    labels = json.loads(labels_path.read_text())
    labels[0]["label"] = "long_tail" if labels[0]["label"] == "normal_core" else "normal_core"
    write(labels_path, labels)
    with pytest.raises(ValueError, match="labels.json differs from replay"):
        validate_label_import(args, ai_input.prepared, ai_input.output, metadata)
    ai_input.source.write_text(ai_input.source.read_text() + "\n")
    with pytest.raises(ValueError, match="AI annotation JSON missing or hash mismatch"):
        validate_label_import(args, ai_input.prepared, ai_input.output, metadata)


def test_ai_replay_can_relocate_unchanged_source_json(ai_input):
    metadata = import_ai_labels(ai_input.prepared, [ai_input.source], ai_input.output)
    source_name = ai_input.source.name
    shutil.move(str(ai_input.source), ai_input.output / source_name)
    evidence, identity = validate_label_import(
        SimpleNamespace(label_protocol="ai-visual-pilot"), ai_input.prepared,
        ai_input.output, metadata)
    assert evidence[0]["resolved_path"] == str(ai_input.output / source_name)
    assert identity["label_scope"] == LABEL_SCOPE


def test_ai_import_rejects_changed_frozen_rubric(ai_input):
    (ai_input.prepared / "rubric.md").write_text("changed rubric")
    with pytest.raises(ValueError, match="Prepared file changed after split freeze"):
        import_ai_labels(ai_input.prepared, [ai_input.source], ai_input.output)
    assert not ai_input.output.exists()


def test_full_ai_handoff_opt_in_and_both_class_gate(handoff):  # noqa: F811
    annotations = handoff.annotations_dir.parent / "ai-pilot-annotations"
    source = handoff.annotations_dir.parent / "ai-visual.json"
    rows = annotations_for(handoff.prepared_dir)
    write(source, rows)
    import_ai_labels(handoff.prepared_dir, [source], annotations)
    handoff.annotations_dir = annotations
    with pytest.raises(ValueError, match="Manual label protocol rejects AI"):
        validate_workflow(handoff)
    handoff.label_protocol = "ai-visual-pilot"
    report = validate_workflow(handoff)
    assert report["status"] == "validated" and report["human_reviewed"] is False
    assert report["label_scope"] == LABEL_SCOPE
    assert report["review_csv_evidence"] == [] and len(report["ai_annotation_evidence"]) == 1
    selected = json.loads((handoff.prepared_dir / "selected_samples.json").read_text())
    split_of = {row["sample_token"]: row["split"] for row in selected}
    for row in rows:
        if split_of[row["sample_token"]] == "test":
            row["label"] = "normal_core"
    write(source, rows)
    single_class = annotations.parent / "single-class-ai"
    import_ai_labels(handoff.prepared_dir, [source], single_class)
    handoff.annotations_dir = single_class
    with pytest.raises(ValueError, match="Frozen test split must contain both classes"):
        validate_workflow(handoff)
