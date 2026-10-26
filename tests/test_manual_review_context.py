"""Synthetic checks for truthful provenance after a human sees AI labels."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_runs import (  # noqa: E402
    align_rows,
    label_scope_from_manifest,
    write_evaluation_artifacts,
)
from evaluation_fixture import create_fixture  # noqa: E402
from prepare_driving import (  # noqa: E402
    POST_AI_LABEL_ORIGIN,
    POST_AI_REVIEW_CONTEXT,
    import_labels,
    post_ai_review_scope,
    prepare,
    sha256,
)
from run_real_driving import train_workflow, validate_label_import  # noqa: E402
from summarize_runs import collect_runs  # noqa: E402
from test_prepare_driving import synthetic_nuscenes, write_reviews  # noqa: E402
from test_real_driving_workflow import handoff  # noqa: E402,F401


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def manual_input(tmp_path):
    prepared = tmp_path / "prepared"
    prepare(synthetic_nuscenes(tmp_path / "synthetic-data"), "v1.0-mini", prepared)
    csv_path = tmp_path / "reviewed.csv"
    rows = write_reviews(prepared, csv_path)
    for row in rows:
        row["review_context"] = POST_AI_REVIEW_CONTEXT
    write_csv(csv_path, rows)
    return SimpleNamespace(prepared=prepared, source=csv_path, rows=rows,
                           output=tmp_path / "annotations")


def test_manual_context_import_and_exact_replay(manual_input):
    data = manual_input
    before = data.source.read_bytes()
    metadata = import_labels(data.prepared, [data.source], data.output)
    scope = post_ai_review_scope()
    assert metadata["review_context"] == POST_AI_REVIEW_CONTEXT
    assert metadata["label_scope"] == scope
    assert metadata["review_inputs_sha256"][str(data.source.resolve())] == sha256(data.source)
    assert data.source.read_bytes() == before
    labels = json.loads((data.output / "labels.json").read_text())
    manifest = json.loads((data.output / "manifest.json").read_text())
    for row in labels + manifest:
        assert row["review_context"] == POST_AI_REVIEW_CONTEXT
        assert row["label_origin"] == POST_AI_LABEL_ORIGIN
        assert row["reviewer"] == "synthetic-unit-test-reviewer"
    assert label_scope_from_manifest(manifest) == scope
    evidence, identity = validate_label_import(
        SimpleNamespace(label_protocol="manual"), data.prepared, data.output, metadata)
    assert evidence[0]["sha256"] == sha256(data.source)
    assert identity["label_scope"] == scope
    assert identity["human_reviewed"] is True
    assert identity["human_ground_truth_verified"] is False
    # Scope cannot be appended or changed after the frozen import.
    labels[0]["review_context"] = ""
    (data.output / "labels.json").write_text(json.dumps(labels))
    with pytest.raises(ValueError, match="labels.json differs from replay"):
        validate_label_import(SimpleNamespace(), data.prepared, data.output, metadata)


def test_omitted_context_preserves_legacy_manual_outputs(manual_input):
    data = manual_input
    for row in data.rows:
        row.pop("review_context")
    write_csv(data.source, data.rows)
    metadata = import_labels(data.prepared, [data.source], data.output)
    assert "label_scope" not in metadata and "review_context" not in metadata
    for name in ("labels.json", "manifest.json", "excluded_rows.json"):
        rows = json.loads((data.output / name).read_text())
        assert all("review_context" not in row and "label_origin" not in row for row in rows)
    _, identity = validate_label_import(SimpleNamespace(), data.prepared, data.output, metadata)
    assert identity["label_scope"] == "Recorded manual judgments"
    manifest = json.loads((data.output / "manifest.json").read_text())
    assert label_scope_from_manifest(manifest) is None


@pytest.mark.parametrize("case", ["unknown", "mixed", "two_reviewers", "uncertain_reviewer",
                                  "ai_origin", "ai_source"])
def test_invalid_or_misrepresented_manual_context_fails_before_output(manual_input, case):
    data = manual_input
    if case == "unknown":
        data.rows[0]["review_context"] = "independent-human-review"
    elif case == "mixed":
        data.rows[0]["review_context"] = ""
    elif case == "two_reviewers":
        data.rows[0]["reviewer"] = "second-synthetic-reviewer"
    elif case == "uncertain_reviewer":
        data.rows[0].update(label="uncertain", review_status="uncertain",
                            reviewer="second-synthetic-reviewer")
    elif case == "ai_origin":
        data.rows[0]["label_origin"] = "ai_visual_provisional"
    else:
        data.rows[0]["label_source"] = "ai_visual_provisional"
    write_csv(data.source, data.rows)
    with pytest.raises(ValueError):
        import_labels(data.prepared, [data.source], data.output)
    assert not data.output.exists()


def context_fixture(path):
    manifest = create_fixture(path)
    for row in manifest:
        row.update(review_context=POST_AI_REVIEW_CONTEXT,
                   label_origin=POST_AI_LABEL_ORIGIN, review_status="reviewed",
                   reviewer="synthetic-single-reviewer")
    return manifest, json.loads((path / "rows.json").read_text())


@pytest.mark.parametrize("case", ["unknown", "mixed", "two_reviewers", "ai_origin",
                                  "ai_status", "ai_source"])
def test_evaluator_rejects_mixed_or_ai_manual_scope(tmp_path, case):
    manifest, _ = context_fixture(tmp_path / "fixture")
    if case == "unknown":
        manifest[0]["review_context"] = "unknown"
    elif case == "mixed":
        manifest[0].pop("review_context")
    elif case == "two_reviewers":
        manifest[0]["reviewer"] = "another-synthetic-reviewer"
    elif case == "ai_origin":
        manifest[0]["label_origin"] = "ai_visual_provisional"
    elif case == "ai_status":
        manifest[0]["review_status"] = "ai_provisional"
    else:
        manifest[0]["label_source"] = "ai_visual_provisional"
    with pytest.raises(ValueError):
        label_scope_from_manifest(manifest)


@pytest.mark.parametrize("field,value", [("review_context", "independent-review"),
                                         ("reviewer", "another-reviewer"),
                                         ("label_origin", "ai_visual_provisional")])
def test_predictions_cannot_override_manifest_provenance(tmp_path, field, value):
    manifest, scores = context_fixture(tmp_path / "fixture")
    scores[0][field] = value
    with pytest.raises(ValueError):
        align_rows(scores, manifest)


@pytest.mark.parametrize("claim", ["independent_human_review", "human_ground_truth_verified"])
def test_predictions_reject_forged_nested_review_scope(tmp_path, claim):
    manifest, scores = context_fixture(tmp_path / "fixture")
    scores[0]["label_scope"] = dict(post_ai_review_scope(), **{claim: True})
    with pytest.raises(ValueError, match="label_scope"):
        align_rows(scores, manifest)


def test_predictions_accept_identical_canonical_nested_review_scope(tmp_path):
    manifest, scores = context_fixture(tmp_path / "fixture")
    scores[0]["label_scope"] = post_ai_review_scope()
    aligned = align_rows(scores, manifest)
    assert aligned[0]["label_scope"] == post_ai_review_scope()
    assert aligned[0]["review_context"] == POST_AI_REVIEW_CONTEXT


def test_unscoped_predictions_preserve_preexisting_nested_scope(tmp_path):
    data = tmp_path / "fixture"
    manifest = create_fixture(data)
    scores = json.loads((data / "rows.json").read_text())
    existing_scope = {"legacy_reference": "preserve original recorded metadata"}
    scores[0]["label_scope"] = existing_scope
    aligned = align_rows(scores, manifest)
    assert aligned[0]["label_scope"] == existing_scope
    assert "review_context" not in aligned[0]


def test_scope_predictions_and_comparison_keep_post_ai_exposure_provenance(tmp_path):
    data = tmp_path / "fixture"
    manifest, scores = context_fixture(data)
    checkpoint = tmp_path / "synthetic-checkpoint.bin"
    checkpoint.write_bytes(b"SYNTHETIC software fixture; not model evidence")
    scope = post_ai_review_scope()
    expected = {
        "label_protocol": "manual", "review_context": POST_AI_REVIEW_CONTEXT,
        "human_reviewed": True, "human_ground_truth_verified": False,
        "independent_human_review": False, "prior_ai_labels_exposed": True,
        "reviewer_count": 1, "official_benchmark": False,
    }
    assert all(scope[key] == value for key, value in expected.items())
    assert "post-AI-exposure" in scope["interpretation"]
    run_dirs = [tmp_path / method for method in ("logistic", "mlp")]
    for run_dir in run_dirs:
        result = write_evaluation_artifacts(
            run_dir, copy.deepcopy(scores), manifest, config={"method": run_dir.name},
            seed=42, checkpoint_path=checkpoint,
            input_paths={"features": data / "features.npy", "meta": data / "meta.json"},
            review_k=4, bootstrap_samples=0)
        metadata = json.loads((run_dir / "run_metadata.json").read_text())
        assert result["label_scope"] == metadata["label_scope"] == scope
        assert metadata["config"]["label_scope"] == scope
        predictions = json.loads((run_dir / "predictions.json").read_text())
        with (run_dir / "predictions.csv").open(newline="") as handle:
            csv_predictions = list(csv.DictReader(handle))
        for row in predictions + csv_predictions:
            assert row["review_context"] == POST_AI_REVIEW_CONTEXT
            assert row["reviewer"] == "synthetic-single-reviewer"
            assert row["label_origin"] == POST_AI_LABEL_ORIGIN
    table, summaries = collect_runs(run_dirs)
    for row in table + summaries:
        assert all(row[key] == value for key, value in expected.items())
        assert row["interpretation"] == scope["interpretation"]


def test_synthetic_sae_legacy_outputs_keep_review_scope(tmp_path):
    data = tmp_path / "fixture"
    manifest, _ = context_fixture(data)
    # A valid frozen manifest may retain excluded records without review context.
    manifest.append({"sample_token": "synthetic-excluded", "split": "excluded",
                     "scene_token": "synthetic-excluded-scene",
                     "exclusion_reason": "synthetic-exclusion"})
    (data / "manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "synthetic-sae"
    script = Path(__file__).resolve().parents[1] / "scripts/sae_abstopk_tail_reward.py"
    subprocess.run(
        [sys.executable, str(script), "--features", str(data / "features.npy"),
         "--meta", str(data / "meta.json"), "--labels", str(data / "labels.json"),
         "--manifest", str(data / "manifest.json"), "--output-dir", str(output),
         "--device", "cpu", "--epochs", "1", "--hidden-dim", "8", "--k", "2",
         "--batch-size", "80", "--review-k", "4", "--bootstrap-samples", "0"],
        check=True, capture_output=True, text=True)
    scope = post_ai_review_scope()
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["label_scope"] == metrics["config"]["label_scope"] == scope
    rows = json.loads((output / "rows.json").read_text())
    assert len(rows) == len(manifest) - 1
    for row in rows:
        assert row["label_scope"] == scope
        assert row["review_context"] == POST_AI_REVIEW_CONTEXT
        assert row["label_origin"] == POST_AI_LABEL_ORIGIN
        assert row["reviewer"] == "synthetic-single-reviewer"


def test_driver_records_consistent_context_without_fitting(handoff, monkeypatch):  # noqa: F811
    source = handoff.annotations_dir.parent / "reviewed.csv"
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["review_context"] = POST_AI_REVIEW_CONTEXT
    context_source = source.with_name("reviewed-with-context.csv")
    write_csv(context_source, rows)
    annotations = handoff.annotations_dir.with_name("context-annotations")
    import_labels(handoff.prepared_dir, [context_source], annotations)
    handoff.annotations_dir = annotations
    handoff.label_protocol = "manual"
    commands = []

    def successful_command(argv, **kwargs):
        commands.append(argv)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("run_real_driving.subprocess.run", successful_command)
    result = train_workflow(handoff)
    assert len(commands) == 5  # Four fits and the comparison were stubbed.
    scope = post_ai_review_scope()
    records = [result]
    for name in ("validation.json", "runner_config.json", "workflow_summary.json"):
        records.append(json.loads((handoff.output_dir / name).read_text()))
    for record in records:
        assert record["label_scope"] == scope
        assert record["label_protocol"] == "manual"
        assert record["review_context"] == POST_AI_REVIEW_CONTEXT
        assert record["human_reviewed"] is True
        assert record["human_ground_truth_verified"] is False
