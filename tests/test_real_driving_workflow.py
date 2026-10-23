"""Synthetic handoff integration checks; no real images, VLM or human judgments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import extract  # noqa: E402
from extraction_contract import build_contract, sha256  # noqa: E402
from prepare_driving import CAMERAS, import_labels, prepare  # noqa: E402
from run_real_driving import train_workflow, validate_workflow, write_json  # noqa: E402
from test_prepare_driving import synthetic_nuscenes, write_reviews  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_real_driving.py"


@pytest.fixture
def handoff(tmp_path):
    data = synthetic_nuscenes(tmp_path / "synthetic-data", frames=2)
    prepared, annotations, extracted = (
        tmp_path / name for name in ("prepared", "annotations", "extract"))
    prepare(data, "v1.0-mini", prepared, frames_per_scene=2)
    reviewed = tmp_path / "reviewed.csv"
    write_reviews(prepared, reviewed)
    import_labels(prepared, [reviewed], annotations)
    selected = json.loads((prepared / "selected_samples.json").read_text())
    model = tmp_path / "synthetic-model"
    model.mkdir()
    (model / "config.json").write_text('{"synthetic_test_only": true}\n')
    (model / "fixture.safetensors").write_bytes(b"NOT MODEL WEIGHTS; synthetic handoff test")
    samples = [dict(row, channels={camera: data / "samples" / row["images"][camera]
                                   for camera in CAMERAS}) for row in selected]
    args = SimpleNamespace(model_revision="a" * 40, model_dir=model, layers=[28],
                           prompt="SYNTHETIC handoff fixture; no VLM inference or real evidence.",
                           dtype="bfloat16", min_pixels=3136, max_pixels=1600000,
                           caption=False, max_new_tokens=256)
    contract = build_contract(args, samples, data / "v1.0-mini", CAMERAS)
    extracted.mkdir()
    write_json(extracted / "extraction_config.json", contract)
    meta = [{key: row[key] for key in
             ("sample_token", "scene_token", "scene_name", "timestamp", "images")}
            for row in selected]
    generator = np.random.default_rng(7)
    values = generator.normal(size=(len(meta), 6)).astype(np.float32)
    for index, row in enumerate(meta):
        row["row_index"] = index
        values[index, :2] += 2 * int(row["sample_token"].rsplit("-", 1)[1]) - 1
    rows = {(28, kind): list(values) for kind in ("mean", "last")}
    extract.save_checkpoint(extracted, [28], rows, meta, [])
    state = json.loads((extracted / "checkpoint_state.json").read_text())
    write_json(extracted / "extraction_summary.json", {
        "status": "completed", "selection_complete": True, "rows_in_cache": len(meta),
        "selected_rows_total": len(meta), "remaining_rows": 0, "model_revision": "a" * 40,
        "extraction_config_sha256": sha256(extracted / "extraction_config.json"),
        "checkpoint_generation": state["generation"],
        "feature_files": {name: {"sha256": record["sha256"],
                                 "bytes": (extracted / name).stat().st_size}
                          for name, record in state["files"].items() if name.endswith(".npy")}})
    return SimpleNamespace(
        prepared_dir=prepared, annotations_dir=annotations, extract_dir=extracted,
        output_dir=tmp_path / "runs", layer=28, pooling="mean", seed=42,
        epochs=1, hidden_dim=8, k=2, review_k=2, command="train")


def rewrite(path, mutation):
    value = json.loads(path.read_text())
    mutation(value)
    write_json(path, value)


def cli_args(args, command="validate"):
    argv = [sys.executable, str(SCRIPT), command, "--prepared-dir", str(args.prepared_dir),
            "--annotations-dir", str(args.annotations_dir), "--extract-dir", str(args.extract_dir)]
    if getattr(args, "feature_protocol", "cosmos") != "cosmos":
        argv.extend(["--feature-protocol", args.feature_protocol, "--layer", str(args.layer),
                     "--pooling", args.pooling])
    return argv


def test_valid_handoff_is_read_only_and_portable(handoff, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Validation must not load a VLM or restore/prune caches")
    monkeypatch.setattr(extract, "load_vlm", forbidden)
    monkeypatch.setattr(extract, "_mirror_checkpoint", forbidden)
    before = {str(path): path.read_bytes()
              for path in handoff.extract_dir.rglob("*") if path.is_file()}
    report = validate_workflow(handoff)
    assert report["status"] == "validated" and report["extracted_rows"] == 20
    assert report["support"]["test"] == {"normal": 2, "tail": 2, "scenes": 2}
    assert "not established" in report["provenance_limit"]
    assert before == {str(path): path.read_bytes()
                      for path in handoff.extract_dir.rglob("*") if path.is_file()}
    # A transferred WORK bundle keeps reviewed.csv alongside its three directories.
    original = handoff.annotations_dir.parent
    copied = original / "copied-work"
    copied.mkdir()
    for name in ("prepared", "annotations", "extract"):
        shutil.copytree(original / name, copied / name)
    shutil.move(str(original / "reviewed.csv"), copied / "reviewed.csv")
    relocated = SimpleNamespace(**{
        **vars(handoff), "prepared_dir": copied / "prepared",
        "annotations_dir": copied / "annotations", "extract_dir": copied / "extract"})
    report = validate_workflow(relocated)
    assert report["review_csv_evidence"][0]["recorded_path"] == str(original / "reviewed.csv")
    assert report["review_csv_evidence"][0]["resolved_path"] == str(copied / "reviewed.csv")


@pytest.mark.parametrize("case,message", [
    ("partial", "selection is incomplete"),
    ("missing_config", "Required handoff file missing"),
    ("missing_state", "Required handoff file missing"),
    ("swapped_features", "consumer hash/file mismatch"),
    ("changed_config", "contract hash mismatch"),
    ("manifest", "manifest.json differs from replay"),
    ("coordinated_labels", "labels.json differs from replay"),
    ("missing_reviews", "Review CSV missing or hash mismatch"),
    ("source_version", "source_version differs"),
])
def test_bad_handoff_fails_before_any_training(handoff, monkeypatch, case, message):
    if case == "partial":
        rewrite(handoff.extract_dir / "extraction_summary.json",
                lambda value: value.update(selection_complete=False))
    elif case.startswith("missing_") and case in ("missing_config", "missing_state"):
        name = "extraction_config.json" if case == "missing_config" else "checkpoint_state.json"
        (handoff.extract_dir / name).unlink()
    elif case == "swapped_features":
        path = handoff.extract_dir / "layer28_mlp_output_mean.npy"
        np.save(path, np.load(path)[::-1])
    elif case == "changed_config":
        rewrite(handoff.extract_dir / "extraction_config.json",
                lambda value: value.update(dtype="float16"))
    elif case == "manifest":
        rewrite(handoff.annotations_dir / "manifest.json",
                lambda value: value[0].update(split="changed"))
    elif case == "coordinated_labels":
        rewrite(handoff.annotations_dir / "labels.json",
                lambda value: value[0].update(label="long_tail"))
        rewrite(handoff.annotations_dir / "manifest.json", lambda value: value[0].update(label=1))
    elif case == "missing_reviews":
        (handoff.annotations_dir.parent / "reviewed.csv").unlink()
    elif case == "source_version":
        rewrite(handoff.extract_dir / "extraction_config.json",
                lambda value: value.update(source_version="v1.0-trainval"))
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs:
                        pytest.fail("Training started for invalid handoff"))
    with pytest.raises(ValueError, match=message):
        train_workflow(handoff)
    assert not handoff.output_dir.exists()


@pytest.mark.parametrize("mutation,message", [("swap", "row order"), ("scene", "metadata differs")])
def test_committed_metadata_identity_is_checked_beyond_hashes(handoff, mutation, message):
    state_path = handoff.extract_dir / "checkpoint_state.json"
    state = json.loads(state_path.read_text())
    path = handoff.extract_dir / state["files"]["meta.json"]["path"]
    meta = json.loads(path.read_text())
    if mutation == "swap":
        meta[0], meta[1] = meta[1], meta[0]
        for index, row in enumerate(meta):
            row["row_index"] = index
    else:
        meta[0]["scene_token"] = "synthetic-wrong-scene"
    write_json(path, meta)
    state["files"]["meta.json"]["sha256"] = sha256(path)
    write_json(state_path, state)
    shutil.copyfile(path, handoff.extract_dir / "meta.json")
    with pytest.raises(ValueError, match=message):
        validate_workflow(handoff)


def test_cli_failure_emits_json_report_without_creating_run(handoff):
    rewrite(handoff.extract_dir / "extraction_summary.json",
            lambda value: value.update(selection_complete=False))
    report_path = handoff.annotations_dir.parent / "failed-validation.json"
    result = subprocess.run([*cli_args(handoff), "--output", str(report_path)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    report = json.loads(result.stdout)
    assert report["status"] == "failed" and report["training_started"] is False
    assert json.loads(report_path.read_text()) == report
    assert not handoff.output_dir.exists()


def test_frozen_test_split_requires_both_classes_after_valid_import(handoff):
    reviewed = handoff.annotations_dir.parent / "reviewed.csv"
    with reviewed.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    changed = 0
    for row in rows:
        if row["split"] == "test" and row["label"] == "long_tail":
            row["label"] = "normal_core"
            changed += 1
    assert changed > 0
    with reviewed.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    annotations = handoff.annotations_dir.parent / "one-class-annotations"
    import_labels(handoff.prepared_dir, [reviewed], annotations)
    handoff.annotations_dir = annotations
    with pytest.raises(ValueError, match="Frozen test split must contain both classes"):
        validate_workflow(handoff)


def test_cli_rejects_malformed_summary_root_as_json(handoff):
    write_json(handoff.extract_dir / "extraction_summary.json", [])
    result = subprocess.run(cli_args(handoff), capture_output=True, text=True)
    assert result.returncode != 0
    report = json.loads(result.stdout)
    assert report["status"] == "failed" and report["training_started"] is False
    assert "extraction_summary.json" in report["error"]
    assert not handoff.output_dir.exists()


@pytest.mark.parametrize("fixture_name", ["handoff", "cpu_handoff"])
def test_actual_tiny_cpu_four_method_synthetic_workflow(request, fixture_name):
    """Fit real estimators on synthetic arrays, never claim a real-data result."""
    handoff = request.getfixturevalue(fixture_name)
    command = [*cli_args(handoff, "train"), "--output-dir", str(handoff.output_dir),
               "--review-k", "2", "--epochs", "1", "--hidden-dim", "8", "--k", "2"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout)
    assert summary["status"] == "completed"
    assert summary["feature_protocol"] == getattr(handoff, "feature_protocol", "cosmos")
    if summary["feature_protocol"] == "cpu-resnet18":
        assert summary["encoder"] == "resnet18-imagenet1k-v1"
        assert "concatenated" in summary["feature_semantics"]
    table = json.loads((handoff.output_dir / "comparison/main_table.json").read_text())
    assert {row["method"] for row in table} == {
        "logistic", "mlp", "sae_abstopk_none", "sae_abstopk_norm"}
    assert len({row["manifest_sha256"] for row in table}) == 1
    assert len({row["features_sha256"] for row in table}) == 1
    assert all(row["seed"] == 42 and row["requested_k"] == 2 for row in table)
    commands = json.loads((handoff.output_dir / "commands.json").read_text())
    assert len(commands) == 5 and all(row["returncode"] == 0 for row in commands)
    assert all(Path(row["log"]).is_file() for row in commands)
    configs = [json.loads((handoff.output_dir / method / "run_metadata.json").read_text())["config"]
               for method in summary["methods"]]
    assert all(config["feature_provenance"] == configs[0]["feature_provenance"]
               for config in configs)
    assert configs[2]["device"] == configs[3]["device"] == "cpu"
    for key in ("hidden_dim", "tail_ratio", "dropout", "k", "sparsity", "alpha", "beta",
                "reward_coeff", "reward_max", "epochs", "batch_size", "lr", "weight_decay",
                "act_threshold"):
        assert configs[2][key] == configs[3][key]
    before = (handoff.output_dir / "workflow_summary.json").read_bytes()
    with pytest.raises(ValueError, match="fresh, empty"):
        train_workflow(handoff)
    assert before == (handoff.output_dir / "workflow_summary.json").read_bytes()


def write_cpu_checkpoint(args, values, meta, *, last_values=None):
    """Synthetic array evidence only; this fixture does not run the image encoder."""
    from cpu_image_encoder import WEIGHTS_SHA256

    if last_values is None:
        last_values = values
    extract.save_checkpoint(args.extract_dir, [1],
                            {(1, "mean"): list(values), (1, "last"): list(last_values)}, meta, [])
    state = json.loads((args.extract_dir / "checkpoint_state.json").read_text())
    write_json(args.extract_dir / "extraction_summary.json", {
        "status": "completed", "selection_complete": True, "rows_in_cache": len(meta),
        "feature_protocol": "cpu-resnet18-six-camera-v1", "encoder": "resnet18-imagenet1k-v1",
        "device": "cpu",
        "selected_rows_total": len(meta), "remaining_rows": 0, "model_revision": WEIGHTS_SHA256,
        "extraction_config_sha256": sha256(args.extract_dir / "extraction_config.json"),
        "checkpoint_generation": state["generation"],
        "feature_files": {name: {"sha256": record["sha256"],
                                 "bytes": (args.extract_dir / name).stat().st_size}
                          for name, record in state["files"].items() if name.endswith(".npy")}})


@pytest.fixture
def cpu_handoff(handoff):
    from cpu_image_encoder import WEIGHTS_SHA256

    source = handoff.extract_dir
    base = json.loads((source / "extraction_config.json").read_text())
    meta = json.loads((source / "meta.json").read_text())
    values = np.tile(np.load(source / "layer28_mlp_output_mean.npy"), (1, 512))
    contract = {key: base[key] for key in
                ("source_version", "metadata_sha256", "samples", "camera_order")}
    contract.update({
        "schema": "cpu-resnet18-six-camera-v1", "encoder": "resnet18-imagenet1k-v1",
        "layers": [1], "hook": "resnet18 avgpool; six-camera concatenation",
        "pooling": ["camera_concat"], "dtype": "float32", "view_feature_dim": 512,
        "feature_dim": 3072, "resize_short_side": 256, "crop_size": 224,
        "interpolation": "PIL bilinear", "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "storage_aliases": {
            "layer1_mlp_output_mean.npy": "camera_concat",
            "layer1_mlp_output_last_token.npy": "camera_concat (identical compatibility alias)",
        },
        "model_revision": WEIGHTS_SHA256,
        "model_files_sha256": {"resnet18-f37072fd.pth": WEIGHTS_SHA256},
        "source_sha256": {name: sha256(SCRIPT.parent / name)
                          for name in ("cpu_image_encoder.py", "extract_cpu.py")},
        "fixture_scope": "SYNTHETIC ARRAY TEST ONLY; no pretrained inference or real judgments",
    })
    handoff.extract_dir = source.parent / "cpu-extract"
    handoff.extract_dir.mkdir()
    handoff.layer = 1
    handoff.feature_protocol = "cpu-resnet18"
    write_json(handoff.extract_dir / "extraction_config.json", contract)
    write_cpu_checkpoint(handoff, values, meta)
    return handoff


def test_cpu_handoff_is_explicit_read_only_and_has_distinct_feature_semantics(cpu_handoff):
    before = {str(path): path.read_bytes()
              for path in cpu_handoff.extract_dir.rglob("*") if path.is_file()}
    report = validate_workflow(cpu_handoff)
    assert report["feature_protocol"] == "cpu-resnet18"
    assert report["encoder"] == "resnet18-imagenet1k-v1"
    assert report["feature_dim"] == 3072 and report["layer"] == 1
    assert report["pooling"] == "camera_concat" and report["storage_pooling"] == "mean"
    assert "concatenated" in report["feature_semantics"]
    assert "separate from VLM" in report["feature_semantics"]
    assert before == {str(path): path.read_bytes()
                      for path in cpu_handoff.extract_dir.rglob("*") if path.is_file()}
    result = subprocess.run(cli_args(cpu_handoff), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["feature_protocol"] == "cpu-resnet18"


def test_cpu_and_vlm_protocols_cannot_be_selected_interchangeably(handoff, cpu_handoff):
    cpu_handoff.feature_protocol = "cosmos"
    with pytest.raises(ValueError, match="selected protocol"):
        validate_workflow(cpu_handoff)
    # Both fixture names share the original handoff object; its VLM cache is preserved.
    handoff.extract_dir = handoff.extract_dir.parent / "extract"
    handoff.feature_protocol = "cpu-resnet18"
    with pytest.raises(ValueError, match="selected protocol"):
        validate_workflow(handoff)


@pytest.mark.parametrize("key,value,message", [
    ("encoder", "different-image-encoder", "encoder mismatch"),
    ("model_revision", "b" * 64, "model_revision mismatch"),
    ("model_files_sha256", {"resnet18-f37072fd.pth": "b" * 64},
     "model_files_sha256 mismatch"),
    ("crop_size", 128, "crop_size mismatch"),
    ("pooling", ["mean_all_tokens"], "pooling mismatch"),
    ("source_sha256", {}, "source hashes"),
])
def test_cpu_contract_changes_fail_before_training(cpu_handoff, monkeypatch, key, value, message):
    rewrite(cpu_handoff.extract_dir / "extraction_config.json",
            lambda config: config.update({key: value}))
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs:
                        pytest.fail("Training started for invalid CPU handoff"))
    with pytest.raises(ValueError, match=message):
        train_workflow(cpu_handoff)
    assert not cpu_handoff.output_dir.exists()


@pytest.mark.parametrize("layer,pooling", [(28, "mean"), (1, "last_token")])
def test_cpu_storage_alias_cannot_be_misreported_as_vlm_layer_or_pooling(
        cpu_handoff, layer, pooling):
    cpu_handoff.layer, cpu_handoff.pooling = layer, pooling
    with pytest.raises(ValueError, match="require --layer 1 --pooling mean"):
        validate_workflow(cpu_handoff)


def test_cpu_array_width_must_match_six_full_image_embeddings(cpu_handoff):
    meta = json.loads((cpu_handoff.extract_dir / "meta.json").read_text())
    values = np.load(cpu_handoff.extract_dir / "layer1_mlp_output_mean.npy")[:, :-1]
    write_cpu_checkpoint(cpu_handoff, values, meta)
    with pytest.raises(ValueError, match="six concatenated 512-dimensional views"):
        validate_workflow(cpu_handoff)


def test_cpu_compatibility_arrays_must_be_identical(cpu_handoff):
    meta = json.loads((cpu_handoff.extract_dir / "meta.json").read_text())
    values = np.load(cpu_handoff.extract_dir / "layer1_mlp_output_mean.npy")
    last = values.copy()
    last[0, 0] += 1
    write_cpu_checkpoint(cpu_handoff, values, meta, last_values=last)
    with pytest.raises(ValueError, match="compatibility arrays must contain identical"):
        validate_workflow(cpu_handoff)


def test_cpu_protocol_does_not_bypass_manual_label_replay(cpu_handoff):
    rewrite(cpu_handoff.annotations_dir / "labels.json",
            lambda labels: labels[0].update(label="long_tail"))
    with pytest.raises(ValueError, match="labels.json differs from replay"):
        validate_workflow(cpu_handoff)
