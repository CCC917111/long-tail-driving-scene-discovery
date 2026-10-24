"""Regression checks for the trained-model screening -> browser boundary."""

import csv
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from neuron_explorer import build_explorer, parse_args  # noqa: E402
import screen  # noqa: E402
from screen import build_model, validate_feature_compatibility, validate_inputs  # noqa: E402


def test_screening_override_survives_browser_reload(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    config = dict(hidden_dim=2, tail_ratio=.5, dropout=0., k=1,
                  sparsity="abstopk", reward="norm")
    model = build_model(config, 2, "cpu")
    # A fixed analytic fixture: .03 is active at .01 but quiet at .05.
    with torch.no_grad():
        model.encoder.weight.zero_()
        model.encoder.bias.copy_(torch.tensor([0., .03]))
    torch.save(model.state_dict(), run / "best_model.pth")
    np.savez(run / "standardizer.npz", mean=np.zeros((1, 2)), std=np.ones((1, 2)))
    (run / "metrics.json").write_text(json.dumps({"config": config,
                                                 "decision": {"eta": .01}}))
    features = tmp_path / "features.npy"
    np.save(features, np.zeros((2, 2), np.float32))
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps([{"sample_token": "a"}, {"sample_token": "b"}]))
    output = tmp_path / "screening"
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/screen.py"),
               "--run-dir", str(run), "--features", str(features), "--meta", str(meta),
               "--output-dir", str(output), "--eta", ".05", "--device", "cpu"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    explorer = build_explorer(parse_args(["units", "--run", str(output)]))
    assert explorer.eta == .05
    with (output / "screening.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert all(row["prediction"] == explorer.units_of_frame(row["sample_token"])["prediction"]
               == "normal" for row in rows)
    exported = list(csv.DictReader(explorer.review_csv().decode().splitlines()))
    assert {row["sample_token"] for row in exported} == {"a", "b"}
    assert all(row["reviewer_label"] == row["review_notes"] == "" for row in exported)
    assert "label" not in exported[0]
    metadata = json.loads((output / "screening_metadata.json").read_text())
    assert metadata["feature_protocol_validation"]["status"] == "unverified"


@pytest.mark.parametrize("failure", ["duplicate", "missing", "nan", "empty",
                                     "wrong_dimension", "zero_std", "nan_std"])
def test_invalid_screening_inputs_fail_before_inference(failure):
    x = np.zeros((2, 3), np.float32)
    meta = [{"sample_token": "a"}, {"sample_token": "b"}]
    mean, std = np.zeros((1, 3)), np.ones((1, 3))
    if failure == "duplicate":
        meta[1]["sample_token"] = "a"
    elif failure == "missing":
        meta[1] = {}
    elif failure == "nan":
        x[0, 0] = np.nan
    elif failure == "empty":
        x, meta = x[:0], []
    elif failure == "wrong_dimension":
        std = np.ones((1, 4))
    elif failure == "zero_std":
        std[0, 0] = 0
    else:
        std[0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_inputs(x, meta, mean, std)


@pytest.fixture
def cpu_feature_contract(tmp_path):
    """Recorded configuration fixture, without downloaded weights or learned scores."""
    from cpu_image_encoder import WEIGHTS_SHA256
    from prepare_driving import CAMERAS

    provenance = {
        "schema": "cpu-resnet18-six-camera-v1", "encoder": "resnet18-imagenet1k-v1",
        "model_revision": WEIGHTS_SHA256,
        "model_files_sha256": {"resnet18-f37072fd.pth": WEIGHTS_SHA256},
        "camera_order": list(CAMERAS), "layers": [1],
        "hook": "resnet18 avgpool; six-camera concatenation", "pooling": ["camera_concat"],
        "storage_aliases": {"layer1_mlp_output_mean.npy": "camera_concat",
                            "layer1_mlp_output_last_token.npy":
                            "camera_concat (identical compatibility alias)"},
        "dtype": "float32", "resize_short_side": 256, "crop_size": 224,
        "interpolation": "PIL bilinear", "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225], "view_feature_dim": 512, "feature_dim": 3072,
        "samples": [{"sample_token": "synthetic-training-only"}],
    }
    run, new = tmp_path / "run", tmp_path / "new"
    run.mkdir()
    new.mkdir()
    (run / "run_metadata.json").write_text(json.dumps({
        "config": {"feature_provenance": provenance}}))
    candidate = dict(provenance, samples=[{"sample_token": "synthetic-unlabelled-new"}],
                     metadata_sha256={"scene.json": "different-new-dataset"})
    path = new / "extraction_config.json"
    path.write_text(json.dumps(candidate))
    features = new / "layer1_mlp_output_mean.npy"
    np.save(features, np.zeros((2, 3072), np.float32))
    return SimpleNamespace(run=run, features=features, config=path, candidate=candidate)


def test_cpu_screening_matches_protocol_without_requiring_training_sample_ids(cpu_feature_contract):
    data = cpu_feature_contract
    report = validate_feature_compatibility(data.run, data.features)
    assert report["status"] == "matched"
    assert report["encoder"] == "resnet18-imagenet1k-v1"
    assert report["config_sha256"] == screen.file_sha256(data.config)
    assert report["sample_identity_compared"] is False
    assert report["input_array_origin_authenticated"] is False
    explicit = data.config.parent / "explicit.json"
    data.config.rename(explicit)
    assert validate_feature_compatibility(data.run, data.features, explicit)["status"] == "matched"


@pytest.mark.parametrize("key,value", [
    ("encoder", "other-same-dimension-encoder"),
    ("model_revision", "b" * 64),
    ("crop_size", 128),
    ("camera_order", ["CAM_FRONT"] * 6),
])
def test_cpu_screening_rejects_same_dimension_wrong_feature_protocol_before_model(
        cpu_feature_contract, monkeypatch, key, value):
    data = cpu_feature_contract
    data.candidate[key] = value
    data.config.write_text(json.dumps(data.candidate))
    # The dimensions still match. The representation mismatch must stop scoring.
    assert np.load(data.features).shape == (2, 3072)
    (data.run / "metrics.json").write_text(json.dumps({"config": {}, "decision": {"eta": .01}}))
    output = data.features.parent / "screening"
    args = SimpleNamespace(run_dir=data.run, features=data.features, meta=None, output_dir=output,
                           feature_config=None, batch_size=64, top_neurons=5)
    monkeypatch.setattr(screen, "parse_args", lambda: args)
    monkeypatch.setattr(screen, "build_model", lambda *args, **kwargs:
                        pytest.fail("Model constructed for mismatched CPU features"))
    monkeypatch.setattr(screen.torch, "load", lambda *args, **kwargs:
                        pytest.fail("Model weights loaded for mismatched CPU features"))
    monkeypatch.setattr(screen, "encode", lambda *args, **kwargs:
                        pytest.fail("Scoring started for mismatched CPU features"))
    with pytest.raises(ValueError, match=f"feature protocol mismatch: {key}"):
        screen.main()
    assert not output.exists()


def test_cpu_screening_requires_new_feature_contract(cpu_feature_contract):
    data = cpu_feature_contract
    data.config.unlink()
    with pytest.raises(ValueError, match="requires --feature-config"):
        validate_feature_compatibility(data.run, data.features)
