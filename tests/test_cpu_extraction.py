"""CPU-extraction software fixtures; no real driving images or efficacy claims."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import cpu_image_encoder as encoder  # noqa: E402
import extract_cpu  # noqa: E402
from extraction_contract import sha256  # noqa: E402
from prepare_driving import CAMERAS, prepare  # noqa: E402
from test_prepare_driving import synthetic_nuscenes  # noqa: E402


class SyntheticEncoder(torch.nn.Module):
    """Cheap deterministic algebra to test orchestration, never pretrained inference."""

    def forward(self, images):
        return images.mean(dim=(1, 2, 3)).unsqueeze(1).repeat(1, encoder.VIEW_FEATURE_DIM)


@pytest.fixture
def source(tmp_path, monkeypatch):
    data = synthetic_nuscenes(tmp_path / "synthetic-data", frames=2)
    # Solid distinct cameras make the fixed concatenation order directly testable.
    for camera_index, camera in enumerate(CAMERAS):
        for path in (data / "samples" / camera).glob("*.png"):
            Image.new("RGB", (16, 24), (camera_index * 30, 45, 90)).save(path)
    prepared = tmp_path / "prepared"
    prepare(data, "v1.0-mini", prepared, frames_per_scene=2)
    weights = tmp_path / encoder.WEIGHTS_NAME
    weights.write_bytes(b"SYNTHETIC test stub; not actual pretrained model weights")
    monkeypatch.setattr(encoder, "WEIGHTS_SHA256", sha256(weights))
    monkeypatch.setattr(encoder, "load_encoder", lambda path: SyntheticEncoder())
    return SimpleNamespace(prepared_dir=prepared, data_root=data, weights=weights,
                           output_dir=tmp_path / "extract", max_samples=1,
                           save_every=1, threads=1, resume=True)


def test_cpu_resume_frozen_order_shapes_and_compatibility_aliases(source, monkeypatch):
    first = extract_cpu.extract(source)
    assert first["rows_in_cache"] == 1 and not first["selection_complete"]
    source.max_samples = 0
    complete = extract_cpu.extract(source)
    assert complete["selection_complete"] and complete["rows_in_cache"] == 20
    assert complete["device"] == "cpu" and complete["feature_protocol"] == encoder.SCHEMA
    config = json.loads((source.output_dir / "extraction_config.json").read_text())
    assert config["encoder"] == encoder.ENCODER and config["pooling"] == ["camera_concat"]
    assert config["model_revision"] == sha256(source.weights)
    assert config["model_files_sha256"] == {encoder.WEIGHTS_NAME: sha256(source.weights)}
    assert config["feature_dim"] == 3072 and config["layers"] == [1]
    expected = json.loads((source.prepared_dir / "selected_samples.json").read_text())
    actual = json.loads((source.output_dir / "meta.json").read_text())
    assert [row["sample_token"] for row in actual] == [row["sample_token"] for row in expected]
    values = np.load(source.output_dir / "layer1_mlp_output_mean.npy")
    assert values.shape == (20, 3072) and values.dtype == np.float32
    alias = np.load(source.output_dir / "layer1_mlp_output_last_token.npy")
    np.testing.assert_array_equal(values, alias)
    first_sample = expected[0]
    means = []
    for camera in CAMERAS:
        with Image.open(source.data_root / "samples" / first_sample["images"][camera]) as image:
            means.append(float(encoder.preprocess(image).mean()))
    np.testing.assert_allclose(values[0].reshape(6, 512)[:, 0], means, rtol=1e-6)
    monkeypatch.setattr(encoder, "load_encoder",
                        lambda path: pytest.fail("Complete resume loads model"))
    again = extract_cpu.extract(source)
    assert again["selection_complete"] and again["new_rows_this_invocation"] == 0


def test_image_tamper_rejected_before_cache_or_model(source, monkeypatch):
    row = json.loads((source.prepared_dir / "selected_samples.json").read_text())[0]
    path = source.data_root / "samples" / row["images"][CAMERAS[0]]
    path.write_bytes(b"tampered source image")
    monkeypatch.setattr(encoder, "load_encoder",
                        lambda path: pytest.fail("Model loaded before validation"))
    with pytest.raises(ValueError, match="Image changed after preparation"):
        extract_cpu.extract(source)
    assert not source.output_dir.exists()


def test_weights_tamper_and_unverified_weights_are_rejected(source):
    source.weights.write_bytes(b"different or unverified weights")
    with pytest.raises(ValueError, match="checkpoint SHA256 mismatch"):
        extract_cpu.extract(source)
    assert not source.output_dir.exists()


def test_changed_preprocessing_source_rejects_resume(source, monkeypatch):
    extract_cpu.extract(source)
    monkeypatch.setattr(encoder, "CROP_SIZE", 192)
    with pytest.raises(ValueError, match="Extraction contract changed"):
        extract_cpu.extract(source)


def test_partial_failure_keeps_committed_rows_and_resumes(source, monkeypatch):
    source.max_samples = 0
    actual = encoder.encode_views
    calls = 0

    def fail_second(model, paths):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic simulated process failure")
        return actual(model, paths)

    monkeypatch.setattr(encoder, "encode_views", fail_second)
    with pytest.raises(RuntimeError, match="simulated process failure"):
        extract_cpu.extract(source)
    failed = json.loads((source.output_dir / "extraction_summary.json").read_text())
    assert failed["status"] == "failed_or_interrupted" and failed["rows_in_cache"] == 1
    monkeypatch.setattr(encoder, "encode_views", actual)
    assert extract_cpu.extract(source)["selection_complete"]


def test_pretrained_architecture_parameters_match_standard_resnet18():
    model = encoder.ResNet18()
    state = model.state_dict()
    assert sum(parameter.numel() for parameter in model.parameters()) == 11689512
    assert state["conv1.weight"].shape == (64, 3, 7, 7)
    assert state["layer4.0.downsample.0.weight"].shape == (512, 256, 1, 1)
    assert state["fc.weight"].shape == (1000, 512)
    # This validates architecture layout only; no randomly initialized feature is extracted.


def test_cpu_cli_defaults_do_not_accept_gpu_or_another_encoder(tmp_path):
    args = extract_cpu.parse_args(["--prepared-dir", str(tmp_path), "--data-root", str(tmp_path),
                                   "--weights", str(tmp_path / encoder.WEIGHTS_NAME),
                                   "--output-dir", str(tmp_path / "extract")])
    assert args.threads == 2 and args.max_samples == 0 and args.resume
    with pytest.raises(SystemExit):
        extract_cpu.parse_args(["--device", "cuda"])
