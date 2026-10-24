"""Extraction orchestration fixtures; no real VLM inference or data-quality claim."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import extract  # noqa: E402
from extraction_contract import build_contract, freeze_contract  # noqa: E402
from driving_preflight import main as preflight_main  # noqa: E402


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "data"
    meta = root / "v1.0-mini"
    meta.mkdir(parents=True)
    rows, samples = [], []
    for i in range(2):
        samples.append({"token": f"s{i}", "scene_token": "scene0", "timestamp": i})
        for camera_index, camera in enumerate(extract.CAMERAS):
            filename = f"samples/{camera}/{i}.jpg"
            image = root / filename
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), (camera_index * 30, 40, 50)).save(image)
            rows.append({"token": f"image-{i}-{camera}", "sample_token": f"s{i}",
                         "is_key_frame": True, "filename": filename})
    tables = (("sample.json", samples), ("sample_data.json", rows),
              ("scene.json", [{"token": "scene0", "name": "fixture-scene"}]))
    for name, value in tables:
        (meta / name).write_text(json.dumps(value))
    model = tmp_path / "fake-local-model"
    model.mkdir()
    (model / "config.json").write_text('{"fixture": true}')
    (model / "model.safetensors").write_bytes(b"fixture-only-not-a-real-model")
    args = SimpleNamespace(samples_root=root / "samples", meta_dir=meta,
                           model_dir=model, model_revision="a" * 40,
                           output_dir=tmp_path / "output", layers=[1, 2],
                           prompt="fixture scene description", sample_tokens=None,
                           caption=False, max_new_tokens=10, device="cpu", dtype="float16",
                           min_pixels=16, max_pixels=64, max_samples=1,
                           save_every=1, resume=True)
    return args


class FixtureLayer(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.mlp = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.mlp.weight.copy_(torch.eye(2) * scale)


class FixtureModel(torch.nn.Module):
    """Fixed algebra for testing hooks and pooling, not a pretrained VLM."""
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([FixtureLayer(1), FixtureLayer(2)])

    def forward(self, input_ids):
        value = torch.stack([input_ids, input_ids * 2], dim=-1).float()
        for layer in self.model.layers:
            value = layer.mlp(value)
        return value


class FixtureProcessor:
    def apply_chat_template(self, messages, **kwargs):
        assert len(messages[0]["content"]) == 7
        return "fixture rendered prompt"

    def __call__(self, *, text, images, **kwargs):
        assert text == ["fixture rendered prompt"]
        assert len(images) == 6
        return {"input_ids": torch.tensor([[1, 2, 4]])}


def test_extraction_real_control_flow_hooks_pooling_and_resume(source, monkeypatch):
    monkeypatch.setattr(extract, "parse_args", lambda: source)
    monkeypatch.setattr(extract, "load_vlm", lambda *args: (FixtureProcessor(), FixtureModel()))
    extract.main()
    summary = json.loads((source.output_dir / "extraction_summary.json").read_text())
    assert summary["new_rows_this_invocation"] == 1 and not summary["selection_complete"]
    source.max_samples = 0
    extract.main()
    metadata = json.loads((source.output_dir / "meta.json").read_text())
    assert [row["sample_token"] for row in metadata] == ["s0", "s1"]
    assert all(row["scene_token"] == "scene0" and len(row["images"]) == 6 for row in metadata)
    np.testing.assert_allclose(np.load(source.output_dir / "layer1_mlp_output_mean.npy"),
                               [[7 / 3, 14 / 3]] * 2)
    np.testing.assert_allclose(np.load(source.output_dir / "layer2_mlp_output_last_token.npy"),
                               [[8, 16]] * 2)
    summary = json.loads((source.output_dir / "extraction_summary.json").read_text())
    assert summary["selection_complete"] and summary["new_rows_this_invocation"] == 1
    source.prompt = "changed prompt"
    with pytest.raises(ValueError, match="contract changed"):
        extract.main()


def test_model_load_failure_is_recorded(source, monkeypatch):
    def fail(*args):
        raise RuntimeError("fixture model load failed")
    monkeypatch.setattr(extract, "parse_args", lambda: source)
    monkeypatch.setattr(extract, "load_vlm", fail)
    with pytest.raises(RuntimeError, match="fixture model load failed"):
        extract.main()
    report = json.loads((source.output_dir / "extraction_summary.json").read_text())
    assert report["status"] == "failed" and report["phase"] == "model_load"
    assert not report["selection_complete"]


def test_unpinned_model_and_legacy_cache_are_rejected(source):
    samples = extract.build_samples(source.meta_dir, source.samples_root)
    source.model_revision = "main"
    with pytest.raises(ValueError, match="immutable"):
        build_contract(source, samples, source.meta_dir, extract.CAMERAS)
    source.model_revision = "a" * 40
    contract = build_contract(source, samples, source.meta_dir, extract.CAMERAS)
    source.output_dir.mkdir()
    (source.output_dir / "meta.json").write_text('[]')
    with pytest.raises(ValueError, match="Legacy extraction"):
        freeze_contract(source.output_dir, contract, True)


def test_preflight_missing_data_is_actionable_and_strict(tmp_path):
    output = tmp_path / "preflight.json"
    result = preflight_main(["--data-root", str(tmp_path / "absent"),
                             "--model-dir", str(tmp_path / "absent-model"),
                             "--output", str(output), "--require", "data"])
    assert result == 2
    report = json.loads(output.read_text())
    assert report["data"]["metadata_files_missing"]
    assert not report["data"]["verified_real_data"]
    assert report["no_download_or_compute_purchase_performed"]


def test_committed_state_without_contract_is_not_relabelled(source):
    samples = extract.build_samples(source.meta_dir, source.samples_root)
    contract = build_contract(source, samples, source.meta_dir, extract.CAMERAS)
    source.output_dir.mkdir()
    (source.output_dir / "checkpoint_state.json").write_text('{"generation": "fixture"}')
    with pytest.raises(ValueError, match="Legacy extraction"):
        freeze_contract(source.output_dir, contract, True)
    assert not (source.output_dir / "extraction_config.json").exists()


def test_preflight_detects_missing_model_shard(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{}')
    (model / "model-00001-of-00002.safetensors").write_bytes(b"fixture-only")
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "first": "model-00001-of-00002.safetensors",
        "second": "model-00002-of-00002.safetensors"}}))
    output = tmp_path / "preflight.json"
    preflight_main(["--data-root", str(tmp_path / "absent"), "--model-dir", str(model),
                    "--output", str(output)])
    report = json.loads(output.read_text())
    assert not report["model"]["files_present"]
    assert report["model"]["missing_weight_shards"] == ["model-00002-of-00002.safetensors"]
