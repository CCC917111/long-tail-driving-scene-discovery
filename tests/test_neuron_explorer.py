#!/usr/bin/env python3
"""Tests for the decision rule, the activation store and the neuron explorer.

These modules need only numpy and Pillow, so the tests run without torch:

    python -m pytest tests/test_neuron_explorer.py
    python tests/test_neuron_explorer.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import types
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from neuron_explorer import Explorer, make_server  # noqa: E402
from tail_activations import (  # noqa: E402
    CAMERAS,
    ActivationStore,
    predict_long_tail,
    save_activations,
    tail_activation_count,
    tail_score,
)


def _z_t() -> np.ndarray:
    """Four frames, five long-tail units; frame 2 has no active unit."""
    z = np.zeros((4, 5), dtype=np.float32)
    z[0, 1] = 2.5           # strong on unit 1
    z[0, 3] = -0.4          # weak, negative
    z[1, 1] = 1.2
    z[1, 4] = 0.005         # below eta
    z[3, 1] = -3.0          # strongest on unit 1, negative sign
    z[3, 2] = 0.02
    return z


def _meta(n: int) -> list[dict]:
    return [{"sample_token": f"tok{i}", "scene_name": f"scene-{i:04d}",
             "images": {cam: f"{cam}/frame{i}.jpg" for cam in CAMERAS}} for i in range(n)]


# --- Decision rule ----------------------------------------------------------

def test_count_rule_flags_any_active_unit():
    z = _z_t()
    assert tail_activation_count(z, 0.01).tolist() == [2, 1, 0, 2]
    assert predict_long_tail(z, 0.01).tolist() == [1, 1, 0, 1]


def test_negative_activations_count_as_active():
    z = np.array([[0.0, -0.5, 0.0]])
    assert predict_long_tail(z, 0.01).tolist() == [1]


def test_eta_is_a_strict_threshold():
    z = np.array([[0.01, 0.0], [0.0100001, 0.0]])
    assert predict_long_tail(z, 0.01).tolist() == [0, 1]


def test_tail_score_is_the_l2_norm():
    z = np.array([[3.0, 4.0], [0.0, 0.0]])
    assert np.allclose(tail_score(z), [5.0, 0.0])


# --- Activation store -------------------------------------------------------

def test_store_round_trip_is_exact():
    z = _z_t()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "activations.npz"
        save_activations(path, z, _meta(4), labels=np.array([1, 1, 0, 1]),
                         splits=["train", "val", "test", "train"])
        store = ActivationStore(Path(tmp))
    assert len(store) == 4 and store.tail_dim == 5
    assert np.array_equal(store.dense(), z)
    assert store.counts(0.01).tolist() == [2, 1, 0, 2]
    assert store.labels.tolist() == [1, 1, 0, 1]
    assert store.splits[2] == "test"
    rows, values = store.column(1)
    assert sorted(rows.tolist()) == [0, 1, 3]
    assert store.samples[3]["images"]["CAM_BACK"] == "CAM_BACK/frame3.jpg"


def test_store_counts_handle_empty_trailing_rows():
    z = np.zeros((3, 4), dtype=np.float32)
    z[0, 0] = 1.0
    with tempfile.TemporaryDirectory() as tmp:
        save_activations(Path(tmp) / "activations.npz", z, _meta(3))
        store = ActivationStore(Path(tmp))
    assert store.counts(0.01).tolist() == [1, 0, 0]


# --- Explorer ---------------------------------------------------------------

def _explorer(tmp: Path, with_images: bool = True) -> Explorer:
    save_activations(tmp / "activations.npz", _z_t(), _meta(4),
                     labels=np.array([1, 1, 0, 0]))
    root = tmp / "samples"
    if with_images:
        for i in range(4):
            for cam in CAMERAS:
                (root / cam).mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (160, 90), (40 * i, 80, 120)).save(root / cam / f"frame{i}.jpg")
    return Explorer(ActivationStore(tmp), root if with_images else None,
                    glossary={1: "Wheelchair user ahead"}, eta=0.01, neuron_threshold=1.0)


def test_unit_to_frames_is_ordered_by_magnitude():
    with tempfile.TemporaryDirectory() as tmp:
        ex = _explorer(Path(tmp))
        frames = ex.frames_of_unit(1, top=10)
    assert [f["sample_token"] for f in frames] == ["tok3", "tok0", "tok1"]
    assert frames[0]["activation"] == -3.0
    assert frames[0]["label"] == "normal" and frames[1]["label"] == "long_tail"


def test_unit_stats_and_overview_report_purity():
    with tempfile.TemporaryDirectory() as tmp:
        ex = _explorer(Path(tmp))
        stats = ex.unit_stats(1)
        overview = ex.units(top=10)
    assert stats["frames"] == 3 and stats["long_tail_frames"] == 2
    assert abs(stats["purity"] - 2 / 3) < 1e-9
    assert overview[0]["unit"] == 1 and overview[0]["name"] == "Wheelchair user ahead"


def test_frame_to_units_lists_active_units_strongest_first():
    with tempfile.TemporaryDirectory() as tmp:
        ex = _explorer(Path(tmp))
        frame = ex.units_of_frame("tok0")
        quiet = ex.units_of_frame("tok2")
    assert [u["unit"] for u in frame["units"]] == [1, 3]
    assert frame["units"][0]["strong"] and not frame["units"][1]["strong"]
    assert frame["prediction"] == "long_tail"
    assert quiet["units"] == [] and quiet["prediction"] == "normal"


def test_unknown_token_and_out_of_range_unit_are_errors():
    with tempfile.TemporaryDirectory() as tmp:
        ex = _explorer(Path(tmp))
        for call in (lambda: ex.units_of_frame("nope"), lambda: ex.frames_of_unit(99)):
            try:
                call()
            except (KeyError, IndexError):
                continue
            raise AssertionError("expected an error")


def test_image_paths_cannot_escape_the_samples_root():
    with tempfile.TemporaryDirectory() as tmp:
        ex = _explorer(Path(tmp))
        assert ex.image_path("tok0", "CAM_FRONT").name == "frame0.jpg"
        ex.store.samples[0]["images"]["CAM_FRONT"] = "../../etc/passwd"
        assert ex.image_path("tok0", "CAM_FRONT") is None
        assert ex.image_path("tok0", "NOT_A_CAMERA") is None


def test_http_interface_serves_both_directions():
    with tempfile.TemporaryDirectory() as tmp:
        ex = _explorer(Path(tmp))
        server = make_server(ex, "127.0.0.1", 0)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            def get(path):
                with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}") as r:
                    return r.read(), r.headers.get("Content-Type")
            body, ctype = get("/")
            assert ctype.startswith("text/html") and b"Neuron Explorer" in body
            unit = json.loads(get("/api/unit/1?top=5")[0])
            assert unit["n_frames"] == 3 and unit["frames"][0]["sample_token"] == "tok3"
            frame = json.loads(get("/api/sample/tok0")[0])
            assert frame["units"][0]["name"] == "Wheelchair user ahead"
            image, ctype = get("/img/tok0/CAM_FRONT?w=120")
            assert ctype == "image/jpeg" and image[:2] == b"\xff\xd8"
            try:
                get("/api/sample/unknown")
            except urllib.error.HTTPError as err:
                assert err.code == 404
            else:
                raise AssertionError("expected 404")
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and isinstance(value, types.FunctionType)]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} tests passed")
