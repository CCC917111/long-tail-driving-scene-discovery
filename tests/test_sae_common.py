#!/usr/bin/env python3
"""Unit tests for the shared SAE library in ``scripts/sae_common.py``.

Run from the repository root:

    python -m pytest tests/test_sae_common.py
    python tests/test_sae_common.py      # same tests, without pytest

The tests cover the parts that are easy to break silently — the sparsity rule,
the scene-level split, the label protocol and the threshold logic — on small
synthetic arrays, so they run on CPU in seconds and need neither the dataset
nor a VLM.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from sae_common import (  # noqa: E402
    LongTailGuidedSAE,
    best_f1_threshold,
    classification_metrics,
    load_label_map,
    split_indices,
)


# --- Sparsity ---------------------------------------------------------------

def test_abstopk_keeps_exactly_k_units():
    model = LongTailGuidedSAE(input_dim=16, hidden_dim=32, k=8, sparsity="abstopk")
    model.eval()
    z, z_n, z_t = model.encode(torch.randn(4, 16))
    assert z.shape == (4, 32), z.shape
    assert z_n.shape[1] + z_t.shape[1] == 32
    assert ((z != 0).sum(dim=1) <= 8).all(), (z != 0).sum(dim=1)


def test_abstopk_keeps_large_negative_activations():
    """The point of AbsTopK: a strongly negative unit is informative too."""
    model = LongTailGuidedSAE(input_dim=4, hidden_dim=8, k=2, sparsity="abstopk")
    model.eval()
    with torch.no_grad():
        model.encoder.weight.zero_()
        model.encoder.bias.copy_(torch.tensor([-9.0, 5.0, 0.1, 0.0,
                                               0.0, 0.0, 0.0, 0.0]))
    z, _, _ = model.encode(torch.zeros(1, 4))
    kept = torch.nonzero(z[0]).flatten().tolist()
    assert kept == [0, 1], kept

    signed = LongTailGuidedSAE(input_dim=4, hidden_dim=8, k=2, sparsity="topk")
    signed.eval()
    with torch.no_grad():
        signed.encoder.weight.zero_()
        signed.encoder.bias.copy_(model.encoder.bias)
    z_signed, _, _ = signed.encode(torch.zeros(1, 4))
    kept_signed = torch.nonzero(z_signed[0]).flatten().tolist()
    assert 0 not in kept_signed, kept_signed


def test_k_larger_than_hidden_dim_is_rejected():
    try:
        LongTailGuidedSAE(input_dim=4, hidden_dim=8, k=16)
    except ValueError:
        return
    raise AssertionError("k > hidden_dim must be rejected")


# --- Objective --------------------------------------------------------------

def test_tail_reward_lowers_the_loss_for_active_tail_samples():
    """An activated z_t must be cheaper for a tail sample than a quiet one."""
    torch.manual_seed(0)
    model = LongTailGuidedSAE(input_dim=8, hidden_dim=16, k=4, sparsity="abstopk",
                              reward="norm", beta=0.0)
    model.eval()
    x = torch.randn(6, 8)
    is_tail = torch.ones(6)
    with torch.no_grad():
        _, parts, _, _, _ = model(x, is_tail)
    assert parts["tail_reward"].item() <= 0.0, parts["tail_reward"].item()

    quiet = LongTailGuidedSAE(input_dim=8, hidden_dim=16, k=4, sparsity="abstopk",
                              reward="none", beta=0.0)
    quiet.eval()
    quiet.load_state_dict(model.state_dict())
    with torch.no_grad():
        _, quiet_parts, _, _, _ = quiet(x, is_tail)
    assert quiet_parts["tail_reward"].item() == 0.0


def test_normal_penalty_only_applies_to_normal_samples():
    torch.manual_seed(0)
    model = LongTailGuidedSAE(input_dim=8, hidden_dim=16, k=4, beta=1.0,
                              reward="none")
    model.eval()
    x = torch.randn(8, 8)
    with torch.no_grad():
        _, all_tail, _, _, _ = model(x, torch.ones(8))
        _, all_normal, _, _, _ = model(x, torch.zeros(8))
    assert all_tail["normal"].item() == 0.0
    assert all_normal["normal"].item() > 0.0


# --- Split ------------------------------------------------------------------

def test_scene_level_split_keeps_scenes_together():
    """No scene may appear in two splits; that is what prevents leakage."""
    rng = np.random.default_rng(0)
    groups = np.repeat([f"scene-{i:03d}" for i in range(40)], 10)
    labels = (rng.random(len(groups)) < 0.3).astype(np.float32)
    splits = split_indices(labels, groups, val_ratio=0.2, test_ratio=0.2, seed=7)

    seen: dict[str, str] = {}
    for name, idx in splits.items():
        for scene in groups[idx]:
            assert seen.setdefault(scene, name) == name, (
                f"scene {scene} appears in both {seen[scene]} and {name}")

    covered = sum(len(idx) for idx in splits.values())
    assert covered == len(labels), (covered, len(labels))


def test_split_is_stratified_over_long_tail_samples():
    rng = np.random.default_rng(1)
    groups = np.repeat([f"scene-{i:03d}" for i in range(60)], 8)
    labels = (rng.random(len(groups)) < 0.25).astype(np.float32)
    splits = split_indices(labels, groups, val_ratio=0.2, test_ratio=0.2, seed=3)
    for name in ("train", "val", "test"):
        share = labels[splits[name]].mean()
        assert 0.1 < share < 0.45, (name, share)


# --- Labels -----------------------------------------------------------------

def test_label_protocol():
    """normal maps to 0, uncertain is dropped, anything else is long-tail."""
    import json
    import tempfile
    from pathlib import Path

    rows = [
        {"sample_token": "a", "label": "normal_core"},
        {"sample_token": "b", "label": "not_normal_core"},
        {"sample_token": "c", "label": "uncertain"},
        {"sample_token": "d", "labels": {"label": "long_tail"}},
        {"sample_token": "e", "label": ""},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "labels.json"
        path.write_text(json.dumps(rows), encoding="utf-8")
        mapping = load_label_map([path])

    assert mapping["a"] == 0
    assert mapping["b"] == 1
    assert mapping["c"] is None, "uncertain must be dropped, not counted as tail"
    assert mapping["d"] == 1, "the {'labels': {'label': ...}} form must be read"
    assert mapping["e"] is None


# --- Thresholding -----------------------------------------------------------

def test_threshold_is_chosen_where_f1_peaks():
    y = np.array([0, 0, 0, 1, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    threshold = best_f1_threshold(y, score)
    assert 0.3 < threshold <= 0.7, threshold

    metrics = classification_metrics(y, score, threshold)
    assert metrics["long_tail"]["f1"] == 1.0, metrics
    assert metrics["auc"] == 1.0
    assert metrics["long_tail"]["support"] == 3


def test_metrics_report_both_classes():
    y = np.array([0, 1, 0, 1])
    score = np.array([0.1, 0.9, 0.8, 0.2])
    metrics = classification_metrics(y, score, 0.5)
    assert set(metrics) == {"auc", "ap", "threshold", "long_tail", "normal"}
    assert metrics["normal"]["support"] == 2


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and isinstance(value, types.FunctionType)]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} tests passed")
