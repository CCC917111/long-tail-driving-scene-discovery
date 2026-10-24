"""Optional authorised-data integration check; synthetic CI cannot establish it."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from data_manifest import load_frozen_manifest  # noqa: E402
from sae_common import load_dataset  # noqa: E402


def test_authorised_feature_bundle(tmp_path):
    location = os.environ.get("DRIVING_REAL_ASSET_DIR")
    if not location:
        pytest.skip("Real assets unavailable: set DRIVING_REAL_ASSET_DIR to an authorised "
                    "bundle with features.npy, meta.json, labels.json and manifest.json")
    root = Path(location)
    args = SimpleNamespace(features=root / "features.npy", meta=root / "meta.json",
                           labels=[root / "labels.json"], output_dir=tmp_path)
    features, labels, groups, meta = load_dataset(args)
    splits = load_frozen_manifest(root / "manifest.json", meta, labels)
    assert len(features) == len(labels) == len(groups) == len(meta)
    for indices in splits.values():
        assert set(labels[indices].tolist()) == {0, 1}
