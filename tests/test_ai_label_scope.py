"""AI-derived references cannot silently become human-ground-truth results."""

import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_runs import label_scope_from_manifest, write_evaluation_artifacts  # noqa: E402
from evaluation_fixture import create_fixture  # noqa: E402


def test_ai_scope_propagates_to_metrics_and_run_record(tmp_path):
    manifest = create_fixture(tmp_path / "fixture")
    rows = json.loads((tmp_path / "fixture/rows.json").read_text())
    for row in manifest:
        row.update(label_origin="ai_visual_provisional", review_status="ai_provisional",
                   human_review_status="pending")
    result = write_evaluation_artifacts(tmp_path / "run", rows, manifest, config={}, seed=42,
                                        review_k=4, bootstrap_samples=0)
    scope = result["label_scope"]
    assert scope["label_protocol"] == "ai-visual-pilot"
    assert scope["human_reviewed"] is False
    metadata = json.loads((tmp_path / "run/run_metadata.json").read_text())
    assert metadata["label_scope"] == scope


def test_ai_scope_rejects_mixed_or_misrepresented_human_status():
    row = {"split": "train", "label_origin": "ai_visual_provisional",
           "review_status": "ai_provisional", "human_review_status": "pending"}
    for changed in ({"human_review_status": "reviewed"}, {"review_status": "reviewed"}):
        invalid = copy.deepcopy(row)
        invalid.update(changed)
        with pytest.raises(ValueError, match="AI provisional"):
            label_scope_from_manifest([invalid])
    with pytest.raises(ValueError, match="AI provisional"):
        label_scope_from_manifest([row, {"split": "test", "review_status": "reviewed"}])


def test_existing_reference_protocol_keeps_its_original_scope():
    assert label_scope_from_manifest([{"split": "train", "review_status": "reviewed"}]) is None
