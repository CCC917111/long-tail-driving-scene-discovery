"""CPU contracts for frozen evaluation, leakage prevention and cheap controls."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_runs import (  # noqa: E402
    align_rows,
    file_sha256,
    review_budget_metrics,
    scene_bootstrap,
    select_operating_point,
    write_evaluation_artifacts,
)
from evaluation_fixture import create_fixture  # noqa: E402
from supervised_baselines import fit_baseline  # noqa: E402
from summarize_runs import collect_runs  # noqa: E402


def fixture_rows(tmp_path):
    manifest = create_fixture(tmp_path)
    return json.loads((tmp_path / "rows.json").read_text()), manifest


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_alignment_joins_by_token_and_rejects_missing_duplicate_or_changed_truth(self):
        tmp_path = self.tmp_path
        rows, manifest = fixture_rows(tmp_path)
        assert align_rows(rows[::-1], manifest)[0]["sample_token"] == manifest[0]["sample_token"]
        with self.assertRaisesRegex(ValueError, "Missing predictions"):
            align_rows(rows[1:], manifest)
        with self.assertRaisesRegex(ValueError, "Duplicate prediction"):
            align_rows(rows + rows[:1], manifest)
        changed = copy.deepcopy(rows)
        changed[0]["label"] = 1 - changed[0]["label"]
        with self.assertRaisesRegex(ValueError, "label disagrees"):
            align_rows(changed, manifest)

    def test_alignment_rejects_cross_split_scene_leakage_and_nonfinite_scores(self):
        tmp_path = self.tmp_path
        rows, manifest = fixture_rows(tmp_path)
        changed = copy.deepcopy(manifest)
        changed[-1]["scene_token"] = changed[0]["scene_token"]
        with self.assertRaisesRegex(ValueError, "overlaps"):
            align_rows(rows, changed)
        rows[0]["score"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            align_rows(rows, manifest)

    def test_unreachable_precision_target_is_predict_none_not_test_tuned(self):
        point = select_operating_point([1, 0], [0.1, 0.9], target_precision=0.9)
        assert point["status"] == "target_unreachable"
        assert point["threshold"] is None
        # A tied score is indivisible: never claim only the positive would be reviewed.
        tied = select_operating_point([1, 0], [0.5, 0.5], target_precision=0.9)
        assert tied["status"] == "target_unreachable"

    def test_operating_point_and_test_predictions_do_not_depend_on_test_labels(self):
        tmp_path = self.tmp_path
        rows, manifest = fixture_rows(tmp_path / "data")
        first = write_evaluation_artifacts(tmp_path / "first", rows, manifest, config={}, seed=7,
                                           review_k=3, bootstrap_samples=0)
        changed_rows, changed_manifest = copy.deepcopy(rows), copy.deepcopy(manifest)
        for item in changed_rows + changed_manifest:
            if item["split"] == "test":
                item["label"] = 1 - item["label"]
        second = write_evaluation_artifacts(tmp_path / "second", changed_rows, changed_manifest,
                                            config={}, seed=7, review_k=3, bootstrap_samples=0)
        assert first["operating_point"] == second["operating_point"]
        first_rows = json.loads((tmp_path / "first/predictions.json").read_text())
        second_rows = json.loads((tmp_path / "second/predictions.json").read_text())
        assert [r["operating_prediction"] for r in first_rows] == [
            r["operating_prediction"] for r in second_rows]
        assert first["splits"]["test"]["operating"]["ap"] != second[
            "splits"]["test"]["operating"]["ap"]

    def test_review_budget_handles_ties_and_reports_analytic_random_expectation(self):
        rows = [{"sample_token": "b", "score": 0.8, "label": 0},
                {"sample_token": "a", "score": 0.8, "label": 1},
                {"sample_token": "c", "score": 0.1, "label": 1}]
        result = review_budget_metrics(rows, 1)
        assert result["selected_tokens"] == ["a"]
        assert result["precision_at_k"] == 1
        assert result["random_expected_tail_found"] == 2 / 3
        assert review_budget_metrics(rows, 10)["effective_k"] == 3

    def test_bootstrap_resamples_whole_scenes_and_reports_undefined_draws(self):
        # Pure positive/negative scenes make single-class draws possible only when
        # complete scenes are resampled together. Frame bootstrap would hide this.
        rows = [{"scene_token": f"scene-{label}", "label": label,
                 "operating_prediction": label, "score": float(label)}
                for label in (0, 1) for _ in range(20)]
        result = scene_bootstrap(rows, seed=3, samples=100)
        assert result == scene_bootstrap(rows, seed=3, samples=100)
        assert result["unit"] == "scene"
        assert result["threshold_refitted"] is False
        assert result["intervals"]["auc"]["undefined_samples"] > 0
        assert result["intervals"]["auc"]["valid_samples"] < 100

    def test_result_bundle_records_inputs_checkpoint_and_deliberate_failures(self):
        tmp_path = self.tmp_path
        data = tmp_path / "data"
        rows, manifest = fixture_rows(data)
        checkpoint = data / "dummy-checkpoint.bin"
        checkpoint.write_bytes(b"synthetic-not-a-trained-model")
        out = tmp_path / "evaluation"
        result = write_evaluation_artifacts(
            out, rows, manifest, config={"fixture": True}, seed=5, checkpoint_path=checkpoint,
            input_paths={"features": data / "features.npy"}, review_k=4, bootstrap_samples=20,
            fixed_rule={"rule": "synthetic score >= 0"})
        metadata = json.loads((out / "run_metadata.json").read_text())
        assert metadata["checkpoint"]["sha256"] == file_sha256(checkpoint)
        assert metadata["inputs"]["features"]["sha256"] == file_sha256(data / "features.npy")
        assert metadata["source"]["commit"]
        assert result["provenance"]["predictions_sha256"] == file_sha256(out / "predictions.json")
        predictions = json.loads((out / "predictions.json").read_text())
        errors = {r["operating_error"] for r in predictions if r["split"] == "test"}
        assert {"fp", "fn"} <= errors
        assert (out / "pr_curve_test.csv").read_text().startswith("precision,recall,threshold")
        assert len((out / "review_queue_test.csv").read_text().splitlines()) == 5

    def _check_supervised_control(self, method):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(40, 6)).astype(np.float32)
        y = (x[:, 0] > 0).astype(int)
        first, history = fit_baseline(method, x[:24], y[:24], x[24:], y[24:], seed=7,
                                      epochs=6, hidden_dim=8)
        second, _ = fit_baseline(method, x[:24], y[:24], x[24:], y[24:], seed=7,
                                 epochs=6, hidden_dim=8)
        np.testing.assert_allclose(first.predict_proba(x), second.predict_proba(x))
        if method == "mlp":
            assert first.early_stopping is False
            assert history["best_epoch"] == int(np.argmin(history["val_log_loss"])) + 1
            assert len(history["val_log_loss"]) == 6

    def test_main_table_rejects_incompatible_experiments_and_tampered_predictions(self):
        tmp_path = self.tmp_path
        rows, manifest = fixture_rows(tmp_path / "data")
        checkpoint = tmp_path / "checkpoint.bin"
        checkpoint.write_bytes(b"synthetic")
        runs = [tmp_path / "logistic", tmp_path / "mlp"]
        for run_dir in runs:
            write_evaluation_artifacts(
                run_dir, rows, manifest, config={"method": run_dir.name}, seed=1,
                checkpoint_path=checkpoint, input_paths={"features": tmp_path / "data/features.npy",
                                                         "meta": tmp_path / "data/meta.json"},
                review_k=3, bootstrap_samples=0)
        table, summary = collect_runs(runs)
        assert len(table) == len(summary) == 2
        assert summary[0]["ap_std"] is None  # One seed has no measured seed variability.
        metadata_path = runs[1] / "run_metadata.json"
        metadata_original = json.loads(metadata_path.read_text())
        metadata_different_protocol = json.loads(metadata_path.read_text())
        metadata_different_protocol["config"]["feature_provenance"] = {
            "schema": "cpu-resnet18-six-camera-v1", "encoder": "resnet18-imagenet1k-v1"}
        metadata_path.write_text(json.dumps(metadata_different_protocol))
        with self.assertRaisesRegex(ValueError, "feature protocol"):
            collect_runs(runs)
        metadata_path.write_text(json.dumps(metadata_original))
        evaluation_path = runs[1] / "evaluation.json"
        evaluation = json.loads(evaluation_path.read_text())
        evaluation["splits"]["test"]["operating"]["ap"] = -1
        evaluation_path.write_text(json.dumps(evaluation))
        with self.assertRaisesRegex(ValueError, "metrics disagree"):
            collect_runs(runs)
        evaluation["splits"]["test"]["operating"]["ap"] = table[1]["ap"]
        evaluation_path.write_text(json.dumps(evaluation))
        metadata_path = runs[1] / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        original_meta_hash = metadata["inputs"]["meta"]["sha256"]
        metadata["inputs"]["meta"]["sha256"] = "different-token-feature-order"
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "Runs differ"):
            collect_runs(runs)
        metadata["inputs"]["meta"]["sha256"] = original_meta_hash
        metadata["inputs"]["features"]["sha256"] = "different-feature-matrix"
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "Runs differ"):
            collect_runs(runs)
        (runs[0] / "predictions.json").write_text("[]")
        with self.assertRaisesRegex(ValueError, "modified predictions"):
            collect_runs(runs[:1])

    def test_supervised_controls_are_deterministic_and_selected_on_explicit_validation(self):
        for method in ("logistic", "mlp"):
            with self.subTest(method=method):
                self._check_supervised_control(method)


if __name__ == "__main__":
    unittest.main()
