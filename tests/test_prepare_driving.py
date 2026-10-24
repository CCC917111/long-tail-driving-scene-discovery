"""Synthetic nuScenes-schema fixtures only; no real data or human labels are used."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_driving import CAMERAS, import_labels, prepare  # noqa: E402


def synthetic_nuscenes(root: Path, scenes=10, frames=2):
    """Create five official-shaped tables and explicitly synthetic image files."""
    version = root / "v1.0-mini"
    version.mkdir(parents=True)
    tables = {name: [] for name in ("scene", "sample", "sample_data",
                                    "calibrated_sensor", "sensor")}
    for camera in CAMERAS:
        tables["sensor"].append({"token": f"sensor-{camera}", "modality": "camera",
                                 "channel": camera})
        tables["calibrated_sensor"].append({"token": f"cal-{camera}",
                                            "sensor_token": f"sensor-{camera}"})
    for scene in range(scenes):
        scene_token = f"synthetic-scene-{scene:03d}"
        tables["scene"].append({"token": scene_token, "name": f"SYNTHETIC scene {scene}",
                                "nbr_samples": frames})
        for frame in range(frames):
            token = f"{scene_token}-frame-{frame:02d}"
            tables["sample"].append({"token": token, "scene_token": scene_token,
                                     "timestamp": scene * 10000 + frame})
            for camera in CAMERAS:
                filename = f"samples/{camera}/{token}.png"
                path = root / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                image = Image.new("RGB", (360, 180), (30, 45, 90))
                draw = ImageDraw.Draw(image)
                draw.text((15, 20), "SYNTHETIC nuScenes-schema fixture", fill="white")
                draw.text((15, 60), "NO REAL DRIVING IMAGE OR HUMAN LABEL", fill="white")
                draw.text((15, 100), token, fill="white")
                draw.text((15, 140), camera, fill="white")
                image.save(path)
                tables["sample_data"].append({
                    "token": f"sd-{token}-{camera}", "sample_token": token,
                    "calibrated_sensor_token": f"cal-{camera}",
                    "is_key_frame": True, "filename": filename})
    # A normal nuScenes reader must not require intermediate sweep images.
    tables["sample_data"].append({
        "token": "synthetic-unused-sweep", "sample_token": "synthetic-scene-000-frame-00",
        "calibrated_sensor_token": "cal-CAM_FRONT", "is_key_frame": False,
        "filename": "sweeps/missing.jpg"})
    tables["sample"].reverse()  # Sampling must use timestamps, not input JSON order.
    for name, rows in tables.items():
        (version / f"{name}.json").write_text(json.dumps(rows), encoding="utf-8")
    (root / "SYNTHETIC.txt").write_text("Unit-test schema and images only; not nuScenes data.\n")
    return root


def write_reviews(prepared: Path, destination: Path, mutate=None):
    with (prepared / "manual_review.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        frame_index = int(row["sample_token"].rsplit("-", 1)[1])
        row.update(label="normal_core" if frame_index % 2 == 0 else "long_tail",
                   review_status="reviewed", reviewer="synthetic-unit-test-reviewer",
                   review_notes="Synthetic test assertions, not human judgments.")
    if mutate:
        mutate(rows)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


class TestPrepareDriving(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic-nuscenes-test-")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self, **kwargs):
        return synthetic_nuscenes(self.root / "synthetic-data", **kwargs)

    def test_uniform_sampling_scene_split_hashes_and_pending_review(self):
        data, out = self.fixture(frames=9), self.root / "prepared"
        report = prepare(data, "v1.0-mini", out, frames_per_scene=3)
        rows = json.loads((out / "raw_manifest.json").read_text())
        assert len(rows) == 30
        assert report["scene_counts"] == {"train": 6, "val": 2, "test": 2}
        assert report["official_split_used"] is False and report["labels_created"] is False
        assert report["rubric_version"].startswith("sha256:")
        rubric_hash = hashlib.sha256((out / "rubric.md").read_bytes()).hexdigest()
        assert report["rubric_sha256"] == rubric_hash
        assert "not an official" in report["scope"]
        scene_splits = {}
        for row in rows:
            assert row["scene_position"] in {0, 4, 8}
            assert row["label"] is None and row["review_status"] == "pending"
            assert scene_splits.setdefault(row["scene_token"], row["split"]) == row["split"]
            assert set(row["images"]) == set(CAMERAS)
            for info in row["image_files"].values():
                digest = hashlib.sha256((data / info["filename"]).read_bytes()).hexdigest()
                assert info["sha256"] == digest
        again = self.root / "prepared-again"
        prepare(data, "v1.0-mini", again, frames_per_scene=3)
        assert (out / "split_scenes.json").read_bytes() == (
            again / "split_scenes.json").read_bytes()
        with (out / "manual_review.csv").open(newline="") as handle:
            review = list(csv.DictReader(handle))
        assert all(r["label"] == "" and r["review_status"] == "pending" for r in review)
        page = (out / "review.html").read_text()
        assert "file://" in page and "Download review CSV" in page and "no server" in page

    def test_missing_and_ambiguous_cameras_are_traceable_not_replaced(self):
        data = self.fixture()
        (data / "samples/CAM_FRONT/synthetic-scene-000-frame-00.png").unlink()
        table = data / "v1.0-mini/sample_data.json"
        rows = json.loads(table.read_text())
        duplicate = dict(next(r for r in rows if r["sample_token"] ==
                              "synthetic-scene-001-frame-00" and
                              r["calibrated_sensor_token"] == "cal-CAM_BACK"),
                         token="duplicate-cam")
        rows.append(duplicate)
        table.write_text(json.dumps(rows))
        out = self.root / "prepared"
        result = prepare(data, "v1.0-mini", out)
        excluded = json.loads((out / "excluded_rows.json").read_text())
        assert result["candidate_frames"] == 20
        assert result["included_frames"] == 18 and len(excluded) == 2
        assert any("missing_file:CAM_FRONT" in r["exclusion_reason"] for r in excluded)
        assert any("duplicate_camera:CAM_BACK" in r["exclusion_reason"] for r in excluded)
        assert all(r["split"] == "excluded" and r["original_split"] for r in excluded)

    def test_duplicate_tokens_and_unknown_scenes_fail_before_outputs(self):
        data = self.fixture()
        table = data / "v1.0-mini/sample.json"
        samples = json.loads(table.read_text())
        table.write_text(json.dumps(samples + samples[:1]))
        with self.assertRaisesRegex(ValueError, "duplicate token"):
            prepare(data, "v1.0-mini", self.root / "not-created")
        samples[0]["scene_token"] = "unknown-scene"
        table.write_text(json.dumps(samples))
        with self.assertRaisesRegex(ValueError, "Unknown scene_token"):
            prepare(data, "v1.0-mini", self.root / "not-created")
        assert not (self.root / "not-created").exists()

    def test_parent_paths_are_excluded_without_reading_outside_root(self):
        data = self.fixture()
        table = data / "v1.0-mini/sample_data.json"
        rows = json.loads(table.read_text())
        rows[0]["filename"] = "samples/CAM_FRONT/../../../outside.png"
        table.write_text(json.dumps(rows))
        out = self.root / "prepared"
        prepare(data, "v1.0-mini", out)
        excluded = json.loads((out / "excluded_rows.json").read_text())
        assert "invalid_camera_path" in excluded[0]["exclusion_reason"]
        assert excluded[0]["image_files"]["CAM_FRONT"]["sha256"] is None

    def test_import_keeps_frozen_splits_and_supports_existing_manifest_loader(self):
        import numpy as np
        from data_manifest import load_frozen_manifest

        data, prepared = self.fixture(), self.root / "prepared"
        prepare(data, "v1.0-mini", prepared)
        reviews = self.root / "synthetic-reviews.csv"
        write_reviews(prepared, reviews)
        out = self.root / "imported"
        result = import_labels(prepared, [reviews], out)
        assert result["included_labels"] == 20 and result["split_recomputed"] is False
        assert result["ready_for_current_training_contract"] is True
        meta = json.loads((prepared / "selected_samples.json").read_text())
        manifest = json.loads((out / "manifest.json").read_text())
        values = {r["sample_token"]: r["label"] for r in manifest}
        splits = load_frozen_manifest(out / "manifest.json", meta,
                                      np.array([values[r["sample_token"]] for r in meta]))
        assert set(splits) == {"train", "val", "test"}
        original = {r["sample_token"]: r["split"] for r in meta}
        assert all(r["split"] == original[r["sample_token"]] for r in manifest)
        assert all(r["source_version"] == "v1.0-mini" and
                   "not an official" in r["evaluation_scope"] for r in manifest)

    def test_pending_and_uncertain_are_excluded_instead_of_becoming_positive(self):
        prepared = self.root / "prepared"
        prepare(self.fixture(), "v1.0-mini", prepared)
        reviews = self.root / "synthetic-reviews.csv"

        def change(rows):
            rows[0].update(label="not_normal_core", review_status="pending", reviewer="")
            rows[1].update(label="uncertain", review_status="uncertain")

        rows = write_reviews(prepared, reviews, change)
        out = self.root / "imported"
        report = import_labels(prepared, [reviews], out)
        labels = {r["sample_token"]: r for r in json.loads((out / "labels.json").read_text())}
        manifest = json.loads((out / "manifest.json").read_text())
        assert labels[rows[0]["sample_token"]]["label"] == ""
        assert labels[rows[1]["sample_token"]]["label"] == "uncertain"
        assert not ({r["sample_token"] for r in rows[:2]} &
                    {r["sample_token"] for r in manifest})
        assert report["included_labels"] == 18 and report["excluded_rows"] == 2

    def test_review_duplicates_unknown_labels_reviewer_and_split_tampering_rejected(self):
        prepared = self.root / "prepared"
        prepare(self.fixture(), "v1.0-mini", prepared)
        path = self.root / "synthetic-reviews.csv"
        cases = [
            (lambda rows: rows.append(dict(rows[0])), "duplicate"),
            (lambda rows: rows.append(dict(rows[0], label="long_tail")), "conflicting labels"),
            (lambda rows: rows[0].update(label="unexpected"), "Unknown label"),
            (lambda rows: rows[0].update(reviewer=""), "requires a reviewer"),
            (lambda rows: rows[0].update(split="changed"), "frozen scene/split"),
            (lambda rows: rows[0].update(sample_token="not-in-prepared"), "Unknown or excluded"),
        ]
        for mutation, message in cases:
            with self.subTest(message=message):
                write_reviews(prepared, path, mutation)
                with self.assertRaisesRegex(ValueError, message):
                    import_labels(prepared, [path], self.root / "not-created")

    def test_raw_manifest_mutation_and_overwriting_are_rejected(self):
        prepared = self.root / "prepared"
        data = self.fixture()
        prepare(data, "v1.0-mini", prepared)
        with self.assertRaisesRegex(ValueError, "fresh, empty"):
            prepare(data, "v1.0-mini", prepared)
        path = self.root / "synthetic-reviews.csv"
        write_reviews(prepared, path)
        raw = prepared / "raw_manifest.json"
        raw.write_text(raw.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "changed after split freeze"):
            import_labels(prepared, [path], self.root / "not-created")

    def test_cli_prepare_and_import_on_explicit_synthetic_schema(self):
        script = Path(__file__).resolve().parents[1] / "scripts/prepare_driving.py"
        prepared = self.root / "cli-prepared"
        result = subprocess.run([sys.executable, str(script), "prepare", "--data-root",
                                 str(self.fixture()), "--version", "v1.0-mini", "--output-dir",
                                 str(prepared)], capture_output=True, text=True, check=True)
        assert json.loads(result.stdout)["included_frames"] == 20
        reviews = self.root / "synthetic-cli-reviews.csv"
        write_reviews(prepared, reviews)
        result = subprocess.run([sys.executable, str(script), "import-labels", "--prepared-dir",
                                 str(prepared), "--reviews", str(reviews), "--output-dir",
                                 str(self.root / "cli-imported")], capture_output=True,
                                text=True, check=True)
        assert json.loads(result.stdout)["included_labels"] == 20
        if shutil.which("node"):
            page = (prepared / "review.html").read_text()
            javascript = page.split("<script>", 1)[1].split("</script>", 1)[0]
            script_path = self.root / "review-script.js"
            script_path.write_text(javascript)
            subprocess.run(["node", "--check", str(script_path)], check=True,
                           capture_output=True, text=True)
            runner = """
const fs=require('fs'),vm=require('vm');
const page=fs.readFileSync(process.argv[2],'utf8');
const script=page.split('<script>')[1].split('</script>')[0];
const elements={'#cards':{},'#export':{},'#reviewer':{value:'synthetic-JS-reviewer'}};
const values={'.label':{value:'normal_core'},'.status':{value:'reviewed'},
'.notes':{value:'Quoted "note", with comma\\nand newline'}};
const context={document:{querySelector:s=>elements[s],
querySelectorAll:()=>[{dataset:{i:'0'},querySelector:s=>values[s]}],
createElement:()=>({click:()=>{}})},
Blob:class{constructor(parts){context.csv=parts.join('')}},
URL:{createObjectURL:()=> 'blob:synthetic',revokeObjectURL:()=>{}},setTimeout:()=>{}};
vm.runInNewContext(script,context);elements['#export'].onclick();
process.stdout.write(JSON.stringify(context.csv));
"""
            exported = subprocess.run(["node", "-", str(prepared / "review.html")],
                                      input=runner, check=True, capture_output=True, text=True)
            import io
            records = list(csv.DictReader(io.StringIO(json.loads(exported.stdout))))
            assert len(records) == 1 and records[0]["label"] == "normal_core"
            assert records[0]["reviewer"] == "synthetic-JS-reviewer"
            assert records[0]["review_notes"] == 'Quoted "note", with comma\nand newline'


if __name__ == "__main__":
    unittest.main()
