"""Synthetic prepared records only; no real images or human decisions are used."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from notebook_review import ReviewSession  # noqa: E402
from prepare_driving import REVIEW_FIELDS, import_labels  # noqa: E402


def synthetic_preparation(directory: Path) -> Path:
    directory.mkdir()
    selected = [{"sample_token": f"synthetic-{index}", "scene_token": f"synthetic-scene-{index}",
                 "split": split, "source_version": "v1.0-mini", "images": {},
                 "evaluation_scope": "Synthetic test; no real driving result"}
                for index, split in enumerate(("train", "val", "test"))]
    values = {
        "selected_samples.json": selected,
        "raw_manifest.json": [dict(row, exclusion_reason="") for row in selected],
        "split_scenes.json": [{key: row[key] for key in ("scene_token", "split")}
                              for row in selected],
    }
    for name, value in values.items():
        (directory / name).write_text(json.dumps(value))
    (directory / "rubric.md").write_text("Synthetic test rubric; not human annotations.\n")
    hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
              for name in (*values, "rubric.md")}
    (directory / "prepare_metadata.json").write_text(json.dumps({
        "prepared_files": hashes, "rubric_version": "synthetic-rubric-v1",
        "rubric_sha256": hashes["rubric.md"],
        "scope": "Synthetic internal schema test; not a real nuScenes result.",
    }))
    return directory


class TestNotebookReview(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic-notebook-review-")
        self.root = Path(self.temporary.name)
        self.prepared = synthetic_preparation(self.root / "prepared")
        self.csv = self.root / "work" / "reviewed.csv"

    def tearDown(self):
        self.temporary.cleanup()

    def write_rows(self, rows):
        self.csv.parent.mkdir(parents=True, exist_ok=True)
        with self.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def test_new_session_persists_all_pending_rows(self):
        session = ReviewSession(self.prepared, self.csv)
        self.assertEqual(session.progress(), {
            "total": 3, "reviewed": 0, "uncertain": 0, "pending": 3,
            "normal_core": 0, "not_normal_core": 0,
        })
        with self.csv.open(newline="") as handle:
            self.assertEqual(list(csv.DictReader(handle)), session.rows)
        self.assertEqual([row["sample_token"] for row in session.selected],
                         [row["sample_token"] for row in session.rows])

    def test_record_reload_and_import_preserve_human_fields_and_frozen_split(self):
        session = ReviewSession(self.prepared, self.csv)
        session.record("synthetic-0", "normal_core", "test-reviewer-A",
                       'A note, with "quotes"\nand newline')
        session.record("synthetic-1", "not_normal_core", "test-reviewer-B",
                       "Synthetic decision only")
        session.record("synthetic-2", "uncertain", "", "Insufficient synthetic evidence")
        resumed = ReviewSession(self.prepared, self.csv)
        self.assertEqual(resumed.rows, session.rows)
        self.assertEqual(resumed.rows[0]["reviewer"], "test-reviewer-A")
        self.assertEqual(resumed.rows[1]["reviewer"], "test-reviewer-B")
        self.assertEqual(resumed.progress()["reviewed"], 2)
        self.assertEqual(resumed.progress()["uncertain"], 1)
        result = import_labels(self.prepared, [self.csv], self.root / "annotations")
        self.assertEqual(result["included_labels"], 2)
        manifest = json.loads((self.root / "annotations" / "manifest.json").read_text())
        self.assertEqual([(row["sample_token"], row["split"]) for row in manifest],
                         [("synthetic-0", "train"), ("synthetic-1", "val")])

    def test_partial_csv_resume_fills_only_missing_tokens(self):
        original = ReviewSession(self.prepared, self.csv)
        original.record("synthetic-1", "not_normal_core", "test-reviewer", "Keep this note")
        self.write_rows([original.rows[1]])
        resumed = ReviewSession(self.prepared, self.csv)
        self.assertEqual(resumed.rows[1], original.rows[1])
        self.assertEqual(resumed.progress()["pending"], 2)
        self.assertEqual(len(resumed.rows), 3)

    def test_invalid_decisions_leave_csv_and_memory_unchanged(self):
        session = ReviewSession(self.prepared, self.csv)
        before = self.csv.read_bytes()
        for args in (("synthetic-0", "normal_core", ""),
                     ("synthetic-0", "inferred-label", "test-reviewer"),
                     ("unknown-token", "normal_core", "test-reviewer")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                session.record(*args)
            self.assertEqual(self.csv.read_bytes(), before)
        self.assertEqual(session.progress()["pending"], 3)

    def test_failed_atomic_replace_keeps_previous_snapshot_and_session(self):
        session = ReviewSession(self.prepared, self.csv)
        before = self.csv.read_bytes()
        with patch("notebook_review.os.replace", side_effect=OSError("synthetic write failure")):
            with self.assertRaises(OSError):
                session.record("synthetic-0", "normal_core", "test-reviewer")
        self.assertEqual(self.csv.read_bytes(), before)
        self.assertEqual(session.progress()["pending"], 3)
        self.assertEqual(list(self.csv.parent.iterdir()), [self.csv])

    def test_duplicate_unknown_conflicting_and_invalid_csvs_are_rejected_before_writes(self):
        rows = ReviewSession(self.prepared, self.csv).rows
        cases = [
            [rows[0], rows[0]],
            [dict(rows[0], sample_token="synthetic-unknown")],
            [dict(rows[0], scene_token="other-scene")],
            [dict(rows[0], split="test")],
            [dict(rows[0], label="fabricated")],
            [dict(rows[0], review_status="automatic")],
            [dict(rows[0], label="normal_core", review_status="reviewed", reviewer="")],
        ]
        for case in cases:
            with self.subTest(case=case):
                self.write_rows(case)
                before = self.csv.read_bytes()
                with self.assertRaises(ValueError):
                    ReviewSession(self.prepared, self.csv)
                self.assertEqual(self.csv.read_bytes(), before)

    def test_frozen_rubric_change_rejects_resume(self):
        ReviewSession(self.prepared, self.csv)
        before = self.csv.read_bytes()
        (self.prepared / "rubric.md").write_text("Tampered synthetic rubric")
        with self.assertRaisesRegex(ValueError, "split freeze: rubric.md"):
            ReviewSession(self.prepared, self.csv)
        self.assertEqual(self.csv.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
