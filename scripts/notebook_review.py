"""Persistent manual six-camera review for Jupyter and Colab.

This helper records explicit human decisions only. Keep one live session per CSV;
the resulting full CSV is accepted by ``prepare_driving.py import-labels``.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import os
from pathlib import Path
import tempfile

from prepare_driving import CAMERAS, REVIEW_FIELDS, normalise_review_label


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ReviewSession:
    """Resume a reviewed CSV without changing its frozen sample allocation."""

    def __init__(self, prepared_dir: Path, reviews_csv: Path):
        self.prepared_dir = Path(prepared_dir).resolve()
        self.reviews_csv = Path(reviews_csv).resolve()
        metadata = json.loads((self.prepared_dir / "prepare_metadata.json").read_text())
        for name in ("raw_manifest.json", "selected_samples.json", "split_scenes.json",
                     "rubric.md"):
            if _sha256(self.prepared_dir / name) != metadata["prepared_files"][name]:
                raise ValueError(f"Prepared file changed after split freeze: {name}")
        self.rubric = (self.prepared_dir / "rubric.md").read_text(encoding="utf-8")
        self.rubric_version = metadata["rubric_version"]
        self.scope = metadata["scope"]
        self.selected = json.loads((self.prepared_dir / "selected_samples.json").read_text())
        if not isinstance(self.selected, list) or not self.selected:
            raise ValueError("Preparation has no selected samples to review")
        self._selected_by_token = {}
        for sample in self.selected:
            token = sample.get("sample_token")
            if not isinstance(token, str) or not token.strip() or token in self._selected_by_token:
                raise ValueError("Prepared selection has missing or duplicate sample tokens")
            if not sample.get("scene_token") or sample.get("split") not in ("train", "val", "test"):
                raise ValueError(f"Prepared selection has invalid scene/split for {token}")
            self._selected_by_token[token] = sample
        loaded = self._load_reviews() if self.reviews_csv.exists() else {}
        self.rows = []
        for sample in self.selected:
            token = sample["sample_token"]
            self.rows.append(loaded.get(token, {
                "sample_token": token, "scene_token": sample["scene_token"],
                "split": sample["split"], "label": "", "review_status": "pending",
                "reviewer": "", "review_notes": "",
            }))
        self._row_by_token = {row["sample_token"]: row for row in self.rows}
        # A single complete snapshot avoids duplicate tokens across resumed exports.
        self._persist()

    def _load_reviews(self) -> dict:
        result = {}
        with self.reviews_csv.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            required = {"sample_token", "label", "review_status", "reviewer"}
            if not required <= set(fields) or len(fields) != len(set(fields)):
                raise ValueError("Review CSV needs unique sample_token, label, review_status, "
                                 "reviewer columns")
            for line, source in enumerate(reader, 2):
                if None in source or any(not isinstance(value, str) for value in source.values()):
                    raise ValueError(f"Malformed review CSV row {line}")
                token = source["sample_token"].strip()
                if token in result:
                    raise ValueError(f"Duplicate review sample_token {token}")
                if token not in self._selected_by_token:
                    raise ValueError(f"Unknown or excluded review sample_token {token}")
                selected = self._selected_by_token[token]
                for key in ("scene_token", "split"):
                    if source.get(key) and source[key] != selected[key]:
                        raise ValueError(f"Review attempted to change frozen {key} for {token}")
                normalise_review_label(source["label"])
                status = source["review_status"].strip().lower() or "pending"
                if status not in ("pending", "reviewed", "uncertain"):
                    raise ValueError(f"Unknown review_status {status!r}")
                reviewer = source["reviewer"].strip()
                if status == "reviewed" and not reviewer:
                    raise ValueError(f"Reviewed label requires a reviewer for {token}")
                result[token] = {
                    "sample_token": token, "scene_token": selected["scene_token"],
                    "split": selected["split"], "label": source["label"],
                    "review_status": status, "reviewer": reviewer,
                    "review_notes": source.get("review_notes", ""),
                }
        return result

    def _persist(self) -> None:
        self.reviews_csv.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                             dir=self.reviews_csv.parent,
                                             prefix=f".{self.reviews_csv.name}.",
                                             delete=False) as handle:
                temporary = Path(handle.name)
                writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
                writer.writeheader()
                writer.writerows(self.rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.reviews_csv)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def record(self, sample_token: str, label: str, reviewer: str, notes: str = "") -> None:
        """Atomically save one explicit normal, tail or uncertain decision."""
        if sample_token not in self._row_by_token:
            raise ValueError(f"Unknown review sample_token {sample_token}")
        if label not in ("normal_core", "not_normal_core", "uncertain"):
            raise ValueError("Choose normal_core, not_normal_core or uncertain")
        if not isinstance(reviewer, str) or not isinstance(notes, str):
            raise ValueError("Reviewer and notes must be text")
        reviewer = reviewer.strip()
        if label != "uncertain" and not reviewer:
            raise ValueError("A reviewed label requires a reviewer")
        row = self._row_by_token[sample_token]
        previous = dict(row)
        row.update(label=label, review_status="uncertain" if label == "uncertain" else "reviewed",
                   reviewer=reviewer, review_notes=notes)
        try:
            self._persist()
        except Exception:
            row.clear()
            row.update(previous)
            raise

    def progress(self) -> dict[str, int]:
        summary = {"total": len(self.rows), "reviewed": 0, "uncertain": 0, "pending": 0,
                   "normal_core": 0, "not_normal_core": 0}
        for row in self.rows:
            binary = normalise_review_label(row["label"])
            if row["review_status"] == "reviewed" and binary is not None:
                summary["reviewed"] += 1
                summary["not_normal_core" if binary else "normal_core"] += 1
            elif (row["review_status"] == "uncertain"
                  or row["label"].strip().lower() in ("uncertain", "unknown")):
                summary["uncertain"] += 1
            else:
                summary["pending"] += 1
        return summary

    def widget(self, data_root: Path):
        """Return a notebook widget; each decision button saves the CSV immediately."""
        try:
            import ipywidgets as widgets
            from IPython.display import display
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Notebook review requires ipywidgets, IPython and Pillow. "
                              "Install them with: %pip install ipywidgets ipython Pillow") from exc

        samples_root = (Path(data_root).resolve() / "samples").resolve()
        slider = widgets.IntSlider(min=0, max=len(self.rows) - 1, description="Frame",
                                   continuous_update=False)
        reviewer = widgets.Text(description="Reviewer", placeholder="Your identifier")
        notes = widgets.Textarea(description="Notes", layout=widgets.Layout(width="95%"))
        previous = widgets.Button(description="Previous")
        following = widgets.Button(description="Next")
        summary, heading, message = widgets.HTML(), widgets.HTML(), widgets.HTML()
        views = widgets.GridBox(layout=widgets.Layout(grid_template_columns="repeat(3, 1fr)",
                                                      grid_gap="8px", width="100%"))
        rubric = widgets.Accordion(children=[widgets.HTML(
            "<pre style='white-space:pre-wrap'>" + html.escape(self.rubric) + "</pre>")])
        rubric.set_title(0, f"Frozen annotation rubric ({self.rubric_version})")
        rubric.selected_index = None

        def refresh(change=None):
            row, sample = self.rows[slider.value], self.selected[slider.value]
            token = row["sample_token"]
            scene_name = str(sample.get("scene_name", sample["scene_token"]))
            heading.value = (f"<b>{slider.value + 1}/{len(self.rows)}: {html.escape(token)}</b>"
                             f"<br>{html.escape(scene_name)}"
                             f" · {html.escape(row['split'])} (internal)"
                             f" · saved: {html.escape(row['label'] or 'pending')}")
            summary.value = html.escape(str(self.progress()))
            notes.value = row["review_notes"]
            if row["reviewer"]:
                reviewer.value = row["reviewer"]
            cards = []
            for camera in CAMERAS:
                relative = Path(sample["images"][camera])
                path = (samples_root / relative).resolve()
                if (relative.is_absolute() or ".." in relative.parts
                        or samples_root not in path.parents):
                    raise ValueError(f"Invalid prepared camera path for {token}: {camera}")
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    image.thumbnail((600, 400))
                    buffer = io.BytesIO()
                    image.save(buffer, format="JPEG", quality=90)
                cards.append(widgets.VBox([widgets.Label(camera), widgets.Image(
                    value=buffer.getvalue(), format="jpeg", layout=widgets.Layout(width="100%"))]))
            views.children = cards
            previous.disabled = slider.value == slider.min
            following.disabled = slider.value == slider.max

        def decide(label):
            def click(_button):
                token = self.rows[slider.value]["sample_token"]
                try:
                    self.record(token, label, reviewer.value, notes.value)
                except (ValueError, OSError) as exc:
                    message.value = f"<b>Not saved:</b> {html.escape(str(exc))}"
                    return
                message.value = f"Saved {html.escape(label)} for {html.escape(token)}."
                if slider.value < slider.max:
                    slider.value += 1
                else:
                    refresh()
            return click

        buttons = []
        for label in ("normal_core", "not_normal_core", "uncertain"):
            button = widgets.Button(description=label)
            button.on_click(decide(label))
            buttons.append(button)
        previous.on_click(lambda _button: setattr(
            slider, "value", max(slider.min, slider.value - 1)))
        following.on_click(lambda _button: setattr(
            slider, "value", min(slider.max, slider.value + 1)))
        slider.observe(refresh, names="value")
        refresh()
        help_output = widgets.Output()
        with help_output:
            display(widgets.HTML(
                "<p>Choose a decision to save it. Notes are saved with that choice. "
                "Previous/Next does not save unsubmitted edits. Resume uses the same CSV; "
                "download or retain it before deleting this runtime.</p>"))
        return widgets.VBox([widgets.HTML(html.escape(self.scope)), rubric, summary,
                             widgets.HBox([previous, following, slider]), heading, views,
                             reviewer, notes, widgets.HBox(buttons), message, help_output])
