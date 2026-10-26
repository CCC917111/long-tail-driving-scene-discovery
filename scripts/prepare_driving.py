#!/usr/bin/env python3
"""Prepare already-authorised nuScenes files for a new, human-labelled CPU smoke.

This tool never downloads data, accepts terms, infers labels or reconstructs an
old experiment. Mini uses a frozen INTERNAL 60/20/20 scene split for a proof of
concept; its internal test split is not an official nuScenes test benchmark.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

CAMERAS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK",
           "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
TABLES = ("scene", "sample", "sample_data", "calibrated_sensor", "sensor")
SPLITS = ("train", "val", "test")
NORMAL = {"normal", "normal_core"}
TAIL = {"not_normal_core", "long_tail", "long-tail", "tail"}
UNDECIDED = {"", "unknown", "uncertain", "none"}
REVIEW_FIELDS = ("sample_token", "scene_token", "split", "label", "review_status",
                 "reviewer", "review_notes")
VERSION_RATIOS = {"v1.0-mini": (0.6, 0.2, 0.2), "v1.0-trainval": (0.7, 0.15, 0.15)}
DEFAULT_RUBRIC = Path(__file__).resolve().parents[1] / "latest_grading_criteria.md"
POST_AI_REVIEW_CONTEXT = "post_ai_exposure_single_reviewer"
POST_AI_LABEL_ORIGIN = "human_post_ai_exposure"


def post_ai_review_scope() -> dict:
    return {"label_protocol": "manual", "review_context": POST_AI_REVIEW_CONTEXT,
            "human_reviewed": True, "human_ground_truth_verified": False,
            "independent_human_review": False, "prior_ai_labels_exposed": True,
            "reviewer_count": 1, "official_benchmark": False,
            "interpretation": "Agreement with post-AI-exposure human review by a single "
                              "reviewer; not independent human ground truth or an official "
                              "benchmark result"}


def review_context_scope(rows: list[dict]) -> dict | None:
    """Validate the optional recorded context, including excluded judgments."""
    contexts = [row.get("review_context", "") for row in rows]
    if any(context not in ("", POST_AI_REVIEW_CONTEXT) for context in contexts):
        raise ValueError("Unknown review_context")
    if not any(contexts) and not any(row.get("label_origin") == POST_AI_LABEL_ORIGIN
                                     for row in rows):
        return None
    if not all(context == POST_AI_REVIEW_CONTEXT for context in contexts):
        raise ValueError("Mixed or missing review_context in manual judgments")
    if any(row.get("label_origin") == "ai_visual_provisional"
           or row.get("label_source") == "ai_visual_provisional"
           or row.get("review_status") == "ai_provisional"
           or row.get("label_protocol") == "ai-visual-pilot" for row in rows):
        raise ValueError("AI provisional labels cannot carry a manual review_context")
    reviewers = {row["reviewer"].strip() for row in rows
                 if isinstance(row.get("reviewer"), str) and row["reviewer"].strip()}
    if len(reviewers) != 1:
        raise ValueError("Single-reviewer review_context requires exactly one nonblank reviewer")
    return post_ai_review_scope()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def fresh_directory(path: Path) -> Path:
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Use a fresh, empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def indexed_table(path: Path) -> dict:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path.name} must be a JSON array")
    indexed = {}
    for row in rows:
        token = row.get("token") if isinstance(row, dict) else None
        if not isinstance(token, str) or not token.strip() or token in indexed:
            raise ValueError(f"{path.name}: missing or duplicate token {token!r}")
        indexed[token] = row
    return indexed


def scene_assignments(scene_tokens: list[str], version: str, seed: int) -> dict:
    if version not in VERSION_RATIOS:
        raise ValueError(f"Unsupported version: {version}")
    if len(scene_tokens) < 5:
        raise ValueError("At least five scenes are required for a three-way internal split")
    ratios = VERSION_RATIOS[version]
    targets = [len(scene_tokens) * value for value in ratios]
    counts = [math.floor(value) for value in targets]
    remaining = len(scene_tokens) - sum(counts)
    order = sorted(range(3), key=lambda i: (-(targets[i] - counts[i]), i))
    for i in order[:remaining]:
        counts[i] += 1
    if not all(counts):
        raise ValueError("Every internal split must contain a scene")
    tokens = sorted(scene_tokens)
    random.Random(seed).shuffle(tokens)
    result, start = {}, 0
    for split, count in zip(SPLITS, counts):
        result.update({token: split for token in tokens[start:start + count]})
        start += count
    return result


def uniform_positions(length: int, count: int) -> list[int]:
    count = min(length, count)
    if count < 1:
        return []
    if count == 1:
        return [length // 2]
    return [i * (length - 1) // (count - 1) for i in range(count)]


def inspect_image(data_root: Path, filename, camera: str) -> dict:
    result = {"filename": filename, "sha256": None, "bytes": None}
    if not isinstance(filename, str):
        return dict(result, error=f"invalid_filename:{camera}")
    relative = Path(filename)
    if (relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 3
            or relative.parts[:2] != ("samples", camera)):
        return dict(result, error=f"invalid_camera_path:{camera}")
    path = (data_root / relative).resolve()
    if data_root not in path.parents:
        return dict(result, error=f"image_outside_data_root:{camera}")
    if not path.is_file():
        return dict(result, error=f"missing_file:{camera}")
    if path.stat().st_size == 0:
        return dict(result, error=f"empty_file:{camera}")
    try:
        return dict(result, sha256=sha256(path), bytes=path.stat().st_size, error=None)
    except OSError:
        return dict(result, error=f"unreadable_file:{camera}")


def write_review_html(path: Path, rows: list[dict], data_root: Path, scope: str,
                      rubric_text: str, rubric_version: str) -> None:
    cards = []
    for row in rows:
        item = {key: row[key] for key in ("sample_token", "scene_token", "scene_name", "split")}
        item["images"] = {camera: (data_root / info["filename"]).as_uri()
                          for camera, info in row["image_files"].items()}
        cards.append(item)
    payload = json.dumps(cards, ensure_ascii=False).replace("<", "\\u003c")
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Driving manual review</title><style>
body{font:15px/1.5 system-ui;margin:24px;background:#f7f8fb;color:#172033}
header{position:sticky;top:0;background:#fff;padding:12px;border:1px solid #ddd;z-index:2}
article{background:white;padding:16px;margin:20px 0;border:1px solid #ddd}
.views{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}
figure{margin:0}img{width:100%;aspect-ratio:16/9;object-fit:contain;background:#222}
input,select,button{font:inherit;margin:6px;padding:6px}input.notes{width:40%}
small{display:block;color:#586174}@media(max-width:650px){.views{grid-template-columns:1fr 1fr}}
</style><header><strong>New manual labels; no automatic ground truth</strong>
<p>__SCOPE__</p><label>Reviewer <input id="reviewer" placeholder="Your identifier"></label>
<button id="export">Download review CSV</button>
<small>Use the agreed rubric. Leave pending or choose uncertain when needed. To include a label,
set status reviewed and supply a reviewer. Download before closing; edits are not saved to disk.
This page reads local camera files and uses no server or account.</small></header>
<details><summary>Frozen annotation rubric (__RUBRIC_VERSION__)</summary>
<pre style="white-space:pre-wrap">__RUBRIC__</pre></details>
<main id="cards"></main><script>
const rows=__ROWS__;
const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',
'"':'&quot;',"'":'&#39;'}[c]));
document.querySelector('#cards').innerHTML=rows.map((r,i)=>`<article data-i="${i}">
<h3>${esc(r.sample_token)}</h3><p>${esc(r.scene_name)} · ${esc(r.split)} (internal)</p>
<div class="views">${Object.entries(r.images).map(([cam,url])=>`<figure>
<img loading="lazy" src="${esc(url)}">
<figcaption>${esc(cam)}</figcaption></figure>`).join('')}</div>
<label>Label <select class="label"><option value="">Pending</option>
<option>normal_core</option><option>not_normal_core</option><option>uncertain</option>
</select></label>
<label>Status <select class="status"><option>pending</option><option>reviewed</option>
<option>uncertain</option></select></label>
<input class="notes" placeholder="Review notes"></article>`
).join('');
document.querySelector('#export').onclick=()=>{
const quote=x=>'"'+String(x??'').replaceAll('"','""')+'"';
const content=[['sample_token','scene_token','split','label',
'review_status','reviewer','review_notes']];
document.querySelectorAll('article').forEach(a=>{const r=rows[+a.dataset.i];content.push([
r.sample_token,r.scene_token,r.split,a.querySelector('.label').value,
a.querySelector('.status').value,document.querySelector('#reviewer').value.trim(),
a.querySelector('.notes').value]);});
const blob=new Blob([content.map(r=>r.map(quote).join(',')).join('\\r\\n')+'\\r\\n'],
{type:'text/csv;charset=utf-8'});const a=document.createElement('a');
a.href=URL.createObjectURL(blob);
a.download='reviewed.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};
</script></html>"""
    page = page.replace("__ROWS__", payload).replace("__SCOPE__", html.escape(scope))
    page = page.replace("__RUBRIC__", html.escape(rubric_text)).replace(
        "__RUBRIC_VERSION__", html.escape(rubric_version))
    path.write_text(page, encoding="utf-8")


def prepare(data_root: Path, version: str, output_dir: Path, *, frames_per_scene: int = 8,
            seed: int = 42, rubric_version: str | None = None,
            rubric_file: Path = DEFAULT_RUBRIC) -> dict:
    if frames_per_scene < 1:
        raise ValueError("frames-per-scene must be positive")
    data_root = Path(data_root).resolve()
    tables_dir = data_root / version
    tables = {name: indexed_table(tables_dir / f"{name}.json") for name in TABLES}
    assignments = scene_assignments(list(tables["scene"]), version, seed)
    scope = ("INTERNAL mini proof-of-concept 60/20/20 scene split; not an official test benchmark."
             if version == "v1.0-mini" else
             "INTERNAL 70/15/15 scene split; not the official nuScenes benchmark split.")
    rubric_file = Path(rubric_file).resolve()
    rubric_text, rubric_hash = rubric_file.read_text(encoding="utf-8"), sha256(rubric_file)
    if not rubric_text.strip():
        raise ValueError("Annotation rubric must not be empty")
    rubric_version = rubric_version or f"sha256:{rubric_hash[:16]}"
    grouped = defaultdict(list)
    for sample in tables["sample"].values():
        if sample.get("scene_token") not in tables["scene"]:
            raise ValueError(f"Unknown scene_token for sample {sample['token']}")
        timestamp = sample.get("timestamp")
        if type(timestamp) not in (int, float) or not math.isfinite(timestamp):
            raise ValueError(f"Invalid timestamp for sample {sample['token']}")
        grouped[sample["scene_token"]].append(sample)
    if set(grouped) != set(tables["scene"]):
        raise ValueError("Every source scene must have samples")
    cameras = defaultdict(lambda: defaultdict(list))
    for data in tables["sample_data"].values():
        if not data.get("is_key_frame"):
            continue
        if data.get("sample_token") not in tables["sample"]:
            raise ValueError("Key-frame sample_data references an unknown sample")
        calibration = tables["calibrated_sensor"].get(data.get("calibrated_sensor_token"))
        sensor = tables["sensor"].get(calibration.get("sensor_token")) if calibration else None
        if sensor is None:
            raise ValueError("sample_data has an invalid calibrated sensor reference")
        if sensor.get("modality") == "camera" and sensor.get("channel") in CAMERAS:
            cameras[data["sample_token"]][sensor["channel"]].append(data)
    raw = []
    for scene_token in sorted(grouped):
        samples = sorted(grouped[scene_token], key=lambda row: (row["timestamp"], row["token"]))
        for position in uniform_positions(len(samples), frames_per_scene):
            sample = samples[position]
            token, errors, image_files, images = sample["token"], [], {}, {}
            for camera in CAMERAS:
                candidates = cameras[token][camera]
                if len(candidates) != 1:
                    errors.append(f"{'missing' if not candidates else 'duplicate'}_camera:{camera}")
                    continue
                info = inspect_image(data_root, candidates[0].get("filename"), camera)
                image_files[camera] = info
                if info["error"]:
                    errors.append(info["error"])
                else:
                    images[camera] = Path(info["filename"]).relative_to("samples").as_posix()
            raw.append({"sample_token": token, "scene_token": scene_token,
                        "scene_name": tables["scene"][scene_token].get("name", scene_token),
                        "timestamp": sample["timestamp"], "scene_position": position,
                        "source_version": version, "evaluation_scope": scope,
                        "split": assignments[scene_token],
                        "label": None, "label_source": "pending-manual-review",
                        "review_status": "pending", "reviewer": "", "images": images,
                        "image_files": image_files, "exclusion_reason": ";".join(errors)})
    selected = [row for row in raw if not row["exclusion_reason"]]
    excluded = [dict(row, original_split=row["split"], split="excluded")
                for row in raw if row["exclusion_reason"]]
    out = fresh_directory(output_dir)
    write_json(out / "raw_manifest.json", raw)
    write_json(out / "selected_samples.json", [{key: row[key] for key in
               ("sample_token", "scene_token", "scene_name", "timestamp", "images", "split",
                "source_version", "evaluation_scope")}
               for row in selected])
    scene_rows = [{"scene_token": token, "split": assignments[token]}
                  for token in sorted(assignments)]
    write_json(out / "split_scenes.json", scene_rows)
    write_json(out / "excluded_rows.json", excluded)
    with (out / "manual_review.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in selected:
            writer.writerow(dict(row, label="", review_notes=""))
    (out / "rubric.md").write_text(rubric_text, encoding="utf-8")
    write_review_html(out / "review.html", selected, data_root, scope, rubric_text, rubric_version)
    metadata = {"schema_version": "new-nuscenes-preparation-v1", "source_version": version,
                "data_root": str(data_root), "scope": scope, "official_split_used": False,
                "split_before_labels": True, "split_seed": seed,
                "target_ratios": dict(zip(SPLITS, VERSION_RATIOS[version])),
                "scene_counts": dict(Counter(assignments.values())),
                "source_samples": len(tables["sample"]), "source_scenes": len(assignments),
                "frames_per_scene": frames_per_scene, "candidate_frames": len(raw),
                "included_frames": len(selected), "excluded_frames": len(excluded),
                "rubric_version": rubric_version, "rubric_sha256": rubric_hash,
                "rubric_source_path": str(rubric_file), "labels_created": False,
                "image_validation": "file exists, nonempty, readable for SHA256; not decoded",
                "source_metadata": {name: {"path": str(tables_dir / f"{name}.json"),
                                           "sha256": sha256(tables_dir / f"{name}.json")}
                                    for name in TABLES},
                "prepared_files": {name: sha256(out / name) for name in
                                   ("raw_manifest.json", "selected_samples.json",
                                    "split_scenes.json", "rubric.md", "manual_review.csv",
                                    "review.html")}}
    write_json(out / "prepare_metadata.json", metadata)
    return metadata


def normalise_review_label(raw: str) -> int | None:
    value = raw.strip().lower()
    if value in NORMAL:
        return 0
    if value in TAIL:
        return 1
    if value in UNDECIDED:
        return None
    raise ValueError(f"Unknown label {raw!r}; use normal_core/not_normal_core/uncertain")


def import_labels(prepared_dir: Path, reviews: list[Path], output_dir: Path) -> dict:
    prepared_dir = Path(prepared_dir).resolve()
    preparation = json.loads((prepared_dir / "prepare_metadata.json").read_text())
    for name in ("raw_manifest.json", "selected_samples.json", "split_scenes.json", "rubric.md"):
        if sha256(prepared_dir / name) != preparation["prepared_files"][name]:
            raise ValueError(f"Prepared file changed after split freeze: {name}")
    raw = json.loads((prepared_dir / "raw_manifest.json").read_text())
    eligible = {row["sample_token"]: row for row in raw if not row["exclusion_reason"]}
    reviewed, source_hashes, context_rows = {}, {}, []
    for path in map(Path, reviews):
        source_hashes[str(path.resolve())] = sha256(path)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"sample_token", "label", "review_status", "reviewer"} <= set(
                    reader.fieldnames or []):
                raise ValueError("Review CSV requires sample_token, label, review_status, reviewer")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError("Review CSV has duplicate column names")
            for line, row in enumerate(reader, 2):
                if None in row or any(not isinstance(row.get(key), str) for key in
                                      ("sample_token", "label", "review_status", "reviewer")):
                    raise ValueError(f"Malformed review CSV row {line}")
                token = row["sample_token"].strip()
                label = normalise_review_label(row["label"])
                if token in reviewed:
                    kind = ("conflicting labels" if reviewed[token]["binary"] != label
                            else "duplicate")
                    raise ValueError(f"Review {kind} for token {token}")
                if token not in eligible:
                    raise ValueError(f"Unknown or excluded review sample_token {token}")
                source = eligible[token]
                if any(row.get(key) and row[key] != source[key]
                       for key in ("scene_token", "split")):
                    raise ValueError(f"Review attempted to change frozen scene/split for {token}")
                status = row["review_status"].strip().lower() or "pending"
                if status not in {"pending", "reviewed", "uncertain"}:
                    raise ValueError(f"Unknown review_status {status!r}")
                reviewer = row["reviewer"].strip()
                if status == "reviewed" and not reviewer:
                    raise ValueError(f"Reviewed label requires a reviewer for {token}")
                context = row.get("review_context", "")
                if not isinstance(context, str):
                    raise ValueError(f"Malformed review_context for {token}")
                context = context.strip()
                if context not in ("", POST_AI_REVIEW_CONTEXT):
                    raise ValueError(f"Unknown review_context {context!r}")
                context_rows.append(dict(row, review_context=context, reviewer=reviewer))
                reviewed[token] = {"binary": label, "raw_label": row["label"],
                                   "review_status": status, "reviewer": reviewer,
                                   "review_notes": row.get("review_notes", ""),
                                   "label_source": str(path.resolve()), "review_csv_line": line}
                if context:
                    reviewed[token].update(review_context=context,
                                           label_origin=POST_AI_LABEL_ORIGIN)
    label_scope = review_context_scope(context_rows)
    labels, manifest, excluded = [], [], []
    for source in raw:
        token = source["sample_token"]
        if source["exclusion_reason"]:
            excluded.append(dict(source, original_split=source["split"], split="excluded"))
            continue
        missing = {"binary": None, "raw_label": "", "review_status": "pending", "reviewer": "",
                   "review_notes": "", "label_source": "missing"}
        review = reviewed.get(token, missing)
        active = review["review_status"] == "reviewed" and review["binary"] is not None
        record = {key: value for key, value in review.items() if key not in ("binary", "raw_label")}
        uncertain = (review["review_status"] == "uncertain" or
                     review["raw_label"].strip().lower() in {"uncertain", "unknown"})
        label_text = ("long_tail" if review["binary"] else "normal_core") if active else (
            "uncertain" if uncertain else "")
        record.update(sample_token=token, scene_token=source["scene_token"], label=label_text,
                      submitted_label=review["raw_label"], source_version=source["source_version"],
                      evaluation_scope=source["evaluation_scope"],
                      rubric_version=preparation["rubric_version"])
        labels.append(record)
        if active:
            manifest.append({"sample_token": token, "scene_token": source["scene_token"],
                             "label": review["binary"], "split": source["split"],
                             "label_source": review["label_source"], "review_status": "reviewed",
                             "reviewer": review["reviewer"], "exclusion_reason": "",
                             "source_version": source["source_version"],
                             "evaluation_scope": source["evaluation_scope"],
                             "rubric_version": preparation["rubric_version"],
                             "rubric_sha256": preparation["rubric_sha256"]})
            if label_scope is not None:
                manifest[-1].update(review_context=review["review_context"],
                                    label_origin=review["label_origin"])
        else:
            excluded.append(dict(source, split="excluded", original_split=source["split"],
                                 exclusion_reason="pending_or_uncertain_manual_review", **record))
    out = fresh_directory(output_dir)
    write_json(out / "labels.json", labels)
    write_json(out / "manifest.json", manifest)
    write_json(out / "excluded_rows.json", excluded)
    support = {split: {"normal": sum(r["split"] == split and r["label"] == 0 for r in manifest),
                       "tail": sum(r["split"] == split and r["label"] == 1 for r in manifest)}
               for split in SPLITS}
    metadata = {"schema_version": "new-manual-label-import-v1", "scope": preparation["scope"],
                "split_recomputed": False, "review_inputs_sha256": source_hashes,
                "prepare_metadata_sha256": sha256(prepared_dir / "prepare_metadata.json"),
                "raw_manifest_sha256": sha256(prepared_dir / "raw_manifest.json"),
                "rubric_version": preparation["rubric_version"],
                "rubric_sha256": preparation["rubric_sha256"], "included_labels": len(manifest),
                "excluded_rows": len(excluded), "support": support,
                "ready_for_current_training_contract": all(all(v.values())
                                                           for v in support.values())}
    if label_scope is not None:
        metadata.update(review_context=POST_AI_REVIEW_CONTEXT, label_scope=label_scope)
    write_json(out / "import_metadata.json", metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Prepare already-authorised local nuScenes files")
    prep.add_argument("--data-root", type=Path, required=True)
    prep.add_argument("--version", choices=tuple(VERSION_RATIOS), required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--frames-per-scene", type=int, default=8)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--rubric-file", type=Path, default=DEFAULT_RUBRIC)
    prep.add_argument("--rubric-version", help="Defaults to a version derived from rubric SHA256")
    labels = commands.add_parser("import-labels", help="Import explicit human CSV judgments")
    labels.add_argument("--prepared-dir", type=Path, required=True)
    labels.add_argument("--reviews", type=Path, nargs="+", required=True)
    labels.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        report = prepare(args.data_root, args.version, args.output_dir,
                         frames_per_scene=args.frames_per_scene, seed=args.seed,
                         rubric_version=args.rubric_version, rubric_file=args.rubric_file)
    else:
        report = import_labels(args.prepared_dir, args.reviews, args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
