#!/usr/bin/env python3
"""Evaluate token-aligned scores without choosing an operating point on test.

This module is shared by the SAE and supervised baseline entry points. The CLI
can rebuild an evaluation from saved per-sample scores and a frozen manifest.
Synthetic inputs exercise the software contract; they are not model evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

PROTOCOL_VERSION = "scene-disjoint-evaluation-v1"
SPLITS = ("train", "val", "test")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n",
                    encoding="utf-8")


def _binary_label(value, context: str) -> int:
    if value not in (0, 1):
        raise ValueError(f"{context}: label/prediction must be 0 or 1, got {value!r}")
    return int(value)


def align_rows(rows: list[dict], manifest: list[dict]) -> list[dict]:
    """Require exactly the same included tokens, labels, splits and scene IDs.

    Excluded manifest entries may be retained with a nonempty exclusion_reason
    and split='excluded' (or no split); they must not have prediction rows.
    """
    expected, scene_split = {}, {}
    for entry in manifest:
        token = entry.get("sample_token")
        if not isinstance(token, str) or not token.strip():
            raise ValueError("Manifest sample_token must be a nonempty string")
        if token in expected:
            raise ValueError(f"Duplicate manifest token: {token}")
        expected[token] = entry
        if entry.get("exclusion_reason") and entry.get("split") not in SPLITS:
            continue
        split, scene = entry.get("split"), entry.get("scene_token")
        if split not in SPLITS:
            raise ValueError(f"{token}: invalid manifest split {split!r}")
        if not isinstance(scene, str) or not scene.strip():
            raise ValueError(f"{token}: scene_token is required for official evaluation")
        _binary_label(entry.get("label"), token)
        if scene in scene_split and scene_split[scene] != split:
            raise ValueError(f"Scene {scene!r} overlaps {scene_split[scene]} and {split}")
        scene_split[scene] = split
    included = {t: e for t, e in expected.items() if e.get("split") in SPLITS}
    by_token = {}
    for row in rows:
        token = row.get("sample_token")
        if token in by_token:
            raise ValueError(f"Duplicate prediction token: {token}")
        if token not in included:
            raise ValueError(f"Unexpected prediction token: {token}")
        entry = included[token]
        if row.get("split") != entry["split"]:
            raise ValueError(f"{token}: prediction split disagrees with manifest")
        if _binary_label(row.get("label"), token) != entry["label"]:
            raise ValueError(f"{token}: prediction label disagrees with manifest")
        if row.get("scene_token", entry["scene_token"]) != entry["scene_token"]:
            raise ValueError(f"{token}: prediction scene disagrees with manifest")
        try:
            score = float(row["score"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"{token}: a finite numeric score is required") from exc
        if not np.isfinite(score):
            raise ValueError(f"{token}: score must be finite")
        result = dict(row, scene_token=entry["scene_token"], score=score,
                      label=int(entry["label"]))
        if "prediction" in row:
            result["prediction"] = _binary_label(row["prediction"], token)
        by_token[token] = result
    missing = set(included) - set(by_token)
    if missing:
        raise ValueError(f"Missing predictions for {len(missing)} manifest tokens")
    aligned = [by_token[token] for token in included]
    for split in SPLITS:
        part = [r for r in aligned if r["split"] == split]
        if not part:
            raise ValueError(f"Empty {split} split")
        if len({r["label"] for r in part}) < 2:
            raise ValueError(f"{split} must contain both classes")
    predictions_present = ["prediction" in row for row in aligned]
    if any(predictions_present) and not all(predictions_present):
        raise ValueError("Fixed-rule prediction must be present for every row or none")
    return aligned


def select_operating_point(y, scores, target_precision: float) -> dict:
    """Maximise validation recall subject to the specified precision target.

    Ties prefer higher precision, then the higher threshold. The final PR point
    (predict nobody) has no threshold and is never treated as a feasible point.
    An unreachable target produces an explicit predict-none operating point.
    """
    if not 0 < target_precision <= 1:
        raise ValueError("target_precision must be in (0, 1]")
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if y.ndim != 1 or scores.shape != y.shape or not len(y):
        raise ValueError("Validation labels and scores must be nonempty aligned vectors")
    if not np.isfinite(scores).all() or set(y.tolist()) != {0, 1}:
        raise ValueError("Validation requires finite scores and both binary classes")
    precision, recall, thresholds = precision_recall_curve(y, scores)
    eligible = np.flatnonzero(precision[:-1] >= target_precision)
    result = {"selection_split": "val", "target_precision": float(target_precision),
              "selection_rule": "max recall, then precision, then threshold",
              "comparison": "score >= threshold"}
    if not len(eligible):
        return dict(result, status="target_unreachable", threshold=None,
                    validation_precision=0.0, validation_recall=0.0)
    best = max(eligible, key=lambda i: (recall[i], precision[i], thresholds[i]))
    return dict(result, status="selected", threshold=float(thresholds[best]),
                validation_precision=float(precision[best]),
                validation_recall=float(recall[best]))


def binary_metrics(y, prediction, score) -> dict:
    y, prediction, score = np.asarray(y), np.asarray(prediction), np.asarray(score)
    tp = int(((y == 1) & (prediction == 1)).sum())
    fp = int(((y == 0) & (prediction == 1)).sum())
    tn = int(((y == 0) & (prediction == 0)).sum())
    fn = int(((y == 1) & (prediction == 0)).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {
        "n": len(y), "tail_support": int((y == 1).sum()),
        "normal_support": int((y == 0).sum()),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-15),
        "ap": float(average_precision_score(y, score)) if (y == 1).any() else None,
        "auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
    }


def review_budget_metrics(rows: list[dict], review_k: int) -> dict:
    if review_k <= 0:
        raise ValueError("review_k must be positive")
    ranked = sorted(rows, key=lambda row: (-row["score"], row["sample_token"]))
    selected = ranked[:review_k]
    tails = sum(row["label"] for row in rows)
    found = sum(row["label"] for row in selected)
    prevalence = tails / len(rows)
    return {
        "requested_k": int(review_k), "effective_k": len(selected),
        "tie_break": "sample_token ascending", "tail_found": found,
        "precision_at_k": found / len(selected), "recall_at_k": found / max(tails, 1),
        "random_expected_tail_found": len(selected) * prevalence,
        "random_expected_precision_at_k": prevalence,
        "random_expected_recall_at_k": len(selected) / len(rows) if tails else 0.0,
        "selected_tokens": [row["sample_token"] for row in selected],
    }


def scene_bootstrap(rows: list[dict], *, seed: int, samples: int) -> dict:
    """Percentile intervals conditional on the frozen validation threshold.

    Entire test scenes, with all their frames, are sampled with replacement.
    Single-class resamples have undefined AUC; no-positive resamples have
    undefined AP. Undefined values are omitted with counts reported per metric.
    """
    if samples < 0:
        raise ValueError("bootstrap_samples must be nonnegative")
    scenes = sorted({row["scene_token"] for row in rows})
    result = {"unit": "scene", "seed": seed, "requested_samples": samples,
              "scene_count": len(scenes), "threshold_refitted": False,
              "confidence": 0.95, "intervals": {}}
    if not samples or len(scenes) < 2:
        return dict(result, status="disabled" if not samples else "too_few_scenes")
    grouped = [[row for row in rows if row["scene_token"] == scene] for scene in scenes]
    rng = np.random.default_rng(seed)
    values = {key: [] for key in ("ap", "auc", "precision", "recall", "f1")}
    for _ in range(samples):
        resample = [row for i in rng.integers(0, len(scenes), size=len(scenes))
                    for row in grouped[i]]
        metrics = binary_metrics([r["label"] for r in resample],
                                 [r["operating_prediction"] for r in resample],
                                 [r["score"] for r in resample])
        for key in values:
            if metrics[key] is not None:
                values[key].append(metrics[key])
    for key, numbers in values.items():
        result["intervals"][key] = {
            "lower": float(np.quantile(numbers, 0.025)) if numbers else None,
            "upper": float(np.quantile(numbers, 0.975)) if numbers else None,
            "valid_samples": len(numbers), "undefined_samples": samples - len(numbers),
        }
    return dict(result, status="computed")


def _source_metadata(output_dir: Path) -> dict:
    root = Path(__file__).resolve().parents[1]

    def git(*arguments):
        try:
            return subprocess.check_output(["git", "-C", str(root), *arguments],
                                           stderr=subprocess.DEVNULL).decode().strip()
        except (subprocess.SubprocessError, OSError):
            return None

    status = git("status", "--porcelain")
    patch = git("diff", "--binary", "HEAD")
    if patch:
        (output_dir / "source.patch").write_text(patch + "\n", encoding="utf-8")
    else:
        (output_dir / "source.patch").unlink(missing_ok=True)
    source_files = {str(path.relative_to(root)): file_sha256(path)
                    for path in sorted((root / "scripts").glob("*.py"))}
    snapshot = output_dir / "source_snapshot.zip"
    snapshot_paths = [*sorted((root / "scripts").glob("*.py")),
                      *sorted((root / "tests").glob("*.py")),
                      root / "requirements.txt", root / "requirements-cpu-lock.txt",
                      root / "Makefile"]
    with zipfile.ZipFile(snapshot, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in snapshot_paths:
            if path.is_file():
                archive.write(path, str(path.relative_to(root)))
    payload = json.dumps(source_files, sort_keys=True).encode()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(status) if status is not None
            else None, "git_status": status, "source_files_sha256": source_files,
            "source_tree_sha256": hashlib.sha256(payload).hexdigest(),
            "source_snapshot_sha256": file_sha256(snapshot),
            "tracked_diff_sha256": file_sha256(output_dir / "source.patch") if patch else None}


def _write_pr_curve(output_dir: Path, split: str, rows: list[dict]) -> None:
    precision, recall, thresholds = precision_recall_curve(
        [r["label"] for r in rows], [r["score"] for r in rows])
    with (output_dir / f"pr_curve_{split}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["precision", "recall", "threshold"])
        writer.writerows(zip(precision, recall, [*thresholds, None]))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        (output_dir / f"pr_curve_{split}.png").unlink(missing_ok=True)
        return
    fig, axes = plt.subplots()
    axes.step(recall, precision, where="post")
    axes.set(xlabel="Tail recall", ylabel="Tail precision", title=f"{split} PR curve",
             xlim=(0, 1), ylim=(0, 1.01))
    fig.savefig(output_dir / f"pr_curve_{split}.png", bbox_inches="tight")
    plt.close(fig)


def label_scope_from_manifest(manifest: list[dict]) -> dict | None:
    """Preserve AI-label limitations in every metric bundle and rebuilt table."""
    active = [row for row in manifest if row.get("split") in SPLITS]

    def is_ai(row):
        return (row.get("label_origin") == "ai_visual_provisional"
                or row.get("label_source") == "ai_visual_provisional"
                or row.get("review_status") == "ai_provisional")

    if not any(is_ai(row) for row in active):
        return None
    if not all(is_ai(row) and row.get("review_status") == "ai_provisional"
               and row.get("human_review_status") == "pending" for row in active):
        raise ValueError("AI pilot labels must remain consistently AI provisional "
                         "and human pending")
    return {"label_protocol": "ai-visual-pilot", "human_reviewed": False,
            "interpretation": "Agreement with provisional AI visual labels; "
                              "not human-ground-truth accuracy or a benchmark result"}


def write_evaluation_artifacts(output_dir: Path, rows: list[dict], manifest: list[dict], *,
                               config: dict, seed: int, checkpoint_path: Path | None = None,
                               input_paths: dict | None = None, review_k: int = 100,
                               target_precision: float = 0.9, bootstrap_samples: int = 1000,
                               fixed_rule: dict | None = None) -> dict:
    """Save reproducible metrics, PR curves and review queues from saved scores."""
    aligned = align_rows(rows, manifest)
    label_scope = label_scope_from_manifest(manifest)
    if review_k <= 0 or bootstrap_samples < 0:
        raise ValueError("review_k must be positive; bootstrap_samples must be nonnegative")
    val = [row for row in aligned if row["split"] == "val"]
    operating_point = select_operating_point([r["label"] for r in val],
                                             [r["score"] for r in val], target_precision)
    threshold = operating_point["threshold"]
    for row in aligned:
        row["operating_prediction"] = int(threshold is not None and row["score"] >= threshold)
        row["operating_error"] = (
            "fp" if row["operating_prediction"] == 1 and row["label"] == 0 else
            "fn" if row["operating_prediction"] == 0 and row["label"] == 1 else None)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation = {"protocol_version": PROTOCOL_VERSION, "seed": int(seed),
                  "score_direction": "higher means more likely tail",
                  "operating_point": operating_point, "fixed_rule": fixed_rule,
                  "split_summary": {}, "splits": {}}
    if label_scope is not None:
        evaluation["label_scope"] = label_scope
    for split in SPLITS:
        part = [r for r in aligned if r["split"] == split]
        evaluation["split_summary"][split] = {
            "n_frames": len(part), "n_scenes": len({r["scene_token"] for r in part}),
            "tail_support": sum(r["label"] for r in part),
            "normal_support": sum(r["label"] == 0 for r in part)}
    for split in ("val", "test"):
        part = [r for r in aligned if r["split"] == split]
        y, scores = [r["label"] for r in part], [r["score"] for r in part]
        metrics = {"scene_count": len({r["scene_token"] for r in part}),
                   "operating": binary_metrics(y, [r["operating_prediction"] for r in part],
                                               scores),
                   "review_budget": review_budget_metrics(part, review_k)}
        if "prediction" in part[0]:
            metrics["fixed"] = binary_metrics(y, [r["prediction"] for r in part], scores)
        evaluation["splits"][split] = metrics
        queue_path = output_dir / f"review_queue_{split}.csv"
        with queue_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "rank", "sample_token", "scene_token", "score", "operating_prediction",
                "reviewer_label", "review_notes"])
            writer.writeheader()
            by_token = {row["sample_token"]: row for row in part}
            for rank, token in enumerate(metrics["review_budget"]["selected_tokens"], 1):
                row = by_token[token]
                writer.writerow({"rank": rank, "sample_token": token,
                                 "scene_token": row["scene_token"], "score": row["score"],
                                 "operating_prediction": row["operating_prediction"]})
        _write_pr_curve(output_dir, split, part)
    test_rows = [row for row in aligned if row["split"] == "test"]
    evaluation["test_scene_bootstrap"] = scene_bootstrap(
        test_rows, seed=seed, samples=bootstrap_samples)
    _write_json(output_dir / "evaluation_manifest.json", manifest)
    _write_json(output_dir / "predictions.json", aligned)
    columns = ["sample_token", "scene_token", "split", "label", "score", "prediction",
               "operating_prediction", "operating_error"]
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(aligned)
    versions = {}
    for package in ("numpy", "scikit-learn", "scipy", "joblib", "threadpoolctl",
                    "torch", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    resolved = sorted({f"{dist.metadata['Name']}=={dist.version}"
                       for dist in importlib.metadata.distributions() if dist.metadata["Name"]})
    (output_dir / "requirements-resolved.txt").write_text(
        "\n".join(resolved) + "\n", encoding="utf-8")
    inputs = {str(name): {"path": str(path), "sha256": file_sha256(Path(path)),
                          "bytes": Path(path).stat().st_size}
              for name, path in (input_paths or {}).items()}
    checkpoint = ({"path": str(checkpoint_path), "sha256": file_sha256(checkpoint_path),
                   "bytes": Path(checkpoint_path).stat().st_size} if checkpoint_path else None)
    metadata = {"protocol_version": PROTOCOL_VERSION,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "config": config, "seed": int(seed), "python": platform.python_version(),
                "dependencies": versions, "source": _source_metadata(output_dir),
                "inputs": inputs, "checkpoint": checkpoint,
                "manifest_sha256": file_sha256(output_dir / "evaluation_manifest.json"),
                "predictions_sha256": file_sha256(output_dir / "predictions.json")}
    if label_scope is not None:
        metadata["label_scope"] = label_scope
    _write_json(output_dir / "run_metadata.json", metadata)
    evaluation["provenance"] = {"run_metadata": "run_metadata.json",
                                "manifest_sha256": metadata["manifest_sha256"],
                                "predictions_sha256": metadata["predictions_sha256"]}
    _write_json(output_dir / "evaluation.json", evaluation)
    return evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--review-k", type=int, required=True)
    parser.add_argument("--target-precision", type=float, default=0.9)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    config = json.loads(args.config.read_text()) if args.config else {}
    config["evaluation_command"] = vars(args)
    result = write_evaluation_artifacts(
        args.output_dir, json.loads(args.rows.read_text()), json.loads(args.manifest.read_text()),
        config=config, seed=args.seed, checkpoint_path=args.checkpoint,
        input_paths={"rows": args.rows, "manifest": args.manifest}, review_k=args.review_k,
        target_precision=args.target_precision, bootstrap_samples=args.bootstrap_samples)
    print(json.dumps(result["splits"]["test"], indent=2))


if __name__ == "__main__":
    main()
