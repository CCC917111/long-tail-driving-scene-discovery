#!/usr/bin/env python3
"""Rebuild a controlled-comparison table from traceable per-run bundles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate_runs import (align_rows, binary_metrics, file_sha256, label_scope_from_manifest,
                           review_budget_metrics, select_operating_point)


def collect_runs(run_dirs: list[Path]) -> tuple[list[dict], list[dict]]:
    rows, shared, seen, method_configs = [], None, set(), {}
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        evaluation = json.loads((run_dir / "evaluation.json").read_text())
        metadata = json.loads((run_dir / "run_metadata.json").read_text())
        for key, filename in (("manifest_sha256", "evaluation_manifest.json"),
                              ("predictions_sha256", "predictions.json")):
            actual = file_sha256(run_dir / filename)
            if metadata[key] != actual or evaluation["provenance"][key] != actual:
                raise ValueError(f"{run_dir}: modified {filename}; rebuild evaluation")
        if not {"features", "meta"} <= metadata["inputs"].keys() or not metadata.get("checkpoint"):
            raise ValueError(f"{run_dir}: comparison needs feature, meta and checkpoint hashes")
        manifest = json.loads((run_dir / "evaluation_manifest.json").read_text())
        label_scope = label_scope_from_manifest(manifest)
        if (evaluation.get("label_scope") != label_scope
                or metadata.get("label_scope") != label_scope):
            raise ValueError(f"{run_dir}: label-scope provenance differs from manifest")
        predictions = align_rows(json.loads((run_dir / "predictions.json").read_text()), manifest)
        val = [row for row in predictions if row["split"] == "val"]
        point = select_operating_point([r["label"] for r in val], [r["score"] for r in val],
                                       evaluation["operating_point"]["target_precision"])
        if point != evaluation["operating_point"]:
            raise ValueError(f"{run_dir}: operating point disagrees with saved validation scores")
        test_rows = [row for row in predictions if row["split"] == "test"]
        threshold = point["threshold"]
        operating_predictions = [int(threshold is not None and row["score"] >= threshold)
                                 for row in test_rows]
        operating = binary_metrics(
            [r["label"] for r in test_rows], operating_predictions, [r["score"] for r in test_rows])
        test = evaluation["splits"]["test"]
        budget = review_budget_metrics(test_rows, test["review_budget"]["requested_k"])
        fixed = (binary_metrics([r["label"] for r in test_rows],
                                [r["prediction"] for r in test_rows],
                                [r["score"] for r in test_rows])
                 if "prediction" in test_rows[0] else {})
        if (operating != test["operating"] or budget != test["review_budget"] or
                fixed != test.get("fixed", {}) or
                test["scene_count"] != len({r["scene_token"] for r in test_rows})):
            raise ValueError(f"{run_dir}: metrics disagree with saved test predictions")
        provenance = metadata["config"].get("feature_provenance", {})
        feature_protocol = provenance.get("schema", "unrecorded")
        encoder = provenance.get("encoder", "unrecorded")
        feature_semantics = provenance.get("hook", "unrecorded")
        feature_config_hash = metadata["inputs"].get("feature_config", {}).get("sha256")
        contract = (evaluation["protocol_version"], metadata["manifest_sha256"],
                    metadata["inputs"]["features"]["sha256"], budget["requested_k"],
                    metadata["inputs"]["meta"]["sha256"],
                    evaluation["operating_point"]["target_precision"],
                    metadata["source"]["source_tree_sha256"], feature_protocol,
                    encoder, feature_semantics, feature_config_hash)
        if shared is None:
            shared = contract
        elif shared != contract:
            raise ValueError("Runs differ in protocol, manifest, features, meta, "
                             "K, target, source or feature protocol")
        config = metadata["config"]
        method = config.get("method") or f"sae-{config.get('sparsity')}-{config.get('reward')}"
        keys = (("c", "max_iter") if method == "logistic" else
                ("hidden_dim", "epochs", "batch_size", "lr", "l2") if method == "mlp" else
                ("hidden_dim", "tail_ratio", "dropout", "k", "sparsity", "reward", "alpha",
                 "beta", "reward_coeff", "reward_max", "epochs", "batch_size", "lr",
                 "weight_decay", "act_threshold"))
        model_config = {key: config.get(key) for key in keys}
        config_hash = hashlib.sha256(json.dumps(model_config, sort_keys=True).encode()).hexdigest()
        if method in method_configs and method_configs[method] != config_hash:
            raise ValueError(f"{method}: model/training configuration differs across seeds")
        method_configs[method] = config_hash
        seed = metadata["seed"]
        if (method, seed) in seen:
            raise ValueError(f"Duplicate method/seed: {method}/{seed}")
        seen.add((method, seed))
        point = operating
        rows.append({
            "method": method, "seed": seed, "run_dir": str(run_dir), "n": point["n"],
            "label_protocol": (label_scope["label_protocol"] if label_scope
                               else "recorded-reference"),
            "human_reviewed": False if label_scope else None,
            "feature_protocol": feature_protocol, "encoder": encoder,
            "feature_semantics": feature_semantics,
            "feature_config_sha256": feature_config_hash,
            "test_scenes": test["scene_count"], "tail_support": point["tail_support"],
            "ap": point["ap"], "auc": point["auc"],
            "fixed_precision": fixed.get("precision"), "fixed_recall": fixed.get("recall"),
            "fixed_f1": fixed.get("f1"), "operating_precision": point["precision"],
            "operating_recall": point["recall"], "operating_f1": point["f1"],
            "validation_threshold": evaluation["operating_point"]["threshold"],
            "target_status": evaluation["operating_point"]["status"],
            "requested_k": budget["requested_k"], "effective_k": budget["effective_k"],
            "precision_at_k": budget["precision_at_k"], "tail_found": budget["tail_found"],
            "random_expected_tail_found": budget["random_expected_tail_found"],
            "commit": metadata["source"]["commit"], "dirty": metadata["source"]["dirty"],
            "source_sha256": metadata["source"]["source_tree_sha256"],
            "model_config_sha256": config_hash,
            "checkpoint_sha256": metadata["checkpoint"]["sha256"],
            "manifest_sha256": metadata["manifest_sha256"],
            "features_sha256": metadata["inputs"]["features"]["sha256"],
            "meta_sha256": metadata["inputs"]["meta"]["sha256"],
        })
    summaries = []
    for method in sorted({row["method"] for row in rows}):
        group = [row for row in rows if row["method"] == method]
        summary = {"method": method, "n_seeds": len(group),
                   "label_protocol": group[0]["label_protocol"],
                   "human_reviewed": group[0]["human_reviewed"],
                   "seeds": ",".join(str(row["seed"]) for row in group)}
        for metric in ("ap", "fixed_precision", "fixed_recall", "fixed_f1",
                       "operating_precision", "operating_recall", "operating_f1",
                       "precision_at_k", "tail_found"):
            values = [row[metric] for row in group if row[metric] is not None]
            summary[f"{metric}_mean"] = float(np.mean(values)) if values else None
            summary[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summaries.append(summary)
    return rows, summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, summaries = collect_runs(args.runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in (("main_table", rows), ("seed_summary", summaries)):
        with (args.output_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(content[0]))
            writer.writeheader()
            writer.writerows(content)
        (args.output_dir / f"{name}.json").write_text(
            json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Rebuilt {len(rows)} run rows and {len(summaries)} method summaries")


if __name__ == "__main__":
    main()
