#!/usr/bin/env python3
"""Validate a local driving-data handoff, then run four controlled CPU methods.

No downloads, model loading or split generation occur here. Validation checks
recorded provenance and file consistency; it cannot establish data authenticity
or the truth of a recorded manual judgment. Synthetic fixtures are software
checks, never real-driving experiment evidence.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

from data_manifest import load_frozen_manifest, validate_tokens
from extraction_contract import sha256
from prepare_driving import CAMERAS, TABLES, VERSION_RATIOS, import_labels
from sae_common import load_dataset

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path):
    if not path.is_file():
        raise ValueError(f"Required handoff file missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_object(path: Path) -> dict:
    value = read_json(path)
    require(isinstance(value, dict), f"{path.name} must be a JSON object")
    return value


def rows_by_token(rows, source: str) -> dict:
    require(isinstance(rows, list) and all(isinstance(row, dict) for row in rows),
            f"{source} must be a JSON array of records")
    validate_tokens(rows, source=source)
    return {row["sample_token"]: row for row in rows}


def normalise_sources(value, aliases: dict):
    """Only relocate recorded CSV paths when replaying a copied handoff."""
    if isinstance(value, dict):
        return {aliases.get(key, key): normalise_sources(item, aliases)
                for key, item in value.items()}
    if isinstance(value, list):
        return [normalise_sources(item, aliases) for item in value]
    return aliases.get(value, value) if isinstance(value, str) else value


def replay_import(prepared: Path, annotations: Path, metadata: dict,
                  label_protocol: str = "manual") -> list[dict]:
    if label_protocol == "ai-visual-pilot":
        from ai_pilot_labels import import_ai_labels

        evidence_key, source_name, importer = (
            "annotation_inputs_sha256", "AI annotation JSON", import_ai_labels)
    else:
        evidence_key, source_name, importer = "review_inputs_sha256", "Review CSV", import_labels
    evidence = metadata.get(evidence_key)
    require(isinstance(evidence, dict) and evidence,
            f"Label import has no {source_name.lower()} hashes")
    paths, aliases, resolved = [], {}, []
    for recorded, digest in evidence.items():
        require(isinstance(recorded, str) and isinstance(digest, str),
                f"Invalid recorded {source_name.lower()} provenance")
        name = Path(recorded).name
        candidates = [Path(recorded), annotations / name, annotations.parent / name]
        matching = [path.resolve() for path in candidates
                    if path.is_file() and sha256(path) == digest]
        require(bool(matching), f"{source_name} missing or hash mismatch: {recorded}; "
                f"copy its unchanged bytes to {annotations / name} or {annotations.parent / name}")
        path = matching[0]
        require(str(path) not in aliases, f"{source_name} sources resolve to the same file")
        paths.append(path)
        aliases[str(path)] = recorded
        resolved.append({"recorded_path": recorded, "resolved_path": str(path), "sha256": digest})
    with tempfile.TemporaryDirectory(prefix="driving-import-validation-") as temporary:
        replay = Path(temporary) / "annotations"
        importer(prepared, paths, replay)
        files = ("labels.json", "manifest.json", "excluded_rows.json", "import_metadata.json")
        for filename in files:
            expected = normalise_sources(read_json(replay / filename), aliases)
            require(read_json(annotations / filename) == expected,
                    f"Imported {filename} differs from replay of frozen review CSVs")
    return resolved


def validate_label_import(args, prepared: Path, annotations: Path, imported: dict) -> tuple:
    """Keep AI silver-label pilots opt-in and separate from manual judgments."""
    label_protocol = getattr(args, "label_protocol", "manual")
    require(label_protocol in ("manual", "ai-visual-pilot"), "Unsupported label protocol")
    if label_protocol == "ai-visual-pilot":
        from ai_pilot_labels import LABEL_SCOPE, SCHEMA

        require(imported.get("schema_version") == SCHEMA
                and imported.get("label_protocol") == label_protocol
                and imported.get("human_review_complete") is False
                and imported.get("label_scope") == LABEL_SCOPE,
                "AI pilot requires an explicit AI-provisional label import")
        identity = {"label_protocol": label_protocol, "label_scope": LABEL_SCOPE,
                    "human_reviewed": False, "human_ground_truth_verified": False}
    else:
        require(imported.get("schema_version") == "new-manual-label-import-v1",
                "Manual label protocol rejects AI-pilot or unsupported label imports; "
                "use --label-protocol ai-visual-pilot explicitly for AI silver labels")
        identity = {"label_protocol": "manual", "label_scope": "Recorded manual judgments",
                    "human_reviewed": True, "human_ground_truth_verified": False}
    require(imported.get("split_recomputed") is False, "Unfrozen or unsupported label import")
    require(imported.get("prepare_metadata_sha256") == sha256(prepared / "prepare_metadata.json")
            and imported.get("raw_manifest_sha256") == sha256(prepared / "raw_manifest.json"),
            "Label import references different preparation or raw manifest")
    return replay_import(prepared, annotations, imported, label_protocol), identity


def validate_feature_protocol(config: dict, args, version: str) -> dict:
    """Keep CPU image features and the existing VLM protocol explicitly separate."""
    protocol = getattr(args, "feature_protocol", "cosmos")
    require(protocol in ("cosmos", "cpu-resnet18"), "Unsupported feature protocol")
    schema = ("raw-six-camera-extraction-v1" if protocol == "cosmos"
              else "cpu-resnet18-six-camera-v1")
    require(config.get("schema") == schema and config.get("source_version") == version,
            "Extraction schema/source_version differs from preparation or selected protocol")
    if protocol == "cpu-resnet18":
        from cpu_image_encoder import WEIGHTS_SHA256

        require(args.layer == 1 and args.pooling == "mean",
                "CPU ResNet18 features require --layer 1 --pooling mean; "
                "the array stores ordered camera concatenation")
        require(isinstance(WEIGHTS_SHA256, str)
                and re.fullmatch(r"[0-9a-f]{64}", WEIGHTS_SHA256) is not None,
                "CPU ResNet18 official weights hash has not been pinned")
        expected = {
            "encoder": "resnet18-imagenet1k-v1", "layers": [1],
            "hook": "resnet18 avgpool; six-camera concatenation", "pooling": ["camera_concat"],
            "dtype": "float32", "view_feature_dim": 512, "feature_dim": 3072,
            "resize_short_side": 256, "crop_size": 224,
            "interpolation": "PIL bilinear", "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
            "storage_aliases": {
                "layer1_mlp_output_mean.npy": "camera_concat",
                "layer1_mlp_output_last_token.npy": "camera_concat (identical compatibility alias)",
            },
            "model_revision": WEIGHTS_SHA256,
            "model_files_sha256": {"resnet18-f37072fd.pth": WEIGHTS_SHA256},
        }
        for key, value in expected.items():
            require(config.get(key) == value, f"CPU ResNet18 extraction {key} mismatch")
        sources = config.get("source_sha256", {})
        require(isinstance(sources, dict)
                and {"cpu_image_encoder.py", "extract_cpu.py"} <= sources.keys()
                and all(isinstance(digest, str)
                        and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
                        for digest in sources.values()),
                "CPU ResNet18 extraction source hashes are missing or invalid")
        return {"feature_protocol": protocol, "encoder": expected["encoder"],
                "feature_semantics": "Six ordered 512-dimensional pretrained image embeddings "
                                     "concatenated to 3072 dimensions; separate from VLM features"}

    require(re.fullmatch(r"[0-9a-fA-F]{40}", config.get("model_revision", "")) is not None,
            "Extraction needs a pinned 40-character model revision")
    pooling = "mean_all_tokens" if args.pooling == "mean" else "last_token"
    require(pooling in config.get("pooling", []), "Requested pooling absent from extraction config")
    require(config.get("hook") == "1-indexed decoder layer MLP output", "Extraction hook mismatch")
    prompt = config.get("prompt")
    require(isinstance(prompt, str)
            and hashlib.sha256(prompt.encode()).hexdigest() == config.get("prompt_sha256"),
            "Extraction prompt hash mismatch")
    model_hashes = config.get("model_files_sha256", {})
    require(isinstance(model_hashes, dict) and "config.json" in model_hashes
            and any(name.endswith((".safetensors", ".bin")) for name in model_hashes)
            and all(re.fullmatch(r"[0-9a-f]{64}", value or "") for value in model_hashes.values()),
            "Extraction model file hashes are missing or invalid")
    return {"feature_protocol": protocol, "encoder": "pinned-vlm-decoder-mlp",
            "feature_semantics": "Frozen VLM decoder MLP output with configured token pooling"}


def validate_workflow(args) -> dict:
    """Read-only validation; use the extractor to resume/repair interrupted mirrors."""
    prepared, annotations, extracted = (
        Path(getattr(args, name)).resolve()
        for name in ("prepared_dir", "annotations_dir", "extract_dir"))
    require(args.layer > 0, "layer must be positive")
    preparation = read_object(prepared / "prepare_metadata.json")
    require(preparation.get("schema_version") == "new-nuscenes-preparation-v1",
            "Unsupported preparation schema")
    version = preparation.get("source_version")
    require(version in VERSION_RATIOS, "source_version must be v1.0-mini or v1.0-trainval")
    require(preparation.get("split_before_labels") is True
            and preparation.get("official_split_used") is False,
            "Preparation must record its frozen internal scene split")
    recorded_files = preparation.get("prepared_files", {})
    prepared_files = ("raw_manifest.json", "selected_samples.json", "split_scenes.json",
                      "rubric.md")
    for filename in prepared_files:
        path = prepared / filename
        require(path.is_file() and recorded_files.get(filename) == sha256(path),
                f"Prepared file changed after split freeze: {filename}")
    raw = rows_by_token(read_json(prepared / "raw_manifest.json"), "prepared raw manifest")
    eligible = {token: row for token, row in raw.items() if not row.get("exclusion_reason")}
    selected = rows_by_token(read_json(prepared / "selected_samples.json"), "prepared selection")
    require(set(selected) == set(eligible) and selected,
            "Prepared selection differs from raw manifest")
    scene_rows = read_json(prepared / "split_scenes.json")
    require(isinstance(scene_rows, list), "Frozen scene assignments must be an array")
    scene_splits = {row["scene_token"]: row["split"] for row in scene_rows}
    require(len(scene_splits) == len(scene_rows), "Duplicate frozen scene assignment")
    for token, row in eligible.items():
        require(row.get("source_version") == version
                and row.get("split") == scene_splits.get(row.get("scene_token")),
                f"Prepared raw scene/split/source mismatch for {token}")
        require(all(selected[token].get(key) == row.get(key) for key in selected[token]),
                f"Prepared selection metadata differs from raw manifest for {token}")

    imported = read_object(annotations / "import_metadata.json")
    reviews, label_identity = validate_label_import(args, prepared, annotations, imported)

    config_path = extracted / "extraction_config.json"
    config = read_object(config_path)
    state_path = extracted / "checkpoint_state.json"
    read_object(state_path)  # Require a committed checkpoint, never adopt legacy arrays.
    summary = read_object(extracted / "extraction_summary.json")
    require(summary.get("status") == "completed" and summary.get("selection_complete") is True,
            "Extraction selection is incomplete; finish/resume extraction before training")
    feature_identity = validate_feature_protocol(config, args, version)
    if feature_identity["feature_protocol"] == "cpu-resnet18":
        require(summary.get("feature_protocol") == config["schema"]
                and summary.get("encoder") == config["encoder"] and summary.get("device") == "cpu",
                "CPU extraction summary must identify its image encoder and CPU feature protocol")
    require(config.get("camera_order") == list(CAMERAS), "Extraction camera order mismatch")
    layers = config.get("layers")
    require(isinstance(layers, list) and layers and all(type(x) is int and x > 0 for x in layers)
            and len(layers) == len(set(layers)) and args.layer in layers,
            "Requested layer is absent or invalid in extraction config")
    for name in TABLES:
        require(config.get("metadata_sha256", {}).get(f"{name}.json")
                == preparation.get("source_metadata", {}).get(name, {}).get("sha256")
                and config.get("metadata_sha256", {}).get(f"{name}.json") is not None,
                f"Extraction source metadata differs from preparation: {name}")
    contract_rows = config.get("samples")
    contract = rows_by_token(contract_rows, "extraction contract")
    require(set(contract) == set(selected),
            "Extraction contract does not cover exact prepared selection")
    for token, row in contract.items():
        source = eligible[token]
        require(row.get("scene_token") == source.get("scene_token")
                and row.get("image_sha256") == {camera: source["image_files"][camera]["sha256"]
                                                for camera in CAMERAS},
                f"Extraction scene/image hash mismatch against prepared raw manifest: {token}")

    # This verifier is shared with load_checkpoint, without its mirror repair or pruning.
    from extract import _read_committed
    state, _, meta, _ = _read_committed(extracted, layers)
    if feature_identity["feature_protocol"] == "cpu-resnet18":
        require(state["files"]["layer1_mlp_output_mean.npy"]["sha256"]
                == state["files"]["layer1_mlp_output_last_token.npy"]["sha256"],
                "CPU ResNet18 compatibility arrays must contain identical camera concatenations")
    for name, record in state["files"].items():
        path = extracted / name
        require(path.is_file() and not path.is_symlink() and sha256(path) == record["sha256"],
                f"Extraction consumer hash/file mismatch: {name}; "
                "resume extract.py to repair mirrors")
    count = len(contract)
    require(len(meta) == count and summary.get("rows_in_cache") == count
            and summary.get("selected_rows_total") == count and summary.get("remaining_rows") == 0,
            "Extraction counts do not cover the complete prepared selection")
    require(summary.get("extraction_config_sha256") == state["contract_sha256"]
            and summary.get("model_revision") == config["model_revision"],
            "Extraction summary config/revision mismatch")
    if "checkpoint_generation" in summary:
        require(summary["checkpoint_generation"] == state["generation"],
                "Stale extraction summary generation")
    feature_names = {name for name in state["files"] if name.endswith(".npy")}
    require(set(summary.get("feature_files", {})) == feature_names,
            "Extraction summary feature inventory mismatch")
    for name in feature_names:
        record = summary["feature_files"][name]
        require(record.get("sha256") == state["files"][name]["sha256"]
                and record.get("bytes") == (extracted / name).stat().st_size,
                f"Extraction summary feature hash/size mismatch: {name}")
    require([row["sample_token"] for row in meta] == [row["sample_token"] for row in contract_rows],
            "Extraction metadata row order differs from pinned contract sample order")
    for row in meta:
        source = eligible[row["sample_token"]]
        keys = ("scene_token", "timestamp", "scene_name", "images")
        require(all(row.get(key) == source.get(key) for key in keys),
                f"Extraction metadata differs from prepared raw manifest: {row['sample_token']}")

    feature_name = f"layer{args.layer}_mlp_output_{args.pooling}.npy"
    inputs = {"features": extracted / feature_name, "meta": extracted / "meta.json",
              "labels": annotations / "labels.json", "manifest": annotations / "manifest.json",
              "feature_config": config_path}
    dataset_args = SimpleNamespace(features=inputs["features"], meta=inputs["meta"],
                                   labels=[inputs["labels"]], output_dir=None)
    with redirect_stdout(io.StringIO()):
        features, labels, _, retained = load_dataset(dataset_args)
    if feature_identity["feature_protocol"] == "cpu-resnet18":
        require(features.shape[1] == 3072,
                "CPU ResNet18 feature array must contain six concatenated 512-dimensional views")
    splits = load_frozen_manifest(inputs["manifest"], retained, labels)
    support = {}
    for split, indices in splits.items():
        require(set(labels[indices].astype(int).tolist()) == {0, 1},
                f"Frozen {split} split must contain both classes; do not re-split or tune on test")
        support[split] = {"normal": int((labels[indices] == 0).sum()),
                          "tail": int((labels[indices] == 1).sum()),
                          "scenes": len({retained[i]["scene_token"] for i in indices})}
    manifest = read_json(inputs["manifest"])
    for row in manifest:
        source = eligible[row["sample_token"]]
        require(row["split"] == source["split"] and row["scene_token"] == source["scene_token"]
                and row.get("source_version") == version,
                "Imported manifest changed frozen scene/split/source")
    require(imported.get("ready_for_current_training_contract") is True,
            "Label import was not ready for both-class training")
    return {"status": "validated", "schema": "real-driving-handoff-v1", "source_version": version,
            **feature_identity, **label_identity,
            "evaluation_scope": preparation["scope"], "official_benchmark": False,
            "provenance_limit": ("Recorded file provenance and review records; data authenticity "
                                 "and judgment truth are not established."),
            "selection_complete": True, "extracted_rows": count, "labelled_rows": len(labels),
            "feature_dim": int(features.shape[1]), "layer": args.layer,
            "pooling": ("camera_concat" if feature_identity["feature_protocol"] == "cpu-resnet18"
                        else args.pooling), "storage_pooling": args.pooling,
            "support": support,
            "review_csv_evidence": reviews if label_identity["label_protocol"] == "manual" else [],
            "ai_annotation_evidence": (reviews if label_identity["label_protocol"]
                                       == "ai-visual-pilot" else []),
            "checkpoint_generation": state["generation"],
            "inputs": {key: {"path": str(path), "sha256": sha256(path)}
                       for key, path in inputs.items()},
            "handoff_hashes": {"prepare_metadata": sha256(prepared / "prepare_metadata.json"),
                               "import_metadata": sha256(annotations / "import_metadata.json"),
                               "checkpoint_state": sha256(state_path),
                               "extraction_summary": sha256(extracted / "extraction_summary.json")}}


def train_workflow(args) -> dict:
    require(args.review_k > 0 and args.epochs > 0 and args.hidden_dim >= 2
            and 0 < args.k <= args.hidden_dim,
            "Invalid review budget, epochs, hidden dimension or k")
    output = Path(args.output_dir).resolve()
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
            f"Use a fresh, empty output directory: {output}")
    report = validate_workflow(args)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "validation.json", report)
    settings = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()}
    write_json(output / "runner_config.json",
               {**settings, "device": "cpu", "selection": "validation only; no test tuning",
                **{key: report[key] for key in
                   ("feature_protocol", "encoder", "feature_semantics", "label_protocol",
                    "label_scope", "human_reviewed", "human_ground_truth_verified")}})
    (output / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    inputs = report["inputs"]
    common = ["--features", inputs["features"]["path"], "--meta", inputs["meta"]["path"],
              "--labels", inputs["labels"]["path"], "--manifest", inputs["manifest"]["path"],
              "--feature-config", inputs["feature_config"]["path"], "--seed", str(args.seed),
              "--review-k", str(args.review_k), "--epochs", str(args.epochs),
              "--hidden-dim", str(args.hidden_dim)]
    commands, run_dirs = [], []
    for method in ("logistic", "mlp", "sae_abstopk_none", "sae_abstopk_norm"):
        destination = output / method
        run_dirs.append(destination)
        if method in ("logistic", "mlp"):
            argv = [sys.executable, str(ROOT / "scripts/supervised_baselines.py"),
                    "--method", method]
        else:
            argv = [sys.executable, str(ROOT / "scripts/sae_abstopk_tail_reward.py"),
                    "--device", "cpu", "--sparsity", "abstopk",
                    "--reward", method.rsplit("_", 1)[1],
                    "--k", str(args.k)]
        commands.append({"name": method,
                         "argv": [*argv, *common, "--output-dir", str(destination)]})
    commands.append({"name": "summarize", "argv": [
        sys.executable, str(ROOT / "scripts/summarize_runs.py"),
        "--runs", *map(str, run_dirs), "--output-dir", str(output / "comparison")]})
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1")
    for record in commands:
        record["command"] = shlex.join(record["argv"])
        record["log"] = str(output / f"{record['name']}.log")
        write_json(output / "commands.json", commands)
        print(f"[run] {record['name']}; log: {record['log']}", file=sys.stderr, flush=True)
        started = time.monotonic()
        with Path(record["log"]).open("w", encoding="utf-8") as log:
            result = subprocess.run(record["argv"], cwd=ROOT, env=environment, stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
        record.update(returncode=result.returncode, elapsed_seconds=time.monotonic() - started)
        write_json(output / "commands.json", commands)
        require(result.returncode == 0, f"{record['name']} failed; inspect {record['log']}")
    result = {"status": "completed", "output_dir": str(output),
              "methods": [path.name for path in run_dirs],
              "main_table": str(output / "comparison/main_table.csv"),
              "validation": str(output / "validation.json"),
              "commands": str(output / "commands.json"),
              "source_version": report["source_version"],
              **{key: report[key] for key in
                 ("feature_protocol", "encoder", "feature_semantics", "label_protocol",
                  "label_scope", "human_reviewed", "human_ground_truth_verified")},
              "evaluation_scope": report["evaluation_scope"],
              "provenance_limit": report["provenance_limit"]}
    write_json(output / "workflow_summary.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "train"):
        sub = subparsers.add_parser(command)
        for name in ("prepared-dir", "annotations-dir", "extract-dir"):
            sub.add_argument(f"--{name}", type=Path, required=True)
        sub.add_argument("--layer", type=int, default=28)
        sub.add_argument("--pooling", choices=("mean", "last_token"), default="mean")
        sub.add_argument("--feature-protocol", choices=("cosmos", "cpu-resnet18"), default="cosmos",
                         help="CPU ResNet18 requires its own cache and --layer 1 --pooling mean")
        sub.add_argument("--label-protocol", choices=("manual", "ai-visual-pilot"),
                         default="manual",
                         help="AI silver labels require explicit opt-in; not human-ground-truth "
                              "accuracy and not interchangeable with manual labels")
        if command == "validate":
            sub.add_argument("--output", type=Path)
        else:
            sub.add_argument("--output-dir", type=Path, required=True)
            sub.add_argument("--review-k", type=int, required=True)
            sub.add_argument("--seed", type=int, default=42)
            sub.add_argument("--epochs", type=int, default=10)
            sub.add_argument("--hidden-dim", type=int, default=256)
            sub.add_argument("--k", type=int, default=32)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = validate_workflow(args) if args.command == "validate" else train_workflow(args)
        status = 0
    except (ValueError, OSError, KeyError, TypeError, IndexError, AttributeError) as exc:
        report = {"status": "failed", "command": args.command, "error": str(exc),
                  "training_started": False if args.command == "validate" else None}
        status = 1
    if args.command == "validate" and args.output:
        try:
            write_json(args.output, report)
        except OSError as exc:
            report = {"status": "failed", "command": args.command, "error": str(exc),
                      "training_started": False}
            status = 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
