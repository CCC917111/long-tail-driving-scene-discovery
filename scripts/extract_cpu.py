#!/usr/bin/env python3
"""Extract frozen pretrained ResNet18 six-camera features entirely on CPU.

This is a separate image-only reference protocol, not Cosmos/VLM replication.
It never downloads assets, creates labels or changes a prepared scene split.
The layer1 *_mean.npy and *_last_token.npy names are compatibility aliases of
the identical 3072-dimensional camera-concatenated vector; neither describes
token pooling or an MLP hook for this protocol. Only one process may write a cache.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import torch

import cpu_image_encoder as encoder
from data_manifest import validate_tokens
from extract import (CHECKPOINT_STATE, _write_json_atomic, load_checkpoint,
                     save_checkpoint)
from extraction_contract import freeze_contract, sha256
from prepare_driving import CAMERAS, TABLES, VERSION_RATIOS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare_inputs(args):
    """Verify frozen selection, source tables and every image before inference."""
    prepared, data_root = Path(args.prepared_dir).resolve(), Path(args.data_root).resolve()
    preparation = read_json(prepared / "prepare_metadata.json")
    require(preparation.get("schema_version") == "new-nuscenes-preparation-v1"
            and preparation.get("split_before_labels") is True,
            "A frozen prepare_driving.py preparation is required")
    version = preparation.get("source_version")
    require(version in VERSION_RATIOS, "Unsupported prepared source version")
    for name in ("raw_manifest.json", "selected_samples.json", "split_scenes.json", "rubric.md"):
        require(sha256(prepared / name) == preparation.get("prepared_files", {}).get(name),
                f"Prepared file changed after split freeze: {name}")
    raw = read_json(prepared / "raw_manifest.json")
    selected = read_json(prepared / "selected_samples.json")
    for rows, name in ((raw, "raw manifest"), (selected, "prepared selection")):
        require(isinstance(rows, list) and rows and all(isinstance(row, dict) for row in rows),
                f"Invalid {name}")
        validate_tokens(rows, source=name)
    eligible = {row["sample_token"]: row for row in raw if not row.get("exclusion_reason")}
    require(set(eligible) == {row["sample_token"] for row in selected},
            "Prepared selection differs from eligible raw manifest")
    split_rows = read_json(prepared / "split_scenes.json")
    require(isinstance(split_rows, list), "Invalid frozen scene assignments")
    splits = {row["scene_token"]: row["split"] for row in split_rows}
    require(len(splits) == len(split_rows), "Duplicate frozen scene assignments")
    metadata_hashes = {}
    for name in TABLES:
        path = data_root / version / f"{name}.json"
        digest = sha256(path)
        require(digest == preparation.get("source_metadata", {}).get(name, {}).get("sha256"),
                f"Source metadata changed after preparation: {name}")
        metadata_hashes[path.name] = digest
    contract_samples, image_paths = [], []
    for row in selected:
        token = row["sample_token"]
        source = eligible[token]
        require(all(source.get(key) == value for key, value in row.items()),
                f"Prepared selection metadata differs from raw manifest: {token}")
        require(isinstance(row.get("scene_token"), str) and row["scene_token"].strip()
                and row.get("split") == splits.get(row["scene_token"])
                and row.get("source_version") == version,
                f"Prepared selection has invalid scene/split/source: {token}")
        require(set(row.get("images", {})) == set(CAMERAS)
                and set(source.get("image_files", {})) == set(CAMERAS),
                f"Exactly six camera records are required: {token}")
        images, paths = {}, []
        for camera in CAMERAS:
            relative = Path(row["images"][camera])
            require(not relative.is_absolute() and ".." not in relative.parts
                    and len(relative.parts) == 2 and relative.parts[0] == camera,
                    f"Invalid prepared camera path: {token}/{camera}")
            path = (data_root / "samples" / relative).resolve()
            require(data_root in path.parents and path.is_file(),
                    f"Missing or outside-root camera image: {token}/{camera}")
            recorded = source["image_files"][camera]
            require(recorded.get("filename") == f"samples/{relative.as_posix()}",
                    f"Prepared camera filename mismatch: {token}/{camera}")
            digest = sha256(path)
            require(digest == recorded.get("sha256")
                    and path.stat().st_size == recorded.get("bytes"),
                    f"Image changed after preparation: {token}/{camera}")
            images[camera] = digest
            paths.append(path)
        contract_samples.append({"sample_token": token, "scene_token": row["scene_token"],
                                 "image_sha256": images})
        image_paths.append(paths)
    return preparation, selected, contract_samples, image_paths, metadata_hashes


def build_contract(args, preparation, samples, metadata_hashes):
    revision = encoder.verify_weights(args.weights)
    packages = {name: importlib.metadata.version(name) for name in ("torch", "numpy", "Pillow")}
    return {
        "schema": encoder.SCHEMA, "source_version": preparation["source_version"],
        "encoder": encoder.ENCODER, "model_revision": revision,
        "revision_status": "pinned official ImageNet checkpoint SHA256; not a Hub commit",
        "model_files_sha256": {encoder.WEIGHTS_NAME: revision},
        "metadata_sha256": metadata_hashes, "samples": samples,
        "camera_order": list(CAMERAS), "layers": encoder.LAYERS,
        "pooling": ["camera_concat"], "hook": encoder.HOOK,
        "storage_aliases": {
            "layer1_mlp_output_mean.npy": "camera_concat",
            "layer1_mlp_output_last_token.npy": "camera_concat (identical compatibility alias)",
        },
        "dtype": "float32", "device": "cpu", "threads": args.threads,
        "resize_short_side": encoder.RESIZE_SHORT_SIDE,
        "crop_size": encoder.CROP_SIZE, "interpolation": "PIL bilinear",
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "view_feature_dim": encoder.VIEW_FEATURE_DIM, "feature_dim": encoder.FEATURE_DIM,
        "packages": packages,
        "source_sha256": {name: sha256(Path(__file__).with_name(name))
                          for name in ("cpu_image_encoder.py", "extract_cpu.py",
                                       "extract.py", "extraction_contract.py")},
        "protocol_scope": "Image-only CPU reference experiment; not Cosmos/VLM reproduction",
    }


def write_summary(output, meta, selected_count, revision, *, status, started, new_rows, error=None):
    state_path = output / CHECKPOINT_STATE
    state = read_json(state_path) if state_path.is_file() else {}
    result = {
        "status": status, "feature_protocol": encoder.SCHEMA, "encoder": encoder.ENCODER,
        "device": "cpu", "elapsed_seconds": time.perf_counter() - started,
        "rows_in_cache": len(meta), "new_rows_this_invocation": new_rows,
        "selected_rows_total": selected_count, "remaining_rows": selected_count - len(meta),
        "selection_complete": selected_count == len(meta), "model_revision": revision,
        "extraction_config_sha256": sha256(output / "extraction_config.json"),
        "checkpoint_generation": state.get("generation"),
        "feature_files": {name: {"sha256": record["sha256"],
                                 "bytes": (output / name).stat().st_size}
                          for name, record in state.get("files", {}).items()
                          if name.endswith(".npy")},
        "provenance_limit": "Recorded input/weights hashes; not evidence of driving accuracy",
    }
    if error is not None:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    _write_json_atomic(output / "extraction_summary.json", result)
    return result


def extract(args):
    require(args.max_samples >= 0 and args.save_every > 0 and 1 <= args.threads <= 2,
            "max-samples must be nonnegative; save-every positive; threads between 1 and 2")
    started = time.perf_counter()
    preparation, selected, samples, image_paths, metadata_hashes = prepare_inputs(args)
    contract = build_contract(args, preparation, samples, metadata_hashes)
    output = Path(args.output_dir).resolve()
    freeze_contract(output, contract, args.resume)
    rows, meta, _ = load_checkpoint(output, encoder.LAYERS)
    require(len(meta) <= len(selected)
            and [row["sample_token"] for row in meta]
            == [row["sample_token"] for row in selected[:len(meta)]],
            "Cached rows are not an ordered prefix of the frozen prepared selection")
    if meta:
        mean, last = (np.stack(rows[(1, kind)]) for kind in ("mean", "last"))
        require(mean.shape == (len(meta), encoder.FEATURE_DIM) and np.array_equal(mean, last),
                "Cached CPU feature width/compatibility aliases differ from the frozen contract")
    for row in meta:
        source = selected[row["row_index"]]
        require(all(row.get(key) == source.get(key) for key in
                    ("sample_token", "scene_token", "scene_name", "timestamp", "images")),
                "Cached metadata differs from prepared selection")
    initial_count = len(meta)
    pending = list(zip(selected[initial_count:], image_paths[initial_count:]))
    if args.max_samples:
        pending = pending[:args.max_samples]
    success, error = False, None
    try:
        if pending:
            torch.set_num_threads(args.threads)
            model = encoder.load_encoder(args.weights)
            for index, (sample, paths) in enumerate(pending, 1):
                value = encoder.encode_views(model, paths)
                require(value.shape == (encoder.FEATURE_DIM,) and value.dtype == np.float32
                        and np.isfinite(value).all(), "Invalid CPU camera-concatenated feature")
                for kind in ("mean", "last"):
                    rows[(1, kind)].append(value.copy())
                keys = ("sample_token", "scene_token", "scene_name", "timestamp", "images")
                meta.append({**{key: sample[key] for key in keys},
                             "row_index": len(meta)})
                if index % args.save_every == 0:
                    save_checkpoint(output, encoder.LAYERS, rows, meta, [])
                print(f"[cpu-extract] {len(meta)}/{len(selected)} samples", flush=True)
        success = True
    except BaseException as exc:
        error = exc
        raise
    finally:
        save_checkpoint(output, encoder.LAYERS, rows, meta, [])
        summary = write_summary(output, meta, len(selected), contract["model_revision"],
                                status="completed" if success else "failed_or_interrupted",
                                started=started, new_rows=len(meta) - initial_count, error=error)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared-dir", "data-root", "weights", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0, help="0 completes the full selection")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    return parser.parse_args(argv)


def main(argv=None):
    report = extract(parse_args(argv))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
