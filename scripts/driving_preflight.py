#!/usr/bin/env python3
"""Inspect local prerequisites without downloading data or provisioning compute."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

TABLES = ("sample.json", "sample_data.json", "scene.json", "sensor.json",
          "calibrated_sensor.json")
GPU_GUIDANCE = "https://github.com/nvidia-cosmos/cosmos-reason1#inference"


def inspect_environment(data_root: Path, version: str, model_dir: Path) -> dict:
    metadata = data_root / version
    missing_tables = [name for name in TABLES if not (metadata / name).is_file()]
    packages = {}
    for name in ("torch", "transformers", "accelerate", "huggingface_hub", "Pillow", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = {"available": False, "devices": [], "probe_error": None}
    try:
        import torch
        gpu["available"] = torch.cuda.is_available()
        if gpu["available"]:
            gpu["devices"] = [{"name": torch.cuda.get_device_properties(i).name,
                               "memory_gib": round(torch.cuda.get_device_properties(i).total_memory
                                                   / 2 ** 30, 2)}
                              for i in range(torch.cuda.device_count())]
    except (ImportError, RuntimeError, OSError) as exc:
        gpu["probe_error"] = f"{type(exc).__name__}: {exc}"
    model_files = sorted(model_dir.glob("*.safetensors")) + sorted(model_dir.glob("*.bin"))
    missing_shards, index_error = [], None
    for index in model_dir.glob("*.index.json"):
        try:
            references = set(json.loads(index.read_text())["weight_map"].values())
            missing_shards.extend(name for name in references if not (model_dir / name).is_file())
        except (ValueError, KeyError, TypeError, OSError) as exc:
            index_error = f"{index.name}: {type(exc).__name__}"
    model_present = ((model_dir / "config.json").is_file() and bool(model_files)
                     and not missing_shards and not index_error)
    missing_gpu_packages = [name for name in ("torch", "transformers", "accelerate")
                            if packages[name] is None]
    gpu_prerequisites = gpu["available"] and model_present and not missing_gpu_packages
    result = {
        "kind": "driving-local-prerequisite-check", "python": sys.version.split()[0],
        "data": {"root": str(data_root.resolve()), "version": version,
                 "metadata_files_missing": missing_tables,
                 "samples_directory_exists": (data_root / "samples").is_dir(),
                 "verified_real_data": False},
        "model": {"directory": str(model_dir.resolve()), "files_present": model_present,
                  "missing_weight_shards": sorted(set(missing_shards)),
                  "index_error": index_error, "weights_verified": False},
        "packages": packages, "gpu": gpu,
        "disk_free_gib": round(shutil.disk_usage(Path.cwd()).free / 2 ** 30, 2),
        "ready_for_raw_data_validation": not missing_tables and (data_root / "samples").is_dir(),
        "gpu_extraction_prerequisites_present": gpu_prerequisites,
        "missing_gpu_packages": missing_gpu_packages,
        "guidance": {"source": GPU_GUIDANCE,
                     "upstream_minimum": "one GPU with at least 24 GB memory",
                     "six_camera_fit_verified": False,
                     "next_gpu_check": "one frame; measure peak VRAM and elapsed time first"},
        "no_download_or_compute_purchase_performed": True,
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("dataset/nuscenes"))
    parser.add_argument("--version", choices=("v1.0-mini", "v1.0-trainval"), default="v1.0-mini")
    parser.add_argument("--model-dir", type=Path, default=Path("models/Cosmos-Reason1-7B"))
    parser.add_argument("--output", type=Path, default=Path("output/from_zero/preflight.json"))
    parser.add_argument("--require", choices=("data", "gpu", "all"),
                        help="Exit 2 if prerequisites are missing; this does not validate data")
    args = parser.parse_args(argv)
    report = inspect_environment(args.data_root, args.version, args.model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    requirements = {"data": report["ready_for_raw_data_validation"],
                    "gpu": report["gpu_extraction_prerequisites_present"]}
    if args.require and not all(requirements[k] for k in
                                (("data", "gpu") if args.require == "all" else (args.require,))):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
