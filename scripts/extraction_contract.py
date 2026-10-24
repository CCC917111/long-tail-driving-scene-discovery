"""Identity and resume guards for fresh, raw-image feature extraction."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import tempfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_contract(args, samples: list[dict], metadata_dir: Path, cameras) -> dict:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.model_revision or ""):
        raise ValueError("Fresh extraction requires --model-revision: the immutable 40-character "
                         "commit used to download these local model files")
    model_dir = args.model_dir
    weights = sorted(model_dir.glob("*.safetensors")) + sorted(model_dir.glob("*.bin"))
    if not (model_dir / "config.json").is_file() or not weights:
        raise ValueError("Local model config/weights are missing; download the pinned model first")
    model_files = sorted({*weights, *model_dir.glob("*.json"), *model_dir.glob("*.txt"),
                          *model_dir.glob("*.model"), *model_dir.glob("*.jinja")})
    inputs = []
    seen = set()
    for sample in samples:
        token = sample.get("sample_token")
        if not token or token in seen or not sample.get("scene_token"):
            raise ValueError("Extraction samples need unique tokens and recorded scene tokens")
        seen.add(token)
        images = {cam: sha256(sample["channels"][cam]) for cam in cameras}
        inputs.append({"sample_token": token, "scene_token": sample["scene_token"],
                       "image_sha256": images})
    if not inputs:
        raise ValueError("No complete six-camera samples selected")
    packages = {}
    for name in ("torch", "transformers", "accelerate", "numpy", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "schema": "raw-six-camera-extraction-v1",
        "source_version": metadata_dir.name,
        "model_revision": args.model_revision.lower(),
        "revision_status": "user-supplied download revision; local files independently hashed",
        "model_files_sha256": {p.name: sha256(p) for p in model_files},
        "metadata_sha256": {p.name: sha256(p) for p in sorted(metadata_dir.glob("*.json"))},
        "samples": inputs, "camera_order": list(cameras), "layers": list(args.layers),
        "hook": "1-indexed decoder layer MLP output",
        "pooling": ["mean_all_tokens", "last_token"],
        "prompt": args.prompt,
        "prompt_sha256": hashlib.sha256(args.prompt.encode("utf-8")).hexdigest(),
        "dtype": args.dtype, "min_pixels": args.min_pixels, "max_pixels": args.max_pixels,
        "caption": args.caption, "max_new_tokens": args.max_new_tokens,
        "packages": packages,
        "source_sha256": {name: sha256(Path(__file__).with_name(name))
                          for name in ("extract.py", "annotate_normal_core.py",
                                       "extraction_contract.py")},
    }


def freeze_contract(output_dir: Path, contract: dict, resume: bool) -> None:
    path = output_dir / "extraction_config.json"
    has_rows = ((output_dir / "meta.json").exists()
                or (output_dir / "checkpoint_state.json").exists())
    if has_rows and not resume:
        raise ValueError("Existing extraction rows must be preserved; use a fresh output directory")
    if has_rows and not path.is_file():
        raise ValueError("Legacy extraction has no verified contract; preserve it and use a fresh "
                         "directory rather than inventing its provenance")
    if path.exists():
        if json.loads(path.read_text()) != contract:
            raise ValueError("Extraction contract changed; cannot mix model/prompt/images/settings "
                             "in one cache. Use a fresh output directory")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_dir,
                                         prefix=".extraction_config-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(contract, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
