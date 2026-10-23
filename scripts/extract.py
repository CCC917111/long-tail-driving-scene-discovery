#!/usr/bin/env python3
"""Extract VLM hidden-state representations for nuScenes keyframes.

For every keyframe sample, all 6 synchronized camera images are passed to the
VLM (default: Cosmos-Reason1-7B) together with an open-ended scene-description
prompt (see docs/prompt_templates.md). The VLM is *not* asked to classify the
scene. Forward hooks capture the output of the MLP block (i.e. the output of
``mlp.down_proj``) of one or more decoder layers, and two token summaries are
stored per layer:

  - mean over all tokens      -> layer{L}_mlp_output_mean.npy        (N, hidden)
  - last (prompt-final) token -> layer{L}_mlp_output_last_token.npy  (N, hidden)

Layer numbers are 1-indexed decoder blocks, so for the 28-block language model
of Cosmos-Reason1-7B ``--layers 28`` is the last block.

``meta.json`` (row-aligned with the arrays) stores sample_token, scene_token,
scene_name, timestamp and the relative path of each camera image, so that the
SAE scripts can join labels and split the data by scene and the neuron explorer
can show the frames. Optionally (``--caption``) the generated scene description
is written to ``captions.jsonl``.

What one sample is
------------------
* One keyframe: a single timestamp, no temporal context. nuScenes keyframes are
  annotated at 2 Hz; the intermediate sweeps are not used.
* Exactly six synchronized views, always in the order CAM_FRONT,
  CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT.
  Keyframes with a missing view are skipped, and the prompt names the views in
  this order, so the order is part of the input.
* Each view is resized by the VLM processor to between ``--min-pixels`` and
  ``--max-pixels`` pixels (3,136 to 1,600,000). A 1600 x 900 nuScenes image
  (1.44 MP) is used at full resolution.
* One forward pass per sample with the six images and the scene prompt; the
  generated text is not needed for the features.

Sample discovery reuses ``ensure_meta`` / ``build_samples`` from
scripts/annotate_normal_core.py, so both scripts see exactly the same set of
keyframes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import uuid
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from annotate_normal_core import CAMERAS, build_samples, ensure_meta, load_vlm
from extraction_contract import build_contract, freeze_contract, sha256

SCENE_PROMPT = """You are given 6 synchronized camera images from the ego vehicle at a single timestamp. The cameras are provided in this order: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT.

Assume that you are the driver of the ego vehicle and need to understand the surrounding driving environment from these multi-camera views.

Your task is to provide a concise driving-scene description that covers the following aspects:
- Overall scene type, such as urban road, highway, residential area, intersection, or parking area.
- Weather and lighting conditions, including rain, fog, snow, night, glare, low visibility, or wet reflective roads.
- Road layout and traffic conditions, including lane structure, intersections, congestion, road boundaries, and traffic flow.
- Dynamic objects, including vehicles, pedestrians, cyclists, scooters, and other vulnerable road users.
- Potential hazards or important interactions that may affect the ego vehicle.
- Any visible risky situations or behaviors that may require cautious driving.

Provide the description in one paragraph with fewer than 150 words."""


def parse_args() -> argparse.Namespace:
    proj = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples-root", type=Path,
                   default=proj / "dataset" / "nuscenes" / "nuscenes_extracted" / "samples",
                   help="Directory containing CAM_FRONT/, CAM_FRONT_LEFT/, ... with jpgs.")
    p.add_argument("--meta-tgz", type=Path,
                   default=proj / "dataset" / "nuscenes" / "v1.0-trainval_meta.tgz")
    p.add_argument("--meta-extract-dir", type=Path, default=proj / "dataset" / "nuscenes")
    p.add_argument("--meta-dir", type=Path,
                   help="Already extracted v1.0-mini/ or v1.0-trainval/ metadata directory")
    p.add_argument("--model-dir", type=Path, default=proj / "models" / "Cosmos-Reason1-7B")
    p.add_argument("--model-revision", required=True,
                   help="Immutable 40-character upstream commit used to download local weights")
    p.add_argument("--output-dir", type=Path, default=proj / "output" / "extract")
    p.add_argument("--layers", type=int, nargs="+", default=[28],
                   help="1-indexed decoder layers whose MLP output is stored")
    p.add_argument("--prompt", type=str, default=SCENE_PROMPT)
    p.add_argument("--sample-tokens", type=Path, default=None,
                   help="Optional JSON (e.g. the annotation file) restricting extraction to "
                        "the sample_tokens it contains")
    p.add_argument("--caption", action="store_true",
                   help="Also generate and save the scene description text (slower)")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--min-pixels", type=int, default=3136)
    p.add_argument("--max-pixels", type=int, default=1_600_000)
    p.add_argument("--max-samples", type=int, default=0, help="0 means all samples")
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--resume", action="store_true", default=True,
                   help="Enabled by default. Pass --no-resume to start fresh.")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    return p.parse_args()


def array_paths(output_dir: Path, layer: int) -> dict[str, Path]:
    return {
        "mean": output_dir / f"layer{layer}_mlp_output_mean.npy",
        "last": output_dir / f"layer{layer}_mlp_output_last_token.npy",
    }


CHECKPOINT_SCHEMA = "extraction-checkpoint-v1"
CHECKPOINT_STATE = "checkpoint_state.json"


def _sync_directory(path: Path) -> None:
    """Persist directory entry changes on the local filesystem."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_state(output_dir: Path, state: dict) -> None:
    """The atomic state replacement is the checkpoint's only commit point."""
    _write_json_atomic(output_dir / CHECKPOINT_STATE, state)


def _generation_path(output_dir: Path, relative: str) -> Path:
    parts = Path(relative).parts if isinstance(relative, str) else ()
    if (len(parts) != 2 or parts[0] != ".checkpoints"
            or not re.fullmatch(r"[0-9a-f]{32}", parts[1])):
        raise ValueError("Cannot resume: invalid checkpoint generation path")
    generation = output_dir / relative
    if generation.is_symlink() or generation.resolve().parent != (
            output_dir.resolve() / ".checkpoints"):
        raise ValueError("Cannot resume: checkpoint generation escapes output directory")
    return generation


def _validate_metadata(meta, captions) -> None:
    if not isinstance(meta, list) or not meta:
        raise ValueError("Checkpoint metadata must contain at least one sample")
    tokens = []
    for index, row in enumerate(meta):
        if (not isinstance(row, dict)
                or not isinstance(row.get("sample_token"), str)
                or not row["sample_token"].strip()
                or not isinstance(row.get("scene_token"), str)
                or not row["scene_token"].strip()
                or row.get("row_index", index) != index):
            raise ValueError("Checkpoint metadata has invalid tokens or row indices")
        tokens.append(row["sample_token"])
    if len(set(tokens)) != len(tokens):
        raise ValueError("Checkpoint metadata contains duplicate sample tokens")
    if not isinstance(captions, list):
        raise ValueError("Checkpoint captions must be a list")
    if captions:
        if (any(not isinstance(row, dict) or not isinstance(row.get("caption"), str)
                for row in captions)
                or [row.get("sample_token") for row in captions] != tokens):
            raise ValueError("Checkpoint captions must match metadata tokens and order")


def _validate_array(array, row_count: int, width: int | None = None) -> None:
    if (array.ndim != 2 or array.shape[0] != row_count or array.shape[1] < 1
            or array.dtype != np.dtype("float32")
            or (width is not None and array.shape[1] != width)
            or not np.isfinite(array).all()):
        raise ValueError("Checkpoint feature matrix has invalid shape, dtype or nonfinite values")


def _expected_checkpoint_names(layers: list[int], captions: bool) -> set[str]:
    names = {path.name for layer in layers
             for path in array_paths(Path(), layer).values()}
    return names | {"meta.json"} | ({"captions.jsonl"} if captions else set())


def _read_committed(output_dir: Path, layers: list[int]):
    """Verify authoritative generation before trusting any consumer mirrors."""
    state = json.loads((output_dir / CHECKPOINT_STATE).read_text(encoding="utf-8"))
    if (state.get("schema") != CHECKPOINT_SCHEMA or state.get("layers") != layers
            or type(state.get("row_count")) is not int or state["row_count"] < 1
            or type(state.get("has_captions")) is not bool):
        raise ValueError("Cannot resume: invalid checkpoint state or changed --layers")
    generation = _generation_path(output_dir, state.get("generation"))
    contract_path = output_dir / "extraction_config.json"
    if (not contract_path.is_file()
            or sha256(contract_path) != state.get("contract_sha256")):
        raise ValueError("Cannot resume: checkpoint extraction contract hash mismatch")
    names = _expected_checkpoint_names(layers, state["has_captions"])
    files = state.get("files")
    if not isinstance(files, dict) or set(files) != names:
        raise ValueError("Cannot resume: checkpoint file inventory mismatch")
    for name, record in files.items():
        path = generation / name
        if (not isinstance(record, dict)
                or record.get("path") != f"{state['generation']}/{name}"
                or not path.is_file() or path.is_symlink()
                or sha256(path) != record.get("sha256")):
            raise ValueError(f"Cannot resume: checkpoint hash/file mismatch for {name}")
    meta = json.loads((generation / "meta.json").read_text(encoding="utf-8"))
    captions = []
    if state["has_captions"]:
        captions = [json.loads(line) for line in
                    (generation / "captions.jsonl").read_text(encoding="utf-8").splitlines()]
    _validate_metadata(meta, captions)
    if len(meta) != state["row_count"]:
        raise ValueError("Cannot resume: checkpoint metadata row count mismatch")
    widths = state.get("feature_widths")
    if not isinstance(widths, dict) or set(widths) != {str(layer) for layer in layers}:
        raise ValueError("Cannot resume: checkpoint feature width inventory mismatch")
    rows = {}
    for layer in layers:
        width = widths[str(layer)]
        if type(width) is not int or width < 1:
            raise ValueError("Cannot resume: invalid checkpoint feature width")
        for kind, path in array_paths(generation, layer).items():
            array = np.load(path, allow_pickle=False)
            _validate_array(array, len(meta), width)
            rows[(layer, kind)] = list(array)
    return state, rows, meta, captions


def _mirror_checkpoint(output_dir: Path, state: dict) -> None:
    """Export consumer paths; a later resume repairs an interrupted export."""
    generation = _generation_path(output_dir, state["generation"])
    for name, record in state["files"].items():
        target = output_dir / name
        if target.is_file() and not target.is_symlink() and sha256(target) == record["sha256"]:
            continue
        temporary = output_dir / f".{name}.{uuid.uuid4().hex}.tmp"
        try:
            with (generation / name).open("rb") as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    if not state["has_captions"]:
        (output_dir / "captions.jsonl").unlink(missing_ok=True)
    _sync_directory(output_dir)


def _prune_generations(output_dir: Path, state: dict) -> None:
    """Bound disk usage after successful export; one writer per cache is required."""
    keep = {state["generation"], state.get("previous_generation")}
    for path in (output_dir / ".checkpoints").iterdir():
        if (path.is_dir() and not path.is_symlink()
                and re.fullmatch(r"[0-9a-f]{32}", path.name)
                and f".checkpoints/{path.name}" not in keep):
            shutil.rmtree(path)


def save_checkpoint(output_dir: Path, layers: list[int], rows: dict, meta: list[dict],
                    captions: list[dict]) -> None:
    """Publish immutable data first, atomic pointer second, consumer mirrors last.

    Current and previous committed generations are retained. Use a single writer
    per cache; consumer files are for readers after extraction/resume completes.
    """
    if not meta:
        return
    _validate_metadata(meta, captions)
    contract_path = output_dir / "extraction_config.json"
    if not contract_path.is_file():
        raise ValueError("Cannot checkpoint without a verified extraction contract")
    previous = None
    state_path = output_dir / CHECKPOINT_STATE
    if state_path.exists():
        previous_state, _, _, _ = _read_committed(output_dir, layers)
        previous = previous_state["generation"]
    elif ((output_dir / "meta.json").exists()
          or any(output_dir.glob("layer*_mlp_output_*.npy"))
          or (output_dir / "captions.jsonl").exists()):
        raise ValueError("Untracked legacy checkpoint; preserve it and use a fresh directory")
    arrays, widths = {}, {}
    for layer in layers:
        for kind, path in array_paths(Path(), layer).items():
            array = np.stack(rows[(layer, kind)]).astype(np.float32)
            _validate_array(array, len(meta), widths.get(str(layer)))
            widths[str(layer)] = array.shape[1]
            arrays[path.name] = array
    relative = f".checkpoints/{uuid.uuid4().hex}"
    generation = _generation_path(output_dir, relative)
    generation.mkdir(parents=True)
    for name, array in arrays.items():
        with (generation / name).open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
    with (generation / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=1, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    if captions:
        with (generation / "captions.jsonl").open("w", encoding="utf-8") as handle:
            for row in captions:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    _sync_directory(generation)
    _sync_directory(generation.parent)
    state = {
        "schema": CHECKPOINT_SCHEMA, "generation": relative,
        "previous_generation": previous, "layers": layers, "row_count": len(meta),
        "feature_widths": widths, "has_captions": bool(captions),
        "contract_sha256": sha256(contract_path),
        "files": {name: {"path": f"{relative}/{name}", "sha256": sha256(generation / name)}
                  for name in _expected_checkpoint_names(layers, bool(captions))},
    }
    _publish_state(output_dir, state)
    _mirror_checkpoint(output_dir, state)
    _prune_generations(output_dir, state)


def load_checkpoint(output_dir: Path, layers: list[int]):
    """Restore only a hash-verified committed generation, repairing root mirrors."""
    rows = {(layer, kind): [] for layer in layers for kind in ("mean", "last")}
    if not (output_dir / CHECKPOINT_STATE).exists():
        if ((output_dir / "meta.json").exists()
                or any(output_dir.glob("layer*_mlp_output_*.npy"))
                or (output_dir / "captions.jsonl").exists()):
            raise ValueError("Untracked legacy checkpoint; preserve it and use a fresh directory")
        return rows, [], []
    state, rows, meta, captions = _read_committed(output_dir, layers)
    _mirror_checkpoint(output_dir, state)
    _prune_generations(output_dir, state)
    print(f"[resume] verified {len(meta)} rows from {state['generation']}")
    return rows, meta, captions


def _refresh_complete_summary(args, meta: list[dict], selected_count: int) -> None:
    """Avoid leaving a stale partial summary after recovering a completed export."""
    path = args.output_dir / "extraction_summary.json"
    summary = {}
    if path.is_file():
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            pass
    state = json.loads((args.output_dir / CHECKPOINT_STATE).read_text(encoding="utf-8"))
    summary.update({
        "status": "completed", "cache_verified_on_resume": True,
        "rows_in_cache": len(meta), "new_rows_this_invocation": 0,
        "selected_rows_total": selected_count, "remaining_rows": selected_count - len(meta),
        "selection_complete": selected_count == len(meta),
        "model_revision": args.model_revision,
        "checkpoint_generation": state["generation"],
        "extraction_config_sha256": state["contract_sha256"],
        "feature_files": {name: {"sha256": record["sha256"],
                                 "bytes": (args.output_dir / name).stat().st_size}
                          for name, record in state["files"].items() if name.endswith(".npy")},
        "timing_scope": "retained prior-invocation timings; this invocation verified cache only",
    })
    _write_json_atomic(path, summary)


def get_decoder_layers(model):
    """Return the list of language-model decoder layers across transformers versions."""
    for path in ("model.language_model.layers", "model.layers", "language_model.model.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("Could not locate the decoder layers of this model")


def main() -> None:
    args = parse_args()
    if (not args.layers or len(set(args.layers)) != len(args.layers)
            or any(layer < 1 for layer in args.layers)):
        raise ValueError("layers must contain unique positive decoder layer indices")
    if args.save_every < 1 or args.max_samples < 0:
        raise ValueError("save-every must be positive and max-samples nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meta_dir = args.meta_dir or ensure_meta(args.meta_tgz, args.meta_extract_dir)
    samples = build_samples(meta_dir, args.samples_root)
    if args.sample_tokens is not None:
        with args.sample_tokens.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rows = data["samples"] if isinstance(data, dict) else data
        tokens = [r.get("sample_token") for r in rows]
        if not tokens or any(not isinstance(t, str) or not t for t in tokens):
            raise ValueError("sample-tokens must contain nonempty sample_token entries")
        if len(set(tokens)) != len(tokens):
            raise ValueError("sample-tokens contains duplicate tokens")
        wanted = set(tokens)
        samples = [s for s in samples if s["sample_token"] in wanted]
        absent = wanted - {s["sample_token"] for s in samples}
        if absent:
            raise ValueError(f"{len(absent)} selected samples lack metadata or six camera files")
        print(f"[plan] restricted to {len(samples)} samples listed in {args.sample_tokens}")

    contract = build_contract(args, samples, meta_dir, CAMERAS)
    freeze_contract(args.output_dir, contract, args.resume)

    rows, meta, captions = (load_checkpoint(args.output_dir, args.layers) if args.resume
                            else ({(layer, k): [] for layer in args.layers for k in ("mean", "last")},
                                  [], []))
    done = {m["sample_token"] for m in meta}
    selected = {s["sample_token"] for s in samples}
    if len(done) != len(meta) or not done.issubset(selected):
        raise ValueError("Cached metadata contains duplicate or unselected sample tokens")
    pending = [s for s in samples if s["sample_token"] not in done]
    if args.max_samples > 0:
        pending = pending[: args.max_samples]
    print(f"[plan] total={len(samples)} done={len(done)} to_process={len(pending)}")
    if not pending:
        _refresh_complete_summary(args, meta, len(samples))
        return

    started = time.perf_counter()
    is_cuda = str(args.device).startswith("cuda")
    if is_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU unavailable; run driving_preflight.py before extraction")
        torch.cuda.reset_peak_memory_stats(args.device)
    try:
        processor, model = load_vlm(args.model_dir, args.device, args.dtype,
                                    args.min_pixels, args.max_pixels)
    except Exception as exc:
        failed = {"status": "failed", "phase": "model_load", "error": str(exc),
                  "elapsed_seconds": time.perf_counter() - started,
                  "model_revision": args.model_revision, "device": args.device,
                  "new_rows_this_invocation": 0, "rows_in_cache": len(meta),
                  "selection_complete": False,
                  "peak_cuda_allocated_bytes": (torch.cuda.max_memory_allocated(args.device)
                                                if is_cuda else None),
                  "peak_cuda_reserved_bytes": (torch.cuda.max_memory_reserved(args.device)
                                               if is_cuda else None)}
        (args.output_dir / "extraction_summary.json").write_text(json.dumps(failed, indent=2))
        raise
    model.eval()
    model_load_seconds = time.perf_counter() - started
    # From transformers 4.52 on, `model.model` is the full vision-language backbone;
    # in older versions it is the text-only decoder, so the full model is used.
    inner = getattr(model, "model", None)
    backbone = inner if inner is not None and hasattr(inner, "visual") else model
    decoder_layers = get_decoder_layers(model)
    n_layers = len(decoder_layers)
    for layer in args.layers:
        if not 1 <= layer <= n_layers:
            raise ValueError(f"--layers must be in [1, {n_layers}], got {layer}")

    captures: dict[int, torch.Tensor] = {}

    def make_hook(layer: int):
        def hook(_module, _inputs, output):
            captures[layer] = output.detach()
        return hook

    handles = [decoder_layers[layer - 1].mlp.register_forward_hook(make_hook(layer))
               for layer in args.layers]

    t_start = time.time()
    success = False
    try:
        for i, sample in enumerate(pending, start=1):
            images = [Image.open(sample["channels"][cam]).convert("RGB") for cam in CAMERAS]
            messages = [{
                "role": "user",
                "content": [*[{"type": "image", "image": im} for im in images],
                            {"type": "text", "text": args.prompt}],
            }]
            prompt = processor.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True)
            inputs = processor(text=[prompt], images=images, return_tensors="pt", padding=True)
            inputs = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v)
                      for k, v in inputs.items()}

            captures.clear()
            with torch.no_grad():
                # Run the backbone only: the hooks capture what we need and skipping
                # the LM head avoids materialising (seq_len x vocab) logits.
                backbone(**inputs)
            new_rows = {}
            for layer in args.layers:
                tokens = captures[layer][0].float()          # [seq_len, hidden]
                new_rows[(layer, "mean")] = tokens.mean(dim=0).cpu().numpy()
                new_rows[(layer, "last")] = tokens[-1].cpu().numpy()
                if not all(np.isfinite(new_rows[(layer, kind)]).all()
                           for kind in ("mean", "last")):
                    raise ValueError(f"Nonfinite features for {sample['sample_token']} layer {layer}")
            captures.clear()

            caption = None
            if args.caption:
                with torch.no_grad():
                    gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                         do_sample=False)
                caption = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:],
                                                 skip_special_tokens=True)[0].strip()

            for im in images:
                im.close()

            # Commit the sample only after everything succeeded, so that the feature
            # arrays and meta.json always stay row-aligned (also on interruption).
            for key, value in new_rows.items():
                rows[key].append(value)
            if caption is not None:
                captions.append({"sample_token": sample["sample_token"], "caption": caption})
            meta.append({
                "row_index": len(meta),
                "sample_token": sample["sample_token"],
                "scene_token": sample.get("scene_token"),
                "scene_name": sample.get("scene_name"),
                "timestamp": sample.get("timestamp"),
                "images": {cam: f"{cam}/{Path(sample['channels'][cam]).name}"
                           for cam in CAMERAS},
            })

            if i == 1 or i % 10 == 0:
                avg = (time.time() - t_start) / i
                print(f"[prog] {i}/{len(pending)} | avg {avg:.1f}s/sample | "
                      f"ETA {(len(pending) - i) * avg / 60:.0f}min")
            if i % args.save_every == 0:
                save_checkpoint(args.output_dir, args.layers, rows, meta, captions)
        success = True
    finally:
        for h in handles:
            h.remove()
        save_checkpoint(args.output_dir, args.layers, rows, meta, captions)
        summary = {
            "status": "completed" if success else "failed_or_interrupted",
            "model_load_seconds": model_load_seconds,
            "forward_seconds_including_image_reads": time.time() - t_start,
            "rows_in_cache": len(meta), "new_rows_this_invocation": len(meta) - len(done),
            "selected_rows_total": len(samples), "remaining_rows": len(samples) - len(meta),
            "selection_complete": len(samples) == len(meta),
            "device": args.device, "model_revision": args.model_revision,
            "extraction_config_sha256": sha256(args.output_dir / "extraction_config.json"),
            "peak_cuda_allocated_bytes": (torch.cuda.max_memory_allocated(args.device)
                                          if is_cuda else None),
            "peak_cuda_reserved_bytes": (torch.cuda.max_memory_reserved(args.device)
                                         if is_cuda else None),
            "packages": {}, "feature_files": {},
        }
        from importlib.metadata import PackageNotFoundError, version
        for package in ("torch", "transformers", "accelerate", "numpy", "Pillow"):
            try:
                summary["packages"][package] = version(package)
            except PackageNotFoundError:
                summary["packages"][package] = None
        for layer in args.layers:
            for path in array_paths(args.output_dir, layer).values():
                if path.exists():
                    summary["feature_files"][path.name] = {"sha256": sha256(path),
                                                           "bytes": path.stat().st_size}
        (args.output_dir / "extraction_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"[done] {len(meta)} samples written to {args.output_dir}")
    for layer in args.layers:
        for path in array_paths(args.output_dir, layer).values():
            print(f"[done] file: {path}")


if __name__ == "__main__":
    main()
