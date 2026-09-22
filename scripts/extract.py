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
scene_name and timestamp so that the SAE scripts can join labels and split the
data by scene. Optionally (``--caption``) the generated scene description is
written to ``captions.jsonl``.

Sample discovery reuses ``ensure_meta`` / ``build_samples`` from
scripts/annotate_normal_core.py, so both scripts see exactly the same set of
keyframes.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from annotate_normal_core import CAMERAS, build_samples, ensure_meta, load_vlm

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
    p.add_argument("--model-dir", type=Path, default=proj / "models" / "Cosmos-Reason1-7B")
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


def save_checkpoint(output_dir: Path, layers: list[int], rows: dict, meta: list[dict],
                    captions: list[dict]) -> None:
    if not meta:
        return
    for layer in layers:
        for kind, path in array_paths(output_dir, layer).items():
            tmp = path.with_suffix(".tmp.npy")
            np.save(tmp, np.stack(rows[(layer, kind)]).astype(np.float32))
            os.replace(tmp, path)
    tmp = output_dir / "meta.json.tmp"
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    os.replace(tmp, output_dir / "meta.json")
    if captions:
        with (output_dir / "captions.jsonl").open("w", encoding="utf-8") as f:
            for row in captions:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_checkpoint(output_dir: Path, layers: list[int]):
    meta_path = output_dir / "meta.json"
    rows = {(layer, kind): [] for layer in layers for kind in ("mean", "last")}
    if not meta_path.exists():
        return rows, [], []
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    for layer in layers:
        for kind, path in array_paths(output_dir, layer).items():
            if not path.exists():
                raise FileNotFoundError(f"Cannot resume: {path} is missing. "
                                        f"Use --no-resume or the original --layers.")
            arr = np.load(path)
            if len(arr) != len(meta):
                raise ValueError(f"Cannot resume: {path} has {len(arr)} rows, "
                                 f"meta.json has {len(meta)}")
            rows[(layer, kind)] = list(arr)
    captions = []
    cap_path = output_dir / "captions.jsonl"
    if cap_path.exists():
        with cap_path.open("r", encoding="utf-8") as f:
            captions = [json.loads(line) for line in f if line.strip()]
    print(f"[resume] loaded {len(meta)} rows from {output_dir}")
    return rows, meta, captions


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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meta_dir = ensure_meta(args.meta_tgz, args.meta_extract_dir)
    samples = build_samples(meta_dir, args.samples_root)
    if args.sample_tokens is not None:
        with args.sample_tokens.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rows = data["samples"] if isinstance(data, dict) else data
        wanted = {r["sample_token"] for r in rows if "sample_token" in r}
        samples = [s for s in samples if s["sample_token"] in wanted]
        print(f"[plan] restricted to {len(samples)} samples listed in {args.sample_tokens}")

    rows, meta, captions = (load_checkpoint(args.output_dir, args.layers) if args.resume
                            else ({(l, k): [] for l in args.layers for k in ("mean", "last")},
                                  [], []))
    done = {m["sample_token"] for m in meta}
    pending = [s for s in samples if s["sample_token"] not in done]
    if args.max_samples > 0:
        pending = pending[: args.max_samples]
    print(f"[plan] total={len(samples)} done={len(done)} to_process={len(pending)}")
    if not pending:
        return

    processor, model = load_vlm(args.model_dir, args.device, args.dtype,
                                args.min_pixels, args.max_pixels)
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
            })

            if i == 1 or i % 10 == 0:
                avg = (time.time() - t_start) / i
                print(f"[prog] {i}/{len(pending)} | avg {avg:.1f}s/sample | "
                      f"ETA {(len(pending) - i) * avg / 60:.0f}min")
            if i % args.save_every == 0:
                save_checkpoint(args.output_dir, args.layers, rows, meta, captions)
    finally:
        for h in handles:
            h.remove()
        save_checkpoint(args.output_dir, args.layers, rows, meta, captions)

    print(f"[done] {len(meta)} samples written to {args.output_dir}")
    for layer in args.layers:
        for path in array_paths(args.output_dir, layer).values():
            print(f"[done] file: {path}")


if __name__ == "__main__":
    main()
