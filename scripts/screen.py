#!/usr/bin/env python3
"""Screen your own driving data for long-tail samples with a trained SAE.

This is the entry point for using the method on new, unlabelled data rather
than for reproducing the benchmark: point it at a run directory produced by one
of the SAE training scripts and at features extracted from your own frames, and
it returns a ranked list of which frames are worth a human's attention — and,
for each one, which learned long-tail neurons fired, so the shortlist comes
with a reason rather than only a score.

Inputs
------
--run-dir   A directory written by scripts/sae_abstopk_tail_reward.py (or an
            ablation). It must contain best_model.pth, standardizer.npz and
            metrics.json; the SAE hyper-parameters and the decision threshold
            are read from metrics.json, so the screening run is configured
            exactly like the run that produced the model.
--features  An (N, D) .npy matrix written by scripts/extract.py for your own
            frames, using the same VLM, the same layer and the same pooling as
            the training run. D must match the run's input dimension.
--meta      The row-aligned meta.json written next to it.

Example:
    python scripts/screen.py \
        --run-dir output/sae_abstopk_tail_reward \
        --features output/extract_mydata/layer28_mlp_output_mean.npy \
        --meta output/extract_mydata/meta.json \
        --glossary results/neuron_glossary.csv \
        --output-dir output/screening
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from sae_common import LongTailGuidedSAE

MODEL_FILE = "best_model.pth"
STANDARDIZER_FILE = "standardizer.npz"
METRICS_FILE = "metrics.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Directory of a trained SAE run (best_model.pth, "
                        "standardizer.npz, metrics.json)")
    p.add_argument("--features", type=Path, required=True,
                   help="(N, D) .npy features of the frames to screen")
    p.add_argument("--meta", type=Path, required=True,
                   help="meta.json written alongside the features")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where the screening results are written")
    p.add_argument("--glossary", type=Path, default=None,
                   help="CSV mapping z_t neurons to human-readable features, "
                        "e.g. results/neuron_glossary.csv")
    p.add_argument("--threshold", type=float, default=None,
                   help="Decision threshold on ||z_t||. Default: the threshold "
                        "the training run chose on its validation split.")
    p.add_argument("--neuron-threshold", type=float, default=1.0,
                   help="|z_t| above which a neuron counts as activated")
    p.add_argument("--top-neurons", type=int, default=5,
                   help="How many activated neurons to report per frame")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_run(run_dir: Path) -> tuple[dict, float]:
    """Reads the SAE configuration and validation threshold of a training run."""
    metrics_path = run_dir / METRICS_FILE
    if not metrics_path.exists():
        raise SystemExit(f"{metrics_path} not found; --run-dir must point at a "
                         f"directory written by an SAE training script")
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    config = metrics.get("config", {})
    threshold = metrics.get("val", {}).get("threshold")
    if threshold is None:
        raise SystemExit(f"{metrics_path} has no validation threshold; pass "
                         f"--threshold explicitly")
    return config, float(threshold)


def build_model(config: dict, input_dim: int, device: str) -> LongTailGuidedSAE:
    """Rebuilds the SAE exactly as the training run configured it."""
    return LongTailGuidedSAE(
        input_dim=input_dim,
        hidden_dim=int(config.get("hidden_dim", 4096)),
        tail_ratio=float(config.get("tail_ratio", 0.5)),
        dropout=float(config.get("dropout", 0.2)),
        k=int(config.get("k", 512)),
        sparsity=str(config.get("sparsity", "abstopk")),
        reward=str(config.get("reward", "norm")),
    ).to(device)


def load_glossary(path: Path | None) -> dict[int, str]:
    """Reads a neuron -> feature-name mapping, if one was given."""
    if path is None:
        return {}
    glossary: dict[int, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                neuron = int(row["neuron"])
            except (KeyError, TypeError, ValueError):
                continue
            name = (row.get("feature") or "").strip()
            if name:
                glossary[neuron] = name
    print(f"[glossary] {len(glossary)} named neurons from {path}")
    return glossary


@torch.no_grad()
def encode(model: LongTailGuidedSAE, x: np.ndarray, batch_size: int,
           device: str) -> np.ndarray:
    """Returns the long-tail subspace activations z_t of every row."""
    chunks = []
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[start:start + batch_size]).to(device)
        _, _, z_t = model.encode(batch)
        chunks.append(z_t.cpu().numpy())
    return np.concatenate(chunks)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config, run_threshold = load_run(args.run_dir)
    threshold = args.threshold if args.threshold is not None else run_threshold

    features = np.load(args.features).astype(np.float32)
    if features.ndim != 2:
        raise SystemExit(f"expected a 2-D feature matrix, got {features.shape}")
    with args.meta.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    if len(meta) != len(features):
        raise SystemExit(f"{args.meta} has {len(meta)} rows but the features "
                         f"have {len(features)}")

    standardizer = np.load(args.run_dir / STANDARDIZER_FILE)
    mean, std = standardizer["mean"], standardizer["std"]
    if mean.shape[-1] != features.shape[1]:
        raise SystemExit(
            f"the run was trained on {mean.shape[-1]}-dimensional features but "
            f"{args.features} is {features.shape[1]}-dimensional; extract your "
            f"frames with the same VLM layer and pooling as the training run")
    # The training statistics are reused unchanged: the new frames are mapped
    # into the space the SAE was fitted in, not re-centred on themselves.
    x = ((features - mean) / std).astype(np.float32)

    model = build_model(config, features.shape[1], args.device)
    state = torch.load(args.run_dir / MODEL_FILE, map_location=args.device)
    model.load_state_dict(state)
    model.eval()
    print(f"[model] {args.run_dir / MODEL_FILE} | sparsity={model.sparsity} "
          f"k={model.k} tail_dim={model.tail_dim} | threshold={threshold:.4f}")

    z_t = encode(model, x, args.batch_size, args.device)
    score = np.linalg.norm(z_t, axis=1)
    predicted_tail = score >= threshold

    glossary = load_glossary(args.glossary)
    magnitude = np.abs(z_t)
    active = magnitude > args.neuron_threshold

    rows = []
    for i, entry in enumerate(meta):
        order = np.argsort(-magnitude[i])
        top = [int(j) for j in order[:args.top_neurons]
               if magnitude[i, j] > args.neuron_threshold]
        rows.append({
            "sample_token": entry.get("sample_token", ""),
            "scene": entry.get("scene_name") or entry.get("scene_token") or "",
            "tail_score": float(score[i]),
            "prediction": "long_tail" if predicted_tail[i] else "normal",
            "active_neurons": int(active[i].sum()),
            "top_neurons": " ".join(str(j) for j in top),
            "reasons": "; ".join(glossary[j] for j in top if j in glossary),
        })

    rows.sort(key=lambda r: -r["tail_score"])
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    fields = ["rank", "sample_token", "scene", "tail_score", "prediction",
              "active_neurons", "top_neurons", "reasons"]
    table = args.output_dir / "screening.csv"
    with table.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})

    flagged = [row["sample_token"] for row in rows
               if row["prediction"] == "long_tail"]
    with (args.output_dir / "long_tail_samples.json").open("w", encoding="utf-8") as f:
        json.dump(flagged, f, indent=1)

    print(f"\nscreened {len(rows)} frames | flagged {len(flagged)} "
          f"({100 * len(flagged) / max(len(rows), 1):.1f}%) as long-tail")
    if glossary:
        named: dict[str, int] = {}
        for row in rows:
            if row["prediction"] != "long_tail":
                continue
            for name in filter(None, row["reasons"].split("; ")):
                named[name] = named.get(name, 0) + 1
        if named:
            print("\nwhy they were flagged:")
            for name, count in sorted(named.items(), key=lambda kv: -kv[1]):
                print(f"  {count:5d}  {name}")
    print(f"\nranked table: {table}")
    print(f"flagged tokens: {args.output_dir / 'long_tail_samples.json'}")


if __name__ == "__main__":
    main()
