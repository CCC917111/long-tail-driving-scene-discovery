#!/usr/bin/env python3
"""Shared code for the long-tail-guided sparse autoencoder (SAE) experiments.

The three entry-point scripts

    scripts/sae_abstopk_tail_reward.py       (final method)
    scripts/ablation_topk_sae.py             (ablation: signed Top-K, no reward)
    scripts/ablation_sae_cosmos_baseline.py  (ablation: reward before selection)

only differ in their default hyper-parameters; all data handling, training and
evaluation logic lives here so that the variants are compared under exactly
the same protocol.

Pipeline
--------
1. Load VLM hidden-state features (``--features``, an ``(N, D)`` ``.npy`` file
   written by ``scripts/extract.py``) and the row-aligned ``meta.json``.
2. Join them with normal / long-tail labels (``--labels``). Rows without a
   usable label (missing, ``unknown`` or ``uncertain``) are dropped instead of
   being silently treated as long-tail.
3. Split into train / validation / test. When scene information is available
   in ``meta.json`` the split is done at the *scene* level, so near-duplicate
   keyframes from the same nuScenes scene never end up on both sides.
4. Standardise features with statistics computed on the training split only.
5. Train the SAE; the checkpoint with the lowest validation loss is kept.
6. Score every sample with ``||z_t||_2``. The decision threshold is chosen on
   the validation split (max F1) and then applied unchanged to the held-out
   test split, which is where the headline metrics are reported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

NORMAL_LABELS = {"normal", "normal_core"}
DROP_LABELS = {"", "unknown", "uncertain", "none"}


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def build_parser(description: str, defaults: dict) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--features", type=Path, required=True,
                   help="(N, D) .npy feature matrix from scripts/extract.py, "
                        "e.g. layer28_mlp_output_last_token.npy")
    p.add_argument("--meta", type=Path, required=True,
                   help="meta.json written by scripts/extract.py (one entry per feature row)")
    p.add_argument("--labels", type=Path, nargs="+", required=True,
                   help="One or more label files. Accepted formats: the JSON array written by "
                        "scripts/annotate_normal_core.py, or {'samples': [{'sample_token', "
                        "'labels': {'label'}}]}.")
    p.add_argument("--output-dir", type=Path, required=True, help="Where results are written")
    # Split
    p.add_argument("--val-ratio", type=float, default=0.15, help="Fraction of samples for validation")
    p.add_argument("--test-ratio", type=float, default=0.15, help="Fraction of samples for testing")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    # Model
    p.add_argument("--hidden-dim", type=int, default=defaults["hidden_dim"],
                   help="Number of SAE latent units")
    p.add_argument("--tail-ratio", type=float, default=0.5,
                   help="Fraction of latent units reserved for the long-tail subspace z_t")
    p.add_argument("--dropout", type=float, default=defaults["dropout"],
                   help="Dropout on the pre-sparsity latent code (training only)")
    p.add_argument("--k", type=int, default=512, help="Number of latent units kept per sample")
    p.add_argument("--sparsity", choices=["abstopk", "topk"], default=defaults["sparsity"],
                   help="abstopk: keep the k largest |z|; topk: keep the k largest signed z")
    p.add_argument("--reward", choices=["none", "norm", "pre_selection"],
                   default=defaults["reward"],
                   help="none: no tail reward; norm: reward the achieved ||z_t|| of tail "
                        "samples (capped); pre_selection: add a constant bonus to the z_t "
                        "selection scores of tail samples before Top-K (training only)")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Extra reconstruction weight for tail samples")
    p.add_argument("--beta", type=float, default=defaults["beta"],
                   help="Weight of the ||z_t||^2 penalty on normal samples")
    p.add_argument("--reward-coeff", type=float, default=defaults["reward_coeff"],
                   help="Strength of the tail reward (beta_tail in the README loss)")
    p.add_argument("--reward-max", type=float, default=2.0,
                   help="Cap of the tail reward (tau in the README loss)")
    # Optimisation
    p.add_argument("--epochs", type=int, default=200, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-5, help="Adam weight decay")
    # Evaluation
    p.add_argument("--act-threshold", type=float, default=0.01,
                   help="|z| above this value counts as an active latent unit")
    p.add_argument("--neuron-threshold", type=float, default=1.0,
                   help="|z_t| threshold used for the per-neuron interpretability report")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="torch device")
    return p


# ---------------------------------------------------------------------------
#  Data
# ---------------------------------------------------------------------------

def _normalise_label(raw) -> int | None:
    """Map a raw label string to 0 (normal), 1 (long-tail) or None (drop)."""
    if raw is None:
        return None
    lab = str(raw).strip().lower()
    if lab in DROP_LABELS:
        return None
    return 0 if lab in NORMAL_LABELS else 1


def load_label_map(paths: list[Path]) -> dict[str, int | None]:
    token_to_label: dict[str, int | None] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rows = data["samples"] if isinstance(data, dict) else data
        for row in rows:
            token = row.get("sample_token")
            if token is None:
                continue
            raw = row.get("label")
            if raw is None and isinstance(row.get("labels"), dict):
                raw = row["labels"].get("label")
            token_to_label[token] = _normalise_label(raw)
    return token_to_label


def load_dataset(args) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[dict]]:
    print(f"[data] features: {args.features}")
    features = np.load(args.features).astype(np.float32)
    if features.ndim != 2:
        raise ValueError(f"Expected a 2-D feature matrix, got shape {features.shape}")

    with args.meta.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    if len(meta) != len(features):
        raise ValueError(f"meta.json has {len(meta)} rows but features have {len(features)}")

    label_map = load_label_map(args.labels)
    labels = np.array([
        -1 if label_map.get(m.get("sample_token")) is None else label_map[m["sample_token"]]
        for m in meta
    ])
    keep = labels >= 0
    n_missing = sum(1 for m in meta if m.get("sample_token") not in label_map)
    print(f"[data] rows: {len(meta)} | labelled: {keep.sum()} | "
          f"no label: {n_missing} | uncertain/unknown: {(~keep).sum() - n_missing}")

    features = features[keep]
    labels = labels[keep].astype(np.float32)
    meta = [m for m, k in zip(meta, keep) if k]

    groups = None
    for key in ("scene_token", "scene_name"):
        if all(m.get(key) for m in meta):
            groups = np.array([m[key] for m in meta])
            print(f"[data] grouping split by '{key}' ({len(set(groups))} scenes)")
            break
    if groups is None:
        print("[data] WARNING: meta.json has no scene information; falling back to a "
              "sample-level split. Neighbouring keyframes of the same scene may then leak "
              "between train and test.")

    print(f"[data] normal: {(labels == 0).sum()} | long-tail: {(labels == 1).sum()} | "
          f"dim: {features.shape[1]}")
    return features, labels, groups, meta


def split_indices(labels: np.ndarray, groups: np.ndarray | None, val_ratio: float,
                  test_ratio: float, seed: int) -> dict[str, np.ndarray]:
    """Random train/val/test split in which whole groups (scenes) are assigned together.

    Long-tail samples tend to cluster in scenes (e.g. a whole rainy scene), so the
    split is greedily stratified: groups are visited in random order and each one
    goes to the split that is furthest below its target share of long-tail
    samples (for groups containing long-tail samples) or of samples overall.
    """
    n = len(labels)
    rng = np.random.default_rng(seed)
    if groups is None:
        groups = np.arange(n)
    unique, inverse = np.unique(groups, return_inverse=True)
    size = np.bincount(inverse)
    n_tail = np.bincount(inverse, weights=labels)
    order = rng.permutation(len(unique))

    ratios = {"train": 1.0 - val_ratio - test_ratio, "val": val_ratio, "test": test_ratio}
    total_tail = max(float(labels.sum()), 1.0)
    count = {k: 0.0 for k in ratios}
    tail = {k: 0.0 for k in ratios}
    split_of_group = np.empty(len(unique), dtype=object)
    for g in order:
        if n_tail[g] > 0:
            deficit = {k: ratios[k] - tail[k] / total_tail for k in ratios}
        else:
            deficit = {k: ratios[k] - count[k] / n for k in ratios}
        choice = max(deficit, key=deficit.get)
        split_of_group[g] = choice
        count[choice] += size[g]
        tail[choice] += n_tail[g]

    assignment = split_of_group[inverse]
    return {name: np.where(assignment == name)[0] for name in ("train", "val", "test")}


# ---------------------------------------------------------------------------
#  Model
# ---------------------------------------------------------------------------

class LongTailGuidedSAE(nn.Module):
    """Single-layer SAE whose latent code is split into z_n (normal) and z_t (tail)."""

    def __init__(self, input_dim: int, hidden_dim: int, tail_ratio: float = 0.5,
                 dropout: float = 0.2, k: int = 512, sparsity: str = "abstopk",
                 reward: str = "norm", alpha: float = 1.0, beta: float = 0.1,
                 reward_coeff: float = 0.5, reward_max: float = 2.0):
        super().__init__()
        if k > hidden_dim:
            raise ValueError(f"k={k} must not exceed hidden_dim={hidden_dim}")
        self.k = k
        self.sparsity = sparsity
        self.reward = reward
        self.alpha = alpha
        self.beta = beta
        self.reward_coeff = reward_coeff
        self.reward_max = reward_max

        self.tail_dim = int(hidden_dim * tail_ratio)
        self.normal_dim = hidden_dim - self.tail_dim

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x: torch.Tensor, is_tail: torch.Tensor | None = None,
               apply_reward: bool = False):
        z = self.dropout(self.encoder(x))

        scores = z.abs() if self.sparsity == "abstopk" else z.clone()
        if self.reward == "pre_selection" and apply_reward and is_tail is not None:
            # Constant bonus on the z_t selection scores of tail samples, so that
            # Top-K prefers tail units for them. Only the *selection* changes; the
            # kept values are the original activations. Used during training only.
            bonus = min(self.reward_coeff, self.reward_max)
            scores = scores.clone()
            scores[:, self.normal_dim:] += bonus * is_tail.unsqueeze(1)

        _, topk_idx = torch.topk(scores, k=self.k, dim=1)
        mask = torch.zeros_like(z).scatter_(1, topk_idx, 1.0)
        z = z * mask
        return z, z[:, :self.normal_dim], z[:, self.normal_dim:]

    def forward(self, x: torch.Tensor, is_tail: torch.Tensor, apply_reward: bool = False):
        z, z_n, z_t = self.encode(x, is_tail, apply_reward)
        x_hat = self.decoder(z)

        # Reconstruction, up-weighted for tail samples.
        weight = 1 + self.alpha * is_tail.unsqueeze(1)
        l_recon = (weight * (x_hat - x) ** 2).mean()

        # Keep z_t quiet on normal samples.
        zero = torch.zeros((), device=x.device)
        normal = is_tail == 0
        l_normal = self.beta * (z_t[normal].norm(dim=1) ** 2).mean() if normal.any() else zero

        # Reward z_t activity on tail samples, capped at reward_max. Note that
        # -c * min(||z_t||, tau) = c * max(0, tau - ||z_t||) - c * tau, i.e. this is
        # the hinge term of the README loss up to a constant.
        tail = is_tail == 1
        if self.reward == "norm" and tail.any():
            l_tail = -self.reward_coeff * torch.clamp(z_t[tail].norm(dim=1),
                                                      max=self.reward_max).mean()
        else:
            l_tail = zero

        total = l_recon + l_normal + l_tail
        parts = {"total": total, "recon": l_recon, "normal": l_normal, "tail_reward": l_tail}
        return total, parts, z_n, z_t, x_hat


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------

def train(model: LongTailGuidedSAE, x_train, y_train, x_val, y_val, args) -> dict:
    device = torch.device(args.device)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
                            batch_size=args.batch_size)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10,
                                                     factor=0.5)
    ckpt = args.output_dir / "best_model.pth"
    history = {"train": [], "val": []}
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {"total": 0.0, "recon": 0.0, "normal": 0.0, "tail_reward": 0.0}
        for x, t in train_loader:
            x, t = x.to(device), t.to(device)
            loss, parts, *_ = model(x, t, apply_reward=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for key in sums:
                sums[key] += parts[key].item()
        train_parts = {key: v / len(train_loader) for key, v in sums.items()}

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for x, t in val_loader:
                x, t = x.to(device), t.to(device)
                loss, *_ = model(x, t, apply_reward=False)
                val_total += loss.item()
        val_total /= len(val_loader)

        history["train"].append(train_parts["total"])
        history["val"].append(val_total)
        scheduler.step(val_total)
        if val_total < best_val:
            best_val = val_total
            torch.save(model.state_dict(), ckpt)

        if epoch == 1 or epoch % 20 == 0:
            print(f"[epoch {epoch:3d}/{args.epochs}] "
                  f"train total={train_parts['total']:.4f} recon={train_parts['recon']:.4f} "
                  f"normal={train_parts['normal']:.4f} "
                  f"tail_reward={train_parts['tail_reward']:.4f} | "
                  f"val total={val_total:.4f} (best {best_val:.4f})")

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return history


def save_loss_curve(history: dict, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.figure()
    plt.plot(history["train"], label="train loss")
    plt.plot(history["val"], label="val loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig(path)
    plt.close()


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_all(model: LongTailGuidedSAE, x: np.ndarray, batch_size: int, device: str):
    z_n_all, z_t_all = [], []
    for i in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[i:i + batch_size]).to(device)
        _, z_n, z_t = model.encode(batch)
        z_n_all.append(z_n.cpu().numpy())
        z_t_all.append(z_t.cpu().numpy())
    return np.concatenate(z_n_all), np.concatenate(z_t_all)


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[int(np.argmax(f1[:-1]))])


def classification_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = (score >= threshold).astype(int)
    p, r, f, support = precision_recall_fscore_support(y, pred, labels=[1, 0], zero_division=0)
    return {
        "auc": float(roc_auc_score(y, score)),
        "ap": float(average_precision_score(y, score)),
        "threshold": threshold,
        "long_tail": {"precision": float(p[0]), "recall": float(r[0]), "f1": float(f[0]),
                      "support": int(support[0])},
        "normal": {"precision": float(p[1]), "recall": float(r[1]), "f1": float(f[1]),
                   "support": int(support[1])},
    }


def activation_stats(name: str, z_n: np.ndarray, z_t: np.ndarray, y: np.ndarray,
                     threshold: float) -> None:
    """Share of samples with at least one active unit in z_n / z_t, per class."""
    zn_active = (np.abs(z_n) > threshold).any(axis=1)
    zt_active = (np.abs(z_t) > threshold).any(axis=1)
    zt_count = (np.abs(z_t) > threshold).sum(axis=1)
    print(f"\n{'=' * 70}\nActivation statistics - {name}\n{'=' * 70}")
    for cls_name, mask in (("Normal", y == 0), ("Long-tail", y == 1)):
        n = int(mask.sum())
        if n == 0:
            continue
        print(f"{cls_name} samples: {n}")
        print(f"  - z_n active: {zn_active[mask].sum()}/{n} ({zn_active[mask].mean() * 100:.1f}%)")
        print(f"  - z_t active: {zt_active[mask].sum()}/{n} ({zt_active[mask].mean() * 100:.1f}%)")
        print(f"  - mean number of active z_t units: {zt_count[mask].mean():.1f}")


def neuron_report(z_t: np.ndarray, y: np.ndarray, threshold: float, path: Path,
                  top: int = 20) -> None:
    """Per-neuron activation purity on long-tail samples (training split)."""
    active = np.abs(z_t) > threshold
    n_active = active.sum(axis=0)
    n_tail = active[y == 1].sum(axis=0)
    purity = np.divide(n_tail, n_active, out=np.zeros(len(n_active)), where=n_active > 0)
    coverage = n_tail / max(int((y == 1).sum()), 1)
    order = np.lexsort((-n_tail, -purity))
    with path.open("w", encoding="utf-8") as f:
        f.write("z_t_neuron,n_active,n_active_long_tail,purity,long_tail_coverage\n")
        for j in order:
            if n_active[j] == 0:
                continue
            f.write(f"{j},{n_active[j]},{n_tail[j]},{purity[j]:.4f},{coverage[j]:.4f}\n")
    print(f"\nTop z_t neurons by long-tail purity (|z_t| > {threshold}, training split):")
    shown = 0
    for j in order:
        if n_active[j] < 5:
            continue
        print(f"  neuron {j:5d}: purity={purity[j] * 100:5.1f}% "
              f"({n_tail[j]}/{n_active[j]}), coverage={coverage[j] * 100:.1f}%")
        shown += 1
        if shown >= top:
            break


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

def run(description: str, defaults: dict) -> None:
    args = build_parser(description, defaults).parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"[config] sparsity={args.sparsity} reward={args.reward} hidden_dim={args.hidden_dim} "
          f"k={args.k} beta={args.beta} reward_coeff={args.reward_coeff} device={args.device}")

    features, labels, groups, meta = load_dataset(args)
    splits = split_indices(labels, groups, args.val_ratio, args.test_ratio, args.seed)
    for name, idx in splits.items():
        n_tail = int(labels[idx].sum())
        print(f"[split] {name}: {len(idx)} samples ({len(idx) - n_tail} normal, {n_tail} long-tail)")
        if len(idx) == 0 or n_tail == 0 or n_tail == len(idx):
            raise ValueError(f"Split '{name}' does not contain both classes; "
                             f"adjust --val-ratio/--test-ratio or --seed.")

    # Standardise with training statistics only (no information from val/test).
    mean = features[splits["train"]].mean(axis=0, keepdims=True)
    std = features[splits["train"]].std(axis=0, keepdims=True) + 1e-8
    x = ((features - mean) / std).astype(np.float32)

    model = LongTailGuidedSAE(
        input_dim=x.shape[1], hidden_dim=args.hidden_dim, tail_ratio=args.tail_ratio,
        dropout=args.dropout, k=args.k, sparsity=args.sparsity, reward=args.reward,
        alpha=args.alpha, beta=args.beta, reward_coeff=args.reward_coeff,
        reward_max=args.reward_max,
    ).to(args.device)

    tr, va, te = splits["train"], splits["val"], splits["test"]
    history = train(model, x[tr], labels[tr], x[va], labels[va], args)
    save_loss_curve(history, args.output_dir / "training_loss_curve.png")

    z_n, z_t = encode_all(model, x, args.batch_size, args.device)
    score = np.linalg.norm(z_t, axis=1)

    for name in ("train", "val", "test"):
        idx = splits[name]
        activation_stats(name, z_n[idx], z_t[idx], labels[idx], args.act_threshold)

    threshold = best_f1_threshold(labels[va], score[va])
    metrics = {
        "config": vars(args),
        "score": "||z_t||_2",
        "val": classification_metrics(labels[va], score[va], threshold),
        "test": classification_metrics(labels[te], score[te], threshold),
    }
    print(f"\n{'=' * 70}\nLong-tail detection with ||z_t||_2 "
          f"(threshold {threshold:.4f} chosen on validation)\n{'=' * 70}")
    for name in ("val", "test"):
        m = metrics[name]
        print(f"{name:>4}: AUC={m['auc']:.4f} AP={m['ap']:.4f} | long_tail "
              f"P={m['long_tail']['precision']:.4f} R={m['long_tail']['recall']:.4f} "
              f"F1={m['long_tail']['f1']:.4f} | normal P={m['normal']['precision']:.4f} "
              f"R={m['normal']['recall']:.4f} F1={m['normal']['f1']:.4f}")

    neuron_report(z_t[tr], labels[tr], args.neuron_threshold,
                  args.output_dir / "neuron_report.csv")

    split_name = np.empty(len(x), dtype=object)
    for name, idx in splits.items():
        split_name[idx] = name
    np.save(args.output_dir / "z_n_general.npy", z_n)
    np.save(args.output_dir / "z_t_longtail.npy", z_t)
    np.save(args.output_dir / "is_tail.npy", labels)
    with (args.output_dir / "rows.json").open("w", encoding="utf-8") as f:
        json.dump([{"sample_token": m.get("sample_token"), "split": s, "label": int(l),
                    "score": float(sc)} for m, s, l, sc in zip(meta, split_name, labels, score)],
                  f, indent=1)
    np.savez(args.output_dir / "standardizer.npz", mean=mean, std=std)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nAll outputs saved to: {args.output_dir}")
