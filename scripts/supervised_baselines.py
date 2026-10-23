#!/usr/bin/env python3
"""Logistic and small-MLP controls on exactly the SAE's frozen VLM features.

Requires an existing scene-disjoint manifest. Standardisation is fitted only
on train. The MLP checkpoint is chosen by explicit validation log loss, never
by sklearn's internal random validation split or by test performance.
"""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier

from evaluate_runs import write_evaluation_artifacts


def fit_baseline(method: str, x_train, y_train, x_val, y_val, *, seed: int,
                 c: float = 1.0, hidden_dim: int = 64, epochs: int = 100,
                 batch_size: int = 64, lr: float = 1e-3, l2: float = 1e-4,
                 max_iter: int = 1000):
    """Return a fitted estimator and its auditable validation-selection history."""
    if c <= 0 or hidden_dim <= 0 or epochs <= 0 or batch_size <= 0 or lr <= 0 or l2 < 0:
        raise ValueError("Invalid baseline hyperparameters")
    if method == "logistic":
        model = LogisticRegression(C=c, solver="lbfgs", max_iter=max_iter, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                model.fit(x_train, y_train)
            except ConvergenceWarning as exc:
                raise ValueError("Logistic regression did not converge; "
                                 "increase --max-iter") from exc
        history = {"selection": "single fixed-C fit; no checkpoint search",
                   "best_epoch": None, "iterations": int(model.n_iter_.max()),
                   "val_log_loss": [float(log_loss(y_val, model.predict_proba(x_val),
                                                   labels=[0, 1]))]}
        return model, history
    if method != "mlp":
        raise ValueError(f"Unknown baseline: {method}")
    model = MLPClassifier(hidden_layer_sizes=(hidden_dim,), activation="relu", solver="adam",
                          alpha=l2, batch_size=min(batch_size, len(y_train)),
                          learning_rate_init=lr, max_iter=1, shuffle=True,
                          early_stopping=False, random_state=seed)
    best, best_loss, best_epoch = None, float("inf"), None
    history = {"selection": "minimum explicit validation log loss",
               "train_loss": [], "val_log_loss": []}
    for epoch in range(1, epochs + 1):
        model.partial_fit(x_train, y_train, classes=np.array([0, 1]))
        loss = float(log_loss(y_val, model.predict_proba(x_val), labels=[0, 1]))
        if not np.isfinite(loss) or not np.isfinite(model.loss_):
            raise ValueError("Nonfinite MLP loss; check inputs and learning rate")
        history["train_loss"].append(float(model.loss_))
        history["val_log_loss"].append(loss)
        if loss < best_loss:
            best, best_loss, best_epoch = copy.deepcopy(model), loss, epoch
    history["best_epoch"] = best_epoch
    return best, history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("logistic", "mlp"), required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--labels", type=Path, nargs="+", required=True)
    parser.add_argument("--feature-config", type=Path,
                        help="Recovered extraction provenance JSON; never infer missing history")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Frozen manifest from the first SAE run; never generate a new split")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c", type=float, default=1.0, help="Logistic inverse L2 strength")
    parser.add_argument("--max-iter", type=int, default=1000, help="Logistic solver iteration cap")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2", type=float, default=1e-4, help="MLP sklearn alpha (L2 coefficient)")
    parser.add_argument("--review-k", type=int, required=True,
                        help="Review budget fixed before comparing test results")
    parser.add_argument("--target-precision", type=float, default=0.9)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    return parser


def main() -> None:
    # Shared strict feature/label loading keeps these controls aligned with SAE.
    from data_manifest import load_frozen_manifest
    from sae_common import load_dataset

    args = build_parser().parse_args()
    args.allow_sample_split = False
    if args.review_k <= 0 or not 0 < args.target_precision <= 1 or args.bootstrap_samples < 0:
        raise ValueError("Invalid evaluation budget/precision/bootstrap settings")
    provenance = (json.loads(args.feature_config.read_text(encoding="utf-8"))
                  if args.feature_config else
                  {"status": "not supplied; historical origin unverified"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    features, labels, _, meta = load_dataset(args)
    splits = load_frozen_manifest(args.manifest, meta, labels)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    for name, indices in splits.items():
        if set(labels[indices].astype(int).tolist()) != {0, 1}:
            raise ValueError(f"{name} must contain both classes")
    tr, va = splits["train"], splits["val"]
    mean = features[tr].mean(axis=0, keepdims=True)
    std = features[tr].std(axis=0, keepdims=True) + 1e-8
    x = ((features - mean) / std).astype(np.float32)
    model, history = fit_baseline(
        args.method, x[tr], labels[tr], x[va], labels[va], seed=args.seed, c=args.c,
        hidden_dim=args.hidden_dim, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, l2=args.l2, max_iter=args.max_iter)
    score = model.predict_proba(x)[:, list(model.classes_).index(1)]
    split_of = {int(i): name for name, indices in splits.items() for i in indices}
    rows = [{"sample_token": row["sample_token"], "label": int(labels[i]),
             "split": split_of[i], "score": float(score[i]), "prediction": int(score[i] >= 0.5)}
            for i, row in enumerate(meta) if i in split_of]
    checkpoint = args.output_dir / "model.joblib"
    joblib.dump({"estimator": model, "mean": mean, "std": std,
                 "method": args.method, "feature_dim": x.shape[1]}, checkpoint)
    np.savez(args.output_dir / "standardizer.npz", mean=mean, std=std)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    (args.output_dir / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    input_paths = {"features": args.features, "meta": args.meta, "manifest": args.manifest,
                   **{f"labels_{i}": path for i, path in enumerate(args.labels)}}
    config = dict(vars(args), estimator_params=model.get_params(),
                  score="P(long_tail | frozen features)", training_history=history,
                  feature_provenance=provenance)
    if args.feature_config:
        input_paths["feature_config"] = args.feature_config
    result = write_evaluation_artifacts(
        args.output_dir, rows, manifest, config=config, seed=args.seed,
        checkpoint_path=checkpoint, input_paths=input_paths, review_k=args.review_k,
        target_precision=args.target_precision, bootstrap_samples=args.bootstrap_samples,
        fixed_rule={"rule": "score >= 0.5", "threshold": 0.5})
    print(json.dumps(result["splits"]["test"], indent=2))


if __name__ == "__main__":
    main()
