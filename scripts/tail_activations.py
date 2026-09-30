#!/usr/bin/env python3
"""The long-tail decision rule and the activation store, without torch.

Everything that happens *after* the sparse autoencoder has produced its code
lives here, so that the decision rule is defined in exactly one place and the
neuron explorer can read a run without importing torch.

Decision rule
-------------
With the sparse code split as ``z = [z_n, z_t]``, the number of active units in
the long-tail subspace is

    c_tail(x) = #{ j : |z_t,j(x)| > eta }

and a sample is predicted long-tail when ``c_tail(x) >= 1``, i.e. when at least
one long-tail-sensitive unit fires. ``eta`` defaults to 0.01.

``||z_t||_2`` is kept alongside as a continuous score. It is not the decision;
it orders flagged samples by how strongly the long-tail subspace responds and
gives the threshold-free AUC / AP numbers.

Activation store
----------------
``activations.npz`` holds the non-zero entries of ``z_t`` for every sample in
compressed sparse row form, next to the sample identifiers and, when the
extraction recorded them, the relative image path of every camera view. AbsTopK
keeps at most ``k`` units per sample, so the file stays small even for the full
dataset.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

DEFAULT_ETA = 0.01
DEFAULT_NEURON_THRESHOLD = 1.0

CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


# ---------------------------------------------------------------------------
#  Decision rule
# ---------------------------------------------------------------------------

def tail_activation_count(z_t: np.ndarray, eta: float = DEFAULT_ETA) -> np.ndarray:
    """``c_tail``: the number of long-tail units with ``|z_t| > eta`` per sample."""
    z_t = np.asarray(z_t)
    if z_t.ndim != 2:
        raise ValueError(f"z_t must be 2-D (samples x units), got {z_t.shape}")
    return (np.abs(z_t) > eta).sum(axis=1)


def predict_long_tail(z_t: np.ndarray, eta: float = DEFAULT_ETA) -> np.ndarray:
    """1 where at least one long-tail unit is active, else 0."""
    return (tail_activation_count(z_t, eta) >= 1).astype(int)


def tail_score(z_t: np.ndarray) -> np.ndarray:
    """``||z_t||_2``: the continuous ranking score."""
    return np.linalg.norm(np.asarray(z_t, dtype=np.float64), axis=1)


# ---------------------------------------------------------------------------
#  Activation store
# ---------------------------------------------------------------------------

def save_activations(path: Path, z_t: np.ndarray, meta: list[dict],
                     labels: np.ndarray | None = None,
                     splits: list[str] | None = None) -> None:
    """Writes the non-zero entries of ``z_t`` plus per-sample metadata."""
    z_t = np.asarray(z_t, dtype=np.float32)
    if len(meta) != len(z_t):
        raise ValueError(f"{len(meta)} metadata rows for {len(z_t)} activation rows")

    nz_rows, nz_cols = np.nonzero(z_t)
    indptr = np.zeros(len(z_t) + 1, dtype=np.int64)
    np.add.at(indptr, nz_rows + 1, 1)
    indptr = np.cumsum(indptr)

    samples = [{
        "sample_token": m.get("sample_token", ""),
        "scene": m.get("scene_name") or m.get("scene_token") or "",
        "images": m.get("images") or {},
    } for m in meta]

    arrays = {
        "indptr": indptr,
        "indices": nz_cols.astype(np.int32),
        "values": z_t[nz_rows, nz_cols].astype(np.float32),
        "tail_dim": np.array(z_t.shape[1], dtype=np.int64),
        "samples": np.array(json.dumps(samples)),
    }
    if labels is not None:
        arrays["labels"] = np.asarray(labels, dtype=np.int8)
    if splits is not None:
        arrays["splits"] = np.array(json.dumps(list(splits)))
    np.savez_compressed(path, **arrays)


class ActivationStore:
    """Read-only view of ``activations.npz``."""

    def __init__(self, path: Path):
        path = Path(path)
        if path.is_dir():
            path = path / "activations.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found; it is written by the SAE training scripts and "
                f"by scripts/screen.py")
        with np.load(path, allow_pickle=False) as data:
            self.indptr = data["indptr"]
            self.indices = data["indices"]
            self.values = data["values"]
            self.tail_dim = int(data["tail_dim"])
            self.samples = json.loads(str(data["samples"]))
            self.labels = data["labels"].astype(int) if "labels" in data else None
            self.splits = json.loads(str(data["splits"])) if "splits" in data else None
        self.path = path
        self.row_of = {s["sample_token"]: i for i, s in enumerate(self.samples)}

    def __len__(self) -> int:
        return len(self.samples)

    def row(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """(unit indices, values) of the non-zero ``z_t`` entries of sample ``i``."""
        start, end = self.indptr[i], self.indptr[i + 1]
        return self.indices[start:end], self.values[start:end]

    def dense(self) -> np.ndarray:
        """The full ``(N, tail_dim)`` matrix; fine for tests and small runs."""
        out = np.zeros((len(self), self.tail_dim), dtype=np.float32)
        for i in range(len(self)):
            cols, vals = self.row(i)
            out[i, cols] = vals
        return out

    def column(self, unit: int) -> tuple[np.ndarray, np.ndarray]:
        """(sample rows, values) of every non-zero entry of one unit."""
        if not 0 <= unit < self.tail_dim:
            raise IndexError(f"unit {unit} is outside the long-tail subspace "
                             f"[0, {self.tail_dim})")
        hit = np.nonzero(self.indices == unit)[0]
        rows = np.searchsorted(self.indptr, hit, side="right") - 1
        return rows, self.values[hit]

    def counts(self, eta: float = DEFAULT_ETA) -> np.ndarray:
        """``c_tail`` of every sample, from the sparse entries."""
        active = (np.abs(self.values) > eta).astype(np.int64)
        cumulative = np.concatenate([[0], np.cumsum(active)])
        return cumulative[self.indptr[1:]] - cumulative[self.indptr[:-1]]


# ---------------------------------------------------------------------------
#  Neuron glossary
# ---------------------------------------------------------------------------

def load_glossary(path: Path | None) -> dict[int, str]:
    """Reads a ``neuron -> feature name`` mapping such as results/neuron_glossary.csv."""
    if path is None:
        return {}
    glossary: dict[int, str] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                neuron = int(row["neuron"])
            except (KeyError, TypeError, ValueError):
                continue
            name = (row.get("feature") or "").strip()
            if name:
                glossary[neuron] = name
    return glossary
