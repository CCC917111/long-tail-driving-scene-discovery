#!/usr/bin/env python3
"""Create deterministic CPU-only features with known splits and deliberate errors.

Every sample_token and scene_token starts with synthetic-. These data contain
no nuScenes imagery or VLM features and cannot substantiate model performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def create_fixture(output_dir: Path, seed: int = 17) -> list[dict]:
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta, manifest, annotations, scores = [], [], [], []
    features = []
    for split, scenes in (("train", 12), ("val", 4), ("test", 4)):
        for scene in range(scenes):
            for frame in range(4):
                token = f"synthetic-{split}-{scene:02d}-{frame}"
                scene_token = f"synthetic-{split}-{scene:02d}"
                label = frame % 2
                feature = rng.normal(0, 0.3, size=8)
                feature[0] += 2 * label - 1
                # Keep a real software failure path: high-confidence FP and FN.
                if split == "test" and scene == 0 and frame in (0, 1):
                    feature[0] *= -1
                score = float(feature[0])
                features.append(feature)
                meta.append({"sample_token": token, "scene_token": scene_token})
                annotations.append({"sample_token": token,
                                    "label": "long_tail" if label else "normal_core"})
                manifest.append({"sample_token": token, "scene_token": scene_token,
                                 "label": label, "split": split,
                                 "label_source": "synthetic-fixture",
                                 "review_status": "synthetic-not-human-reviewed",
                                 "exclusion_reason": None})
                scores.append({"sample_token": token, "split": split, "label": label,
                               "score": score, "prediction": int(score >= 0)})
    np.save(output_dir / "features.npy", np.asarray(features, dtype=np.float32))
    for filename, data in (("meta.json", meta), ("labels.json", annotations),
                           ("manifest.json", manifest), ("rows.json", scores)):
        (output_dir / filename).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    (output_dir / "SYNTHETIC.txt").write_text(
        "CPU software fixture only. No real driving data, trained VLM or scientific result.\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    manifest = create_fixture(args.output_dir, args.seed)
    print(f"Wrote {len(manifest)} SYNTHETIC samples to {args.output_dir}")


if __name__ == "__main__":
    main()
