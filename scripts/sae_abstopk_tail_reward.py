#!/usr/bin/env python3
"""Final method: long-tail-guided SAE with AbsTopK sparsity and a tail-activation reward.

* AbsTopK: for every sample the k latent units with the largest |z| are kept
  (sign preserved), so strongly negative activations are not discarded.
* Tail reward: for long-tail training samples the achieved ||z_t|| is rewarded,
  capped at --reward-max so the norm cannot grow without bound.
* Normal samples are penalised for any z_t activity (beta * ||z_t||^2).

At inference no labels are used: a sample is scored by ||z_t||_2 and flagged as
long-tail above a threshold selected on the validation split.

Example:
    python scripts/sae_abstopk_tail_reward.py \\
        --features output/extract/layer28_mlp_output_last_token.npy \\
        --meta output/extract/meta.json \\
        --labels output/annotations/labels.json \\
        --output-dir output/sae_abstopk_tail_reward
"""

from sae_common import run

DEFAULTS = {
    "sparsity": "abstopk",
    "reward": "norm",
    "hidden_dim": 7168,
    "dropout": 0.2,
    "beta": 0.1,
    "reward_coeff": 0.5,
}

if __name__ == "__main__":
    run(__doc__, DEFAULTS)
