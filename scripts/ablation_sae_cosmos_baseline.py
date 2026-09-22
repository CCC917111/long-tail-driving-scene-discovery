#!/usr/bin/env python3
"""Ablation ("Train SAE Cosmos"): AbsTopK with the reward applied before selection.

Instead of rewarding the achieved ||z_t|| (final method), a constant bonus is
added to the z_t selection scores of long-tail training samples before the
Top-K step, which biases *which* units are kept but not their values. The bonus
is only used during training; inference is label-free, exactly as in the final
method. This variant also uses a smaller latent space (hidden_dim = input dim),
stronger dropout and a weaker normal-sample penalty, matching the original
experiment.

Example:
    python scripts/ablation_sae_cosmos_baseline.py \\
        --features output/extract/layer28_mlp_output_last_token.npy \\
        --meta output/extract/meta.json \\
        --labels output/annotations/labels.json \\
        --output-dir output/ablation_sae_cosmos_baseline
"""

from sae_common import run

DEFAULTS = {
    "sparsity": "abstopk",
    "reward": "pre_selection",
    "hidden_dim": 3584,
    "dropout": 0.4,
    "beta": 0.01,
    "reward_coeff": 1.0,
}

if __name__ == "__main__":
    run(__doc__, DEFAULTS)
