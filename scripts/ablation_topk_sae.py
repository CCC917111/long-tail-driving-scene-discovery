#!/usr/bin/env python3
"""Ablation ("TopK SAE"): signed Top-K sparsity and no tail reward.

Same architecture and training protocol as scripts/sae_abstopk_tail_reward.py,
but the k largest *signed* activations are kept (negative activations are
dropped) and there is no reward term for long-tail samples; the long-tail
subspace is shaped only by the normal-sample penalty and the up-weighted tail
reconstruction.

Because this variant differs from the final method in two ways, the effect of
the sparsity rule alone can be isolated with:
    python scripts/ablation_topk_sae.py --reward norm ...

Example:
    python scripts/ablation_topk_sae.py \\
        --features output/extract/layer28_mlp_output_last_token.npy \\
        --meta output/extract/meta.json \\
        --labels output/annotations/labels.json \\
        --output-dir output/ablation_topk_sae
"""

from sae_common import run

DEFAULTS = {
    "sparsity": "topk",
    "reward": "none",
    "hidden_dim": 7168,
    "dropout": 0.2,
    "beta": 0.1,
    "reward_coeff": 0.5,
}

if __name__ == "__main__":
    run(__doc__, DEFAULTS)
