User page
=============
Welcome to the long-tail driving data mining project user main page!

This page is the practical guide to running the pipeline end to end: what you
need, what each command does, and what it writes where. Applying a trained
model to your own frames has its own page,
[Screening Your Own Driving Data](apply.md).

## Requirements

- Python 3.10 or newer, and transformers 4.49 or newer for Qwen2.5-VL support.
- A CUDA GPU for the two VLM stages — bf16 inference, 24 GB VRAM or more is
  comfortable. The SAE training and the screening step also run on CPU.
- Cosmos-Reason1-7B weights:
  https://huggingface.co/nvidia/Cosmos-Reason1-7B
- nuScenes metadata (`v1.0-trainval_meta.tgz`) and keyframe images
  (`v1.0-trainval{01..10}_keyframes.tgz`), extracted so that `samples/CAM_*/`
  exists: https://www.nuscenes.org/nuscenes

```bash
python -m venv .venv && source .venv/bin/activate
make install                       # pip install -r requirements.txt
pip install pytest flake8          # only needed for make test / make lint
```

Every `make` target takes overrides, so a different data root or layer is a
variable rather than an edit:

```bash
make extract DATA=/mnt/nuscenes LAYERS="15 18 21 24 27 28"
```

## 1. Labelling

```bash
make label DATA=<nuscenes>
```

which is

```bash
python scripts/annotate_normal_core.py \
    --samples-root <nuscenes>/samples \
    --meta-tgz <nuscenes>/v1.0-trainval_meta.tgz \
    --output output/annotations/labels.json
```

All six synchronised camera views of a keyframe go to the VLM together with the
normal-core rubric. The label and its confidence are read from the
output-token logits of the three possible answers rather than parsed out of
free text, so the confidence is calibrated rather than rhetorical. The run
writes incrementally and resumes where it stopped.

Output: `labels.json`, one entry per keyframe with the sample token, the label
(`normal_core` / `not_normal_core` / `uncertain`) and the confidence.

## 2. Feature extraction

```bash
make extract DATA=<nuscenes> LAYERS=28
```

which is

```bash
python scripts/extract.py \
    --samples-root <nuscenes>/samples \
    --meta-tgz <nuscenes>/v1.0-trainval_meta.tgz \
    --sample-tokens output/annotations/labels.json \
    --layers 28 \
    --output-dir output/extract
```

The same six views go through the VLM again, this time with the open-ended
scene-description prompt, and forward hooks capture the MLP block output
(`down_proj`) of each requested decoder layer. `--layers` is 1-indexed, so `28`
is the last block of the 7B language model. Several layers can be extracted in
one pass, which is how the per-layer sweep in the results was produced.

Outputs per layer, plus one shared index:

| File | Contents |
|---|---|
| `layer{L}_mlp_output_mean.npy` | `(N, D)` float32, mean-pooled over tokens |
| `layer{L}_mlp_output_last_token.npy` | `(N, D)` float32, last-token representation |
| `meta.json` | `N` row-aligned entries with `sample_token` and scene |

`meta.json` is what lets the SAE split by scene instead of by sample, so keep
it with the features.

## 3. Training the SAE

```bash
make train
```

which is

```bash
python scripts/sae_abstopk_tail_reward.py \
    --features output/extract/layer28_mlp_output_mean.npy \
    --meta output/extract/meta.json \
    --labels output/annotations/labels.json \
    --output-dir output/sae_abstopk_tail_reward
```

The flags worth knowing — run any script with `--help` for the rest:

| Flag | Default | Effect |
|---|---|---|
| `--hidden-dim` | per variant | Number of SAE latent units |
| `--tail-ratio` | 0.5 | Fraction of units reserved for `z_t` |
| `--k` | 512 | Units kept per sample by the sparsity rule |
| `--sparsity` | `abstopk` | `abstopk` keeps the largest `\|z\|`, `topk` the largest signed `z` |
| `--reward` | per variant | `none`, `norm` (reward the achieved `\|\|z_t\|\|`), `pre_selection` |
| `--alpha` | 1.0 | Extra reconstruction weight for tail samples |
| `--beta` | per variant | Weight of the `\|\|z_t\|\|^2` penalty on normal samples |
| `--reward-coeff` | per variant | Strength of the tail reward |
| `--reward-max` | 2.0 | Cap of the tail reward, `tau` in the loss |
| `--val-ratio` / `--test-ratio` | 0.15 | Split sizes, taken at scene level |
| `--epochs` | 200 | Training epochs; the best validation loss is kept |
| `--neuron-threshold` | 1.0 | `\|z_t\|` threshold for the interpretability report |
| `--seed` | 42 | Seed for the split and the initialisation |

`--labels` accepts several files and both label formats: the JSON array written
by the labeller, and `{"samples": [{"sample_token", "labels": {"label"}}]}`.
`normal` and `normal_core` count as normal, `uncertain` / `unknown` / empty are
excluded, and every other label counts as long-tail.

Written to `--output-dir`:

- `metrics.json` — validation and test AUC / AP / precision / recall / F1, the
  chosen threshold, and the full configuration of the run.
- `neuron_report.csv` — per-unit activation counts, purity and coverage.
- `z_n_general.npy`, `z_t_longtail.npy` — the learned sparse codes.
- `rows.json` — per-sample token, split assignment, label and score.
- `standardizer.npz`, `best_model.pth` — everything `screen.py` needs.
- `training_loss_curve.png`.

## 4. Ablations

```bash
make ablations
```

Runs the two variants against the same features, labels and split, so the
comparison in [`results/README.md`](../results/README.md) isolates the sparsity
rule and the reward placement rather than confounding them with data changes.

## 5. Screening new data

```bash
make screen RUN=output/sae_abstopk_tail_reward FEATURES=output/extract_mydata
```

See [Screening Your Own Driving Data](apply.md) for the input contract and the
output columns.

## Tests and style

```bash
make test          # unit tests, CPU only, no dataset or VLM needed
make lint          # flake8
make clean         # __pycache__ and .pytest_cache
```

## Troubleshooting

**The extraction run is slower than expected.** Six images per sample through a
7B VLM is the cost. `--max-pixels` trades resolution for throughput, and
`--max-samples` is useful for a smoke test before committing to a full pass.

**Out of memory during extraction.** Lower `--max-pixels` first. Extraction
holds one sample at a time, so batch size is not the lever here.

**A split does not contain both classes.** The SAE scripts stop rather than
train on a degenerate split. Raise `--val-ratio` / `--test-ratio`, or change
`--seed`: with few labelled scenes, a greedy scene-level split can put all the
long-tail scenes on one side.

**`screen.py` reports a dimension mismatch.** The features being screened come
from a different layer or pooling than the run was trained on. Extract them
with the same `--layers` value and the same `_mean` / `_last_token` file.
