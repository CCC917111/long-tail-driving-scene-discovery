Programming Reference
=============
Welcome to the long-tail driving data mining project programming reference page!

This page lists what every script provides. `N` is the number of frames, `D`
the hidden dimension of the VLM layer, `P` the SAE latent width and `V` the
number of camera views.

## `scripts/annotate_normal_core.py`

Command-line entry point. Labels keyframes `normal_core` / `not_normal_core` /
`uncertain` by prompting a VLM with the rubric in
[`latest_grading_criteria.md`](../latest_grading_criteria.md) and reading the
output-token logits of the three answers, so the confidence is calibrated
rather than parsed from prose. Writes incrementally and resumes by default.

Key flags: `--samples-root`, `--meta-tgz`, `--output`. Run with `--help` for
the full list.

## `scripts/extract.py`

Command-line entry point. Feeds the `V` synchronised views of every keyframe to
the VLM with the scene-description prompt and captures the MLP block output
(`down_proj`) of the requested decoder layers through forward hooks.

| Flag | Purpose |
|---|---|
| `--layers` | Decoder layers to capture, 1-indexed; several in one pass |
| `--sample-tokens` | Restrict extraction to the tokens in a label file |
| `--caption` | Also generate and store the scene description text |
| `--max-pixels` / `--min-pixels` | Resolution budget per image |
| `--max-samples` | Stop after N samples, for a smoke test |

Outputs per layer `L`: `layer{L}_mlp_output_mean.npy` and
`layer{L}_mlp_output_last_token.npy`, both `(N, D)` float32, plus a row-aligned
`meta.json` carrying each row's `sample_token` and scene.

## `scripts/sae_common.py`

The shared library behind the three SAE entry points.

| Name | Signature | Purpose |
|---|---|---|
| `NORMAL_LABELS`, `DROP_LABELS` | `set[str]` | The label protocol: what counts as normal, what is dropped |
| `build_parser(description, defaults)` | `-> ArgumentParser` | The common CLI, with per-variant defaults |
| `load_label_map(paths)` | `-> dict[str, int \| None]` | Reads both label formats; `None` means drop the row |
| `load_dataset(args)` | `-> (features, labels, groups, meta)` | Loads and joins features, labels and metadata |
| `split_indices(labels, groups, val_ratio, test_ratio, seed)` | `-> dict[str, ndarray]` | Greedily stratified split at group (scene) level |
| `LongTailGuidedSAE` | `nn.Module` | The model; see below |
| `train(model, x_train, y_train, x_val, y_val, args)` | `-> dict` | Training loop, keeps the best validation checkpoint |
| `encode_all(model, x, batch_size, device)` | `-> (z_n, z_t)` | Batched inference over a feature matrix |
| `best_f1_threshold(y, score)` | `-> float` | Threshold at maximum F1 on the given split |
| `classification_metrics(y, score, threshold)` | `-> dict` | AUC, AP and per-class precision / recall / F1 |
| `activation_stats(name, z_n, z_t, y, threshold)` | `-> None` | Prints how often each subspace fires, per class |
| `neuron_report(z_t, y, threshold, path)` | `-> None` | Per-unit purity and coverage, written as CSV |
| `run(description, defaults)` | `-> None` | The whole pipeline; what the entry points call |

`LongTailGuidedSAE(input_dim, hidden_dim, tail_ratio, dropout, k, sparsity,
reward, alpha, beta, reward_coeff, reward_max)`:

- `encode(x, is_tail=None, apply_reward=False)` returns `(z, z_n, z_t)` for an
  `(B, D)` input. `z` is `(B, P)` with at most `k` non-zero entries; `z_n` and
  `z_t` are the two halves, `(B, P - tail_dim)` and `(B, tail_dim)`.
- `forward(x, is_tail, apply_reward=False)` returns
  `(total_loss, parts, z_n, z_t, x_hat)`, where `parts` holds `total`,
  `recon`, `normal` and `tail_reward` separately, so each term of the objective
  can be logged on its own.
- `k > hidden_dim` raises `ValueError`.
- `sparsity` is `abstopk` (keep the largest `|z|`) or `topk` (largest signed
  `z`); `reward` is `none`, `norm` or `pre_selection`.

## The three SAE entry points

They contain only a `DEFAULTS` dictionary and a call to `run(__doc__,
DEFAULTS)`, so any behavioural difference between them is a hyper-parameter.

| Script | Sparsity | Reward |
|---|---|---|
| `sae_abstopk_tail_reward.py` | `abstopk` | `norm` — reward the achieved `\|\|z_t\|\|`, capped |
| `ablation_topk_sae.py` | `topk` | `none` |
| `ablation_sae_cosmos_baseline.py` | `abstopk` | `pre_selection` — bonus added to the `z_t` selection scores before Top-K |

Each writes `metrics.json`, `neuron_report.csv`, `z_n_general.npy`,
`z_t_longtail.npy`, `is_tail.npy`, `rows.json`, `standardizer.npz`,
`best_model.pth` and `training_loss_curve.png` to `--output-dir`.

## `scripts/screen.py`

Command-line entry point. Applies a trained run to new, unlabelled features.
[Screening Your Own Driving Data](apply.md) documents the workflow.

| Name | Purpose |
|---|---|
| `load_run(run_dir)` | Reads the SAE configuration and validation threshold from `metrics.json` |
| `build_model(config, input_dim, device)` | Rebuilds the SAE exactly as the run configured it |
| `load_glossary(path)` | Reads a `neuron,feature` CSV into a mapping |
| `encode(model, x, batch_size, device)` | Returns `z_t` for every row |

Flags: `--run-dir`, `--features`, `--meta`, `--output-dir`, `--glossary`,
`--threshold`, `--neuron-threshold`, `--top-neurons`, `--batch-size`,
`--device`. Outputs `screening.csv` (ranked, with reasons) and
`long_tail_samples.json` (the flagged tokens).

The script stops with an explanatory error if the feature dimension does not
match the run, or if `meta.json` and the feature matrix disagree on the number
of rows.

## `scripts/pipeline.py`

A clip-level variant of the same idea: a continuous VLM-derived tail score
instead of binary labels, a deeper encoder (`DeepLongTailSAE`), a pairwise
margin loss, mixup and tail oversampling. Self-contained, with its own CLI.

## `results/neuron_glossary.csv`

The named long-tail units of the layer-28 run. `screen.py` reads `neuron` and
`feature`; `dataset`, `activation_ratio`, `purity`, `n_activated`,
`n_activated_target` and `notes` carry the evidence for each name.
[The Interpretable Neurons](neurons.md) explains how to rebuild it.
