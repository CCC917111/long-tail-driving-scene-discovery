# Programming reference

`N` is the number of samples, `D` the feature dimension, `P` the SAE latent
width, `d_t` the tail-subspace width and `V = 6` the camera count. Current CPU
runs use `D = 3072`; the original Cosmos route uses `D = 3584` and a default
`P = 7168`. Run a script with `--help` for its complete CLI.

## Current CPU and data workflow

| Script | Purpose |
|---|---|
| [`prepare_driving.py`](../scripts/prepare_driving.py) | Select six-view timestamps, freeze scene allocation, create review files and import submitted judgments. |
| [`import_mini_parts.py`](../scripts/import_mini_parts.py) | Validate the browser helper's mini ZIPs and reconstruct the selected local data root. |
| [`acquire_cpu_encoder.py`](../scripts/acquire_cpu_encoder.py) | Acquire and verify the pinned official ResNet18 checkpoint. |
| [`extract_cpu.py`](../scripts/extract_cpu.py) | Extract six ordered 512-dimensional image vectors, concatenate them and resume a checked cache. |
| [`cpu_image_encoder.py`](../scripts/cpu_image_encoder.py) | Define the encoder, preprocessing and feature identity contract. |
| [`run_real_driving.py`](../scripts/run_real_driving.py) | Validate a prepared handoff and run the controlled four-method comparison. |
| [`new_scene_e2e.py`](../scripts/new_scene_e2e.py) | Acquire the recorded author-demo batch and apply existing CPU models to new inputs. |
| [`export_offline_explorer.py`](../scripts/export_offline_explorer.py) | Build the offline sample, score and sparse-unit viewer. |
| [`export_unlabelled_review.py`](../scripts/export_unlabelled_review.py) | Export a review interface for new inputs with blank decisions. |

See the [CPU route](cpu-route.md), [new-scene run](new-scene-e2e.md) and
[developer guide](developer.md) for commands and recorded execution scopes.

## VLM annotation and extraction

## `scripts/annotate_normal_core.py`

Command-line entry point. Labels keyframes `normal_core` / `not_normal_core` /
`uncertain` by prompting a VLM with the rubric in
[`latest_grading_criteria.md`](../latest_grading_criteria.md) and reading the
output-token logits of the three answers. Confidence is a model score, not
empirically calibrated human-label accuracy. Writes incrementally and resumes by default.

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

Outputs per layer `L`: `layer{L}_mlp_output_mean.npy` (the representation used
by the method) and `layer{L}_mlp_output_last_token.npy` (kept for the pooling
comparison), both `(N, D)` float32, plus a row-aligned `meta.json` carrying each
row's `sample_token`, `scene_token`, `scene_name`, `timestamp` and `images`, the
path of each camera view relative to `--samples-root`.

A sample is one keyframe with all six views present, in the order `CAM_FRONT`,
`CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`, `CAM_BACK`, `CAM_BACK_LEFT`,
`CAM_BACK_RIGHT`; keyframes with a missing view are skipped.

## `scripts/sae_common.py`

The shared library behind the three SAE entry points. Retained examples require
unique sample tokens and explicit scene tokens; missing scenes and unrecognised
labels are errors. See [strict inputs](user.md#strict-data-contract).

The common CLI requires `--review-k`; `--freeze-manifest-only` writes a manifest
without fitting, `--manifest` reuses a frozen split, and `--feature-config` retains
recovered extraction settings. [Evaluation](evaluation.md) documents provenance,
validation operating points and exported predictions.

| Name | Signature | Purpose |
|---|---|---|
| `NORMAL_LABELS`, `TAIL_LABELS`, `DROP_LABELS` | `set[str]` | Explicit whitelist for normal, tail and excluded labels |
| `build_parser(description, defaults)` | `-> ArgumentParser` | The common CLI, with per-variant defaults |
| `load_label_map(paths)` | `-> dict[str, int \| None]` | Reads both label formats; `None` means drop the row |
| `load_dataset(args)` | `-> (features, labels, groups, meta)` | Loads and joins features, labels and metadata |
| `split_indices(labels, groups, val_ratio, test_ratio, seed)` | `-> dict[str, ndarray]` | Greedily stratified split at group (scene) level |
| `LongTailGuidedSAE` | `nn.Module` | The model; see below |
| `train(model, x_train, y_train, x_val, y_val, args)` | `-> dict` | Training loop, keeps the best validation checkpoint |
| `encode_all(model, x, batch_size, device)` | `-> (z_n, z_t)` | Batched inference over a feature matrix |
| `classification_metrics(y, pred, score)` | `-> dict` | Per-class precision / recall / F1 of `pred`, AUC and AP of `score` |
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

They contain a `DEFAULTS` dictionary and a call to `run(__doc__, DEFAULTS)`.
The historical defaults change multiple variables, including width/dropout/beta
in the pre-selection variant; they do not constitute single-factor ablations.
Use one entry point with otherwise identical settings for reward none/norm.

| Script | Sparsity | Reward |
|---|---|---|
| `sae_abstopk_tail_reward.py` | `abstopk` | `norm` — reward the achieved `\|\|z_t\|\|`, capped |
| `ablation_topk_sae.py` | `topk` | `none` |
| `ablation_sae_cosmos_baseline.py` | `abstopk` | `pre_selection` — bonus added to the `z_t` selection scores before Top-K |

Each writes `metrics.json`, `neuron_report.csv`, `z_n_general.npy`,
`z_t_longtail.npy`, `is_tail.npy`, `activations.npz`, `rows.json`,
`standardizer.npz`, `best_model.pth` and `training_loss_curve.png` to
`--output-dir`. `metrics.json` records the configuration, the decision rule and
its `eta`, and the validation and test metrics; `rows.json` gives every frame's
split, label, prediction, number of active long-tail units and
`||z_t||_2`.

## `scripts/tail_activations.py`

The decision rule and the activation store. Needs numpy only.

| Name | Signature | Purpose |
|---|---|---|
| `tail_activation_count(z_t, eta=0.01)` | `(N, d_t) -> (N,)` | `c_tail`, the number of units with `\|z_t\| > eta` |
| `predict_long_tail(z_t, eta=0.01)` | `(N, d_t) -> (N,)` | 1 where `c_tail >= 1`, else 0 |
| `tail_score(z_t)` | `(N, d_t) -> (N,)` | `\|\|z_t\|\|_2`, the ranking score |
| `save_activations(path, z_t, meta, labels=None, splits=None)` | `-> None` | Writes the non-zero `z_t` entries in CSR form with per-frame metadata |
| `ActivationStore(path)` | class | Reads `activations.npz`: `row(i)`, `column(unit)`, `counts(eta)`, `dense()`, `samples`, `labels`, `splits` |
| `load_glossary(path)` | `-> dict[int, str]` | Reads a `neuron,feature` CSV |
| `CAMERAS` | `tuple[str]` | The six views in their fixed order |

## `scripts/screen.py`

Command-line entry point. Applies a trained run to new, unlabelled features.
[Screening Your Own Driving Data](apply.md) documents the workflow.

| Name | Purpose |
|---|---|
| `load_run(run_dir)` | Reads the SAE configuration and `eta` from `metrics.json` |
| `build_model(config, input_dim, device)` | Rebuilds the SAE exactly as the run configured it |
| `encode(model, x, batch_size, device)` | Returns `z_t` for every row |

Flags: `--run-dir`, `--features`, `--meta`, `--feature-config`, `--output-dir`,
`--glossary`, `--eta`, `--neuron-threshold`, `--top-neurons`, `--batch-size`,
`--device`.
Outputs `screening.csv` (flagged frames first, with reasons),
`long_tail_samples.json` (the flagged tokens) and `activations.npz` (for the
explorer).

The script rejects dimension/row mismatches, duplicate tokens, nonfinite
features and invalid standardisers. For a CPU-trained run, it also compares the
recorded encoder, pinned weights and preprocessing with `--feature-config` or
the adjacent `extraction_config.json`. This comparison checks declared feature
identity; it cannot authenticate arbitrary input arrays. Legacy/VLM runs retain
explicitly unverified feature identity beyond dimensional checks.

## `scripts/neuron_explorer.py`

Command-line entry point with four sub-commands: `serve` (local web interface),
`unit ID` (top frames of a unit), `sample TOKEN` (active units of a frame) and
`units` (units ranked by how many frames they fire on). Common flags: `--run`,
`--samples-root`, `--glossary`, `--eta`, `--neuron-threshold` (default 1.0),
and `--nuscenes-meta` for runs whose `meta.json` carries no image paths.
Needs numpy and Pillow only.

| Name | Purpose |
|---|---|
| `Explorer(store, samples_root, glossary, eta, neuron_threshold)` | Both lookup directions over one `ActivationStore` |
| `Explorer.frames_of_unit(unit, top, min_activation)` | Frames with the largest `\|z_t\|` on a unit |
| `Explorer.units_of_frame(token, top)` | Active units of a frame, strongest first |
| `Explorer.units(top)` / `unit_stats(unit)` | Frames per unit and, with labels, long-tail purity |
| `Explorer.image_path(token, camera)` | Image of one view; never resolves outside `--samples-root` |
| `make_server(explorer, host, port)` | The web interface (`/`, `/api/...`, `/img/...`) |
| `unit_sheet(...)`, `frame_sheet(...)` | Contact sheets for `--save` |

## `results/neuron_glossary.csv`

This historical glossary contains proposed names for layer-28 units.
`screen.py` reads `neuron` and `feature`; the other columns record supporting
counts and notes. These names have not been validated for the current CPU
models and must not be applied to their unit IDs. The [explorer guide](neurons.md)
describes how to build a checkpoint-specific glossary.

## Controlled comparison tools

`data_manifest.py` validates frozen splits; `supervised_baselines.py` trains
Logistic/MLP on the same contract; `evaluate_runs.py` exports metrics and review
queues; `summarize_runs.py` checks and rebuilds tables.
`evaluation_fixture.py` and `synthetic_demo.py` generate synthetic software
examples. See [evaluation](evaluation.md) and [verification](testing.md) for
commands, artifacts and what has actually run.
