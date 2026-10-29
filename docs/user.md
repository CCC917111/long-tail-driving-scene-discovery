# Running the pipeline

Use the [CPU synthetic quickstart](../README.md#cpu-quickstart) to try the
software without driving data or a GPU. For real six-camera images, start with
the completed [ResNet18 CPU route](cpu-route.md). This guide describes the
original VLM-feature interfaces and the shared training workflow. The
[verification record](feedback-status.md#verification-record) distinguishes
completed runs from setup templates.

## Environment and real assets

The CPU quickstart was verified with Python 3.12; its pinned package versions
are in `requirements-cpu-lock.txt`. For VLM labelling/extraction, use a
CUDA-compatible environment with the GPU requirements described in
[from zero](from-zero.md). Labelling and extraction require the model weights
and six-camera images; SAE training, cached-feature evaluation and the explorer can
run on CPU. GPU memory and runtime should be measured on a small batch before
scheduling extraction. The old 24 GB recommendation is not a measured bound
for every image resolution or dependency version.

To reproduce the historical VLM experiment, recover these assets:

- Authorised nuScenes v1.0-trainval metadata and keyframes, with `samples/CAM_*/`.
- Original labels and manual review records, with sample and scene tokens.
- Frozen VLM features and row-aligned `meta.json`, plus model revision, layer,
  pooling, prompt/rubric versions and input hashes where recoverable.
- Original checkpoints, standardisers, run configurations, predictions and logs
  when evaluating historical claims.

The repository excludes original camera images and trained weights. The
current CPU results preserve derived evidence separately from private inputs;
source availability does not grant permission to redistribute dataset images.

## Labelling and feature extraction

These are templates for a configured real-data environment, not a claim that
VLM inference has run in this round:

```bash
python scripts/annotate_normal_core.py \
  --samples-root /path/to/nuscenes/samples \
  --meta-tgz /path/to/nuscenes/v1.0-trainval_meta.tgz \
  --output output/annotations/labels.json
python scripts/extract.py \
  --samples-root /path/to/nuscenes/samples \
  --meta-tgz /path/to/nuscenes/v1.0-trainval_meta.tgz \
  --sample-tokens output/annotations/labels.json \
  --layers 28 --output-dir output/extract
```

Both scripts accept `--model-dir`. The labeller reads answer-token logits;
its confidence is a model score, not empirically calibrated human accuracy.
Independent human review is a separate operation. Preserve label sources,
review decisions and exclusions.

Extraction uses one timestamp and exactly six views, in this order:
`CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`, `CAM_BACK`, `CAM_BACK_LEFT`,
`CAM_BACK_RIGHT`. A keyframe with missing views is skipped. It stores mean and
last-token MLP-output features per requested layer, plus row-aligned tokens,
scene metadata, timestamps and image paths in `meta.json`. Keep the files
together. A new extraction does not establish the provenance of an old run.

## Strict data contract

The feature matrix must be finite, nonempty `(N, D)` data with `N` metadata
rows. Tokens must be unique and nonempty. Every retained labelled sample needs
a nonempty **`scene_token`**; `scene_name` alone is insufficient, and there is
no sample-level fallback. Recover missing scene metadata before training.

The label loader accepts an array of rows or `{"samples": [...]}`, with a
`label` field or nested `labels.label`. Labels are trimmed and lowercased:

| Values | Meaning |
|---|---|
| `normal`, `normal_core` | Normal, 0 |
| `not_normal_core`, `long_tail`, `long-tail`, `tail` | Tail, 1 |
| `uncertain`, `unknown`, `none`, empty or missing label | Excluded, not a tail example |
| Any other value | Error |

Duplicate label tokens, including repetitions across label files, and
conflicting labels are errors. Missing/unusable labels are recorded in
`excluded_rows.json` when a run directory is provided. Inspect the exclusions
before freezing a protocol; do not silently translate an unfamiliar label to
positive.

## Freeze one manifest

Choose the review budget before reading test results. Here `100` is an example
budget to replace with the study's predeclared K. This command validates the
inputs and freezes a scene-level 70/15/15 assignment without fitting a model:

```bash
python scripts/sae_abstopk_tail_reward.py \
  --features output/extract/layer28_mlp_output_mean.npy \
  --meta output/extract/meta.json --labels output/annotations/labels.json \
  --review-k 100 --freeze-manifest-only --seed 42 \
  --output-dir output/protocol
```

Review `output/protocol/manifest.json`, including actual scene/frame counts,
class support and exclusions. Reuse this exact manifest for every method and
model seed. An explicitly supplied `--manifest` is optional for initial
creation, but required by the comparison workflow once frozen. Do not generate
a different split when changing training seeds.

Use `--feature-config /path/to/recovered_extraction_config.json` on evaluated
runs when a real extraction record is available. Its contents and hash are
retained; without it the extraction provenance is explicitly unverified. Do
not invent a config that appears to document missing historical settings.

## Train the matched core models

The full configuration and evaluator outputs are described in
[controlled evaluation](evaluation.md). For the two core SAE groups use the
same entry point and arguments, changing only `--reward` and output directory:

```bash
python scripts/sae_abstopk_tail_reward.py \
  --features output/extract/layer28_mlp_output_mean.npy \
  --meta output/extract/meta.json --labels output/annotations/labels.json \
  --manifest output/protocol/manifest.json --review-k 100 --seed 17 \
  --reward none --output-dir output/sae_none_seed17
python scripts/sae_abstopk_tail_reward.py \
  --features output/extract/layer28_mlp_output_mean.npy \
  --meta output/extract/meta.json --labels output/annotations/labels.json \
  --manifest output/protocol/manifest.json --review-k 100 --seed 17 \
  --reward norm --output-dir output/sae_norm_seed17
```

Train-only statistics standardise the features; validation loss selects the
checkpoint. `--review-k` is required. `--target-precision` defaults to 0.9,
and `--bootstrap-samples` to 1,000. Target selection uses validation only;
its test precision is not guaranteed. Use separate output directories for
new runs. Historical variant entry points also require K but have confounded
defaults; they are not the controlled reward/no-reward comparison.

The Makefile offers `manifest`, `train`, `train-no-reward` and `baselines`
helpers, each requiring an explicit `REVIEW_K`. Its default manifest path is
`output/protocol/manifest.json`. Inspect flags before long jobs. The VLM-feature
commands above are templates; the completed real-image CPU fits use the
separately documented ResNet18
feature configuration.

SAE runs retain the model/standardiser, sparse activations, per-sample rows,
manifest and exclusions, metrics and evaluation artifacts. `neuron_report.csv`
is a training-label inspection report, not independent semantic test purity.
The [result-bundle contract](evaluation.md#result-bundle) lists provenance,
FP/FN markers, PR curves, review queues and table-rebuilding requirements.

## Browse and screen

For a recovered trained run with authorised images:

```bash
python scripts/neuron_explorer.py serve \
  --run output/sae_norm_seed17 --samples-root /path/to/nuscenes/samples \
  --glossary /path/to/that_run_glossary.csv
```

The page defaults to `http://127.0.0.1:8765`. It links units to high-activation
frames and frames to active units. `units`, `unit` and `sample` subcommands
provide terminal queries; see [the explorer guide](neurons.md). Historical unit
IDs and names are not transferable to newly trained weights: create and
validate a glossary for each run.

`screen.py` applies a trusted SAE checkpoint and its standardiser to new
features. Extraction model, revision, layer, pooling and preprocessing must
match; dimension equality alone does not prove compatibility. See
[screening inputs](apply.md). The evaluator's review queues deliberately omit
reference labels so reviewers can make independent decisions.

## Troubleshooting

Missing scenes, duplicated tokens or unknown labels are data errors to fix,
not reasons to weaken validation. A split lacking either class cannot support
the current metrics: inspect the eligible scenes and protocol before choosing
a new split using train/validation considerations. Never choose a seed because
its test score looks better. Missing explorer images usually mean the wrong
`samples` root or paths in metadata. A blocked socket in the execution sandbox
prevents the HTTP test; it does not verify the server route.
