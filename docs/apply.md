# Screening driving data

`scripts/screen.py` ranks new six-camera samples for human review using an
existing SAE checkpoint. It exports sample IDs, scores and active unit IDs so a
reviewer can inspect candidates and build an annotation queue. This is a data
selection tool; candidate status does not establish a hazard or a safety outcome.

The current real-data route uses [CPU ResNet18 features](cpu-route.md). Its
[new-scene replay](../results/new-scene-human-review-20261028/README.md) applied
models trained on the old mini cohort to nine new timestamps. The old
validation-selected threshold of `6.074954873` selected five candidates, with
60% precision and 75% recall against the submitted references. Those judgments
followed chat AI-reference disclosure, and class and scene coincide in this
batch. The result is a descriptive check, not a generalization estimate.

## What you need

- A trained SAE directory containing `best_model.pth`, `standardizer.npz` and
  `metrics.json`. Preserve its configuration and provenance files too.
- New features extracted with the same encoder, weights, camera order and
  preprocessing as the training run, plus row-aligned metadata.
- For a CPU run, the matching `extraction_config.json`. The screener compares
  the declared feature protocol with training. It cannot authenticate arbitrary
  arrays supplied alongside that configuration.

CPU extraction and screening require no GPU. A VLM checkpoint instead requires
its original extraction contract and enough GPU memory for that model. Features
from the two routes are incompatible even if filenames look similar.

## Input format

| Field | Requirement |
|---|---|
| Sample | One timestamp, without temporal context. |
| Views | Exactly six synchronized views in this order: `CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`, `CAM_BACK`, `CAM_BACK_LEFT`, `CAM_BACK_RIGHT`. |
| Features | One finite float32 row per sample, with the training run's dimension and representation. Current CPU features have 3,072 dimensions. |
| Metadata | Unique `sample_token`, scene identifier and six image paths, aligned with the feature rows. |
| Normalization | Use the saved training standardizer. Do not fit a new one on the screening batch. |

The CPU extractor resizes each image to a 256-pixel short side, takes a 224-pixel
center crop and concatenates six 512-dimensional vectors. It records encoder,
weights, preprocessing and image hashes. See [the CPU protocol](cpu-route.md)
for the full contract and its limits.

## Running it

For a completed CPU cache, use the existing helper:

```bash
bash scripts/cloud.sh cpu-screen OUTPUT=output/real-cpu FEATURES=path/to/new/extract
```

Or invoke the screener directly:

```bash
python scripts/screen.py \
    --run-dir output/real-cpu/comparison_runs/sae_abstopk_norm \
    --features path/to/new/extract/layer1_mlp_output_mean.npy \
    --meta path/to/new/extract/meta.json \
    --feature-config path/to/new/extract/extraction_config.json \
    --output-dir output/screening \
    --device cpu
```

Use the actual trained run path. The CPU `layer1_mlp_output_mean.npy` name is a
storage compatibility alias; it is not a language-model layer or token average.
Labels are unnecessary for new-input scoring, although SAE fitting uses labels.

The screener uses the original activation rule: a sample is flagged when at
least one tail unit has `|z_t| > eta`, with `eta` read from `metrics.json`
(default `0.01`). It ranks by tail-code norm. This rule is separate from the
validation-selected score threshold used in evaluation. In the nine-row replay,
the original activation rule selected all nine; the frozen validation threshold
selected five. Changing `--eta` creates a different screening policy and must
be recorded. Do not tune either rule against the new evaluation labels.

## What you get back

`screening.csv` contains one row per sample, with flagged samples first:

| Column | Meaning |
|---|---|
| `rank` | Position in the output, starting at 1. |
| `sample_token` | Identifier from `meta.json`. |
| `scene` | Scene name or token for grouping. |
| `prediction` | `long_tail` if a tail unit passes `eta`; otherwise `normal`. |
| `active_tail_units` | Number of tail units with `\|z_t\| > eta`. |
| `tail_score` | Tail-code L2 norm, used for ranking. |
| `strong_units` | Number of units above the inspection threshold, default `\|z_t\| > 1`. |
| `top_neurons` | Strongest unit IDs above that inspection threshold. |
| `reasons` | Optional names from a checkpoint-specific glossary. |

`long_tail_samples.json` contains the flagged IDs. `activations.npz` opens in
the [explorer](neurons.md#browse-a-trained-run):

```bash
python scripts/neuron_explorer.py serve --run output/screening \
    --samples-root path/to/data/samples
```

Current CPU units use numeric IDs. Historical Cosmos glossary names have not
been validated for these models and must not be copied onto them. Even a
checkpoint-specific name remains an interpretation requiring review; it is not
a verified explanation of a prediction.

## Working with the result

Choose a review budget before measuring yield, inspect the top-ranked samples
and retain uncertain decisions. Report the selected count and class support
alongside precision and recall. For a new corpus, measure yield with new
references rather than carrying forward an earlier precision estimate.

Grouping by scene can reveal repeated candidates from one short sequence. A
queue dominated by neighboring timestamps may offer less diversity than its
size suggests. Inspect real false positives and false negatives as well as
high-scoring examples before deciding whether the ranking is useful.

## Original VLM extraction route

The VLM route remains available but is separate from the completed CPU runs.
Use the training model revision, prompt, layer and pooling. The original
configuration uses layer 28 mean pooling and 3,584-dimensional Cosmos features.
Matching dimensions alone does not establish representation compatibility.

```bash
python scripts/extract.py \
    --samples-root <your_data>/samples --meta-tgz <your_data>/meta.tgz \
    --layers 28 --output-dir output/extract_mydata

make screen RUN=output/sae_abstopk_tail_reward FEATURES=output/extract_mydata
```

The equivalent screening command is:

```bash
python scripts/screen.py \
    --run-dir output/sae_abstopk_tail_reward \
    --features output/extract_mydata/layer28_mlp_output_mean.npy \
    --meta output/extract_mydata/meta.json \
    --glossary results/neuron_glossary.csv \
    --output-dir output/screening
```

Use a glossary only after verifying that it belongs to that checkpoint. The
original extractor accepts Pillow-readable images, resizes within its configured
pixel budget (default 3,136–1,600,000 pixels) and skips keyframes missing a view.
Its output is a feature matrix and row-aligned `meta.json`. Historical wheelchair
examples and raw-VLM comparisons remain unverified from original outputs; see
[the evidence record](../results/README.md).

## Adapting it to your own definition of "rare"

The learned target follows the training labels. To change it, version the
[rubric](../latest_grading_criteria.md), collect new judgments and fit a new run
under a new frozen manifest. Reuse features only when the extraction contract
is unchanged, and evaluate their suitability for the new target. See
[common tasks](tasks.md) for backbone, camera-rig and glossary changes.
