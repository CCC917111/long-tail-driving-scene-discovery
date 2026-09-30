Screening Your Own Driving Data
=============
Welcome to the long-tail driving data mining project application page!

The benchmark answers "how good is this method". This page answers the question
a team with a drive log actually has: **here are a hundred thousand frames
nobody has looked at — which ones should a human annotate first, and why?**

## What the tool is for

You have driving data and a limited annotation budget. Labelling everything is
impossible, labelling at random spends the budget on the same lane-following
frames over and over, and the frames that matter — the ones that would require
the ego vehicle to slow down, yield or keep a larger margin — are rare by
definition.

`scripts/screen.py` takes your frames, flags every frame on which at least one
learned long-tail unit fires, and lists the flagged frames first, strongest
first, each with the units that fired and, where those units have names, a
plain-language reason. You annotate from the top of that list.

What it is not: a perception model or a safety monitor. It does not detect
objects, and it makes no claim about a frame beyond "this looks like the kind of
situation the SAE learned to associate with defensive driving". It is a
prioritisation tool for a data pipeline.

Two properties are worth knowing before you choose it over the alternatives,
because they are what it was built for:

- It finds samples that are **not** feature-space outliers. The wheelchair frame
  in the case study ranks above only 1.4% of samples by KNN outlier degree —
  globally it looks like an ordinary urban street, and what makes it matter is
  one local detail.
- It finds samples the model is **confident** about. Low-confidence filtering
  selects that frame in 0 of 20 trials. Uncertainty and long-tail value are not
  the same axis.

## What you need

- A trained SAE run directory, produced by `make train` (or by running
  `scripts/sae_abstopk_tail_reward.py` directly). It contains `best_model.pth`,
  `standardizer.npz` and `metrics.json`; the repository ships the code, not the
  weights.
- The same VLM used for that run, to extract features from your frames.
- A CUDA GPU for the extraction step (bf16 inference, 24 GB or more is
  comfortable). The screening step itself runs on CPU in seconds.

## Input format

The frames go through `scripts/extract.py`, exactly as in training, so the
input contract is that script's contract:

| What | Requirement |
|---|---|
| Sample | One keyframe: a single timestamp, no temporal context. A drive is screened frame by frame. |
| Views | Exactly six synchronised cameras per sample, in the nuScenes layout and order: `samples/CAM_FRONT/<file>.jpg`, then `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`, `CAM_BACK`, `CAM_BACK_LEFT`, `CAM_BACK_RIGHT`. A keyframe with a missing view is skipped. |
| Images | Any format Pillow reads; each view is resized to between 3,136 and 1,600,000 pixels. |
| Index | A metadata source mapping a `sample_token` to its six image paths and its scene — the nuScenes `v1.0-*` tables for nuScenes-format data. |
| Layer | The same `--layers` value as the training run, `28` for the reported run. |
| Pooling | Mean over tokens (`*_mean.npy`), as in training. |

A run trained on this six-camera rig is applied to data from the same rig; the
prompt names the six views in this order, so the order is part of the input.

Extraction writes two files that travel together:

```
layer28_mlp_output_mean.npy   (N, 3584) float32, one row per frame
meta.json                     N entries, row-aligned: sample_token, scene, image paths
```

`screen.py` checks that `D` matches the run's input dimension and that
`meta.json` has as many rows as the feature matrix, and stops with an
explanatory error rather than silently producing nonsense if either is wrong.

## Running it

```bash
# 1. Extract features from your frames with the same VLM, layer and pooling
python scripts/extract.py \
    --samples-root <your_data>/samples --meta-tgz <your_data>/meta.tgz \
    --layers 28 --output-dir output/extract_mydata

# 2. Screen them with a trained SAE run
make screen RUN=output/sae_abstopk_tail_reward FEATURES=output/extract_mydata
```

which is

```bash
python scripts/screen.py \
    --run-dir output/sae_abstopk_tail_reward \
    --features output/extract_mydata/layer28_mlp_output_mean.npy \
    --meta output/extract_mydata/meta.json \
    --glossary results/neuron_glossary.csv \
    --output-dir output/screening
```

No labels are needed at this point. Labels are only used to *train* the SAE; at
screening time the method is unsupervised.

The decision is the rule used in training: a frame is flagged when at least one
long-tail unit has `|z_t| > eta`, with `eta` read from the run's
`metrics.json` (0.01). Nothing is re-tuned on your data. `--eta` changes the
floor if you want to experiment; to shorten the list instead, take the top of
the ranking or filter on the units you care about.

## What you get back

`screening.csv`, one row per frame, flagged frames first:

| Column | Meaning |
|---|---|
| `rank` | Position in the list, 1 = the most strongly flagged frame |
| `sample_token` | Frame identifier, carried over from `meta.json` |
| `scene` | Scene name or token, so you can group frames by drive |
| `prediction` | `long_tail` if at least one long-tail unit is active, else `normal` |
| `active_tail_units` | `c_tail`, the number of long-tail units with `\|z_t\| > eta` |
| `tail_score` | `\|\|z_t\|\|_2`, how strongly the long-tail subspace responds; orders the list |
| `strong_units` | How many units fire above the interpretability threshold (`\|z_t\| > 1`) |
| `top_neurons` | The strongest of those unit indices |
| `reasons` | Their names, for the units that have one |

plus `long_tail_samples.json`, just the flagged tokens, ready to be handed to an
annotation tool, and `activations.npz`, which the
[neuron explorer](neurons.md#browsing-the-units) opens directly:

```bash
python scripts/neuron_explorer.py serve --run output/screening \
    --samples-root <your_data>/samples --glossary results/neuron_glossary.csv
```

The console prints the flagged share and a tally of the reasons:

```
screened 12480 frames | flagged 1163 (9.3%) as long-tail

why they were flagged:
    612  Rain / wet road
     97  Glare on a wet or rainy road
     41  Pedestrians crossing a construction zone
     14  Wheelchair user ahead
```

That tally is often the first genuinely useful output: it is a profile of what
kinds of rare situations your archive contains, computed without anyone having
labelled it.

## Working with the result

- **Spend the annotation budget top-down.** Take the first N rows of
  `screening.csv`. With the reported long-tail precision of 0.90, most of what
  a human opens is worth opening.
- **Fill a specific gap.** If your model fails on wet-road glare, filter on the
  unit for it rather than on the score — see
  [The Interpretable Neurons](neurons.md). The units act as retrieval keys for
  categories nobody labelled in advance.
- **Compare drives.** Group by `scene` and compare flagged shares to find which
  collection routes actually contribute rare data and which ones repeat what
  you already have.
- **Look at the units, not only the score.** A frame that activates several
  unrelated units is often a genuinely compound scene; a frame with one very
  strong unit is usually a clean example of that one category. The explorer's
  frame view shows this at a glance.

## Adapting it to your own definition of "rare"

The method has no built-in taxonomy — it learns whatever the training labels
call long-tail. To retarget it, relabel and retrain rather than editing the
model: write your own rubric in place of
[`latest_grading_criteria.md`](../latest_grading_criteria.md), produce labels
with `scripts/annotate_normal_core.py` or by hand, and train a new SAE. Only the
labels change; the pipeline is identical. [Common Tasks](tasks.md) has the
details, including how to swap in a different VLM backbone.
