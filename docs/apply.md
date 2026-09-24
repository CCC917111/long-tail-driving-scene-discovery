Screening Your Own Driving Data
=============
Welcome to the long-tail mining project application page!

The benchmark answers "how good is this method". This page answers the question
a team with a drive log actually has: **here are a hundred thousand frames
nobody has looked at — which ones should a human annotate first, and why?**

## What the tool is for

You have driving data and a limited annotation budget. Labelling everything is
impossible, labelling at random spends the budget on the same lane-following
frames over and over, and the frames that matter — the ones that would require
the ego vehicle to slow down, yield or keep a larger margin — are rare by
definition.

`scripts/screen.py` takes your frames and returns them ranked by how strongly
they activate the learned long-tail subspace, each with the activated neurons
and, where those neurons have names, a plain-language reason. You annotate from
the top of that list.

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

| What | Layout |
|---|---|
| Images | Six synchronised camera views per sample, in the nuScenes layout: `samples/CAM_FRONT/<file>.jpg`, `samples/CAM_FRONT_LEFT/...`, `CAM_FRONT_RIGHT`, `CAM_BACK`, `CAM_BACK_LEFT`, `CAM_BACK_RIGHT` |
| Index | A metadata source that maps a `sample_token` to its six image paths — the nuScenes `v1.0-*_meta.tgz` for nuScenes data |
| Layer | The same `--layers` value as the training run, e.g. `28` |
| Pooling | The same pooling as the training run: mean-pooled (`*_mean.npy`) or last-token (`*_last_token.npy`) |

Fewer than six views also works — the prompt in
[`prompt_templates.md`](prompt_templates.md) describes the views it is given —
but a run trained on six-view features should be applied to six-view features.
Mixing them changes the feature distribution the SAE was fitted on.

Extraction writes two files that travel together:

```
layer28_mlp_output_mean.npy   (N, D) float32, one row per frame
meta.json                     N entries, row-aligned, with sample_token and scene
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

The decision threshold is not re-tuned on your data — it is read from
`metrics.json`, the value the training run chose on its own validation split
and applied unchanged to its held-out test split. `--threshold` overrides it if
you would rather trade precision for recall; raising it shortens the shortlist,
lowering it lengthens it.

## What you get back

`screening.csv`, one row per frame, sorted by score:

| Column | Meaning |
|---|---|
| `rank` | Position in the ranking, 1 = most long-tail-like |
| `sample_token` | Frame identifier, carried over from `meta.json` |
| `scene` | Scene name or token, so you can group frames by drive |
| `tail_score` | `\|\|z_t\|\|_2`, the activation strength of the long-tail subspace |
| `prediction` | `long_tail` or `normal`, at the threshold in use |
| `active_neurons` | How many long-tail units the frame activates |
| `top_neurons` | The strongest activated unit indices |
| `reasons` | Their names, for the units that have one |

and `long_tail_samples.json`, just the flagged tokens, ready to be handed to an
annotation tool.

The console prints the flagged share and a tally of the reasons:

```
screened 12480 frames | flagged 1163 (9.3%) as long-tail

why they were flagged:
    612  Rain / wet road
     97  Glare on a wet or rainy road
     41  Pedestrian in a construction zone
     14  Wheelchair user ahead
```

That tally is often the first genuinely useful output: it is a profile of what
kinds of rare situations your archive contains, computed without anyone having
labelled it.

## Working with the result

- **Spend the annotation budget top-down.** Take the first N rows of
  `screening.csv`. With the reported test-split precision of 0.90, most of what
  a human opens is worth opening.
- **Fill a specific gap.** If your model fails on wet-road glare, filter on the
  unit for it rather than on the score — see
  [The Interpretable Neurons](neurons.md). The units act as retrieval keys for
  categories nobody labelled in advance.
- **Compare drives.** Group by `scene` and compare flagged shares to find which
  collection routes actually contribute rare data and which ones repeat what
  you already have.
- **Look at `active_neurons`, not only `tail_score`.** A frame that activates
  several unrelated units is often a genuinely compound scene; a frame with one
  very strong unit is usually a clean example of that one category.

## Adapting it to your own definition of "rare"

The method has no built-in taxonomy — it learns whatever the training labels
call long-tail. To retarget it, relabel and retrain rather than editing the
model: write your own rubric in place of
[`latest_grading_criteria.md`](../latest_grading_criteria.md), produce labels
with `scripts/annotate_normal_core.py` or by hand, and train a new SAE. Only the
labels change; the pipeline is identical. [Common Tasks](tasks.md) has the
details, including how to swap in a different VLM backbone.
