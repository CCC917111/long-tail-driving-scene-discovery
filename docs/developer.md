# Developer guide

This repository contains the capstone work of Yike Chen, Zihua Chen and Yusong
Zhao at the Chinese University of Hong Kong, Shenzhen, plus later AI-assisted
engineering and evaluation work. The [contribution statement](../README.md#team-and-contribution)
records the team's historical roles; [feedback status](feedback-status.md)
records the later changes and evidence.

The original method uses frozen vision-language-model features and a
tail-guided sparse autoencoder (SAE) to find candidate long-tail driving frames.
The completed current experiment uses six-view pretrained ResNet18 image
features on CPU. It tests the data, training, screening and review workflow;
it does not reproduce the Cosmos experiment.

The current intake contains 89 timestamps from 12 scenes and 534 camera images.
The models were fitted on the older 80-timestamp cohort, using 77 binary human
references and excluding three uncertain decisions. Those decisions came from
one reviewer after AI-reference disclosure. A later nine-timestamp batch was
screened with the saved model and old validation threshold, without fitting or
tuning on its labels. See [the fitted run](../results/human-ai-exposed-20261026/README.md)
and [new-scene review](../results/new-scene-human-review-20261028/README.md)
for the limits of both comparisons.

## Layout

```text
scripts/
  annotate_normal_core.py         VLM labels from the normal-core rubric
  extract.py                     per-keyframe, per-layer VLM feature extraction
  extract_cpu.py                  six-view CPU image features
  sae_common.py                  shared SAE model, training and evaluation
  tail_activations.py             decision rule and numpy activation store
  sae_abstopk_tail_reward.py      AbsTopK entry point, with configurable reward
  ablation_topk_sae.py            historical signed-TopK variant
  ablation_sae_cosmos_baseline.py historical pre-selection tail-bonus variant
  neuron_explorer.py             unit/frame explorer and command-line queries
  screen.py                      saved-SAE inference on new features
docs/                            setup, protocols and evidence guides
tests/                           contract, workflow and browser checks
results/                         derived result bundles and historical records
latest_grading_criteria.md        normal-core annotation rubric
```

Check the actual script names and flags before adapting a command. The
[CPU route](cpu-route.md) is the starting point for the completed image-feature
workflow; [from zero](from-zero.md) also documents the unexecuted VLM/GPU route.

## Data flow

Both encoders produce a feature matrix and row-aligned sample metadata. The
VLM route reads six camera views with a prompt and records the selected layer
and pooling. The CPU route concatenates six ResNet18 view embeddings into
3,072-dimensional features. These feature protocols are separate and cannot
share a trained checkpoint.

```text
six camera views + sample/scene metadata
  -> feature extractor -> features (N, D), meta.json, extraction configuration
  -> reviewed labels -> strict label import, exclusions, frozen scene manifest
  -> training-only mean/std -> matched baseline and SAE fits
  -> saved checkpoint, standardizer, predictions, evaluation and activations
  -> explorer: unit -> frames; frame -> active units
  -> screening: compatible new features -> scores and a blank review queue
```

The loader rejects duplicate tokens, unsupported labels and missing scene
identifiers. Controlled comparisons reuse one frozen scene manifest across
methods and seeds. The SAE splits sparse activations into normal and tail
subspaces; the fixed activation-count rule and validation-selected operating
point are separate outputs. See [evaluation](evaluation.md) for their contracts.

Feature extraction writes resumable checkpoints. Use a fresh cache when
changing the encoder, revision, preprocessing or feature settings. Measure
runtime and memory before choosing a larger budget; this project has not
established a VLM speed or cost advantage.

## Where to change behavior

| Change | Entry point |
|---|---|
| SAE architecture or objective | `scripts/sae_common.py`, `LongTailGuidedSAE` |
| Split and metrics contracts | `scripts/data_manifest.py`, `scripts/evaluate_runs.py`; version the protocol |
| SAE variant | Entry point with its own `DEFAULTS`; use matched settings for comparisons |
| Definition of long-tail | `latest_grading_criteria.md` and annotation prompts |
| VLM or selected layer | `scripts/extract.py`; re-extract and retrain |
| Activation decision rule | `scripts/tail_activations.py`, `predict_long_tail` |
| New-data screening | `scripts/screen.py`, [screening guide](apply.md) |
| Unit inspection and naming | `scripts/neuron_explorer.py`, [explorer guide](neurons.md) |
| Design rationale | [Design](design.md) |

Unit IDs are specific to a fitted checkpoint. A high activation identifies an
inspectable association; it does not by itself prove a semantic explanation.
Keep newly fitted units unnamed until a documented review supports a name.

## Before submitting a change

Follow [coding conventions](coding.md). Run the relevant checks, then the
repository checks required for the change:

```bash
make test
make lint
```

Add a focused test when a change can alter data meaning, the evaluation
protocol or model behavior. The [testing record](testing.md) distinguishes
synthetic contracts, real-asset audits, HTTP checks and browser interaction.
Historical default variants are not controlled ablations merely because they
share an implementation.
