# Project card: Long-Tail Driving Scene Discovery

**Task:** rank multi-camera driving frames for human review and inspect the
sparse features behind the ranking.

**Context:** a team capstone with Yike Chen, Zihua Chen and Yusong Zhao,
followed by a separately documented AI-assisted CPU engineering pilot.

**Research question:** under identical frozen features, scenes and review
budget, does a tail-guided sparse autoencoder offer useful ranking or
inspection over Logistic regression and a small MLP?

## Contribution and implementation

The original contribution statement records Yike's work on SAE feature
extraction and training, precision-oriented evaluation, and parts of the
ablation/interpretation analysis. Teammates own the annotation platform and
most feature interpretation; see the [contribution table](../README.md#team-and-contribution).
Original reports and commit/run evidence remain necessary to substantiate
individual implementation claims.

The follow-up CPU route adds strict token/scene/label contracts, pinned
six-camera ResNet18 features, four methods under one manifest, validation-only
selection, checkpoint/source provenance, saved-result reconstruction and an
offline explorer. This follow-up was developed with AI assistance and is
documented separately from Yike's original capstone contributions. Yike should
review and understand it before presenting it as current engineering work.

## What is demonstrated

| Evidence | What it establishes |
|---|---|
| 80 real timestamps, 480 original images, ten scenes | Real inputs passed transport, decoding and frozen-split preparation. |
| Four trained methods and independent checkpoint reloads | The CPU training/evaluation path completed with traceable outputs. |
| 73 provisional AI labels, seven uncertain exclusions | A scoped pipeline pilot; independent human truth is absent. |
| Reward saturation diagnosis | Identical SAE checkpoints were explained by a zero-gradient capped reward; no reward gain was established. |
| Offline explorer and CI | Cached outputs can be inspected/exported; synthetic browser checks cover the UI. |

The pilot uses a custom 6/2/2-scene mini allocation and one seed. Test has
14 frames from two scenes. SAE AP is 0.9216 **against AI references**, while all
methods fail the validation 90% precision target. These values do not establish
human-validated accuracy or general SAE superiority.

## A 60-second explanation

“Our team capstone studies how to find driving frames that deserve human
review. My recorded role covered SAE feature extraction and training,
precision-oriented evaluation and parts of the ablation analysis. The original
method uses frozen VLM features and a sparse representation with normal and
tail subspaces.

“In a later AI-assisted CPU follow-up, we made a small real-image pipeline
traceable from scene allocation to saved predictions and an offline explorer.
It compares Logistic regression, MLP and matched SAEs with and without reward.
The labels are provisional AI judgments, so I treat the metrics as a pipeline
check. An instructive result is that the reward was saturated and had zero
gradient: both SAE checkpoints were identical. Independent human review and
more balanced scene coverage are needed before a stronger claim.”

Use this script after verifying the recorded contribution and being able to
explain the code, inputs and limitations in your own words.

## Suggested current CV wording

> Implemented feature-extraction and sparse-autoencoder training code for a
> team capstone on long-tail driving-scene discovery; led precision-oriented
> evaluation and contributed to ablation studies.

This follows the existing contribution statement. Do not add an AI-pilot
accuracy number or claim sole ownership of the follow-up code. A future
result bullet should state the feature protocol, scene split, label source
and independently verified outcome.

[Demo and walkthrough](../demo/README.md) · [Protocol](ai-pilot.md) ·
[Derived evidence](../results/cpu-mini-ai-pilot-20261023/README.md) ·
[Historical claims](../results/README.md)
