# Presentation and interview copy

Use the [interactive demo guide](../demo/README.md) to inspect six-camera inputs,
saved model scores, activations and review exports alongside these notes.

These CV bullets and interview scripts describe Yike's work on the team
capstone and the later repository modernization. The original team was Yike
Chen, Zihua Chen and Yusong Zhao. Yike leads the later CPU workflow, evaluation
and review-tool integration. The CPU ResNet18 experiment uses a separate
feature protocol from the original Cosmos/VLM route.

## CV bullets

- Led the modernization of a team capstone into a reproducible CPU pipeline for six-camera driving-scene screening, integrating input validation, checkpoint reloads, CI, and a browser review tool with CSV export.
- Implemented feature-extraction and SAE-training code for the original capstone; led a matched four-model CPU evaluation with scene-disjoint splits and validation-only thresholds, documenting reward saturation and preserving per-sample predictions and review provenance.

Use both under **Long-Tail Driving Scene Discovery — Project Lead**. Keep any
numerical result tied to its small, single-reviewer reference protocol.

## 60-second introduction

I lead the modernization of a team capstone that screens six-camera driving
frames for human review. My original contributions included feature extraction,
sparse-autoencoder training, and precision-oriented evaluation.

I led the integration of validated images, a frozen CPU encoder, four matched
models, saved predictions, and a browser review tool.
We kept scenes separate and selected thresholds only on validation data.

One useful failure was that the SAE's fixed rule flagged every frame in a new
nine-frame batch. An earlier validation threshold reduced that to five
candidates, but missed one positive reference. That exposed the
precision–recall tradeoff.

The references came from one reviewer who had already seen model-generated
labels. This remains a small pilot; its main result is a reproducible engineering
workflow. Stronger model claims need broader data and independent evaluation.

## Three-minute explanation

I lead the modernization and delivery of a GitHub project that grew out of our
team capstone on long-tail driving-scene discovery. My original responsibilities
included feature extraction, sparse-autoencoder training, precision-oriented
evaluation, and parts of the ablations. The later repository work added CPU
execution, reproducibility checks, and review tools, with the original team
contributions retained.

The task is to rank six-camera timestamps for human review. We define a normal
core through visible driving conditions, rather than assuming that a rare image
must be dangerous. A single timestamp cannot establish a sudden cut-in or hard
braking, so the labels focus on factors visible across its six views.

For the CPU follow-up, we use a pinned ResNet18 encoder and concatenate the six
view representations. Across two batches, the repository records 89 timestamps,
534 images, and 12 scenes. Those are cumulative inputs, not 89 training rows:
the four fitted models use the earlier cohort, and the later nine frames were
used for inference and review. The comparison shares frozen features and scene
assignments across Logistic regression, a small MLP, and two SAE variants.
Normalization uses training data, while model and threshold selection use
validation data. Saved predictions, input hashes, checkpoint reloads, CI, and a
review browser make the decisions inspectable.

Two failures shaped the result. First, the reward and no-reward SAE checkpoints
were identical. We traced this to a reward cap that was already saturated, so
the run supplied no evidence of a reward benefit. Second, the SAE's fixed
activation rule selected all nine new frames. Replaying a threshold already
selected on the old validation set selected five: three positive references,
two false positives, and one missed positive. We preserved that tradeoff rather
than tuning on the new labels. Fewer selected items did not establish measured
human time savings.

The current models trained with only three normal frames from one scene and
one seed. The references came from one reviewer who had already seen
model-generated labels; the new batch also confounds class with scene. My next
research step would be broader normal training data and independently reviewed
held-out scenes. If the SAE then fails to beat a cheaper baseline, I would choose the
cheaper model. The completed outcome today is a traceable screening and review
workflow with explicit limits.

## Delivery notes

Lead with the task and your responsibilities. Credit the original team and
distinguish the capstone work from the later CPU workflow. Yike's original
contributions were feature extraction, SAE training, precision-oriented
evaluation and parts of the ablations.

If asked about scale, distinguish the 89 input timestamps from the training
set. The later nine timestamps were used for inference and review; the fitted
models still have only three normal training frames from one scene. If asked
about AP, name the 15-frame, two-scene test set and the single reviewer's prior
access to model-generated labels. These were not independently blinded annotations.

Use the new-batch failure to explain the threshold decision: the fixed rule
selected all nine, while the old validation threshold selected five with two
false positives and one missed positive. No review-time savings were measured.
Likewise, identical SAE weights and a saturated reward cap are a diagnosed
negative result, not evidence that the reward improved the algorithm.

## Evidence for follow-up questions

| Interview question | Evidence to open |
|---|---|
| What did you lead and implement? | [Project role and contribution record](project-card.md) |
| How did you prevent scene leakage and select the model? | [Fitted comparison and frozen split](../results/human-ai-exposed-20261026/README.md) |
| What failed, and how did you inspect it? | [Fixed-rule FP/FN cases](../results/human-ai-exposed-20261026/failure_cases.json) |
| Was the later threshold chosen using the new labels? | [Frozen-policy source and replay](../results/new-scene-human-review-20261028/README.md#replay-of-policies-frozen-before-these-new-scenes) |
| Can someone inspect the workflow? | [Demo access and verification limits](../demo/README.md) |

For an interview, open an example in the offline explorer and trace its saved
score, active units and review export. The [demo guide](../demo/README.md)
explains the downloadable HTML and generated-data CPU example.
