# Long-Tail Driving Scene Discovery

**Project lead: Yike Chen.** Yike leads the repository modernization and
technical delivery: the CPU workflow, experiment organization, review tools,
reproducibility materials and demonstration.

The project began as a team capstone by **Yike Chen, Zihua Chen and Yusong
Zhao**. Yike's original contributions included feature extraction, SAE training,
precision-oriented evaluation and parts of the ablation and interpretation
analysis. Zihua's annotation-platform work and Yusong's learned-feature analysis
remain credited in the [original contribution table](../README.md#team-and-contribution).

[**Interactive demo guide**](../demo/README.md) ·
[**CV bullets and interview scripts**](presentation.md)

## Task and approach

Rank six-camera driving timestamps for human review and inspect the sparse
features behind each score. The experiment asks whether an SAE offers useful
ranking or inspection compared with logistic regression and a small MLP under
the same feature, scene and review-budget settings.

The completed CPU route uses a pinned ResNet18 encoder, train-only scaling,
scene-disjoint splits and validation-only model and threshold selection.
Saved predictions and checkpoint reloads connect the result tables to their
trained models. A browser explorer displays six camera views, sparse
activations and error cases, then exports a review queue.

Yike led the integration of this route, the four-method evaluation, the review
of 80 original timestamps and nine later inputs, and the GitHub documentation
and demonstration. The CPU experiment is a separate feature protocol from
the original Cosmos/VLM capstone route.

## Completed evidence

The latest fitted comparison uses 77 binary judgments from one reviewer who
had seen the AI-label table; three uncertain decisions are excluded. Its test
set contains 15 timestamps from two scenes. The models use one seed and have
only three normal training frames from one scene.

| Evidence | What it shows |
|---|---|
| 80 timestamps, 480 images and ten scenes in the fitted cohort | Validated six-camera inputs and completed CPU feature extraction. |
| Four trained methods and a separate reload audit | Saved scores, scalers, thresholds and result tables can be reconstructed. |
| 77 final binary judgments and three uncertain decisions | A completed single-reviewer reference set after AI-label exposure. |
| SAE AP 0.9353 and four positives in the top four | Ranking performance on the small test set under those references. |
| Two identical SAE checkpoints | No observed reward benefit; the reward cap was already saturated. |
| Offline explorer and blank review export | Scores, activations and reference disagreements can be inspected and reviewed. |

The SAE's validation-selected operating point selects two test timestamps,
both positive under the recorded references, and recalls two of ten positives.
Its top-four ranking is a separate policy; its fixed activation rule flags all
test frames. The higher AP than the original AI-reference run reflects changed
references and retained IDs, not improved SAE weights.

## New-scene follow-up

Nine later timestamps add 54 images from two scenes. Across both batches, the
repository records **89 timestamps, 12 scenes and 534 images**. The later nine
were used for inference and review, not model fitting.

The reviewer submitted five normal and four non-normal judgments after the
AI reference table was disclosed in chat; per-row timing is unknown. Replaying
the old validation-selected threshold of **6.074954873** selects five candidates:
three positive references, two false positives and one missed positive
(precision 60%, recall 75%). The fixed SAE rule selects all nine. No threshold
was tuned on these new labels, and no review-time savings were measured.

All five normal references come from one new scene and all four non-normal
references from the other. That class/scene confound and the prior AI-label
exposure limit what the batch can establish about generalization. The
[review record and frozen-policy replay](../results/new-scene-human-review-20261028/README.md)
preserve the submission and policy source. The
[research expansion plan](research-expansion.md) describes broader training
and independent held-out review as future work.

## Demonstration and presentation

The [offline explorer](../demo/driving_pilot.html) loads the four original image
ZIPs locally and displays six camera views, saved scores, sparse activations
and MLP false-positive and false-negative bookmarks. Export a blank review
queue to inspect the handoff to a reviewer. The [demo guide](../demo/README.md)
explains the required inputs and the generated-data CPU example.

The [presentation copy](presentation.md) includes two CV bullets and short and
extended interview introductions. Use the result bundles below to answer
questions about model selection, label provenance and the reward diagnostic.

[Latest fitted results](../results/human-ai-exposed-20261026/README.md) ·
[Supplemental review](../results/new-scene-human-review-20261028/README.md) ·
[Original AI pilot](../results/cpu-mini-ai-pilot-20261023/README.md) ·
[Historical result record](../results/README.md)
