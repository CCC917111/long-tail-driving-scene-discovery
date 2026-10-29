# Long-Tail Driving Scene Discovery

**Project lead: Yike Chen.** Yike leads the repository modernization and
technical delivery: the CPU workflow, experiment organization, review tools,
reproducibility materials and demonstration.

The project began as a team capstone by **Yike Chen, Zihua Chen and Yusong
Zhao**. Yike's original contributions included feature extraction, SAE training,
precision-oriented evaluation and parts of the ablation and interpretation
analysis. Zihua's annotation-platform work and Yusong's learned-feature analysis
remain credited in the [original contribution table](../README.md#team-and-contribution).
The later modernization used AI-assisted implementation and documentation.

[**Watch the 90-second walkthrough**](https://github.com/CCC917111/long-tail-driving-scene-discovery/blob/main/demo/driving-walkthrough-90s.mp4) ·
[**Download the MP4**](https://github.com/CCC917111/long-tail-driving-scene-discovery/raw/refs/heads/main/demo/driving-walkthrough-90s.mp4) ·
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

The public GitHub video linked above uses original images and cached trained
outputs, with English captions and no spoken narration. Scripted navigation
shows an SAE reference-positive case, an MLP false positive and false negative
under the same fixed threshold, followed by a blank new-review CSV export.
The [recording record](../results/presentation-english-20261029/README.md) identifies
the cases, source images and attribution.

The [English presentation copy](presentation.md) includes two CV bullets,
a 122-word introduction and a 368-word explanation. Delivery time depends on
speaking pace.

The optional [hosted review tool](https://yike-driving-review-20261028.upbeatperch12.chatgpt.site)
and its [presentation page](https://yike-driving-review-20261028.upbeatperch12.chatgpt.site/presentation/)
require the owner's ChatGPT sign-in. The [demo guide](../demo/README.md)
explains the public video, downloadable HTML and generated-data CPU example.

[Latest fitted results](../results/human-ai-exposed-20261026/README.md) ·
[Supplemental review](../results/new-scene-human-review-20261028/README.md) ·
[Original AI pilot](../results/cpu-mini-ai-pilot-20261023/README.md) ·
[Historical result record](../results/README.md)
