# Project card: Long-Tail Driving Scene Discovery

**Project lead: Yike Chen.** Yike is responsible for the GitHub project's
repository-wide optimization and delivery: the CPU workflow, experiment
organization, review process, reproducibility materials and demonstration.
Implementation and documentation in the modernization phase used AI assistance.

**Context:** the repository builds on a team capstone by Yike Chen, Zihua Chen
and Yusong Zhao. The [original contribution table](../README.md#team-and-contribution)
retains their respective capstone work. Project leadership and modernization
responsibility are separate from claiming sole authorship of teammates' modules.

**Task:** rank six-camera driving timestamps for human review and inspect the
sparse features behind the ranking.

**Question:** with identical features, scenes and review budget, does an SAE
provide useful ranking or inspection over Logistic regression and a small MLP?

**Review delivery:** the [hosted review tool](https://yike-driving-review-20261028.upbeatperch12.chatgpt.site)
is owner-private and requires the owner's ChatGPT sign-in. Its 54 new images
load from the same origin; external GitHub readers are not automatically granted
access. The [source-linked HTML fallback](../demo/driving_new_scenes.html) needs
Internet access to `raw.githubusercontent.com`.

## What Yike led

- Repository optimization and integration of the reproducible CPU route.
- A controlled four-method evaluation with frozen scenes, train-only scaling,
  validation-only selection and traceable saved predictions.
- Review of all 80 timestamps, retaining 77 binary judgments and three uncertain
  decisions, with the final review's prior AI-label exposure explicitly recorded.
- A later nine-frame review from two additional scenes, preserving the original
  submission, chat AI-reference disclosure and descriptive frozen-policy checks.
- Delivery of the GitHub documentation, offline explorer and presentation flow.

Yike's original recorded capstone contributions include feature-extraction and
SAE training code, precision-oriented evaluation, and parts of the ablation and
interpretation analysis. Teammates' annotation-platform and interpretation work
remain credited. The modernization used AI coding assistance; interview answers
should distinguish leadership, personal implementation and tool-assisted work.

## Evidence to show

| Evidence | What it establishes |
|---|---|
| 80 real timestamps, 480 original images, ten scenes | Actual input validation and six-camera CPU extraction. |
| Four trained methods and independent reload audit | Scores, train-only scalers, thresholds and tables can be reconstructed. |
| 77 final binary judgments, three uncertain exclusions | A completed single-reviewer reference set after AI-label disclosure. |
| SAE AP 0.9353 and four-of-four top-budget hits | Ranking on 15 test timestamps in two scenes under those references. |
| Two identical SAE checkpoints | No incremental reward benefit was observed; the cap saturation was diagnosed. |
| Latest offline explorer | Cached reviewed-reference scores, activations, errors and blank review export are inspectable. |
| Nine additional real timestamps, 54 images, two scenes | New-input CPU inference and same-origin review delivery; no new model fit. |
| Five normal / four non-normal supplemental judgments | Complete single-reviewer submission after chat AI-reference disclosure; per-row timing remains unknown. |
| Old validation-selected policy on those nine rows | Five candidates, precision 60%, recall 75%, without new-label threshold tuning. |

The SAE operating threshold selects two test timestamps, both positive under
the recorded judgments, and recalls two of ten positives. Its fixed activation
rule flags all test timestamps. Keep these policies separate. One seed, three
normal training frames from one scene, two test scenes and prior AI exposure
limit research claims. The [new-scene expansion plan](research-expansion.md)
separates proposed work from completed experiments.

The [supplemental replay](../results/new-scene-human-review-20261028/frozen_policy_evaluation.json)
uses the earlier validation-selected threshold. It is separate from both the
fixed activation rule, which flags all nine, and the old 15-frame test table.
All five new normal references belong to one scene and all four non-normal
references to another. That class/scene confound and prior AI disclosure
prevent treating this batch as an independent generalization benchmark.

## Final presentation materials

The [final English CV bullets, 60-second introduction and three-minute story](presentation.md)
are ready to use. The scripts contain 122 and 368 words respectively; actual
delivery time depends on speaking pace.

The [approximately 90-second recorded walkthrough](https://yike-driving-review-20261028.upbeatperch12.chatgpt.site/presentation/)
uses original images and saved trained outputs. It is a silent recording with
English captions and scripted browser navigation, not Yike's recorded voice or
a live inference service. The Site remains owner-private with ChatGPT sign-in.

The video shows a selected SAE reference-positive frame, one MLP false positive
and one MLP false negative under the same fixed threshold, then exports a blank
new-review queue. [Recording evidence](../results/presentation-20261029/README.md)
links each example to its saved prediction and records the private video hash.

Use the role wording that matches the work you can explain. The small pilot's
AP is scoped to post-AI-exposure, single-reviewer references; the current CV
wording emphasizes the verified engineering and leadership rather than an
unqualified accuracy number.

[Demo and 90-second walkthrough](../demo/README.md) ·
[Latest reviewed-reference results](../results/human-ai-exposed-20261026/README.md) ·
[Supplemental review and frozen-policy replay](../results/new-scene-human-review-20261028/README.md) ·
[Original AI pilot](../results/cpu-mini-ai-pilot-20261023/README.md) ·
[Historical claims](../results/README.md)
