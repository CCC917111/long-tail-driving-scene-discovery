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

## What Yike led

- Repository optimization and integration of the reproducible CPU route.
- A controlled four-method evaluation with frozen scenes, train-only scaling,
  validation-only selection and traceable saved predictions.
- Review of all 80 timestamps, retaining 77 binary judgments and three uncertain
  decisions, with the final review's prior AI-label exposure explicitly recorded.
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

The SAE operating threshold selects two test timestamps, both positive under
the recorded judgments, and recalls two of ten positives. Its fixed activation
rule flags all test timestamps. Keep these policies separate. One seed, three
normal training frames from one scene, two test scenes and prior AI exposure
limit research claims. The [new-scene expansion plan](research-expansion.md)
separates proposed work from completed experiments.

## A 60-second explanation

“I lead the optimization and delivery of this GitHub project, which grew out
of our team capstone on finding driving frames worth human review. My recorded
capstone work covered SAE feature extraction and training, precision-oriented
evaluation and parts of the ablations.

“For the repository modernization, we used AI-assisted development to connect
validated six-camera inputs, CPU features, four matched screening methods and
an offline explorer. I completed the image review, with prior exposure to the
AI labels recorded. On this small reference set, the SAE ranks candidates well,
but its reward and no-reward checkpoints are identical. That negative result
led us to diagnose a saturated reward rather than claim a reward gain. The
next research step is independent labels and more normal examples and held-out
scenes; the current release demonstrates a traceable engineering workflow.”

## A three-minute explanation

1. **Problem and role:** candidate discovery, Yike's lead role, original team
   contributions and the AI-assisted modernization phase.
2. **Data contract:** one timestamp contains six views; scenes are frozen before
   labels; unknown/uncertain labels remain separate; scaling uses train only.
3. **Controlled comparison:** same features and manifest for Logistic, MLP and
   matched SAEs; explain AP, fixed-budget precision/recall and validation selection.
4. **One failure:** inspect an MLP fixed-rule FP or FN and explain the visible
   reference disagreement without inferring motion from one frame.
5. **One negative finding:** reward saturation and identical SAE weights.
6. **Limits and decision:** explain the small scene coverage and label exposure,
   then describe the data-dependent expansion plan without claiming it ran.

## CV wording

> Led repository-wide optimization and delivery of a team capstone on long-tail
> driving-scene discovery, integrating a reproducible CPU pipeline, matched
> four-model evaluation and an offline activation explorer with AI-assisted
> development.

> Implemented feature-extraction and sparse-autoencoder training code for the
> original capstone; led precision-oriented evaluation and contributed to
> ablation studies.

Use the role wording that matches the work you can explain. The small pilot's
AP is scoped to post-AI-exposure, single-reviewer references; the current CV
wording emphasizes the verified engineering and leadership rather than an
unqualified accuracy number.

[Demo and 90-second walkthrough](../demo/README.md) ·
[Latest reviewed-reference results](../results/human-ai-exposed-20261026/README.md) ·
[Original AI pilot](../results/cpu-mini-ai-pilot-20261023/README.md) ·
[Historical claims](../results/README.md)
