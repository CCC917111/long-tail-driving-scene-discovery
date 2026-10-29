# Presentation and interview copy

[Watch or download the approximately 90-second captioned walkthrough](https://yike-driving-review-20261028.upbeatperch12.chatgpt.site/presentation/) (owner-private; use the owner’s ChatGPT sign-in). The recording is silent, uses scripted browser navigation and shows original images with cached trained outputs. It performs no new inference or annotation.

Final wording for Yike's SWE/MLE portfolio, based on the recorded project
artifacts. The original work was a team capstone; Yike led the repository
modernization and delivery with AI-assisted implementation and documentation.
The CPU ResNet18 follow-up is a separate feature protocol from the original
Cosmos/VLM route.

## CV bullets

- Led the AI-assisted modernization of a team capstone into a reproducible CPU pipeline for six-camera driving-scene screening, integrating input validation, checkpoint reloads, CI, and a browser-based human-review and CSV-export workflow.
- Implemented feature-extraction and SAE-training code for the original capstone; led a matched four-model CPU evaluation with scene-disjoint splits and validation-only thresholds, documenting reward saturation and preserving per-sample predictions and review provenance.

Use these bullets together under **Long-Tail Driving Scene Discovery — Project
Lead**. They emphasize verified implementation, evaluation and delivery without
presenting the small pilot's metrics as independent benchmark accuracy.

## 60-second introduction

I lead the modernization of a team capstone that screens six-camera driving
frames for human review. My original contributions included feature extraction,
sparse-autoencoder training, and precision-oriented evaluation.

With AI-assisted development, I connected validated images, a frozen CPU
encoder, four matched models, saved predictions, and a browser review tool.
We kept scenes separate and selected thresholds only on validation data.

One useful failure was that the SAE's fixed rule flagged every frame in a new
nine-frame batch. An earlier validation threshold reduced that to five
candidates, but missed one positive reference. That exposed the
precision–recall tradeoff.

This remains a small, single-reviewer pilot with AI-label exposure recorded.
Its main result is a reproducible engineering workflow; stronger model claims
need broader data and independent evaluation.

## Three-minute explanation

I lead the modernization and delivery of a GitHub project that grew out of our
team capstone on long-tail driving-scene discovery. My original responsibilities
included feature extraction, sparse-autoencoder training, precision-oriented
evaluation, and parts of the ablations. The later repository work used
AI-assisted development, with the original team contributions retained.

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
one seed. The references came from one reviewer with AI-label disclosure
recorded; the new batch also confounds class with scene. My next research step
would be broader normal training data and independently reviewed held-out
scenes. If the SAE then fails to beat a cheaper baseline, I would choose the
cheaper model. The completed outcome today is a traceable screening and review
workflow with explicit limits.

## 中文使用说明

- **开场先讲任务和职责。** Yike 是项目 lead，负责后续仓库优化和交付；原始 capstone 是团队合作，后续实现和文档使用过 AI 辅助。不要把团队模块都说成本人独立完成。
- **区分输入总量和训练量。** 89 组、534 张、12 个场景是累计输入；新增九组没有用于重训。现有模型训练集中的正常样本仍只有三组，来自一个场景。
- **把失败讲成一次技术决策。** 固定规则全选九组；沿用旧 validation 阈值后选五组，仍有两个误报和一个漏报。不要把减少四个候选说成已经测得省了多少人工时间。
- **指标需要带协议。** 如被追问 AP，说明旧测试集只有 15 组、两个场景，标签来自 AI 信息披露后的单人复核。新九组的五个正常样本全部来自一个场景，四个非正常样本来自另一个场景。
- **对负结果坦诚。** 两种 SAE 权重相同，reward 饱和导致没有观察到增益。可以解释诊断和下一步，不能把它说成已验证的新算法优势。

## Evidence for follow-up questions

| Interview question | Evidence to open |
|---|---|
| What did you lead and implement? | [Project role and contribution record](project-card.md) |
| How did you prevent scene leakage and select the model? | [Fitted comparison and frozen split](../results/human-ai-exposed-20261026/README.md) |
| What failed, and how did you inspect it? | [Fixed-rule FP/FN cases](../results/human-ai-exposed-20261026/failure_cases.json) |
| Was the later threshold chosen using the new labels? | [Frozen-policy source and replay](../results/new-scene-human-review-20261028/README.md#replay-of-policies-frozen-before-these-new-scenes) |
| Can someone inspect the workflow? | [Demo access and verification limits](../demo/README.md) |

The hosted review Site is owner-private and requires the owner's ChatGPT
sign-in. External reviewers can use the documented public HTML fallback or
the generated-data CPU quickstart; they are not automatically granted access
to the private Site.
