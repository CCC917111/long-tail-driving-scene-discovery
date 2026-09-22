# Long-Tail Driving Scene Discovery

Finding rare ("long-tail") driving scenarios in large autonomous-driving datasets by looking inside the representations of a vision-language model (VLM), rather than relying on hand-written rules or frequency counts.

The core idea: instead of trying to enumerate every possible rare or risky driving situation in advance, define a tight, high-confidence notion of a **"normal" driving scene**, learn what that looks like inside a VLM's internal representations, and then treat anything that deviates from that learned "normal manifold" as a candidate long-tail sample worth a closer look. The long-tail definition is behavior-oriented rather than purely frequency-based: a sample counts as long-tail if it is both statistically rare and likely to require extra defensive driving (slowing down, yielding, keeping a larger safety margin, paying closer attention to nearby risks).

Built on [NVIDIA Cosmos-Reason1-7B](https://huggingface.co/nvidia/Cosmos-Reason1-7B) (a Qwen2.5-VL-based multimodal reasoning model) and the [nuScenes](https://www.nuscenes.org/nuscenes) autonomous-driving dataset, with additional evaluation on a private Huawei "golden" driving dataset (not included here).

## Why this problem

Autonomous-driving perception stacks are usually evaluated on average-case performance, but the failures that matter happen in the tail of the distribution — unusual lighting, occlusions, atypical road layouts, rare object configurations. Long-tail data is hard to find because it's rare by definition: keyword search and simple heuristics don't scale, and manually reviewing millions of frames isn't feasible. Purely unsupervised outlier detection also isn't a great fit — many safety-relevant long-tail scenes (e.g. a wheelchair user near the ego path, see [Case study](#case-study-a-sample-anomaly-detection-and-confidence-filtering-both-miss) below) are not visual outliers in feature space at all; what makes them long-tail is a local, safety-relevant detail, not global scene novelty.

This project explores a representation-based alternative: use a large VLM as a feature extractor, define what "normal"/"long-tail" means as precisely as possible in behavioral terms, and train a sparse autoencoder (SAE) to carve the VLM's dense hidden state into an interpretable, long-tail-sensitive subspace whose activation can be used directly as a long-tail detector.

## Method

1. **Define "normal"/"long-tail" precisely.** The project defines a strict normal core: a single-frame, multi-camera driving scene where the ego vehicle would not need to take any extra defensive action (see [`latest_grading_criteria.md`](latest_grading_criteria.md) for the full annotation rubric, with inclusion/exclusion criteria and a normal_core / not_normal_core / uncertain protocol). A sample is treated as a long-tail candidate if it is both relatively rare in natural driving data and likely to require additional defensive-driving behavior.

2. **Annotate at scale with a VLM.** [`scripts/annotate_normal_core.py`](scripts/annotate_normal_core.py) feeds all 6 synchronized camera views of a nuScenes keyframe into a VLM together with the normal-core rubric, and reads off a calibrated label and confidence by inspecting the model's output-token logits for the three possible answers (A/B/C), rather than just parsing free text. The exact prompt text is in [`docs/prompt_templates.md`](docs/prompt_templates.md).

3. **Extract internal representations.** [`scripts/extract.py`](scripts/extract.py) feeds the same 6 synchronized camera views of every keyframe into Cosmos-Reason1-7B and uses forward hooks to capture the output of the MLP block (`down_proj` output) of one or more chosen decoder layers (`--layers`, 1-indexed, so `28` is the last block of the 7B language model), stored both mean-pooled over all tokens and as the last-token representation (optionally together with the generated scene description). Critically, the VLM is *not* asked to classify the scene; it is prompted with an open-ended scene-description prompt (also in [`docs/prompt_templates.md`](docs/prompt_templates.md)) so that the extracted hidden state reflects general scene understanding rather than a fixed category list. The row-aligned `meta.json` keeps the `sample_token` and scene of each row, which is what the SAE scripts use to join labels and to split data by scene.

4. **Learn a sparse, long-tail-sensitive decomposition of that representation.** The extracted hidden feature is fed into a sparse autoencoder that splits the latent code into a normal-feature subspace `z_n` and a long-tail-sensitive subspace `z_t`, using **AbsTopK** sparsity (top-`k` latent dimensions by *absolute* activation are kept, `k = 512`) and a tail-guided training objective:

   ```
   L_i = (1 + alpha * y_i) * ||h_hat_i - h_i||^2          # reconstruction, up-weighted for tail samples
       + beta_normal * (1 - y_i) * ||z_t,i||^2            # suppress z_t on normal samples
       + beta_tail   * y_i * max(0, tau - ||z_t,i||_2)    # push z_t above margin tau on tail samples
   ```

   In the code the tail term is implemented as a capped reward, `-beta_tail * min(||z_t||, tau)`, which equals the hinge above up to a constant. Labels are only used during training: at inference a sample is scored by `||z_t||_2` and flagged as long-tail above a threshold chosen on the validation split. This repository implements three variants of this idea, which share all data handling, training and evaluation code in [`scripts/sae_common.py`](scripts/sae_common.py) and are compared in [Ablation study](#ablation-study) below:
   - [`scripts/sae_abstopk_tail_reward.py`](scripts/sae_abstopk_tail_reward.py) — **the final method**: AbsTopK sparsity + a reward term on the *achieved* `||z_t||` for tail samples.
   - [`scripts/ablation_topk_sae.py`](scripts/ablation_topk_sae.py) — ablation ("TopK SAE"): plain (signed) Top-K and no tail reward.
   - [`scripts/ablation_sae_cosmos_baseline.py`](scripts/ablation_sae_cosmos_baseline.py) — ablation ("Train SAE Cosmos"): AbsTopK where a bonus is added to the tail-subspace selection scores *before* Top-K, rather than rewarding the resulting activation norm.
   - [`scripts/pipeline.py`](scripts/pipeline.py) — an earlier, more experimental iteration of this idea (clip-level features with a continuous VLM tail score, `DeepLongTailSAE` with a deeper encoder, a pairwise margin loss, mixup and tail-oversampling augmentation) kept for reference; it predates the AbsTopK + tail-reward formulation above and is not the version the results below were produced with.

## Results

The numbers below are the results reported in the project report. The SAE scripts in this repository use a stricter evaluation protocol than the original experiment scripts: unlabelled / `uncertain` samples are excluded instead of being counted as long-tail, features are standardised with training-split statistics only, the data is split by nuScenes *scene* (so near-identical keyframes of one scene cannot appear in both training and evaluation data), and precision/recall/F1 are measured on a held-out test split with the decision threshold fixed on the validation split. Re-running them may therefore give somewhat lower, but more reliable, numbers than the tables below.

### Main result: long-tail filtering accuracy

Evaluated on a manually-verified long-tail/normal split of nuScenes (labels obtained by pre-labeling with Qwen3.5-397B-A17B and manually verifying), comparing three VLM baselines used directly as classifiers against Cosmos-Reason1-7B equipped with the final SAE (`scripts/sae_abstopk_tail_reward.py`, layer 28):

| Model | long_tail Precision | long_tail Recall | long_tail F1 | normal Precision | normal Recall | normal F1 |
|---|---:|---:|---:|---:|---:|---:|
| Cosmos-Reason1-7B (raw) | 0.1429 | 0.0130 | 0.0238 | 0.8838 | 0.9897 | 0.9338 |
| Qwen3.5-9B (raw) | 0.2900 | 0.3580 | 0.3204 | 0.9103 | 0.8815 | 0.8957 |
| Qwen3-VL-2B-Instruct (raw) | 0.3611 | 0.1135 | 0.1728 | 0.8843 | 0.9712 | 0.9257 |
| **Cosmos-Reason1-7B + SAE (ours)** | **0.9023** | **0.7984** | **0.8472** | 0.8896 | 0.9495 | 0.9186 |

The raw Cosmos-Reason1-7B model is actually the *weakest* of the three baselines at directly classifying long-tail scenes — but once its hidden representations are re-decomposed by the SAE, it becomes the best-performing method by a wide margin. This is the central finding of the project: **the improvement doesn't come from a bigger or better VLM, it comes from the SAE re-decomposing and amplifying long-tail-relevant signal that was already present but diluted in the VLM's dense hidden state.**

On the earlier, smaller-scale annotation-agreement pass (10,120 nuScenes keyframes, used to sanity-check the normal-core rubric before the full pipeline above), Cosmos-Reason1-7B and Qwen3.5-9B independently labeled scenes normal_core / not_normal_core / uncertain and agreed on 83.9% of frames (8,489 / 10,120):

| Annotator | normal_core | not_normal_core | uncertain | Mean confidence |
|---|---:|---:|---:|---:|
| Cosmos-Reason1-7B | 96.9% | 1.1% | 2.0% | 0.970 |
| Qwen3.5-9B | 86.0% | 14.0% | 0.0% | 0.994 |

Most disagreements were Qwen3.5-9B flagging a frame as not_normal_core where Cosmos-Reason1-7B called it normal_core, consistent with Qwen3.5-9B applying a stricter reading of the rubric — a useful secondary signal in its own right, since frames both models agree are normal are strong negative examples for SAE training, and frames where they disagree are natural candidates for human review.

### Ablation study

Ablating the tail-reward term (at layer 28) shows it is responsible for most of the gain — without it, AUC/precision/recall/F1 all drop noticeably. Comparing AbsTopK against plain (signed) Top-K sparsity across hidden layers 15–28 shows AbsTopK + tail reward is consistently the strongest configuration, with performance generally improving in deeper layers and peaking at layer 28 (the model's last hidden layer):

| Method | Layer | Val AUC | Val AP | Best F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| **AbsTopK + Tail Reward** | **L28** | **0.9063** | **0.9082** | **0.8472** | **0.9023** | **0.7984** |
| AbsTopK + Tail Reward | L27 | 0.9033 | 0.9068 | 0.8386 | 0.9212 | 0.7695 |
| AbsTopK + Tail Reward | L24 | 0.9004 | 0.8999 | 0.8294 | 0.9051 | 0.7654 |
| AbsTopK + Tail Reward | L21 | 0.8772 | 0.8120 | 0.8043 | 0.8374 | 0.7737 |
| AbsTopK + Tail Reward | L18 | 0.8733 | 0.8118 | 0.7868 | 0.7876 | 0.7860 |
| AbsTopK + Tail Reward | L15 | 0.8791 | 0.8829 | 0.8091 | 0.9100 | 0.7284 |
| TopK SAE | L28 | 0.8735 | 0.8846 | 0.8269 | 0.9669 | 0.7222 |
| Train SAE Cosmos | L28 | 0.8657 | 0.8623 | 0.8071 | 0.8750 | 0.7490 |

(Per-layer results for all three SAE variants are obtained by extracting several layers at once, e.g. `extract.py --layers 15 18 21 24 27 28`, and running each SAE script on the corresponding `layer{L}_*.npy` file. Note that the "TopK SAE" variant differs from the final method both in the sparsity rule and in having no tail reward; `ablation_topk_sae.py --reward norm` isolates the effect of the sparsity rule alone.) Two further analysis experiments (see the project report) show that mean-pooled features slightly outperform last-token features as SAE input, and that this holds consistently across layers — both consistent with deeper, more semantically-aggregated hidden states being more suitable for the SAE to learn long-tail-relevant structure from.

### Neuron-level interpretability

Individual SAE neurons in the tail subspace correspond to specific, human-interpretable long-tail categories rather than firing on arbitrary noise. Thresholding `|z_t|` at 1.0 and looking at which samples activate each neuron (each SAE script writes the per-neuron activation counts and purity to `neuron_report.csv`; the category names below come from manual inspection of the top-activating samples):

| Dataset | Long-tail feature | Neuron | Score ratio | Activated-sample purity |
|---|---|---:|---:|---|
| nuScenes | Rainy | 2169 | 43.8% | 100.0% (723/723) |
| nuScenes | Glare | 396 | 5.6% | 100.0% (92/92) |
| nuScenes | Disabled persons (wheelchair) | 3058 | 42.4% | 77.8% (14/18) |
| nuScenes | Pedestrians crossing construction zone | 1542 | 6.1% | 53.8% (71/132) |
| nuScenes | Nearby truck | 1843 | 4.2% | 57.1% (12/21) |
| Huawei golden dataset | Snowy | — | 68.5% | 100.0% (137/137) |

Some neurons capture compound risk patterns rather than single visual attributes: neuron 396 fires on glare caused by rain or wet roads specifically (not just any bright light), and neuron 1542 fires on the combination of pedestrians *and* nearby construction, not either alone. Grouping related neurons for the same scenario also gives good category-level coverage — e.g. snowy and rainy/foggy scenarios in the Huawei dataset, and vulnerable-road-user scenarios in nuScenes:

| Dataset | Long-tail feature | Coverage (>=1 related neuron activated) |
|---|---|---:|
| Huawei golden dataset | Snowy | 100.0% |
| Huawei golden dataset | Rainy / Foggy | 92.02% |
| nuScenes | Vulnerable road users | 57.6% |

### Case study: SAE finds a sample that anomaly detection and confidence filtering both miss

A wheelchair user appears near the ego vehicle's path in a sample from nuScenes `part07 / scene-0675`. This sample is a useful stress test because it is genuinely hard for three common long-tail-mining strategies:

- **Unsupervised anomaly detection (KNN in feature space)** does not prioritize it: its outlier degree ranks above only 1.4% of samples, and only 2 of its 10 nearest neighbors are themselves long-tail. The road structure, vehicle distribution and overall scene are all visually close to a normal scene — the thing that makes it long-tail is a small, local detail, not global novelty.
- **Confidence-based / uncertainty-driven filtering** also misses it: every VLM assigns it a high-confidence prediction, so a low-confidence filter would never flag it (0/20 trials across the tested VLMs and re-runs).
- **Semantic-driven methods (directly asking an MLLM)** are inconsistent across model strength: strong MLLMs recognize it reliably (GPT-4o and Gemini 3 Pro Preview both hit "rideable wheelchair" 5/5 times), but a weaker MLLM (Claude Haiku 4.5) does not hit that specific category in any of 5 trials — so this strategy's reliability depends heavily on which (and how expensive a) model you're willing to run at scale.

By contrast, this sample is found by combining two signals from this project's method: every VLM judges the scene as requiring defensive driving under the behavior-oriented long-tail definition (5/5), and SAE neuron 3058 fires on it — the same neuron that reliably fires on other, similar wheelchair-nearby scenes elsewhere in the dataset. The case shows the method can surface long-tail samples that are neither feature-space outliers nor low-confidence predictions, but that still carry real safety relevance.

(Full per-frame annotations and extracted representations are not included in this repository — they run into the gigabytes — but the scripts above reproduce them end to end from public nuScenes data.)

## Repository layout

```
.
├── scripts/
│   ├── annotate_normal_core.py         # VLM-based normal/not-normal/uncertain labeling
│   ├── extract.py                      # Per-keyframe, per-layer hidden-state extraction
│   ├── sae_common.py                   # Shared SAE model, scene-level split, training, evaluation
│   ├── sae_abstopk_tail_reward.py      # Final method: AbsTopK + tail-activation reward
│   ├── ablation_topk_sae.py            # Ablation: signed Top-K, no tail reward
│   ├── ablation_sae_cosmos_baseline.py # Ablation: reward applied before Top-K selection
│   └── pipeline.py                     # Earlier clip-level iteration (DeepLongTailSAE, margin loss)
├── docs/
│   └── prompt_templates.md             # Exact VLM prompts used for the baselines and for feature extraction
├── latest_grading_criteria.md          # The "normal core" annotation rubric
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python >= 3.10 and transformers >= 4.49 (Qwen2.5-VL support). Annotation and feature extraction need a CUDA GPU (bf16 inference, >= 24 GB VRAM recommended); the SAE scripts also run on CPU.

External data/models (not included in this repo — see each script's `--help` for the exact paths):

- Cosmos-Reason1-7B weights from Hugging Face: https://huggingface.co/nvidia/Cosmos-Reason1-7B
- nuScenes metadata (`v1.0-trainval_meta.tgz`) and keyframe images (`v1.0-trainval{01..10}_keyframes.tgz`, extracted so that `samples/CAM_*/` exists): https://www.nuscenes.org/nuscenes

## Usage

```bash
# 1. Label keyframes as normal_core / not_normal_core / uncertain
python scripts/annotate_normal_core.py \
    --samples-root <nuscenes>/samples --meta-tgz <nuscenes>/v1.0-trainval_meta.tgz \
    --output output/annotations/labels.json

# 2. Extract VLM hidden states for the labelled keyframes (one or more layers)
python scripts/extract.py \
    --samples-root <nuscenes>/samples --meta-tgz <nuscenes>/v1.0-trainval_meta.tgz \
    --sample-tokens output/annotations/labels.json --layers 28 \
    --output-dir output/extract

# 3a. Train the final SAE (AbsTopK + tail-activation reward) and evaluate it
python scripts/sae_abstopk_tail_reward.py \
    --features output/extract/layer28_mlp_output_last_token.npy \
    --meta output/extract/meta.json \
    --labels output/annotations/labels.json \
    --output-dir output/sae_abstopk_tail_reward

# 3b. (optional) Ablations: same arguments, different script
python scripts/ablation_topk_sae.py            --features ... --meta ... --labels ... --output-dir output/ablation_topk
python scripts/ablation_sae_cosmos_baseline.py --features ... --meta ... --labels ... --output-dir output/ablation_pre_selection
```

`annotate_normal_core.py` and `extract.py` write their results incrementally and resume interrupted runs by default. Each SAE script writes `metrics.json` (validation and test AUC / AP / precision / recall / F1), `neuron_report.csv`, the learned `z_n` / `z_t` codes, the per-sample scores and split assignment (`rows.json`) and the model checkpoint to `--output-dir`; run any script with `--help` for all hyper-parameters. `--labels` also accepts label files in the `{"samples": [{"sample_token", "labels": {"label"}}]}` format. Labels `normal` / `normal_core` are treated as normal, `uncertain` / `unknown` / empty labels are excluded, and every other label (e.g. `not_normal_core`, `long_tail`) counts as long-tail.

## Acknowledgments

Built on Cosmos-Reason1-7B (NVIDIA) and the nuScenes dataset (Motional). Developed as part of a research collaboration project.

## License

MIT — see LICENSE.
