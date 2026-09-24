Results
=============

All numbers are measured on a manually-verified long-tail / normal split of
nuScenes: samples were pre-labelled with a large VLM under the
defensive-driving rubric in
[`latest_grading_criteria.md`](../latest_grading_criteria.md) and then verified
by hand, so the evaluation labels are human-checked rather than model-generated.

The protocol behind every table: unlabelled and `uncertain` samples are
excluded rather than counted as long-tail, features are standardised with
training-split statistics only, the split is taken at the nuScenes *scene*
level so near-identical keyframes cannot appear on both sides, and the decision
threshold is chosen on the validation split and applied unchanged to the
held-out test split.

## Main result: long-tail filtering

Three VLMs used directly as classifiers, against Cosmos-Reason1-7B equipped
with the final SAE (`scripts/sae_abstopk_tail_reward.py`, layer 28):

| Model | long_tail P | long_tail R | long_tail F1 | normal P | normal R | normal F1 |
|---|---:|---:|---:|---:|---:|---:|
| Cosmos-Reason1-7B (raw) | 0.1429 | 0.0130 | 0.0238 | 0.8838 | 0.9897 | 0.9338 |
| Qwen3.5-9B (raw) | 0.2900 | 0.3580 | 0.3204 | 0.9103 | 0.8815 | 0.8957 |
| Qwen3-VL-2B-Instruct (raw) | 0.3611 | 0.1135 | 0.1728 | 0.8843 | 0.9712 | 0.9257 |
| **Cosmos-Reason1-7B + SAE (ours)** | **0.9023** | **0.7984** | **0.8472** | 0.8896 | 0.9495 | 0.9186 |

All four use the same behaviour-oriented long-tail definition as their decision
criterion; the baselines are prompted to apply it directly, while the SAE-based
method never asks the VLM for a label at all and reads its hidden state
instead.

## Annotation agreement

An earlier smaller-scale pass over 10,120 nuScenes keyframes sanity-checked the
rubric before the full pipeline: two VLMs independently labelled every frame
and agreed on 83.9% of them (8,489 / 10,120).

| Annotator | normal_core | not_normal_core | uncertain | Mean confidence |
|---|---:|---:|---:|---:|
| Cosmos-Reason1-7B | 96.9% | 1.1% | 2.0% | 0.970 |
| Qwen3.5-9B | 86.0% | 14.0% | 0.0% | 0.994 |

Most disagreements are Qwen3.5-9B flagging a frame that Cosmos-Reason1-7B calls
normal, consistent with a stricter reading of the rubric. The disagreement is a
useful signal in itself: frames both models call normal are strong negatives
for SAE training, and frames they disagree on are natural candidates for human
review.

## Ablation: the objective and the sparsity rule

Ablating the tail-reward term at layer 28 accounts for most of the gain.
Comparing AbsTopK against plain signed Top-K across layers 15-28 shows AbsTopK
with the tail reward is the strongest configuration at every depth, with
performance improving in deeper layers and peaking at layer 28, the model's
last hidden layer:

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

Deeper layers carrying more of the signal matches what the decomposition is
being asked to do: later hidden states hold higher-level, semantically
aggregated information, which is where behaviour-relevant structure lives.

Two further analyses: mean-pooled features slightly outperform last-token
features as SAE input, and that holds consistently across layers — again
consistent with more aggregated representations suiting the task.

The per-layer numbers are produced by extracting several layers in one pass,
`extract.py --layers 15 18 21 24 27 28`, and running each SAE script on the
corresponding `layer{L}_*.npy` file. The "TopK SAE" row differs from the final
method in both the sparsity rule and the absence of a tail reward;
`ablation_topk_sae.py --reward norm` isolates the effect of the sparsity rule
alone.

## Interpretable neurons

Thresholding `|z_t|` at 1.0 and inspecting which frames activate each unit.
*Activation ratio* is the share of that scenario's frames that activate the
unit; *purity* is the share of activated frames that really belong to the
scenario. Names come from reading the top-activating frames:

| Long-tail feature | Unit | Activation ratio | Purity |
|---|---:|---:|---|
| Rain / wet road | 2169 | 43.8% | 100.0% (723/723) |
| Glare on a wet or rainy road | 396 | 5.6% | 100.0% (92/92) |
| Wheelchair user ahead | 3058 | 42.4% | 77.8% (14/18) |
| Pedestrian in a construction zone | 1542 | 6.1% | 53.8% (71/132) |
| Truck close to the ego vehicle | 1843 | 4.2% | 57.1% (12/21) |

A single unit captures one sub-pattern; asking how often *at least one* unit of
a related group fires gives the coverage of a whole scenario family:

| Scenario | Coverage by a related group |
|---|---:|
| Snowy | 100.0% |
| Rainy / foggy | 92.0% |
| Vulnerable road users | 57.6% |

The machine-readable version is
[`neuron_glossary.csv`](neuron_glossary.csv), which the screening tool joins
against so a flagged frame carries a reason.
[`docs/neurons.md`](../docs/neurons.md) covers how the units are scored and how
to rebuild the glossary for another run.

## Case study: a sample anomaly detection and confidence filtering both miss

A wheelchair user appears near the ego vehicle's path in a frame from nuScenes
`part07 / scene-0675`. It is a useful stress test because it is genuinely hard
for three common mining strategies:

- **Unsupervised anomaly detection (KNN in feature space)** does not prioritise
  it. Its outlier degree ranks above only 1.4% of samples, and only 2 of its 10
  nearest neighbours are themselves long-tail. The road structure, the vehicle
  distribution and the overall scene are all close to a normal frame — what
  makes it long-tail is a small local detail, not global novelty.
- **Confidence-based filtering** misses it too. Every VLM assigns it a
  high-confidence prediction, so a low-confidence filter never surfaces it
  (0/20 trials across the tested VLMs and re-runs).
- **Asking an MLLM directly** depends on which model you can afford to run at
  scale. GPT-4o and Gemini 3 Pro Preview both hit "rideable wheelchair" 5/5
  times; Claude Haiku 4.5 does not hit that category in any of 5 trials,
  attributing the scene to traffic cones or strollers instead.

The method finds it by combining two signals. Every VLM judges the scene as
requiring defensive driving under the behaviour-oriented definition (5/5), and
SAE unit 3058 fires on it — the same unit that fires on other wheelchair-nearby
frames elsewhere in the dataset. The frame is neither a feature-space outlier
nor a low-confidence prediction, and it still carries real safety relevance.

## Reproducing

```bash
make label DATA=<nuscenes>            # writes output/annotations/labels.json
make extract DATA=<nuscenes> LAYERS=28
make train                            # writes output/sae_abstopk_tail_reward/
make ablations                        # the two ablation variants
```

Each SAE run writes `metrics.json` (validation and test AUC / AP / precision /
recall / F1), `neuron_report.csv`, the learned `z_n` and `z_t` codes, the
per-sample scores and split assignment in `rows.json`, the standardiser and the
model checkpoint. [`docs/user.md`](../docs/user.md) documents every flag.
