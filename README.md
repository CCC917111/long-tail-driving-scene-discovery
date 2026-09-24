Long-Tail Driving Scene Discovery
=============
Welcome to the long-tail driving data mining project main page!
This page is about how to run this software.

#### Install

```bash
python -m venv .venv && source .venv/bin/activate
make install                      # pip install -r requirements.txt
```

#### Label

```bash
# Label keyframes normal_core / not_normal_core / uncertain with a VLM
make label DATA=<nuscenes>
```

#### Extract

```bash
# Capture the VLM's hidden state for every labelled keyframe
make extract DATA=<nuscenes> LAYERS=28
```

#### Train

```bash
# Train the tail-guided sparse autoencoder and evaluate it
make train
```

#### Screen

```bash
# Rank your own unlabelled frames by how long-tail they look
make screen RUN=output/sae_abstopk_tail_reward FEATURES=output/extract_mydata
```

#### Run Tests

```bash
make test                         # python -m pytest tests -q
```

#### Check Style

```bash
make lint                         # flake8, 100 columns
```

#### Clean the Project

```bash
make clean
```

Labelling and extraction need a CUDA GPU and a local copy of the VLM weights;
the SAE steps run on CPU. The full usage guide, with every flag and the files
each command writes, is in [`docs/user.md`](docs/user.md).

----------------

User page
=============
Welcome to the long-tail driving data mining project user main page!

This project finds rare, safety-relevant driving scenarios in large
autonomous-driving datasets by looking *inside* the representations of a
vision-language model, instead of relying on hand-written rules, keyword
search or frequency counts. It contains the full pipeline — the annotation
rubric and the VLM labeller that applies it, the hidden-state extractor, the
sparse autoencoder that decomposes those hidden states, the evaluation against
VLM baselines, and a screening tool that applies a trained model to new,
unlabelled driving data.

# Abstract

Long-tail scenarios are rare but safety-critical in autonomous driving, and
their insufficient coverage is a central obstacle to reliable deployment.
Existing mining methods rely on explicit semantic descriptions, anomaly scores,
uncertainty estimates or synthetic data, which miss visually subtle or
high-confidence long-tail cases. This project proposes a Sparse Autoencoder
(SAE) framework for mining long-tail driving data from the hidden
representations of a frozen vision-language model. It adopts a
driving-behavior-oriented definition, treating underrepresented samples that
require additional defensive-driving behaviour as long-tail candidates. Given a
driving sample, the method extracts internal hidden features from a frozen VLM
and trains a tail-guided SAE to decompose them into sparse normal and
long-tail-sensitive latent features. At inference a sample is identified as a
long-tail candidate when the learned long-tail subspace is activated.
Experiments on nuScenes show that the method improves long-tail filtering
performance over VLM baselines and learns interpretable neurons associated with
specific risk patterns, and a case study shows it identifies safety-critical
samples that anomaly- and uncertainty-based mining miss.

![Pipeline](docs/figures/pipeline.svg)

## Why this problem

Autonomous-driving perception stacks are usually evaluated on average-case
performance, but the failures that matter happen in the tail of the
distribution — unusual lighting, occlusions, atypical road layouts, rare object
configurations. Long-tail data is hard to find because it is rare by
definition: keyword search and simple heuristics do not scale, and manually
reviewing millions of frames is not feasible.

Purely unsupervised outlier detection is not a good fit either. Many
safety-relevant long-tail scenes are not visual outliers in feature space at
all; what makes them long-tail is a local, safety-relevant detail rather than
global scene novelty. A wheelchair user near the ego path sits in an otherwise
completely ordinary urban street.

This project takes a representation-based route: use a large VLM as a feature
extractor, define "normal" and "long-tail" as precisely as possible in
behavioural terms, and train a sparse autoencoder to carve the VLM's dense
hidden state into an interpretable, long-tail-sensitive subspace whose
activation is itself the detector.

## Method

**1. Define "normal" and "long-tail" precisely.** The project defines a strict
normal core: a single-frame, multi-camera driving scene where the ego vehicle
would not need to take any extra defensive action. The full rubric, with
inclusion and exclusion criteria and a normal_core / not_normal_core /
uncertain protocol, is in
[`latest_grading_criteria.md`](latest_grading_criteria.md). A sample counts as
a long-tail candidate if it is both relatively rare in natural driving data and
likely to require additional defensive driving — slowing down, yielding,
keeping a larger safety margin, paying closer attention to nearby risks.

**2. Annotate at scale with a VLM.**
[`scripts/annotate_normal_core.py`](scripts/annotate_normal_core.py) feeds all
six synchronised camera views of a keyframe to a VLM together with the rubric,
and reads off a calibrated label and confidence by inspecting the output-token
logits for the three possible answers, rather than parsing free text. The exact
prompts are in [`docs/prompt_templates.md`](docs/prompt_templates.md).

**3. Extract internal representations.**
[`scripts/extract.py`](scripts/extract.py) feeds the same six views into
Cosmos-Reason1-7B and uses forward hooks to capture the MLP block output
(`down_proj`) of one or more decoder layers, stored both mean-pooled over
tokens and as the last-token representation. The VLM is deliberately *not*
asked to classify the scene: it is given an open-ended scene-description
prompt, so the hidden state reflects general scene understanding rather than a
fixed category list. A row-aligned `meta.json` keeps each row's sample token
and scene, which is what lets the SAE split data by scene.

**4. Learn a sparse, long-tail-sensitive decomposition.** The hidden feature
goes into a sparse autoencoder whose latent code is split into a normal
subspace `z_n` and a long-tail-sensitive subspace `z_t`, with **AbsTopK**
sparsity — the top `k = 512` latent units by *absolute* activation are kept —
and a tail-guided objective:

```
L_i = (1 + alpha * y_i) * ||h_hat_i - h_i||^2          # reconstruction, up-weighted for tail samples
    + beta_normal * (1 - y_i) * ||z_t,i||^2            # suppress z_t on normal samples
    + beta_tail   * y_i * max(0, tau - ||z_t,i||_2)    # push z_t above margin tau on tail samples
```

In the code the tail term is implemented as a capped reward,
`-beta_tail * min(||z_t||, tau)`, which equals the hinge above up to a
constant.

**5. Score and decide.** Labels are used only during training. At inference a
sample is scored by `||z_t||_2` and flagged as long-tail above a threshold
chosen on the validation split and then applied unchanged to the held-out test
split.

The evaluation protocol is strict throughout: unlabelled and `uncertain`
samples are excluded rather than counted as long-tail, features are standardised
with training-split statistics only, and the data is split by nuScenes *scene*,
so near-identical keyframes from one scene can never appear on both sides of
the split.

Three variants of the idea share all data handling, training and evaluation
code in [`scripts/sae_common.py`](scripts/sae_common.py):

- [`scripts/sae_abstopk_tail_reward.py`](scripts/sae_abstopk_tail_reward.py) —
  **the final method**: AbsTopK sparsity plus a reward on the achieved
  `||z_t||` of tail samples.
- [`scripts/ablation_topk_sae.py`](scripts/ablation_topk_sae.py) — ablation:
  plain signed Top-K, no tail reward.
- [`scripts/ablation_sae_cosmos_baseline.py`](scripts/ablation_sae_cosmos_baseline.py)
  — ablation: AbsTopK where the bonus is added to the tail-subspace *selection*
  scores before Top-K, rather than rewarding the resulting activation norm.
- [`scripts/pipeline.py`](scripts/pipeline.py) — a clip-level variant of the
  same idea, with a continuous VLM tail score, a deeper encoder
  (`DeepLongTailSAE`), a pairwise margin loss, mixup and tail oversampling.

## Results

Evaluated on a manually-verified long-tail/normal split of nuScenes, comparing
three VLM baselines used directly as classifiers against Cosmos-Reason1-7B
equipped with the final SAE at layer 28:

| Model | long_tail Precision | long_tail Recall | long_tail F1 | normal F1 |
|---|---:|---:|---:|---:|
| Cosmos-Reason1-7B (raw) | 0.1429 | 0.0130 | 0.0238 | 0.9338 |
| Qwen3.5-9B (raw) | 0.2900 | 0.3580 | 0.3204 | 0.8957 |
| Qwen3-VL-2B-Instruct (raw) | 0.3611 | 0.1135 | 0.1728 | 0.9257 |
| **Cosmos-Reason1-7B + SAE (ours)** | **0.9023** | **0.7984** | **0.8472** | 0.9186 |

The raw Cosmos-Reason1-7B model is the *weakest* of the three baselines at
directly classifying long-tail scenes — and once its hidden representations are
re-decomposed by the SAE, it becomes the best method by a wide margin. That is
the central finding: the improvement does not come from a bigger or better VLM,
it comes from the SAE re-decomposing and amplifying long-tail-relevant signal
that was already present but diluted in the dense hidden state.

Ablating the tail reward accounts for most of the gain, and AbsTopK beats
signed Top-K at every layer tested, with performance improving in deeper layers
and peaking at layer 28. The full tables, including the per-layer sweep and the
wheelchair case study, are in [`results/README.md`](results/README.md).

## The interpretable neurons

Because the latent code is sparse, a flagged frame comes with an explanation:
which units fired. Several of those units turn out to stand for one specific,
nameable situation.

| Unit | Feature | Activation ratio | Purity |
|---:|---|---:|---:|
| 2169 | Rain / wet road | 43.8% | 100.0% (723/723) |
| 396 | Glare on a wet or rainy road | 5.6% | 100.0% (92/92) |
| 3058 | Wheelchair user ahead | 42.4% | 77.8% (14/18) |
| 1542 | Pedestrian in a construction zone | 6.1% | 53.8% (71/132) |
| 1843 | Truck close to the ego vehicle | 4.2% | 57.1% (12/21) |

Two of these are more than "the model learned to see rain". Unit 396 is not a
brightness detector — it fires on glare *caused by* rain or a wet surface, and
keeps doing so across different parts of the dataset. Unit 1542 is a
conjunction: pedestrians **in** construction areas, not either alone. A
rule-based miner would need someone to have written that combination down in
advance.

Once a unit has a name it becomes a retrieval key: filter on the unit instead
of on the score and you get a categorical search over a corpus nobody labelled
for that category. The named units live in
[`results/neuron_glossary.csv`](results/neuron_glossary.csv) so the screening
tool can attach them as reasons, and
[`docs/neurons.md`](docs/neurons.md) explains how they are scored, why single
units mean anything at all, and how to rebuild the glossary for a run of your
own.

## Screening your own driving data

The benchmark says how good the method is; this is how you point it at your own
archive. Extract features from your frames with the same VLM and layer, then:

```bash
make screen RUN=output/sae_abstopk_tail_reward FEATURES=output/extract_mydata
```

No labels are needed — labels are only used to *train* the SAE. You get back
`screening.csv`, every frame ranked by `||z_t||`, with the activated units and
their names, plus `long_tail_samples.json` ready to hand to an annotation tool:

```
screened 12480 frames | flagged 1163 (9.3%) as long-tail

why they were flagged:
    612  Rain / wet road
     97  Glare on a wet or rainy road
     41  Pedestrian in a construction zone
     14  Wheelchair user ahead
```

That tally is a profile of what rare situations your archive contains,
computed without anyone having labelled it. The input contract, the output
columns and how to work with the ranking are in
[`docs/apply.md`](docs/apply.md).

----------------

Developer page
=============
Welcome to the long-tail driving data mining project developer main page!

----------------
## Abstract

The repository is organised as a set of command-line scripts around one shared
library. `scripts/sae_common.py` holds the SAE model, the scene-level split,
the training loop and the evaluation protocol; the three SAE entry points
differ only in their default hyper-parameters, which is what makes the ablation
a fair comparison. `annotate_normal_core.py` and `extract.py` are the two VLM
stages, `screen.py` applies a trained run to new data, `docs/` holds the pages
below and `results/` holds the reported numbers and the neuron glossary.

For more information about the software, select the following pages.

----------------
## [Screening Your Own Driving Data](docs/apply.md)

What the tool is for, the input contract, what the ranking contains and how to
spend an annotation budget with it.

## [The Interpretable Neurons](docs/neurons.md)

How a unit is scored, which units are named, why sparsity makes single units
mean something, and how to rebuild the glossary for your own run.

## [Developer Overview](docs/developer.md)

The layout of the repository, the path a frame takes through it, and where to
start for the change you have in mind.

## [Programming Reference](docs/reference.md)

The module-by-module map: what each script provides, the public functions and
classes, their arguments and the array shapes they expect.

## [High-level Design](docs/design.md)

The method in detail, with the equations of the sparse decomposition and the
reasoning behind the objective, the sparsity rule and the evaluation protocol.

## [Coding Style](docs/coding.md)

The style the code follows and how to check it.

## [Common Tasks](docs/tasks.md)

How to extend the project: swap the VLM, change the long-tail definition, add
an SAE variant, or run on a different dataset.

## [Testing](docs/testing.md)

The testing strategy, what each test covers and how to run the suite.

----------------
## Repository layout

```
.
├── scripts/
│   ├── annotate_normal_core.py         # VLM labelling against the normal-core rubric
│   ├── extract.py                      # Per-keyframe, per-layer hidden-state extraction
│   ├── sae_common.py                   # SAE model, scene-level split, training, evaluation
│   ├── sae_abstopk_tail_reward.py      # Final method: AbsTopK + tail-activation reward
│   ├── ablation_topk_sae.py            # Ablation: signed Top-K, no tail reward
│   ├── ablation_sae_cosmos_baseline.py # Ablation: reward applied before Top-K selection
│   ├── pipeline.py                     # Clip-level variant: DeepLongTailSAE, margin loss
│   └── screen.py                       # Apply a trained run to your own frames
├── docs/                               # Developer documentation, prompts and figures
├── tests/                              # Unit tests for the shared SAE library
├── results/                            # Reported metrics and the neuron glossary
├── latest_grading_criteria.md          # The normal-core annotation rubric
├── requirements.txt
└── Makefile                            # install / label / extract / train / screen / test
```

## Setup

Requires Python 3.10 or newer and transformers 4.49 or newer (Qwen2.5-VL
support). Annotation and feature extraction need a CUDA GPU (bf16 inference,
24 GB VRAM or more recommended); the SAE and screening steps also run on CPU.

External data and models, not included here — see each script's `--help` for
the exact paths:

- Cosmos-Reason1-7B weights:
  https://huggingface.co/nvidia/Cosmos-Reason1-7B
- nuScenes metadata (`v1.0-trainval_meta.tgz`) and keyframe images
  (`v1.0-trainval{01..10}_keyframes.tgz`, extracted so that `samples/CAM_*/`
  exists): https://www.nuscenes.org/nuscenes

Per-frame annotations and extracted representations are not included in the
repository — they run into the gigabytes — but the scripts reproduce them end
to end from public nuScenes data.

## References

1. Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2023).
   *Sparse Autoencoders Find Highly Interpretable Features in Language Models.*
   [arXiv:2309.08600](https://arxiv.org/abs/2309.08600)
2. Templeton, A., et al. (2024). *Scaling Monosemanticity: Extracting
   Interpretable Features from Claude 3 Sonnet.* Transformer Circuits Thread.
3. Pach, M., Karthik, S., Bouniot, Q., Belongie, S., & Akata, Z. (2025).
   *Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models.*
   [arXiv:2504.02821](https://arxiv.org/abs/2504.02821)
4. Ye, M., et al. (2024). *VLMine: Long-Tail Data Mining with Vision Language
   Models.* [arXiv:2409.15486](https://arxiv.org/abs/2409.15486)
5. Liu, H. X., & Feng, S. (2022). *"Curse of Rarity" for Autonomous Vehicles.*
   [arXiv:2207.02749](https://arxiv.org/abs/2207.02749)

## Acknowledgments

Built on Cosmos-Reason1-7B (NVIDIA) and the nuScenes dataset (Motional).
Developed as part of a research collaboration project.

## License

MIT — see [LICENSE](LICENSE).
