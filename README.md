# Long-Tail Driving Scene Discovery

Finding rare ("long-tail") driving scenarios in large autonomous-driving datasets by looking inside the representations of a vision-language model (VLM), rather than relying on hand-written rules or frequency counts.

The core idea: instead of trying to enumerate every possible rare or risky driving situation in advance, define a tight, high-confidence notion of a **"normal" driving scene**, learn what that looks like inside a VLM's internal representations, and then treat anything that deviates from that learned "normal manifold" as a candidate long-tail sample worth a closer look.

Built on [NVIDIA Cosmos-Reason1-7B](https://huggingface.co/nvidia/Cosmos-Reason1-7B) (a Qwen2.5-VL-based multimodal reasoning model) and the [nuScenes](https://www.nuscenes.org/nuscenes) autonomous-driving dataset.

## Why this problem

Autonomous-driving perception stacks are usually evaluated on average-case performance, but the failures that matter happen in the tail of the distribution — unusual lighting, occlusions, atypical road layouts, rare object configurations. Long-tail data is hard to find because it's rare by definition: keyword search and simple heuristics don't scale, and manually reviewing millions of frames isn't feasible.

This project explores a representation-based alternative: use a large VLM as a feature extractor, define what "normal" looks like as precisely as possible, and let outlier detection in the learned feature space surface candidate long-tail frames for human review.

## Method

1. Define "normal" precisely. Rather than trying to exhaustively label every long-tail category, the project defines a strict normal core: a single-frame, multi-camera driving scene where the ego vehicle would not need to take any extra defensive action (see latest_grading_criteria.md for the full annotation rubric, with inclusion/exclusion criteria and a normal_core / not_normal_core / uncertain protocol).

2. Annotate at scale with a VLM. scripts/annotate_normal_core.py feeds all 6 synchronized camera views of a nuScenes keyframe into a VLM together with the normal-core rubric, and reads off a calibrated label and confidence by inspecting the model's output-token logits for the three possible answers (A/B/C), rather than just parsing free text.

3. Extract internal representations. scripts/extract.py runs Cosmos-Reason1-7B over each 6-camera clip and pulls out the last-layer MLP's input and output activations (mean-pooled and last-token variants), plus a free-form natural-language scene description, for every clip.

4. Learn the normal manifold and score deviation. scripts/pipeline.py takes the pre-MLP hidden states, trains a sparse autoencoder (DeepLongTailSAE, with a pairwise margin loss, mixup, and tail-oversampling) to reconstruct "normal" samples well and rare samples poorly, and reports pre-/post-SAE separability metrics (AUC, raw L2 distance, mean-difference and logistic-regression baselines) plus a semantic interpretability pass that surfaces the top activating clips per SAE neuron.

## Results

Two VLMs (Cosmos-Reason1-7B and Qwen3.5-9B) were run as independent annotators of the normal-core rubric over 10,120 nuScenes keyframes:

| Annotator | normal_core | not_normal_core | uncertain | Mean confidence |
|---|---:|---:|---:|---:|
| Cosmos-Reason1-7B | 96.9% | 1.1% | 2.0% | 0.970 |
| Qwen3.5-9B | 86.0% | 14.0% | 0.0% | 0.994 |

The two models agree on 83.9% of frames (8,489 / 10,120). Most disagreements are Qwen3.5-9B flagging a frame as not_normal_core where Cosmos-Reason1-7B calls it normal_core — consistent with Qwen3.5-9B applying a stricter interpretation of the rubric, and it suggests that a single VLM's "normal" judgment is not fully calibrated across models. This kind of inter-model disagreement is itself a useful signal: frames both models are confident are normal are strong negatives for training the SAE, and frames where the two disagree are natural candidates for human review.

(Full per-frame annotations and extracted representations are not included in this repository — they run into the gigabytes — but the scripts above reproduce them end to end from public nuScenes data.)

## Repository layout

```
.
├── scripts/
│   ├── annotate_normal_core.py   # VLM-based normal/not-normal/uncertain labeling
│   ├── extract.py                # Cosmos-Reason1 representation + caption extraction
│   ├── pipeline.py               # Hidden-state extraction -> SAE training -> interpretability
│   └── .env.example              # API key template (for the optional DSPy prompt-optimization path)
├── latest_grading_criteria.md    # The "normal core" annotation rubric
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python >= 3.10, a CUDA GPU (bf16 inference, >= 24 GB VRAM recommended), and transformers >= 4.49 (for Qwen2_5_VLForConditionalGeneration).

External data/models (not included in this repo — see each script's --help for exact paths):

- Cosmos-Reason1-7B weights from Hugging Face: https://huggingface.co/nvidia/Cosmos-Reason1-7B
- nuScenes keyframe archives (v1.0-trainval0{1..10}_keyframes.tgz): https://www.nuscenes.org/nuscenes
- A clip-level annotation file with tail_score per clip (produced upstream, or substitute your own)

## Usage

```bash
# 1. Label frames as normal_core / not_normal_core / uncertain
python scripts/annotate_normal_core.py --samples-root <nuscenes_root> --output annotations.json

# 2. Extract VLM representations for a batch of clips
python scripts/extract.py --parts 4 5 6 --tgz-dir <nuscenes_dir> --output-dir output/cosmos_extract_output

# 3. Train the sparse autoencoder and run the interpretability analysis
python scripts/pipeline.py --parts 1 2 3
```

Each script is independently runnable and supports --resume for interrupted runs; see the top-of-file docstring in each script for the full set of options.

## Acknowledgments

Built on Cosmos-Reason1-7B (NVIDIA) and the nuScenes dataset (Motional). Developed as part of a research collaboration project.

## License

MIT — see LICENSE.

