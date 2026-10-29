# A CPU-only image-feature experiment

This route runs on a CPU. A fixed ImageNet
ResNet18 encoder produces one 512-dimensional vector per camera. Six ordered
views are concatenated into 3,072 dimensions. Logistic regression, MLP and the
reward/no-reward SAE then share those features, the same frozen scenes, labels
and evaluation rules. This is an image-only reference experiment and does not
reproduce Cosmos hidden states or inherit the historical VLM results.

Use the self-contained [CPU notebook](../notebooks/driving_cpu_from_zero.ipynb)
in a Python 3.12 Jupyter/Colab CPU runtime. It embeds the source, acquires official
data and the small encoder, persists manual review, checks the feature cache and
runs the four methods. The runtime must normally permit the official download
hosts. No GPU selection, paid service or cloud booking is needed.

## Completed runs

The project has processed 89 real timestamps from 12 scenes: the original
80-timestamp mini cohort and a separate nine-timestamp supplement, totaling
534 camera images. The model fits still use only the original cohort.
[Yike's review](human-review.md) retained 77 binary decisions and three uncertain
rows after prior AI-label disclosure. All four methods used the frozen 6/2/2
scene allocation and seed 42. Train contains only three normal timestamps,
all from one scene; test contains 15 retained timestamps in two scenes.
The SAE's test AP was 0.9353 against those single-reviewer references.
Reward and no-reward checkpoints were identical, so this run found no reward gain.

The nine later timestamps contain five normal and four non-normal human
judgments, also submitted after AI-reference disclosure. The old
validation-selected threshold, `6.074954873387325`, selected five candidates:
precision 60%, recall 75%. No new fitting or threshold tuning occurred. All
normal references came from one scene and all non-normal references from the
other, confounding class with scene. See the
[supplemental replay](../results/new-scene-human-review-20261028/README.md).

### Encoder and fixture checks

On 2026-10-23, the official 46,830,571-byte pretrained checkpoint was downloaded
from PyTorch's HTTPS host and pinned by its full SHA256:

```text
f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec
```

The encoder extracted 20 synthetic six-view samples (120 drawings), including
a one-sample checkpoint followed by a verified resume. Extraction took about
4.74 seconds with two CPU threads; peak process RSS was about 354 MiB. These
numbers describe that local fixture and include no real driving images or human
judgments. They do not measure accuracy or promise real-dataset throughput.

The same 20-sample fixture passed strict handoff validation, all four CPU model
fits and table rebuilding. A further 10 unlabelled synthetic samples had zero
token overlap with training; their extracted features passed the screening
encoder/weights/preprocessing check. A six-view synthetic false-positive preview
and a blank-label review export are retained with the local verification artifacts.

### Original real-data extraction

On 2026-10-23, four user-generated offline transfer parts were imported from
an archive recorded as `v1.0-mini.tgz` (4,168,148,189 bytes). The imported subset
contains 80 real timestamps from ten scenes and 480 original camera JPEGs.
Five original metadata tables and all selected image hashes were verified.
The custom mini split was frozen before labels: six/two/two scenes and
48/16/16 timestamps for train/validation/test. No selected frames were excluded.

The pinned encoder completed a one-sample smoke run and resumed the remaining
79 timestamps with two CPU threads. The completed cache has shape `(80, 3072)`;
the smoke invocation took 2.18 seconds and the resume invocation took 25.79
seconds, including process startup and input checks. These are local run times,
not a throughput benchmark. Logs and the preserved smoke/full summaries are
under `output/real-cpu-mini-20261023/` in the execution workspace.

Before the human submission, the [AI-label pilot](ai-pilot.md) fitted all four
models on 73 retained timestamps and screened seven uncertain timestamps.
That experiment remains separate from the later human-review run. The original
archive checksum was not independently verified; transfer and file hashes
establish which bytes were used, not official-source authenticity. The worker's
network policy blocked the official download host during acquisition, so the
local upload route supplied the inputs.

If the mini archive is already downloaded on your computer, the
[offline small-part transfer tool](local-upload.md) selects a tiny six-view
subset locally and makes uploadable ZIPs, without a separate hosting service.

## Run with local data

Keep CPU outputs separate from the Cosmos cache. From the repository root:

```bash
bash scripts/cloud.sh setup
bash scripts/cloud.sh acquire-cpu-encoder
bash scripts/cloud.sh acquire-mini OUTPUT=output/real-cpu
bash scripts/cloud.sh prepare-real OUTPUT=output/real-cpu
```

The acquisition command also accepts an existing official archive at
`dataset/downloads/v1.0-mini.tgz`. Follow the frozen rubric, inspect all six views
and complete `reviewed.csv` using the [manual-review workflow](from-zero.md#2-prepare-and-label-on-cpu).
Then import it without changing the already frozen scene allocation:

```bash
bash scripts/cloud.sh import-labels OUTPUT=output/real-cpu REVIEWS=output/real-cpu/reviewed.csv
bash scripts/cloud.sh extract-cpu OUTPUT=output/real-cpu MAX_SAMPLES=1
```

Preserve `extract/extraction_summary.json` as the one-sample resource check.
Complete the same cache, validate it and train with a review budget chosen
before reading test results:

```bash
bash scripts/cloud.sh extract-cpu OUTPUT=output/real-cpu MAX_SAMPLES=0
bash scripts/cloud.sh cpu-validate OUTPUT=output/real-cpu
bash scripts/cloud.sh cpu-train OUTPUT=output/real-cpu REVIEW_K=10
```

`REVIEW_K=10` is an example budget. Uncertain or pending labels stay excluded;
if a frozen split lacks a class, preserve that limitation and obtain more data.
Open the trained SAE in the existing explorer with
`bash scripts/cloud.sh cpu-explore OUTPUT=output/real-cpu`. To screen another
completed CPU cache, use `bash scripts/cloud.sh cpu-screen OUTPUT=output/real-cpu
FEATURES=path/to/new/extract`. The screener checks the recorded encoder, weights
and preprocessing against training; the new samples do not need labels.
These CPU entry points omit historical neuron names. Do not reuse a Cosmos
glossary for newly learned CPU-model units; name units from train/validation
evidence and evaluate those frozen interpretations separately.

## Identity and limitations

The feature schema is `cpu-resnet18-six-camera-v1`; validation requires the exact
official weights, preprocessing, image hashes and 3,072 dimensions. CPU caches
cannot pass the Cosmos validator or be mixed into its results table. The legacy
`layer1_mlp_output_mean.npy` and `layer1_mlp_output_last_token.npy` files are
identical storage compatibility aliases of the concatenated camera features;
they do not represent language-model layers, token pooling or an MLP hook.

ImageNet pretraining supplies generic visual features. Resizing to a 256-pixel
short side and taking a 224-pixel center crop may discard small hazards and
peripheral context; this needs failure analysis on real images. Learned SAE
units still require train/validation interpretation and independent test checks.
Human labels, scene separation and honest negative findings remain necessary.

The minimal encoder architecture is adapted from torchvision with
[BSD attribution](third_party_resnet18.txt). Rebuild the embedded source with
`bash scripts/cloud.sh cpu-notebook` after source changes.
