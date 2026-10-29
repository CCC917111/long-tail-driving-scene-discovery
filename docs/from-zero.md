# Starting with no nuScenes data and no GPU

The completed [CPU image-feature route](cpu-route.md) uses pretrained ResNet18
and needs no GPU. Across the fitted cohort and supplemental intake, it has
processed 89 six-view timestamps from 12 scenes: 534 original images. The
models were fitted on the older 80 timestamps, using 77 binary human references
and excluding three uncertain decisions. That review followed AI-reference
disclosure. The later nine-timestamp batch used the saved model and old
validation threshold, with no new fit or tuning. See [the fitted result](../results/human-ai-exposed-20261026/README.md)
and [supplemental review](../results/new-scene-human-review-20261028/README.md)
for the evaluation limits.

This page documents acquisition, human review and the optional original
Cosmos/VLM route. VLM extraction has not run in this environment. Prepare and
review data on CPU, measure one frame on a suitable GPU, cache the features,
then return to CPU for fitting and browsing. Commands run from the repository
root. Official download and hardware references below were checked on
2026-10-20–21; check their current requirements before use.

The portable version is
[`notebooks/driving_from_zero.ipynb`](../notebooks/driving_from_zero.ipynb).
It embeds the current source, runs in Jupyter/Colab, saves review decisions as
they are made, and includes a small review backup/restore before changing
runtimes. It uses the selected runtime's normal network policy and does not
book or purchase a GPU. Bootstrap, local fixture preparation, backup/restore
and CPU training have software checks; no complete real-data/GPU notebook run
has been performed. Rebuild it with `make notebook` after source changes.

## 1. Obtain the small official dataset

Run `bash scripts/cloud.sh setup` to select the project's local `.venv`,
the verified shared `/workspace/yike-top2-venv`, or a new project-local CPU
venv. Set `DRIVING_PYTHON` to select another interpreter. Setup records CPU
lock comparisons and prerequisite checks under `output/cloud/`. Platform
network access and GPU allocation are configured separately.

The [official mini tutorial](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/tutorials/nuscenes_tutorial.ipynb)
publishes `https://www.nuscenes.org/data/v1.0-mini.tgz` as a direct download.
Try that published mini route first; do not assume portal registration is
required for every mini download. The
[AWS Open Data registry](https://github.com/awslabs/open-data-registry/blob/main/datasets/motional-nuscenes.yaml)
also identifies Motional's official CloudFront distribution
`d36yt3mvayqw5m.cloudfront.net` and S3 bucket `motional-nuscenes` in
`ap-northeast-1`.

Read the [data Terms of Use](https://www.nuscenes.org/terms-of-use); use the
[download portal](https://www.nuscenes.org/download) if the published mini link
needs to be refreshed. Current archive size, checksum and any access prompts
must be checked at download time. The devkit's Apache licence and this
repository's MIT licence do not grant rights to redistribute the data.

On 2026-10-21–22 this workspace's managed proxy rejected archive HEAD requests
with HTTP 403 before contacting the origin. Its enforced allowlist does not
include nuScenes/Motional download hosts. That attempt downloaded no archive
and did not establish the origin's availability. A later [local upload](local-upload.md) supplied the real images
used by the CPU experiment. Add the official hosts through the managed
environment's network settings before retrying;
editing `/etc/codex/network-policy.json` does not change the enforced policy.

The acquisition entry point can also consume an existing archive without a
network request:

```bash
python scripts/acquire_nuscenes_mini.py --archive dataset/downloads/v1.0-mini.tgz \
  --data-root dataset/nuscenes --report output/real-mini/acquisition.json
```

It preserves incomplete downloads, records byte hashes, and stages regular
metadata/keyframe camera files before publishing the data folder. Sweeps and
other sensors are omitted. An optional `--expected-sha256` checks a digest
obtained separately from a trusted source; locally computing a SHA256 does
not establish official archive authenticity. Existing nonempty data folders
are preserved. The recorded acquisition step is separate from preparation
and does not create labels.

Extract to:

```text
dataset/nuscenes/
  v1.0-mini/{scene,sample,sample_data,calibrated_sensor,sensor}.json
  samples/CAM_*/...
```

The [official split definitions](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/splits.py)
contain eight `mini_train` scenes and two `mini_val` scenes, used for debugging
and visualisation. **There is no official `mini_test`.** This project's mini
preparation creates a custom 6/2/2-scene proof-of-concept split; its test is not
an official nuScenes benchmark split. Ten scenes do not support a reliable
long-tail generalisation claim.

## 2. Prepare and label on CPU

Use the Python 3.12 CPU environment from the
[quickstart](../README.md#cpu-quickstart). First write a prerequisite inventory,
then prepare the downloaded dataset into a new output directory:

```bash
python scripts/driving_preflight.py --data-root dataset/nuscenes \
  --version v1.0-mini --model-dir models/Cosmos-Reason1-7B \
  --output output/real-mini/preflight.json
python scripts/prepare_driving.py prepare --data-root dataset/nuscenes \
  --version v1.0-mini --frames-per-scene 8 --seed 42 \
  --rubric-file latest_grading_criteria.md --output-dir output/real-mini/prepared
```

Preparation copies `rubric.md`, records its SHA256/version and embeds the
frozen rubric in the review page. It checks five metadata tables, camera
references, file existence/nonempty content and hashes; it does not decode
every image. Inspect `prepare_metadata.json`, `raw_manifest.json` and
`excluded_rows.json`. Scene splits are frozen **before labelling**. Open `output/real-mini/prepared/review.html` locally to inspect
six views and enter human decisions, then download the completed CSV. Keep a
copy as `output/real-mini/reviewed.csv`.

Label each selected six-view timestamp using the
[normal-core rubric](../latest_grading_criteria.md): `normal_core`,
`not_normal_core` or `uncertain`. Keep reviewer identity, notes and review
status. Leave uncertain cases uncertain; do not derive this binary label from
object counts, scene descriptions or official detection classes. Review a
small overlapping set independently with a second person when possible.
Compare and adjudicate disagreements before import, keeping the original
independent CSVs. Import one final decision per timestamp; multiple CSV inputs
must contain disjoint tokens.

Set `review_status=reviewed` and a reviewer name for completed decisions.
Pending/uncertain rows stay recorded but are excluded from training. Import:

```bash
python scripts/prepare_driving.py import-labels \
  --prepared-dir output/real-mini/prepared --reviews output/real-mini/reviewed.csv \
  --output-dir output/real-mini/annotations
```

The output includes `labels.json`, the original scene allocation in
`manifest.json`, exclusions and `import_metadata.json`. Check its per-split
normal/tail support and `ready_for_current_training_contract`. If a split has
one class only, do not fabricate a comparative score or choose a new split by
its test result. Preserve this PoC and plan more suitable data before a formal
experiment. This stage requires neither GPU nor VLM labels.

## 3. Obtain one suitable GPU session

NVIDIA's [Cosmos-Reason1 instructions](https://github.com/nvidia-cosmos/cosmos-reason1#inference)
list one GPU with at least 24 GB memory for their inference example. This is
not a guarantee that six high-resolution camera images fit. The
[model card](https://huggingface.co/nvidia/Cosmos-Reason1-7B) lists testing on
H100, A100 and GB200 and the NVIDIA Open Model License. Keep this project on
**Cosmos-Reason1-7B**; switching to another model changes the feature contract.

Use an existing school/lab allocation if available. If trying
[Colab](https://research.google.com/colaboratory/faq.html), its official FAQ says
GPU availability, model and usage limits vary; a free 24 GB GPU is not promised.
No GPU booking or purchase has been made. Without suitable access, continue with the separate [CPU route](cpu-route.md).

Use a separate GPU environment: choose a CUDA-compatible build with the
[PyTorch installer](https://pytorch.org/get-started/locally/), then install
`requirements-gpu.txt`. It follows [upstream lower bounds](https://github.com/nvidia-cosmos/cosmos-reason1/blob/main/pyproject.toml):
Python >=3.10, torch >=2.7.1, transformers >=4.51.3, accelerate >=1.10.1.
Neither this file nor the CPU lock is a verified GPU lockfile.

Choose an actual model commit from its
[official revision history](https://huggingface.co/nvidia/Cosmos-Reason1-7B/commits/main).
With the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli),
replace `MODEL_COMMIT` below with that commit, preview download size, then
download once to the GPU machine's persistent storage:

```bash
hf download nvidia/Cosmos-Reason1-7B --revision MODEL_COMMIT --dry-run
hf download nvidia/Cosmos-Reason1-7B --revision MODEL_COMMIT \
  --local-dir models/Cosmos-Reason1-7B
```

Follow the model's access terms. The [model file listing](https://huggingface.co/nvidia/Cosmos-Reason1-7B/tree/main)
displays about 16.6 GB; confirm actual download and free disk space with the
revision-specific dry run rather than treating that size as fixed.

## 4. Measure one frame, then cache once

On the GPU machine, copy the authorised data, reviewed preparation and pinned
model into the same relative layout, then run the existence/CUDA check:

```bash
python scripts/driving_preflight.py --data-root dataset/nuscenes \
  --version v1.0-mini --model-dir models/Cosmos-Reason1-7B \
  --output output/real-mini/gpu-preflight.json --require all
make extract PYTHON=python DATA=dataset/nuscenes NU_VERSION=v1.0-mini \
  OUTPUT=output/real-mini MODEL_REVISION=MODEL_COMMIT MAX_SAMPLES=1
```

`MODEL_COMMIT` must be the same **40-character hex commit** used for download.
Without `--require`, the inventory can exit successfully while prerequisites
are missing; `--require all` exits 2 when data/model/CUDA prerequisites are
absent. Neither mode validates model weights or guarantees sufficient VRAM.

This uses `--meta-dir`, frozen selected tokens, layer 28 and the pinned model. Its default pixel ceiling is 1.6 million per camera:
**24 GB is not a verified fit for this setting**. Inspect `extraction_summary.json`
for model-load/forward time, peak CUDA allocation/reservation, progress and
errors. If it OOMs, stop; use `scripts/extract.py --max-pixels` to try a smaller
pixel budget in a fresh output directory, then hold that setting fixed.

**Do not train on the one-frame smoke cache.** Preserve its summary/log, then
resume the remaining selection with the same settings and cache:

```bash
make extract PYTHON=python DATA=dataset/nuscenes NU_VERSION=v1.0-mini \
  OUTPUT=output/real-mini MODEL_REVISION=MODEL_COMMIT MAX_SAMPLES=0
```

Resume rejects a changed extraction contract or legacy untracked cache. For
custom smoke settings, reuse the same direct command, changing only
`--max-samples 1` to `--max-samples 0`.

Use one extraction process per cache on a filesystem supporting atomic rename
and file sync. Each save publishes a complete immutable
generation through `checkpoint_state.json`; resume verifies its hashes, shapes,
finite values and metadata before restoring the public `.npy`/JSON files.
An interruption during that export is recoverable from the committed generation.
Current and previous generations are retained to bound disk usage. Keep the
hidden `.checkpoints/` directory and state file when copying a cache for resume;
do not train from its public files while extraction is writing them.

Bring `output/real-mini/extract/` and the preparation/labels back to CPU.
Keep `meta.json`, `layer28_mlp_output_mean.npy`, `extraction_config.json` and
`extraction_summary.json` together. The config records local file hashes and
the declared download revision; it does not independently certify the Hub
origin of arbitrary local weights. Repeated baseline/SAE training reuses this
cache and needs no new VLM inference.

## 5. Return to CPU

Require `selection_complete=true` in the final extraction summary and valid
per-split class support. **Reuse `output/real-mini/annotations/manifest.json`**
from label import. Do not run `make manifest` or regenerate a 70/15/15 split:
that would replace the mini PoC's already frozen 6/2/2 allocation.

Use the handoff validator before training. It checks complete committed
features, source/sample/image alignment and the frozen manifest, and replays
label import from the hash-verified CSV evidence. Retain the unchanged review
CSV at its recorded location, beside `annotations/`, or inside `annotations/`
under its original basename when moving files. Validation does not certify
data authenticity or the truth of a person's recorded decision.

```bash
make real-validate PYTHON=python OUTPUT=output/real-mini
make real-train PYTHON=python OUTPUT=output/real-mini REVIEW_K=10 \
  SEED=42 EPOCHS=10 HIDDEN_DIM=256 TOP_K=32
```

K=10 and the short training budget are illustrative; choose them before
reading test outcomes. The driver runs Logistic, MLP and matched reward/no-reward
SAE using the imported manifest, feature config and one review budget. It
preserves commands/logs and reconstructs `comparison_runs/comparison/main_table.csv`.
The [controlled evaluation guide](evaluation.md) describes the artifacts.
Start with one seed. A mini-only table must say
**custom real-data proof of concept**; it cannot replace the historical
full-data result or establish a new benchmark.

Open the trained activation cache without the old run's glossary:

```bash
python scripts/neuron_explorer.py serve \
  --run output/real-mini/comparison_runs/sae_abstopk_norm \
  --samples-root dataset/nuscenes/samples
```

Inspect FP/FN images and export the review CSV. Retain model/data versions,
exclusions, human review records and all predictions. The [synthetic end-to-end
checks](e2e.md) verify the software path. The original VLM checkpoint and a
human-validated scene-disjoint VLM result remain to be produced.

Early preparation/import checks used a watermarked 80-frame schema fixture.
The later local upload, ResNet18 extraction and CPU fits used real camera
images; their evidence is separate from those fixture checks. The first
[CPU AI-reference pilot](ai-pilot.md) was followed by the explicitly
[post-disclosure human-reference fit](human-review.md). Neither is a Cosmos
replication or independent blind-label evaluation.
