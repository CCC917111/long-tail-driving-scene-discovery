# New real scenes: CPU inference to human review

The [hosted review tool](https://yike-driving-review-20261028.upbeatperch12.chatgpt.site)
lets the owner inspect nine new six-camera timestamps and export decisions.

The private Site serves all 54 verified original images from the same
origin. No HTML download or external raw-GitHub image access is required.
Desktop/mobile real-browser checks, CSV export and failed-image retry passed.
Those checks used the same-origin bundle on a temporary local server; native
publishing status separately verifies deployment. An authenticated production
browser session was not recorded.
Sign in with the owner's ChatGPT account. External GitHub readers do not
automatically have access. The
[repair evidence](../results/review-page-repair-20261028/README.md) records the
same-origin image delivery, browser checks and deployment. A public source-linked
HTML fallback is described separately below.

The original 2026-10-27 `new-scene-real-cpu-e2e-v1` run connected **54 original JPEGs →
9 six-camera feature vectors → the saved SAE → review HTML → CSV export**.
It used a CPU and an existing trained checkpoint, with no new fitting or
accuracy evaluation at that time. The nine image groups were then unlabelled.
Yike's later [2026-10-28 submission](../results/new-scene-human-review-20261028/README.md)
contains five normal and four non-normal judgments. The AI reference table was
disclosed in chat before submission; per-row decision timing is unknown. None
of these nine rows has been used to retrain the existing models.

## Open the review page

1. Open the [owner-private Site](https://yike-driving-review-20261028.upbeatperch12.chatgpt.site)
   and sign in with the owner's ChatGPT account. No HTML download or ZIP upload
   is needed; images load from the Site itself.
2. Wait until the current group's six views load. The three judgment buttons
   remain disabled while images are missing. The image-retry button retries loading while
   preserving the current unsubmitted name and notes.
3. Inspect the six views. Enter your name, add a short reason, then choose
   `normal_core`, `not_normal_core` or `uncertain`. The next group opens after
   a decision; previous groups remain editable.
4. Download `reviewed-new-scenes.csv` to preserve your work. Browser storage is
   a convenience, so export before changing device/browser or clearing storage.

Scores and numeric SAE units stay hidden initially. The model-signal control reveals
them and records that subsequent judgments followed model-signal exposure.
The source HTML contains the cached model values, so this convenience is not
proof of an independent blind study. The CSV is a fresh review record, not a
new benchmark table. The page itself displays no source scene descriptions,
DriveLM answers or AI reference labels. The separate chat AI-label disclosure
is recorded with the later submission even when browser model signals stayed
hidden. A fresh browser context starts with nine pending rows;
automated UI tests use a disposable `CI_TEST_ONLY` reviewer, never real labels.

## Downloadable and local alternatives

The public [driving_new_scenes.html](../demo/driving_new_scenes.html) is a
source-linked fallback. Download it with GitHub's **Download raw file** arrow
and open the saved file in Chrome/Edge. It requires Internet access to
`raw.githubusercontent.com` to load the author's original JPEGs. That external
image dependency failed for the project owner and motivated the same-origin
hosted delivery above; the fallback is not the hosted Site.

The private local export `output/new-scene-e2e-final-20261027/review.html`
uses sibling `images/CAM_*/...jpg` files and works without Internet after the
complete folder is copied. Keep that folder together. The public HTML embeds
numeric outputs and author image URLs, not pixels or model weights.

## What actually ran

| Stage | Recorded output |
|---|---|
| Source | Official [OpenDriveLab/DriveLM](https://github.com/OpenDriveLab/DriveLM) author demo, commit `1de72a74b257e5373400fa68239e99bd5d20580a` |
| Input validation | 54 JPEGs, 6,114,069 bytes; original Git blob hashes and SHA256 checked, every image decoded |
| Cohort | Nine timestamps in two scenes; no sample or scene overlap with the original 80-frame/10-scene cohort |
| Encoder | Pinned ImageNet ResNet18, the same resize/crop/normalisation and camera order as training |
| Feature cache | Finite float32 `(9, 3072)` array; SHA256 `65c452c559b574034e1534afc03f955de37937c8bc468da85f55a95429ac7d7e` |
| Inference | Existing `sae_abstopk_none`; original checkpoint and train-only standardiser reused unchanged |
| Fixed rule | At least one tail unit with `abs(z) > 0.01`; all nine frames were flagged |
| Independent replay | Re-encoding original JPEGs and reloading the saved SAE reproduce every feature, activation and score exactly |
| Delivery | Six-camera review page, numeric units, editable decisions, exposure context and CSV export |

The checkpoint SHA256 is
`250a475bf430ade3a9a4381430a1b4cee366a01a2448747602e2aff76baabbcb`.
Its training labels came from the earlier single-reviewer submission after
AI-label disclosure. That is the **model's training provenance**. At the
2026-10-27 inference run these nine frames had no human reference labels; the
later supplemental submission is retained separately with its own provenance.

All nine fixed-rule flags are model outputs, not nine verified tail examples.
The batch was selected by the upstream demo authors, spans only two scenes,
and lacks original log metadata. It is a data-ingestion and inference smoke
test, not the proposed representative research expansion. No log-disjointness,
balanced training improvement or statistical rarity is claimed.
nuScenes images retain their original dataset terms; code licences do not
replace those terms. Raw images and weights remain outside this public
repository's Git history; the same-origin image copies belong to the separate
owner-private Site.

## Descriptive frozen-policy replay

The [saved replay](../results/new-scene-human-review-20261028/frozen_policy_evaluation.json)
uses the old validation-selected operating threshold,
`6.074954873387325`, without adjusting it to the nine new judgments. It selects
five candidates (three reference positives,
two reference negatives): precision 60%, recall 75%. The fixed activation rule
instead selects all nine: precision 44.4%, recall 100%. These are two different
policies on unchanged model scores, not a model improvement from retraining.

All five normal references come from one scene; all four non-normal references
come from the other. This class/scene confound, author selection and prior
AI-reference disclosure prevent interpreting the numbers as an independent
generalization benchmark.

The original run files and hashes are preserved. Its
[execution source snapshot](../results/new-scene-e2e-20261027/execution_source.zip)
retains the exporter bytes recorded at execution time; current source includes
the later image-loading guard and retry repair. The
[later browser/deployment evidence](../results/review-page-repair-20261028/README.md)
documents that UI repair without rewriting the earlier result files.

## Replay with the real assets

Use the pinned CPU environment from the main quickstart. With the original
trained run and its original feature metadata available:

```bash
python scripts/acquire_cpu_encoder.py
python scripts/new_scene_e2e.py \
  --acquisition results/new-scene-e2e-20261027/source_manifest.json \
  --data-root dataset/drivelm-author-demo-20261027 \
  --weights models/resnet18/resnet18-f37072fd.pth \
  --run-dir output/real-cpu-mini-20261023/human-ai-exposed-20261026/comparison_runs/sae_abstopk_none \
  --output-dir output/new-scene-replay \
  --download
```

`--download` fetches only missing images from the pinned author source and
checks each SHA256 before extraction. A cached run needs no network access.
Use a fresh output directory. If the original cohort metadata moved, pass
`--training-meta PATH`; its hash must match the original training record.
The trained SAE and training metadata are private assets, not bundled into
the public HTML. A clean clone can inspect the cached source-linked fallback;
reproducing inference additionally requires those recorded model assets.

The adapter does not manufacture nuScenes metadata tables, scene names,
labels or train/test assignments. Its timestamp comes from the front-camera
filename and is explicitly marked as such, rather than an official sample
timestamp. It validates the six original views and calls the existing CPU
encoder, screening CLI and neuron explorer.

Browser verification (Playwright and Chromium are separate dependencies):

```bash
python scripts/verify_new_scene_review.py \
  --html output/new-scene-replay/review.html \
  --output-dir output/new-scene-review-browser \
  --browser-executable /usr/bin/chromium
```

The verifier starts and closes a temporary localhost server. It checks all
54 original views in desktop/mobile layouts, initial blank CSV rows, reviewer
validation, notes, decisions, signal exposure and saved-state restoration.
Simulated test decisions never enter training or evaluation.

[Source manifest and execution evidence](../results/new-scene-e2e-20261027/README.md)
· [Earlier training and evaluation](../results/human-ai-exposed-20261026/README.md)
