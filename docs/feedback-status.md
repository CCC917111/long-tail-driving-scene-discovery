# Feedback implementation status

Updated 2026-10-28. Original audited version: `b57eaffa21d54ad67d98d441935cf6b17cdc5294`.
This file distinguishes completed software and descriptive review checks from
independent human-reference benchmarks. Stable model-effectiveness and independent
blind-label claims remain unestablished.

The [hosted review tool](https://yike-driving-review-20261028.upbeatperch12.chatgpt.site)
is owner-private and requires the owner's ChatGPT sign-in. It serves all 54
new images from the same origin. The [downloadable fallback](../demo/driving_new_scenes.html)
still requires access to the author's image URLs on `raw.githubusercontent.com`.

## Current round

| Area | Implemented / prepared | Still required |
|---|---|---|
| P0 data contracts | Strict scene-token and label validation; frozen manifest interface; input/run provenance | Recover historical assets, review manifest, explain six-frame gap and raw-VLM supports |
| Fresh real-data setup | Four real mini parts imported; five tables / 480 camera images checked; 80 CPU ResNet18 feature vectors extracted on a frozen 6/2/2-scene split | Independent human decisions and adequate normal/tail scene support; original VLM/GPU extraction remains separate |
| Post-AI-exposure human-review CPU run | Yike supplied 80 judgments: 77 reviewed binary labels (12 normal / 65 tail), three uncertain, no pending; all four CPU methods fitted and independently reconstructed; [record](../results/human-ai-exposed-20261026/README.md) | Single reviewer after the full AI label table, empty notes, one seed, two test scenes and three normal training frames; no observed reward increment |
| Supplemental new-scene check | Nine real timestamps / 54 images from two additional scenes; new-input CPU inference and exact independent replay; Yike completed five normal / four non-normal references; [record](../results/new-scene-human-review-20261028/README.md) | Chat AI-reference disclosure before submission; per-row timing unknown; empty notes; all normals in one scene and all non-normals in another; no new model fit |
| Frozen working-point replay | Old validation-selected threshold selects five of nine, precision 60% and recall 75%, versus the fixed activation rule selecting all nine; [saved replay](../results/new-scene-human-review-20261028/frozen_policy_evaluation.json) | Descriptive comparison with submitted references; no threshold adjustment on new labels and no independent generalization claim |
| Explicit AI-label pilot | 80 six-view AI decisions; 73 retained / seven uncertain; four real-image model fits, reload/result audit and unlabelled screening; [record](ai-pilot.md) | Independent human truth, more normal training scenes and formal multi-seed evidence; no observed reward increment |
| Portable execution | Embedded-source notebook; staged archive acquisition; saved review/backup; strict four-method CPU driver | Real download/GPU execution and Colab visual verification |
| P1 comparison | Four matched real-image CPU fits against explicit AI references, then a separately recorded post-disclosure human-reference fit; shared evaluator and saved-output reconstruction | Independent human-reference evaluation and formal multi-seed evidence; all validation targets were unreachable in the original AI pilot; no observed reward increment |
| P2 demonstration | Latest reviewed-reference offline explorer with 77 cached timestamps / 308 method values, same-rule MLP FP/FN bookmarks and blank new-review export; original 73-frame AI viewer archived byte-for-byte; 90-second script prepared | A narrated real-image recording, independent human error review and verified semantic interpretation; current Chrome status is recorded in publication PR checks |
| New review delivery | Owner-private Site with same-origin original images; local Chromium desktop/mobile bundle checks, CSV export, loading guards, failed-image retry and successful native publication; [repair evidence](../results/review-page-repair-20261028/README.md) | Owner ChatGPT sign-in is required; authenticated production browsing was not recorded; external GitHub reviewers have no automatic access |
| Documentation | Historical results isolated; original contribution wording retained; AI-assisted follow-up identified; [derived pilot bundle](../results/cpu-mini-ai-pilot-20261023/README.md) published | Original contribution evidence and human-validated result claims |

The original README contribution table records Yike's stated capstone work.
Yike is the project lead and is responsible for the GitHub repository's overall
optimization and delivery, as confirmed in the project discussion. This role
covers the modernization phase while preserving the original team-role table
and the documented AI-assisted implementation history.
The 2026-10-19–23 changes to validation, provenance, baseline evaluation, data
handoff, the offline explorer, fixtures and documentation are AI-assisted
follow-up work. They must be reviewed and
understood by Yike before being presented as current engineering work; they
must not be retroactively attributed to the original capstone period. No
unsupported statement about earlier AI use is made.

## Verification record

The 2026-10-27 new-scene execution has a separate original
[source snapshot](../results/new-scene-e2e-20261027/execution_source.zip).
The later exporter loading/retry repair and private deployment are recorded in
[2026-10-28 evidence](../results/review-page-repair-20261028/README.md), rather
than replacing the old run's hashes. All 54 real original images were decoded
in desktop/mobile review checks. The temporary test decisions are not human
annotations. Source-linked fallback CI and the owner-private Site have distinct
verification scopes.

The following earlier local and CI records retain their original dates/scopes:

The final local CPU selection passed **205 tests and 18 subtests**, with one
HTTP test deselected because the sandbox denied loopback binding. The optional
real-asset integration actually ran against the real feature bundle and
explicitly AI-provisional labels. Lint passed. This is a local non-HTTP pass;
the HTTP-dependent case was not counted as passed locally.

At `ba8efddd46b357f492965b941172a8fd1f6dd948`, GitHub Actions passed
[CPU tests/lint and the HTTP end-to-end workflow](https://github.com/CCC917111/long-tail-driving-scene-discovery/actions/runs/40039920171)
and the [offline Chrome browser workflow](https://github.com/CCC917111/long-tail-driving-scene-discovery/actions/runs/40039920157).
The earlier [CPU run](https://github.com/CCC917111/long-tail-driving-scene-discovery/actions/runs/40039177713)
records 205 tests and 18 subtests passed, with the private-real-asset test
explicitly skipped in CI.

Chrome 155 passed desktop/mobile navigation, six-image rendering, cached score
checks and actual CSV download using generated, watermarked images in synthetic
ZIPs. Independent Node checks separately verified all 480 actual local image
bytes and all 292 displayed model values against saved outputs. The browser
check is UI evidence, not a real-image browser run or label-quality measurement.
See [testing](testing.md) for the scope of each check.

The synthetic end-to-end run trained all four core models on 60 scenes / 240
frames, reloaded their checkpoints and train-only standardisers, rebuilt the
main table and screened 40 independent unlabelled samples. Both real HTTP
explorers passed 14 route checks each (28 total), including image responses
and `/export/review.csv` without reference labels. The screening override
`eta=0.05` survives reload, distinct from the trained run's `eta=0.01`.

This round also repaired screening threshold persistence and validation of
duplicate tokens, nonfinite values and standardiser inputs. These are software
fixes, not new real-data model results. See [end-to-end verification](e2e.md)
for commands and artifacts. CPU dependency versions are in
`requirements-cpu-lock.txt`; GPU/VLM lower bounds are in
`requirements-gpu.txt` and have not been verified on a CUDA machine.

The new raw-data preparation and label-import CLI passed against an explicitly
synthetic 80-frame, 10-scene fixture, including consumption by the shared data
loader and frozen-manifest loader. Extraction hooks and resume were exercised
with a fixture model. The earlier preflight reported no real dataset, local VLM
weights or CUDA device. A real mini subset and pinned ResNet18 weights are now
available; local Cosmos weights and CUDA remain absent.

The 2026-10-23 real CPU run imported 80 timestamps / ten scenes / 480 JPEGs,
froze an internal mini allocation of 48/16/16 frames, and completed extraction
into an `(80, 3072)` cache after a one-frame smoke and verified resume. It created
no labels. Raw logs and hashes are retained under
`output/real-cpu-mini-20261023/`; [CPU route](cpu-route.md) records the scope.
No VLM inference or original-checkpoint evaluation has run in
this round. A subsequent explicitly AI-labelled pilot trained and reloaded all
four methods on 73 retained timestamps and screened seven uncertain timestamps.
Its complete result bundle, source snapshot, hashes and numeric offline explorer
are published without original images, raw metadata or weights. The user loads
their own images locally. Scores remain agreement with provisional AI references:
two test scenes, only three normal training examples, one seed, identical SAE
reward/no-reward checkpoints and unreachable validation precision targets limit
the conclusions. The [pilot record](ai-pilot.md) includes these failures.

On 2026-10-26, the final human CSV was received and checked against all 80
frozen sample IDs. Its SHA-256 is
`c39de8776325f6d72ee22738dcf2c95ee653db439f416de0a4e9403ea8f8a43a`.
Every row names Yike; the final submission contains 12 `normal_core`, 65
`not_normal_core` and three `uncertain` judgments. All binary rows are
`reviewed`, all uncertain rows have status `uncertain`, and none remain pending.
All review notes are blank, including the three uncertainty reasons; no reasons
have been inferred or added. The earlier first-ten submission is retained as a
pre-exposure checkpoint, together with the four later revisions to those labels.
The completed submission followed display of the full AI label table, so it is
post-AI-exposure single-reviewer evidence, not blind independent ground truth.
The frozen split is unchanged: train has 3 normal / 45 tail; validation has
4 normal / 10 tail plus two uncertain; test has 5 normal / 10 tail plus one
uncertain. The 77 binary rows meet the current class-support import contract;
the three uncertain rows remain excluded from supervised training/evaluation.
The subsequent controlled CPU rerun fitted Logistic, MLP and both SAE variants
with the retained 77 rows, original ResNet18 cache and unchanged training
settings. Independent reconstruction verified the inputs/source hashes,
train-only scalers, all four checkpoint scores, validation-only thresholds,
saved metrics and comparison table. On the 15-row test split, SAE AP is 0.9353,
precision@4 is 1.0 and recall@4 is 0.4; its validation-selected operating point
has test precision 1.0 and recall 0.2. Both SAE checkpoints are byte-identical
to each other and to the earlier AI pilot, so no reward improvement was
observed. Logistic remains byte-identical; MLP's validation-selected checkpoint
moved from epoch 4 to 5 with the same training settings and loss trajectory.
These scores measure agreement with the post-AI-exposure judgments. The
training source snapshots are retained; a later provenance guard in the current
evaluator does not rewrite those snapshots or the saved run evidence.
See the [completed record](../results/human-ai-exposed-20261026/README.md) and
[review workflow](human-review.md) for the full comparison and limits.

## Historical claims needing recovery

- Recover labels and review records, sample/scene tokens, extraction settings,
  feature arrays, checkpoint, standardiser, original commands and old logs.
- Reconcile raw-VLM predictions with the stated test supports before making a
  matched-test comparison; preserve each original file rather than rewriting it.
- Explain why 824 scenes at eight keyframes would yield 6,592 samples while the
  historical report lists 6,586. Record excluded tokens and reasons.
- Keep historical random-frame 80/20 results separate from current scene-level
  70/15/15 results. Do not infer old provenance from the current implementation.
- Verify contribution records with the original report, commits and team records.
  The annotation platform and most feature interpretation belong to the named
  teammates in the existing contribution table.

## Human-validated experiment completion gate

The CPU software pilot has completed the shared-input fits, saved-output
reconstruction and an inspection tool. The remaining gate concerns research
claims and independent human evidence:

1. Validate and freeze an authorised manifest with no duplicate tokens,
   conflicting labels or scene overlap, with provenance/exclusion records.
2. Use the same feature/metadata hashes and manifest for Logistic, MLP,
   AbsTopK without reward and AbsTopK with reward. Change only reward between
   the two SAE core configurations. Freeze the review budget in advance.
3. Run a small smoke before estimating training/download/annotation costs.
   Begin with one seed; run core groups with three seeds within the agreed
   resource budget. Report mean and spread separately from scene intervals.
4. Rebuild the main table from saved predictions and retain config, source,
   seed, logs, checkpoint hash, feature settings and input hashes for every row.
5. Review real high-confidence errors and disagreements with independent human
   decisions. Name units using train/validation, freeze them, then measure test
   purity/coverage. VLM agreement is not human-ground-truth accuracy.
6. Show real FP/FN examples and one explanation in the explorer; export the
   candidate review list. Document data/weight redistribution conditions.

A valid outcome may show the SAE loses to a cheaper baseline. End the round
when claims can be traced, a small example runs, failures are explainable and
individual responsibilities are evidenced. Estimate download, annotation and
extraction costs from the smoke run; obtain the user's budget decision before
buying compute or other paid resources.
