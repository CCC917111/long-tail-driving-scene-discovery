# Feedback implementation status

Updated 2026-10-23. Original audited version: `b57eaffa21d54ad67d98d441935cf6b17cdc5294`.
This file distinguishes software work from experimental evidence. No new
human-validated real-data effectiveness result has been established in this round.

## Current round

| Area | Implemented / prepared | Still required |
|---|---|---|
| P0 data contracts | Strict scene-token and label validation; frozen manifest interface; input/run provenance | Recover historical assets, review manifest, explain six-frame gap and raw-VLM supports |
| Fresh real-data setup | Four real mini parts imported; five tables / 480 camera images checked; 80 CPU ResNet18 feature vectors extracted on a frozen 6/2/2-scene split | Human decisions and label-support checks; original VLM/GPU extraction remains separate |
| Explicit AI-label pilot | 80 six-view AI decisions; 73 retained / seven uncertain; four real-image model fits, reload/result audit and unlabelled screening; [record](ai-pilot.md) | Independent human truth, more normal training scenes and formal multi-seed evidence; no observed reward increment |
| Portable execution | Embedded-source notebook; staged archive acquisition; saved review/backup; strict four-method CPU driver | Real download/GPU execution and Colab visual verification |
| P1 comparison software | Shared evaluator; Logistic/MLP and reward/no-reward SAE paths; validation operating point, review exports and table rebuilding | Real-data smoke, four matched runs, core multi-seed results, real FP/FN review |
| P2 demonstration | Existing neuron explorer; four-model synthetic workflow, reload/screening checks and real HTTP routes validated | Authorised real cached examples, trained real weights, 90-second recording with a mistake and review export |
| Documentation | Historical results isolated; contribution wording retained; current AI-assisted refactor identified | Original contribution evidence, real result bundle |

The original README contribution table records Yike's stated capstone work.
The 2026-10-19–22 changes to validation, provenance, baseline evaluation, fixtures
and documentation are AI-assisted follow-up work. They must be reviewed and
understood by Yike before being presented as current engineering work; they
must not be retroactively attributed to the original capstone period. No
unsupported statement about earlier AI use is made.

## Verification record

The offline CPU selection passed 122 tests and 18 subtests; one real-asset test
was skipped and one HTTP test deselected. The HTTP unit test passed separately
with loopback access. The trained synthetic explorer passed desktop/mobile
Chromium navigation, image loading, search and CSV-download checks; the screened
explorer's Chromium check remains pending. A combined full-suite pass is not
claimed. Lint passed. CI configuration has not run remotely.

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
No VLM inference, original-checkpoint evaluation or human-label audit has run in
this round. A subsequent explicitly AI-labelled pilot produced supervised scores
against provisional AI references. It remains separate from human-validated
effectiveness results; the [pilot record](ai-pilot.md) includes its failures.

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

## Real-data completion gate

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
