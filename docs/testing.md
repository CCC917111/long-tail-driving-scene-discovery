# Testing and verification

## New-scene review and image-delivery repair: 2026-10-28

The [owner-private review Site](https://yike-driving-review-20261028.upbeatperch12.chatgpt.site)
requires the owner's ChatGPT sign-in and serves all 54 original images from the
same origin. Local desktop/mobile Chromium checks of that bundle exercised all nine six-camera
groups, initial blank review CSVs, reviewer validation, notes, signal disclosure,
refresh persistence and CSV downloads. A separate failed-image test verified
that missing views disable judgment buttons and that retry restores the images
while retaining unsubmitted name and notes. These tests use temporary
`CI_TEST_ONLY` judgments and create no human training labels. The
[repair evidence](../results/review-page-repair-20261028/README.md) preserves
these checks and the successful private deployment separately from the original
2026-10-27 run. The deployed archive matches the checked bundle; native publishing
status verifies deployment. An authenticated production browser session was not
recorded.

The original [new-scene inference audit](../results/new-scene-e2e-20261027/independent_audit.json)
freshly re-encoded the 54 JPEGs and reloaded the saved SAE; features, activations
and scores matched exactly. Its recorded source hashes describe execution time.
The [execution snapshot](../results/new-scene-e2e-20261027/execution_source.zip)
preserves the older exporter before the subsequent image-loading repair; original
run metadata is not rewritten to match newer source.

Yike's later [nine-row human submission](../results/new-scene-human-review-20261028/README.md)
is kept separately with chat AI-reference disclosure provenance. Browser
`model_signal_exposed=false` records only browser signal visibility; it does
not establish blindness to the chat AI table. The
[frozen-policy replay](../results/new-scene-human-review-20261028/frozen_policy_evaluation.json)
applies the old validation-selected threshold without tuning on those nine
labels: five candidates, precision 60%, recall 75%. Both classes are perfectly
confounded with the two source scenes, so this is descriptive agreement rather
than independent generalization evidence.

The new source-linked HTML browser workflow depends on the original author's
public image URLs and tests that fallback's UI. It does not authenticate into
or certify the owner-private hosted Site. New source/test files cover six-view
acquisition integrity and loading/failure/retry state; publication checks report
the CI status of the actual submitted commit.

## Earlier fitted pilot and cached viewers

The latest offline viewer now binds the reviewed-reference run's 77 timestamps
and all 308 method values. Independent Node checks loaded the actual four ZIPs,
compared all 480 image files byte-for-byte with the originals, and checked all
cached values, blank review fields and script syntax. The old AI viewer is
archived byte-identically with its 73-frame/292-value record. Separate synthetic
core checks pass for both. The updated browser CI runs both HTML versions at
desktop/mobile sizes using synthetic images and checks case/metric tables,
navigation and CSV downloads. The latest publication PR checks establish the
current Chrome result; core-only checks do not claim browser verification.

The 2026-10-26 [post-AI-exposure user-review run](../results/human-ai-exposed-20261026/README.md)
completed all four real CPU fits and an independent checkpoint/scaler/metric
audit. A local full non-HTTP selection passed **225 tests and 18 subtests**,
including the configured real-asset integration; one socket-dependent HTTP case
was deselected. That record predates the final nested-scope guard hardening.
After hardening, 24 context tests plus 12 evaluation/AI-scope tests and two
subtests passed, with lint/compile clean. The publication PR's GitHub Actions
checks cover the final source and generated-data HTTP workflow. These checks
preserve single-reviewer, post-AI-exposure scope; they do not authenticate
judgment correctness or establish independent blind human truth.

The default CPU suite uses synthetic arrays and images to validate software
contracts. A separately configured integration test and independent audits also
ran against the real mini CPU pilot. The original pilot's labels remain
provisional AI decisions; the later fitted comparison uses single-reviewer
judgments after AI-label disclosure. None of these checks establishes independent
human-validated driving accuracy or reproduces the original Cosmos experiment.

## Earlier local record: 2026-10-23

On 2026-10-23, in the verified Python 3.12 CPU environment:

```bash
DRIVING_REAL_ASSET_DIR=/path/to/real_bundle \
  python -m pytest tests -q -k 'not http_interface'
python -m flake8 scripts tests --max-line-length 100
```

The final local selection passed **205 tests and 18 subtests**, with **one HTTP
test deselected** because this sandbox denied loopback binding. The optional
real-asset test actually ran against the real feature bundle and explicitly
AI-provisional labels; it was not skipped in this local record. Lint passed.
This is a non-HTTP local pass, not a claim that the sandbox completed the HTTP
test. Resolved dependencies are in
[`requirements-cpu-lock.txt`](../requirements-cpu-lock.txt).

## GitHub Actions

At commit `ba8efddd46b357f492965b941172a8fd1f6dd948`, both workflows passed:

- [CPU contracts and synthetic demo](https://github.com/CCC917111/long-tail-driving-scene-discovery/actions/runs/40039920171):
  `make test lint` and `make e2e`, including actual temporary HTTP servers.
- [Offline explorer browser smoke](https://github.com/CCC917111/long-tail-driving-scene-discovery/actions/runs/40039920157):
  Chrome 155 at desktop and mobile viewports, six loaded images, sample/unit
  navigation, exact-token search, cached model values and an actual review CSV
  download, with no JavaScript errors.

The earlier [CPU run](https://github.com/CCC917111/long-tail-driving-scene-discovery/actions/runs/40039177713)
records **205 passed, one real-asset test skipped and 18 subtests passed**.
CI has no private driving assets, so its real-asset skip is expected and distinct
from the configured local integration pass.

The browser workflow uses visibly watermarked, generated images and synthetic
metadata in four ZIPs. It preserves the shipped demo's runtime/UI scripts and
tests their rendering and interaction. It does not upload private nuScenes
images, run real-image inference or validate labels. The workflow retains its
verification JSON and desktop/mobile screenshots as temporary CI artifacts.

## Real mini and independent offline checks

The [CPU image-feature route](cpu-route.md) imported ten real scenes / 80
timestamps / 480 original JPEGs, checked the images and metadata, and extracted
finite `(80, 3072)` pretrained ResNet18 features. The subsequent
[AI-reference pilot](ai-pilot.md) retained 73 labelled timestamps and completed
four model fits, checkpoint reloads, train-only normalization checks, saved-score
evaluation and table reconstruction. Seven uncertain timestamps were screened
separately; their scenes can overlap the labelled scenes.

Independent Node checks used the actual four user-held ZIPs, verified all 480
image bytes and matched all 292 displayed model values to saved predictions.
The exported review CSV retains blank human-decision fields. These real-byte
and numeric checks are separate from the synthetic Chrome interaction check;
a Chrome session with the user's actual images has not been verified here.
The [derived result bundle](../results/cpu-mini-ai-pilot-20261023/README.md)
preserves the audits and explicit AI-label scope. It includes no original image
bytes or trained model weights.

Wrong encoder, weights and preprocessing identities are rejected before
screening model loading. Legacy screening records unverified feature identity
explicitly. AI labels require their own importer and an explicit pilot protocol;
the default manual route rejects an AI bundle.

## Synthetic end-to-end checks

The synthetic end-to-end workflow also passed: four trained models on 60
scenes / 240 frames, matching checkpoint reloads and train-only standardisers,
reconstructed comparison table, and screening of 40 independent unlabelled
samples. Actual temporary HTTP servers passed **14 route checks each** for the
trained and screened explorer (**28 total**), including six JPEGs per sample,
search, review CSV and an expected unknown-token 404. Effective activation
thresholds were 0.01 for training and 0.05 after screening override/reload.

The trained synthetic explorer passed Chromium click checks at desktop and
mobile viewports: unit/sample navigation, six loaded images, exact-token search
and downloaded review CSV. Its report is
`output/e2e-20261020-final/ui-trained/ui_check.json`. The screened explorer has
HTTP verification but its Chromium check has not completed. The
[one-command workflow](e2e.md) describes commands and evidence files. All images,
features and labels in this workflow are generated; these checks do not measure
real driving performance.

## What the CPU suite checks

| File | Contracts |
|---|---|
| `tests/test_sae_common.py` | AbsTopK/sign behavior, loss terms, scene splits, label whitelist, token conflicts and strict input validation |
| `tests/test_neuron_explorer.py` | Fixed activation rule, activation-store round trip, sample/unit queries, image paths and HTTP route integration |
| `tests/test_evaluation.py` | Frozen manifests, prediction alignment, operating points, review budget, scene resampling, metrics and result/provenance integrity |
| `tests/test_screening.py` | Reloaded screening thresholds, duplicate/nonfinite inputs and standardiser validation |
| `tests/test_prepare_driving.py` | Five-table preparation, six-camera selection, frozen scenes, rubric hashes and strict manual-review CSV import |
| `tests/test_extraction_contract.py` | Extraction hooks/pooling/resume with a fixture model, immutable settings, legacy-cache rejection, load failure reports and prerequisite checks |
| `tests/test_extract_checkpoints.py` | Atomic checkpoint publication, interrupted-export recovery, corruption rejection and bounded generation retention |
| `tests/test_acquire_mini.py` | Transfer completeness/resume/checksums, safe archive staging and preserving existing data |
| `tests/test_cpu_extraction.py`, `tests/test_acquire_cpu_encoder.py` | CPU encoder layout, six-view order, pinned-weight enforcement, image tampering and checkpoint resume |
| `tests/test_notebook_review.py` | Saved review decisions, partial CSV resume, per-row reviewer identity and actual label import |
| `tests/test_real_driving_workflow.py` | Complete real-input handoff gates, immutable input evidence and four actual CPU estimators on a fixture |
| `tests/test_portable_notebook.py`, `tests/test_portable_handoff.py` | Notebook syntax/bootstrap/local preparation and review backup/restore without downloads/GPU |
| `tests/test_import_mini_parts.py`, `tests/test_local_upload_html.py` | ZIP completeness, checksums, safe paths and local browser handoff contracts |
| `tests/test_ai_pilot_labels.py`, `tests/test_ai_label_scope.py` | Split-blind AI-label evidence, uncertainty exclusion, explicit protocol opt-in and scope propagation |
| `tests/test_offline_explorer.py` | Cached result integrity, bound local-image loading and review exports without invented semantic unit names |
| `tests/test_new_scene_e2e.py` | Original camera hashes/decoding, six-view completeness, safe paths and rejection of labels/splits in new intake |
| `tests/test_unlabelled_review_export.py` | Unlabelled review scope, source-image binding, CSV/exposure state, loading guard, stale callbacks and retry recovery |
| `tests/test_real_assets.py` | Optional integration against a supplied real feature/label/scene bundle; explicit skip when absent and no label-quality claim |

Run `python -m pytest tests -q` with loopback access to include the HTTP
route, as the CPU workflow does. A local non-HTTP pass does not establish a
local combined full-suite pass. All checks require their installed dependencies; the
standalone evaluator checks can also run with
`python tests/test_evaluation.py -q` as described in [evaluation](evaluation.md).

The raw-data setup CLI also consumed a retained synthetic nuScenes-schema
fixture: 10 scenes, 80 frames and 480 watermarked images, with a frozen 6/2/2
scene allocation. Generated test labels imported successfully and both
`load_dataset` and `load_frozen_manifest` consumed the resulting files.
`output/from_zero_fixture/consumption_check.json` records this check. This
does not establish real nuScenes decoding, human label quality or VLM inference;
follow [the real-data setup guide](from-zero.md) for those steps.

The portable notebook's embedded-source bootstrap and local fixture preparation
were executed. Review save/reload/import and backup/restore passed fixture tests.
The new runner also completed four actual CPU estimator runs and table rebuilding
on a tiny synthetic handoff. Notebook GPU/download cells and visual widget behavior
in Colab remain unverified. The 2026-10-22 live acquisition attempt failed at the
managed proxy with HTTP 403 and produced no real archive or labels in that
attempt. The later local upload succeeded, as recorded above.

## Remaining real-asset work

The current CPU pilot does not complete the following checks:

1. Obtain independent human labels and review/exclusion records, with enough
   normal and tail scenes to support a meaningful held-out evaluation.
2. Run original VLM extraction with recorded model revision, layer, hook,
   pooling and prompt versions. The completed image-only ResNet18 extraction
   uses a separate feature protocol.
3. Recover historical weights, features, labels and logs, then compare original
   predictions without rewriting their provenance.
4. Collect independent human decisions with appropriate scene support. Actual
   browser viewing is verified for the supplemental 54-image batch, but those
   submitted labels followed chat AI-reference disclosure and do not establish
   independent blind human truth. The older viewers retain their separately
   documented browser-verification scope.

The optional cached-feature integration entry point is:

```bash
DRIVING_REAL_ASSET_DIR=/path/to/real_bundle python -m pytest tests/test_real_assets.py -q
```

The directory must contain `features.npy`, `meta.json`, `labels.json` and
`manifest.json`. Without `DRIVING_REAL_ASSET_DIR`, the test explicitly skips;
synthetic substitutes do not count as real-data integration. This cached-data
check does not run VLM extraction, verify the original trained checkpoint or
turn provisional AI labels into human ground truth.

## New checks

Test failures that could silently change data meaning or a conclusion:
missing scenes, duplicate/conflicting tokens, unrecognised labels, overlapping
splits, nonfinite features, changed token-to-feature alignment, test-dependent
selection, altered predictions or mismatched metadata. Keep tests small enough
for CPU CI. Semantic names and human judgement need evidence records beyond
unit assertions. Refer to [feedback status](feedback-status.md) for the real-data
completion gate and current limits.
