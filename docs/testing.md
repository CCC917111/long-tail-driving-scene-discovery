# Testing and verification

CPU checks use synthetic arrays and images. They validate software contracts,
not nuScenes label quality, original VLM checkpoints or real
model effectiveness. Real-asset checks require a separate configured run.

## Latest local record

On 2026-10-23, in the verified Python 3.12 CPU environment:

```bash
python -m pytest tests -q -k 'not http_interface'
python -m flake8 scripts tests --max-line-length 100
```

The offline test selection passed **150 tests and 18 subtests**, with **1
real-asset test skipped** because no input directory is configured and **1
HTTP test deselected**. The HTTP unit test subsequently passed on its own with
loopback access during the earlier end-to-end verification. The final combined
full-suite run is pending; these separate checks are not a recorded combined
full-suite pass. Lint passed. CPU CI is
configured but has not run remotely. Resolved CPU dependencies are in
[`requirements-cpu-lock.txt`](../requirements-cpu-lock.txt).

After adding the CPU screening feature-identity guard, its targeted selection
passed **14 tests**. That later guard change was checked separately from the
150-test record above. Wrong encoder/weights/preprocessing are rejected before
model loading, and legacy screening records its unverified feature identity.

The separate [CPU image-feature route](cpu-route.md) also ran official pretrained
ResNet18 inference on 20 synthetic six-view samples, verified a one-sample resume,
trained all four CPU methods and rebuilt their table. Ten further unlabelled
synthetic samples used a disjoint token set and a matching extraction protocol.
This verifies execution, not driving accuracy. No real-data or full interactive
notebook session has completed.

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
| `tests/test_real_assets.py` | Optional integration against a supplied real feature/label/scene bundle; explicit skip when absent |

Run `python -m pytest tests -q` with loopback access to include the HTTP
route. A passing non-HTTP selection or isolated HTTP check does not establish
a passing combined full suite. All checks require their installed dependencies; the
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
managed proxy with HTTP 403; no real archive or labels were produced.

## Real-asset integration checklist

These checks have **not run** in this round because the real assets are absent:

1. Load a small authorised nuScenes set, resolve all six cameras and scenes,
   and confirm label provenance, review/exclusion records and sample IDs.
2. Extract a small VLM batch with recorded model revision, layer, hook, pooling,
   prompt/rubric versions and image hashes; verify row alignment and finite
   features. Record runtime and memory before estimating a full extraction.
3. Load recovered trained weights/standardiser, score the frozen manifest and
   compare saved per-sample outputs and historical logs without rewriting them.
4. Open authorised real cached images in the explorer, inspect real FP/FN
   examples and export a review queue for independent human decisions.

The optional cached-feature integration entry point is:

```bash
DRIVING_REAL_ASSET_DIR=/path/to/real_bundle python -m pytest tests/test_real_assets.py -q
```

The directory must contain `features.npy`, `meta.json`, `labels.json` and
`manifest.json`. Without `DRIVING_REAL_ASSET_DIR`, the test explicitly skips;
synthetic substitutes do not count as real-data integration. This cached-data
check does not run VLM extraction or verify the original trained checkpoint.

## New checks

Test failures that could silently change data meaning or a conclusion:
missing scenes, duplicate/conflicting tokens, unrecognised labels, overlapping
splits, nonfinite features, changed token-to-feature alignment, test-dependent
selection, altered predictions or mismatched metadata. Keep tests small enough
for CPU CI. Semantic names and human judgement need evidence records beyond
unit assertions. Refer to [feedback status](feedback-status.md) for the real-data
completion gate and current limits.
