# New-scene CPU E2E evidence

**Later review, 2026-10-28:** all nine human decisions are now recorded
[separately](../new-scene-human-review-20261028/README.md), with a record of the
model-generated references disclosed before submission. The original execution
artifacts below remain unchanged.

**Completed inference and review-interface check. No accuracy was measured in this
run.** The official DriveLM author demo supplied 54 original JPEGs in nine six-camera
timestamps across two scenes, pinned to commit
`1de72a74b257e5373400fa68239e99bd5d20580a`. Both scenes and all nine sample IDs are
absent from the original mini cohort. Original log metadata is unavailable.

The existing no-reward SAE checkpoint and its train-only standardiser were reused
unchanged. All nine fixed-rule outputs are tail candidates; these are not verified
positive labels. This run involved no fitting, split allocation, human labeling or
accuracy evaluation. Training-label provenance remains the earlier single-reviewer
submission after model-label exposure.

| Evidence | What it records |
|---|---|
| [source_manifest.json](source_manifest.json) | Source commit, 54 exact image URLs/hashes and nine scene/sample mappings; no upstream QA |
| [extraction_config.json](extraction_config.json) | CPU encoder/preprocessing identity, source hashes and original input-manifest hash |
| [flow.json](flow.json) | Completed real-data execution, `(9, 3072)` features, unchanged training assets and scope |
| [screening.csv](screening.csv) | Nine model scores, fixed-rule outputs and numeric active units; no human labels |
| [screening_metadata.json](screening_metadata.json) | Actual checkpoint/standardiser/feature hashes and decision rule |
| [independent_audit.json](independent_audit.json) | Fresh original-image encoding and model reload; all features, activations and scores match exactly |
| [explorer_browser.json](explorer_browser.json) | Actual Chromium desktop/mobile neuron navigation, six-view decoding, search and CSV |
| [review_browser.json](review_browser.json) | Actual Chromium desktop/mobile review flow; all 54 original images per viewport, blank initial CSV, decisions, notes and exposure persistence |

`flow.json` records `browser_ui_verified: false` at pipeline export time. Browser checks
occurred later and are recorded separately in the two browser reports. The review check
uses local copies of the real original JPEGs. Automated test judgments are marked
`CI_TEST_ONLY`; they are excluded from the result record and are not human labels. The
source-linked public HTML uses the same image URLs whose bytes were checked at
acquisition; its separate CI tests report their own status. No raw images, weights or
rendered image screenshots are committed here.

For compatibility, `screening_metadata.json.label_scope` is the inherited old
**training-reference** scope, also stored as `model_training_label_scope`. The current
frames are unlabelled (`kind: unlabelled-screening`); the review export explicitly
records `has_reference_labels: false` and `human_reviewed: false`. Do not interpret the
inherited training scope as new human review of these frames.

[Open/download the new review HTML](../../demo/driving_new_scenes.html) · [Execution,
usage and replay](../../docs/new-scene-e2e.md)

## Recorded execution source

[execution_source.zip](execution_source.zip) preserves all seven source files whose
SHA256 values were recorded in extraction_config.json, including the exporter before its
later image-loading repair. It also preserves the exact independent audit source and
licence notices. [execution_source_manifest.json](execution_source_manifest.json) lists
every archive entry hash and the immutable public base used for other dependencies. The
[independent audit script](verify_e2e.py) retains its original worker paths and requires
the real private image/cache/checkpoint assets. This snapshot preserves the execution
source. It does not include the private data and model assets required to rerun it.

The later [review-page repair](../review-page-repair-20261028/README.md) changes the
exporter and delivery mechanism. Original flow, extraction and audit hashes remain
unchanged. The original execution source and later repair are recorded separately.
