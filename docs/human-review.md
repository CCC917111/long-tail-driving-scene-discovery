# Human review and model-label checks

This workflow covers the original mini cohort: **80 timestamps, with six
original camera images per timestamp**. Use the frozen
[normal-core rubric](../latest_grading_criteria.md), SHA256:
`f19373f55ffc930f67c340ae5fffde061cd68dc0b3128f10ca7b14bf9cf0e71c`.

A human review means a person inspected the images and made the decision.
Agreement between model annotation passes does not establish human ground
truth or label accuracy.

## Submitted labels and prior model-label disclosure: 2026-10-26

Yike's final CSV contains all 80 frozen sample tokens, with no missing,
duplicate or extra rows. Its SHA256 is
`c39de8776325f6d72ee22738dcf2c95ee653db439f416de0a4e9403ea8f8a43a`.
Every row names `Yike` as reviewer. The CSV records 77 binary `reviewed`
decisions: 12 `normal_core` and 65 `not_normal_core`. The other three rows
have both label and status set to `uncertain`; none remain `pending`.
All 80 `review_notes` fields are blank, including the three uncertain rows.
No reasons were added or inferred.

An earlier CSV, with only the first ten rows completed, is preserved with
SHA256 `c56adb638a09276ff027fb039b049ec8be90d251aaa3b3477c3263760d4c49ec`.
Yike saw the complete model-generated label table before submitting the final version.
The final labels are therefore **single-reviewer judgments after model-label
disclosure**, not an independent blind reference. Four of the first ten rows
changed: zero-based indices 0, 3 and 9 changed from `normal_core` to
`not_normal_core`; index 8 changed from `normal_core` to `uncertain`.
Both original files and the revision record are retained. Of the remaining
70 rows, 68 became `reviewed` and two became `uncertain`.

The original 6/2/2-scene allocation was unchanged:

| Frozen split | Normal | Not normal | Uncertain, excluded |
|---|---:|---:|---:|
| Train | 3 | 45 | 0 |
| Validation | 4 | 10 | 2 |
| Test | 5 | 10 | 1 |

The 77 binary decisions meet the importer's requirement for both classes in
each split. This check establishes usable class support, not annotation
independence or model quality. All three normal training timestamps belong to
one scene. The three uncertain decisions remain excluded from supervised
training and evaluation.

All four CPU methods were fitted using these labels, the existing ResNet18
features and unchanged training settings. An independent replay checked input
and source hashes, train-only normalization, all four reloaded checkpoints,
validation-selected thresholds, saved metrics and the comparison table. The
metrics measure agreement with the human references collected after model-label disclosure. Both SAE
checkpoints are byte-identical to each other and to the original provisional-label pilot
checkpoints; no reward gain was observed. Changed validation decisions moved
the MLP's selected checkpoint from epoch four to epoch five, while its settings
and ten-epoch training-loss history remained unchanged.

The execution source snapshot is preserved. A later provenance-field guard
repair is recorded separately. See the
[human-review run](../results/human-ai-exposed-20261026/README.md), the original
[provisional-label pilot](ai-pilot.md) and the separate
[model-label recheck](../results/ai-label-recheck-20261024/README.md).

## 1. Prepare for an independent review

1. Download the [blank review tool](../tools/review_upload.html), save the
   HTML file and open it in desktop Chrome or Edge.
2. Select all four original `driving-mini-part-…zip` files together and use
   the load-and-validate control. Confirm that the tool shows 80 frames.
   Images are read locally. For a first independent review, use a separate
   browser profile and check that no earlier answers have been restored.
   The tool restores local records for the same data and rubric; downloading
   the HTML again does not clear them.
3. Enter the actual reviewer's name. Read the full rubric, then inspect a few
   examples to learn the controls.
4. During formal review, use the blank tool and avoid old model-generated labels, model
   scores, predictions and split assignments. Use the scored explorer later
   for error analysis.
5. Record any rubric clarifications and freeze them before formal annotation.
   A changed rubric needs a new version and another check of affected samples;
   do not revise the rubric recorded for an existing experiment.

## 2. Check every timestamp in the same order

Review the **whole six-view timestamp**, not just the front camera.

1. Inspect front, front-left, front-right, rear, rear-left and rear-right in
   the displayed order. Note visible blur, glare, low light, rain, fog or
   occlusion.
2. Check the drivable area and static features: junctions, lane boundaries,
   construction, cones, barriers, road closures and obstacles in the lane.
3. Check traffic participants, including nearby vehicles, pedestrians and
   cyclists relative to the visible driving path. Consider whether large or
   parked vehicles hide a relevant area. A distant person or vehicle alone
   does not make a frame abnormal.
4. Choose one label under the frozen rubric:

   | Label | Decision rule |
   |---|---|
   | `normal_core` | The visible scene is clear and routine, with no evident factor requiring additional defensive driving. |
   | `not_normal_core` | A visible factor calls for extra caution, or a static difficulty meets the frozen rubric. |
   | `uncertain` | A key detail is unclear, information is insufficient or the decision is unstable. |

5. For difficult, non-normal or uncertain decisions, record the camera, visible
   evidence and its relevance. For example: "CAM_FRONT_RIGHT: a barrier is
   close to the visible travel area and restricts its boundary." Avoid events
   that a still image cannot establish, such as an imminent collision, and do
   not infer statistical rarity from appearance.
6. Open a camera thumbnail in another tab to inspect the original JPEG at a
   larger scale. Finish before moving to the next timestamp: changing frames
   releases the previous frame's Blob links. If the evidence remains unclear,
   choose `uncertain` and retain the reason. The link lifecycle was checked in
   code; it did not establish a complete human review session in the original
   execution sandbox.

Judge each timestamp on its evidence. Do not revise labels to balance classes,
improve a score or agree with another label set.

## 3. Save work and obtain a second human review

- Export `reviewed.csv` after each batch and again when finished. Browser
  storage is a convenience, not a substitute for the exported file.
- Keep each reviewer's original CSV with their real name or stable identifier.
- Ideally, a second person reviews the same frames independently on another
  computer or browser profile. If the budget permits only a subset, choose it
  in advance, add uncertain cases and record which frames had one or two
  reviewers. Do not describe the remainder as double-reviewed.
- Lock both original files before comparing disagreements. Inspect the
  original images together and save the final decision, participants and
  reason separately. Preserve the original judgments; unresolved cases stay
  `uncertain`.
- Import one final decision per token. Keep overlapping reviewer CSVs as
  evidence rather than concatenating them into duplicate-token input.
- A model disagreement list can guide extra review after the first independent
  human pass. Showing old model-generated decisions earlier can anchor judgments. A
  disagreement-focused check is not a random-sample estimate of label quality.

## 4. Import labels and define a new experiment

Preserve the original CSV and any second-reviewer or adjudication records. The
importer checks token coverage, identity, status, allowed labels, completeness,
the frozen scene allocation and class support in each split. Pending and
uncertain rows remain excluded from supervised fitting and evaluation.
A name in a CSV is not proof that a person inspected the images.

Human labels define a **new experiment version**. Retain the frozen 6/2/2 scene
allocation, recalculate supports, set review budget K before inspecting the new
test results, retrain all four models and select checkpoints and operating
points using validation only. Relabelling old predictions alone does not mean
the old model was trained on human references. Preserve the provisional-label pilot's labels,
predictions and tables. If class support is insufficient, record the limit and
collect suitable data rather than changing scenes or judgments to obtain a score.

Report the number of reviewers, their coverage, disagreements and the rubric.
Human review does not remove the mini cohort's limited size and scene coverage.

## Historical model-label consistency check

The [2026-10-24 consistency record](../results/ai-label-recheck-20261024/README.md)
compares additional vision-model annotations with the original provisional
labels under a fixed rubric and sample assignment. These are model-to-model
comparisons. They do not replace human review, establish annotation accuracy
or change the submitted human decisions.
