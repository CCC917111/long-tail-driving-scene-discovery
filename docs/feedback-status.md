# Feedback implementation status

Updated 2026-10-29. The original audited version was
`b57eaffa21d54ad67d98d441935cf6b17cdc5294`. The current publication baseline is
`e4d8fc4c9a3bf877c98f24a42558dadc174c063b`, after PRs
[#4](https://github.com/CCC917111/long-tail-driving-scene-discovery/pull/4) and
[#5](https://github.com/CCC917111/long-tail-driving-scene-discovery/pull/5) merged.

The CPU pipeline, saved-output audits and review interfaces are complete.
Research conclusions remain limited by the small cohorts, AI-disclosed
single-reviewer references and scene/class confounding. Stable model gains and
independent blind-label accuracy have not been established.

## Current round

| Area | Completed | Remaining limitation |
|---|---|---|
| Data contracts | Strict labels and scene IDs, frozen manifests, provenance and exclusion records. | Historical assets, raw-VLM supports and the six-frame count gap remain unresolved. |
| Real mini inputs | Four transferred parts validated; 80 timestamps, ten scenes and 480 images; pinned CPU ResNet18 cache and frozen 6/2/2-scene allocation. | The original archive checksum was not independently verified; current features do not reproduce Cosmos. |
| Human-reference CPU fit | 77 binary judgments and three uncertain exclusions; Logistic, MLP and both SAE variants fitted and independently reconstructed. | One reviewer after AI-label exposure, blank notes, seed 42, two test scenes and three normal training samples from one scene. |
| New-scene replay | Nine timestamps and 54 images from two additional scenes; existing models applied without fitting; all nine human judgments received. | AI-reference disclosure preceded submission; per-row timing is unknown; all five normals are in one scene and all four tails in the other. Original log IDs are unavailable. |
| Frozen score policy | Old validation threshold `6.074954873` selected five of nine: precision 60%, recall 75%. | Descriptive agreement with submitted references; no tuning on these labels or independent generalization estimate. The original activation rule selected all nine. |
| Reward comparison | Matched reward/no-reward configurations, common inputs and rebuilt result tables. | Both SAE checkpoints are identical; no reward benefit was observed. |
| Review and demonstration | Offline explorer with 77 retained timestamps and 308 cached method values, saved MLP FP/FN bookmarks, blank review export and an [English recording](../demo/driving-walkthrough-90s.mp4). | Unit IDs have no validated semantic names; example context does not prove an error's cause. |
| Portable execution | Embedded-source CPU notebook, staged acquisition, persisted review and strict four-method driver. | Real Colab execution and the original GPU/VLM route have not been verified. |

Total real input coverage is 89 timestamps, 12 scenes and 534 images. The nine
supplemental timestamps were scored with existing models; they did not enlarge
the fitting cohort.

The [private review Site](https://yike-driving-review-20261028.upbeatperch12.chatgpt.site)
requires the owner's ChatGPT sign-in and serves all 54 supplemental images from
the same origin. The [downloadable fallback](../demo/driving_new_scenes.html)
uses the author's `raw.githubusercontent.com` image URLs. External GitHub
reviewers do not automatically have access to the private Site. See the
[presentation record](../results/presentation-20261029/README.md) for the public
recording and the distinct original private-delivery receipts.

Yike leads the project and the repository modernization and delivery. The
[original contribution table](../README.md) retains the capstone team's roles,
including the annotation platform and most feature interpretation by named
teammates. Validation, provenance, baselines, data handoff, the offline explorer,
fixtures and documentation added on 2026-10-19–23 were AI-assisted follow-up
work. That work belongs to the modernization phase and should be reviewed and
understood by Yike before presentation. It must not be attributed retroactively
to the original capstone period; earlier AI use is not inferred.

## Human-reference provenance and results

The final old-cohort CSV was received on 2026-10-26 and matched all 80 frozen
sample IDs. Its SHA256 is
`c39de8776325f6d72ee22738dcf2c95ee653db439f416de0a4e9403ea8f8a43a`.
Every row names Yike: 12 `normal_core`, 65 `not_normal_core` and three
`uncertain`. Binary rows have status `reviewed`; the uncertain rows are excluded
from supervised fitting and evaluation. None remain pending.

All notes are blank, including uncertainty reasons. The first ten judgments
are preserved as a pre-exposure checkpoint, along with four subsequent
revisions. The full submission followed display of the complete AI label table.
It is single-reviewer evidence after AI-label exposure, not blind independent
reference data.

| Old split | Normal | Tail | Uncertain |
|---|---:|---:|---:|
| Train | 3 | 45 | 0 |
| Validation | 4 | 10 | 2 |
| Test | 5 | 10 | 1 |

The CPU rerun reused the ResNet18 cache, frozen scenes and training settings.
Independent reconstruction checked source/input hashes, training-only scalers,
checkpoint scores, validation thresholds, metrics and the comparison table. On
the 15-row test subset, SAE AP was `0.9353`, precision@4 was `1.0` and recall@4
was `0.4`. Its validation-selected operating point had precision `1.0` and
recall `0.2`. These measure agreement with the disclosed references.

Both SAE checkpoints are byte-identical to each other and to the earlier
AI-label pilot. Logistic is also unchanged. MLP checkpoint selection moved
from epoch 4 to 5 under the same settings and training loss trajectory. The
[completed result bundle](../results/human-ai-exposed-20261026/README.md) retains
source snapshots; a later evaluator provenance guard does not rewrite them.
The [nine-row supplement](../results/new-scene-human-review-20261028/README.md)
is reported separately.

## Verification record

The following checks retain their original dates and scopes; software checks
do not validate label quality.

- The final local CPU selection passed **205 tests and 18 subtests**, plus lint.
  One HTTP test was deselected because the sandbox denied loopback binding.
  The optional real-asset integration ran with the actual feature bundle and
  explicitly provisional AI labels; that private-asset check was skipped in CI.
- At `ba8efddd46b357f492965b941172a8fd1f6dd948`, Actions passed the
  [CPU/lint/HTTP workflow](https://github.com/CCC917111/long-tail-driving-scene-discovery/actions/runs/40039920171)
  and [offline Chrome workflow](https://github.com/CCC917111/long-tail-driving-scene-discovery/actions/runs/40039920157).
  The [earlier CPU run](https://github.com/CCC917111/long-tail-driving-scene-discovery/actions/runs/40039177713)
  records the 205 tests and 18 subtests.
- Chrome 155 checked desktop/mobile navigation, six-view rendering, cached
  values and CSV download with watermarked synthetic ZIPs. Separate Node checks
  verified all 480 local real-image bytes and 292 displayed AI-pilot values
  against saved outputs. Those checks covered different artifacts.
- The synthetic E2E run fitted four models on 60 scenes/240 samples, reloaded
  checkpoints and training scalers, rebuilt the table and screened 40 separate
  unlabelled samples. Two HTTP explorers passed 14 route checks each, including
  images and blank-label `/export/review.csv`. The screening override `eta=0.05`
  persisted through reload, separately from the training `eta=0.01`.
- Synthetic preparation/import checks used an 80-sample, ten-scene fixture.
  Extraction hooks and resume used a fixture model. These preceded real mini
  acquisition and do not constitute VLM execution.
- Real CPU extraction on 2026-10-23 passed a one-sample smoke and resume into an
  `(80, 3072)` cache on the frozen 48/16/16 timestamp allocation. Logs and hashes
  remain under `output/real-cpu-mini-20261023/` in the execution workspace.
- The later [AI-label pilot](ai-pilot.md) fitted four methods on 73 retained
  timestamps and screened seven uncertain ones. It used one seed and two test
  scenes; all validation precision targets were unreachable and both SAEs were
  identical. Its published bundle excludes original images, metadata and weights.
- The 2026-10-27 new-scene run retains its original
  [execution source](../results/new-scene-e2e-20261027/execution_source.zip).
  The [2026-10-28 repair record](../results/review-page-repair-20261028/README.md)
  covers loading guards, retry, CSV export and private publication separately.
  All 54 real images decoded in local desktop/mobile checks. Temporary UI test
  decisions were discarded; authenticated production browsing was not recorded.

Screening persistence, duplicate-token, nonfinite-input and standardizer checks
are software fixes. See [testing](testing.md) and [E2E verification](e2e.md) for
artifacts and commands. CPU versions are pinned in `requirements-cpu-lock.txt`;
`requirements-gpu.txt` has not been verified on CUDA. Local Cosmos weights and a
CUDA device remain unavailable.

## Historical claims needing recovery

Recover the original labels/review records, sample and scene IDs, extraction
settings, feature arrays, checkpoint, standardizer, commands and logs. Reconcile
raw-VLM predictions and supports on the same test IDs before claiming a matched
comparison. Explain why 824 scenes at eight keyframes imply 6,592 samples while
the report lists 6,586; retain exclusion IDs and reasons. Preserve historical
random-frame 80/20 results separately from current scene-level protocols and
verify capstone contributions against the report, commits and team records.

<a id="human-validated-experiment-completion-gate"></a>

## Next research milestone

The [expansion plan](research-expansion.md) calls for new scene/log coverage,
blind human review and adequate normal training support. Freeze that inventory,
rubric, manifest and review budget before model results. Use one shared cache and
selection protocol for Logistic, MLP and the two SAE configurations; vary only
reward in the core SAE comparison. Retain seeds, attempted settings, source
hashes and per-sample outputs so every table can be reconstructed.

Report model-seed spread separately from scene intervals and reviewer
disagreement. Name units on train/validation, freeze the interpretations and
check them on independent samples. A result favoring a cheaper baseline remains
a valid outcome. Estimate transfer, review and compute costs from a small run
before committing paid resources.
