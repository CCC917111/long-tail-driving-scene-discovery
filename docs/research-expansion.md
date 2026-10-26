# Driving research expansion

This is a data audit and a proposed next experiment, dated 2026-10-26. The
expanded experiment has **not** run. Its immediate blocker is original camera
data and matching metadata for additional scenes. No data download, paid
service, GPU reservation, new labels, or new training is established here.

The next useful step is to identify any already-authorized trainval assets and
their available scene/image coverage. A larger experiment should follow that
inventory; an empty experiment framework cannot supply the missing evidence.

## What is actually available

The real local dataset is `dataset/nuscenes-mini-user-20261023`. Its five
preserved source tables contain:

| Table or asset | Count |
|---|---:|
| `scene.json` | 10 scenes |
| `sample.json` | 404 sample tokens |
| `sample_data.json` | 31,206 rows |
| `calibrated_sensor.json` | 120 rows |
| `sensor.json` | 12 rows |
| Six-camera keyframe metadata | 2,424 rows, covering all 404 samples |
| Samples with all six original JPEGs locally | 80 |
| Remaining samples with zero of six JPEGs locally | 324 |
| Original local JPEGs | 480, totaling 85,378,496 bytes |
| Original local camera sweeps | 0 |

Every sample has six camera keyframe references in metadata. Only the selected
80 have the referenced image bytes. The other 324 are metadata-only: 194 in
the old train scenes, 64 in validation scenes, and 66 in test scenes. There are
no partially complete additional samples.

| Scene | Source samples | Complete local timestamps | Old frozen split |
|---|---:|---:|---|
| scene-0757 | 41 | 8 | test |
| scene-0916 | 41 | 8 | test |
| scene-0553 | 41 | 8 | train |
| scene-0655 | 41 | 8 | train |
| scene-0796 | 40 | 8 | validation |
| scene-0061 | 39 | 8 | train |
| scene-1077 | 41 | 8 | train |
| scene-1094 | 40 | 8 | train |
| scene-1100 | 40 | 8 | train |
| scene-0103 | 40 | 8 | validation |

The ten scenes represent eight distinct `log_token` values. The six old
training scenes span four logs: scene-1077, scene-1094, and scene-1100 share
one log. Validation and test each span two logs. No log token crosses the old
splits. `log.json` is absent; the existing tables cannot join a log's location
or capture date. Scene descriptions are available, but do not establish labels.

The [completed human-review experiment](../results/human-ai-exposed-20261026/README.md)
has 77 included judgments and three uncertain exclusions:

| Old split | Normal | Tail | Included |
|---|---:|---:|---:|
| Train | 3 | 45 | 48 |
| Validation | 4 | 10 | 14 |
| Test | 5 | 10 | 15 |

All three normal training judgments come from scene-0655. These judgments were
submitted by one reviewer after AI-label exposure. Their recorded scope must
remain visible; they are not independent blind reference labels.

The real browser handoff records four uploaded ZIPs, eight selected timestamps
per scene, and a source `v1.0-mini.tgz` size of 4,168,148,189 bytes. The source
archive SHA256 is null. Per-file hashes identify the transported bytes; they
do not independently verify official-source authenticity. The genuine full
mini archive is not present on this worker.

A read-only workspace inventory, including ignored files but excluding secret
directories, DWTS, environments, dependencies, and caches, found no additional
original nuScenes scene assets. The four user ZIPs contain the same 480 mini
JPEGs. Other asset-looking tables and archives are synthetic fixtures. In
particular, `output/upload-helper-20261023/node/v1.0-mini.tgz` is a
9,033,232-byte **synthetic** fixture, supported by `synthetic_only: true` in
its `pack_verification.json` and `source_real_nuscenes: false` in
`output/upload-helper-20261023/verification.json`. It is not the user's real
full archive.

## Which expansion addresses the research limitation

An authorized full mini archive can supply up to 324 currently missing
timestamps. That improves temporal coverage within the same ten scenes. It
does **not** add held-out scenes, independent logs, or new evidence of
generalization to an unseen scene. All ten mini scenes and all 80 reviewed
timestamps are now known development data for a future study, including the
two scenes previously called test.

True scene expansion needs original six-camera keyframe images and matching
metadata for additional scenes, for example from official `v1.0-trainval`.
New scene tokens must be checked against all ten old scenes; new held-out
logs must also be checked against all eight old logs and new training logs.
Additional metadata without the referenced image bytes is insufficient.

## Proposed v2 study

Preserve the original 6/2/2-scene assignments, labels, caches, checkpoints, and
both prior result bundles unchanged. Record v2 in fresh data/output/result
locations. Its primary comparison should use new, blind-reviewed data; retain
the old mini experiments as development evidence. Any reuse of old exposed
labels for fitting must be specified and disclosed before running v2.

After inspecting available assets, use a modest acquisition and review plan.
An initial **planning target** is 50 additional scenes: 30 for training, ten
for validation, and ten for test, with eight evenly spaced timestamps per
scene (400 six-view reviews). A normal-training **planning goal** is at least
100 judged normal timestamps distributed across at least ten training scenes.
These numbers organize effort; they are not a power calculation, a minimum
sample size for valid science, or a promise that the dataset contains those
labels. Revise the plan from the asset inventory and review budget before
freezing it. Scene/log diversity matters more than obtaining many adjacent
frames from one sequence.

1. **Freeze acquisition and allocation before reference labels and model
   results.** Select candidates using source metadata and a recorded sampling
   rule, without model scores or normal/tail labels. Group scenes by source
   log. Allocate whole logs to train, validation, and test when available;
   held-out scenes must be disjoint from every old mini scene and, where
   source identifiers permit, every old mini log. Keep unverified log
   separation explicit rather than claiming it. Use `log.location` and
   `date_captured`, when supplied, to spread sampling across locations and
   capture sessions. Record geography per split. A specifically geographic
   holdout, if feasible, must be declared before labels; avoid a geographic
   generalization claim from scene separation alone. Hash the selected sample
   list, image inventory, split assignments, rubric, and exclusions.
2. **Collect blind human judgments.** Present timestamps in randomized order
   with all six views consistently positioned, without model predictions,
   earlier AI decisions, split names, or scene descriptions. Freeze the
   normal/tail/uncertain rubric first. Have a
   second human independently review a prespecified overlap, such as 20% of
   each split, selected before judgments, plus documented adjudication of
   disagreements. Retain both original judgments, uncertainty, reviewer
   identities, and reason notes. Report overlap size, class counts, and
   disagreement; do not describe one reviewer or a small overlap as complete
   independent ground truth.
3. **Check class support honestly.** Report normal/tail/uncertain counts by
   split, scene, and log, and the number of training scenes contributing
   normals. Do not relabel examples, move scenes between frozen splits, or
   repeatedly draw new splits to manufacture balanced classes. If training
   normals are scarce, add training-only scenes under a prespecified,
   label-independent sampling rule; preserve the held-out allocation. If a
   held-out split lacks useful class support, retain that fact and undefined
   metrics. A later expanded evaluation cohort needs a new version and an
   untouched allocation; existing held-out judgments cannot silently become
   a fresh test.
4. **Run one controlled comparison.** Use the existing pinned CPU ResNet18
   six-camera feature protocol, extracting each new sample once and verifying
   source image, encoder, preprocessing, and cache hashes. The same manifest,
   cache, review exclusions, training-only normalization, and selection rules
   feed logistic regression, the small MLP, AbsTopK SAE without reward, and
   AbsTopK SAE with norm reward. Freeze settings from development evidence;
   validation alone may select checkpoints and the operating threshold. Keep
   attempted configurations visible. When resources allow, use three fixed
   model seeds on this same allocation and report each run plus mean and
   sample standard deviation. A deterministic method may yield the same
   result across seeds. Keep a one-seed result labeled as one seed; do not
   create an extensive experiment matrix before establishing data support.
5. **Evaluate the frozen held-out set once.** Predeclare K and the target
   validation precision. Report test AP, Precision@K, Recall@K, class support,
   and precision/recall at the threshold selected on validation. Keep the
   top-K policy, validation-selected operating point, and original fixed
   decision rule separate. An unreachable validation precision target must
   remain explicit. Report the analytic uniform-random top-K expectation as
   an expectation. Inspect representative errors after final evaluation;
   resulting changes belong to a subsequent study.

Use scene-cluster bootstrap intervals as described in
[evaluation.md](evaluation.md), conditional on a fixed model and validation
threshold. With few scenes, intervals are unstable and cannot manufacture
independent evidence or establish significance. Multiple scenes from one log
also limit a scene-only independence assumption; retain the log grouping and
state that limitation. Seed spread and reviewer disagreement are separate
uncertainties, not quantities captured by those intervals. Report undefined
single-class resamples and avoid treating overlapping intervals as a
significance test. A planning target of ten test scenes does not guarantee a
precise estimate or a detectable difference.

This remains a custom image-feature study. The existing preparer's trainval
70/15/15 allocation is an internal split, not the official nuScenes benchmark
split, and it currently groups by scene rather than log. It must not silently
replace the v2 allocation described above. No Cosmos, cross-dataset,
vehicle-safety, or reward-improvement claim follows from successful execution.

## Acquisition and one concrete handoff

If the local files are uncertain, first search the computer's Downloads folder
for `trainval` or `nuscenes`. Record the matching file/folder names and displayed
sizes. A filename containing `v1.0-trainval` is a candidate for inspection;
`v1.0-mini.tgz` and the four `driving-mini-part` files are the old mini source.
Filenames alone do not verify contents. Share that small inventory first so the
required metadata and selected camera paths can be identified before a large
transfer. The current evidence does not establish that any trainval asset is
already available on the user's computer.

The public official devkit sources were checked through the read-only GitHub
connector on 2026-10-26:

- [Official nuScenes setup](https://github.com/nutonomy/nuscenes-devkit/blob/master/README.md#nuscenes-setup)
  directs users to the [download page](https://www.nuscenes.org/download), to
  create an account, agree to the [terms](https://www.nuscenes.org/terms-of-use),
  and log in to see the archives. It describes the standard full devkit layout
  with `samples`, `sweeps`, `maps`, and `v1.0-*` tables, and says the full
  devkit requires all archives. This project's selected camera-only handoff
  is a narrower input contract; it is not a complete standard devkit install.
- [Official FAQ](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/faqs.md)
  states: "For non-commercial use nuScenes and nuImages are free," with
  educational and some research use as examples. Free use is subject to the
  dataset's terms. No purchase or paid service is part of this proposal.
- [Official schema](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md)
  confirms that multiple scenes can originate from the same log, identifies
  `scene.log_token` and `log.location`/`date_captured`, and defines keyframe
  image references in `sample_data`.

No current official trainval download size or archive inventory was verified,
so no total download size is quoted. No account was logged into, terms
accepted, or dataset downloaded during this documentation work.

**Input handoff:** identify an already-authorized trainval data root or supply
an equivalent transfer bundle preserving original paths and bytes:

```text
<data-root>/
  v1.0-trainval/
    scene.json
    sample.json
    sample_data.json
    calibrated_sensor.json
    sensor.json
    log.json                 # include for location/date joins when available
  samples/
    CAM_FRONT/...
    CAM_FRONT_LEFT/...
    CAM_FRONT_RIGHT/...
    CAM_BACK/...
    CAM_BACK_LEFT/...
    CAM_BACK_RIGHT/...
```

The required pixels are the six original keyframe JPEGs for every selected
sample, with metadata referencing them. Include source/version and archive
inventory/checksums already available; preserve missing provenance as missing.
If trainval exists only as archives on the user's computer, begin with its
metadata and archive inventory so the assistant can identify the specific
disjoint scenes and required image paths before a large transfer. There is no
need for the user to choose sample tokens, create labels in advance, or buy
hosting. If only mini is available, that supports the within-scene extension
only, and the true new-scene experiment remains blocked.

The current [browser helper](../tools/prepare_upload.html) and
[importer](../scripts/import_mini_parts.py) are mini-only: exactly two or eight
timestamps per scene, five to ten source scenes in the helper, and at most
80 selected timestamps in the importer. They cannot currently ingest
trainval or expand mini beyond eight timestamps per scene. The Python
[preparer](../scripts/prepare_driving.py) already reads trainval and arbitrary
positive frame counts, but does not create missing pixels or implement the
proposed log-aware split. These constraints should be adapted against the
actual supplied assets, rather than building an empty transfer or training
framework now.

Once that input is accessible, the assistant can autonomously validate source
coverage, identify old-scene/log overlaps, compute the candidate sample/image
inventory, estimate transfer and review volume, adapt the minimal necessary
handoff, and prepare a concrete frozen review batch in fresh directories.
After actual human review returns, it can validate class support, extract and
cache CPU features, run the same four methods, audit saved predictions, and
produce the comparison with its limitations. Human judgments and missing
original pixels cannot be supplied by scaffolding. The outstanding input is
whether additional authorized trainval assets exist and can be made accessible;
the stronger experiment is not yet executed.
