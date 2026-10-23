# First real mini CPU extraction

On 2026-10-23, the local upload route successfully moved a real nuScenes mini
subset into the managed workspace and completed feature extraction without a
GPU or paid service. This is a data/extraction smoke run, not a model-quality
result or a reproduction of the historical Cosmos experiment.

| Check | Observed result |
|---|---|
| Transport | Four complete `nuscenes-mini-browser-handoff-v1` ZIP parts |
| Selection | Eight uniformly spaced timestamps per scene; 80 timestamps from ten scenes |
| Input images | 480 unique original JPEGs, all decoded as 1600 × 900 RGB |
| Metadata | Five original JSON tables; transported and decoded hashes verified |
| Frozen split | Six/two/two scenes; 48/16/16 frames; zero scene or token intersections |
| Features | Finite `float32` array with shape `(80, 3072)` |
| Encoder | Pinned ImageNet ResNet18; six 512-dimensional views concatenated |
| Execution | One-timestamp smoke, then resume to all 80 timestamps |
| Labels / model scores | No reviewed labels; no supervised real-data models or accuracy metrics |

The mini allocation is a custom 60/20/20 scene split for a proof of concept;
nuScenes mini has no official test split. It is separate from the planned
70/15/15 trainval protocol. The recorded source archive is 4,168,148,189 bytes;
its full checksum was not independently verified. File hashes establish the
exact transported inputs, not their official-source authenticity.

## Reproduction record

Local paths are `dataset/nuscenes-mini-user-20261023` and
`output/real-cpu-mini-20261023`. The latter contains:

- `upload_import.json`: part completeness, selection and transport record.
- `prepared/`: frozen manifest, scene allocation, rubric and blank review CSV.
- `extract-smoke-run.json` and `extract-full-run.json`: exact extraction
  commands, wall times, exit codes and preserved stage summaries.
- `extract/`: arrays, metadata, input/encoder contract and atomic checkpoint.
- `independent_audit.json`: read-only checks of source hashes, every image,
  selection, split intersections and cache alignment.
- `verification.json` and `execution_source.zip`: execution inventory, Git base
  plus dirty-source hashes and the actual runtime source snapshot.

The extraction invocations took 2.18 seconds and 25.79 seconds respectively,
including interpreter startup and input checks. These describe this one local
run; they are not a performance comparison. The targeted software checks
passed 34 tests and 16 subtests. Browser interaction has not been verified in
this sandbox.

## Remaining human input

Inspect all six original views under the frozen normal-core rubric and choose
`normal_core`, `not_normal_core` or `uncertain`. The classification concerns
visible conditions requiring extra defensive driving, not frequency or scene
descriptions. Keep reviewer identity and notes. Pending/uncertain samples remain
excluded; model output must not be recorded as a human decision.

The [offline reviewer](local-upload.md#review-the-same-parts-locally) opens the
same four local ZIPs and exports the required CSV. Its actual-part parser,
image bytes and blank CSV import were checked; browser interaction remains
unverified in the execution sandbox.

Import the resulting CSV without changing the frozen split:

```bash
python scripts/prepare_driving.py import-labels \
  --prepared-dir output/real-cpu-mini-20261023/prepared \
  --reviews output/real-cpu-mini-20261023/reviewed.csv \
  --output-dir output/real-cpu-mini-20261023/annotations
```

Check `annotations/import_metadata.json` before training. The current four-model
driver requires both classes in each frozen split. If support is insufficient,
record that limitation and obtain more suitable data; do not change decisions
or shuffle scenes to obtain a score. Use the same complete cache and manifest
for all methods, and choose the review budget before reading model results.
