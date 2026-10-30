# Controlled offline evaluation

The evaluator compares logistic regression, a small MLP and two SAE variants
on a frozen feature matrix and scene allocation. The current
[real CPU run](../results/human-ai-exposed-20261026/README.md) uses 77 binary
references from one reviewer who had seen model-generated labels. Its SAE test
AP of 0.9353 covers 15 timestamps in two scenes at seed 42; both SAE variants
are identical.
Historical random-frame results remain separate. The synthetic fixture below
checks execution and deliberately includes mistakes; its metrics are test data.

## Shared data and model selection

All methods must use the same encoder features, labels and frozen manifest. Create
the manifest with the SAE entry point once, review it, then supply its path with
`--manifest` to every subsequent run. Do not regenerate splits for different
model seeds. Manifests identify `sample_token`, `scene_token`, binary `label`,
`split` (`train`/`val`/`test`), label source, review status and exclusions.
The evaluator joins scores by token and rejects missing/duplicate predictions,
label/split conflicts, absent scene IDs and scenes spanning multiple splits.

Standardisation uses train mean and standard deviation only. Logistic regression
uses one fixed regularisation strength; its validation log loss is recorded but
does not choose among unreported models. The MLP uses explicit validation log
loss after each epoch, with the earliest minimum saved. It does not use sklearn's
internal random validation split. SAE checkpoint selection remains validation
loss. If tuning hyperparameters, retain every tried configuration and use only
validation for that choice. Test labels must never select a checkpoint, feature
layer, prompt, threshold or review budget.

`--target-precision` selects a score threshold on validation: maximum recall
subject to the target precision; ties prefer higher precision, then the higher
threshold. The same threshold is applied unchanged to test. If the target is
unreachable, the output explicitly says `target_unreachable` and predicts no
positives. This empirical validation target is not a guarantee of test precision.
Original fixed-rule metrics (SAE activation count or baseline probability 0.5)
are retained separately from this fitted operating point.

## Trained synthetic end-to-end workflow

```bash
make e2e PYTHON=.venv/bin/python OUTPUT=output/e2e-example
```

Use a new output path. This trains four small models on 60 synthetic scenes /
240 frames with one frozen 42/9/9-scene split, verifies checkpoint reloads and
train-only standardisation, rebuilds the comparison, then screens 40 independent
unlabelled samples. The MLP and both SAE configurations use two epochs by default.
`make e2e` includes `--http`, which starts actual loopback servers and verifies
14 routes each for trained and screened data, including six JPEG images,
search, review export and an expected unknown-token 404.

This workflow passed locally. The screened run retains its effective `eta=0.05`
through reload, separately from the trained run's `eta=0.01`. The review CSV
contains blank reviewer fields and omits reference labels. Browser rendering
and click checks have a separate verification record in [testing](testing.md).

Every feature, image and source identifier is synthetic. No nuScenes input,
VLM extraction, original real checkpoint or scientific model result is involved.
See [end-to-end commands](e2e.md) for output files, standalone HTTP checks and
the optional real Chromium UI verifier.

Choose `--review-k` before reading test results. Precision@K and tail count use
descending score, with token order breaking ties. The random comparison is the
analytic expectation under uniform sampling without replacement from that same
split; it is not a separately trained method. If K exceeds split size, both
requested and effective K are reported.

## CPU fixture commands

Install the repository's Python requirements first. These commands use only
synthetic arrays. The baseline commands import the shared SAE data loader, so
torch must also be installed even though these two estimators run on CPU.
The evaluation contracts can be checked without torch or pytest with
`python tests/test_evaluation.py -q` (numpy and scikit-learn are required).
Use fresh output paths: training and evaluation refuse to overwrite an existing
checkpoint or evaluation bundle.

```bash
python scripts/evaluation_fixture.py --output-dir output/evaluation_fixture
python scripts/evaluate_runs.py \
  --rows output/evaluation_fixture/rows.json \
  --manifest output/evaluation_fixture/manifest.json \
  --output-dir output/fixture_evaluation --seed 17 --review-k 4 --bootstrap-samples 100
python scripts/supervised_baselines.py --method logistic \
  --features output/evaluation_fixture/features.npy \
  --meta output/evaluation_fixture/meta.json --labels output/evaluation_fixture/labels.json \
  --manifest output/evaluation_fixture/manifest.json \
  --output-dir output/fixture_logistic --seed 17 --review-k 4 --bootstrap-samples 100
python scripts/supervised_baselines.py --method mlp \
  --features output/evaluation_fixture/features.npy \
  --meta output/evaluation_fixture/meta.json --labels output/evaluation_fixture/labels.json \
  --manifest output/evaluation_fixture/manifest.json \
  --output-dir output/fixture_mlp --seed 17 --review-k 4 --bootstrap-samples 100 \
  --hidden-dim 16 --epochs 10
python scripts/summarize_runs.py \
  --runs output/fixture_logistic output/fixture_mlp --output-dir output/fixture_comparison
```

For real experiments, replace every fixture asset with recovered, authorised
data. Confirm extraction model revision, layer, pooling, input hashes and
prompt/rubric version before comparing models. File hashes establish identity;
they cannot recover missing historical provenance or human label quality.
The supervised CLI accepts `--feature-config recovered_extraction_config.json`
to retain recovered extraction settings and their file hash. An absent file is
explicitly recorded as unverified provenance, without inventing old settings.

## Result bundle

Each evaluated run saves:

| File | Evidence |
| --- | --- |
| `evaluation.json` | Protocol version, supports/confusion counts, AP/AUC, fixed-rule and validation-selected precision/recall/F1, precision@K and random expectation |
| `predictions.json`, `predictions.csv` | Every included token, scene, split, label, score, fixed prediction, frozen operating prediction and FP/FN marker |
| `review_queue_val.csv`, `review_queue_test.csv` | Top-K candidates with score/rank and blank reviewer fields; reference labels are omitted to support independent review |
| `evaluation_manifest.json` | Exact manifest content used by evaluation |
| `pr_curve_val.csv`, `pr_curve_test.csv` | Full PR curve data; PNG plots are also saved when matplotlib is installed |
| `run_metadata.json` | Configuration, seed, dependency versions, commit, dirty status, hashes of Python source files, inputs, checkpoint, manifest and predictions |
| `source.patch` | Tracked working-tree diff when the run used uncommitted changes; new script hashes are also in metadata |
| `source_snapshot.zip`, `requirements-resolved.txt` | Python source (including untracked scripts), tests, Makefile/requirements, and installed package versions for local uncommitted runs |

The supervised commands additionally save the fitted estimator and train-only
normalisation in `model.joblib`, `standardizer.npz`, `training_history.json`,
`manifest.json` and the exact invocation in `command.txt`. A standalone rescore
records the evaluator's current code version; it does **not** establish the
training code version of an old checkpoint. Retain original training logs and
configuration beside it. Load only trusted joblib checkpoints.

`summarize_runs.py` rebuilds `main_table.csv/json` (one row per run) and
`seed_summary.csv/json` (per-method mean and sample standard deviation). A
single seed has no estimated seed standard deviation, shown as null. It checks
saved prediction/manifest hashes and refuses to combine different protocols,
manifests, feature matrices, token-to-feature metadata, source versions, K
values or precision targets. Core metrics are recomputed from saved predictions
and checked against the evaluation JSON before the table is written.
Within each method, model and training settings must also match across seeds.
Keep paper reference values and historical runs in separate tables.

The test uncertainty estimate resamples complete scenes with replacement. Its
95% percentile intervals are conditional on a fixed model and validation
threshold; they do not capture model-seed or annotation uncertainty. Undefined
AP/AUC values from single-class resamples are counted explicitly. Do not compare
overlapping intervals as a significance test. Run multiple model seeds against
the same manifest, retain each run, and report mean and spread separately.

Include real FP/FN images and investigate their causes before making claims.
Synthetic mistakes only demonstrate that the export and inspection path works.
