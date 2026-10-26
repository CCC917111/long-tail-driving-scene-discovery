# Final user review after AI-label exposure

Yike submitted decisions for all **80 six-camera timestamps** on 2026-10-26:
**77 binary judgments** (12 normal, 65 tail), **three uncertain**, zero pending.
This is **single-reviewer human review after the full AI-label table was shown**.
Metrics describe agreement with those recorded judgments; they do not establish
independent blind human ground truth or an official benchmark result.

All four CPU models were rerun using the same pinned ResNet18 cache, unchanged
6/2/2-scene allocation and carried-over seed-42 settings. Checkpoint reloads,
train-only normalization, saved-score evaluation and table reconstruction passed
an independent audit. The original AI pilot remains unchanged.

## Data and review evidence

| Frozen split | Scenes | Normal | Tail | Uncertain | Included |
|---|---:|---:|---:|---:|---:|
| Train | 6 | 3 | 45 | 0 | 48 |
| Validation | 2 | 4 | 10 | 2 | 14 |
| Test | 2 | 5 | 10 | 1 | 15 |

All 80 rows declare reviewer `Yike`. All notes are blank, including the three
uncertain reasons; no explanation has been invented. The uploaded bytes are
preserved in [reviewed-original.csv](reviewed-original.csv), SHA256
`c39de8776325f6d72ee22738dcf2c95ee653db439f416de0a4e9403ea8f8a43a`.
[reviewed-with-context.csv](reviewed-with-context.csv) adds only the assistant's
`review_context=post_ai_exposure_single_reviewer` marker; the five user fields
are unchanged. Its hash and known exposure are in
[review_evidence.json](review_evidence.json).

[first10-before-ai-table.csv](first10-before-ai-table.csv) preserves the earlier
partial submission. Four of those decisions changed in the final file: page
frames 1, 4 and 10 changed normal to tail; frame 9 changed normal to uncertain.
The earlier file is not overwritten or retroactively labelled post-exposure.
The final file, including its first ten rows, carries the final exposure scope.

## Observed results

**Seed 42 · 10 epochs · hidden width 256 · AbsTopK 32 · review budget K=4**

| Method | Test AP | ROC-AUC | Precision@4 | Recall@4 | Test operating precision / recall |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.6143 | 0.4000 | 0.50 | 0.20 | no candidates / 0 |
| Small MLP | 0.5984 | 0.3200 | 0.50 | 0.20 | no candidates / 0 |
| AbsTopK SAE, no reward | 0.9353 | 0.8400 | 1.00 | 0.40 | 1.00 / 0.20 |
| AbsTopK SAE, norm reward | 0.9353 | 0.8400 | 1.00 | 0.40 | 1.00 / 0.20 |

The operating threshold maximizes validation recall subject to empirical
validation precision >=90%, with the existing tie rules. Logistic has no
qualifying threshold and predicts none. MLP selects threshold
`0.9992990493774414` on one validation positive, but selects no test candidates.
The SAEs select threshold `6.074954873387325`: validation precision 1.00 and
recall 0.30; applying it unchanged to test produces **two TP, zero FP, eight FN
and five TN**. Its precision 1.00 rests on only two selected timestamps.

The SAE's separate top-four policy finds four of ten reference positives.
Uniform random sampling of four from these 15 frames has an analytic expectation
of 2.67 positives; this is not an executed random experiment or measured time
saving. The original SAE activation rule instead flags all 15 test frames,
with precision 2/3, recall 1 and F1 0.80. These three policies are kept separate.

## What changed from the original AI pilot

The reference labels and retained validation/test IDs changed, while the feature
cache and training labels did not. These are **different evaluation references**,
so a change from the old AP is not evidence that the model learned a better
representation.

- Logistic checkpoint bytes match the original AI pilot.
- Both new SAE checkpoints match one another **and the original pilot**:
  SHA256 `250a475bf430ade3a9a4381430a1b4cee366a01a2448747602e2aff76baabbcb`.
  No incremental reward benefit was observed. The earlier capped-reward
  saturation diagnosis remains relevant; no reward cap was tuned on test.
- The MLP's unchanged ten-epoch trajectory selects epoch five instead of four
  under the new validation references. Its selected checkpoint therefore differs.

Train has only three normal frames, all from one scene; test has only two scenes.
One seed, this class/scene coverage and prior AI exposure sharply limit inference.
Scene-bootstrap intervals condition on the fixed model/threshold and cannot
create more independent scene evidence. There is no cross-dataset, Cosmos,
vehicle-safety or independent-human accuracy claim.

## Inspect and reproduce

- [main_table.csv](main_table.csv) / [main_table.json](main_table.json): one row per run.
- Each method folder retains `evaluation.json`, `predictions.json/.csv`,
  `run_metadata.json` and available training history/legacy metrics.
- [manifest.json](manifest.json), [import_metadata.json](import_metadata.json)
  and [handoff_validation.json](handoff_validation.json) bind labels, scene
  allocation, review scope and feature identity. Every active prediction row
  and metric bundle retains the post-exposure scope.
- [independent_audit.json](independent_audit.json) and
  [checkpoint_audit.json](checkpoint_audit.json) record actual reload errors,
  hashes, scaler checks, metrics, PR/threshold/bootstrap and table reconstruction.
- [failure_cases.json](failure_cases.json) selects three illustrative errors
  under the same MLP fixed-0.5 rule, relative to the final recorded judgments.
- [uncertain-screening.csv](uncertain-screening.csv) retains scores for all three
  excluded timestamps. Their tokens are absent from fitting/validation/test, but
  scenes can overlap. This is inference smoke, not a new generalization metric.
- [verification.json](verification.json) separates the local full selection,
  later provenance-guard tests and remote CI status.

The actual training source is preserved as
[training_source_snapshot.zip](training_source_snapshot.zip). Run metadata records
its hash and the dirty-source digest. Current repository code includes later
hardening that rejects a prediction-supplied label scope contradicting the
manifest; it does not change the model or metric math. The audit used preserved
training sources and confirms the completed artifacts already have truthful
scope. No run metadata or source snapshot was rewritten to claim later code.

With authorized local assets and the frozen preparation/cache, replay in fresh
output directories using the preserved context CSV:

```bash
python scripts/prepare_driving.py import-labels \
  --prepared-dir output/real-cpu-mini-20261023/prepared \
  --reviews results/human-ai-exposed-20261026/reviewed-with-context.csv \
  --output-dir output/human-review-reproduction/annotations

python scripts/run_real_driving.py train \
  --feature-protocol cpu-resnet18 --label-protocol manual \
  --prepared-dir output/real-cpu-mini-20261023/prepared \
  --annotations-dir output/human-review-reproduction/annotations \
  --extract-dir output/real-cpu-mini-20261023/extract --layer 1 --pooling mean \
  --output-dir output/human-review-reproduction/comparison_runs \
  --review-k 4 --seed 42 --epochs 10 --hidden-dim 256 --k 32
```

Exact executed commands/settings are in [commands.json](commands.json),
[runner_config.json](runner_config.json) and [protocol_plan.json](protocol_plan.json).
No raw image bytes, original metadata tables, feature arrays or weights are
published in this bundle. Obtain data under nuScenes terms. The downloadable
[offline HTML demo](../../demo/README.md) still replays the separately labelled
**original AI pilot**, not these new reviewed-reference runs.
