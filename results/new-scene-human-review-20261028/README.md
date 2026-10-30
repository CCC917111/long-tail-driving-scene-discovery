# Supplemental new-scene human review

Yike submitted all nine groups on 2026-10-28: five normal_core, four not_normal_core, no
uncertain or pending rows. All notes are blank. The original 1115-byte CSV is preserved
byte-for-byte, SHA256
`b470eab6e17eaa0e6e2cc93a52ddeefd2d6b4ce1c9eff58fcc83da619a3653d2`.

The model-generated reference table was disclosed before submission. The page fields
`model_hidden_single_reviewer` and `model_signal_exposed=false` describe only whether
the browser showed model signals. They do not establish blindness to the earlier
reference labels. Per-row judgment timing is unknown. `provenance.json` records this
separately without modifying the submitted file.

All nine decisions agree with the provisional model-generated visual references. This is reference
agreement, not an estimate of annotation-model accuracy.

The previously frozen SAE rule, `abs(z_t) > 0.01` for at least one tail unit, flags all
nine: TP=4, FP=5, TN=0, FN=0 relative to this submission. Precision is 44.44%, recall
100%, F1 61.54%; no review items are removed by this rule. No training, threshold
tuning, formal train/validation/test allocation or independent benchmark evaluation was
performed for this review batch.

All five normal references belong to scene 54cdaaae372d421fa4734d66f51a8c48; all four
non-normal references belong to scene f0f120e4d4b0441da90ec53b16ee169d. Class and scene
are completely confounded in this author-selected batch. This batch cannot establish
generalization performance.

The directory preserves the original CSV, unchanged notes in
`normalized_references.json`, separate model-generated visual references, the nine-row comparison,
audit and provenance records, and `import_and_compare.py` for rebuilding the comparison
from the original inputs. No original images or model weights are committed in this
evidence record.

The preceding ingestion/inference record is in [the original E2E
record](../new-scene-e2e-20261027/README.md). Its no-human-label statements describe
execution time, before this later submission.

## Replay of policies frozen before these new scenes

The SAE L2-score operating threshold **6.074954873387325** was selected on the old
validation split, not these nine labels. Its selection is independently visible in
immutable public commit
[`fb1f2f93`](https://github.com/CCC917111/long-tail-driving-scene-discovery/blob/fb1f2f93ad46958b2994a7f939b4b39c76ffa476/results/human-ai-exposed-20261026/sae_abstopk_none/evaluation.json),
committed 2026-10-26, before new-scene inference on 2026-10-27 and this review. The
original budget K=4 is carried over unchanged from the earlier protocol; it was not
chosen to improve these new results.

| Saved policy | Review candidates | TP / FP / TN / FN | Precision | Recall | F1 |
|---|---:|---|---:|---:|---:|
| Any active tail unit, eta=0.01 | 9 | 4 / 5 / 0 / 0 | 44.44% | 100% | 61.54% |
| Old validation-selected L2 threshold | 5 | 3 / 2 / 3 / 1 | 60.00% | 75.00% | 66.67% |
| Original fixed budget, top 4 scores | 4 | 2 / 2 / 3 / 2 | 50.00% | 50.00% | 50.00% |

The frozen threshold retains groups **4, 5, 7, 8 and 9**. Groups 4 and 5 are false
positives relative to this submission, and group 6 is a false negative. It leaves four
of nine items unselected and recovers three of four tail references. This measures queue
size; human review time was not measured. Its new-batch precision is 60%, so the old
validation 90% precision target is not guaranteed under this small scene shift. None of
these policies was selected or tuned on the new labels; all three are reported so their
tradeoffs can be compared. Random selection of four from this batch has an expected 1.78
positive references, compared with two in the saved ranking; that expectation is not an
additional experiment or proof of a stable gain.

Both recorded SAE variants have the same checkpoint, original standardizer, threshold
and replayed decisions. They are not independent repeated experiments, and this provides
no reward benefit claim. Actual private checkpoint and standardizer hashes were checked
against the new inference record. The class/ scene confound and post-disclosure review
scope above still apply.

Artifacts:

- [frozen_policy_public_source.json](frozen_policy_public_source.json): immutable
  published policy, commit timestamp and source link.
- [frozen_policy_evaluation.json](frozen_policy_evaluation.json): input hashes, checked
  model identities, policy settings, scope and all metrics.
- [frozen_policy_evaluation.csv](frozen_policy_evaluation.csv): all nine saved scores,
  references and predictions under each policy.
- [replay_frozen_policy.py](replay_frozen_policy.py): standard-library replay; no model
  fitting, threshold search, new split or external API call.

From the repository root, reconstruct the metrics into a separate output:

```bash
python results/new-scene-human-review-20261028/replay_frozen_policy.py \
  --output-prefix output/frozen-policy-replay/policies
```

When the original private run is available, add `--private-training-root
output/real-cpu-mini-20261023/human-ai-exposed-20261026/comparison_runs` to check actual
checkpoint/standardizer bytes as well. The public replay only requires saved numeric
outputs and references; private weights are not included in the repository.
