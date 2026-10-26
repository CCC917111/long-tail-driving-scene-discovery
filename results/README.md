# Results and evidence status

No real-data result has been established under the original VLM/Cosmos
v1.0-trainval scene-disjoint 70/15/15 protocol. The historical numbers below
are reports retained from `b57eaffa`.
They have not been regenerated from original predictions, logs or checkpoints
in this round. New controlled results must be published in a separate table
with one identifiable run per row.

## Latest mini CPU run: post-AI-exposure human review

The [2026-10-26 record](human-ai-exposed-20261026/README.md) records Yike's
completed 80-row human CSV: 77 reviewed binary labels (12 normal / 65 tail),
three uncertain and no pending rows. The final submission followed exposure to
the full AI label table and has one reviewer; it is not blind independent ground
truth. The original first-ten pre-exposure checkpoint and its four subsequent
label revisions are retained. All notes are empty, including uncertainty
reasons; no reasons have been invented. The raw final CSV SHA-256 is
`c39de8776325f6d72ee22738dcf2c95ee653db439f416de0a4e9403ea8f8a43a`.
The 77 eligible rows retain the frozen internal mini 6/2/2-scene allocation.
All four CPU fits are now complete with the original ResNet18 feature cache
and unchanged training settings; the linked record contains the independently
reconstructed comparison. On 15 test timestamps from two scenes, both SAE
variants have AP 0.9353, precision@4 1.0 and recall@4 0.4. Their validation-selected
operating point has test precision 1.0 and recall 0.2. The SAE checkpoints are
byte-identical to each other and to the earlier AI pilot: no reward benefit
was observed. These are agreement results against one reviewer's judgments
after AI-label exposure, with only three normal training frames and one seed.
The fit-time source snapshot is preserved; the current evaluator also contains
a later provenance guard, recorded separately from the fitted runs.

The earlier [CPU mini AI-label pilot](cpu-mini-ai-pilot-20261023/README.md)
contains real-image CPU results against provisional AI references, with 73
binary decisions and seven uncertain exclusions. Its results retain that
scope and are separate from this human-review submission and the historical
VLM/Cosmos reports below.

## Original VLM/trainval 70/15/15 protocol: no real-data results yet

The planned comparison shares frozen VLM features, one scene-disjoint manifest,
train-only normalisation and validation-only model/operating-point selection:
Logistic regression, small MLP, AbsTopK without reward and the same AbsTopK with
reward. The [evaluation protocol](../docs/evaluation.md) defines metrics,
predeclared review budget, scene uncertainty and reproducible table artifacts.
Synthetic outputs demonstrate this software path and are not nuScenes results.

## Historical protocol and unresolved counts

The report describes 6,586 nuScenes v1.0-trainval keyframes: 2,356 long-tail
and 4,230 normal, selected from 824 scenes. It also describes eight keyframes
per scene, which would total 6,592. The six-frame difference remains unexplained
without the historical manifest and exclusion records. Missing-view filtering
exists in the code but has not been verified as the cause of this difference.

The reported SAE split was random **by frame**, 80/20: 5,268 training and
1,318 held out (486 tail / 832 normal). The checkpoint with the lowest held-out
loss was selected, so this was not an independent validation/test protocol.
Frames from one scene could straddle the split. These results do not transfer
to the current scene-disjoint 70/15/15 protocol.

The reported labels were VLM pre-labels followed by manual review. The original
labels and review records are absent from this checkout and need recovery.
VLM confidence or agreement is not independently measured human-label accuracy.

## Historical SAE configuration

| Method | Tail precision | Tail recall | Tail F1 | Normal precision | Normal recall | Normal F1 | AP | ROC AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cosmos-Reason1-7B + AbsTopK SAE, L28 | 0.9023 | 0.7984 | 0.8472 | 0.8896 | 0.9495 | 0.9186 | 0.9082 | 0.9063 |

The original decision was at least one tail unit with `|z_t| > 0.01`.
AP and ROC AUC used `||z_t||_2`. The six class precision/recall/F1 values are
arithmetically compatible with TP=388, FN=98, FP=42, TN=790 under the stated
supports. This compatibility does not verify the predictions or recover the
run. Historical AP/AUC cannot be reconstructed from this confusion matrix.

## Historical raw-VLM reports: comparison not verified

| Raw VLM | Tail precision | Tail recall | Tail F1 | Normal precision | Normal recall | Normal F1 |
|---|---:|---:|---:|---:|---:|---:|
| Cosmos-Reason1-7B | 0.1429 | 0.0130 | 0.0238 | 0.8838 | 0.9897 | 0.9338 |
| Qwen3.5-9B | 0.2900 | 0.3580 | 0.3204 | 0.9103 | 0.8815 | 0.8957 |
| Qwen3-VL-2B-Instruct | 0.3611 | 0.1135 | 0.1728 | 0.8843 | 0.9712 | 0.9257 |

None of these rows corresponds to an integer binary confusion matrix with
486 tail and 832 normal supports when rounded to the reported four decimals.
The evaluation IDs, aggregation method or transcription may differ. Recover
and rescore the original per-frame predictions before claiming a matched-test
recall gain, common ground truth or causal benefit from the SAE.

## Historical configuration and layer comparisons

| Configuration | Layer | ROC AUC | AP | Tail F1 | Tail precision | Tail recall |
|---|---|---:|---:|---:|---:|---:|
| AbsTopK + norm reward | L28 | 0.9063 | 0.9082 | 0.8472 | 0.9023 | 0.7984 |
| AbsTopK + norm reward | L27 | 0.9033 | 0.9068 | 0.8386 | 0.9212 | 0.7695 |
| AbsTopK + norm reward | L24 | 0.9004 | 0.8999 | 0.8294 | 0.9051 | 0.7654 |
| AbsTopK + norm reward | L21 | 0.8772 | 0.8120 | 0.8043 | 0.8374 | 0.7737 |
| AbsTopK + norm reward | L18 | 0.8733 | 0.8118 | 0.7868 | 0.7876 | 0.7860 |
| AbsTopK + norm reward | L15 | 0.8791 | 0.8829 | 0.8091 | 0.9100 | 0.7284 |
| Signed TopK, no reward | L28 | 0.8735 | 0.8846 | 0.8269 | 0.9669 | 0.7222 |
| “Train SAE Cosmos” / pre-selection reward | L28 | 0.8657 | 0.8623 | 0.8071 | 0.8750 | 0.7490 |

These are historical configuration reports, not controlled single-factor
ablations. The TopK row changes both sparsity and reward. The pre-selection
variant also changes latent width (3,584 vs 7,168), dropout (0.4 vs 0.2),
normal penalty (0.01 vs 0.1) and reward settings. Therefore the 0.0401 F1
difference cannot establish the isolated effect of adding a tail reward.
Layer results are not monotonic; L28 is the highest reported F1 among these
rows. Selecting layers requires validation only in new experiments.

## Historical interpretation and agreement

| Interpreted feature | Unit | Reported activation ratio | Reported purity |
|---|---:|---:|---|
| Rain / wet road | 2169 | 43.8% | 100.0% (723/723) |
| Glare on a wet or rainy road | 396 | 5.6% | 100.0% (92/92) |
| Wheelchair user ahead | 3058 | 42.4% | 77.8% (14/18) |
| Pedestrians crossing a construction zone | 1542 | 6.1% | 53.8% (71/132) |
| Truck close to the ego vehicle | 1843 | 4.2% | 57.1% (12/21) |

These reports use `|z_t| > 1.0`. Their sample memberships, scenario labels and
exploration/test boundaries are not available here. They are not independent
test purity/coverage estimates. The [glossary](neuron_glossary.csv) and
[example figure](../docs/figures/neuron_examples.webp) preserve interpretation
material; names must be frozen on train/validation before test measurement.

The historical report also lists 8,489 / 10,120 agreement between two VLM
annotators and a wheelchair case in `scene-0675`. The annotation records,
KNN outputs and repeated VLM trials are missing. Preserve those as historical
unverified observations, without claiming accuracy against human truth or a
systematic advantage over alternative mining methods.

## Rebuild new results

Recover authorised features, metadata, labels and extraction settings first.
Use the [user guide](../docs/user.md) to freeze a manifest, then the
[evaluation workflow](../docs/evaluation.md) to write per-sample predictions,
metrics, provenance and reconstructable main tables. Keep original artifacts
unchanged and new runs in distinct directories. A current-code rerun is a new
experiment, not a reconstruction of missing historical provenance.
