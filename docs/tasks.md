# Common tasks

Use the [user guide](user.md) for commands and the
[evaluation protocol](evaluation.md) for controlled comparisons. Historical
configuration comparisons are documented separately in
[results](../results/README.md).

## Recover and freeze a study

Keep original labels, review records, features, predictions and logs unchanged.
Recover `sample_token` and `scene_token`, reconcile duplicate/conflicting
labels, and record exclusions before freezing a manifest. The current loader
requires a nonempty `scene_token` for every retained example; it never falls
back to sample-level splitting. `scene_name` is display metadata.

Use `--freeze-manifest-only --review-k K` once, inspect actual scene/frame counts
and distributions, then pass that `--manifest` to every method and seed. Changing
a model seed must not silently change the split. Record model revision, layer,
pooling, prompt/rubric versions and input hashes in recovered extraction
settings; supply `--feature-config` rather than inferring those settings from
feature dimensions.

## Change the long-tail definition

Keep the [rubric](../latest_grading_criteria.md) and
[classification prompt](prompt_templates.md) consistent. New labels and a new
rubric version create a new study; keep the old manifest/results intact.
The loader accepts only the [label whitelist](user.md#strict-data-contract),
with unknown/unusable labels excluded and unrecognised strings rejected.

The extraction prompt requests a scene description. Features can be reused
only if the extraction contract is unchanged; test whether they suit a new
rubric instead of assuming the original performance carries over.

## Compare reward and sparsity

The core reward comparison uses `sae_abstopk_tail_reward.py` twice with the
same architecture, optimiser, seed, features and manifest, changing only
`--reward none` versus `--reward norm`. Match review K, target precision,
training budget and checkpoint-selection rule across all compared runs.

The legacy `ablation_topk_sae.py` changes both signed/absolute selection and
reward. `ablation_sae_cosmos_baseline.py` additionally changes width, dropout,
normal penalty and reward settings. Running `make ablations REVIEW_K=K` does
not make these single-factor experiments. For a later sparsity-only comparison,
hold reward and all other settings fixed and retain both exact configs.

Changing `--k`, latent width or tail ratio is a validation experiment. Inspect
reconstruction, dead units, activation coverage and real errors alongside
ranking/review metrics; semantic interpretation is not established by high
training-label purity alone.

## Swap a backbone or camera rig

For another supported VLM, set `--model-dir`, choose a decoder layer using
validation only, and capture a documented hook output and pooling. A new
architecture may need a different hook. Re-extract and train a new run; the
same feature dimension does not imply compatible representations.

For another dataset, adapt metadata loading in `annotate_normal_core.py` and
write a stable sample identifier and scene token per row. Keep a whole temporal
scene in one split. For a new rig, update camera lists, their fixed order and
both prompts before new extraction. Do not apply old six-camera weights to
features from an incompatible rig.

## Name and inspect units

Explore units on train/validation only, recording scenario labels, reviewers,
thresholds and evidence samples. Freeze names and thresholds before measuring
test purity/coverage. A new seed, layer or latent width needs a new glossary;
the historical glossary is not a universal category mapping. See
[the explorer guide](neurons.md).

For a demonstration, show a frame's score and active units, compare high-unit
examples, inspect a real false positive or false negative, then export the
review queue. Synthetic images and weights must be labelled as synthetic.

## Change a decision or reuse a run

The original count rule lives in
[`tail_activations.py`](../scripts/tail_activations.py). Changing it requires a
new protocol/version; do not retune it after looking at test outcomes. The
validation-selected score operating point is reported separately. The existing
`screen.py` uses the fixed activation rule; preserve that distinction when
presenting a screening list.

For new data, use the trained run's standardiser and matching extraction
contract. Do not re-centre on the new evaluation corpus. Retain original
config/weights and verify hashes; [screening](apply.md) describes the inputs.
