Testing
=============
Welcome to the long-tail driving data mining project testing page!

The suite runs on CPU in seconds and needs neither the dataset nor the VLM:
every test builds a small SAE on synthetic arrays. Its job is to pin down the
properties that are easy to break silently — a sparsity rule that quietly drops
negative activations, a split that leaks near-duplicate frames, a label
protocol that counts `uncertain` as long-tail — rather than to check numbers
that depend on initialisation.

## Overall structure

```bash
make test                            # python -m pytest tests -q
python tests/test_sae_common.py      # the same tests, without pytest
```

The file works both under pytest and as a script: running it directly executes
every `test_*` function in order and prints one line per test. It needs `torch`
and `scikit-learn`, which `requirements.txt` already installs, plus `pytest`
for the first form.

## Unit tests

### Sparsity

- **AbsTopK keeps exactly `k` units.** The code has at most `k` non-zero
  entries and the two subspaces together account for the full latent width.
- **AbsTopK keeps large negative activations.** With the encoder pinned to a
  known bias, a strongly negative unit survives AbsTopK and is dropped by
  signed Top-K. This is the whole difference between the final method and the
  `topk` ablation, expressed as one assertion.
- **`k > hidden_dim` is rejected.** A configuration error should fail in the
  constructor, not inside `torch.topk`.

### Objective

- **The tail reward only fires for tail samples, and lowers the loss.** With
  `reward="norm"` the reward term is non-positive; with `reward="none"` it is
  exactly zero for the same weights and inputs.
- **The normal penalty only applies to normal samples.** An all-tail batch has
  a zero penalty term; an all-normal batch with the same weights does not.

### Split

- **Scenes stay together.** No scene may appear in two splits, and every sample
  is assigned exactly once. This is the property that keeps near-identical
  keyframes of one scene from straddling the train/test boundary.
- **The split is stratified over long-tail samples.** Each split ends up with a
  long-tail share in the same range, which is what the greedy assignment in
  `split_indices` exists to achieve: long-tail frames cluster inside scenes, so
  a naive random assignment of scenes produces very uneven splits.

### Labels

- **The label protocol.** `normal` and `normal_core` map to 0, anything else
  maps to 1, and `uncertain` / empty map to `None` so the row is dropped rather
  than counted as long-tail. Both accepted label formats are exercised,
  including the nested `{"labels": {"label": ...}}` form.

### Thresholding

- **The threshold lands where F1 peaks,** and the metrics computed at it are
  the expected perfect scores on a separable toy problem.
- **Both classes are reported,** with the right support, so a future change to
  the metric block cannot silently drop the normal class.

## What is not covered

The two VLM stages need model weights and a GPU, so they are exercised by
running them rather than by unit tests; `--max-samples` makes a smoke test
cheap. `screen.py` is covered indirectly — it is a thin composition of
`LongTailGuidedSAE.encode` and the stored run configuration — and its failure
modes are the explicit dimension and row-count checks, which report the problem
instead of producing a silently wrong ranking.

## Adding a test

Add a `test_*` function to `tests/test_sae_common.py`, keep it free of dataset
and VLM access, and prefer a property over a golden value: assert that a
negative activation survives, that a scene cannot straddle a split, that an
`uncertain` row is dropped — not that a metric equals a number produced by
today's random seed.
