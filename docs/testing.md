Testing
=============
Welcome to the long-tail driving data mining project testing page!

The suite runs on CPU in seconds and needs neither the dataset nor the VLM:
every test works on small synthetic arrays and images. Its job is to pin down
the properties that are easy to break silently — a sparsity rule that quietly
drops negative activations, a split that leaks near-duplicate frames, a label
protocol that counts `uncertain` as long-tail, a decision rule that ignores a
negative activation, an image route that can be walked out of its directory —
rather than to check numbers that depend on initialisation.

## Overall structure

```bash
make test                                # python -m pytest tests -q
python tests/test_sae_common.py          # the SAE library, without pytest
python tests/test_neuron_explorer.py     # rule, store and explorer, without pytest
```

| File | Covers | Needs |
|---|---|---|
| `tests/test_sae_common.py` | sparsity, objective, split, labels, metrics | torch, scikit-learn |
| `tests/test_neuron_explorer.py` | decision rule, activation store, explorer queries, web interface | numpy, Pillow |

Both files work under pytest and as scripts: running one directly executes
every `test_*` function in order and prints one line per test. Everything they
need is in `requirements.txt`.

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

### Decision rule and metrics

- **One active unit is enough.** `c_tail` counts the units with `|z_t| > eta`
  and a frame is long-tail when the count is at least one; a frame with no
  active unit is normal.
- **Sign does not matter, the threshold is strict.** A negative activation
  counts as active, and an activation exactly at `eta` does not.
- **A quiet tail subspace means normal.** With the encoder pinned so that
  AbsTopK keeps only `z_n` units, the frame is normal; move one strong
  activation into `z_t` and it becomes long-tail.
- **The metrics score the rule.** Precision and recall are computed from the
  count rule's predictions and AUC from `||z_t||_2`, checked on a hand-worked
  example; both classes are reported with the right support.

### Activation store and explorer

- **The store round-trips exactly.** Writing `z_t` sparsely and reading it back
  gives the same matrix, the same counts, labels, splits and image paths, also
  when the last frames have no active unit.
- **Both lookup directions are ordered by magnitude.** A unit's frames come
  strongest first regardless of sign; a frame's units likewise, with the
  firing threshold marked.
- **Purity is computed per unit** from the labels.
- **Unknown tokens and out-of-range units are errors,** not empty answers.
- **Image paths cannot leave `--samples-root`,** whatever `meta.json` says.
- **The web interface answers both directions,** serves the page, the JSON
  endpoints and JPEG thumbnails, and returns 404 for an unknown frame.

## What is not covered

The two VLM stages need model weights and a GPU, so they are exercised by
running them rather than by unit tests; `--max-samples` makes a smoke test
cheap. `screen.py` is a thin composition of `LongTailGuidedSAE.encode`, the
decision rule and the activation store, all of which are tested; its own
failure modes are the explicit dimension and row-count checks, which report the
problem instead of producing a silently wrong list.

## Adding a test

Add a `test_*` function to the file that matches what it tests, keep it free of
dataset and VLM access, and prefer a property over a hard-coded number: assert that a
negative activation survives, that a scene cannot straddle a split, that an
`uncertain` row is dropped — not that a metric equals a number produced by
today's random seed.
