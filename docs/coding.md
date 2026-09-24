Coding Style
=============
Welcome to the long-tail driving data mining project coding style page!

The Python code follows [PEP 8](https://peps.python.org/pep-0008/) with the
conventions below applied consistently across the scripts.

## Formatting

- Four-space indentation, 100 columns as the hard limit checked by the linter.
- `from __future__ import annotations` at the top of every module, so type
  hints stay readable on older interpreters.
- Imports in three groups — standard library, third party, then this package —
  separated by a blank line and alphabetised within each group.
- Section banners (`# ---- Data ----`) separate the stages of a long module;
  when a module needs more than about five of them, it is split instead.

## Naming and interfaces

- `snake_case` for functions and variables, `CamelCase` for classes,
  `UPPER_CASE` for module-level constants.
- A leading underscore marks a helper that is not part of a module's public
  surface, such as `_normalise_label`.
- Every public function carries a docstring that states its intent, its
  arguments and the array shapes it expects and returns. Shapes are written as
  `(N, D)` and named consistently: `N` frames, `D` hidden dimension, `P` latent
  width.
- Constants that encode a protocol — `NORMAL_LABELS`, `DROP_LABELS` — live in
  one place and are imported, never re-spelled as literals at the call site.

## Comments

- Comments explain intent, not mechanics: why `uncertain` frames are dropped
  rather than counted, why the standardiser uses training statistics only, why
  the capped reward equals the hinge in the paper up to a constant.
- Anything a future reader would find surprising gets a comment, including
  deliberate deviations and the reason for them.

## Correctness habits

- Arguments are validated where a wrong value would otherwise fail deep inside
  a framework call: `LongTailGuidedSAE` rejects `k > hidden_dim`, the training
  scripts stop when a split does not contain both classes, and `screen.py`
  checks the feature dimension against the run it was trained on.
- Randomness is seeded from a single `--seed` flag, and the full configuration
  is dumped to `metrics.json` next to the checkpoint, so a run can be repeated.
- Long-running stages write incrementally and resume, because a VLM pass over a
  dataset is measured in hours.
- No credentials, absolute paths or machine-specific settings in the source;
  data roots, model directories and every hyper-parameter are flags with
  defaults.

## Experiment hygiene

- Variants that are meant to be compared share their implementation. The three
  SAE entry points hold nothing but a `DEFAULTS` dictionary, so an ablation
  cannot accidentally differ in a second place.
- Anything reported in the README or in `results/` is produced by a script in
  this repository, with the flags given next to it.

## Checking

```bash
make lint          # python -m flake8 scripts tests --max-line-length 100
```
