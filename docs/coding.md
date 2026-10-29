# Coding conventions

Python code follows [PEP 8](https://peps.python.org/pep-0008/). Use these
conventions when extending the scripts.

## Formatting and names

- Use four spaces and a 100-column line limit, as checked by the linter.
- Add `from __future__ import annotations` to Python modules.
- Group imports as standard library, third party and local modules, with a
  blank line between groups and alphabetical order within each group.
- Use `snake_case` for functions and variables, `CamelCase` for classes and
  `UPPER_CASE` for constants. Prefix private helpers with an underscore.
- Keep protocol constants such as `NORMAL_LABELS` and `DROP_LABELS` in one
  shared definition.

Public functions should document their purpose, arguments and expected array
shapes. Use consistent shape names: `(N, D)` for frames and feature dimension,
and `P` for latent width. Split long modules by responsibility when section
banners stop making them easy to navigate.

## Comments and validation

Explain choices a reader cannot infer from the code: why uncertain labels are
excluded, why scaling uses training statistics only, or how a reward relates
to a hinge penalty. Avoid comments that repeat an assignment or loop.

Validate inputs at the boundary. Reject invalid dimensions, duplicate tokens,
nonfinite features, missing scenes and incompatible feature identities before
starting a fit or loading a screening model. Preserve exclusions and failure
reports so a later reviewer can reconstruct what happened.

Keep configuration in command-line flags and recorded run files. Do not commit
credentials or machine-specific paths. Seed randomness through `--seed` and
save the effective configuration alongside the checkpoint. Long extraction
stages should checkpoint incrementally and reject incompatible resume settings.

## Experiment records

Comparable variants should share the same training implementation. The SAE
entry points provide `DEFAULTS`; the controlled reward comparison uses the
same entry point and flags except for reward and output directory. Record the
exact commands, frozen manifest, feature provenance and evaluation artifacts
for any reported run. Shared code alone does not make different default
configurations a controlled ablation.

Avoid importing torch in utilities that only need numpy and Pillow. The
activation store and explorer can inspect saved runs without torch. The
explorer uses the standard-library HTTP server and a self-contained page, so
it needs no front-end build.

## Lint

```bash
make lint          # python -m flake8 scripts tests --max-line-length 100
```

[`.flake8`](../.flake8) contains the configuration. The files containing long
embedded prompts or HTML have line-length exemptions; other lint rules still
apply.
