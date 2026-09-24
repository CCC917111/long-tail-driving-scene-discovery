Common Tasks
=============
Welcome to the long-tail driving data mining project common tasks page!

This page gives the recipes for the changes people most often want to make.

## Change what counts as long-tail

The definition lives in two places that must agree: the rubric in
[`latest_grading_criteria.md`](../latest_grading_criteria.md) and the
classification prompt in [`prompt_templates.md`](prompt_templates.md). Edit
both, re-run the labelling stage, and re-train — nothing in the model or the
training code encodes the taxonomy, so a different definition is a data change.

The feature-extraction prompt is deliberately *not* part of this. It asks for a
scene description, not a judgement, so it stays the same when the definition
changes; that separation is what lets the same extracted features be reused for
several label sets.

## Swap the VLM backbone

`extract.py` loads the model from `--model-dir` and hooks the MLP block output
of the layers in `--layers`. For another Qwen2.5-VL-family model, point it at
the new weights and pick a layer — deeper is generally better, and the last
block is a good first guess. For a different architecture, the hook target is
the one thing to adjust: it needs the residual-stream-adjacent output of a
decoder block, which is `down_proj` in this family.

Then re-extract and re-train. The SAE reads an `(N, D)` matrix and does not
care where it came from, so a new backbone only changes `D`.

## Add an SAE variant

Copy one of the three entry points, change its `DEFAULTS`, and give it a
docstring — that docstring becomes the `--help` text. If the variant needs
behaviour that no flag expresses, add it to `LongTailGuidedSAE` behind a new
choice of `--sparsity` or `--reward` rather than forking the model, so the
variants keep sharing one training loop and one evaluation protocol.

Then add a case to `tests/test_sae_common.py` that pins the new behaviour: the
existing sparsity tests are a good template.

## Tune the sparsity

`--k` is the number of latent units kept per sample and `--hidden-dim` the
latent width; their ratio is the real knob. A smaller `k` forces more
specialisation and usually cleaner single units, at the cost of reconstruction
quality. `--tail-ratio` sets how much of the code is reserved for `z_t`; at
0.5 the two subspaces are the same size.

After a change, look at `neuron_report.csv` before looking at the F1: if the
high-purity units have disappeared, the code has become too dense or too tight
regardless of what the headline metric says.

## Run on a different dataset

The pipeline assumes multi-view frames with a stable identifier per sample:

1. Adapt the metadata loading in `annotate_normal_core.py` and `extract.py`
   to your index format. Both need a mapping from `sample_token` to the image
   paths of its views.
2. Keep writing `meta.json` with a `scene_token` or `scene_name` per row. That
   field is what makes the split scene-level; without it the code warns and
   falls back to a sample-level split, which leaks near-duplicate frames.
3. Everything downstream is dataset-agnostic.

Fewer or more than six views works — the prompt describes the views it is
given — as long as training and inference use the same configuration.

## Use the units as retrieval keys

Name the high-purity units for your run, put them in a CSV in the format of
[`results/neuron_glossary.csv`](../results/neuron_glossary.csv), and filter
`screening.csv` on `top_neurons` instead of on `tail_score`. That turns the SAE
into a categorical search over an unlabelled corpus.
[The Interpretable Neurons](neurons.md) has the full procedure.

## Re-use a trained run on new data

Nothing needs retraining to screen new frames: `screen.py` reads the model, the
standardiser and the threshold from the run directory, so the new frames are
mapped into the space the SAE was fitted in rather than re-centred on
themselves. Extract them with the same layer and pooling, then see
[Screening Your Own Driving Data](apply.md).
