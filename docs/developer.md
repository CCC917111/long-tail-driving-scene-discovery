Developer page
=============
Welcome to the long-tail driving data mining project developer main page!

This page is the entry point for working on the code: how the repository is
laid out, how a frame travels through it, and where to start when you want to
change something.

## Abstract

The project mines long-tail autonomous-driving data from the hidden
representations of a frozen vision-language model. A behaviour-oriented rubric
defines what counts as long-tail, a VLM applies that rubric at scale to produce
weak labels, a second VLM pass captures hidden states, and a tail-guided sparse
autoencoder decomposes those states into a normal subspace and a
long-tail-sensitive subspace whose activation is the detector. The same trained
model then screens unlabelled data, and the sparse units it learned double as
named, reusable descriptions of specific risk patterns.

## Layout

```
scripts/
  annotate_normal_core.py         VLM labelling against the normal-core rubric
  extract.py                      per-keyframe, per-layer hidden-state extraction
  sae_common.py                   model, split, training loop, evaluation protocol
  sae_abstopk_tail_reward.py      final method  (AbsTopK + tail reward)
  ablation_topk_sae.py            ablation     (signed Top-K, no reward)
  ablation_sae_cosmos_baseline.py ablation     (reward before Top-K selection)
  pipeline.py                     clip-level variant, DeepLongTailSAE
  screen.py                       apply a trained run to your own frames
docs/                             these pages, the prompts and the figures
tests/                            unit tests for the shared SAE library
results/                          reported metrics and the neuron glossary
latest_grading_criteria.md        the normal-core annotation rubric
```

## Data flow

```
six camera views of a keyframe
  -> annotate_normal_core.py   rubric prompt + answer-token logits
                               -> labels.json   {sample_token, label, confidence}
  -> extract.py                scene-description prompt, forward hook on layer L
                               -> layer{L}_mlp_output_mean.npy  (N, D)
                               -> meta.json                     (N, sample_token, scene)
  -> sae_common.load_dataset   join features + labels, drop uncertain rows
  -> split_indices             scene-level, greedily stratified
  -> standardise               training-split mean/std only
  -> LongTailGuidedSAE         z = AbsTopK(f_enc(h)) = [z_n, z_t]
  -> ||z_t||_2                 score; threshold fixed on validation
  -> metrics.json, neuron_report.csv, best_model.pth, standardizer.npz
  -> screen.py                 same model, new frames, no labels
                               -> screening.csv  (rank, score, units, reasons)
```

The two VLM stages are the expensive ones and both write incrementally, so an
interrupted run resumes instead of starting over. Everything downstream of the
`.npy` files is cheap and CPU-friendly, which is what makes the layer sweep and
the ablations practical.

## Where to start

| You want to | Start in |
|---|---|
| Change the model or the objective | `sae_common.py`, class `LongTailGuidedSAE` |
| Change the split or the metrics | `sae_common.py`, `split_indices` / `classification_metrics` |
| Add an SAE variant | A new entry point with its own `DEFAULTS` |
| Change what counts as long-tail | `latest_grading_criteria.md` and the prompts |
| Change the VLM or the layer | `extract.py`, then re-train |
| Apply the method to new data | `screen.py`, [Screening Your Own Driving Data](apply.md) |
| Understand a design decision | [High-level Design](design.md) |
| Name the learned units | [The Interpretable Neurons](neurons.md) |

## Conventions

The code follows the [Coding Style](coding.md) page: four-space indentation,
docstrings that name array shapes, argument validation where a wrong value
would otherwise fail deep inside a framework call, no credentials or
machine-specific paths in the source, and every experimental knob exposed as a
flag with a default.

Before opening a change, run:

```bash
make test
make lint
```

New behaviour comes with a test in `tests/`; the [Testing](testing.md) page
explains what the suite already covers and what kind of test fits here.
