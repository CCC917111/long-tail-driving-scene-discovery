The Interpretable Neurons
=============
Welcome to the long-tail mining project neuron page!

This is the part of the project that makes the difference between a detector
and a tool. A long-tail detector answers "is this frame worth looking at?". The
sparse autoencoder answers that *and* which of its latent units fired — and
several of those units turn out to stand for a specific, nameable driving
situation. The shortlist therefore comes with a reason attached, and the units
themselves can be used as retrieval keys.

## Why single units mean anything at all

Nothing forces a neural network to put one concept in one unit; in a dense
representation, concepts are typically spread across many directions at once.
Two properties of this SAE push against that:

- **AbsTopK sparsity.** Only the `k = 512` latent units with the largest
  absolute activation survive per sample, out of a much wider latent layer. A
  unit that fires on everything is not useful under that budget, so units
  specialise.
- **The tail-guided objective.** The long-tail subspace `z_t` is pushed down on
  normal samples and pushed up on long-tail samples, so the specialisation that
  happens inside `z_t` is specialisation *towards long-tail structure* rather
  than towards whatever happens to reconstruct the data best.

The result is checkable rather than assumed: take a unit, take every frame
where `|z_t| > 1`, and look at them.

## How a unit is scored

Every SAE run writes `neuron_report.csv` with one row per `z_t` unit:

| Column | Meaning |
|---|---|
| `z_t_neuron` | Index of the unit inside `z_t` |
| `n_active` | How many frames activate it |
| `n_active_long_tail` | How many of those are long-tail |
| `purity` | `n_active_long_tail / n_active` — when it fires, how often is it right |
| `long_tail_coverage` | Share of all long-tail frames it fires on |

Purity and coverage pull in opposite directions, and both matter. A unit with
high purity and low coverage is a precise detector of one narrow situation; a
unit with high coverage and low purity is a broad "something is unusual"
signal. The units worth naming are the high-purity ones, and a *group* of them
is what covers a whole scenario family.

## The named units

Naming is the one manual step: read the top-activating frames of a
high-purity unit and write down what they have in common. These are the units
named on the nuScenes run at layer 28, and they are also the contents of
[`results/neuron_glossary.csv`](../results/neuron_glossary.csv), which
`screen.py` joins against so that a flagged frame carries a reason:

| Unit | Feature | Activation ratio | Purity |
|---:|---|---:|---:|
| 2169 | Rain / wet road | 43.8% | 100.0% (723/723) |
| 396 | Glare on a wet or rainy road | 5.6% | 100.0% (92/92) |
| 3058 | Wheelchair user ahead | 42.4% | 77.8% (14/18) |
| 1542 | Pedestrian in a construction zone | 6.1% | 53.8% (71/132) |
| 1843 | Truck close to the ego vehicle | 4.2% | 57.1% (12/21) |

Two of these are more interesting than "the model learned to see rain".

**Unit 396 is not a brightness detector.** It fires on glare specifically
caused by rain or a wet road surface, and it keeps doing so on different parts
of nuScenes. A plain brightness or exposure feature would fire on any strong
light source; this one is tied to a compound of weather and illumination, which
is exactly the kind of condition that degrades perception.

**Unit 1542 is a conjunction.** It fires on pedestrians *in* construction
areas — not on pedestrians alone, not on construction alone. A rule-based or
keyword-based miner would need someone to have thought of that combination in
advance and written it down. Here it fell out of the decomposition.

## Groups cover a scenario, single units cover a pattern

One unit usually captures one sub-pattern of a scenario. Asking instead how
often *at least one* unit of a related group fires gives the coverage of the
whole category, and the two views together are what makes the representation
compositional: high-purity single units for explanation, groups for recall.

| Scenario | Coverage by a related group |
|---|---:|
| Snowy | 100.0% |
| Rainy / foggy | 92.0% |
| Vulnerable road users | 57.6% |

## Using the units as retrieval keys

Once a unit has a name, it is a query. Run the screening tool over an
unlabelled archive and filter `screening.csv` on `top_neurons` instead of on
`tail_score`:

```bash
python scripts/screen.py --run-dir output/sae_abstopk_tail_reward \
    --features output/extract_archive/layer28_mlp_output_mean.npy \
    --meta output/extract_archive/meta.json \
    --glossary results/neuron_glossary.csv \
    --output-dir output/screening

# every frame in the archive with a wheelchair user ahead
awk -F, 'NR==1 || $7 ~ /(^| )3058( |$)/' output/screening/screening.csv
```

That is a categorical search over a corpus that was never labelled for the
category, driven by a unit that nobody specified in advance. It is the same
mechanism as the case study in [`results/README.md`](../results/README.md),
where unit 3058 surfaces a wheelchair frame that anomaly detection ranks in the
bottom 1.4% of outliers and that every tested VLM is confidently sure about.

## Rebuilding the glossary for your own run

Unit indices belong to the run that produced them: a different layer, seed or
latent width gives a different numbering. The glossary shipped here goes with
the layer-28 run described in the results. For a run of your own:

1. Train the SAE; `neuron_report.csv` appears in the output directory.
2. Take the units with high purity and at least a handful of activations.
3. For each, list the frames where `|z_t| > 1` — `z_t_longtail.npy` and
   `rows.json` in the same directory give you activations and sample tokens —
   and look at the top ones.
4. Write `neuron,feature` rows into a CSV in the format of
   `results/neuron_glossary.csv` and pass it to `screen.py --glossary`.

Only `neuron` and `feature` are read by the tool; the remaining columns are
there so the evidence for a name travels with the name.
