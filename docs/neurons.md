The Interpretable Neurons
=============
Welcome to the long-tail driving data mining project neuron page!

This is the part of the project that makes the difference between a detector
and a tool. A long-tail detector answers "is this frame worth looking at?". The
sparse autoencoder answers that *and* which of its latent units fired — and
several of those units turn out to stand for a specific, nameable driving
situation. The shortlist therefore comes with a reason attached, and the units
themselves can be used as retrieval keys. Numbers in brackets refer to the
[references on the main page](../README.md#references).

## Why single units mean anything at all

Nothing forces a neural network to put one concept in one unit; in a dense
representation, concepts are spread across many directions at once. Sparse
dictionary learning is the standard way to pull them apart: an SAE trained to
reconstruct activations under a sparsity constraint recovers features that are
far more monosemantic than the raw neurons, in language models [19]–[21] and in
vision-language models [22]. Two properties of this SAE add to that:

- **AbsTopK sparsity** [18]. Only the `k = 512` latent units with the largest
  absolute activation survive per sample, out of 7,168. A unit that fires on
  everything is not useful under that budget, so units specialise.
- **The tail-guided objective.** The long-tail subspace `z_t` is pushed down on
  normal samples and up on long-tail samples, so the specialisation inside
  `z_t` is specialisation *towards long-tail structure* rather than towards
  whatever happens to reconstruct the data best.

The result is checkable rather than assumed: take a unit, take every frame
where `|z_t| > 1`, and look at them. The [neuron explorer](#browsing-the-units)
does exactly that.

## How a unit is scored

A unit *fires* on a frame when its activation exceeds 1 in magnitude; that
filters the small activations that are enough for the long-tail decision
(`|z_t| > 0.01`) but too weak to say anything about a particular unit. Every
SAE run writes `neuron_report.csv` with one row per `z_t` unit, computed on the
training split:

| Column | Meaning |
|---|---|
| `z_t_neuron` | Index of the unit inside `z_t` |
| `n_active` | How many frames it fires on |
| `n_active_long_tail` | How many of those are long-tail |
| `purity` | `n_active_long_tail / n_active` — when it fires, how often is it right |
| `long_tail_coverage` | Share of all long-tail frames it fires on |

For a named scenario the same two ideas are measured against that scenario: the
**activation ratio** is the share of the scenario's frames the unit fires on,
and the **purity** is the share of the frames it fires on that belong to the
scenario. Purity and coverage pull in opposite directions, and both matter. A
unit with high purity and low coverage is a precise detector of one narrow
situation; a unit with high coverage and low purity is a broad "something is
unusual" signal. The units worth naming are the high-purity ones, and a *group*
of them is what covers a whole scenario family.

## The named units

Naming is the one manual step: read the top-activating frames of a high-purity
unit and write down what they have in common. These are the units named on the
nuScenes run at layer 28. They are also the contents of
[`results/neuron_glossary.csv`](../results/neuron_glossary.csv), which the
screening tool and the explorer read so that a flagged frame carries a reason:

| Unit | Feature | Activation ratio | Purity |
|---:|---|---:|---:|
| 2169 | Rain / wet road | 43.8% | 100.0% (723/723) |
| 396 | Glare on a wet or rainy road | 5.6% | 100.0% (92/92) |
| 3058 | Wheelchair user ahead | 42.4% | 77.8% (14/18) |
| 1542 | Pedestrians crossing a construction zone | 6.1% | 53.8% (71/132) |
| 1843 | Truck close to the ego vehicle | 4.2% | 57.1% (12/21) |

![Frames that activate units 396, 1542 and 3058](figures/neuron_examples.webp)

Three of these are more interesting than "the model learned to see rain".

**Unit 396 is not a brightness detector.** It fires on glare specifically
caused by rain or a wet road surface, and it does so on different parts of
nuScenes (top row). A plain brightness or exposure feature would fire on any
strong light source; this one is tied to a compound of weather and
illumination, which is exactly the kind of condition that degrades perception.

**Unit 1542 is a conjunction.** It fires on pedestrians *in* construction
areas (middle row) — not on pedestrians alone, not on construction alone. A
rule-based or keyword-based miner would need someone to have thought of that
combination in advance. Here it fell out of the decomposition.

**Unit 3058 is a rare category found without asking for it.** Nobody labelled
wheelchairs; the unit fires on wheelchair users ahead of the vehicle across a
whole stretch of nuScenes (bottom row), including the frame of the
[case study](../README.md#case-study-a-frame-the-usual-methods-miss) that
anomaly detection and confidence filtering both miss.

## Groups cover a scenario, single units cover a pattern

One unit usually captures one sub-pattern of a scenario. Asking instead how
often *at least one* unit of a related group fires gives the coverage of the
whole category: for vulnerable road users, a group of related units fires on
**57.6%** of the vulnerable-road-user frames. The two views together are what
makes the representation compositional — high-purity single units for
explanation, groups for recall.

## Browsing the units

[`scripts/neuron_explorer.py`](../scripts/neuron_explorer.py) turns both
directions of this analysis into a local web page:

```bash
python scripts/neuron_explorer.py serve \
    --run output/sae_abstopk_tail_reward \
    --samples-root /data/nuscenes/samples \
    --glossary results/neuron_glossary.csv
# open http://127.0.0.1:8765
```

- The left column lists the long-tail units, named ones first, each with the
  number of frames it fires on and, for a training run, the share of them
  labelled long-tail.
- **Unit → frames.** Selecting a unit, or typing its number, shows the frames
  that activate it most, strongest first, with the activation, the scene and
  the label. A camera selector switches the thumbnails between the six views.
- **Frame → units.** Selecting a frame, or searching for a sample token or a
  scene name, shows its six views in their physical layout and every long-tail
  unit it activates as a ranked bar chart; units above the firing threshold are
  drawn dark, weaker ones light. Every unit links back to its own frames.

The same queries are available without a browser, and `--save` writes them as
image sheets:

```bash
python scripts/neuron_explorer.py units  --run output/sae_abstopk_tail_reward --top 30
python scripts/neuron_explorer.py unit 396 --run output/sae_abstopk_tail_reward \
    --samples-root /data/nuscenes/samples --top 16 --camera CAM_FRONT --save unit396.jpg
python scripts/neuron_explorer.py sample <sample_token> --run output/sae_abstopk_tail_reward \
    --samples-root /data/nuscenes/samples --save frame.jpg
```

The explorer reads `activations.npz`, which training and screening both write,
so it opens your own screened data exactly like the benchmark run. It needs
numpy and Pillow only.

## Using the units as retrieval keys

Once a unit has a name, it is a query. Screen an unlabelled archive and filter
`screening.csv` on the `top_neurons` column (the eighth) instead of on the
score:

```bash
make screen RUN=output/sae_abstopk_tail_reward FEATURES=output/extract_archive

# every frame in the archive with a wheelchair user ahead
awk -F, 'NR==1 || $8 ~ /(^| )3058( |$)/' output/screening/screening.csv
```

That is a categorical search over a corpus that was never labelled for the
category, driven by a unit nobody specified in advance.

## Naming the units of your own run

Unit indices belong to the run that produced them: a different layer, seed or
latent width gives a different numbering, so the glossary shipped here goes
with the layer-28 run described in the results. For a run of your own:

1. Train the SAE; `neuron_report.csv` and `activations.npz` appear in the
   output directory.
2. Open the explorer, or run `neuron_explorer.py units`, and take the units
   with high purity and at least a handful of frames.
3. For each, look at its strongest frames and write down what they share.
4. Write `neuron,feature` rows into a CSV in the format of
   `results/neuron_glossary.csv` and pass it to `--glossary`.

Only `neuron` and `feature` are read by the tools; the remaining columns are
there so the evidence for a name travels with the name.
