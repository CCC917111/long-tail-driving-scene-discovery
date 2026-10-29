# Inspecting sparse units

The neuron explorer links a frame to active tail units and each unit to
high-activation frames. A unit's human-assigned name is a hypothesis to check;
sparsity alone does not establish a category detector or causal explanation.

Historical unit names, the sample figure and reported purity/activation ratios
are retained in [historical results](../results/README.md#historical-interpretation-and-agreement).
Their original sample memberships and review records are absent. The shipped
[glossary](../results/neuron_glossary.csv) belongs to that historical run; do not
apply its semantic names to a new checkpoint or synthetic model.

## Training diagnostics and semantic evaluation

Every SAE run writes `neuron_report.csv` using **training** binary labels:

| Column | Meaning |
|---|---|
| `z_t_neuron` | Unit index within the tail subspace |
| `n_active` | Frames above the interpretation activation threshold |
| `n_active_long_tail` | Active frames with a tail reference label |
| `purity` | Fraction of active training frames labelled tail |
| `long_tail_coverage` | Fraction of all tail training frames activating the unit |

The default interpretation threshold is magnitude 1.0, distinct from the
original fixed-decision floor of 0.01. Binary training purity does not tell you
whether a unit means rain, a wheelchair or construction. For a named scenario,
record scenario-specific labels and measure purity/coverage against those
labels. Explore names and thresholds on train/validation, freeze them, then
measure them on held-out test scenes. Report empty-support cases explicitly.

## Browse a trained run

For authorised real cached images and a matching run-specific glossary:

```bash
python scripts/neuron_explorer.py serve \
  --run output/sae_norm_seed17 --samples-root /path/to/nuscenes/samples \
  --glossary /path/to/that_run_glossary.csv
```

The default address is `http://127.0.0.1:8765`. HTTP serving requires an
environment that allows a loopback socket. The initial CPU check used static
queries and synthetic image previews. The [downloadable explorer](../demo/driving_pilot.html)
uses saved CPU-model outputs and loads authorised images from the original
local ZIPs. See [testing](testing.md) for browser-check scope.

The interface provides two directions: a unit's strongest frames (ranked by
activation magnitude), and a frame's six views plus active units. Unit indices,
thresholds, labels and split metadata help inspect mistakes. Use the terminal
commands when an HTTP server is unavailable:

```bash
python scripts/neuron_explorer.py units --run output/sae_norm_seed17 --top 30
python scripts/neuron_explorer.py unit 5 --run output/sae_norm_seed17 \
  --samples-root /path/to/nuscenes/samples --top 16 --save unit5.jpg
python scripts/neuron_explorer.py sample SAMPLE_TOKEN --run output/sae_norm_seed17 \
  --samples-root /path/to/nuscenes/samples --save frame.jpg
```

Replace unit 5 and `SAMPLE_TOKEN` with entries present in the run; they are
illustrative arguments, not known real examples. The explorer reads
`activations.npz` from training or screening, and needs NumPy and Pillow for
these cached queries.

## Name units and demonstrate a mistake

1. Inspect train/validation top activations, including disagreements and
   confident mistakes. Keep the evidence sample IDs and independent reviewer
   decisions rather than only a convenient montage.
2. Record a tentative name and activation threshold; test alternative
   explanations using contrasting examples. High activation alone does not
   prove a concept-specific response.
3. Save `neuron,feature` rows in a new run-specific CSV, freeze names and
   thresholds, then evaluate semantic purity/coverage on test scenes.
4. Show one frame's score and units, compare similar high-activation examples,
   explain an FP or FN and export the evaluation review queue. The queue omits
   reference labels to support independent human review.

A named unit can be used as a retrieval key in `screening.csv`, but its
precision and coverage on a new corpus must be measured. Groups of units add
another selection choice to freeze using train/validation; historical group
coverage is not a guaranteed outcome for new data.
