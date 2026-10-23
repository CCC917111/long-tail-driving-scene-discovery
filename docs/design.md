# High-level design

The detailed extraction, SAE equations and research references live in
[method and historical context](methods-history.md). This page describes
implementation decisions and their limits. The
[historical results](../results/README.md) and
[current evaluation protocol](evaluation.md) use different evidence scopes.

## Representation and decision

The pipeline consumes one timestamp with six ordered camera views. A frozen
VLM produces a dense feature vector under a scene-description prompt. A linear
encoder, sparse selection and linear decoder learn a code `[z_n, z_t]`.
The objective combines reconstruction, suppression of tail-code activity on
normal-labelled training samples, and optional capped reward on tail-labelled
samples. These constraints encourage task sensitivity; they do not guarantee
semantic purity, causal explanations or better performance than a classifier.

AbsTopK keeps the largest activation magnitudes with their signs. Signed TopK
uses signed order, which can discard large negative activations. Whether that
difference helps this task requires a controlled comparison with every other
setting held fixed. Shared code reduces implementation drift, but different
entry-point defaults still change multiple variables.

The original fixed rule flags a sample when any tail activation has magnitude
above `eta` (default 0.01). The tail-code norm gives a continuous ranking.
The current evaluator also selects a separate score threshold on validation
using a declared precision target. Original-rule metrics and fitted-operating-
point metrics are retained separately; test labels select neither threshold.
Screening currently uses the original activation-count rule.

## Data and reproducibility

- A retained sample has a unique sample token, an explicit scene token and a
  label from the documented whitelist. Missing scene tokens are errors;
  unrecognised label strings are never silently positive.
- A frozen manifest keeps whole scenes together. Reuse it across model seeds
  and methods; model initialisation must not redefine test data.
- Feature mean and standard deviation are fitted on train only. Validation
  chooses checkpoints and any allowed hyperparameters; test evaluates frozen
  decisions once.
- Record model revision, layer, pooling and prompt/rubric versions. Matching
  array dimensions cannot establish matching feature semantics. Input hashes
  identify files but do not recover missing historical provenance.
- Per-sample predictions, provenance, supports, confusion counts and metric
  calculations travel together so tables can be rebuilt and checked.

The original random-frame 80/20 results used held-out loss to select a
checkpoint. They have not been rerun under this protocol. Missing original
assets, raw-VLM support discrepancies and the six-frame count difference
remain explicitly documented.

## Inspectability and limitations

The activation store links tokens, scenes and images to sparse activations.
The explorer can retrieve high-activation examples or a frame's active units.
Human names in a glossary are hypotheses about unit behavior. Training-label
purity is a binary-label diagnostic; semantic purity/coverage requires separate
scenario labels, frozen names/thresholds and independent test samples.

A useful output is a review shortlist with known yield and honest failure
examples. The four core methods must use matching data and budgets to determine
whether the SAE's quality, inspection value and cost justify it. No positive
result is assumed in the implementation.
