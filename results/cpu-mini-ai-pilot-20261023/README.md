# CPU mini pilot: provisional model-generated references

This bundle contains the pilot annotations, predictions, diagnostics and training source
snapshot. Its references are provisional model-generated judgments, with no human verification in
this run. Scores measure agreement with those references.

- [Run explanation and exact scope](../../docs/ai-pilot.md)
- [Archived model-reference explorer](../../demo/driving_ai_pilot.html): save the HTML,
  open it locally in Chrome or Edge, and select the four original ZIP parts. Images
  remain local. This archive preserves the original HTML bytes;
  `demo/driving_pilot.html` contains the later reviewed-reference demo.
- `ai_visual_labels.json` preserves all 80 decisions, evidence and uncertainty;
  `manifest.json` retains 73 active labels and the original scene split.
- `main_table.json`, `main_table.csv` and each method directory preserve the four
  completed runs and their saved outputs.
- `reward_diagnostic.json` records zero reward gradient from cap saturation; the two SAE
  checkpoints are identical.
- `failure_cases.json` contains three disagreements under the same MLP fixed rule.
- `independent_audit.json` and `offline_verification.json` record completed checks and
  their limits.
- `browser_ci_verification.json` records the later Chrome desktop/mobile UI check using
  synthetic, watermarked ZIPs, separate from the original-image Node checks.
  `completion.json` is the earlier point-in-time record, before that CI run.
- `training_source_snapshot.zip` preserves the exact working-tree source used by all
  four fits, including uncommitted changes. Its hash is in each run record; current
  source includes later provenance hardening and the offline exporter.

The model checkpoints and feature cache are retained in the execution workspace, with
hashes/sizes in the run metadata. They are not stored in Git. Original driving images
and original metadata tables are not included. Obtain input data through authorized
nuScenes access; this bundle does not change its terms.

The [human-review workflow](../../docs/human-review.md) is separate. When citing this
pilot, describe its references as provisional model-generated labels and distinguish the later CPU
implementation from the original capstone work.
