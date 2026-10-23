# CPU mini pilot: provisional AI references

This bundle contains derived annotations, numeric predictions, diagnostics and the source snapshot used by the run. **No labels have been verified by a human.** Scores measure agreement with provisional AI labels; they are not benchmark accuracy.

- [Run explanation and exact scope](../../docs/ai-pilot.md)
- [Download the offline explorer](../../demo/driving_pilot.html): save HTML, open it locally in Chrome/Edge and choose the same four user-held ZIP parts. Images remain local.
- `ai_visual_labels.json` preserves all 80 decisions, evidence and uncertainty; `manifest.json` retains 73 active labels and the original scene split.
- `main_table.json` / `.csv` and each method directory preserve the four actual saved-output comparisons.
- `reward_diagnostic.json` records zero reward gradient from cap saturation; the two SAE checkpoints are identical.
- `failure_cases.json` contains three disagreements under the same MLP fixed rule.
- `independent_audit.json` and `offline_verification.json` record completed checks and their limits.
- `training_source_snapshot.zip` preserves the actual dirty source used by all four fits. Its hash is in each run record; current source includes later provenance hardening and the offline exporter.

The model checkpoints and feature cache are retained in the execution workspace, with hashes/sizes in the run metadata. They are not stored in Git. Original driving images and original metadata tables are not included. Obtain input data through authorized nuScenes access; this bundle does not change its terms.

The default human-review route remains separate. Do not submit these provisional scores as human-validated application results or attribute the AI-assisted follow-up code to the original capstone period.
