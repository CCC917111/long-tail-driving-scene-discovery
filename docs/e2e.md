# Driving end-to-end verification

This workflow uses generated images and features to verify training,
evaluation, checkpoint reload, screening and review export. Use Python 3.12
with the [pinned CPU dependencies](../requirements-cpu-lock.txt). For real-image
results and their review limits, see [the CPU route](cpu-route.md).

## One command

From the repository root, after installing the CPU requirements:

```bash
make e2e PYTHON=.venv/bin/python OUTPUT=output/e2e-example
```

Use a fresh output location; the workflow refuses to overwrite an existing
run. This Make target includes `--http` and requires loopback networking. It
trains Logistic, MLP, AbsTopK without reward and the same SAE with reward on
60 scenes / 240 synthetic frames, using one frozen split. It then checks
reloaded predictions and train-only standardisers, rebuilds the comparison,
and screens 40 independent unlabelled synthetic samples.

To run the cached-feature workflow without HTTP checks:

```bash
.venv/bin/python scripts/synthetic_demo.py --output-dir output/cpu-example
```

Add `--http` to that direct command to run the real temporary servers. The
server check verifies HTML, summary/units/sample/search APIs, six JPEG images,
review CSV and an expected missing-token 404. Socket failures fail the check;
they are not replaced by a simulated service or successful skip.

## Evidence files

Under `output/e2e-example/synthetic_e2e/`:

| File / directory | Evidence |
|---|---|
| `demo.json`, `commands.txt`, `logs/` | Run scope, exact child commands and logs |
| `comparison/main_table.csv` | Rebuilt four-run table; synthetic metrics only |
| `reload_check.json` | Checkpoint predictions and train-only normalisation match |
| `screening_check.json`, `screening/` | 40 independent unlabelled tokens; effective screening `eta=0.05` retained on reload |
| `browser/smoke.json`, `sample.png`, `unit.png` | Dynamically selected inspectable synthetic FP and unit examples; PNGs are under `browser/` |
| `browser/trained/http_check.json`, `browser/screened/http_check.json` | 14 actual HTTP route checks each, thresholds, responses and server cleanup |
| `browser_command.txt`, `screening_browser_command.txt` | Commands to browse the respective activation stores |

The trained explorer uses `eta=0.01`; the screened explorer uses its saved
0.05 override. `/export/review.csv` downloads candidate scores and blank
reviewer fields without reference labels.

## Real browser checks

`scripts/verify_browser_ui.py` uses Playwright and installed Chromium. Those
browser dependencies are separate from the CPU training lock. In the recorded managed environment, the global `python` contains Playwright;
the shared training venv does not. Select an interpreter with Playwright and
Chromium installed.

Read the current sample token/unit from `browser/smoke.json`, then supply them:

```bash
python scripts/verify_browser_ui.py \
  --run output/e2e-example/synthetic_e2e/sae_norm \
  --samples-root output/e2e-example/synthetic_e2e/dataset/samples \
  --sample-token SAMPLE_TOKEN --unit UNIT --neuron-threshold 0.01 \
  --output-dir output/browser-ui-example
```

Replace `SAMPLE_TOKEN` and `UNIT` with the current run's entries. The script
starts a real loopback server and Chromium, checks unit/frame links, search,
six distinct decoded camera images and the CSV download at desktop and mobile
sizes. It records page/request/HTTP errors, screenshots and `ui_check.json`,
and exits nonzero on failure. See [testing](testing.md#latest-local-record) for
the completed checks and any remaining verification limits.
