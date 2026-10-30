# Driving Scene Explorer

[CV bullets and interview scripts](../docs/presentation.md) ·
[Project role and contributions](../docs/project-card.md)

## Use the offline viewer

The latest viewer replays the 2026-10-26 comparison using one reviewer's final
judgments. The reviewer had seen model-generated reference labels before the
submission, so the references are not independently blinded. It retains 12
normal and 65 tail references and excludes three uncertain decisions. The
viewer contains 77 timestamps and 308 cached method scores.

1. Open [driving_pilot.html](driving_pilot.html) on GitHub and choose
   **Download raw file**. GitHub's source preview does not run the viewer.
2. Open the saved HTML in desktop Chrome or Edge.
3. Select all four original ZIP parts, `driving-mini-part-001-of-004.zip`
   through `driving-mini-part-004-of-004.zip`, and click **Open images**.
4. Open a case bookmark or select an active unit to inspect its high-activation
   frames. The table shows fixed-rule and operating-point decisions separately.
5. Export the review CSV. New-review fields start blank; cached reference
   labels remain a separate column.

The HTML embeds numeric outputs but no original image bytes. Image navigation
requires the matching ZIP parts, whose identities and hashes are checked;
images stay in your browser. Without those assets, run the
[synthetic CPU example](../README.md#run-a-cpu-example).

## Inspect results and errors

The explorer has three MLP error bookmarks: page frame 16 is a false positive,
and page frames 10 and 1 are false negatives. These examples did not tune the
models. The SAE's fixed activation rule flags every test frame; its top-four
ranking and validation-selected operating point answer different questions.

The fitted comparison uses one seed, two test scenes and only three normal
training frames from one scene. Its references are not independent blind
labels. See the [full result record](../results/human-ai-exposed-20261026/README.md)
for predictions, review revisions and policy definitions.

## Review new scenes

The [new-scene HTML](driving_new_scenes.html) loads nine six-camera timestamps
from original author image URLs on `raw.githubusercontent.com`. Download it
and open it with an Internet connection. It includes all 54 image links and
requires no account or ZIP upload.

The [nine-frame review record](../results/new-scene-human-review-20261028/README.md)
contains five normal and four non-normal judgments submitted after disclosure
of a model-generated reference table. Model signals are hidden by default in the
review UI, but that does not remove the prior disclosure. The models were
not retrained on this batch. All normal references come from one scene and
all non-normal references from the other.

## Verification

The saved viewer checks reconstruct manifests, labels, splits, scores and
selected input images. Separate Node checks loaded all four original ZIPs and
compared all 480 image files byte for byte with the originals; all 308 scores in the latest viewer
match saved predictions. Its 77 exported rows have blank new-review fields.
The archive retains its original bytes and separate 292-score check.

Browser CI exercises desktop and mobile navigation, search and CSV downloads
with synthetic watermarked ZIPs. Checks of the separate new-scene bundle cover
all 54 original images, CSV export and failed-image retry. These checks have
different scopes; see [testing](../docs/testing.md) and the
[new-scene repair record](../results/review-page-repair-20261028/README.md).

[Latest fitted results](../results/human-ai-exposed-20261026/README.md) ·
[Earlier provisional-label pilot](../results/cpu-mini-ai-pilot-20261023/README.md) ·
[New-scene execution guide](../docs/new-scene-e2e.md) ·
[Research expansion plan](../docs/research-expansion.md)
