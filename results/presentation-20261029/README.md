# Walkthrough recordings and presentation delivery

The current [English walkthrough](../../demo/driving-walkthrough-90s.mp4) is available
from the repository. See the [demo guide](../../demo/README.md) for playback, source and
licensing information.

## Original private recording — 2026-10-29

The original private recording is 89.96 seconds, H.264, 1440×1000, 2,708,662 bytes, with
SHA256 `e1b293d456d126e4c77e1537162d5390e8598b59018bd0d2e55ce0863903e679`. It is silent
with English captions. Navigation is scripted in real Chromium; original image ZIPs and
saved trained outputs are used without new inference, training or human judgments.

The [private presentation
page](https://yike-driving-review-20261028.upbeatperch12.chatgpt.site/presentation/)
requires the owner's ChatGPT sign-in. It provides MP4/subtitle/script downloads and copy
controls for the final CV and interview text. At the time of the original delivery,
video pixels and posters were hosted only on that private Site. The JSON receipts below
describe that delivery; they are preserved unchanged and do not describe the later
public recording.

## Examples and review scope in the original recording

The walkthrough uses the earlier 77 retained references from the original 80-frame
cohort. Labels came from a single reviewer after AI-label disclosure. The separate
nine-frame supplement is described in the written interview story; it is reported
separately from the old test table.

- SAE selected reference-positive frame: `a9def8f915cf4bfe9c74b864384ec6c7`, saved score
  6.490295, above the earlier validation threshold 6.074955.
- MLP fixed-rule FP: `b4ff30109dd14c89b24789dc5713cf8c`, score 0.897683, normal
  reference.
- MLP fixed-rule FN: `c1eed31234b94e9f8e22fbf3428b0ac2`, score 0.065045, positive
  reference.

Both MLP cases use probability >= 0.5. Cases were selected after evaluation for
explanation, not tuning. Visible scene context does not establish the cause of an error.
The units have numeric IDs and no validated semantic names. The exported queue has 77
rows and blank new-review fields.

## Original delivery evidence

- [recording_receipt.json](recording_receipt.json): exact video properties/hash, source
  HTML and ZIP hashes, label scope, chapter timing and blank CSV export.
- [case_evidence.json](case_evidence.json): saved scores/references/policies for the
  three cases.
- [site_browser.json](site_browser.json): actual desktop/mobile player playback, MP4
  download-byte checks, copy controls, speech word counts and layout checks.
- [delivery.json](delivery.json): private page URL, page hash and delivery scope.
- [English captions](../../demo/presentation/captions.srt) and [final CV/interview
  copy](../../docs/presentation.md).

Browser checks used the prepared static Site bundle served over temporary localhost
HTTP. They did not verify an authenticated browser visit after deployment. Native
publication status is recorded separately. The original review payload and browser
storage key were preserved when the presentation link was added.

## Reproduce the original recording with authorized assets

Use Python with Pillow and Playwright, plus installed Chromium, FFmpeg and ffprobe. The
recorder uses the installed system FFmpeg through a checkout-local Playwright media
cache; it downloads no browser/model/data assets. Provide the original four
hash-matching ZIP parts, and use a fresh output directory:

```bash
python scripts/record_presentation.py \
  --parts PART_001.zip PART_002.zip PART_003.zip PART_004.zip \
  --output-dir output/presentation-replay
```

The shipped viewer verifies the original transport and image hashes. `--preview-only`
exercises the chapter navigation and saves screenshots without claiming a completed
video. Original images and weights require the documented access permissions and are not
included in a clean clone.
