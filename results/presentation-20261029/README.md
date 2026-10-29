# Recorded walkthrough and presentation delivery

An approximately 90-second real-image UI walkthrough is complete. The MP4
is 89.96 seconds, H.264, 1440×1000, 2,708,662 bytes, with SHA256
`e1b293d456d126e4c77e1537162d5390e8598b59018bd0d2e55ce0863903e679`.
It is silent with English captions. Navigation is scripted in real Chromium;
original image ZIPs and saved trained outputs are used without new inference,
training or human judgments.

The [private presentation page](https://yike-driving-review-20261028.upbeatperch12.chatgpt.site/presentation/)
requires the owner's ChatGPT sign-in. It provides MP4/subtitle/script downloads
and copy controls for the final CV and interview text. Video pixels and posters
remain on that private Site and are not in this public repository.

## Recorded examples and scope

The walkthrough uses the earlier 77 retained references from the original
80-frame cohort. Labels came from a single reviewer after AI-label disclosure.
The separate nine-frame supplement is described in the written interview story;
it is not silently combined with the old test table in the video.

- SAE selected reference-positive frame: `a9def8f915cf4bfe9c74b864384ec6c7`,
  saved score 6.490295, above the earlier validation threshold 6.074955.
- MLP fixed-rule FP: `b4ff30109dd14c89b24789dc5713cf8c`, score 0.897683, normal reference.
- MLP fixed-rule FN: `c1eed31234b94e9f8e22fbf3428b0ac2`, score 0.065045, positive reference.

Both MLP cases use probability >= 0.5. Cases were selected after evaluation
for explanation, not tuning. Visible scene context is not a proven causal
explanation of an error. The units have numeric IDs and no validated semantic
names. The exported queue has 77 rows and blank new-review fields.

## Evidence

- [recording_receipt.json](recording_receipt.json): exact video properties/hash,
  source HTML and ZIP hashes, label scope, chapter timing and blank CSV export.
- [case_evidence.json](case_evidence.json): saved scores/references/policies for the three cases.
- [site_browser.json](site_browser.json): actual desktop/mobile player playback,
  MP4 download-byte checks, copy controls, speech word counts and layout checks.
- [delivery.json](delivery.json): private page URL, page hash and delivery scope.
- [English captions](../../demo/presentation/captions.srt) and
  [final CV/interview copy](../../docs/presentation.md).

Browser QA ran against the exact prepared static Site bundle on temporary
localhost HTTP. It does not claim an authenticated post-deployment browser
visit. Native publication status is recorded separately after deployment.
The original review payload and browser storage key were preserved when the
presentation link was added.

## Record again with authorised assets

Use Python with Pillow and Playwright, plus installed Chromium, FFmpeg and
ffprobe. The recorder uses the installed system FFmpeg through a checkout-local
Playwright media cache; it downloads no browser/model/data assets. Provide the
original four hash-matching ZIP parts, and use a fresh output directory:

```bash
python scripts/record_presentation.py \
  --parts PART_001.zip PART_002.zip PART_003.zip PART_004.zip \
  --output-dir output/presentation-replay
```

The shipped viewer verifies the original transport and image hashes.
`--preview-only` exercises the chapter navigation and saves screenshots without
claiming a completed video. Original images/weights require the documented
access conditions; a clean clone does not include them.
