# English walkthrough · 29 October 2026

[View the MP4 on GitHub](../../demo/driving-walkthrough-90s.mp4) · [Download the MP4](https://github.com/CCC917111/long-tail-driving-scene-discovery/raw/refs/heads/main/demo/driving-walkthrough-90s.mp4) · [English subtitles](../../demo/presentation/captions.srt) · [Video license and image credits](../../demo/VIDEO-LICENSE.md)

The 90-second recording shows the current English explorer with original nuScenes images and saved model outputs. It has burned-in English captions and no audio. Scripted browser navigation imports the four original image ZIPs, displays six camera views, inspects unit 67, shows an MLP false positive and false negative, compares four models, and downloads 77 blank review rows.

This is playback of the existing CPU pilot. No training, inference, threshold selection or human annotation was performed for the video. The model scores, image hashes and 77-frame reference set are unchanged by the interface translation. Error examples are relative to one reviewer's judgments after AI-label disclosure.

| Property | Value |
| --- | --- |
| Duration | 90.00 seconds |
| Format | H.264 MP4, 1440 × 1000, no audio stream |
| File size | 2,969,370 bytes |
| SHA-256 | `dc798ec25e51442ebf054853c591fa148c66a6e8e70c33b330aad684ddb7a21b` |
| Reference cohort | 77 binary judgments from the earlier 80 timestamps |
| Model fitting | Existing one-seed CPU run; no new fits |

## Evidence

- [Recording source](recording_source.py): exact recorder used for this MP4; its hash matches the receipt. The current script has a formatting-only line wrap.
- [Recording receipt](recording_receipt.json): exact viewer and recorder hashes, ZIP input hashes, chapter timings, browser errors, MP4 properties and CSV export checksum.
- [Selected cases](case_evidence.json): sample tokens, splits, saved scores, reference labels and decision policies.
- [Presentation browser checks](site_browser.json): desktop/mobile playback, download checksum, copy controls and layout checks against the exact Site bundle.
- [Review browser checks](review_browser.json): all 54 new-scene images load on desktop and mobile; temporary test decisions are not human labels.

The [earlier presentation record](../presentation-20261029/README.md) describes the original owner-private recording, whose interface still contained Chinese. Its receipts and hashes remain historical evidence. [Documentation provenance](../documentation-english-20261029/README.md) preserves exact originals of the rewritten documents and viewers.
