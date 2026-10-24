# Offline Driving Pilot

Inspect real six-camera frames alongside cached scores and trained sparse
activations. The downloaded viewer needs no server, Python installation or
GPU. Images remain on your computer.

> Labels are AI provisional, with no human verification. False positives,
> false negatives and purity here are relative to those AI references.

## Open the viewer

1. Open [driving_pilot.html](driving_pilot.html) on GitHub. Click the download
   arrow (**Download raw file**) and save the `.html` file. GitHub's source
   preview does not run the app.
2. Open the saved file in desktop Chrome or Edge.
3. Select all four original `driving-mini-part-001-of-004.zip` through
   `driving-mini-part-004-of-004.zip` files together. Click **打开原图**.
4. Use a case bookmark, a unit number or an exact sample token to inspect
   activations and six camera views. Download the review CSV when done.

中文：下载 HTML → 在电脑 Chrome/Edge 打开 → 同时选择原来的四个 ZIP →
点击「打开原图」。图片只在本机读取，不上传。

These ZIPs are not distributed in the public repository. The viewer validates
the original transfer identity, selected samples and image hashes. Another
nuScenes archive or generated four-part transfer will not match this cached
pilot. Without matching files, the result summary remains readable but image
navigation cannot initialize. For an independently runnable example, follow
the [generated-data CPU quickstart](../README.md#run-a-cpu-example).

## A 90-second walkthrough

| Time | Show | Explain |
|---|---|---|
| 0–15 s | Scope banner and result table | Candidate screening with provisional AI references. |
| 15–35 s | High-score candidate, six views and active units | Scores and activations come from saved trained models. Unit numbers have no validated semantic names. |
| 35–50 s | A unit and its high-activation frames | Compare examples to form a hypothesis about what the unit responds to. |
| 50–70 s | SAE fixed-rule false-positive bookmark, then MLP false-negative bookmark | Explain the visible disagreements. These bookmarks use different models; they do not form one confusion matrix. |
| 70–90 s | Per-frame four-model table and review CSV | Fixed-rule and validation-selected decisions differ. Export blank human-review fields for independent judgment. |

For three examples under one consistent MLP decision rule, read
[failure_cases.json](../results/cpu-mini-ai-pilot-20261023/failure_cases.json).
This is a presentation script; a video has not been recorded.

## What has been checked

- All 480 original JPEGs matched the transported hashes and decoded correctly.
- The real-part loader reconstructed the images, and 292 displayed method
  values matched saved predictions in independent Node checks.
- GitHub Actions Chrome tested desktop/mobile navigation, six-image rendering,
  search and CSV downloads with watermarked synthetic ZIPs. This UI check is
  separate from the real-image byte check and does not certify a real-image
  browser session.

[Verification details](../docs/testing.md) ·
[Full pilot protocol](../docs/ai-pilot.md) ·
[Saved results](../results/cpu-mini-ai-pilot-20261023/README.md)

## Limits to keep visible

Test contains 14 timestamps from two scenes; training has only three normal
frames. All validation precision targets were unreachable. Reward/no-reward
SAE checkpoints are identical because the reward was saturated. The HTML
replays cached results and does not run inference or collect ground truth.
Use the separate [blank reviewer](../tools/review_upload.html) for independent
annotation.
