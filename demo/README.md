# Offline Driving Explorer

**Recorded walkthrough and final application copy:**
[open the private presentation page](https://yike-driving-review-20261028.upbeatperch12.chatgpt.site/presentation/)
with the owner's ChatGPT sign-in. It contains an approximately 90-second real-image
MP4 with English captions, downloadable subtitles, two final CV bullets and
60-second / three-minute interview scripts. The [text is also public](../docs/presentation.md).
The recording is silent and navigation is scripted; models are not rerun.


**New: [hosted new-scene review tool](https://yike-driving-review-20261028.upbeatperch12.chatgpt.site).**
This owner-private Site requires the owner's ChatGPT sign-in. All 54 images
are served from the same origin, so the owner can inspect nine six-camera
groups and export CSV without downloading HTML or uploading ZIPs. External
GitHub readers do not receive access to this private tool automatically.
Model signals stay hidden until requested; the page does not display human
reference labels. The separate [downloadable fallback](driving_new_scenes.html)
uses original author image URLs and requires access to `raw.githubusercontent.com`.
See the [real-data E2E guide](../docs/new-scene-e2e.md) and
[image-loading repair evidence](../results/review-page-repair-20261028/README.md).

The [later nine-row submission](../results/new-scene-human-review-20261028/README.md)
contains five normal and four non-normal judgments. The AI reference table had
been disclosed in chat before submission; browser model-signal visibility does
not establish blind human review. The two cached result viewers below retain
their original fitted-run reference protocols.

**Project lead: Yike Chen.** The latest viewer replays the completed
single-reviewer reference run after AI-label disclosure. It shows six-camera
images, trained sparse activations, four-model scores and a review export.
The downloaded HTML runs locally in desktop Chrome/Edge; images stay local.

| Viewer | References | Cached timestamps | Model values |
|---|---|---:|---:|
| [Latest driving_pilot.html](driving_pilot.html) | Final user review after AI-label exposure | 77 | 308 |
| [Archived driving_ai_pilot.html](driving_ai_pilot.html) | Original provisional AI labels | 73 | 292 |

The archive preserves the original HTML bytes. The latest viewer does not
replace the older run's predictions or change training. It uses 12 normal and
65 tail references; three uncertain decisions are excluded. One reviewer saw
the full AI-label table before final submission, so these are not independent
blind human labels or an official benchmark.

## Open the viewer

1. Open [driving_pilot.html](driving_pilot.html) on GitHub and click the download
   arrow (**Download raw file**). Save the HTML; GitHub's source preview does
   not run the app.
2. Open the saved file in desktop Chrome or Edge.
3. Select the same four original `driving-mini-part-001-of-004.zip` through
   `driving-mini-part-004-of-004.zip` files together. Click **打开原图**.
4. Open a case bookmark, select an active unit and compare its high-activation
   frames. The per-frame table distinguishes fixed-rule and operating decisions.
5. Export the review CSV. Its new-review fields are blank; cached reference
   labels remain a separately disclosed source.

中文：下载最新 HTML → 电脑 Chrome/Edge 打开 → 同时选择原来的四个 ZIP →
点击「打开原图」。新版本对应最终复核结果；旧 AI 试跑有单独归档。

The HTML has numeric caches and no original image bytes. Matching parts are
required for image navigation; their transfer identity and hashes are checked.
For a runnable generated-data example without authorized nuScenes assets, use
[the CPU quickstart](../README.md#run-a-cpu-example).

## A 90-second walkthrough

| Time | Show | Say |
|---|---|---|
| 0–15 s | Project title, scope banner and result table | “I lead the optimization and delivery of this repository, building on our team capstone. This viewer replays a small CPU experiment using final user judgments after AI-label exposure.” |
| 15–35 s | One six-camera timestamp, its score and active units | “One timestamp contains six views. Scores and sparse activations come from saved trained models; the viewer runs no new inference.” |
| 35–50 s | An active unit and its high-activation frames | “I can inspect which examples activate a direction. These numeric unit IDs have no validated semantic names.” |
| 50–70 s | MLP fixed-rule FP bookmark, then an FN bookmark | “These are disagreements with the final reference labels under the same MLP threshold of 0.5. We retain both mistakes rather than reporting only good examples.” |
| 70–90 s | Four-model table, operating policy and review export | “SAE top four finds four reference positives. Its validation-selected policy separately selects two test frames; both are positive, recalling two of ten. Reward/no-reward weights are identical. I export the review queue for follow-up.” |

The three error bookmarks use the same MLP fixed rule: page frame 16 is an FP,
and page frames 10 and 1 are FNs. These cases are illustrative, selected after
evaluation, and did not tune the models. The fixed SAE activation rule flags
all test frames; it is separate from the ranking and validation policies.
An approximately 90-second English-captioned real-image recording is complete
and available on the private presentation page above. It has no spoken narration.
The [recording receipt](../results/presentation-20261029/README.md) preserves
source/asset hashes and the exact selected examples.

## Verification and limits

- Export reconstruction checks scope, manifests, cached labels/splits/scores,
  selected input images and saved-output table integrity for both reference sets.
- Independent Node checks loaded the actual four local ZIPs and compared all
  480 image bytes with the originals. All 308 displayed method values match
  saved predictions; 77 review rows have blank new-review fields.
- Local synthetic core checks pass for both HTML versions. GitHub Actions
  exercises Chrome desktop/mobile rendering, case and unit navigation, search,
  tables and actual CSV downloads using visibly synthetic ZIP images. This
  remains separate from the original-byte/score checks and does not certify a
  real-image browser session. See the publication PR checks for current UI status.

The separate new-scene Site bundle passed local desktop/mobile checks against
all 54 real original images, CSV export, and a failed-image retry check. These
[repair records](../results/review-page-repair-20261028/README.md) are distinct
from both authenticated production browsing and the synthetic ZIP checks for
the two older viewers. Native publishing status verifies private deployment.
CI for the downloadable
source-linked fallback does not authenticate into or test the owner-private Site.

Test contains 15 timestamps from two scenes; training has three normal frames
from one scene. The two SAE checkpoints match each other and the original AI
pilot; score changes across reference versions are not model improvements.
The [research expansion plan](../docs/research-expansion.md) describes the new
scene data and independent labeling needed for stronger claims; it has not run.

[Latest result record](../results/human-ai-exposed-20261026/README.md) ·
[Original AI pilot](../results/cpu-mini-ai-pilot-20261023/README.md) ·
[Project card and interview story](../docs/project-card.md) ·
[Blank image-review tool](../tools/review_upload.html)
