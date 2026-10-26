# Offline Driving Explorer

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
A walkthrough script is prepared; a narrated real-image recording has not yet
been made.

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

Test contains 15 timestamps from two scenes; training has three normal frames
from one scene. The two SAE checkpoints match each other and the original AI
pilot; score changes across reference versions are not model improvements.
The [research expansion plan](../docs/research-expansion.md) describes the new
scene data and independent labeling needed for stronger claims; it has not run.

[Latest result record](../results/human-ai-exposed-20261026/README.md) ·
[Original AI pilot](../results/cpu-mini-ai-pilot-20261023/README.md) ·
[Project card and interview story](../docs/project-card.md) ·
[Blank image-review tool](../tools/review_upload.html)
