# Model-label recheck — 2026-10-24

View the [80-frame model-label table](all-ai-labels.md) or download the [complete
CSV](all-ai-labels.csv), which includes frame numbers, sample tokens, confidence and
both rounds' original reasoning. The original model-generated labels and the later
recheck appear in separate columns. Neither contains the user's human labels.

This was a **model-to-model agreement check** using the [review
procedure](../../docs/human-review.md). Four separate model-review sessions covered
20 timestamps each: 80 six-camera panels in total. They also inspected 31 individual
camera images where needed. Each frame received one judgment in this round, not four.

Review inputs contained the frozen rubric, image identities without labels and the
images. A fixed seed assigned the groups before inspection. The review inputs excluded
earlier labels, model scores, split assignments and scene descriptions. Separate
sessions can still share model biases; withholding those inputs does not establish
statistical independence or human ground truth.

## Results

| Item | Count |
|---|---:|
| Timestamps reviewed | 80 |
| Agreement with the earlier final model label | 73 |
| Disagreement with the earlier model label | 7 |
| Recheck: normal_core | 12 |
| Recheck: not_normal_core | 63 |
| Recheck: uncertain | 5 |
| Human-verified labels in this record | **0** |

The seven disagreements comprise one normal-to-non-normal change, two
non-normal-to-uncertain changes and four uncertain-to-non-normal changes. Three further
frames were uncertain in both rounds. These ten frames form a priority queue for human
review. The other 70 have not been human-verified.

Earlier labels, model predictions and metrics were preserved; no model was retrained.
Model agreement is not label accuracy. The four-model comparison still uses the
original model-reference pilot labels.

## Records and use

- [summary.json](summary.json): coverage, paired counts, rubric/hash checks and scope.
- [plan.json](plan.json) and [rubric.snapshot.txt](rubric.snapshot.txt): the
  pre-inspection plan and exact rubric bytes. The readable current rubric is
  [here](../../latest_grading_criteria.md).
- `reviewer-0.json` through `reviewer-3.json`: original model-review outputs, with
  per-frame evidence, uncertainty reasons, inspected cameras and panel hashes.
- [paired_decisions.json](paired_decisions.json) and
  [disagreements.json](disagreements.json): all paired decisions and the seven
  disagreements. Both expose the model-generated answers; keep them hidden
  during an independent human judgment.
- [human-priority-ids.json](human-priority-ids.json): identities for the ten priority
  frames, without reference answers. `index` is zero-based; page frame numbers are
  one-based. Match by `sample_token` when possible.
- [human-review-blank.csv](human-review-blank.csv): an empty 80-frame review template
  with every row pending. It contains no human labels.

Use the [image-review tool](../../tools/review_upload.html); model downloads are not
required. Reviewers inspect images under the SOP and export their own CSV. Save human
adjudication of disagreements separately. For partial review, report the coverage and
selection procedure.

This directory does not distribute original JPEGs, complete source metadata or task
bundles containing local paths. The bundles and execution checks remain in the run
workspace, with hashes in the plan. Data access remains subject to nuScenes terms. Every
model-generated row retains `human_review_status=pending`.
