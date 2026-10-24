# 独立 AI 标签复核 · 2026-10-24

这是根据 [复核步骤](../../docs/human-review.md) 执行的 **AI-to-AI 一致性检查**。
四个新会话的 AI agent 各检查 20 个时间点，总计 80 个六视角图像面板，并按需要
打开了 31 张原始摄像头图片。每帧本轮一次 AI 判断；不是每帧四位复核者。

输入仅为冻结 rubric、无标签图像身份和原图；分组在看图前由固定种子冻结。
未向复核 agent 提供旧标签、模型分数、split 或场景描述。新会话仍可能共享模型
偏差，因此输入隔离不等于统计独立，更不等于独立的人类真值。

## 结果

| 项目 | 数量 |
|---|---:|
| 本轮覆盖时间点 | 80 |
| 与先前 AI 最终标签一致 | 73 |
| 与先前 AI 标签有分歧 | 7 |
| 本轮 normal_core | 12 |
| 本轮 not_normal_core | 63 |
| 本轮 uncertain | 5 |
| 已完成人工核验 | **0** |

七处分歧包括：一帧 normal → not-normal、两帧 not-normal → uncertain、四帧
uncertain → not-normal。另有三帧两轮都 uncertain。合计十帧可优先交给实际人员
看图，**这不代表其余七十帧已通过人工核验**。

没有覆盖旧标签、修改旧模型预测/指标或自动重新训练。本轮的 AI 一致性不能
解释为标签准确率；原四模型表格仍对应原来的 AI 试跑版本。

## 记录与使用

- [summary.json](summary.json)：完整覆盖、配对统计、rubric/hash 校验与范围声明。
- [plan.json](plan.json) / [rubric.snapshot.txt](rubric.snapshot.txt)：看图前冻结的
  任务计划和标准原始字节；可阅读的标准见 [canonical rubric](../../latest_grading_criteria.md)。
- `reviewer-0.json` 至 `reviewer-3.json`：四位 AI 的原始提交，含逐帧证据、
  不确定原因、实际打开的摄像头和面板 hash；保留原始字段形式。
- [paired_decisions.json](paired_decisions.json)：全部配对；
  [disagreements.json](disagreements.json)：七处分歧。二者含旧/新 AI 标签，独立
  人工判断前不要打开以免锚定。
- [human-priority-ids.json](human-priority-ids.json)：十个优先检查样本的身份，
  不包含 AI 答案。身份中的 `index` 从零开始；界面「第几帧」从一开始，优先
  用 `sample_token` 确认身份。
- [human-review-blank.csv](human-review-blank.csv)：80 帧空白人类记录，全部
  pending；**不是人工标签数据**。

人工操作使用 [原图复核工具](../../tools/review_upload.html)，不需要下载模型。
实际审阅者按 SOP 看图、导出自己的 CSV；对有分歧项另存人工裁定。只核验部分
帧时，明确记录覆盖数量和选择方式。

原始 JPEG、完整原数据表及本地路径形式的任务包不在此结果目录中分发；任务包
和执行校验脚本保留在运行工作区，计划记录其 hash。数据访问仍遵守 nuScenes
条款。所有 AI 行保持 `human_review_status=pending`。
