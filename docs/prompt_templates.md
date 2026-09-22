# Prompt Templates

These are the exact prompts used in the project, copied verbatim from the
code: (1) the classification prompt that encodes the normal-core rubric and
(2) the open-ended scene-description prompt used to elicit the internal VLM
representations that the SAE is trained on.

## 1. Normal-core classification prompt

Source: `PROMPT_TEMPLATE` in [`scripts/annotate_normal_core.py`](../scripts/annotate_normal_core.py).

This prompt is used in two places:

- **Labelling.** `annotate_normal_core.py` sends it, together with the 6 camera
  images of a nuScenes keyframe, to a VLM and reads the answer letter's logits to
  obtain a `normal_core` / `not_normal_core` / `uncertain` label with a
  calibrated confidence. It is a compact, English version of the rubric in
  [`latest_grading_criteria.md`](../latest_grading_criteria.md).
- **VLM baselines.** The raw-VLM baselines in the README results table
  (Cosmos-Reason1-7B, Qwen3.5-9B, Qwen3-VL-2B-Instruct) use the VLM itself as the
  classifier with this prompt. They can be reproduced by running the same script
  with `--model-dir` pointing at the respective model and treating
  `not_normal_core` as a long-tail prediction.

```
You are given 6 synchronized camera images from an autonomous vehicle at a single timestamp. The cameras are provided in this order: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT.

Your task: assign exactly ONE label to this multi-camera frame.

Definition - normal_core:
A simple driving scene where, based on the visible static and weakly-dynamic information, the ego vehicle does NOT need to take any defensive action beyond routine driving. Concretely, a sample is normal_core only if ALL of the following hold:
- The main road and the relevant traffic participants are clearly visible.
- Lighting and weather do not significantly degrade perception (no night, no strong glare, no rain, no wet reflective surfaces, no fog, no low visibility).
- The ego vehicle has a reasonable safe buffer: no close lead vehicle, no visible brake lights of the lead vehicle, no congestion ahead, no sign of impending deceleration.
- The road ahead is clear: no construction zone, no cones, no barriers, no closure, no abnormally parked vehicles, no lane-intruding obstacles.
- Pedestrians, cyclists, and other vulnerable road users are in routine positions, not near the ego path, and not occluded in a way that might hide risk.
- Large vehicles or buildings do not severely occlude key risk areas.
- The road structure is clear (no unusually complex intersection, no ambiguous lane boundaries, no abnormal topology).
- No camera is severely blurred, occluded, over-exposed, under-exposed, or otherwise abnormal.
- Nothing visible requires extra deceleration, yielding, increased following distance, or other conservative behavior.

If the visible information clearly indicates ANY of the exclusion conditions above (degraded perception conditions, close lead / braking / congestion, VRUs near ego path or occluded, construction / obstacles / abnormal parking, complex or ambiguous road structure, abnormal image quality, or anything requiring extra defensive driving), the sample is NOT normal_core.

If a single-frame multi-camera view is genuinely insufficient to make a stable judgement (strongly ambiguous even after careful inspection), choose uncertain. Do NOT use uncertain as a hedge when the answer is actually clear.

Important: do NOT infer temporal events (no cut-in, no sudden braking, no jaywalking across time). Reason only from what is statically visible in this single multi-camera frame.

Output format (strict):
1. First, write 2 to 4 short sentences analysing the visible static driving-relevant factors (lighting, weather, visibility, road layout, traffic participants, obstacles, construction, image quality).
2. Then, on a new line, output exactly:
Final answer: X
where X is one of:
A (normal_core)       - high-confidence normal scene
B (not_normal_core)   - clearly requires extra defensive driving, or has static difficult factors
C (uncertain)         - single-frame information is genuinely insufficient
```

## 2. Scene-description prompt (feature extraction for the SAE)

Source: `SCENE_PROMPT` in [`scripts/extract.py`](../scripts/extract.py) (the default of `--prompt`).

For the SAE-based method the VLM is *not* asked to classify the sample as
long-tail or normal. Instead, this neutral scene-description prompt is used so
that the hidden states captured by `extract.py` reflect general scene
understanding rather than a fixed category list; the SAE scripts
(`scripts/sae_abstopk_tail_reward.py` and the two ablations) are trained on those
hidden states.

```
You are given 6 synchronized camera images from the ego vehicle at a single timestamp. The cameras are provided in this order: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT.

Assume that you are the driver of the ego vehicle and need to understand the surrounding driving environment from these multi-camera views.

Your task is to provide a concise driving-scene description that covers the following aspects:
- Overall scene type, such as urban road, highway, residential area, intersection, or parking area.
- Weather and lighting conditions, including rain, fog, snow, night, glare, low visibility, or wet reflective roads.
- Road layout and traffic conditions, including lane structure, intersections, congestion, road boundaries, and traffic flow.
- Dynamic objects, including vehicles, pedestrians, cyclists, scooters, and other vulnerable road users.
- Potential hazards or important interactions that may affect the ego vehicle.
- Any visible risky situations or behaviors that may require cautious driving.

Provide the description in one paragraph with fewer than 150 words.
```
