# Prompt Templates

These are the exact prompt templates used in the project, for (1) the VLM
baseline classifiers and (2) the scene-description prompt used to elicit the
internal VLM representations that the SAE is trained on.

## 1. Long-Tail Classification Prompt (VLM baselines)

Used for the VLM-based baselines (Cosmos-Reason1-7B, Qwen3.5-9B,
Qwen3-VL-2B-Instruct) in `README.md`'s baseline comparison table: each VLM is
given this prompt directly and asked to classify the scene itself.

```
You are given 6 synchronized camera images from an autonomous vehicle at a single timestamp. The
cameras are provided in this order: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK,
CAM_BACK_LEFT, CAM_BACK_RIGHT.

Your task: assign exactly ONE label to this multi-camera frame.

Definition - normal_core:
A simple driving scene where, based on the visible static and weakly-dynamic information, the ego
vehicle does NOT need to take any defensive action beyond routine driving.

Concretely, a sample is normal_core only if ALL of the following hold:
- Lighting and weather do not significantly degrade perception (no night, no strong glare, no rain, no wet reflective
  surfaces, no fog, no low visibility).
- The ego vehicle has a reasonable safe buffer: no close lead vehicle, no visible brake lights of the
  lead vehicle, no congestion ahead, no sign of impending deceleration.
- The road ahead is clear: no construction zone, no cones, no barriers, no closure, no abnormally
  parked vehicles, no lane-intruding obstacles.
- Pedestrians, cyclists, and other vulnerable road users are in routine positions, not near the ego
  path, and not occluded in a way that might hide risk. - Large vehicles or buildings do not
  severely occlude key risk areas.
- The road structure is clear (no unusually complex intersection, no ambiguous lane boundaries, no
  abnormal topology).
- No camera is severely blurred, occluded, over-exposed, under-exposed, or otherwise abnormal.
- Nothing visible requires extra deceleration, yielding, increased following distance, or other
  conservative behavior.

If the visible information clearly indicates ANY of the exclusion conditions above (degraded
perception conditions, close lead / braking / congestion, VRUs near ego path / occluded,
construction / obstacles / abnormal parking, complex or ambiguous road structure, abnormal image
quality, or anything requiring extra defensive driving), the sample is NOT normal_core.

If a single-frame multi-camera view is genuinely insufficient to make a stable judgement (strongly
ambiguous even after careful inspection), choose uncertain. Do NOT use uncertain as a hedge when
the answer is actually clear.

Important: do NOT infer temporal events (no cut-in, no sudden braking, no jaywalking across time).
Reason only from what is statically visible in this single multi-camera frame.

Output format (strict):
1. First, write 2 to 4 short sentences analysing the visible static driving-relevant factors
   (lighting, weather, visibility, road layout, traffic participants, obstacles, construction,
   image quality).
2. Then, on a new line, output exactly:
Final answer: X
where X is one of:
A (normal_core) - high-confidence normal scene
B (not_normal_core) - clearly requires extra defensive driving, or has static difficult factors
C (uncertain) - genuinely ambiguous from a single frame
```

This is the same `normal_core` rubric implemented in `latest_grading_criteria.md`
and used by `scripts/annotate_normal_core.py`, repurposed here as a direct
classification prompt for the VLM baselines so they can be scored against the
same long-tail/normal ground truth as the SAE method.

## 2. Feature-Extraction Prompt (SAE method)

For the SAE-based method, the VLM is *not* prompted to directly classify a
sample as long-tail or normal. Instead, this scene-description prompt is used
to elicit rich internal driving-scene representations from the VLM (see
`scripts/extract.py`); the resulting hidden states are what `scripts/pipeline.py`
and `scripts/sae_abstopk_tail_reward.py` train the SAE on.

```
You are given 6 synchronized camera images from the ego vehicle at a single timestamp.
The cameras are provided in this order: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK,
    CAM_BACK_LEFT, CAM_BACK_RIGHT.

Assume that you are the driver of the ego vehicle and need to understand the surrounding driving
    environment from these multi-camera views.

Your task is to provide a concise driving-scene description that covers the following aspects:
- Overall scene type, such as urban road, highway, residential area, intersection, or parking area.
- Weather and lighting conditions, including rain, fog, snow, night, glare, low visibility, or wet
  reflective roads.
- Road layout and traffic conditions, including lane structure, intersections, congestion, road
  boundaries, and traffic flow.
- Dynamic objects, including vehicles, pedestrians, cyclists, scooters, and other vulnerable road
  users.
- Potential hazards or important interactions that may affect the ego vehicle.
- Any visible risky situations or behaviors that may require cautious driving.

Provide the description in one paragraph with fewer than 150 words.
```
