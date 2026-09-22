# Normal Core Definition for Static Driving Frames

This document defines the **normal core** used in this project for single-frame, multi-view autonomous-driving scene analysis. The definition is meant to select ordinary scenes; it does not try to enumerate every possible long-tail scene.

## Motivation

In a single-frame setting many dynamic risks cannot be judged reliably, e.g. hard braking, cut-ins or sudden crossings. This project therefore only considers static or weakly-dynamic driving conditions that are visible in the current multi-view frame.

We do not define "normal" by frequency, and we do not equate "normal" with the most common scenes in the dataset. A normal scene should correspond to a routine, clear, low-ambiguity situation from the perspective of autonomous driving.

## Definition

**Normal core**: a simple driving scene in which, judging only from the information visible in the current single-frame multi-view images, the vehicle does not need to take any defensive action beyond its routine driving policy.

In other words, if the scene does not clearly require the vehicle to slow down, increase its following distance, keep a larger lateral safety margin, be more ready to yield, or otherwise drive more conservatively because of the environment, visibility, road structure, traffic participants or static obstacles, it can be treated as normal core.

## Inclusion Criteria

A sample can usually be considered normal core if it satisfies the following:

- The image is clear overall; the main road area and the traffic participants are recognizable.
- Lighting and weather do not significantly reduce perception reliability.
- The ego vehicle has a reasonable safety space around it and keeps normal distances to vehicles ahead and to the side.
- The road ahead is clearly passable, without obvious blockage, construction, closure or abnormal static obstacles.
- Pedestrians, cyclists, vehicles and other traffic participants are in routine positions and states.
- No significant occlusion hides key risk areas.
- The visible information does not call for extra deceleration, avoidance, yielding or conservative car-following.

## Exclusion Criteria

A sample should not be placed in the normal core if any of the following clearly applies:

- Conditions that increase perception uncertainty: low light, night, strong glare, rain, wet and strongly reflective road surfaces, fog or other low-visibility conditions.
- The lead vehicle is too close, its brake lights are clearly on, there is congestion ahead, or there are other signs that early deceleration is needed.
- Pedestrians, cyclists or other vulnerable road users are close to the ego path, or are occluded in a way that may affect driving.
- Large vehicles, parked vehicles, buildings, etc. severely occlude key areas.
- Construction zones, traffic cones, barriers, road closures, abnormally parked vehicles or obstacles in the lane.
- Complex or unclear road structure, e.g. complex intersections, unclear lane boundaries or abnormal road topology.
- A camera is severely blurred, occluded, over-exposed, under-exposed, or the image quality is otherwise clearly abnormal.
- Any situation that, based on the single-frame visible information, requires extra defensive driving.

## Annotation Protocol

Annotation uses three classes instead of a forced binary decision:

- `normal_core`: a high-confidence normal scene that can be used to learn the normal manifold.
- `not_normal_core`: a scene that clearly requires extra defensive driving or contains static difficulty factors.
- `uncertain`: the single frame does not contain enough information and the annotator cannot make a stable judgement; such samples are not used for training.

When training an SAE or a one-class model, only `normal_core` samples should be treated as normal. `not_normal_core` is used as the long-tail (positive) class in the supervised SAE experiments, while `uncertain` samples are excluded from training and evaluation (see `scripts/sae_common.py`) and kept for later discovery, ranking and manual analysis.

## Relation to Long-Tail Discovery

This project does not attempt to enumerate all long-tail types in advance. We only define a high-confidence normal core and let the model learn the representation distribution of ordinary driving scenes. Samples that deviate from this normal core may correspond to long-tail cases, hard samples or abnormal capture conditions, and are confirmed through interpretable features, top-activating samples and manual review.

## Prompt Principle

When Cosmos-Reason1 is used to extract representations, the prompt should neither give the normal/long-tail definition nor ask the model whether the scene is long-tail. A neutral driving-scene analysis prompt should be used instead, so that the model attends to visible driving-relevant factors rather than performing a classification task. The prompt actually used by `scripts/extract.py` is listed in [`docs/prompt_templates.md`](docs/prompt_templates.md). It asks for a description of the scene, including visible hazards, but never provides the normal/long-tail definition and never asks for a label. A minimal example of the principle:

```text
Analyze this multi-camera autonomous driving frame.
Describe visible static driving-relevant factors, including lighting, weather,
visibility, road layout, traffic participants, obstacles, construction elements,
and unusual static object states. Do not infer temporal events.
```
