# Additional driving-scene data

This source inventory obtained **no new scenes or images**. Current coverage
remains 89 timestamps, 534 original camera images and 12 scenes. No new labels,
model fits, predictions or evaluation results were produced.

## Existing public sample is exhausted

The complete [DriveLM author tree at `1de72a74`](https://github.com/OpenDriveLab/DriveLM/tree/1de72a74b257e5373400fa68239e99bd5d20580a)
contains the same 54 camera JPEGs already used: nine timestamps from two
scenes. It supplies no additional complete six-camera scene groups at that
revision. The original mini contributes the other ten scenes.

## Larger publisher-provided candidates

The [author's download instructions](https://github.com/OpenDriveLab/DriveLM/blob/1de72a74b257e5373400fa68239e99bd5d20580a/challenge/README.md)
identify a larger nuScenes-derived subset:

| File | Publisher-advertised size | Current status |
|---|---:|---|
| `drivelm_nus_imgs_train.zip` | 3.48 GB | Not downloaded |
| `drivelm_nus_imgs_val.zip` | 705 MB | Not downloaded |
| `v1_1_train_nus.json` | 193 MB | Not downloaded |
| `v1_1_val_nus_q_only.json` | 9.86 MB | Not downloaded |

These sizes come from the [publisher's listing](https://huggingface.co/datasets/OpenDriveLab/DriveLM/tree/main),
not measured transfers. That route requires an account, acceptance of its
conditions and sharing account information. Those steps were not performed.
The author's Google Drive mirrors are also candidates; their current access,
object identities and Range support remain unverified. The inventory does not
assign scene counts to archives whose metadata and images were not inspected.

The [official nuScenes route](https://www.nuscenes.org/download) remains another
source subject to registration and dataset terms. Downloading the same mini
again would not add scenes. Original nuScenes images retain their dataset
license and attribution requirements; DriveLM's code license does not relicense
them. DriveLM's selected keyframes also do not represent a random sample of
all driving conditions.

## Acquisition gate for a new study

1. Resolve an authorized access route and inspect its metadata first.
2. Exclude overlap with all 12 existing scenes; recover original log identities
   before claiming log-disjointness.
3. Verify complete six-camera groups and input hashes in a small image batch.
4. Measure download, CPU extraction and human-review costs before committing
   to the larger scene budget.
5. Keep reference labels and predictions hidden until new human judgments are
   submitted. Freeze the study split and decision rules before evaluation.

The proposed 30–50 additional scenes remain a planning target. The
[machine-readable inventory](../results/data_source_inventory/source_inventory.json)
records the exact existing scene IDs, source links and unresolved access fields.
