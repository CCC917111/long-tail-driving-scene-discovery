# Transfer a local mini archive as small parts

Use the self-contained [`tools/prepare_upload.html`](../tools/prepare_upload.html)
when an authorized `v1.0-mini.tgz` is already on your computer and the execution
workspace cannot download it. The tool has no external libraries or network
requests. Download the HTML, then open it locally in desktop Chrome or Edge.
No Python installation or cloud account is needed for this local step.

## User steps

1. Choose the existing `v1.0-mini.tgz` file; keep the original archive.
2. Leave the two-frames-per-scene setting for the initial small smoke run, then
   start extraction. The browser reads the archive in two streaming passes;
   the time depends on archive size and your computer. Keep the tab open.
3. When extraction finishes, download every `driving-mini-part-…zip` file.
   Each part is at most 24 MiB, including its metadata. Upload those small ZIPs
   to the conversation; the original multi-gigabyte archive stays local.

The initial selection uses two timestamps per scene (normally 20 timestamps
over the mini set's ten scenes), each with six unmodified camera images. The
optional eight-frame setting is larger. Selection depends on timestamps and
scene membership, never on labels. This is a custom proof of concept; mini has
no official test split.

On 2026-10-23, a user selected the eight-frame option and uploaded four parts
generated locally from the 4,168,148,189-byte archive. The workspace imported all
80 timestamps, 480 camera images and five original metadata tables successfully,
then completed real CPU feature extraction. The local browser's processing of
the original archive was not directly observed or timed here; this evidence is
the successful generated-part import, not an official archive checksum check.

## Recorded evidence and import

Original bytes of the five metadata tables are preserved, individually gzip
compressed for transport. Sweeps, LiDAR and unselected camera images are omitted.
Every part carries the same selection/file inventory and per-file SHA256 hashes.
The original archive name and size are recorded; its multi-gigabyte SHA256 is
not computed, and neither those fields nor local hashes certify its origin.

Import only after all parts arrive:

```bash
bash scripts/cloud.sh import-mini-parts DATA=dataset/nuscenes OUTPUT=output/real-cpu \
  PARTS="/path/to/driving-mini-part-001-of-002.zip /path/to/driving-mini-part-002-of-002.zip"
bash scripts/cloud.sh prepare-real DATA=dataset/nuscenes OUTPUT=output/real-cpu \
  FRAMES_PER_SCENE=2
```

Use the frame budget recorded in the handoff when running preparation; use 8
only if the source handoff used 8. Import checks completeness, file hashes,
original metadata and the exact selected camera frames before publishing a new
data directory. It refuses to replace an existing nonempty dataset. No labels
are created. After import, continue with [CPU extraction and manual review](cpu-route.md).

## Review the same parts locally

Download [`tools/review_upload.html`](../tools/review_upload.html), open it in
desktop Chrome/Edge and select all the same ZIP parts together. Click
`校验并打开`, fill the actual reviewer's identity and inspect all six views for
each timestamp. The original frozen rubric is embedded with its SHA256. Choose
`normal_core`, `not_normal_core` or `uncertain`, then export `reviewed.csv` and
return that small CSV for strict label import. No split, scene description,
model prediction or prefilled label is shown.

The page stores decisions in local browser storage and supports CSV resume.
Export a backup before closing; local storage can be unavailable or cleared.
Images remain on the user's device, and the tool makes no network requests.
Only source code is published at the
[verified GitHub tool revision](https://github.com/CCC917111/long-tail-driving-scene-discovery/blob/41e3f32aca66ac57b16fbb35686060a0fd8aabcc/tools/review_upload.html).

The actual four parts passed the reviewer's parser and matched the frozen
80-timestamp preparation; all 480 image Blobs were byte-identical to the
imported originals. The exported all-pending CSV was accepted by the Python
importer with zero eligible labels and training readiness false. CSV roundtrip
checks used synthetic decisions only. Actual browser clicking, local-storage
behavior and download interaction remain unverified in this sandbox.
