"""Portable cached explorer integrity and browser-script contracts."""

import json
import copy
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import export_offline_explorer as exporter  # noqa: E402
from export_offline_explorer import (  # noqa: E402
    ROOT, build_payload, digest, reference_from_scope, render_html, script_json,
)
from prepare_driving import post_ai_review_scope  # noqa: E402
from tail_activations import CAMERAS, save_activations  # noqa: E402


def payload(reviewed=False):
    images = {camera: "samples/" + camera + "/fixture.jpg" for camera in CAMERAS}
    frame = {"sample_token": "fixture-token", "scene": "fixture-scene",
             "active_tail_units": 1, "prediction": "long_tail", "tail_score": 2.0,
             "cameras": list(CAMERAS), "label": "normal", "split": "test",
             "images": images, "comparisons": [], "units": [{"unit": 0, "activation": 2.0}]}
    scope = (post_ai_review_scope() if reviewed else
             {"label_protocol": "ai-visual-pilot", "human_reviewed": False})
    unit_frame = {"sample_token": "fixture-token", "activation": 2.0}
    return {"schema": "offline-trained-explorer-v2",
            "reference": reference_from_scope(scope),
            "summary": {"frames": 1, "label_scope": scope}, "label_scope": scope,
            "comparison": [{"method": "fixture <script>", "ap": 0.5,
                            "precision_at_k": 0.5, "target_status": "target_unreachable",
                            "fixed_precision": 0.5, "fixed_recall": 0.5,
                            "operating_precision": 0.0, "operating_recall": 0.0,
                            "validation_threshold": None,
                            "n": 1, "test_scenes": 1, "requested_k": 1, "seed": 42}],
            "bookmarks": [{"sample_token": "fixture-token", "title": "fixture"}],
            "transfer_id": "fixture-transfer", "frames_per_scene": 2, "expected_part_count": 4,
            "rubric": "fixture rubric", "rubric_sha256": "b" * 64,
            "expected_frames": [{"sample_token": "fixture-token", "scene_token": "scene",
                                 "timestamp": 1, "images": images}],
            "expected_images": {path: "a" * 64 for path in images.values()},
            "expected_metadata": {"scene": "c" * 64}, "frames": {"fixture-token": frame},
            "unit_list": [{"unit": 0, "frames": 1}],
            "units": {"0": {"unit": 0, "frames": [unit_frame]}},
            "manifest_sha256": "d" * 64, "checkpoints": {}, "source_sha256": {},
            "sae_checkpoints_identical": True,
            "operating_rules": {"fixture <script>": {
                "test_candidates": 0, "fixed_rule": {"rule": "score >= 0.5"},
                "operating_point": {"status": "target_unreachable", "target_precision": 0.9},
                "test": {"precision": 0.0, "recall": 0.0}}},
            "review_csv": "sample_token,reviewer_label\nfixture-token,\n"}


def test_script_json_cannot_terminate_a_script_element():
    value = {"untrusted": "</script><img src=x>&\u2028\u2029"}
    encoded = script_json(value)
    assert "</script>" not in encoded and "<img" not in encoded
    assert json.loads(encoded) == value


@pytest.mark.parametrize("reviewed", [False, True])
def test_render_keeps_images_and_network_outside_the_export(reviewed):
    source = render_html(payload(reviewed), (ROOT / "tools/review_upload.html").read_text())
    assert "connect-src 'none'" in source
    assert "fetch(" not in source and "data:image/" not in source
    assert "fixture &lt;script&gt;" in source
    assert source.count('<script id="') == 4


@pytest.mark.parametrize("reviewed", [False, True])
def test_local_api_and_expected_image_binding(tmp_path, reviewed):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for offline browser-script contract verification")
    source = render_html(payload(reviewed), (ROOT / "tools/review_upload.html").read_text())
    (tmp_path / "demo.html").write_text(source)
    (tmp_path / "verify.js").write_text(r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const html=fs.readFileSync(process.argv[2],'utf8');
const scripts=Object.fromEntries([...html.matchAll(
 /<script id="([^"]+)">([\s\S]*?)<\/script>/g)].map(x=>[x[1],x[2]]));
for(const [name,code]of Object.entries(scripts))new vm.Script(code,{filename:name});
for(const name of ['offline-payload','review-loader','offline-runtime'])
 vm.runInThisContext(scripts[name]);
const api=DrivingOffline,p=api.payload;
const model={manifest:{transfer_id:p.transfer_id,frames_per_scene:p.frames_per_scene,part_count:4,
 metadata_sha256:p.expected_metadata,
 files:Object.entries(p.expected_images).map(([path,output_sha256])=>({path,output_sha256}))},
 selected:p.expected_frames,rubric_sha256:p.rubric_sha256,
 images:new Map(Object.keys(p.expected_images).map(path=>[path,new Blob(['fixture'])]))};
assert.equal(api.validateLoaded(model),true);
assert.equal(api.api('api/unit/0?top=1').frames[0].sample_token,'fixture-token');
assert.equal(api.api('api/sample/fixture-token').units[0].activation,2.0);
assert.equal(api.api('api/search?q=fixture').length,1);
assert.throws(()=>api.api('api/sample/unknown'));
const bad=structuredClone(model);bad.manifest.transfer_id='different';
assert.throws(()=>api.validateLoaded(bad));
const wrongHash=structuredClone(model);wrongHash.manifest.files[0].output_sha256='0'.repeat(64);
assert.throws(()=>api.validateLoaded(wrongHash));
const wrongParts=structuredClone(model);wrongParts.manifest.part_count=3;
assert.throws(()=>api.validateLoaded(wrongParts));
const wrongMetadata=structuredClone(model);
wrongMetadata.manifest.metadata_sha256.scene='0'.repeat(64);
assert.throws(()=>api.validateLoaded(wrongMetadata));
const wrongCamera=structuredClone(model);
wrongCamera.selected[0].images.CAM_FRONT='samples/wrong.jpg';
assert.throws(()=>api.validateLoaded(wrongCamera));
""")
    result = subprocess.run([node, str(tmp_path / "verify.js"), str(tmp_path / "demo.html")],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture(params=[False, True], ids=["ai-73", "human-77"])
def source_bundle(tmp_path, monkeypatch, request):
    """A cached result bundle with 80 source frames, without training or real pixels."""
    reviewed = request.param
    scope = payload(reviewed)["label_scope"]
    count = 77 if reviewed else 73
    prepared, samples, comparison = (tmp_path / name for name in
                                     ("prepared", "samples", "comparison_runs"))
    selected, raw = [], []
    for index in range(80):
        token = f"fixture-{index:02d}"
        images = {camera: camera + "/" + token + ".jpg" for camera in CAMERAS}
        records = {}
        for camera, filename in images.items():
            path = samples / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic path fixture, no image pixels")
            records[camera] = {"filename": "samples/" + filename, "sha256": digest(path)}
        selected.append({"sample_token": token, "scene_token": f"scene-{index // 8}",
                         "timestamp": index, "images": images})
        raw.append({"sample_token": token, "image_files": records})
    for name, value in (("selected_samples.json", selected), ("raw_manifest.json", raw)):
        write_json(prepared / name, value)
    (prepared / "rubric.md").write_text("fixture rubric", encoding="utf-8")
    prepared_hashes = {name: digest(prepared / name) for name in
                       ("selected_samples.json", "raw_manifest.json", "rubric.md")}
    write_json(prepared / "prepare_metadata.json", {
        "included_frames": 80, "prepared_files": prepared_hashes,
        "source_metadata": {"scene": {"sha256": "c" * 64}},
        "rubric_sha256": digest(prepared / "rubric.md")})
    imported = tmp_path / "import.json"
    write_json(imported, {"status": "completed", "labels_created": False, "parts": 4,
                          "transfer_id": "same-original-four-parts", "frames_per_scene": 8,
                          "selected_samples": 80})
    methods = ("logistic", "mlp", "sae_abstopk_none", "sae_abstopk_norm")
    table = []
    labels = [index % 2 for index in range(count)]
    labels[-3:] = [0, 1, 1]
    splits = ["train"] * (count - 29) + ["val"] * 14 + ["test"] * 15
    activations = np.array([[1 + index / 100, 0] for index in range(count)], dtype=np.float32)
    for method in methods:
        directory = comparison / method
        directory.mkdir(parents=True)
        predictions = []
        status = "selected" if reviewed and method != "logistic" else "target_unreachable"
        threshold = (0.99 if method == "mlp" else 1 + (count - 2) / 100) if status == (
            "selected") else None
        for index in range(count):
            score = float(activations[index, 0]) if method.startswith("sae_") else 0.8
            if method == "mlp" and index >= count - 3:
                score = [0.9, 0.1, 0.2][index - count + 3]
            prediction = 1 if method.startswith("sae_") else int(score >= 0.5)
            operating = int(threshold is not None and score >= threshold)
            predictions.append({"sample_token": selected[index]["sample_token"],
                                "scene_token": selected[index]["scene_token"],
                                "split": splits[index], "label": labels[index], "score": score,
                                "prediction": prediction, "operating_prediction": operating,
                                "operating_error": exporter.error_type(labels[index], operating)})
        write_json(directory / "predictions.json", predictions)
        write_json(directory / "evaluation_manifest.json", {"fixture": scope})
        checkpoint = directory / "checkpoint.bin"
        checkpoint.write_bytes(b"synthetic saved weights")
        write_json(directory / "run_metadata.json", {
            "label_scope": scope, "predictions_sha256": digest(directory / "predictions.json"),
            "manifest_sha256": digest(directory / "evaluation_manifest.json"),
            "checkpoint": {"path": str(checkpoint), "sha256": digest(checkpoint)}})
        candidates = sum(row["operating_prediction"] for row in predictions if row["split"] == (
            "test"))
        point = {"status": status, "threshold": threshold, "target_precision": 0.9}
        test = {"precision": float(bool(candidates)), "recall": candidates / 10,
                "confusion": {"tp": candidates, "fp": 0}}
        write_json(directory / "evaluation.json", {
            "label_scope": scope, "operating_point": point,
            "fixed_rule": {"rule": "c_tail >= 1" if method.startswith("sae_") else
                           "score >= 0.5"}, "splits": {"test": {"operating": test}}})
        if method.startswith("sae_"):
            write_json(directory / "metrics.json", {"label_scope": scope,
                                                    "decision": {"eta": 0.01}})
            save_activations(directory / "activations.npz", activations, selected[:count],
                             labels=np.asarray(labels), splits=splits)
        table.append({"method": method, "ap": 0.9, "precision_at_k": 1.0,
                      "target_status": status, "validation_threshold": threshold,
                      "fixed_precision": 0.5, "fixed_recall": 1.0,
                      "operating_precision": test["precision"], "operating_recall": test["recall"],
                      "n": 15, "test_scenes": 2, "requested_k": 4, "seed": 42})
    write_json(comparison / "comparison/main_table.json", table)
    monkeypatch.setattr(exporter, "collect_runs", lambda directories: (copy.deepcopy(table), []))
    return {"args": (comparison / "sae_abstopk_norm", comparison, prepared, samples, imported),
            "reviewed": reviewed, "count": count, "scope": scope}


def test_both_scopes_bind_cached_rows_to_same_original_selection(source_bundle):
    result = build_payload(*source_bundle["args"])
    assert len(result["frames"]) == result["summary"]["frames"] == source_bundle["count"]
    assert len(result["expected_frames"]) == 80 and len(result["expected_images"]) == 480
    assert result["transfer_id"] == "same-original-four-parts"
    assert result["expected_part_count"] == 4
    assert result["label_scope"] == source_bundle["scope"]
    assert [row["error_type"] for row in result["bookmarks"]] == ["fp", "fn", "fn"]
    assert all(row["method"] == "mlp" for row in result["bookmarks"])
    for frame in result["frames"].values():
        assert len(frame["comparisons"]) == 4
        assert all(path in result["expected_images"] for path in frame["images"].values())
    source = render_html(result, (ROOT / "tools/review_upload.html").read_text())
    if source_bundle["reviewed"]:
        assert 'data-reference-kind="human_ai_exposed"' in source
        assert ("human review pending" not in source
                and "Provisional model-label pilot;" not in source)
        assert "Model-generated label pilot:" not in source and "AP (model reference)" not in source
        assert ("Provisional AI-label pilot;" not in source
                and "AI visual-label pilot:" not in source)
        assert "AP (AI reference)" not in source
        assert "AP (human review reference)" in source
        assert "not independent human ground truth" in source
        assert result["operating_rules"]["mlp"]["operating_point"]["status"] == "selected"
        assert result["operating_rules"]["mlp"]["test_candidates"] == 0
        assert result["operating_rules"]["sae_abstopk_norm"]["test_candidates"] == 2
    else:
        assert 'data-reference-kind="ai_provisional"' in source
        assert "human review pending" in source and "AP (model reference)" in source


@pytest.mark.parametrize("location", ["run_metadata.json", "evaluation.json", "metrics.json"])
def test_changed_or_missing_artifact_scopes_are_rejected(source_bundle, location):
    run = source_bundle["args"][0]
    document = json.loads((run / location).read_text())
    document["label_scope"] = None
    write_json(run / location, document)
    with pytest.raises(ValueError, match="scope|provenance"):
        build_payload(*source_bundle["args"])


@pytest.mark.parametrize("field,message", [
    ("labels", "labels or splits"), ("splits", "labels or splits"),
    ("images", "image mapping"), ("scores", "saved SAE predictions"),
])
def test_activation_cache_must_match_saved_predictions(source_bundle, field, message):
    run = source_bundle["args"][0]
    store = exporter.ActivationStore(run)
    dense = store.dense()
    if field == "labels":
        store.labels[0] = 1 - store.labels[0]
    elif field == "splits":
        store.splits[0] = "test"
    elif field == "images":
        store.samples[0]["images"]["CAM_FRONT"] = store.samples[0]["images"]["CAM_BACK"]
    else:
        dense[0, 0] *= 1.1
    save_activations(run / "activations.npz", dense, store.samples,
                     labels=store.labels, splits=store.splits)
    with pytest.raises(ValueError, match=message):
        build_payload(*source_bundle["args"])


def test_comparison_cannot_mix_reference_scopes(source_bundle):
    other = (payload(not source_bundle["reviewed"])["label_scope"])
    directory = source_bundle["args"][1] / "mlp"
    for name in ("run_metadata.json", "evaluation.json"):
        document = json.loads((directory / name).read_text())
        document["label_scope"] = other
        write_json(directory / name, document)
    with pytest.raises(ValueError, match="Mixed reference scopes"):
        build_payload(*source_bundle["args"])


def test_changed_prediction_hash_is_rejected(source_bundle):
    run = source_bundle["args"][0]
    path = run / "predictions.json"
    document = json.loads(path.read_text())
    document[0]["score"] += 1
    write_json(path, document)
    with pytest.raises(ValueError, match="Run source hash differs"):
        build_payload(*source_bundle["args"])


@pytest.mark.parametrize("change", [
    {"human_reviewed": False}, {"independent_human_review": True},
    {"prior_ai_labels_exposed": False}, {"human_ground_truth_verified": True},
    {"official_benchmark": True}, {"reviewer_count": 2}, {"human_reviewed": 1},
    {"review_context": "independent"}, {"label_protocol": "ai-visual-pilot"},
])
def test_manual_scope_cannot_claim_independence_or_hide_prior_exposure(change):
    with pytest.raises(ValueError):
        reference_from_scope({**post_ai_review_scope(), **change})


def test_scope_cannot_be_missing_or_mix_ai_and_manual_claims():
    for scope in (None, {}, {"label_protocol": "ai-visual-pilot"},
                  {"label_protocol": "manual", "human_reviewed": True},
                  {"label_protocol": "ai-visual-pilot", "human_reviewed": False,
                   "review_context": "post_ai_exposure_single_reviewer"}):
        with pytest.raises(ValueError):
            reference_from_scope(scope)
    for key in post_ai_review_scope():
        scope = post_ai_review_scope()
        del scope[key]
        with pytest.raises(ValueError):
            reference_from_scope(scope)


def test_render_rejects_reference_descriptor_that_disagrees_with_scope():
    result = payload(True)
    result["reference"] = reference_from_scope(payload(False)["label_scope"])
    with pytest.raises(ValueError, match="Payload reference scope differs"):
        render_html(result, (ROOT / "tools/review_upload.html").read_text())


@pytest.mark.parametrize("name,count", [("ai-pilot", 73), ("human-ai-exposed-20261026", 77)])
def test_available_saved_runs_rebuild_with_original_zip_hashes(name, count):
    base = ROOT / "output/real-cpu-mini-20261023"
    comparison = base / name / "comparison_runs"
    if not (comparison / "sae_abstopk_norm/activations.npz").exists():
        pytest.skip("Authorised local saved runs are unavailable")
    cases = ROOT / "results" / (name if name != "ai-pilot" else
                                "cpu-mini-ai-pilot-20261023") / "failure_cases.json"
    result = build_payload(comparison / "sae_abstopk_norm", comparison, base / "prepared",
                           ROOT / "dataset/nuscenes-mini-user-20261023/samples",
                           base / "upload_import.json", cases)
    assert result["summary"]["frames"] == count
    assert len(result["expected_images"]) == 480 and len(result["expected_frames"]) == 80
    assert result["transfer_id"] == "5962d6a1-8e6c-4f25-9d45-02339ecbcd12"
    assert result["sae_checkpoints_identical"]
    if name != "ai-pilot":
        sae = next(row for row in result["comparison"] if row["method"] == "sae_abstopk_norm")
        assert sae["ap"] == pytest.approx(0.9353174603174602)
        assert sae["precision_at_k"] == 1 and sae["operating_recall"] == 0.2
        assert result["operating_rules"]["sae_abstopk_norm"]["test_candidates"] == 2
