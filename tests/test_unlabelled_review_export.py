"""Unlabelled export integrity and review exposure bookkeeping."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from export_unlabelled_review import build_payload, render_html, script_json  # noqa: E402
from tail_activations import CAMERAS, save_activations  # noqa: E402


@pytest.fixture
def batch(tmp_path):
    run, samples = tmp_path / "screening", tmp_path / "samples"
    run.mkdir()
    paths, records = {}, []
    for camera in CAMERAS:
        path = samples / camera / "frame.jpg"
        path.parent.mkdir(parents=True)
        Image.new("RGB", (8, 8), (80, 120, 160)).save(path)
        paths[camera] = camera + "/frame.jpg"
        records.append({"path": "samples/" + paths[camera],
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "source_url": "https://raw.githubusercontent.com/OpenDriveLab/DriveLM/"
                        + "a" * 40 + "/data/nuscenes/samples/" + paths[camera]})
    save_activations(run / "activations.npz", np.array([[1.2, 0]], dtype=np.float32),
                     [{"sample_token": "new-frame", "scene_token": "new-scene", "images": paths}])
    (run / "screening_metadata.json").write_text(json.dumps({
        "kind": "unlabelled-screening", "decision": {"rule": "c_tail >= 1", "eta": 0.01},
        "checkpoint_sha256": "b" * 64, "standardizer_sha256": "c" * 64,
        "model_training_label_scope": {"review_context": "post_ai_exposure_single_reviewer"}}))
    source = tmp_path / "acquisition.json"
    source.write_text(json.dumps({"source_commit": "a" * 40, "images": records}))
    return run, samples, source


def test_payload_keeps_new_batch_unlabelled_and_checks_images(batch):
    run, samples, source = batch
    payload = build_payload(run, samples, source_manifest=source)
    assert payload["scope"] == {
        "has_reference_labels": False, "human_reviewed": False,
        "is_benchmark_evaluation": False, "model_signals_hidden_initially": True}
    assert payload["model_training_label_scope"]["review_context"] == (
        "post_ai_exposure_single_reviewer")
    frame = payload["frames"][0]
    assert "label" not in frame and "split" not in frame
    assert set(frame["images"]) == set(CAMERAS)
    assert frame["images"]["CAM_FRONT"] == "images/CAM_FRONT/frame.jpg"
    assert frame["units"][0]["unit"] == 0
    (samples / "CAM_FRONT/frame.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash differs"):
        build_payload(run, samples, source_manifest=source)


def test_export_rejects_labels_missing_images_and_unpinned_sources(batch):
    run, samples, source = batch
    data = json.loads(source.read_text())
    data["images"][0]["source_url"] = "https://untrusted.example/image.jpg"
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="pinned author"):
        build_payload(run, samples, source_manifest=source, image_mode="source")
    with pytest.raises(ValueError, match="inside the review bundle"):
        build_payload(run, samples, image_prefix="../outside")
    (samples / "CAM_FRONT/frame.jpg").unlink()
    with pytest.raises(ValueError, match="six original"):
        build_payload(run, samples)


def test_labelled_activation_store_is_rejected(batch):
    run, samples, _ = batch
    save_activations(run / "activations.npz", np.ones((1, 2), dtype=np.float32),
                     [{"sample_token": "frame", "scene_token": "scene"}], labels=np.array([1]))
    with pytest.raises(ValueError, match="no reference labels"):
        build_payload(run, samples)


def test_safe_html_and_csv_exposure_transition(batch, tmp_path):
    run, samples, source = batch
    payload = build_payload(run, samples, source_manifest=source)
    payload["model_training_label_scope"]["untrusted"] = "</script><img src=x>\u2028"
    html = render_html(payload)
    assert "</script><img src=x>" not in html
    assert "<section id=\"signals\" hidden>" in html
    assert "connect-src 'none'" in html
    assert "data:image/" not in html
    assert json.loads(script_json({"x": "</script>\u2029"})) == {"x": "</script>\u2029"}
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed for the review CSV runtime check")
    page = tmp_path / "review.html"
    page.write_text(html)
    script = tmp_path / "verify.js"
    script.write_text(r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const html=fs.readFileSync(process.argv[2],'utf8');
const json=html.match(/<script id="payload" type="application\/json">([\s\S]*?)<\/script>/)[1];
const runtime=html.match(/<script id="review-runtime">([\s\S]*?)<\/script>/)[1];
const nodes=new Map();function node(id){if(!nodes.has(id))nodes.set(id,{textContent:'',hidden:false,replaceChildren(){},append(){}});return nodes.get(id)};
global.document={getElementById:id=>id==='payload'?{textContent:json}:node(id),createElement:()=>node('new')};
global.localStorage={getItem:()=>null,setItem(){}};
global.DrivingReviewHeadless=true;vm.runInThisContext(runtime);
const api=DrivingUnlabelledReview;assert.equal(api.state.rows[0].label,'');assert.equal(api.state.exposed,false);
assert.throws(()=>api.decide('normal_core','','notes'));
api.decide('normal_core','Yike','clear, ordinary road');
assert.equal(api.state.rows[0].review_context,'model_hidden_single_reviewer');
assert.equal(api.state.rows[0].model_signal_exposed,false);
api.reveal();assert.equal(api.state.exposed,true);
api.decide('uncertain','Yike','visibility "unclear"');
assert.equal(api.state.rows[0].review_context,'post_model_signal_exposure_single_reviewer');
assert.equal(api.state.rows[0].model_signal_exposed,true);
assert.equal(api.state.rows[0].review_status,'uncertain');
assert.ok(api.csv().includes('visibility ""unclear""'));
assert.equal(api.csv().trim().split('\r\n').length,2);
assert.ok(!api.FIELDS.includes('prediction'));
''')
    result = subprocess.run([node, str(script), str(page)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_six_view_readiness_failures_retry_and_stale_callbacks(batch, tmp_path):
    run, samples, source = batch
    payload = build_payload(run, samples, source_manifest=source)
    payload['frames'].append(dict(payload['frames'][0], sample_token='second-frame'))
    html = render_html(payload)
    assert '<button id="normal" disabled>' in html
    assert '<button id="tail" disabled>' in html
    assert '<button id="uncertain" disabled>' in html
    assert '<button id="clear">' in html
    assert 'id="image-status" role="status" aria-live="polite"' in html
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is needed for the image-readiness runtime check')
    page = tmp_path / 'review.html'
    page.write_text(html)
    script = tmp_path / 'verify-readiness.js'
    script.write_text(r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const html=fs.readFileSync(process.argv[2],'utf8');
const json=html.match(/<script id="payload" type="application\/json">([\s\S]*?)<\/script>/)[1];
const runtime=html.match(/<script id="review-runtime">([\s\S]*?)<\/script>/)[1];
function element(){return {textContent:'',value:'',hidden:false,disabled:false,children:[],attrs:{},
 replaceChildren(){this.children=[]},append(...children){this.children.push(...children)},
 setAttribute(k,v){this.attrs[k]=v}}}
const nodes=new Map();
function node(id){if(!nodes.has(id))nodes.set(id,element());return nodes.get(id)}
global.document={getElementById:id=>id==='payload'?{textContent:json}:node(id),createElement:element};
global.localStorage={getItem:()=>null,setItem(){}};
global.DrivingReviewHeadless=true;vm.runInThisContext(runtime);
const api=DrivingUnlabelledReview,first=api.payload.frames[0].sample_token;
const good={complete:true,naturalWidth:8,naturalHeight:8};
const generation=api.beginImages(first);
assert.equal(node('normal').disabled,true);
assert.equal(node('tail').disabled,true);
assert.equal(node('uncertain').disabled,true);
assert.equal(node('clear').disabled,false);
assert.equal(node('views').attrs['aria-busy'],'true');
assert.throws(()=>api.requireImagesReady());
for(const camera of api.payload.camera_layout.slice(0,5))api.completeImage(first,generation,camera,good);
assert.equal(api.imageState.loaded.size,5);
assert.equal(node('normal').disabled,true);
const last=api.payload.camera_layout.at(-1);
api.completeImage(first,generation,last,{...good,complete:false});
assert.equal(api.imageState.ready,false);
assert.ok(node('image-status').textContent.includes('images 文件夹'));
api.completeImage(first,generation,last,{...good,naturalWidth:0});
assert.equal(api.imageState.ready,false);
api.completeImage(first,generation,last,good);
assert.equal(api.imageState.ready,true);
assert.equal(node('normal').disabled,false);
assert.equal(node('tail').disabled,false);
assert.equal(node('uncertain').disabled,false);
api.requireImagesReady();
const nextGeneration=api.beginImages(first);
for(const camera of api.payload.camera_layout)api.completeImage(first,generation,camera,good);
api.failImage(first,generation,last);
assert.equal(api.imageState.loaded.size,0);
assert.equal(api.imageState.failed.size,0);
assert.equal(api.imageState.ready,false);
for(const camera of api.payload.camera_layout)api.completeImage(first,nextGeneration,camera,good);
assert.equal(api.imageState.ready,true);
node('notes').value='unsaved, typed "reason"\nsecond line';
node('reviewer').value='UI_TEST_ONLY';
api.retryImages();
assert.equal(node('notes').value,'unsaved, typed "reason"\nsecond line');
assert.equal(node('reviewer').value,'UI_TEST_ONLY');
assert.equal(api.state.rows[0].review_status,'pending');
assert.equal(api.state.rows[0].reviewer,'');
assert.equal(api.imageState.ready,false);
const firstImages=node('views').children.map(figure=>figure.children[0]);
assert.equal(firstImages.length,6);
for(const img of firstImages){Object.assign(img,good);img.onload()}
assert.equal(api.imageState.ready,true);
api.state.index=1;
assert.throws(()=>api.requireImagesReady());
api.retryImages();
const secondImages=node('views').children.map(figure=>figure.children[0]);
for(const img of secondImages){Object.assign(img,good);img.onload()}
assert.equal(api.imageState.ready,true);
firstImages[0].onerror();firstImages[1].onload();
assert.equal(api.imageState.ready,true);
assert.equal(api.imageState.failed.size,0);
api.payload.image_mode='source';
secondImages[0].onerror();
assert.equal(api.imageState.ready,false);
assert.equal(node('normal').disabled,true);
assert.equal(node('clear').disabled,false);
assert.ok(node('image-status').textContent.includes('作者原图'));
assert.ok(!node('image-status').textContent.includes('ZIP'));
assert.ok(!node('image-status').textContent.includes('images 文件夹'));
''')
    result = subprocess.run([node, str(script), str(page)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
