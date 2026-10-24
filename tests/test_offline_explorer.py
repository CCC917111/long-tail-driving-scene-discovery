"""Portable cached explorer integrity and browser-script contracts."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from export_offline_explorer import ROOT, render_html, script_json  # noqa: E402
from tail_activations import CAMERAS  # noqa: E402


def payload():
    images = {camera: "samples/" + camera + "/fixture.jpg" for camera in CAMERAS}
    frame = {"sample_token": "fixture-token", "scene": "fixture-scene",
             "active_tail_units": 1, "prediction": "long_tail", "tail_score": 2.0,
             "cameras": list(CAMERAS), "label": "normal", "split": "test",
             "images": images, "comparisons": [], "units": [{"unit": 0, "activation": 2.0}]}
    scope = {"label_protocol": "ai-visual-pilot", "human_reviewed": False}
    unit_frame = {"sample_token": "fixture-token", "activation": 2.0}
    return {"summary": {"frames": 1, "label_scope": scope}, "label_scope": scope,
            "comparison": [{"method": "fixture <script>", "ap": 0.5,
                            "precision_at_k": 0.5, "target_status": "target_unreachable",
                            "n": 1, "test_scenes": 1, "requested_k": 1, "seed": 42}],
            "bookmarks": [{"sample_token": "fixture-token", "title": "fixture"}],
            "transfer_id": "fixture-transfer", "frames_per_scene": 2,
            "rubric": "fixture rubric", "rubric_sha256": "b" * 64,
            "expected_frames": [{"sample_token": "fixture-token", "scene_token": "scene",
                                 "timestamp": 1, "images": images}],
            "expected_images": {path: "a" * 64 for path in images.values()},
            "expected_metadata": {"scene": "c" * 64}, "frames": {"fixture-token": frame},
            "unit_list": [{"unit": 0, "frames": 1}],
            "units": {"0": {"unit": 0, "frames": [unit_frame]}},
            "review_csv": "sample_token,reviewer_label\nfixture-token,\n"}


def test_script_json_cannot_terminate_a_script_element():
    value = {"untrusted": "</script><img src=x>&\u2028\u2029"}
    encoded = script_json(value)
    assert "</script>" not in encoded and "<img" not in encoded
    assert json.loads(encoded) == value


def test_render_keeps_images_and_network_outside_the_export():
    source = render_html(payload(), (ROOT / "tools/review_upload.html").read_text())
    assert "connect-src 'none'" in source
    assert "fetch(" not in source and "data:image/" not in source
    assert "fixture &lt;script&gt;" in source
    assert source.count('<script id="') == 4


def test_local_api_and_expected_image_binding(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for offline browser-script contract verification")
    source = render_html(payload(), (ROOT / "tools/review_upload.html").read_text())
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
const model={manifest:{transfer_id:p.transfer_id,frames_per_scene:p.frames_per_scene,
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
""")
    result = subprocess.run([node, str(tmp_path / "verify.js"), str(tmp_path / "demo.html")],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
