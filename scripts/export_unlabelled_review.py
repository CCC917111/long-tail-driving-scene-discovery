#!/usr/bin/env python3
"""Export a small, private, offline review page for unlabelled SAE screening.

Scores stay hidden until explicitly revealed. This exports observations and a
blank human-review workflow, never reference labels or accuracy metrics. A local
bundle places original JPEGs under images/<CAMERA>/<filename> beside this page.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote, urlparse

from neuron_explorer import Explorer, PAGE, SIX_VIEW_LAYOUT
from tail_activations import ActivationStore, CAMERAS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def script_json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False).replace(
        "<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026").replace(
        "\u2028", "\\u2028").replace("\u2029", "\\u2029")


def build_payload(run, samples_root, *, source_manifest=None,
                  image_mode="local", image_prefix="images"):
    run, samples_root = Path(run).resolve(), Path(samples_root).resolve()
    metadata_path = run / "screening_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    require(metadata.get("kind") == "unlabelled-screening",
            "An unlabelled screening run is required")
    store = ActivationStore(run)
    require(store.labels is None and store.splits is None,
            "Review inputs must have no reference labels or train/test assignments")
    require(len(store) and len(store.row_of) == len(store),
            "Unique nonempty sample tokens are required")
    require(image_mode in {"local", "source"}, "Unknown image mode")
    prefix = Path(image_prefix)
    require(not prefix.is_absolute() and ".." not in prefix.parts,
            "Image prefix must stay inside the review bundle")
    source = json.loads(Path(source_manifest).read_text()) if source_manifest else {}
    source_images = {record["path"].removeprefix("samples/"): record
                     for record in source.get("images", [])}
    if image_mode == "source":
        require(source_images, "Source image mode requires an acquisition manifest")
    explorer = Explorer(store, samples_root, eta=metadata["decision"]["eta"],
                        neuron_threshold=metadata.get("neuron_threshold", 1.0))
    frames = []
    for index, sample in enumerate(store.samples):
        token = sample["sample_token"]
        require(token and sample.get("scene"), "Every frame needs sample and scene identifiers")
        require(set(sample.get("images", {})) == set(CAMERAS),
                "Exactly six camera paths are required")
        frame = explorer._frame(index)
        require(frame["cameras"] == list(CAMERAS), "All six original camera images must exist")
        frame["images"] = {}
        for camera in CAMERAS:
            relative = Path(sample["images"][camera])
            require(not relative.is_absolute() and ".." not in relative.parts
                    and len(relative.parts) == 2 and relative.parts[0] == camera,
                    "Camera images must use CAMERA/filename paths")
            path = explorer.image_path(token, camera)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            record = source_images.get(relative.as_posix())
            if source_manifest:
                require(record and record.get("sha256") == digest,
                        "Image hash differs from acquisition manifest")
            if image_mode == "source":
                url = record["source_url"]
                parsed = urlparse(url)
                expected = "/OpenDriveLab/DriveLM/" + source.get("source_commit", "") + "/"
                require(parsed.scheme == "https" and parsed.netloc == "raw.githubusercontent.com"
                        and len(source.get("source_commit", "")) == 40
                        and parsed.path.startswith(expected),
                        "Image URL must use the pinned author source")
            else:
                url = quote((prefix / relative).as_posix(), safe="/")
            frame["images"][camera] = url
        units, values = store.row(index)
        frame["units"] = sorted([{"unit": int(unit), "activation": float(value)}
                                 for unit, value in zip(units, values)],
                                key=lambda row: -abs(row["activation"]))
        frames.append(frame)
    fingerprint = hashlib.sha256((run / "activations.npz").read_bytes()).hexdigest()
    return {"schema": "unlabelled-review-v1", "frames": frames,
            "camera_layout": [camera for row in SIX_VIEW_LAYOUT for camera in row],
            "scope": {"has_reference_labels": False, "human_reviewed": False,
                      "is_benchmark_evaluation": False, "model_signals_hidden_initially": True},
            "decision": metadata["decision"], "neuron_threshold": explorer.neuron_threshold,
            "model_training_label_scope": metadata.get(
                "model_training_label_scope", metadata.get("label_scope")),
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "standardizer_sha256": metadata["standardizer_sha256"],
            "activation_sha256": fingerprint, "storage_key": "driving-review-" + fingerprint,
            "source_commit": source.get("source_commit"), "image_mode": image_mode}


RUNTIME = r"""
const p=JSON.parse(document.getElementById('payload').textContent),$=id=>document.getElementById(id);
const FIELDS=['sample_token','label','review_status','reviewer','review_notes','review_context','model_signal_exposed'];
const labels=['normal_core','not_normal_core','uncertain'];
const state={rows:p.frames.map(f=>({sample_token:f.sample_token,label:'',review_status:'pending',reviewer:'',review_notes:'',review_context:'model_hidden_single_reviewer',model_signal_exposed:false})),exposed:false,index:0};
const imageState={generation:0,token:null,loaded:new Set(),failed:new Set(),ready:false};
function check(ok,message){if(!ok)throw new Error(message)}
function isCurrentImage(token,generation){return imageState.token===token&&imageState.generation===generation&&p.frames[state.index].sample_token===token}
function updateImageStatus(){
 imageState.ready=imageState.loaded.size===p.camera_layout.length&&imageState.failed.size===0;
 for(const id of ['normal','tail','uncertain'])$(id).disabled=!imageState.ready;
 $('views').setAttribute('aria-busy',String(!imageState.ready));
 if(imageState.ready){$('image-status').textContent='All six camera images are loaded. You can review this frame.';return}
 if(imageState.failed.size){
  const recovery=p.image_mode==='local'?'Place the images folder beside this review page, then retry.':'The original author images could not load. Check your network connection, then retry.';
  $('image-status').textContent='Images failed to load ('+[...imageState.failed].join(', ')+'); '+recovery;
 }else $('image-status').textContent='Loading six-camera images ('+imageState.loaded.size+'/'+p.camera_layout.length+'); review is available once all images have loaded.';
}
function beginImages(token){imageState.generation++;imageState.token=token;imageState.loaded.clear();imageState.failed.clear();updateImageStatus();return imageState.generation}
function failImage(token,generation,camera){if(!isCurrentImage(token,generation))return;imageState.loaded.delete(camera);imageState.failed.add(camera);updateImageStatus()}
function completeImage(token,generation,camera,image){
 if(!isCurrentImage(token,generation)||!p.camera_layout.includes(camera))return;
 if(!image.complete||!(image.naturalWidth>0)||!(image.naturalHeight>0)){failImage(token,generation,camera);return}
 imageState.loaded.add(camera);imageState.failed.delete(camera);updateImageStatus();
}
function requireImagesReady(){check(imageState.ready&&imageState.token===p.frames[state.index].sample_token,'Wait until all six camera images for this frame have loaded before reviewing.')}
function validateRow(row,token){check(row.sample_token===token,'Unknown frame');check(['',...labels].includes(row.label),'Invalid label');check(typeof row.reviewer==='string'&&typeof row.review_notes==='string','Invalid reviewer/notes');check(row.review_status===(row.label===''?'pending':row.label==='uncertain'?'uncertain':'reviewed'),'Invalid review status');check(row.label===''||row.reviewer.trim(),'Reviewer is required');check(typeof row.model_signal_exposed==='boolean','Invalid exposure flag');check(row.review_context===(row.model_signal_exposed?'post_model_signal_exposure_single_reviewer':'model_hidden_single_reviewer'),'Invalid review context');return row}
function context(){return state.exposed?'post_model_signal_exposure_single_reviewer':'model_hidden_single_reviewer'}
function decide(label,reviewer,notes){check(labels.includes(label),'Unknown label');const row={sample_token:p.frames[state.index].sample_token,label,review_status:label==='uncertain'?'uncertain':'reviewed',reviewer:reviewer.trim(),review_notes:notes,review_context:context(),model_signal_exposed:state.exposed};validateRow(row,row.sample_token);state.rows[state.index]=row;return row}
function reveal(){state.exposed=true;save();renderSignals()}
function csv(){const quote=v=>'"'+String(v).replaceAll('"','""')+'"';return '\ufeff'+[FIELDS,...state.rows.map(r=>FIELDS.map(k=>r[k]))].map(r=>r.map(quote).join(',')).join('\r\n')+'\r\n'}
function save(){try{localStorage.setItem(p.storage_key,JSON.stringify(state));$('status').textContent='Saved in this browser. Export a CSV backup.'}catch(_){$('status').textContent='Browser autosave is unavailable. Export a CSV backup.'}}
function renderSignals(){const f=p.frames[state.index];$('signals').hidden=!state.exposed;$('reveal').hidden=state.exposed;if(!state.exposed)return;$('signal-context').textContent='Model signals revealed. Decisions saved from now on are recorded as reviews after model-signal exposure.';$('score').textContent='Candidate screening score ||z_t||₂ = '+f.tail_score.toFixed(6)+'; fixed rule '+p.decision.rule+', eta='+p.decision.eta+'; model output '+f.prediction+'. These outputs are not human labels.';$('units').replaceChildren();for(const unit of f.units){const li=document.createElement('li');li.textContent='Unit #'+unit.unit+': activation '+unit.activation.toFixed(6);$('units').append(li)}$('training-scope').textContent=JSON.stringify(p.model_training_label_scope,null,2)}
function render(){const f=p.frames[state.index],r=state.rows[state.index];$('frame').textContent='Frame '+(state.index+1)+' / '+p.frames.length+' · '+f.sample_token;$('scene').textContent='Scene '+f.scene;$('progress').textContent='Reviewed '+state.rows.filter(r=>r.label).length+' / '+p.frames.length+'; current: '+(r.label||'pending');$('notes').value=r.review_notes;if(r.reviewer)$('reviewer').value=r.reviewer;$('choose').value=String(state.index);$('views').replaceChildren();const generation=beginImages(f.sample_token);for(const cam of p.camera_layout){const figure=document.createElement('figure'),img=document.createElement('img'),caption=document.createElement('figcaption');img.alt=cam;img.onload=()=>completeImage(f.sample_token,generation,cam,img);img.onerror=()=>failImage(f.sample_token,generation,cam);img.src=f.images[cam];caption.textContent=cam;figure.append(img,caption);$('views').append(figure)}$('previous').disabled=state.index===0;$('next').disabled=state.index===p.frames.length-1;renderSignals()}
function retryImages(){const notes=$('notes').value,reviewer=$('reviewer').value;render();$('notes').value=notes;$('reviewer').value=reviewer}
function move(index){state.index=index;save();render()}
function exportCSV(){const url=URL.createObjectURL(new Blob([csv()],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download='reviewed-new-scenes.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),10000)}
function restore(){try{const old=JSON.parse(localStorage.getItem(p.storage_key)||'null');if(!old)return;check(Array.isArray(old.rows)&&old.rows.length===state.rows.length&&typeof old.exposed==='boolean','Invalid saved state');old.rows.forEach((r,i)=>validateRow(r,p.frames[i].sample_token));check(!old.rows.some(r=>r.model_signal_exposed)||old.exposed,'Saved exposure cannot be reset');state.rows=old.rows;state.exposed=old.exposed;state.index=Number.isInteger(old.index)&&old.index>=0&&old.index<p.frames.length?old.index:0}catch(error){$('status').textContent='Local records were not loaded: '+error.message}}
globalThis.DrivingUnlabelledReview={payload:p,state,FIELDS,validateRow,decide,reveal,csv,restore,imageState,beginImages,completeImage,failImage,requireImagesReady,retryImages};
function boot(){for(let i=0;i<p.frames.length;i++){const option=document.createElement('option');option.value=String(i);option.textContent='Frame '+(i+1);$('choose').append(option)}for(const [id,label] of [['normal','normal_core'],['tail','not_normal_core'],['uncertain','uncertain']])$(id).onclick=()=>{try{requireImagesReady();decide(label,$('reviewer').value,$('notes').value);save();if(state.index<p.frames.length-1)state.index++;render()}catch(e){$('status').textContent='Not saved: '+e.message}};$('clear').onclick=()=>{state.rows[state.index]={sample_token:p.frames[state.index].sample_token,label:'',review_status:'pending',reviewer:'',review_notes:'',review_context:context(),model_signal_exposed:state.exposed};save();render()};$('previous').onclick=()=>move(state.index-1);$('next').onclick=()=>move(state.index+1);$('choose').onchange=e=>move(Number(e.target.value));$('retry-images').onclick=retryImages;$('reveal').onclick=reveal;$('export').onclick=exportCSV;$('notes').onchange=()=>{const r=state.rows[state.index];r.review_notes=$('notes').value;r.review_context=context();r.model_signal_exposed=state.exposed;save()};restore();render()}
if(!globalThis.DrivingReviewHeadless)boot();
"""


def render_html(payload):
    require(payload.get("scope", {}).get("has_reference_labels") is False,
            "Unlabelled scope is required")
    style = PAGE.split("<style>", 1)[1].split("</style>", 1)[0]
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self' https://raw.githubusercontent.com; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'; base-uri 'none'; form-action 'none'">
<title>Driving new-scene human review</title><style>""" + style + """
body{max-width:1250px;margin:auto;padding:20px}button,input,select,textarea{font:inherit;padding:8px;margin:4px;max-width:100%}button{cursor:pointer}textarea{display:block;width:min(95%,700px);min-height:70px}p,pre,h1,h2,figure,section{min-width:0;overflow-wrap:anywhere}pre{white-space:pre-wrap}.views img{object-fit:contain;background:#e9edf3}.bar{display:flex;gap:8px;flex-wrap:wrap}.notice{padding:12px;background:#fff1d6;border-radius:6px}[hidden]{display:none!important}
</style></head><body><h1>Driving new-scene human review</h1>
<p class="notice">Judge the current six-camera frame. No reference labels are prefilled. Model signals are hidden initially; revealing them records exposure for subsequent decisions. After reveal, you can inspect the label provenance used to train the model. This page has no evaluation metrics.</p>
<details><summary>Review rubric</summary><p>Normal (normal_core): based only on the current six-camera frame, conditions are clear and routine, with no visible factor requiring extra defensive driving. Outside the normal core (not_normal_core): visible low visibility, perception difficulties from night or rain, key occlusions, obstacles, construction, complex traffic, or road users clearly too close to or near the ego vehicle's path require extra defensive action such as slowing down, yielding or increasing the safety margin. Uncertain (uncertain): the images do not support a stable judgment. Do not infer dynamic events that a single frame cannot show.</p></details>
<div class="bar"><button id="previous">Previous frame</button><select id="choose" aria-label="Choose frame"></select><button id="next">Next frame</button><button id="export">Download reviewed-new-scenes.csv</button></div>
<h2 id="frame"></h2><p id="scene"></p><p id="progress"></p><div id="views" class="views"></div>
<p id="image-status" role="status" aria-live="polite"></p><button id="retry-images" type="button">Retry images</button>
<label>Reviewer name <input id="reviewer" maxlength="200" placeholder="For example, Yike"></label>
<label for="notes">Reason for this decision or uncertainty</label><textarea id="notes" maxlength="10000"></textarea>
<div class="bar"><button id="normal" disabled>Normal · normal_core</button><button id="tail" disabled>Outside normal core · not_normal_core</button><button id="uncertain" disabled>Uncertain · uncertain</button><button id="clear">Clear this decision</button></div>
<p id="status" role="status"></p><button id="reveal">Reveal model signals (records exposure)</button>
<section id="signals" hidden><p id="signal-context" class="notice"></p><p id="score"></p><ul id="units"></ul><details><summary>Model training label provenance (this batch is still unlabelled)</summary><pre id="training-scope"></pre></details></section>
<script id="payload" type="application/json">""" + script_json(payload) + """</script>
<script id="review-runtime">""" + RUNTIME + "</script></body></html>\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--image-mode", choices=("local", "source"), default="local")
    parser.add_argument("--image-prefix", default="images")
    args = parser.parse_args(argv)
    require(not args.output.exists(), "Output exists; preserve it and use a new filename")
    payload = build_payload(args.run, args.samples_root, source_manifest=args.source_manifest,
                            image_mode=args.image_mode, image_prefix=args.image_prefix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(payload), encoding="utf-8")
    report = {"schema": payload["schema"], "frames": len(payload["frames"]),
              "images": len(payload["frames"]) * 6, "scope": payload["scope"],
              "html_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
              "image_mode": args.image_mode, "images_embedded": 0,
              "model_training_label_scope": payload["model_training_label_scope"],
              "browser_executed": False}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
