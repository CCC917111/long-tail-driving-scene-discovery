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
 if(imageState.ready){$('image-status').textContent='六视角图片已全部加载，可以判断。';return}
 if(imageState.failed.size){
  const recovery=p.image_mode==='local'?'请确认 images 文件夹与核验页放在同一目录，然后重试。':'作者原图未能加载，请检查网络后重试。';
  $('image-status').textContent='图片加载失败（'+[...imageState.failed].join('、')+'）；'+recovery;
 }else $('image-status').textContent='正在加载六视角图片（'+imageState.loaded.size+'/'+p.camera_layout.length+'）；全部加载后才能判断。';
}
function beginImages(token){imageState.generation++;imageState.token=token;imageState.loaded.clear();imageState.failed.clear();updateImageStatus();return imageState.generation}
function failImage(token,generation,camera){if(!isCurrentImage(token,generation))return;imageState.loaded.delete(camera);imageState.failed.add(camera);updateImageStatus()}
function completeImage(token,generation,camera,image){
 if(!isCurrentImage(token,generation)||!p.camera_layout.includes(camera))return;
 if(!image.complete||!(image.naturalWidth>0)||!(image.naturalHeight>0)){failImage(token,generation,camera);return}
 imageState.loaded.add(camera);imageState.failed.delete(camera);updateImageStatus();
}
function requireImagesReady(){check(imageState.ready&&imageState.token===p.frames[state.index].sample_token,'请等待当前组的六视角图片全部加载后再判断。')}
function validateRow(row,token){check(row.sample_token===token,'Unknown frame');check(['',...labels].includes(row.label),'Invalid label');check(typeof row.reviewer==='string'&&typeof row.review_notes==='string','Invalid reviewer/notes');check(row.review_status===(row.label===''?'pending':row.label==='uncertain'?'uncertain':'reviewed'),'Invalid review status');check(row.label===''||row.reviewer.trim(),'Reviewer is required');check(typeof row.model_signal_exposed==='boolean','Invalid exposure flag');check(row.review_context===(row.model_signal_exposed?'post_model_signal_exposure_single_reviewer':'model_hidden_single_reviewer'),'Invalid review context');return row}
function context(){return state.exposed?'post_model_signal_exposure_single_reviewer':'model_hidden_single_reviewer'}
function decide(label,reviewer,notes){check(labels.includes(label),'Unknown label');const row={sample_token:p.frames[state.index].sample_token,label,review_status:label==='uncertain'?'uncertain':'reviewed',reviewer:reviewer.trim(),review_notes:notes,review_context:context(),model_signal_exposed:state.exposed};validateRow(row,row.sample_token);state.rows[state.index]=row;return row}
function reveal(){state.exposed=true;save();renderSignals()}
function csv(){const quote=v=>'"'+String(v).replaceAll('"','""')+'"';return '\ufeff'+[FIELDS,...state.rows.map(r=>FIELDS.map(k=>r[k]))].map(r=>r.map(quote).join(',')).join('\r\n')+'\r\n'}
function save(){try{localStorage.setItem(p.storage_key,JSON.stringify(state));$('status').textContent='已保存到当前浏览器；请导出 CSV 备份。'}catch(_){$('status').textContent='浏览器无法自动保存，请导出 CSV 备份。'}}
function renderSignals(){const f=p.frames[state.index];$('signals').hidden=!state.exposed;$('reveal').hidden=state.exposed;if(!state.exposed)return;$('signal-context').textContent='已显示模型信号；此后保存的判断会记录为模型信息暴露后复核。';$('score').textContent='候选筛选得分 ||z_t||₂ = '+f.tail_score.toFixed(6)+'；固定规则 '+p.decision.rule+'，eta='+p.decision.eta+'；模型输出 '+f.prediction+'。这些输出不是人工标签。';$('units').replaceChildren();for(const unit of f.units){const li=document.createElement('li');li.textContent='单元 #'+unit.unit+'：激活 '+unit.activation.toFixed(6);$('units').append(li)}$('training-scope').textContent=JSON.stringify(p.model_training_label_scope,null,2)}
function render(){const f=p.frames[state.index],r=state.rows[state.index];$('frame').textContent='第 '+(state.index+1)+' / '+p.frames.length+' 组 · '+f.sample_token;$('scene').textContent='场景 '+f.scene;$('progress').textContent='已判断 '+state.rows.filter(r=>r.label).length+' / '+p.frames.length+'；当前 '+(r.label||'待判断');$('notes').value=r.review_notes;if(r.reviewer)$('reviewer').value=r.reviewer;$('choose').value=String(state.index);$('views').replaceChildren();const generation=beginImages(f.sample_token);for(const cam of p.camera_layout){const figure=document.createElement('figure'),img=document.createElement('img'),caption=document.createElement('figcaption');img.alt=cam;img.onload=()=>completeImage(f.sample_token,generation,cam,img);img.onerror=()=>failImage(f.sample_token,generation,cam);img.src=f.images[cam];caption.textContent=cam;figure.append(img,caption);$('views').append(figure)}$('previous').disabled=state.index===0;$('next').disabled=state.index===p.frames.length-1;renderSignals()}
function retryImages(){const notes=$('notes').value,reviewer=$('reviewer').value;render();$('notes').value=notes;$('reviewer').value=reviewer}
function move(index){state.index=index;save();render()}
function exportCSV(){const url=URL.createObjectURL(new Blob([csv()],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download='reviewed-new-scenes.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),10000)}
function restore(){try{const old=JSON.parse(localStorage.getItem(p.storage_key)||'null');if(!old)return;check(Array.isArray(old.rows)&&old.rows.length===state.rows.length&&typeof old.exposed==='boolean','Invalid saved state');old.rows.forEach((r,i)=>validateRow(r,p.frames[i].sample_token));check(!old.rows.some(r=>r.model_signal_exposed)||old.exposed,'Saved exposure cannot be reset');state.rows=old.rows;state.exposed=old.exposed;state.index=Number.isInteger(old.index)&&old.index>=0&&old.index<p.frames.length?old.index:0}catch(error){$('status').textContent='未加载本地记录：'+error.message}}
globalThis.DrivingUnlabelledReview={payload:p,state,FIELDS,validateRow,decide,reveal,csv,restore,imageState,beginImages,completeImage,failImage,requireImagesReady,retryImages};
function boot(){for(let i=0;i<p.frames.length;i++){const option=document.createElement('option');option.value=String(i);option.textContent='第 '+(i+1)+' 组';$('choose').append(option)}for(const [id,label] of [['normal','normal_core'],['tail','not_normal_core'],['uncertain','uncertain']])$(id).onclick=()=>{try{requireImagesReady();decide(label,$('reviewer').value,$('notes').value);save();if(state.index<p.frames.length-1)state.index++;render()}catch(e){$('status').textContent='未保存：'+e.message}};$('clear').onclick=()=>{state.rows[state.index]={sample_token:p.frames[state.index].sample_token,label:'',review_status:'pending',reviewer:'',review_notes:'',review_context:context(),model_signal_exposed:state.exposed};save();render()};$('previous').onclick=()=>move(state.index-1);$('next').onclick=()=>move(state.index+1);$('choose').onchange=e=>move(Number(e.target.value));$('retry-images').onclick=retryImages;$('reveal').onclick=reveal;$('export').onclick=exportCSV;$('notes').onchange=()=>{const r=state.rows[state.index];r.review_notes=$('notes').value;r.review_context=context();r.model_signal_exposed=state.exposed;save()};restore();render()}
if(!globalThis.DrivingReviewHeadless)boot();
"""


def render_html(payload):
    require(payload.get("scope", {}).get("has_reference_labels") is False,
            "Unlabelled scope is required")
    style = PAGE.split("<style>", 1)[1].split("</style>", 1)[0]
    return """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self' https://raw.githubusercontent.com; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'; base-uri 'none'; form-action 'none'">
<title>Driving 新场景人工核验</title><style>""" + style + """
body{max-width:1250px;margin:auto;padding:20px}button,input,select,textarea{font:inherit;padding:8px;margin:4px;max-width:100%}button{cursor:pointer}textarea{display:block;width:min(95%,700px);min-height:70px}p,pre,h1,h2,figure,section{min-width:0;overflow-wrap:anywhere}pre{white-space:pre-wrap}.views img{object-fit:contain;background:#e9edf3}.bar{display:flex;gap:8px;flex-wrap:wrap}.notice{padding:12px;background:#fff1d6;border-radius:6px}[hidden]{display:none!important}
</style></head><body><h1>Driving 新场景人工核验</h1>
<p class="notice">请根据当前六视角填写判断。本页不预填 AI 或人工参考标签。模型信号默认隐藏；点击显示后，后续判断会记录暴露情况。训练模型的旧标签来源见显示后的说明。本页没有评估分数。</p>
<details><summary>判断标准</summary><p>正常 normal_core：仅看当前六视角，路况清楚、通行常规，未见需要额外防御驾驶的因素。非正常核心 not_normal_core：有清楚可见的低能见度、夜间或降雨导致的感知困难、关键遮挡、障碍、施工、复杂通行，或明显过近或邻近自车路径且需要额外防御动作的参与者等因素，需额外减速、让行或扩大安全距离。不确定 uncertain：图片不足以稳定判断。不要推断单帧看不见的动态事件。</p></details>
<div class="bar"><button id="previous">上一组</button><select id="choose" aria-label="选择图片组"></select><button id="next">下一组</button><button id="export">下载 reviewed-new-scenes.csv</button></div>
<h2 id="frame"></h2><p id="scene"></p><p id="progress"></p><div id="views" class="views"></div>
<p id="image-status" role="status" aria-live="polite"></p><button id="retry-images" type="button">重试图片</button>
<label>真实标注人 <input id="reviewer" maxlength="200" placeholder="例如 Yike"></label>
<label for="notes">判断依据或不确定原因</label><textarea id="notes" maxlength="10000"></textarea>
<div class="bar"><button id="normal" disabled>正常 · normal_core</button><button id="tail" disabled>非正常核心 · not_normal_core</button><button id="uncertain" disabled>不确定 · uncertain</button><button id="clear">清除此组判断</button></div>
<p id="status" role="status"></p><button id="reveal">显示模型信号（会记录信息暴露）</button>
<section id="signals" hidden><p id="signal-context" class="notice"></p><p id="score"></p><ul id="units"></ul><details><summary>模型训练标签来源（不代表本批图片已标注）</summary><pre id="training-scope"></pre></details></section>
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
