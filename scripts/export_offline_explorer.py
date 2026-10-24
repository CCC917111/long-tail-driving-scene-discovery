#!/usr/bin/env python3
"""Export a cached AI-pilot explorer whose original images stay on the user's PC.

Reuses neuron_explorer.PAGE and Explorer, plus the already checked local ZIP
reader. No pixels, model inference or network calls are embedded in the HTML.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import re

from neuron_explorer import PAGE, Explorer
from summarize_runs import collect_runs
from tail_activations import ActivationStore, CAMERAS, DEFAULT_NEURON_THRESHOLD

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def script_json(value):
    """A literal in a script element must not terminate that element."""
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            .replace("&", "\\u0026").replace("<", "\\u003c")
            .replace(">", "\\u003e").replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029"))


def replace_once(source, original, replacement):
    require(source.count(original) == 1, "Explorer template changed: " + original[:60])
    return source.replace(original, replacement)


def build_payload(run, comparison_dir, prepared_dir, samples_root, import_report):
    run, comparison_dir, prepared_dir = (Path(path).resolve()
                                         for path in (run, comparison_dir, prepared_dir))
    imported = read_json(import_report)
    prepared = read_json(prepared_dir / "prepare_metadata.json")
    selected = read_json(prepared_dir / "selected_samples.json")
    raw = read_json(prepared_dir / "raw_manifest.json")
    require(imported.get("status") == "completed" and imported.get("labels_created") is False,
            "A completed original small-part import report is required")
    for filename in ("selected_samples.json", "raw_manifest.json", "rubric.md"):
        require(digest(prepared_dir / filename) == prepared["prepared_files"][filename],
                "Prepared source changed: " + filename)
    require(prepared["included_frames"] == imported["selected_samples"] == len(selected),
            "Imported selection count differs from preparation")
    raw_by_token = {row["sample_token"]: row for row in raw if not row.get("exclusion_reason")}
    require(len(raw_by_token) == len(selected), "Prepared selection has duplicate/missing rows")
    expected_images = {}
    expected_frames = []
    for row in selected:
        images = {}
        for camera in CAMERAS:
            filename = "samples/" + row["images"][camera]
            record = raw_by_token[row["sample_token"]]["image_files"][camera]
            require(record["filename"] == filename, "Prepared image filename differs")
            expected_images[filename] = record["sha256"]
            images[camera] = filename
        expected_frames.append({key: row[key] for key in
                                ("sample_token", "scene_token", "timestamp")})
        expected_frames[-1]["images"] = images

    recorded_table = read_json(comparison_dir / "comparison/main_table.json")
    require(len(recorded_table) == 4, "Expected the four controlled pilot methods")
    comparison, _ = collect_runs([comparison_dir / row["method"] for row in recorded_table])
    for recorded, rebuilt in zip(recorded_table, comparison):
        require({key: value for key, value in recorded.items() if key != "run_dir"}
                == {key: value for key, value in rebuilt.items() if key != "run_dir"},
                "Comparison table differs from saved per-sample predictions")
    methods, model_predictions, checkpoints = [], {}, {}
    manifest_hash = None
    for summary in comparison:
        name = summary["method"]
        require(name not in model_predictions, "Duplicate method in table")
        directory = comparison_dir / name
        metadata = read_json(directory / "run_metadata.json")
        evaluation = read_json(directory / "evaluation.json")
        scope = metadata.get("label_scope", {})
        require(scope.get("label_protocol") == "ai-visual-pilot"
                and scope.get("human_reviewed") is False
                and evaluation.get("label_scope") == scope,
                "Offline pilot export requires explicit pending-human AI label scope")
        for key, filename in (("predictions_sha256", "predictions.json"),
                              ("manifest_sha256", "evaluation_manifest.json")):
            require(digest(directory / filename) == metadata[key],
                    "Run source hash differs: " + name + "/" + filename)
        if manifest_hash is None:
            manifest_hash = metadata["manifest_sha256"]
        require(manifest_hash == metadata["manifest_sha256"], "Model manifests differ")
        checkpoint = Path(metadata["checkpoint"]["path"])
        if not checkpoint.is_file():
            checkpoint = directory / checkpoint.name
        require(digest(checkpoint) == metadata["checkpoint"]["sha256"],
                "Saved checkpoint differs from run metadata")
        checkpoints[name] = metadata["checkpoint"]["sha256"]
        rows = read_json(directory / "predictions.json")
        require(len({row["sample_token"] for row in rows}) == len(rows),
                "Prediction tokens are not unique")
        model_predictions[name] = {row["sample_token"]: row for row in rows}
        methods.append(name)

    metadata = read_json(run / "run_metadata.json")
    scope = metadata["label_scope"]
    decision = read_json(run / "metrics.json")["decision"]
    store = ActivationStore(run)
    explorer = Explorer(store, samples_root=samples_root, eta=decision["eta"],
                        neuron_threshold=DEFAULT_NEURON_THRESHOLD, label_scope=scope)
    frames = {row["sample_token"]: explorer.units_of_frame(row["sample_token"], top=store.tail_dim)
              for row in store.samples}
    require(all(set(predictions) == set(frames) for predictions in model_predictions.values()),
            "Model token sets differ from trained activation cache")
    image_paths = {row["sample_token"]: row["images"] for row in expected_frames}
    for token, frame in frames.items():
        require(frame["cameras"] and set(frame["cameras"]) == set(CAMERAS),
                "The cached frame has missing original camera images")
        frame["images"] = image_paths[token]
        frame["comparisons"] = [{"method": method, **{
            key: model_predictions[method][token].get(key)
            for key in ("score", "prediction", "operating_prediction", "operating_error")}}
            for method in methods]

    units = {}
    for unit in range(store.tail_dim):
        stats = explorer.unit_stats(unit)
        units[str(unit)] = {**stats, "n_frames": stats["frames"],
                            "frames": [{"sample_token": row["sample_token"],
                                        "activation": row["activation"]}
                                       for row in explorer.frames_of_unit(unit, top=len(store))]}

    def pick(method, condition):
        choices = [row for row in model_predictions[method].values()
                   if row["split"] == "test" and condition(row)]
        return sorted(choices, key=lambda row: (-row["score"], row["sample_token"]))

    bookmarks = []
    for title, method, condition in (
        ("SAE 固定规则误报（相对 AI 初标）", run.name,
         lambda row: row["label"] == 0 and row["prediction"] == 1),
        ("MLP 固定规则漏报（相对 AI 初标）", "mlp",
         lambda row: row["label"] == 1 and row["prediction"] == 0),
        ("SAE 高分候选与激活单元", run.name,
         lambda row: row["label"] == 1),
    ):
        candidates = pick(method, condition)
        if candidates:
            bookmarks.append({"title": title, "method": method,
                              "sample_token": candidates[0]["sample_token"]})
    require(len(bookmarks) == 3, "Pilot does not contain all requested real disagreement examples")
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in (
        ROOT / "scripts/neuron_explorer.py", ROOT / "scripts/tail_activations.py",
        ROOT / "tools/review_upload.html", Path(__file__))}
    return {
        "schema": "offline-trained-explorer-ai-pilot-v1",
        "label_scope": scope, "transfer_id": imported["transfer_id"],
        "frames_per_scene": imported["frames_per_scene"],
        "expected_frames": expected_frames, "expected_images": expected_images,
        "expected_metadata": {name: record["sha256"]
                              for name, record in prepared["source_metadata"].items()},
        "rubric_sha256": prepared["rubric_sha256"],
        "rubric": (prepared_dir / "rubric.md").read_text(encoding="utf-8"),
        "summary": {"frames": len(store), "tail_dim": store.tail_dim,
                    "eta": explorer.eta, "neuron_threshold": explorer.neuron_threshold,
                    "has_labels": True, "cameras": list(CAMERAS), "label_scope": scope},
        "frames": frames, "unit_list": explorer.units(top=store.tail_dim), "units": units,
        "review_csv": explorer.review_csv().decode("utf-8"),
        "comparison": comparison, "methods": methods, "bookmarks": bookmarks,
        "checkpoints": checkpoints, "source_sha256": source_hashes,
        "manifest_sha256": manifest_hash,
        "scope_notes": [
            "Cached outputs from trained CPU ResNet18 + SAE; no Cosmos reproduction.",
            "AI provisional labels; human review pending. Errors are AI-reference disagreements.",
            "All validation precision targets were unreachable: operating predictions are none.",
            "Reward/no-reward SAE checkpoints are identical; no reward improvement is established.",
            "Mini custom scene split with two test scenes; not an official benchmark.",
        ],
    }


RUNTIME = r"""
(function(){
'use strict';
const payload=globalThis.__DRIVING_OFFLINE_PAYLOAD__,reader=globalThis.DrivingReview;
const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
let loaded=null,urls=new Map(),resolveReady;
const ready=new Promise(resolve=>{resolveReady=resolve;});
function check(value,message){if(!value)throw new Error(message);}
function api(path){
 const request=new URL(path,'https://offline.invalid/');
 const parts=request.pathname.split('/').filter(Boolean);
 const top=Math.max(0,Math.min(200,Number(request.searchParams.get('top')||50)));
 if(parts.join('/')==='api/summary')return structuredClone(payload.summary);
 if(parts.join('/')==='api/units')return structuredClone(payload.unit_list.slice(0,top));
 if(parts.length===3&&parts[0]==='api'&&parts[1]==='unit'){
  const unit=payload.units[parts[2]];check(unit,'Unknown unit');
  const frames=unit.frames.slice(0,top).map(row=>{
   const {units,comparisons,images,...frame}=payload.frames[row.sample_token];
   return {...structuredClone(frame),activation:row.activation};
  });
  return {...structuredClone(unit),frames};
 }
 if(parts.length===3&&parts[0]==='api'&&parts[1]==='sample'){
  const frame=payload.frames[decodeURIComponent(parts[2])];
  check(frame,'Unknown frame');return structuredClone(frame);
 }
 if(parts.join('/')==='api/search'){
  const query=(request.searchParams.get('q')||'').trim().toLowerCase();if(!query)return [];
  return Object.values(payload.frames).filter(frame=>
   frame.sample_token.toLowerCase().startsWith(query)||frame.scene.toLowerCase().includes(query)
  ).slice(0,20).map(frame=>structuredClone(frame));
 }
 throw new Error('Unsupported offline API route');
}
function validateLoaded(model){
 check(model.manifest.transfer_id===payload.transfer_id,'分包不属于本次结果包，请选择原来的四个 ZIP。');
 check(model.manifest.frames_per_scene===payload.frames_per_scene,'分包帧数设置不一致');
 check(model.rubric_sha256===payload.rubric_sha256,'标注标准版本不一致');
 check(model.selected.length===payload.expected_frames.length,'所选时间点数量不一致');
 for(let i=0;i<model.selected.length;i++){
  const actual=model.selected[i],expected=payload.expected_frames[i];
  check(actual.sample_token===expected.sample_token&&actual.scene_token===expected.scene_token&&
        actual.timestamp===expected.timestamp,'所选时间点与结果不一致');
  for(const camera of reader.CAMERAS)
   check(actual.images[camera]===expected.images[camera],'六视角映射不一致');
 }
 for(const [name,digest]of Object.entries(payload.expected_metadata))
  check(model.manifest.metadata_sha256[name]===digest,'原始元数据与本轮结果不同');
 const inventory=new Map(model.manifest.files.map(row=>[row.path,row]));
 for(const [path,digest]of Object.entries(payload.expected_images))
  check(inventory.has(path)&&inventory.get(path).output_sha256===digest&&model.images.has(path),
        '图片内容与本轮结果不同');
 return true;
}
function attachLoaded(model){
 validateLoaded(model);for(const url of urls.values())URL.revokeObjectURL(url);
 urls=new Map();loaded=model;resolveReady();return model;
}
function imageURL(token,camera){
 check(loaded,'请先打开 ZIP 分包');const frame=payload.frames[token];
 check(frame&&frame.images[camera],'Unknown image');const path=frame.images[camera];
 if(!urls.has(path))urls.set(path,URL.createObjectURL(loaded.images.get(path)));
 return urls.get(path);
}
function downloadCSV(){
 const blob=new Blob([payload.review_csv],{type:'text/csv;charset=utf-8'});
 const url=URL.createObjectURL(blob),a=document.createElement('a');
 a.href=url;a.download='review.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),10000);
}
globalThis.DrivingOffline={payload,loadParts:reader.loadParts,api,validateLoaded,attachLoaded,imageURL,ready};
if(typeof document==='undefined')return;
const status=document.getElementById('offline-status'),load=document.getElementById('offline-load');
document.getElementById('offline-rubric').textContent=payload.rubric;
document.getElementById('review-export').addEventListener('click',event=>{
 event.preventDefault();downloadCSV();
});
load.onclick=async()=>{load.disabled=true;try{
 const files=Array.from(document.getElementById('offline-parts').files);
 const model=await reader.loadParts(files,text=>{status.textContent=text;});
 attachLoaded(model);document.body.classList.add('offline-loaded');
 status.textContent='校验完成：'+model.selected.length+' 组原图，本页展示 '+
  payload.summary.frames+' 组训练/评估缓存。图像留在你的电脑上。';
}catch(error){status.textContent='未打开：'+error.message;}finally{load.disabled=false;}};
document.querySelectorAll('[data-offline-token]').forEach(link=>{
 link.addEventListener('click',event=>{
  if(!loaded){event.preventDefault();status.textContent='请先选择四个 ZIP 并点击「打开原图」。';}
 });
});
window.addEventListener('unload',()=>{for(const url of urls.values())URL.revokeObjectURL(url);});
})();
"""


def render_html(payload, review_source):
    match = re.search(r"<script>([\s\S]*?)</script>", review_source)
    require(match is not None, "Local review tool has no script")
    marker = "if(typeof document==='undefined')return;"
    require(marker in match[1], "Local review core boundary changed")
    loader = match[1].split(marker, 1)[0] + "})();"
    source = PAGE
    source = replace_once(source, "<title>Neuron Explorer</title>",
                          "<title>Driving AI pilot — trained offline explorer</title>")
    csp = ("<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; "
           "script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src blob:; "
           "connect-src 'none'; base-uri 'none'; form-action 'none'\">\n")
    source = replace_once(source, "<style>", csp + "<style>")
    styles = ("\n.offline-setup{padding:18px 24px;background:#f5f7fc;"
              "border-bottom:1px solid #d8dfec}"
              ".offline-setup h2{margin:0 0 8px}.offline-warning{background:#fff1d6;color:#703d00;"
              "padding:12px 18px;margin:0}.offline-setup table{max-width:1000px}"
              ".offline-setup pre{white-space:pre-wrap}.offline-cases{display:flex;gap:14px;"
              "flex-wrap:wrap;margin:12px 0}.offline-cases a{color:#1c3478}"
              "body:not(.offline-loaded) header form{pointer-events:none;opacity:.5}"
              "#offline-status{white-space:pre-wrap}details{margin:12px 0}"
              ".comparison-table{margin:20px 0}.comparison-table caption{text-align:left;"
              "font-weight:bold;padding:8px 0}\n")
    source = replace_once(source, "</style>", styles + "</style>")
    table_rows = "".join(
        "<tr><td>" + html.escape(row["method"]) + "</td><td>"
        + f"{row['ap']:.4f}</td><td>{row['precision_at_k']:.2f}</td><td>"
        + html.escape(row["target_status"]) + "</td></tr>" for row in payload["comparison"])
    bookmarks = "".join(
        '<a data-offline-token="' + html.escape(row["sample_token"], quote=True)
        + '" href="#/sample/' + html.escape(row["sample_token"], quote=True) + '">'
        + html.escape(row["title"]) + "</a>" for row in payload["bookmarks"])
    first = payload["comparison"][0]
    caption = (f"{first['n']} 个 test 时间点 / {first['test_scenes']} 个场景 · "
               f"K={first['requested_k']} · seed={first['seed']}")
    included = payload["summary"]["frames"]
    excluded = len(payload["expected_frames"]) - included
    setup = ("<section class=\"offline-setup\"><h2>真实数据 · CPU 训练缓存演示</h2>"
             "<p>选择你已下载的全部四个 ZIP，然后查看真实六视角、激活单元和相似高激活样本。"
             "图像不上传；页面不重新训练或调用在线模型。</p>"
             "<input id=\"offline-parts\" type=\"file\" accept=\".zip\" multiple> "
             "<button id=\"offline-load\">打开原图</button>"
             "<p id=\"offline-status\" role=\"status\">请在电脑 Chrome 或 Edge 中打开此 HTML。</p>"
             "<p class=\"offline-warning\">AI 初标试跑，尚未人工核验。以下分数、误报/漏报和纯度"
             "均相对 AI 初标，不代表真实准确率或官方 benchmark。CPU ResNet18 路线与 Cosmos 独立。</p>"
             "<table><caption>" + html.escape(caption) + "</caption>"
             "<tr><th>模型</th><th>AP（AI 参考）</th><th>precision@4</th><th>validation 工作点</th></tr>"
             + table_rows + "</table><p>四组均未达到 validation 90% precision 目标，因此工作点预测全部拒绝。"
             "浏览器顶部的 normal/long-tail 使用原固定激活规则；下面逐帧表分别列出两种预测。"
             "Reward 与 no-reward SAE 的 checkpoint 相同，本轮未证明 reward 增益。</p>"
             "<div class=\"offline-cases\">" + bookmarks + "</div>"
             "<details><summary>冻结标注标准与使用范围</summary><p>"
             + str(included) + " 组 AI 暂定标签参与训练/评估；" + str(excluded)
             + " 组不确定样本未参与。人类核验仍未完成。</p>"
             "<pre id=\"offline-rubric\"></pre></details>"
             "</section>")
    source = replace_once(source, "<header><h1>Neuron Explorer</h1>",
                          setup + "<header><h1>Neuron Explorer · AI pilot</h1>")
    source = replace_once(
        source, '<a href="export/review.csv" download="review.csv">Export review CSV</a>',
        '<a id="review-export" href="#" download="review.csv">Export review CSV</a>')
    original_api = ("const api=p=>fetch(p).then(r=>r.ok?r.json():"
                    "r.json().then(e=>Promise.reject(e.error||r.statusText)));")
    replacement_api = ("const api=async p=>{await DrivingOffline.ready;"
                       "return DrivingOffline.api(p);};")
    source = replace_once(source, original_api, replacement_api)
    original_image = ('function img(tok,cam,w){return `<img loading="lazy" '
                      'src="img/${encodeURIComponent(tok)}/${cam}?w=${w||420}" '
                      'onerror="this.outerHTML=\'<div class=ph>no image</div>\'">`}')
    replacement_image = ('function img(tok,cam,w){return `<img loading="lazy" '
                         'src="${DrivingOffline.imageURL(tok,cam)}">`}')
    source = replace_once(source, original_image, replacement_image)
    comparison_js = r"""
  const comparison=document.createElement('table');comparison.className='comparison-table';
  const caption=document.createElement('caption');
  caption.textContent='同一帧：四模型预测（参考标签为 AI 初标）';comparison.append(caption);
  const heading=document.createElement('tr');
  const titles=['模型','分数','固定预测','validation 工作点预测','相对 AI 参考的工作点错误'];
  for(const title of titles){
   const th=document.createElement('th');th.textContent=title;heading.append(th);
  }
  comparison.append(heading);
  for(const row of f.comparisons){
   const tr=document.createElement('tr');
   const values=[row.method,row.score.toFixed(4),row.prediction===1?'long-tail':'normal',
    row.operating_prediction===1?'long-tail':'normal',row.operating_error||'—'];
   for(const value of values){
    const td=document.createElement('td');td.textContent=value;tr.append(td);
   }
   comparison.append(tr);
  }
  $('#main').append(comparison);
  document.querySelectorAll('.views img').forEach(image=>{
   image.style.cursor='zoom-in';
   image.onclick=()=>window.open(image.src,'_blank','noopener');
  });
"""
    catch_boundary = (" }catch(e){$('#main').innerHTML=`<p class=\"empty\">${esc(e)}</p>`}}"
                      "\nasync function showSearch")
    source = replace_once(source, catch_boundary, comparison_js + catch_boundary)
    scripts = ('<script id="offline-payload">globalThis.__DRIVING_OFFLINE_PAYLOAD__='
               + script_json(payload) + ';</script>\n<script id="review-loader">'
               + loader + '</script>\n<script id="offline-runtime">' + RUNTIME
               + '</script>\n<script id="explorer-ui">')
    source = replace_once(source, "<script>", scripts)
    require("fetch(" not in source, "Offline page must not make network requests")
    return source


def export(args):
    payload = build_payload(args.run, args.comparison_dir, args.prepared_dir,
                            args.samples_root, args.import_report)
    review_source = (ROOT / "tools/review_upload.html").read_text(encoding="utf-8")
    source = render_html(payload, review_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    require(not args.output.exists(), "Use a fresh output file")
    args.output.write_text(source, encoding="utf-8")
    report = {"status": "exported", "output": str(args.output), "bytes": args.output.stat().st_size,
              "sha256": digest(args.output), "frames": payload["summary"]["frames"],
              "source_images_embedded": 0, "label_scope": payload["label_scope"],
              "bookmarks": payload["bookmarks"], "browser_ui_verified": False}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("run", "comparison-dir", "prepared-dir", "samples-root", "import-report", "output")
    for name in names:
        parser.add_argument("--" + name, type=Path, required=True)
    print(json.dumps(export(parser.parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
