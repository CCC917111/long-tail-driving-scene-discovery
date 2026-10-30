#!/usr/bin/env python3
"""Export a scoped cached explorer whose original images stay on the user's PC.

Reuses neuron_explorer.PAGE and Explorer, plus the already checked local ZIP
reader. No pixels, model inference or network calls are embedded in the HTML.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import re

from neuron_explorer import PAGE, Explorer
from prepare_driving import post_ai_review_scope
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


def reference_from_scope(scope):
    """Accept only the two recorded reference protocols, without upgrading either."""
    require(isinstance(scope, dict), "Explicit label scope is required")
    if scope.get("label_protocol") == "ai-visual-pilot":
        require(scope.get("human_reviewed") is False
                and not scope.get("review_context")
                and all(scope.get(key, False) is False for key in
                        ("human_ground_truth_verified", "independent_human_review",
                         "prior_ai_labels_exposed", "official_benchmark"))
                and scope.get("reviewer_count", 0) == 0,
                "AI label scope must be provisional with human review pending")
        return {"kind": "ai_provisional", "label": "provisional model-generated labels",
                "short_label": "model reference",
                "review_context": None,
                "description": "Model-generated provisional labels; human review pending. "
                               "Errors and purity are relative to these model references."}
    canonical = post_ai_review_scope()
    require(all(key in scope and type(scope[key]) is type(value) and scope[key] == value
                for key, value in canonical.items()),
            "Manual label scope requires canonical post-AI-exposure single-reviewer provenance")
    return {"kind": "human_ai_exposed",
            "label": "single-reviewer references after model-label disclosure",
            "short_label": "human review reference", "review_context": canonical["review_context"],
            "description": "Agreement with one human reviewer after model-label disclosure; "
                           "not independent human ground truth or an official benchmark."}


def error_type(label, prediction):
    return "fp" if prediction and not label else "fn" if label and not prediction else None


def failure_bookmarks(predictions, reference, failure_cases=None):
    """Use one saved fixed rule for the illustrative FP and two FNs."""
    rows = [row for row in predictions["mlp"].values() if row["split"] == "test"]
    false_positives = sorted((row for row in rows if row["label"] == 0
                              and row["prediction"] == 1),
                             key=lambda row: (-row["score"], row["sample_token"]))
    false_negatives = sorted((row for row in rows if row["label"] == 1
                              and row["prediction"] == 0),
                             key=lambda row: (row["score"], row["sample_token"]))
    selected = false_positives[:1] + false_negatives[:2]
    require(len(selected) == 3, "Expected an MLP fixed-rule test FP and two FNs")
    if failure_cases is not None:
        recorded = read_json(failure_cases)
        if reference["kind"] == "human_ai_exposed":
            require(reference_from_scope(recorded.get("label_scope")) == reference,
                    "Failure-case reference scope differs")
        cases = recorded["cases"]
        require([row["sample_token"] for row in cases]
                == [row["sample_token"] for row in selected],
                "Failure cases differ from the saved fixed-rule disagreements")
        for case, row in zip(cases, selected):
            score = case.get("score", case.get("model_score"))
            require(case.get("model", case.get("method")) == "mlp"
                    and isinstance(score, (float, int)) and math.isclose(score, row["score"]),
                    "Failure-case model or score differs from saved predictions")
            if reference["kind"] == "human_ai_exposed":
                require(case.get("reference_label") == row["label"]
                        and case.get("error_relative_to_final_review")
                        == error_type(row["label"], row["prediction"]),
                        "Failure-case label or error differs from saved predictions")
    return [{"title": "MLP fixed-rule " + ("false positive" if row["label"] == 0 else
                                           "false negative")
             + " (relative to " + reference["label"] + ")",
             "method": "mlp", "sample_token": row["sample_token"], "split": "test",
             "rule": "fixed probability >= 0.5", "error_type": error_type(
                 row["label"], row["prediction"]),
             "reference_label": row["label"], "score": row["score"],
             "selection": "Illustrative post-evaluation cases; not used for tuning"}
            for row in selected]


def build_payload(run, comparison_dir, prepared_dir, samples_root, import_report,
                  failure_cases=None):
    run, comparison_dir, prepared_dir = (Path(path).resolve()
                                         for path in (run, comparison_dir, prepared_dir))
    imported = read_json(import_report)
    prepared = read_json(prepared_dir / "prepare_metadata.json")
    selected = read_json(prepared_dir / "selected_samples.json")
    raw = read_json(prepared_dir / "raw_manifest.json")
    require(imported.get("status") == "completed" and imported.get("labels_created") is False
            and imported.get("parts") == 4 and imported.get("transfer_id"),
            "A completed original small-part import report is required")
    for filename in ("selected_samples.json", "raw_manifest.json", "rubric.md"):
        require(digest(prepared_dir / filename) == prepared["prepared_files"][filename],
                "Prepared source changed: " + filename)
    require(prepared["included_frames"] == imported["selected_samples"] == len(selected),
            "Imported selection count differs from preparation")
    raw_by_token = {row["sample_token"]: row for row in raw if not row.get("exclusion_reason")}
    require(len(raw_by_token) == len(selected)
            == len({row["sample_token"] for row in selected}),
            "Prepared selection has duplicate/missing rows")
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
    require(len(expected_images) == len(selected) * len(CAMERAS),
            "Original image filenames must be unique for each selected frame and camera")

    recorded_table = read_json(comparison_dir / "comparison/main_table.json")
    require(len(recorded_table) == 4 and {row["method"] for row in recorded_table}
            == {"logistic", "mlp", "sae_abstopk_none", "sae_abstopk_norm"},
            "Expected the four controlled comparison methods")
    comparison, _ = collect_runs([comparison_dir / row["method"] for row in recorded_table])
    for recorded, rebuilt in zip(recorded_table, comparison):
        require({key: value for key, value in recorded.items() if key != "run_dir"}
                == {key: value for key, value in rebuilt.items() if key != "run_dir"},
                "Comparison table differs from saved per-sample predictions")
    methods, model_predictions, checkpoints, operating_rules = [], {}, {}, {}
    manifest_hash, scope, reference = None, None, None
    for summary in comparison:
        name = summary["method"]
        require(name not in model_predictions, "Duplicate method in table")
        directory = comparison_dir / name
        metadata = read_json(directory / "run_metadata.json")
        evaluation = read_json(directory / "evaluation.json")
        current_scope = metadata.get("label_scope")
        current_reference = reference_from_scope(current_scope)
        require(evaluation.get("label_scope") == current_scope,
                "Evaluation and run label scopes differ")
        if scope is None:
            scope, reference = current_scope, current_reference
        require(scope == current_scope, "Mixed reference scopes in comparison runs")
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
        point = evaluation["operating_point"]
        for row in rows:
            prediction = int(point["threshold"] is not None
                             and row["score"] >= point["threshold"])
            require(row.get("operating_prediction") == prediction
                    and row.get("operating_error") == error_type(row["label"], prediction),
                    "Saved operating predictions differ: " + name)
        test = evaluation["splits"]["test"]["operating"]
        operating_rules[name] = {"fixed_rule": evaluation["fixed_rule"],
                                 "operating_point": point, "test": test,
                                 "test_candidates": test["confusion"]["tp"]
                                 + test["confusion"]["fp"]}
        model_predictions[name] = {row["sample_token"]: row for row in rows}
        methods.append(name)

    metadata = read_json(run / "run_metadata.json")
    require(run == comparison_dir / run.name and run.name.startswith("sae_")
            and run.name in methods
            and metadata.get("label_scope") == scope,
            "Activation run must belong to the same comparison and reference scope")
    metrics = read_json(run / "metrics.json")
    require(metrics.get("label_scope") == scope
            or (reference["kind"] == "ai_provisional" and "label_scope" not in metrics),
            "Activation metrics label scope differs")
    decision = metrics["decision"]
    store = ActivationStore(run)
    explorer = Explorer(store, samples_root=samples_root, eta=decision["eta"],
                        neuron_threshold=DEFAULT_NEURON_THRESHOLD, label_scope=scope)
    frames = {row["sample_token"]: explorer.units_of_frame(row["sample_token"], top=store.tail_dim)
              for row in store.samples}
    require(all(set(predictions) == set(frames) for predictions in model_predictions.values()),
            "Model token sets differ from trained activation cache")
    require(len(frames) == len(store) and store.labels is not None and store.splits is not None,
            "Activation cache must contain unique labelled and split-bound samples")
    image_paths = {row["sample_token"]: row["images"] for row in expected_frames}
    for token, frame in frames.items():
        require(token in image_paths, "Cached frame is outside the original ZIP selection")
        require(frame["cameras"] and set(frame["cameras"]) == set(CAMERAS),
                "The cached frame has missing original camera images")
        cached = store.samples[store.row_of[token]]
        require(all("samples/" + cached["images"][camera] == image_paths[token][camera]
                    for camera in CAMERAS), "Cached image mapping differs from original ZIPs")
        for method in methods:
            row = model_predictions[method][token]
            require(frame["label"] == ("long_tail" if row["label"] else "normal")
                    and frame["split"] == row["split"],
                    "Activation labels or splits differ from saved predictions")
        row = model_predictions[run.name][token]
        require(frame["prediction"] == ("long_tail" if row["prediction"] else "normal")
                and math.isclose(frame["tail_score"], row["score"], rel_tol=1e-6, abs_tol=1e-6),
                "Activation cache differs from saved SAE predictions")
        frame["images"] = image_paths[token]
        frame["comparisons"] = [{"method": method, **{
            key: model_predictions[method][token].get(key)
            for key in ("score", "label", "prediction", "operating_prediction", "operating_error")},
            "fixed_error": error_type(model_predictions[method][token]["label"],
                                      model_predictions[method][token]["prediction"])}
            for method in methods]

    units = {}
    for unit in range(store.tail_dim):
        stats = explorer.unit_stats(unit)
        units[str(unit)] = {**stats, "n_frames": stats["frames"],
                            "frames": [{"sample_token": row["sample_token"],
                                        "activation": row["activation"]}
                                       for row in explorer.frames_of_unit(unit, top=len(store))]}

    bookmarks = failure_bookmarks(model_predictions, reference, failure_cases)
    sae_identical = checkpoints["sae_abstopk_none"] == checkpoints["sae_abstopk_norm"]
    operating_notes = []
    for method, rule in operating_rules.items():
        point, test = rule["operating_point"], rule["test"]
        operating_notes.append(
            f"{method}: validation {point['status']}; test candidates="
            f"{rule['test_candidates']}, precision={test['precision']:.4f}, "
            f"recall={test['recall']:.4f}.")
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in (
        ROOT / "scripts/neuron_explorer.py", ROOT / "scripts/tail_activations.py",
        ROOT / "tools/review_upload.html", Path(__file__))}
    return {
        "schema": "offline-trained-explorer-v2", "reference": reference,
        "reference_kind": reference["kind"], "expected_part_count": imported["parts"],
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
        "operating_rules": operating_rules, "sae_checkpoints_identical": sae_identical,
        "checkpoints": checkpoints, "source_sha256": source_hashes,
        "manifest_sha256": manifest_hash,
        "scope_notes": [
            "Cached outputs from trained CPU ResNet18 + SAE; no Cosmos reproduction.",
            reference["description"],
            *operating_notes,
            ("Reward/no-reward SAE checkpoints are identical; no reward improvement is established."
             if sae_identical else "Distinct SAE checkpoints do not by themselves establish reward "
             "improvement."),
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
 check(model.manifest.transfer_id===payload.transfer_id,
       'These parts do not belong to this result bundle. Select the original four ZIPs.');
 check(model.manifest.part_count===payload.expected_part_count,
       'Part count differs from the original four ZIPs');
 check(model.manifest.frames_per_scene===payload.frames_per_scene,'Frames per scene differ');
 check(model.rubric_sha256===payload.rubric_sha256,'Labelling rubric version differs');
 check(model.selected.length===payload.expected_frames.length,'Selected frame count differs');
 for(let i=0;i<model.selected.length;i++){
  const actual=model.selected[i],expected=payload.expected_frames[i];
  check(actual.sample_token===expected.sample_token&&actual.scene_token===expected.scene_token&&
        actual.timestamp===expected.timestamp,'Selected frames differ from these results');
  for(const camera of reader.CAMERAS)
   check(actual.images[camera]===expected.images[camera],'Six-camera image mapping differs');
 }
 for(const [name,digest]of Object.entries(payload.expected_metadata))
  check(model.manifest.metadata_sha256[name]===digest,
        'Original metadata differs from these results');
 const inventory=new Map(model.manifest.files.map(row=>[row.path,row]));
 for(const [path,digest]of Object.entries(payload.expected_images))
  check(inventory.has(path)&&inventory.get(path).output_sha256===digest&&model.images.has(path),
        'Image contents differ from these results');
 return true;
}
function attachLoaded(model){
 validateLoaded(model);for(const url of urls.values())URL.revokeObjectURL(url);
 urls=new Map();loaded=model;resolveReady();return model;
}
function imageURL(token,camera){
 check(loaded,'Open the ZIP parts first');const frame=payload.frames[token];
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
 status.textContent='Verified: '+model.selected.length+' original frames; this page shows '+
  payload.summary.frames+' cached training/evaluation frames. Images stay on your computer.';
}catch(error){status.textContent='Could not open: '+error.message;}finally{load.disabled=false;}};
document.querySelectorAll('[data-offline-token]').forEach(link=>{
 link.addEventListener('click',event=>{
  if(!loaded){event.preventDefault();
   status.textContent='Select all four ZIPs and click "Open original images" first.';}
 });
});
window.addEventListener('unload',()=>{for(const url of urls.values())URL.revokeObjectURL(url);});
})();
"""


def render_html(payload, review_source):
    reference = reference_from_scope(payload.get("label_scope"))
    require(payload.get("reference", reference) == reference, "Payload reference scope differs")
    reviewed = reference["kind"] == "human_ai_exposed"
    match = re.search(r"<script>([\s\S]*?)</script>", review_source)
    require(match is not None, "Local review tool has no script")
    marker = "if(typeof document==='undefined')return;"
    require(marker in match[1], "Local review core boundary changed")
    loader = match[1].split(marker, 1)[0] + "})();"
    source = PAGE
    title = ("Driving reviewed references — trained offline explorer" if reviewed
             else "Driving provisional-label pilot — trained offline explorer")
    source = replace_once(source, "<title>Neuron Explorer</title>",
                          "<title>" + title + "</title>")
    csp = ("<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; "
           "script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src blob:; "
           "connect-src 'none'; base-uri 'none'; form-action 'none'\">\n")
    source = replace_once(source, "<style>", csp + "<style>")
    styles = ("\n.offline-setup{padding:18px 24px;background:#f5f7fc;"
              "border-bottom:1px solid #d8dfec}"
              ".offline-setup h2{margin:0 0 8px}.offline-warning{background:#fff1d6;color:#703d00;"
              "padding:12px 18px;margin:0}.offline-setup table{max-width:1000px}"
              ".offline-setup pre{white-space:pre-wrap;overflow-wrap:anywhere}"
              ".offline-cases{display:flex;gap:14px;"
              "flex-wrap:wrap;margin:12px 0}.offline-cases a{color:#1c3478}"
              "body:not(.offline-loaded) header form{pointer-events:none;opacity:.5}"
              "#offline-status{white-space:pre-wrap}details{margin:12px 0}"
              ".comparison-table{margin:20px 0}.comparison-table caption{text-align:left;"
              "font-weight:bold;padding:8px 0}.offline-results{overflow-x:auto}"
              ".offline-results table{min-width:850px}"
              ".offline-frame-comparison,.offline-frame-activations{overflow-x:auto;max-width:100%}"
              ".offline-frame-comparison .comparison-table{min-width:650px}\n")
    source = replace_once(source, "</style>", styles + "</style>")
    table_rows = "".join(
        "<tr><td>" + html.escape(row["method"]) + "</td><td>"
        + f"{row['ap']:.4f}</td><td>{row['precision_at_k']:.2f}</td><td>"
        + f"{row['fixed_precision']:.2f} / {row['fixed_recall']:.2f}</td><td>"
        + html.escape(row["target_status"]) + "</td><td>"
        + ("—" if row["validation_threshold"] is None
           else f"{row['validation_threshold']:.6f}") + "</td><td>"
        + f"{row['operating_precision']:.2f} / {row['operating_recall']:.2f}</td><td>"
        + str(payload["operating_rules"][row["method"]]["test_candidates"])
        + "</td></tr>" for row in payload["comparison"])
    bookmarks = "".join(
        '<a data-offline-token="' + html.escape(row["sample_token"], quote=True)
        + '" href="#/sample/' + html.escape(row["sample_token"], quote=True) + '">'
        + html.escape(row["title"]) + "</a>" for row in payload["bookmarks"])
    first = payload["comparison"][0]
    caption = (f"{first['n']} test frames / {first['test_scenes']} scenes · "
               f"K={first['requested_k']} · seed={first['seed']}")
    included = payload["summary"]["frames"]
    excluded = len(payload["expected_frames"]) - included
    warning = ("Single-reviewer references after prior model-label disclosure: final decisions "
               "were submitted after seeing provisional model-generated labels. Scores, false "
               "positives/negatives and purity "
               "are relative to these human review references; they are not independent human "
               "ground truth or an official benchmark. The CPU ResNet18 route is separate from "
               "Cosmos." if reviewed else
               "Provisional model-label pilot; human review pending. Scores, false "
               "positives/negatives "
               "and purity are relative to provisional model-generated labels, "
               "not measured accuracy "
               "or an official benchmark. The CPU ResNet18 route is separate from Cosmos.")
    rule_notes = []
    for method, rule in payload["operating_rules"].items():
        point, test = rule["operating_point"], rule["test"]
        if point["status"] == "target_unreachable":
            selection = (f"validation {point['target_precision']:.0%} precision target "
                         "unreachable; "
                         "no threshold selected, so the operating point produces no candidates")
        else:
            selection = (f"validation selected score ≥ {point['threshold']:.6f}; "
                         f"test produces {rule['test_candidates']} candidates, "
                         f"precision={test['precision']:.2f}, recall={test['recall']:.2f}")
        fixed = rule["fixed_rule"]
        rule_notes.append("<li><b>" + html.escape(method) + "</b>: fixed rule "
                          + html.escape(fixed["rule"])
                          + (f" (eta={fixed['eta']})" if "eta" in fixed else "")
                          + "; " + html.escape(selection) + ".</li>")
    reward_note = ("Reward and no-reward SAE checkpoints are identical; this run establishes no "
                   "reward improvement."
                   if payload["sae_checkpoints_identical"] else
                   "Different SAE checkpoints alone do not establish a reward improvement.")
    source_scope = json.dumps({"reference": reference, "label_scope": payload["label_scope"],
                               "manifest_sha256": payload["manifest_sha256"],
                               "checkpoints": payload["checkpoints"],
                               "source_sha256": payload["source_sha256"]},
                              ensure_ascii=False, indent=2)
    setup = ("<section class=\"offline-setup\"><h2>Real data · cached CPU training demo</h2>"
             "<p>Select all four downloaded ZIPs to explore original six-camera images, "
             "activated units and similar high-activation samples. Images stay local; "
             "this page does not retrain or call an online model.</p>"
             "<input id=\"offline-parts\" type=\"file\" accept=\".zip\" multiple> "
             "<button id=\"offline-load\">Open original images</button>"
             "<p id=\"offline-status\" role=\"status\">Open this HTML in desktop Chrome or "
             "Edge.</p>"
             "<p id=\"reference-scope-warning\" class=\"offline-warning\" data-reference-kind=\""
             + reference["kind"] + "\">" + html.escape(warning) + "</p>"
             "<div class=\"offline-results\"><table><caption>" + html.escape(caption) + "</caption>"
             "<tr><th>Model</th><th>AP (" + reference["short_label"] + ")</th><th>precision@"
             + str(first["requested_k"]) + "</th><th>Fixed-rule P / R</th>"
             "<th>Validation operating point</th><th>Threshold</th>"
             "<th>Test operating-point P / R</th>"
             "<th>Test candidates</th></tr>" + table_rows + "</table></div>"
             "<p>The normal/long-tail prediction at the top uses the original SAE fixed activation "
             "rule. Each frame table lists fixed-rule and validation operating-point predictions "
             "separately. AP and precision@K measure ranking; interpret operating-point test "
             "precision together with candidate count and recall. " + reward_note + "</p><ul>"
             + "".join(rule_notes) + "</ul><p>Activated units use numeric identifiers; "
             "semantic names "
             "have not been validated. The following false positive and two false negatives use "
             "the same MLP fixed rule. They were selected after evaluation as illustrations and "
             "were not used for tuning.</p>"
             "<div class=\"offline-cases\">" + bookmarks + "</div>"
             "<details><summary>Frozen labelling rubric and scope</summary><p>"
             + str(included) + " frames with " + reference["label"] + " were used for "
             "training/evaluation; " + str(excluded) + " uncertain frames were excluded.</p>"
             "<pre id=\"offline-rubric\"></pre></details>"
             "<details><summary>Result provenance and reference scope</summary><pre>"
             + html.escape(source_scope) + "</pre></details>"
             "</section>")
    source = replace_once(source, "<header><h1>Neuron Explorer</h1>",
                          setup + "<header><h1>Neuron Explorer · "
                          + ("reviewed references" if reviewed else "provisional model labels")
                          + "</h1>")
    inherited_warning = (" if(S.label_scope&&S.label_scope.human_reviewed===false){\n"
                         "  const warning=document.createElement('p');warning.textContent='"
                         "Model-generated label pilot: labels are provisional and have not "
                         "been verified by a human. Scores, errors and purity refer to "
                         "model-generated labels; this is not "
                         "benchmark accuracy.';\n"
                         "  warning.style.cssText='margin:0;padding:12px 20px;background:#fff1d6;"
                         "color:#703d00';\n"
                         "  document.querySelector('header').after(warning);\n }")
    source = replace_once(source, inherited_warning, "")
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
  const activationTable=$('#main > table');
  if(activationTable){
   const activationWrap=document.createElement('div');
   activationWrap.className='offline-frame-activations';
   activationTable.replaceWith(activationWrap);activationWrap.append(activationTable);
  }
  const comparison=document.createElement('table');comparison.className='comparison-table';
  const caption=document.createElement('caption');
  caption.textContent='Same frame: four-model predictions (reference: '+
   DrivingOffline.payload.reference.label+')';comparison.append(caption);
  const heading=document.createElement('tr');
  const titles=['Model','Score','Reference label','Fixed prediction','Fixed-rule error',
   'Validation operating-point prediction','Operating-point error relative to reference'];
  for(const title of titles){
   const th=document.createElement('th');th.textContent=title;heading.append(th);
  }
  comparison.append(heading);
  for(const row of f.comparisons){
   const tr=document.createElement('tr');
   const values=[row.method,row.score.toFixed(4),row.label===1?'long-tail':'normal',
    row.prediction===1?'long-tail':'normal',row.fixed_error||'—',
    row.operating_prediction===1?'long-tail':'normal',row.operating_error||'—'];
   for(const value of values){
    const td=document.createElement('td');td.textContent=value;tr.append(td);
   }
   comparison.append(tr);
  }
  const comparisonWrap=document.createElement('div');
  comparisonWrap.className='offline-frame-comparison';comparisonWrap.append(comparison);
  $('#main').append(comparisonWrap);
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
                            args.samples_root, args.import_report,
                            getattr(args, "failure_cases", None))
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
    parser.add_argument("--failure-cases", type=Path,
                        help="Optional saved cases to check against computed MLP bookmarks")
    print(json.dumps(export(parser.parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
