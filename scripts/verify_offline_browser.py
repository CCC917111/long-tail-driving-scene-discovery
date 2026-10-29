#!/usr/bin/env python3
"""Exercise the shipped offline explorer using clearly synthetic local pixels.

The fixture copies cached numbers only to test rendering and file binding. It
does not run a model, validate real images or create new experimental results.
Default execution requires an actual browser. --core-only never claims a UI pass.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import traceback
import zipfile

from PIL import Image, ImageDraw

CAMERAS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK",
           "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
SCRIPT = re.compile(r'<script id="([^"]+)">([\s\S]*?)</script>')
TRANSFER_ID = "11111111-1111-4111-8111-111111111111"
SCOPE = ("SYNTHETIC UI FIXTURE: generated pixels and metadata; cached numbers are "
         "copied rendering references, not new model inference or real-data validation.")


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def script_json(value):
    return (json_bytes(value).decode("utf-8").replace("<", "\\u003c")
            .replace(">", "\\u003e").replace("&", "\\u0026")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def extract_payload(source):
    scripts = dict(SCRIPT.findall(source))
    require(set(scripts) == {"offline-payload", "review-loader", "offline-runtime", "explorer-ui"},
            "The shipped explorer script layout changed")
    prefix = "globalThis.__DRIVING_OFFLINE_PAYLOAD__="
    code = scripts["offline-payload"]
    require(code.startswith(prefix) and code.endswith(";"), "Unsupported payload encoding")
    return json.loads(code[len(prefix):-1]), scripts


def validate_payload(payload):
    """Check source contracts without treating cached values as new inference."""
    schema = payload.get("schema")
    require(schema in {"offline-trained-explorer-ai-pilot-v1", "offline-trained-explorer-v2"},
            "Unsupported offline explorer schema")
    scope = payload["label_scope"]
    require(payload["summary"]["label_scope"] == scope, "Summary label scope differs")
    kind = payload.get("reference_kind", "ai_provisional")
    if schema == "offline-trained-explorer-v2":
        require(payload.get("reference_kind") == payload.get("reference", {}).get("kind")
                and payload["reference"]["kind"] in {"ai_provisional", "human_ai_exposed"},
                "Reference kind metadata differs")
        require(payload["reference"]["review_context"] == scope.get("review_context"),
                "Display reference context differs from label scope")
    else:
        require(kind == "ai_provisional", "Archived v1 supports only AI-provisional references")
    if kind == "ai_provisional":
        require(scope.get("label_protocol") == "ai-visual-pilot"
                and scope.get("human_reviewed") is False,
                "AI references must remain explicitly pending human review")
    elif kind == "human_ai_exposed":
        require(scope.get("label_protocol") == "manual"
                and scope.get("human_reviewed") is True
                and scope.get("review_context") == "post_ai_exposure_single_reviewer"
                and scope.get("human_ground_truth_verified") is False
                and scope.get("independent_human_review") is False
                and scope.get("prior_ai_labels_exposed") is True
                and scope.get("reviewer_count") == 1
                and scope.get("official_benchmark") is False,
                "Human references must disclose single-reviewer prior AI exposure")
    else:
        raise ValueError("Unsupported reference kind: " + str(kind))
    frames, methods = payload["frames"], payload["methods"]
    require(len(methods) == len(set(methods)) == 4, "Expected four distinct cached methods")
    require(len(frames) == payload["summary"]["frames"] > 0, "Cached frame count differs")
    selected = {row["sample_token"]: row for row in payload["expected_frames"]}
    require(len(selected) == len(payload["expected_frames"]) and set(frames) <= set(selected),
            "Cached frames must belong to the unique transfer selection")
    paths = set()
    for row in selected.values():
        require(set(row["images"]) == set(CAMERAS), "Transfer selection has missing cameras")
        paths.update(row["images"].values())
    require(set(payload["expected_images"]) == paths, "Expected image hash set differs")
    for token, frame in frames.items():
        require(frame["sample_token"] == token, "Cached frame token differs")
        require(frame["images"] == selected[token]["images"], "Cached image paths differ")
        rows = frame["comparisons"]
        require([row["method"] for row in rows] == methods,
                "Cached per-frame method order differs")
        require(all(math.isfinite(row["score"]) and row["prediction"] in (0, 1)
                    and row["operating_prediction"] in (0, 1) for row in rows),
                "Invalid cached score or prediction")
    require(payload["bookmarks"]
            and all(row["sample_token"] in frames for row in payload["bookmarks"]),
            "Bookmark points outside the cached frames")
    rows = list(csv.DictReader(io.StringIO(payload["review_csv"])))
    require(len(rows) == len(frames)
            and {row["sample_token"] for row in rows} == set(frames),
            "Review CSV differs from cached frame selection")
    require(all(not row["reviewer_label"] and not row["review_notes"] for row in rows),
            "Review CSV must preserve blank fields for the next reviewer")
    return kind


def make_image(index, camera_index, camera):
    color = ((index * 17 + camera_index * 31) % 140 + 40,
             (index * 23 + camera_index * 19) % 140 + 40,
             (index * 11 + camera_index * 41) % 140 + 40)
    image = Image.new("RGB", (320, 180), color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((6, 6, 313, 173), outline="white", width=3)
    draw.rectangle((12, 20, 307, 150), fill="white")
    draw.text((20, 28), "SYNTHETIC UI FIXTURE", fill="black")
    draw.text((20, 54), f"Frame {index:02d} / {camera}", fill="black")
    draw.text((20, 80), "NOT REAL DRIVING DATA", fill="black")
    draw.text((20, 106), "No model inference in this test", fill="black")
    output = io.BytesIO()
    image.save(output, "JPEG", quality=85)
    return output.getvalue()


def prepare_fixture(source_html, output):
    source = source_html.read_text(encoding="utf-8")
    original, scripts = extract_payload(source)
    reference_kind = validate_payload(original)
    payload = copy.deepcopy(original)
    selected = payload["expected_frames"]
    require(len(selected) == 80 and payload["frames_per_scene"] == 8,
            "This UI smoke requires the published 80-frame mini pilot")
    sensors = [{"token": f"synthetic_sensor_{camera}", "modality": "camera", "channel": camera}
               for camera in CAMERAS]
    calibrated = [{"token": f"synthetic_calibration_{camera}",
                   "sensor_token": f"synthetic_sensor_{camera}"} for camera in CAMERAS]
    scenes = sorted({row["scene_token"] for row in selected})
    scene_counts = {scene: sum(row["scene_token"] == scene for row in selected) for scene in scenes}
    require(len(scenes) == 10 and set(scene_counts.values()) == {8},
            "Unexpected scene selection")
    tables = {"scene": [{"token": scene, "name": f"SYNTHETIC scene {index:02d}"}
                        for index, scene in enumerate(scenes)],
              "sample": [], "sample_data": [], "calibrated_sensor": calibrated,
              "sensor": sensors}
    images, records = {}, []
    for index, row in enumerate(selected):
        tables["sample"].append({"token": row["sample_token"], "scene_token": row["scene_token"],
                                 "timestamp": row["timestamp"]})
        for camera_index, camera in enumerate(CAMERAS):
            path = f"samples/{camera}/SYNTHETIC_{index:02d}.jpg"
            body = make_image(index, camera_index, camera)
            images[path] = body
            row["images"][camera] = path
            tables["sample_data"].append({"token": f"synthetic_data_{index}_{camera}",
                                          "sample_token": row["sample_token"],
                                          "calibrated_sensor_token":
                                              f"synthetic_calibration_{camera}",
                                          "is_key_frame": True, "filename": path})
            records.append({"path": path, "sha256": digest(body), "bytes": len(body),
                            "output_path": path, "output_sha256": digest(body),
                            "output_bytes": len(body)})
    transported, metadata_hashes = {}, {}
    for table, value in tables.items():
        raw = json_bytes(value)
        path = f"v1.0-mini/{table}.json.gz"
        compressed = gzip.compress(raw, mtime=0)
        transported[path] = compressed
        metadata_hashes[table] = digest(raw)
        records.append({"path": path, "sha256": digest(compressed), "bytes": len(compressed),
                        "output_path": path[:-3], "output_sha256": digest(raw),
                        "output_bytes": len(raw)})
    transported.update(images)
    manifest = {"schema": "nuscenes-mini-browser-handoff-v1", "transfer_id": TRANSFER_ID,
                "part_count": 4, "frames_per_scene": 8,
                "source_archive": {"name": "SYNTHETIC-UI-FIXTURE.tgz", "bytes": 1,
                                   "last_modified": 0, "sha256": None},
                "version": "v1.0-mini",
                "selected_tokens": [{key: row[key] for key in
                                     ("sample_token", "scene_token", "timestamp")}
                                    for row in selected],
                "camera_order": list(CAMERAS), "files": records,
                "metadata_sha256": metadata_hashes, "labels_created": False,
                "archive_checksum_status": "SYNTHETIC UI FIXTURE; no original archive"}
    output.mkdir(parents=True, exist_ok=True)
    part_paths = []
    entries = list(transported.items())
    for part_index in range(4):
        members = entries[part_index::4]
        part_manifest = {**manifest, "part_index": part_index + 1,
                         "members": [name for name, _ in members]}
        target = output / f"synthetic-part-{part_index + 1:03d}-of-004.zip"
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("HANDOFF.json", json_bytes(part_manifest))
            for name, body in members:
                archive.writestr(name, body)
        require(target.stat().st_size <= 24 * 1024 * 1024, "Synthetic part exceeds tool limit")
        part_paths.append(target)
    payload["transfer_id"] = TRANSFER_ID
    payload["expected_metadata"] = metadata_hashes
    payload["expected_images"] = {path: digest(body) for path, body in images.items()}
    by_token = {row["sample_token"]: row for row in selected}
    for token, frame in payload["frames"].items():
        frame["images"] = by_token[token]["images"]
    payload["fixture_scope"] = SCOPE
    payload["scope_notes"] = [SCOPE, *payload.get("scope_notes", [])]
    payload_code = "globalThis.__DRIVING_OFFLINE_PAYLOAD__=" + script_json(payload) + ";"
    cloned = source.replace(scripts["offline-payload"], payload_code, 1)
    banner = ('<div id="synthetic-ui-fixture" style="padding:18px;background:#ffec87;'
              'color:#392900;font:bold 18px system-ui;border:4px solid #b96200">'
              + SCOPE + "</div>")
    require("<body>" in cloned, "Missing browser page body")
    cloned = cloned.replace("<body>", "<body>" + banner, 1)
    copied_scripts = dict(SCRIPT.findall(cloned))
    for name in ("review-loader", "offline-runtime", "explorer-ui"):
        require(copied_scripts[name] == scripts[name], "Shipped runtime changed in fixture")
    fixture_html = output / "SYNTHETIC-ui-fixture.html"
    fixture_html.write_text(cloned, encoding="utf-8")
    return payload, fixture_html, part_paths, {
        "scope": SCOPE, "source_html_sha256": digest(source_html.read_bytes()),
        "fixture_html_sha256": digest(fixture_html.read_bytes()),
        "selected_timestamps": 80, "synthetic_images": 480, "part_count": 4,
        "cached_reference_frames": len(payload["frames"]),
        "cached_model_values": len(payload["frames"]) * len(payload["methods"]),
        "source_schema": payload["schema"], "reference_kind": reference_kind,
        "source_label_scope": payload["label_scope"],
        "shipped_runtime_scripts_unchanged": True,
        "parts": [{"name": path.name, "bytes": path.stat().st_size,
                   "sha256": digest(path.read_bytes())} for path in part_paths]}


CORE_JS = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
(async()=>{
 const html=fs.readFileSync(process.argv[2],'utf8');
 const scripts=Object.fromEntries([...html.matchAll(
  /<script id="([^"]+)">([\s\S]*?)<\/script>/g)].map(x=>[x[1],x[2]]));
 for(const [name,code]of Object.entries(scripts))new vm.Script(code,{filename:name});
 for(const name of ['offline-payload','review-loader','offline-runtime'])
  vm.runInThisContext(scripts[name]);
 const api=DrivingOffline,files=process.argv.slice(3).map(path=>new Blob([fs.readFileSync(path)]));
 const loaded=await api.loadParts(files);assert(api.validateLoaded(loaded));
 api.attachLoaded(loaded);
 const payload=api.payload,frames=Object.values(payload.frames),methods=payload.methods;
 assert.equal(loaded.selected.length,80);assert.equal(loaded.images.size,480);
 assert.equal(api.api('api/summary').frames,frames.length);
 assert.deepEqual(api.api('api/summary').label_scope,payload.label_scope);
 for(const reference of frames){
  const sample=api.api('api/sample/'+reference.sample_token);
  assert.deepEqual(sample.comparisons,reference.comparisons);
  assert.deepEqual(sample.comparisons.map(row=>row.method),methods);
  assert.equal(sample.cameras.length,6);
  for(const camera of sample.cameras)
   assert(api.imageURL(sample.sample_token,camera).startsWith('blob:'));
 }
 const unit=api.api('api/units?top=1')[0];assert(unit);
 assert(api.api('api/unit/'+unit.unit).frames.length>0);
 const token=frames[0].sample_token;
 assert(api.api('api/search?q='+token).some(frame=>frame.sample_token===token));
 assert.throws(()=>api.api('api/sample/unknown-synthetic-token'));
 const wrongTransfer={...loaded,manifest:{...loaded.manifest,transfer_id:'wrong-transfer'}};
 assert.throws(()=>api.validateLoaded(wrongTransfer));
 const wrongDigest={...loaded,manifest:{...loaded.manifest,
  files:loaded.manifest.files.map((row,index)=>index===0?{...row,
   output_sha256:'0'.repeat(64)}:row)}};
 assert.throws(()=>api.validateLoaded(wrongDigest));
 console.log(JSON.stringify({status:'passed',synthetic_images:480,cached_frames:frames.length,
  cached_model_values:frames.length*methods.length,
  reference_kind:payload.reference_kind||'ai_provisional',
  all_cached_frame_apis_match:true,negative_transfer_and_hash_checks:true,
  browser_verified:false,scope:'Synthetic file transport and cached API binding only'}));
})().catch(error=>{console.error(error);process.exit(1)});
"""


def verify_core(fixture_html, parts, output):
    node = shutil.which("node")
    fallback = Path("/opt/codex/runtimes/codex-primary-runtime/dependencies/node/bin/node")
    if not node and fallback.is_file():
        node = str(fallback)
    require(node is not None, "Node is required for the synthetic core verification")
    script = output / "verify-core.js"
    script.write_text(CORE_JS)
    result = subprocess.run([node, str(script), str(fixture_html), *map(str, parts)],
                            capture_output=True, text=True, check=False, timeout=120)
    (output / "core.stdout.txt").write_text(result.stdout)
    (output / "core.stderr.txt").write_text(result.stderr)
    require(result.returncode == 0, "Synthetic core verification failed: " + result.stderr)
    return json.loads(result.stdout)


def verify_reference_banner(page, payload, expect):
    kind = payload.get("reference_kind", "ai_provisional")
    warning = page.locator(".offline-warning").first
    expect(warning).to_be_visible()
    if kind == "human_ai_exposed":
        expect(warning).to_have_attribute("data-reference-kind", kind)
        expect(warning).to_contain_text("Single-reviewer post-AI-exposure review")
        expect(warning).to_contain_text("not independent human ground truth")
        expect(warning).to_contain_text("or an official benchmark")
        body = page.locator("body").inner_text()
        for stale in ("human review pending", "Provisional AI-label pilot;",
                      "labels are provisional", "not been verified by a human"):
            require(stale not in body, "Human-reviewed UI contains stale AI warning: " + stale)
    else:
        expect(warning).to_contain_text("Provisional AI-label pilot")
        expect(warning).to_contain_text("human review pending")


def verify_summary_table(page, payload, expect):
    rows = page.locator(".offline-setup table").first.locator("tr")
    expect(rows).to_have_count(len(payload["methods"]) + 1)
    v2 = payload["schema"] == "offline-trained-explorer-v2"
    if v2:
        expect(rows.first).to_contain_text("AP (" + payload["reference"]["short_label"] + ")")
    for index, reference in enumerate(payload["comparison"]):
        cells = rows.nth(index + 1).locator("td").all_text_contents()
        require(cells[:3] == [reference["method"], f"{reference['ap']:.4f}",
                              f"{reference['precision_at_k']:.2f}"],
                "Displayed model ranking metrics differ from cache")
        if v2:
            threshold = reference["validation_threshold"]
            expected = [f"{reference['fixed_precision']:.2f} / {reference['fixed_recall']:.2f}",
                        reference["target_status"],
                        "—" if threshold is None else f"{threshold:.6f}",
                        f"{reference['operating_precision']:.2f} / "
                        f"{reference['operating_recall']:.2f}",
                        str(payload["operating_rules"][reference["method"]]["test_candidates"])]
        else:
            expected = [reference["target_status"]]
        require(cells[3:] == expected, "Displayed model operating summary differs from cache")


def verify_frame(page, payload, token, expect, mobile_width=None):
    expect(page.locator("#main .sub").first).to_contain_text(token)
    images = page.locator(".views img")
    expect(images).to_have_count(6)
    for image in images.all():
        image.scroll_into_view_if_needed()
        expect(image).to_be_visible()
        image.evaluate("image=>image.decode()")
        require(image.evaluate("image=>image.naturalWidth===320&&image.naturalHeight===180"),
                "Synthetic six-view image dimensions differ")
    rows = page.locator(".comparison-table tr")
    expect(rows).to_have_count(len(payload["methods"]) + 1)
    v2 = payload["schema"] == "offline-trained-explorer-v2"
    if v2:
        require(rows.first.locator("th").all_text_contents() == [
            "Model", "Score", "Reference label", "Fixed prediction", "Fixed-rule error",
            "Validation operating-point prediction", "Operating-point error relative to reference"],
            "Per-frame comparison columns changed")
        expect(page.locator(".comparison-table caption")).to_contain_text(
            payload["reference"]["label"])
    for index, reference in enumerate(payload["frames"][token]["comparisons"]):
        cells = rows.nth(index + 1).locator("td").all_text_contents()
        require(cells[0] == reference["method"], "Displayed model method differs")
        require(abs(float(cells[1]) - reference["score"]) <= 0.000051,
                "Displayed cached score differs")
        prediction = "long-tail" if reference["prediction"] else "normal"
        operating = "long-tail" if reference["operating_prediction"] else "normal"
        if v2:
            label = "long-tail" if reference["label"] else "normal"
            expected = [label, prediction, reference["fixed_error"] or "—", operating,
                        reference["operating_error"] or "—"]
        else:
            expected = [prediction, operating, reference["operating_error"] or "—"]
        require(cells[2:] == expected, "Displayed cached reference/prediction/errors differ")
    if v2 and mobile_width is not None:
        document_width = page.evaluate("document.documentElement.scrollWidth")
        require(document_width <= mobile_width + 2,
                f"Mobile page overflows: document {document_width}px, viewport {mobile_width}px")


def verify_viewport(browser, payload, fixture_html, parts, output, name, viewport):
    from playwright.sync_api import expect

    errors, external_requests = [], []
    context = browser.new_context(viewport=viewport, accept_downloads=True,
                                  is_mobile=name == "mobile", has_touch=name == "mobile")
    try:
        page = context.new_page()
        page.set_default_timeout(60000)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: external_requests.append(request.url)
                if request.url.startswith(("http://", "https://")) else None)
        page.goto(fixture_html.as_uri())
        expect(page.locator("#synthetic-ui-fixture")).to_contain_text("SYNTHETIC UI FIXTURE")
        verify_reference_banner(page, payload, expect)
        verify_summary_table(page, payload, expect)
        page.locator("#offline-parts").set_input_files(list(map(str, parts)))
        page.locator("#offline-load").click()
        expect(page.locator("#offline-status")).to_contain_text("Verified:", timeout=60000)
        expect(page.locator("#meta")).to_contain_text(str(len(payload["frames"])) + " frames")
        verify_reference_banner(page, payload, expect)
        mobile_width = viewport["width"] if name == "mobile" else None
        for index, bookmark in enumerate(payload["bookmarks"]):
            page.locator("[data-offline-token]").nth(index).click()
            verify_frame(page, payload, bookmark["sample_token"], expect, mobile_width)
        bookmark = payload["bookmarks"][0]
        page.locator("[data-offline-token]").first.click()
        verify_frame(page, payload, bookmark["sample_token"], expect, mobile_width)
        unit_link = page.locator("#main a.u").first
        unit_number = unit_link.inner_text().lstrip("#")
        unit_link.click()
        expect(page.locator("#main h2")).to_contain_text("Unit " + unit_number)
        expect(page.locator("#main .card").first).to_be_visible()
        page.locator("#q").fill(bookmark["sample_token"])
        page.locator("#find button").click()
        expect(page.locator("#main h2")).to_have_text("Search")
        expect(page.locator("#main .card")).to_have_count(1)
        page.locator("#main .card").first.click()
        verify_frame(page, payload, bookmark["sample_token"], expect, mobile_width)
        with page.expect_download() as downloaded:
            page.locator("#review-export").click()
        require(downloaded.value.suggested_filename == "review.csv", "Review filename differs")
        csv_path = output / f"synthetic-review-{name}.csv"
        downloaded.value.save_as(csv_path)
        require(csv_path.read_bytes() == payload["review_csv"].encode("utf-8"),
                "Downloaded review CSV differs from the cached export")
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            review_rows = list(csv.DictReader(handle))
        require(len(review_rows) == len(payload["frames"])
                and all(not row["reviewer_label"] and not row["review_notes"]
                        for row in review_rows),
                "Review CSV must preserve blank fields for the next reviewer")
        page.screenshot(path=str(output / f"{name}.png"), full_page=True)
        require(not errors, "Browser page errors: " + "; ".join(errors))
        require(not external_requests, "Offline page attempted external requests")
        return {"viewport": viewport, "six_images_loaded": True,
                "image_dimensions": [320, 180], "bookmarks_checked": len(payload["bookmarks"]),
                "four_model_table_matches_cache": True, "scoped_reference_banner": True,
                "model_metrics_and_operating_summary_match_cache": True,
                "unit_navigation": True, "search_and_sample_navigation": True,
                "review_csv_downloaded": True, "review_csv_rows": len(review_rows),
                "human_review_fields_blank": True, "page_errors": errors,
                "mobile_document_width_checked": mobile_width is not None
                and payload["schema"] == "offline-trained-explorer-v2",
                "document_width": page.evaluate("document.documentElement.scrollWidth"),
                "external_requests": external_requests}
    finally:
        context.close()


def verify_browser(payload, fixture_html, parts, output):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        executable = (shutil.which("google-chrome") or shutil.which("google-chrome-stable")
                      or shutil.which("chromium") or shutil.which("chromium-browser"))
        if executable is None:
            bundled = Path(playwright.chromium.executable_path)
            require(bundled.is_file(), "No Chrome/Chromium executable is installed")
            executable = str(bundled)
        browser = playwright.chromium.launch(executable_path=executable, headless=True,
                                             args=["--no-sandbox", "--disable-dev-shm-usage"])
        try:
            viewports = {"desktop": {"width": 1440, "height": 1000},
                         "mobile": {"width": 390, "height": 844}}
            results = {name: verify_viewport(browser, payload, fixture_html, parts,
                                             output, name, viewport)
                       for name, viewport in viewports.items()}
            return {"status": "passed", "browser_verified": True,
                    "browser_version": browser.version, "executable": executable,
                    "reference_kind": payload.get("reference_kind", "ai_provisional"),
                    "viewports": results, "scope": SCOPE}
        finally:
            browser.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-html", type=Path, default=Path("demo/driving_pilot.html"))
    parser.add_argument("--output", type=Path, default=Path("output/browser-smoke"))
    parser.add_argument("--core-only", action="store_true",
                        help="Verify synthetic ZIP/API core only; never claim browser verification")
    args = parser.parse_args()
    args.source_html, args.output = args.source_html.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"status": "failed", "browser_verified": False, "scope": SCOPE}
    code = 1
    try:
        payload, fixture_html, parts, fixture = prepare_fixture(args.source_html, args.output)
        report["fixture"] = fixture
        report["core"] = verify_core(fixture_html, parts, args.output)
        if args.core_only:
            report.update(status="core_passed_browser_not_run", browser_verified=False)
        else:
            report["browser"] = verify_browser(payload, fixture_html, parts, args.output)
            report.update(status="passed", browser_verified=True)
        code = 0
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        (args.output / "failure.txt").write_text(traceback.format_exc())
    (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
