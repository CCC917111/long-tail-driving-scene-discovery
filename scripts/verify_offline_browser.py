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
    payload = copy.deepcopy(original)
    selected = payload["expected_frames"]
    require(len(selected) == 80 and payload["frames_per_scene"] == 8,
            "This UI smoke requires the published 80-frame mini pilot")
    require(payload["summary"]["frames"] == 73, "Expected the 73-frame cached pilot")
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
    payload["scope_notes"] = [SCOPE]
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
 assert.equal(loaded.selected.length,80);assert.equal(loaded.images.size,480);
 assert.equal(api.api('api/summary').frames,73);
 for(const bookmark of api.payload.bookmarks){
  const sample=api.api('api/sample/'+bookmark.sample_token);
  assert.equal(sample.comparisons.length,4);assert.equal(sample.cameras.length,6);
  assert(api.imageURL(sample.sample_token,'CAM_FRONT').startsWith('blob:'));
 }
 console.log(JSON.stringify({status:'passed',synthetic_images:480,cached_frames:73,
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


def verify_browser(payload, fixture_html, parts, output):
    from playwright.sync_api import expect, sync_playwright

    errors = []
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
            context = browser.new_context(viewport={"width": 1440, "height": 1000},
                                          accept_downloads=True)
            page = context.new_page()
            page.set_default_timeout(60000)
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(fixture_html.as_uri())
            expect(page.locator("#synthetic-ui-fixture")).to_contain_text("SYNTHETIC UI FIXTURE")
            page.locator("#offline-parts").set_input_files(list(map(str, parts)))
            page.locator("#offline-load").click()
            expect(page.locator("#offline-status")).to_contain_text("校验完成", timeout=60000)
            bookmark = payload["bookmarks"][0]
            page.locator("[data-offline-token]").first.click()
            expect(page.locator("#main .sub").first).to_contain_text(bookmark["sample_token"])
            page.wait_for_function("""()=>{
                const images=[...document.querySelectorAll('.views img')];
                return images.length===6&&images.every(image=>image.complete&&
                    image.naturalWidth===320&&image.naturalHeight===180);
            }""")
            rows = page.locator(".comparison-table tr")
            require(rows.count() == 5, "The per-frame comparison table is missing a model")
            references = payload["frames"][bookmark["sample_token"]]["comparisons"]
            for index, reference in enumerate(references):
                cells = rows.nth(index + 1).locator("td").all_text_contents()
                require(cells[0] == reference["method"], "Displayed model method differs")
                require(abs(float(cells[1]) - reference["score"]) <= 0.000051,
                        "Displayed cached score differs")
                require(cells[2] == ("long-tail" if reference["prediction"] else "normal"),
                        "Displayed fixed-rule prediction differs")
                expected = "long-tail" if reference["operating_prediction"] else "normal"
                require(cells[3] == expected, "Displayed validation operating prediction differs")
                require(cells[4] == (reference["operating_error"] or "—"),
                        "Displayed AI-reference disagreement differs")
            page.screenshot(path=str(output / "desktop.png"), full_page=True)
            unit_link = page.locator("#main a.u").first
            unit_number = unit_link.inner_text().lstrip("#")
            unit_link.click()
            expect(page.locator("#main h2")).to_contain_text("Unit " + unit_number)
            require(page.locator("#main .card").count() > 0, "Unit navigation has no frames")
            page.locator("#q").fill(bookmark["sample_token"][:12])
            page.locator("#find button").click()
            expect(page.locator("#main h2")).to_have_text("Search")
            expect(page.locator("#main .card").first).to_be_visible()
            page.locator("#main .card").first.click()
            expect(page.locator("#main .sub").first).to_contain_text(bookmark["sample_token"])
            with page.expect_download() as downloaded:
                page.locator("#review-export").click()
            csv_path = output / "synthetic-review.csv"
            downloaded.value.save_as(csv_path)
            require(csv_path.read_bytes() == payload["review_csv"].encode("utf-8"),
                    "Downloaded review CSV differs from the cached export")
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                review_rows = list(csv.DictReader(handle))
            blank_human_fields = all(not row["reviewer_label"] and not row["review_notes"]
                                     for row in review_rows)
            require(len(review_rows) == 73 and blank_human_fields,
                    "Review CSV must preserve blank human-review fields")
            page.set_viewport_size({"width": 390, "height": 844})
            page.locator(".views").scroll_into_view_if_needed()
            expect(page.locator(".views img")).to_have_count(6)
            page.screenshot(path=str(output / "mobile.png"), full_page=True)
            require(not errors, "Browser page errors: " + "; ".join(errors))
            return {"status": "passed", "browser_verified": True,
                    "browser_version": browser.version, "executable": executable,
                    "six_images_loaded": True, "image_dimensions": [320, 180],
                    "four_model_table_matches_cache": True, "unit_navigation": True,
                    "search_and_sample_navigation": True, "review_csv_downloaded": True,
                    "human_review_fields_blank": True, "page_errors": errors,
                    "desktop_viewport": [1440, 1000], "mobile_viewport": [390, 844],
                    "mobile_six_image_elements": True,
                    "scope": SCOPE}
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
