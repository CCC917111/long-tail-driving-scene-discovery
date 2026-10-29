#!/usr/bin/env python3
"""Check a supplied new-scene review page in real Chromium.

The browser creates temporary CI_TEST_ONLY decisions solely to verify the UI.
They are not human labels and must never enter a training or evaluation set.
Works with a local image bundle or the exporter's pinned source-image mode.
This verifier uses only the standard library and Playwright, without torch.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import threading
import traceback
from urllib.parse import quote, unquote, urljoin


CAMERAS = {"CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK",
           "CAM_BACK_LEFT", "CAM_BACK_RIGHT"}
FIELDS = ["sample_token", "label", "review_status", "reviewer", "review_notes",
          "review_context", "model_signal_exposed"]


def require(value, message):
    if not value:
        raise AssertionError(message)


@contextmanager
def serve_html(html):
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            if self.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            super().do_GET()

    handler = partial(QuietHandler, directory=str(html.parent))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:" + str(server.server_port) + "/" + quote(html.name)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--browser-executable", type=Path)
    return parser.parse_args(argv)


def read_payload(path):
    source = path.read_text(encoding="utf-8")
    match = re.search(r'<script id="payload" type="application/json">(.*?)</script>',
                      source, flags=re.S)
    require(match is not None, "Missing review payload")
    payload = json.loads(match.group(1))
    require(payload["schema"] == "unlabelled-review-v1", "Unknown review page schema")
    scope = payload["scope"]
    require(scope["has_reference_labels"] is False and scope["human_reviewed"] is False
            and scope["is_benchmark_evaluation"] is False,
            "The supplied review page must remain unlabelled")
    tokens = [frame["sample_token"] for frame in payload["frames"]]
    require(tokens and len(tokens) == len(set(tokens)), "Duplicate or empty review frames")
    require(all("label" not in frame and "split" not in frame
                and set(frame["images"]) == CAMERAS for frame in payload["frames"]),
            "Every new frame must have six cameras and no reference label/split")
    return payload


def row_state(page, index):
    return page.evaluate("i => DrivingUnlabelledReview.state.rows[i]", index)


def signals_hidden(page):
    require(not page.locator("#signals").is_visible(), "Model signals must initially be hidden")
    require(page.locator("#reveal").is_visible(), "Missing explicit model-reveal button")
    require(page.evaluate("DrivingUnlabelledReview.state.exposed") is False,
            "Fresh browser state should have no model-signal exposure")


def download_csv(page, path):
    with page.expect_download() as pending:
        page.locator("#export").click()
    download = pending.value
    download.save_as(path)
    require(download.failure() is None, "Review CSV download failed")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == FIELDS, "Unexpected review CSV fields")
        return list(reader)


def verify_images(page, frame, index):
    page.locator("#choose").select_option(str(index))
    page.wait_for_function(
        "token => document.getElementById('frame').textContent.includes(token)",
        arg=frame["sample_token"])
    page.wait_for_function("""() => {
      const images=[...document.querySelectorAll('#views img')];
      return images.length===6 && images.every(i=>i.complete &&
        i.naturalWidth>0 && i.naturalHeight>0);
    }""")
    records = page.locator("#views img").evaluate_all("""images => images.map(img=>({
      camera:img.alt,url:img.currentSrc,width:img.naturalWidth,height:img.naturalHeight,
      object_fit:getComputedStyle(img).objectFit
    }))""")
    require(len(records) == 6 and {r["camera"] for r in records} == CAMERAS,
            "Each group must display all six named camera views")
    require(len({r["url"] for r in records}) == 6, "Six views must use distinct image URLs")
    for record in records:
        expected = urljoin(page.url, frame["images"][record["camera"]])
        require(unquote(record["url"]) == unquote(expected), "Image URL differs from frame")
        require(record["object_fit"] == "contain", "Review images must preserve the full view")
        record["sample_token"] = frame["sample_token"]
    return records


def verify_viewport(browser, name, viewport, html_url, payload, output, report):
    context = browser.new_context(viewport=viewport, accept_downloads=True)
    page = context.new_page()
    page.set_default_timeout(20000)
    page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
    page.on("console", lambda message: report["console_errors"].append({
        "viewport": name, "text": message.text}) if message.type == "error" else None)
    page.on("requestfailed", lambda request: report["request_failures"].append({
        "viewport": name, "url": request.url, "failure": request.failure}))
    page.on("response", lambda response: report["http_errors"].append({
        "viewport": name, "url": response.url, "status": response.status})
        if response.status >= 400 else None)
    try:
        page.goto(html_url, wait_until="load", timeout=60000)
        signals_hidden(page)
        require(page.locator("#score").inner_text() == ""
                and page.locator("#training-scope").inner_text() == "",
                "Model signal text must not be rendered before explicit reveal")
        tokens = [frame["sample_token"] for frame in payload["frames"]]
        rows = download_csv(page, output / (name + "_initial_blank.csv"))
        require([row["sample_token"] for row in rows] == tokens,
                "Initial CSV tokens differ from the supplied page")
        require(all(row["label"] == row["reviewer"] == row["review_notes"] == ""
                    and row["review_status"] == "pending"
                    and row["review_context"] == "model_hidden_single_reviewer"
                    and row["model_signal_exposed"] == "false" for row in rows),
                "Fresh CSV must contain only blank pending decisions")
        images = []
        for index, frame in enumerate(payload["frames"]):
            images.extend(verify_images(page, frame, index))
            sizes = page.evaluate("""() => ({width:innerWidth,
              document_width:document.documentElement.scrollWidth,
              body_width:document.body.scrollWidth})""")
            require(max(sizes["document_width"], sizes["body_width"]) <= sizes["width"] + 2,
                    f"{name}: horizontal overflow: {sizes}")
        require(len({record["url"] for record in images}) == len(images),
                "Frames unexpectedly reuse original camera images")
        page.locator("#choose").select_option("0")
        page.locator("#normal").click()
        require(row_state(page, 0)["label"] == "", "A reviewer name must be required")
        require("Reviewer is required" in page.locator("#status").inner_text(),
                "Missing reviewer validation feedback")
        reviewer = "CI_TEST_ONLY"
        notes = 'CI test only, not a human judgment.\nQuoted "note".'
        page.locator("#reviewer").fill(reviewer)
        page.locator("#notes").fill(notes)
        page.locator("#normal").click()
        before = row_state(page, 0)
        require(before["label"] == "normal_core" and before["review_notes"] == notes
                and before["review_context"] == "model_hidden_single_reviewer"
                and before["model_signal_exposed"] is False,
                "The first simulated decision must retain its hidden-signal context")
        page.reload(wait_until="load")
        require(row_state(page, 0) == before, "Review decision did not survive refresh")
        signals_hidden(page)
        page.locator("#choose").select_option("0")
        require(page.locator("#notes").input_value() == notes, "Notes were not restored")
        page.screenshot(path=str(output / (name + "_hidden.png")), full_page=True)
        page.locator("#reveal").click()
        require(page.locator("#signals").is_visible(), "Model signals did not appear")
        require(not page.locator("#reveal").is_visible(), "Reveal button should be replaced")
        require("eta=" + str(payload["decision"]["eta"])
                in page.locator("#score").inner_text(), "Wrong saved model threshold")
        require(payload["decision"]["rule"] in page.locator("#score").inner_text(),
                "Wrong saved fixed decision rule")
        require(page.locator("#units li").count() == len(payload["frames"][0]["units"]),
                "Displayed activation units differ from supplied model outputs")
        page.locator("#notes").fill(notes + " After model reveal.")
        page.locator("#uncertain").click()
        after = row_state(page, 0)
        require(after["label"] == after["review_status"] == "uncertain"
                and after["reviewer"] == reviewer
                and after["model_signal_exposed"] is True
                and after["review_context"] == "post_model_signal_exposure_single_reviewer",
                "Revised decision must record model-signal exposure")
        page.reload(wait_until="load")
        require(row_state(page, 0) == after, "Exposed review record did not survive refresh")
        require(page.locator("#signals").is_visible(), "Exposure cannot reset on refresh")
        rows = download_csv(page, output / (name + "_simulated_review.csv"))
        require(rows[0]["review_notes"] == after["review_notes"]
                and rows[0]["model_signal_exposed"] == "true"
                and rows[0]["label"] == "uncertain", "CSV did not retain quoted review state")
        require(all(row["label"] == "" for row in rows[1:]),
                "Unreviewed frames must stay blank after the UI test")
        page.locator("#choose").select_option("0")
        page.screenshot(path=str(output / (name + "_revealed.png")), full_page=True)
        page.locator("#clear").click()
        cleared = row_state(page, 0)
        require(cleared["label"] == "" and cleared["review_status"] == "pending"
                and cleared["model_signal_exposed"] is True,
                "Clear must remove the decision without erasing exposure history")
        return {"viewport": viewport, "image_count": len(images),
                "all_views_decoded": True, "images": images,
                "initial_blank_export_verified": True, "reviewer_required": True,
                "simulated_decision_and_csv_verified": True,
                "model_exposure_and_refresh_verified": True, "no_horizontal_overflow": True}
    finally:
        context.close()


def main(argv=None):
    args = parse_args(argv)
    html, output = args.html.resolve(), args.output_dir.resolve()
    require(not output.exists(), "Preserve prior evidence; use a new output directory")
    output.mkdir(parents=True)
    report = {"schema": "new-scene-review-browser-v1", "status": "running",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "html_sha256": hashlib.sha256(html.read_bytes()).hexdigest(),
              "real_browser_executed": False, "page_errors": [], "console_errors": [],
              "request_failures": [],
              "http_errors": [], "viewports": {}, "human_annotation_performed": False,
              "serving_mode": "temporary_loopback_http",
              "simulated_reviewer": "CI_TEST_ONLY",
              "scope": ("UI verification with fake temporary decisions; "
                        "not human labels or accuracy")}
    try:
        from playwright.sync_api import sync_playwright

        payload = read_payload(html)
        report["frames"] = len(payload["frames"])
        report["image_mode"] = payload["image_mode"]
        with serve_html(html) as html_url, sync_playwright() as playwright:
            launch = {"headless": True, "args": ["--no-sandbox", "--disable-dev-shm-usage"]}
            if args.browser_executable:
                launch["executable_path"] = str(args.browser_executable)
            proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
            if proxy:
                # The temporary server is local IPC. Keep external source images
                # on the inherited proxy and leave TLS verification enabled.
                launch["proxy"] = {"server": proxy, "bypass": "127.0.0.1,localhost"}
            browser = playwright.chromium.launch(**launch)
            try:
                report["real_browser_executed"] = True
                report["browser_version"] = browser.version
                for name, viewport in (("desktop", {"width": 1365, "height": 1000}),
                                       ("mobile", {"width": 390, "height": 844})):
                    report["viewports"][name] = verify_viewport(
                        browser, name, viewport, html_url, payload, output, report)
            finally:
                browser.close()
        require(not report["page_errors"] and not report["request_failures"]
                and not report["http_errors"] and not report["console_errors"],
                "The page reported browser or image request errors")
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error),
                           "traceback": traceback.format_exc()}
        raise
    finally:
        (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: report[key] for key in
                          ("status", "frames", "real_browser_executed",
                           "human_annotation_performed")
                          if key in report}))


if __name__ == "__main__":
    main()
