#!/usr/bin/env python3
"""Verify the explorer in real Chromium, including images and CSV download.

This checks an existing run; it never generates or mocks model/data artifacts.
It starts an ephemeral loopback HTTP server, exercises desktop and mobile
viewports, saves screenshots/downloads, and writes ``ui_check.json`` even on
failure. Loopback networking and an installed Playwright/Chromium are required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import threading
import traceback
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from neuron_explorer import CAMERAS, DEFAULT_NEURON_THRESHOLD, build_explorer, make_server


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="Existing directory with activations.npz and metrics.json")
    parser.add_argument("--samples-root", type=Path, required=True,
                        help="Directory containing the run's actual six-camera images")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-token", required=True)
    parser.add_argument("--unit", type=int,
                        help="Unit linking to the sample; otherwise select an eligible active unit")
    parser.add_argument("--neuron-threshold", type=float, default=DEFAULT_NEURON_THRESHOLD,
                        help="Explorer interpretation threshold; use the intended serving setting")
    parser.add_argument("--browser-executable", type=Path, default=Path("/usr/bin/chromium"))
    return parser.parse_args(argv)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_unit(explorer, token, requested):
    frame = explorer.units_of_frame(token)
    require(set(frame["cameras"]) == set(CAMERAS),
            f"{token}: all six camera image files must exist")
    overview = {item["unit"] for item in explorer.units(top=80)}
    candidates = ([requested] if requested is not None
                  else [item["unit"] for item in frame["units"]])
    for unit in candidates:
        if unit not in overview or not any(item["unit"] == unit for item in frame["units"]):
            continue
        # The UI requests its 48 strongest frames for a unit. Choose a link that
        # actually exists instead of manufacturing a route in the browser.
        if any(item["sample_token"] == token
               for item in explorer.frames_of_unit(unit, top=48)):
            return unit, frame
    raise ValueError(
        "No displayed unit links to this sample among its 48 strongest frames. "
        "Choose a supported sample/unit or explicitly set the intended neuron threshold."
    )


def collect_events(page, viewport, report):
    page.on("pageerror", lambda error: report["page_errors"].append(
        {"viewport": viewport, "error": str(error)}))
    page.on("requestfailed", lambda request: report["request_failures"].append({
        "viewport": viewport, "url": request.url, "method": request.method,
        "resource_type": request.resource_type, "failure": request.failure,
    }))

    def on_response(response):
        record = {"viewport": viewport, "url": response.url, "status": response.status,
                  "content_type": response.headers.get("content-type", "")}
        report["http_responses"].append(record)
        if response.status >= 400:
            report["http_errors"].append(record)

    page.on("response", on_response)


def settle(page):
    page.wait_for_load_state("networkidle")


def wait_for_unit(page, unit):
    page.locator("#main h2").filter(has_text=re.compile(rf"^Unit {unit}(?:\s|$)")).wait_for()
    settle(page)


def wait_for_sample(page, token):
    page.wait_for_function(
        "token => document.querySelector('#main .sub')?.textContent.includes(token) "
        "&& document.querySelectorAll('#main .views figure').length === 6",
        arg=token,
    )


def verify_six_images(page, token, viewport, report):
    wait_for_sample(page, token)
    captions = page.locator("#main .views figcaption").all_text_contents()
    require(len(captions) == 6 and set(captions) == set(CAMERAS),
            f"{viewport}: expected exactly six named camera figures, got {captions}")
    for index in range(6):
        image = page.locator("#main .views img").nth(index)
        image.scroll_into_view_if_needed()
        page.wait_for_function(
            "index => { const img = document.querySelectorAll('#main .views img')[index]; "
            "return img && img.complete && img.naturalWidth > 0 && img.naturalHeight > 0; }",
            arg=index,
        )
    settle(page)
    images = page.locator("#main .views img").evaluate_all(
        "images => images.map(img => ({url: img.currentSrc, width: img.naturalWidth, "
        "height: img.naturalHeight, complete: img.complete}))"
    )
    require(len(images) == 6, f"{viewport}: an image was replaced by an error placeholder")
    require(len({image["url"] for image in images}) == 6,
            f"{viewport}: expected six distinct camera image URLs")
    for camera, image in zip(captions, images):
        require(unquote(urlparse(image["url"]).path) == f"/img/{token}/{camera}",
                f"{viewport}: image URL does not match its sample/camera caption: {image['url']}")
        image["camera"] = camera
        image["sample_token"] = token
        require(any(response["viewport"] == viewport and response["url"] == image["url"]
                    and response["status"] == 200
                    and response["content_type"].startswith("image/")
                    for response in report["http_responses"]),
                f"{viewport}: no successful image HTTP response for {image['url']}")
    return images


def verify_download(page, output, viewport, explorer):
    link = page.get_by_role("link", name="Export review CSV", exact=True)
    require(link.count() == 1, f"{viewport}: expected one Export review CSV link")
    require(link.get_attribute("href") in ("export/review.csv", "/export/review.csv"),
            f"{viewport}: unexpected review export URL")
    with page.expect_download() as pending:
        link.click()
    download = pending.value
    path = output / f"{viewport}_review.csv"
    download.save_as(path)
    require(download.failure() is None, f"{viewport}: review CSV download failed")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    expected = {"sample_token", "tail_score", "reviewer_label", "review_notes"}
    require(expected <= set(fields), f"{viewport}: missing CSV fields: {expected - set(fields)}")
    require(not ({"label", "reference_label", "ground_truth", "true_label", "y_true"}
                 & set(fields)), f"{viewport}: reference labels leaked into the review CSV")
    require(bool(rows), f"{viewport}: downloaded review CSV is empty")
    tokens = [row["sample_token"] for row in rows]
    require(len(set(tokens)) == len(tokens), f"{viewport}: duplicated export sample tokens")
    require(set(tokens) <= set(explorer.store.row_of), f"{viewport}: unknown exported sample")
    for row in rows:
        require(math.isfinite(float(row["tail_score"])), f"{viewport}: nonfinite tail score")
        require(row["reviewer_label"] == "" and row["review_notes"] == "",
                f"{viewport}: independent reviewer fields must initially be blank")
    return {"path": str(path), "url": download.url, "row_count": len(rows),
            "columns": fields, "sha256": sha256(path),
            "suggested_filename": download.suggested_filename,
            "reference_labels_omitted": True}


def verify_viewport(browser, base_url, explorer, token, unit, name, size, output, report):
    context = browser.new_context(viewport=size, accept_downloads=True,
                                  is_mobile=name == "mobile", has_touch=name == "mobile")
    page = context.new_page()
    page.set_default_timeout(20000)
    collect_events(page, name, report)
    result = {"viewport": size, "checks": {}, "screenshots": {}}
    report["viewports"][name] = result
    sample_href = f"#/sample/{quote(token, safe='')}"
    card = f"#main a.card[href='{sample_href}']"
    unit_href = f"#/unit/{unit}"
    try:
        response = page.goto(base_url, wait_until="networkidle")
        require(response is not None and response.status == 200,
                f"{name}: homepage did not return HTTP 200")
        page.get_by_role("heading", name="Neuron Explorer", exact=True).wait_for()
        page.locator(f"#units .unit-row[data-u='{unit}']").wait_for()
        page.wait_for_function(
            "eta => document.querySelector('#meta').textContent.includes(`eta ${eta}`)",
            arg=explorer.eta,
        )
        result["checks"]["homepage_units"] = page.locator("#units .unit-row").count()
        result["summary_text"] = page.locator("#meta").inner_text()
        home_path = output / f"{name}_home.png"
        page.screenshot(path=str(home_path), full_page=True)
        result["screenshots"]["home"] = str(home_path)

        page.locator(f"#units .unit-row[data-u='{unit}']").click()
        wait_for_unit(page, unit)
        page.locator(card).wait_for()
        page.locator(card).scroll_into_view_if_needed()
        settle(page)
        unit_path = output / f"{name}_unit.png"
        page.screenshot(path=str(unit_path), full_page=True)
        result["screenshots"]["unit"] = str(unit_path)
        page.locator(card).click()
        result["images"] = verify_six_images(page, token, name, report)
        result["checks"]["unit_to_sample_six_images"] = True

        page.locator(f"#main a.u[href='{unit_href}']").click()
        wait_for_unit(page, unit)
        page.locator(card).click()
        verify_six_images(page, token, name, report)
        result["checks"]["sample_unit_sample_round_trip"] = True

        page.locator("#q").fill(token)
        page.locator("#find button").click()
        page.get_by_role("heading", name="Search", exact=True).wait_for()
        page.locator(card).wait_for()
        settle(page)
        result["checks"]["search_exact_sample"] = True
        page.locator(card).click()
        verify_six_images(page, token, name, report)
        result["checks"]["search_to_sample_six_images"] = True
        page.evaluate("window.scrollTo(0, 0)")
        sample_path = output / f"{name}_sample.png"
        page.screenshot(path=str(sample_path), full_page=True)
        result["screenshots"]["sample"] = str(sample_path)
        result["layout"] = page.evaluate(
            "({viewport_width: innerWidth, page_width: document.documentElement.scrollWidth})"
        )
        result["download"] = verify_download(page, output, name, explorer)
        result["checks"]["review_csv_download"] = True
        settle(page)
    except Exception:
        try:
            failed_path = output / f"{name}_failure.png"
            page.screenshot(path=str(failed_path), full_page=True, timeout=5000)
            result["screenshots"]["failure"] = str(failed_path)
        except Exception as screenshot_error:
            result["failure_screenshot_error"] = str(screenshot_error)
        raise
    finally:
        context.close()


def verify(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "kind": "real-chromium-explorer-ui-check",
              "started_at": datetime.now(timezone.utc).isoformat(),
              "run": str(args.run.resolve()), "samples_root": str(args.samples_root.resolve()),
              "sample_token": args.sample_token, "browser_executable": str(args.browser_executable),
              "python_version": sys.version, "playwright_version": None,
              "viewports": {}, "page_errors": [], "request_failures": [],
              "http_errors": [], "http_responses": []}
    server = browser = thread = None
    try:
        report["playwright_version"] = version("playwright")
        from playwright.sync_api import sync_playwright

        require(args.run.is_dir(), "--run must be a run directory")
        require(args.samples_root.is_dir(), "--samples-root must be an image directory")
        require(args.browser_executable.is_file(), "Chromium executable does not exist")
        require(math.isfinite(args.neuron_threshold) and args.neuron_threshold >= 0,
                "--neuron-threshold must be finite and nonnegative")
        explorer = build_explorer(argparse.Namespace(
            run=args.run.resolve(), samples_root=args.samples_root.resolve(),
            nuscenes_meta=None, glossary=None, eta=None, neuron_threshold=args.neuron_threshold,
        ))
        unit, frame = select_unit(explorer, args.sample_token, args.unit)
        report.update({"unit": unit, "effective_eta": explorer.eta,
                       "neuron_threshold": explorer.neuron_threshold,
                       "reference_frame": {key: frame[key] for key in
                                           ("sample_token", "prediction", "tail_score",
                                            "total_active_units")}})
        report["input_hashes"] = {
            name: sha256(args.run / name) for name in ("activations.npz", "metrics.json")
            if (args.run / name).is_file()
        }
        report["verifier_sha256"] = sha256(__file__)
        server = make_server(explorer, "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1},
                                  name="explorer-ui-verification", daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_address[1]}/"
        report["server_url"] = base_url
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(
                    executable_path=str(args.browser_executable), headless=True,
                    args=["--no-sandbox", "--disable-dev-shm-usage"],
                )
                report["chromium_version"] = browser.version
                for name, size in (("desktop", {"width": 1280, "height": 900}),
                                   ("mobile", {"width": 390, "height": 844})):
                    verify_viewport(browser, base_url, explorer, args.sample_token, unit,
                                    name, size, output, report)
            finally:
                if browser is not None:
                    browser.close()
                    browser = None
        successful_downloads = {
            item["download"]["url"] for item in report["viewports"].values()
            if "download" in item
        }
        unexpected_failures = []
        for failure in report["request_failures"]:
            # Chromium may finish an attachment navigation with ERR_ABORTED.
            # Retain that event, exempting it only after the actual CSV was
            # downloaded and parsed successfully for this exact export URL.
            expected = (failure["url"] in successful_downloads
                        and urlparse(failure["url"]).path == "/export/review.csv"
                        and "ERR_ABORTED" in (failure["failure"] or ""))
            failure["expected_successful_download_navigation"] = expected
            if not expected:
                unexpected_failures.append(failure)
        require(not report["page_errors"], "Browser page errors were recorded")
        require(not report["http_errors"], "HTTP responses >= 400 were recorded")
        require(not unexpected_failures, "Unexpected failed browser requests were recorded")
        report["status"] = "passed"
        return report
    except Exception as error:
        report.update({"status": "failed", "error": str(error),
                       "error_type": type(error).__name__, "traceback": traceback.format_exc()})
        raise
    finally:
        if server is not None:
            if thread is not None and thread.is_alive():
                server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5)
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        (output / "ui_check.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    report = verify(args)
    print(f"Browser UI {report['status']}: {args.output_dir.resolve() / 'ui_check.json'}")


if __name__ == "__main__":
    main()
