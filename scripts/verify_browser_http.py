#!/usr/bin/env python3
"""Check the explorer through a real, temporary loopback HTTP server.

Socket creation failures are reported and raised. This checker never replaces
an unavailable HTTP service with an in-memory simulation or a successful skip.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from neuron_explorer import make_server
from tail_activations import CAMERAS


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _checkpoint_info(path: Path | None) -> dict | None:
    if path is None:
        return None
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def verify_http(explorer, output_dir: Path, sample_token: str, unit: int, *,
                checkpoint_path: Path | None = None, timeout: float = 5.0) -> dict:
    """Verify HTML, API, six JPEGs and review export, then always stop the server.

    The caller supplies a sample with all six camera images and a valid unit.
    ``checkpoint_path`` optionally links the checked explorer to its source
    trained checkpoint. ``http_check.json`` is saved on success and failure;
    failures are re-raised after cleanup, including socket-permission failures.
    This checks HTTP responses and the HTML entry point, not browser JS rendering.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"kind": "real-loopback-http-check", "status": "running",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "sample_token": sample_token, "unit": int(unit), "routes": [],
              "server_closed": False, "thread_stopped": False}
    server, thread, failure, traceback = None, None, None, None
    started = time.monotonic()

    try:
        _require(math.isfinite(timeout) and timeout > 0, "timeout must be finite and positive")
        _require(isinstance(sample_token, str) and bool(sample_token), "sample_token is required")
        report["checkpoint"] = _checkpoint_info(checkpoint_path)
        expected_frame = explorer.units_of_frame(sample_token)
        server = make_server(explorer, host="127.0.0.1", port=0)
        thread = threading.Thread(target=server.serve_forever,
                                  kwargs={"poll_interval": 0.05}, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        report["temporary_base_url"] = base_url
        # The check must stay on loopback even in sessions with HTTP proxies.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        def get(path: str, content_type: str, status: int = 200):
            record = {"path": path, "expected_status": status, "passed": False}
            report["routes"].append(record)
            try:
                try:
                    response = opener.open(base_url + path, timeout=timeout)
                except urllib.error.HTTPError as exc:
                    response = exc
                with response:
                    body = response.read()
                    record.update(status=response.code,
                                  content_type=response.headers.get("Content-Type", ""),
                                  bytes=len(body))
                _require(record["status"] == status,
                         f"{path}: expected HTTP {status}, got {record['status']}")
                _require(record["content_type"].lower().startswith(content_type),
                         f"{path}: expected {content_type}, got {record['content_type']}")
                return body, record
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                raise

        def get_json(path: str, expected_type, status: int = 200):
            body, record = get(path, "application/json", status)
            payload = json.loads(body.decode("utf-8"))
            _require(isinstance(payload, expected_type), f"{path}: unexpected JSON structure")
            return payload, record

        body, record = get("/", "text/html")
        _require(b"Neuron Explorer" in body and b"<script" in body,
                 "HTML entry point must include the explorer UI and its script")
        record["passed"] = True

        summary, record = get_json("/api/summary", dict)
        _require(summary.get("frames") == len(explorer.store), "Summary frame count differs")
        _require(summary.get("tail_dim") == explorer.store.tail_dim, "Summary tail_dim differs")
        _require(math.isfinite(float(summary["eta"])) and float(summary["eta"]) >= 0 and
                 float(summary["eta"]) == float(explorer.eta), "Summary eta differs")
        _require(math.isfinite(float(summary["neuron_threshold"])) and
                 float(summary["neuron_threshold"]) >= 0, "Summary neuron threshold is invalid")
        report["actual_eta"] = float(summary["eta"])
        report["actual_neuron_threshold"] = float(summary["neuron_threshold"])
        report["frame_count"] = int(summary["frames"])
        record["passed"] = True

        units, record = get_json("/api/units", list)
        _require(all(isinstance(item, dict) and "unit" in item for item in units),
                 "Unit index lacks unit IDs")
        record["unit_count"] = len(units)
        record["passed"] = True

        unit_payload, record = get_json(f"/api/unit/{int(unit)}", dict)
        _require(unit_payload.get("unit") == int(unit), "Unit lookup returned the wrong unit")
        _require(isinstance(unit_payload.get("frames"), list), "Unit lookup lacks frame list")
        record["frame_count"] = len(unit_payload["frames"])
        record["passed"] = True

        encoded_token = urllib.parse.quote(sample_token, safe="")
        frame, record = get_json(f"/api/sample/{encoded_token}", dict)
        for key in ("sample_token", "prediction", "total_active_units", "tail_score", "cameras"):
            _require(frame.get(key) == expected_frame.get(key), f"Sample API differs for {key}")
        _require(math.isfinite(float(frame["tail_score"])), "Sample score is not finite")
        _require(set(frame["cameras"]) == set(CAMERAS), "Sample must have all six camera images")
        record["passed"] = True

        query = urllib.parse.urlencode({"q": sample_token})
        matches, record = get_json(f"/api/search?{query}", list)
        _require(any(row.get("sample_token") == sample_token for row in matches),
                 "Exact-token search did not return the sample")
        record["match_count"] = len(matches)
        record["passed"] = True

        for camera in CAMERAS:
            body, record = get(f"/img/{encoded_token}/{camera}?w=160", "image/jpeg")
            with Image.open(io.BytesIO(body)) as image:
                _require(image.format == "JPEG", f"{camera}: response is not a JPEG")
                _require(image.width > 0 and image.height > 0, f"{camera}: empty image")
                record["width"], record["height"] = image.size
                image.verify()
            record["passed"] = True

        missing = "__http_check_missing_token__"
        while missing in explorer.store.row_of:
            missing += "_"
        error_payload, record = get_json(f"/api/sample/{missing}", dict, 404)
        _require("error" in error_payload, "Unknown-token response must contain an error")
        record["passed"] = True

        body, record = get("/export/review.csv", "text/csv")
        reader = csv.DictReader(io.StringIO(body.decode("utf-8-sig")))
        columns = set(reader.fieldnames or [])
        required = {"sample_token", "tail_score", "reviewer_label", "review_notes"}
        forbidden = {"label", "labels", "ground_truth", "ground_truth_label", "groundtruth",
                     "is_tail", "reference_label", "target", "y_true"}
        _require(required <= columns, "Review CSV lacks required review columns")
        _require(not columns.intersection(forbidden), "Review CSV exposes ground-truth labels")
        rows = list(reader)
        _require(bool(rows), "Review CSV contains no candidates")
        seen = set()
        for row in rows:
            token = row["sample_token"]
            _require(token in explorer.store.row_of and token not in seen,
                     "Review CSV contains unknown or duplicate sample_token")
            seen.add(token)
            score = float(row["tail_score"])
            _require(math.isfinite(score), "Review CSV has a nonfinite tail_score")
            expected_score = float(explorer.units_of_frame(token)["tail_score"])
            _require(math.isclose(score, expected_score, rel_tol=1e-6, abs_tol=1e-6),
                     "Review CSV score differs from the explorer")
            _require(not row["reviewer_label"] and not row["review_notes"],
                     "Fresh review CSV must leave reviewer fields blank")
        record.update(row_count=len(rows), columns=list(reader.fieldnames), passed=True)
        report["review_row_count"] = len(rows)
        report["status"] = "passed"
    except Exception as exc:
        failure, traceback = exc, exc.__traceback__
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        try:
            if server is not None:
                try:
                    if thread is not None and thread.is_alive():
                        server.shutdown()
                finally:
                    server.server_close()
                    report["server_closed"] = True
            if thread is not None:
                thread.join(timeout=timeout)
                report["thread_stopped"] = not thread.is_alive()
                _require(not thread.is_alive(), "HTTP server thread did not stop")
        except Exception as exc:
            report["status"] = "failed"
            report["cleanup_error"] = {"type": type(exc).__name__, "message": str(exc)}
            if failure is None:
                failure, traceback = exc, exc.__traceback__
        report["duration_seconds"] = round(time.monotonic() - started, 3)
        (output_dir / "http_check.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if failure is not None:
        raise failure.with_traceback(traceback)
    return report
