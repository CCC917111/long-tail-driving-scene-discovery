#!/usr/bin/env python3
"""Acquire the published mini archive through normal network policy, or use a local copy.

No login, terms acceptance, proxy changes, paid resources or labels are automated.
Local hashes identify the received bytes; they do not establish an official checksum.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
from urllib.parse import urlparse
import urllib.request

URLS = (
    "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-mini.tgz",
    "https://www.nuscenes.org/data/v1.0-mini.tgz",
    "https://motional-nuscenes.s3.ap-northeast-1.amazonaws.com/public/v1.0/v1.0-mini.tgz",
)
HOSTS = {urlparse(url).hostname for url in URLS} | {
    "nuscenes.org", "motional-nuscenes.s3.amazonaws.com",
    "motional-nuscenes.s3-ap-northeast-1.amazonaws.com",
}
TABLES = {f"{name}.json" for name in
          ("scene", "sample", "sample_data", "calibrated_sensor", "sensor")}
CAMERAS = {"CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
           "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class OfficialRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlparse(newurl)
        if parsed.scheme != "https" or parsed.hostname not in HOSTS:
            raise ValueError("Download redirected outside the recorded official HTTPS hosts")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(archive: Path, url: str, expected_sha256: str | None = None,
             *, opener=None, timeout: float = 30) -> dict:
    if url not in URLS:
        raise ValueError("Choose one of the published official mini URL choices")
    if expected_sha256 and not re.fullmatch(r"[a-fA-F0-9]{64}", expected_sha256):
        raise ValueError("expected-sha256 must be a 64-character hex digest")
    archive = Path(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    report = {"requested_url": url, "archive": str(archive.resolve()),
              "expected_sha256_supplied": expected_sha256 is not None}
    if not archive.exists():
        partial = archive.with_name(archive.name + ".part")
        metadata = archive.with_name(archive.name + ".part.json")
        offset, previous = 0, {}
        if partial.exists():
            if not metadata.is_file():
                raise ValueError("Untracked partial archive; preserve it and choose a fresh path")
            previous = json.loads(metadata.read_text())
            if previous.get("requested_url") != url:
                raise ValueError("Partial archive belongs to a different source URL")
            offset = partial.stat().st_size
            if offset and offset == previous.get("expected_bytes"):
                # A process can stop after the final byte but before the rename.
                # Validate the complete gzip locally instead of requesting bytes=EOF-.
                with gzip.open(partial, "rb") as content:
                    for _ in iter(lambda: content.read(1024 * 1024), b""):
                        pass
                digest = sha256(partial)
                if expected_sha256 and digest != expected_sha256.lower():
                    raise ValueError("Completed partial archive checksum mismatch")
                os.replace(partial, archive)
                metadata.rename(archive.with_name(archive.name + ".download.json"))
                retained = {key: value for key, value in previous.items()
                            if key != "requested_url"}
                return dict(report, **retained,
                            downloaded_this_invocation=False, recovered_completed_partial=True,
                            sha256=digest, bytes=offset,
                            expected_sha256_matched=bool(expected_sha256),
                            official_checksum_independently_verified=False)
        headers = {"User-Agent": "Driving-mini-acquisition/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
            validator = previous.get("etag") or previous.get("last_modified")
            if not validator:
                raise ValueError("Cannot safely resume without a recorded server validator")
            headers["If-Range"] = validator
        opener = opener or urllib.request.build_opener(OfficialRedirects())
        request = urllib.request.Request(url, headers=headers)
        with opener.open(request, timeout=timeout) as response:
            status = response.status
            etag = response.headers.get("ETag")
            modified = response.headers.get("Last-Modified")
            length = response.headers.get("Content-Length")
            expected = int(length) if length is not None else None
            if offset:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                     response.headers.get("Content-Range", ""))
                if status != 206 or not match or int(match[1]) != offset:
                    raise ValueError("Server did not confirm the requested resume range; "
                                     "preserve the partial archive and choose a fresh path")
                if previous.get("etag") and etag != previous["etag"]:
                    raise ValueError("Server ETag changed; refusing to append different bytes")
                expected = int(match[3])
            elif status != 200:
                raise ValueError(f"Expected a full archive response, received HTTP {status}")
            if expected is not None and expected - offset > shutil.disk_usage(archive.parent).free:
                raise ValueError("Insufficient free disk for the remaining archive download")
            transfer = {"requested_url": url, "final_url": response.url,
                        "etag": etag, "last_modified": modified, "expected_bytes": expected}
            write_json(metadata, transfer)
            with partial.open("ab" if offset else "wb") as handle:
                received = offset
                first = True
                next_progress = offset + 64 * 1024 * 1024
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    if first and not offset and block[:2] != b"\x1f\x8b":
                        raise ValueError("Response is not a gzip archive; check the official link")
                    first = False
                    handle.write(block)
                    received += len(block)
                    if received >= next_progress:
                        print(f"[download] {received / 2**20:.0f} MiB received", flush=True)
                        next_progress += 64 * 1024 * 1024
                handle.flush()
                os.fsync(handle.fileno())
            if expected is not None and received != expected:
                raise ValueError(f"Incomplete archive: received {received}, expected {expected}")
            if received == 0:
                raise ValueError("Empty archive response")
        digest = sha256(partial)
        if expected_sha256 and digest != expected_sha256.lower():
            raise ValueError("Archive checksum mismatch; unverified partial file preserved")
        os.replace(partial, archive)
        metadata.rename(archive.with_name(archive.name + ".download.json"))
        report.update(transfer, downloaded_this_invocation=True)
    else:
        digest = sha256(archive)
        if expected_sha256 and digest != expected_sha256.lower():
            raise ValueError("Existing archive checksum mismatch")
        report.update(downloaded_this_invocation=False, source_status="local file supplied")
    report.update(sha256=digest, bytes=archive.stat().st_size,
                  expected_sha256_matched=bool(expected_sha256),
                  official_checksum_independently_verified=False)
    return report


def extract_camera_data(archive: Path, data_root: Path, *, max_extract_bytes=12 * 2**30) -> dict:
    """Stage regular files only; publish complete metadata/keyframe cameras atomically."""
    archive, data_root = Path(archive), Path(data_root)
    if data_root.exists() and (not data_root.is_dir() or any(data_root.iterdir())):
        raise ValueError("Data root must be absent or empty; existing data is preserved")
    data_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".mini-extract-", dir=data_root.parent))
    total, names, camera_files = 0, set(), 0
    try:
        with tarfile.open(archive, mode="r|gz") as bundle:
            for member in bundle:
                parts = PurePosixPath(member.name).parts
                if (member.name.startswith("/") or ".." in parts or "\\" in member.name
                        or member.issym() or member.islnk()):
                    raise ValueError(f"Unsafe archive member: {member.name}")
                if member.isdir():
                    continue
                allowed = ((len(parts) == 2 and parts[0] == "v1.0-mini"
                            and parts[1].endswith(".json"))
                           or (len(parts) == 3 and parts[0] == "samples"
                               and parts[1] in CAMERAS)
                           or (len(parts) == 1 and parts[0] in {
                               "LICENSE", "LICENSE.txt", "README", "README.md", ".v1.0-mini.txt"}))
                if not allowed:
                    continue
                canonical = "/".join(parts)
                if not member.isfile() or canonical in names:
                    raise ValueError(f"Unsupported or duplicate archive member: {member.name}")
                total += member.size
                if total > max_extract_bytes:
                    raise ValueError("Selected archive contents exceed the extraction limit")
                if member.size > shutil.disk_usage(stage).free:
                    raise ValueError("Insufficient free disk to extract the next file")
                destination = stage.joinpath(*parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, destination.open("xb") as output:
                    shutil.copyfileobj(source, output)
                if destination.stat().st_size != member.size:
                    raise ValueError(f"Incomplete extracted member: {member.name}")
                names.add(canonical)
                camera_files += int(parts[0] == "samples")
        if not {f"v1.0-mini/{name}" for name in TABLES} <= names or not camera_files:
            raise ValueError("Archive lacks required mini metadata or camera keyframes")
        record = {"schema": "mini-camera-acquisition-v1", "archive_sha256": sha256(archive),
                  "extracted_files": len(names), "camera_files": camera_files,
                  "extracted_bytes": total, "sweeps_extracted": False,
                  "labels_created": False, "images_decoded": False}
        write_json(stage / "acquisition.json", record)
        os.replace(stage, data_root)
        return record
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("dataset/downloads/v1.0-mini.tgz"))
    parser.add_argument("--data-root", type=Path, default=Path("dataset/nuscenes"))
    parser.add_argument("--url", choices=URLS, default=URLS[0])
    parser.add_argument("--expected-sha256")
    parser.add_argument("--report", type=Path, default=Path("output/real-mini/acquisition.json"))
    args = parser.parse_args(argv)
    report = {"recorded_at": datetime.now(timezone.utc).isoformat(), "status": "started",
              "data_extracted": False, "labels_created": False}
    write_json(args.report, report)
    try:
        report["download"] = download(args.archive, args.url, args.expected_sha256)
        report["extraction"] = extract_camera_data(args.archive, args.data_root)
        report.update(status="completed", data_extracted=True)
    except (OSError, ValueError, EOFError, http.client.HTTPException, tarfile.TarError) as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(args.report, report)
        print(json.dumps(report, indent=2))
        return 2
    except KeyboardInterrupt:
        report.update(status="interrupted", error="Download/extraction interrupted")
        write_json(args.report, report)
        return 130
    write_json(args.report, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
