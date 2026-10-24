"""Archive/transport fixtures only; no real nuScenes download is represented."""

import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from acquire_nuscenes_mini import download, extract_camera_data, main, TABLES, URLS  # noqa: E402


def archive_bytes(extra=()):
    result = io.BytesIO()
    with tarfile.open(fileobj=result, mode="w:gz") as bundle:
        entries = [(f"./v1.0-mini/{name}", b"[]") for name in sorted(TABLES)]
        entries += [("samples/CAM_FRONT/fixture.jpg", b"SYNTHETIC image placeholder"),
                    ("sweeps/CAM_FRONT/unused.jpg", b"not extracted"), *extra]
        for name, value in entries:
            member = tarfile.TarInfo(name)
            member.size = len(value)
            bundle.addfile(member, io.BytesIO(value))
    return result.getvalue()


class Response(io.BytesIO):
    def __init__(self, body, status=200, **headers):
        super().__init__(body)
        self.status, self.headers, self.url = status, headers, URLS[0]


class Opener:
    def __init__(self, response):
        self.response, self.request = response, None

    def open(self, request, timeout):
        self.request = request
        return self.response


def test_actual_download_stream_checksum_and_selective_archive_extract(tmp_path):
    body = archive_bytes()
    archive = tmp_path / "mini.tgz"
    checksum = hashlib.sha256(body).hexdigest()
    opener = Opener(Response(body, **{"Content-Length": str(len(body)), "ETag": '"fixture"'}))
    report = download(archive, URLS[0], checksum, opener=opener)
    assert report["expected_sha256_matched"] and archive.read_bytes() == body
    result = extract_camera_data(archive, tmp_path / "data")
    assert result["camera_files"] == 1 and not result["labels_created"]
    assert not (tmp_path / "data/sweeps").exists()
    assert all((tmp_path / "data/v1.0-mini" / name).is_file() for name in TABLES)


def test_partial_archive_resume_and_range_are_checked(tmp_path):
    body, split = archive_bytes(), 25
    archive = tmp_path / "mini.tgz"
    (tmp_path / "mini.tgz.part").write_bytes(body[:split])
    (tmp_path / "mini.tgz.part.json").write_text(json.dumps({
        "requested_url": URLS[0], "etag": '"fixture"'}))
    opener = Opener(Response(body[split:], 206, **{
        "Content-Range": f"bytes {split}-{len(body)-1}/{len(body)}", "ETag": '"fixture"'}))
    download(archive, URLS[0], opener=opener)
    assert opener.request.get_header("Range") == f"bytes={split}-"
    assert archive.read_bytes() == body


def test_ignored_resume_preserves_existing_partial(tmp_path):
    archive = tmp_path / "mini.tgz"
    partial = tmp_path / "mini.tgz.part"
    partial.write_bytes(b"original partial")
    (tmp_path / "mini.tgz.part.json").write_text(json.dumps({
        "requested_url": URLS[0], "etag": '"fixture"'}))
    with pytest.raises(ValueError, match="resume range"):
        download(archive, URLS[0], opener=Opener(Response(archive_bytes())))
    assert partial.read_bytes() == b"original partial" and not archive.exists()


def test_complete_partial_is_verified_and_published_without_requesting_eof(tmp_path):
    body = archive_bytes()
    archive = tmp_path / "mini.tgz"
    (tmp_path / "mini.tgz.part").write_bytes(body)
    (tmp_path / "mini.tgz.part.json").write_text(json.dumps({
        "requested_url": URLS[0], "expected_bytes": len(body), "etag": '"fixture"'}))

    class NoNetwork:
        def open(self, *args, **kwargs):
            pytest.fail("A completed partial should not make an HTTP416 request")

    report = download(archive, URLS[0], opener=NoNetwork())
    assert report["recovered_completed_partial"] and archive.read_bytes() == body


@pytest.mark.parametrize("kind", ["short", "html", "checksum"])
def test_incomplete_or_wrong_response_is_not_published(tmp_path, kind):
    body = b"<html>sign in</html>" if kind == "html" else archive_bytes()
    extra = 1 if kind == "short" else 0
    opener = Opener(Response(body, **{"Content-Length": str(len(body) + extra)}))
    archive = tmp_path / "mini.tgz"
    with pytest.raises(ValueError):
        download(archive, URLS[0], "0" * 64 if kind == "checksum" else None, opener=opener)
    assert not archive.exists()


@pytest.mark.parametrize("entry", [
    ("../outside", b"bad"), ("/absolute", b"bad"),
    ("samples/CAM_FRONT/fixture.jpg", b"duplicate"),
])
def test_unsafe_archive_never_publishes_a_data_root(tmp_path, entry):
    archive = tmp_path / "mini.tgz"
    archive.write_bytes(archive_bytes([entry]))
    root = tmp_path / "data"
    with pytest.raises(ValueError):
        extract_camera_data(archive, root)
    assert not root.exists() and not (tmp_path / "outside").exists()
    assert not list(tmp_path.glob(".mini-extract-*"))


def test_existing_data_is_preserved_and_local_archive_needs_no_network(tmp_path):
    archive = tmp_path / "mini.tgz"
    archive.write_bytes(archive_bytes())
    data = tmp_path / "data"
    data.mkdir()
    (data / "keep").write_text("existing")
    report = tmp_path / "report.json"
    result = main(["--archive", str(archive), "--data-root", str(data), "--report", str(report)])
    assert result == 2
    assert (data / "keep").read_text() == "existing"
    record = json.loads(report.read_text())
    assert record["status"] == "failed" and not record["data_extracted"]
