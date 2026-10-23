"""Run the browser TAR reader against independent Python-created archive cases."""

import gzip
import io
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node is needed for offline browser JS checks")

NODE_CHECK = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(process.argv[1], 'utf8');
vm.runInThisContext(html.split('<script>')[1].split('</script>')[0]);
const bytes = new Uint8Array(fs.readFileSync(process.argv[2]));
let offset = 0;
let stream = new ReadableStream({pull(controller) {
  if (offset === bytes.length) {controller.close(); return;}
  const end = Math.min(offset + 137, bytes.length);
  controller.enqueue(bytes.slice(offset, end)); offset = end;
}});
if (process.argv[3] === 'gzip') stream = stream.pipeThrough(new DecompressionStream('gzip'));
const paths = [];
DrivingPack.parseTar(stream, async (entry, take) => {
  paths.push(entry.path); await take();
}).then(() => {console.log(JSON.stringify({status: 'passed', paths}));})
  .catch(error => {console.error(error.message); process.exitCode = 2;});
"""


def archive(entries, format=tarfile.PAX_FORMAT):
    result = io.BytesIO()
    with tarfile.open(fileobj=result, mode="w", format=format) as bundle:
        for name, value, link in entries:
            info = tarfile.TarInfo(name)
            if link:
                info.type, info.linkname = tarfile.SYMTYPE, "../outside"
            else:
                info.size = len(value)
            bundle.addfile(info, None if link else io.BytesIO(value))
    return result.getvalue()


def invoke(tmp_path, body, *, gzip_mode=False):
    path = tmp_path / "case.tar"
    path.write_bytes(body)
    return subprocess.run([NODE, "-e", NODE_CHECK, str(ROOT / "tools/prepare_upload.html"),
                           str(path), "gzip" if gzip_mode else "tar"],
                          capture_output=True, text=True, timeout=15)


@pytest.mark.parametrize("format", [tarfile.USTAR_FORMAT, tarfile.PAX_FORMAT, tarfile.GNU_FORMAT])
def test_standard_and_long_names_are_parsed_across_stream_chunks(tmp_path, format):
    # Prefix handling in USTAR; extended path headers in PAX/GNU.
    name = "samples/CAM_FRONT/" + "a" * 90 + ".jpg"
    body = archive([(name, b"SYNTHETIC payload, not an image", False)], format)
    result = invoke(tmp_path, body)
    assert result.returncode == 0, result.stderr
    assert name in result.stdout


@pytest.mark.parametrize("failure", ["path", "symlink", "duplicate", "checksum", "truncated"])
def test_invalid_archives_reject_before_a_handoff_is_published(tmp_path, failure):
    entries = [("file.txt", b"synthetic", False)]
    if failure == "path":
        entries = [("../outside", b"bad", False)]
    elif failure == "symlink":
        entries = [("link", b"", True)]
    elif failure == "duplicate":
        entries *= 2
    body = archive(entries)
    if failure == "checksum":
        body = bytes([body[0] ^ 1]) + body[1:]
    elif failure == "truncated":
        body = body[:1024]  # Member is present, but both end-of-archive blocks are absent.
    result = invoke(tmp_path, body)
    assert result.returncode == 2, result.stdout + result.stderr


def test_gzip_trailer_corruption_is_not_ignored(tmp_path):
    body = bytearray(gzip.compress(archive([("file.txt", b"synthetic", False)])))
    body[-8] ^= 1
    result = invoke(tmp_path, bytes(body), gzip_mode=True)
    assert result.returncode == 2, result.stdout + result.stderr
