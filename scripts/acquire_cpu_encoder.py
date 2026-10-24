#!/usr/bin/env python3
"""Download the pinned official CPU encoder; preserve existing model files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import urlparse
import urllib.request

from cpu_image_encoder import WEIGHTS_NAME, WEIGHTS_SHA256, WEIGHTS_URL, verify_weights
from extraction_contract import sha256


class OfficialRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlparse(newurl)
        if parsed.scheme != "https" or parsed.hostname != "download.pytorch.org":
            raise ValueError("Encoder download redirected outside official HTTPS host")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def acquire(output: Path, *, opener=None) -> dict:
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    downloaded = False
    if output.exists():
        verify_weights(output)
    else:
        opener = opener or urllib.request.build_opener(OfficialRedirects())
        temporary = None
        try:
            request = urllib.request.Request(WEIGHTS_URL,
                                             headers={"User-Agent": "Driving-CPU-reference/1.0"})
            with opener.open(request, timeout=30) as response:
                if response.status != 200:
                    raise ValueError(f"Expected full encoder response; got HTTP {response.status}")
                with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".resnet18-",
                                                 delete=False) as handle:
                    temporary = Path(handle.name)
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        handle.write(block)
                    handle.flush()
                    os.fsync(handle.fileno())
            if sha256(temporary) != WEIGHTS_SHA256:
                raise ValueError("Official encoder SHA256 mismatch; downloaded bytes rejected")
            # Hard-link publication refuses to replace a file created by another process.
            os.link(temporary, output)
            downloaded = True
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return {"status": "verified", "official_url": WEIGHTS_URL, "path": str(output),
            "bytes": output.stat().st_size, "sha256": verify_weights(output),
            "downloaded_this_invocation": downloaded, "feature_protocol": "cpu-resnet18"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/resnet18") / WEIGHTS_NAME)
    args = parser.parse_args(argv)
    print(json.dumps(acquire(args.output), indent=2))


if __name__ == "__main__":
    main()
