"""Verify trusted-weight publication without network requests or large downloads."""

import io
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import acquire_cpu_encoder as acquisition  # noqa: E402


def test_bad_download_never_publishes_a_model(tmp_path):
    class Response(io.BytesIO):
        status = 200
    opener = SimpleNamespace(open=lambda *args, **kwargs: Response(b"not official weights"))
    destination = tmp_path / "weights.pth"
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        acquisition.acquire(destination, opener=opener)
    assert not destination.exists()
    assert not list(tmp_path.glob(".resnet18-*"))


def test_existing_bad_weights_are_preserved_without_network(tmp_path):
    destination = tmp_path / "weights.pth"
    destination.write_bytes(b"preserve these bytes")

    def forbidden(*args, **kwargs):
        pytest.fail("Existing file must be validated before any download")

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        acquisition.acquire(destination, opener=SimpleNamespace(open=forbidden))
    assert destination.read_bytes() == b"preserve these bytes"


def test_redirect_outside_official_host_rejected():
    with pytest.raises(ValueError, match="outside official"):
        acquisition.OfficialRedirects().redirect_request(
            None, None, 302, "redirect", {}, "https://example.com/model.pth")
