"""Tests for the align server."""

import json
import threading
import time
import urllib.request
import urllib.error

import pytest
from PIL import Image

from datasety.align_server import _build_pairs, _make_handler, cmd_align_server
from http.server import HTTPServer


def make_image(path, width, height):
    img = Image.new("RGB", (width, height), color=(255, 0, 0))
    img.save(path)


@pytest.fixture
def setup_dirs(tmp_path):
    target = tmp_path / "target"
    control = tmp_path / "control"
    target.mkdir()
    control.mkdir()
    return target, control


@pytest.fixture
def populated_dirs(setup_dirs):
    target, control = setup_dirs
    for i in range(3):
        make_image(target / f"{i:03d}.jpg", 512, 512)
        make_image(control / f"{i:03d}.jpg", 512, 512)
    # Add a caption for pair 001
    (target / "001.txt").write_text("target caption")
    (control / "001.txt").write_text("control caption")
    return target, control


@pytest.fixture
def server(populated_dirs):
    """Start a test server on a random port."""
    target, control = populated_dirs
    handler = _make_handler(target, control, recursive=False)
    srv = HTTPServer(("127.0.0.1", 0), handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}", target, control
    srv.shutdown()


class TestBuildPairs:
    def test_matched_pairs(self, populated_dirs):
        target, control = populated_dirs
        pairs = _build_pairs(target, control, recursive=False)
        assert len(pairs) == 3
        stems = [p["stem"] for p in pairs]
        assert "000" in stems
        assert "001" in stems

    def test_caption_detection(self, populated_dirs):
        target, control = populated_dirs
        pairs = _build_pairs(target, control, recursive=False)
        by_stem = {p["stem"]: p for p in pairs}
        assert by_stem["001"]["has_target_caption"] is True
        assert by_stem["001"]["has_control_caption"] is True
        assert by_stem["000"]["has_target_caption"] is False

    def test_no_match(self, setup_dirs):
        target, control = setup_dirs
        make_image(target / "a.jpg", 100, 100)
        make_image(control / "b.jpg", 100, 100)
        pairs = _build_pairs(target, control, recursive=False)
        assert len(pairs) == 0


class TestServerAPI:
    def _get(self, base, path):
        req = urllib.request.Request(base + path)
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())

    def _get_raw(self, base, path):
        req = urllib.request.Request(base + path)
        with urllib.request.urlopen(req) as resp:
            return resp.status, resp.read(), resp.headers.get("Content-Type")

    def _post(self, base, path, data):
        body = json.dumps(data).encode()
        req = urllib.request.Request(base + path, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())

    def _delete(self, base, path):
        req = urllib.request.Request(base + path, method="DELETE")
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())

    def test_get_pairs(self, server):
        base, target, control = server
        status, data = self._get(base, "/api/pairs")
        assert status == 200
        assert len(data) == 3

    def test_serve_html(self, server):
        base, target, control = server
        status, body, ctype = self._get_raw(base, "/")
        assert status == 200
        assert "text/html" in ctype

    def test_serve_image(self, server):
        base, target, control = server
        img_path = str(target / "000.jpg")
        status, body, ctype = self._get_raw(
            base, "/api/image?path=" + urllib.request.quote(img_path)
        )
        assert status == 200
        assert "image" in ctype

    def test_serve_image_access_denied(self, server):
        base, target, control = server
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            self._get_raw(base, "/api/image?path=/etc/passwd")
        assert exc_info.value.code == 403

    def test_read_caption(self, server):
        base, target, control = server
        cap_path = str(target / "001.txt")
        status, data = self._get(
            base, "/api/caption?path=" + urllib.request.quote(cap_path)
        )
        assert status == 200
        assert data["text"] == "target caption"
        assert data["exists"] is True

    def test_read_missing_caption(self, server):
        base, target, control = server
        cap_path = str(target / "000.txt")
        status, data = self._get(
            base, "/api/caption?path=" + urllib.request.quote(cap_path)
        )
        assert status == 200
        assert data["exists"] is False

    def test_save_caption(self, server):
        base, target, control = server
        cap_path = str(target / "002.txt")
        status, data = self._post(
            base, "/api/caption", {"path": cap_path, "text": "new caption"}
        )
        assert status == 200
        assert data["ok"] is True
        assert (target / "002.txt").read_text() == "new caption"

    def test_save_caption_non_txt_denied(self, server):
        base, target, control = server
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            self._post(
                base, "/api/caption",
                {"path": str(target / "002.jpg"), "text": "hack"},
            )
        assert exc_info.value.code == 403

    def test_delete_pair(self, server):
        base, target, control = server
        assert (target / "001.jpg").exists()
        assert (control / "001.jpg").exists()
        assert (target / "001.txt").exists()
        assert (control / "001.txt").exists()

        status, data = self._delete(base, "/api/pair?stem=001")
        assert status == 200
        assert len(data["deleted"]) == 4  # 2 images + 2 captions

        assert not (target / "001.jpg").exists()
        assert not (control / "001.jpg").exists()
        assert not (target / "001.txt").exists()
        assert not (control / "001.txt").exists()

    def test_pairs_after_delete(self, server):
        base, target, control = server
        self._delete(base, "/api/pair?stem=000")
        status, data = self._get(base, "/api/pairs")
        stems = [p["stem"] for p in data]
        assert "000" not in stems
