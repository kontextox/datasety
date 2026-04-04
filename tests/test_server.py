"""Tests for the universal datasety server."""

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import HTTPServer

import pytest
from PIL import Image

from datasety.server import (
    COMMAND_SCHEMAS,
    _args_to_argv,
    _build_pairs,
    _compute_stats,
    _make_handler,
    _scan,
)


def make_image(path, width, height, color=(255, 0, 0)):
    img = Image.new("RGB", (width, height), color=color)
    img.save(path)


@pytest.fixture
def dataset_dir(tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    make_image(d / "img001.jpg", 512, 512, color=(255, 0, 0))
    make_image(d / "img002.jpg", 512, 512, color=(0, 255, 0))
    make_image(d / "img003.jpg", 1024, 768, color=(0, 0, 255))
    make_image(d / "portrait.jpg", 512, 1024, color=(128, 128, 128))
    make_image(d / "landscape.jpg", 1024, 512, color=(64, 64, 64))
    (d / "img001.txt").write_text("A test caption")
    return d


@pytest.fixture
def pairs_dirs(tmp_path):
    target = tmp_path / "target"
    control = tmp_path / "control"
    target.mkdir()
    control.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        make_image(target / name, 512, 512)
        make_image(control / name, 512, 512)
    # orphan target (no matching control)
    make_image(target / "orphan.jpg", 256, 256)
    (target / "a.txt").write_text("target caption a")
    (control / "a.txt").write_text("control caption a")
    return target, control


@pytest.fixture
def server(dataset_dir):
    handler = _make_handler(dataset_dir, control_dir=None, recursive=False, compute_hashes=True)
    srv = HTTPServer(("127.0.0.1", 0), handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}", dataset_dir
    srv.shutdown()


@pytest.fixture
def pairs_server(pairs_dirs):
    target, control = pairs_dirs
    handler = _make_handler(target, control_dir=control, recursive=False, compute_hashes=False)
    srv = HTTPServer(("127.0.0.1", 0), handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}", target, control
    srv.shutdown()


# ── Helper functions ──────────────────────────────────────────────────────────


def _get(base, path):
    with urllib.request.urlopen(base + path) as resp:
        return resp.status, json.loads(resp.read())


def _get_raw(base, path):
    with urllib.request.urlopen(base + path) as resp:
        return resp.status, resp.read(), resp.headers.get("Content-Type")


def _post(base, path, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(base + path, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read())


# ── Unit tests ────────────────────────────────────────────────────────────────


class TestScan:
    def test_scan_basic(self, dataset_dir):
        result = _scan(dataset_dir, recursive=False, compute_hashes=False)
        assert result["stats"]["total"] == 5

    def test_scan_captions(self, dataset_dir):
        result = _scan(dataset_dir, recursive=False, compute_hashes=False)
        assert result["stats"]["captions_found"] == 1
        assert result["stats"]["captions_missing"] == 4

    def test_scan_orientations(self, dataset_dir):
        result = _scan(dataset_dir, recursive=False, compute_hashes=False)
        o = result["stats"]["orientations"]
        assert o.get("square", 0) == 3  # 512x512(×2), 1024x768 (ratio 1.33)
        assert o.get("landscape", 0) == 1
        assert o.get("portrait", 0) == 1

    def test_scan_hashes(self, dataset_dir):
        result = _scan(dataset_dir, recursive=False, compute_hashes=True)
        assert len(result["hashes"]) > 0


class TestComputeStats:
    def test_basic(self, dataset_dir):
        result = _scan(dataset_dir, recursive=False, compute_hashes=False)
        stats = _compute_stats(result["images"])
        assert stats["total"] == 5
        assert stats["captions_found"] == 1

    def test_after_delete(self, dataset_dir):
        result = _scan(dataset_dir, recursive=False, compute_hashes=False)
        images = [i for i in result["images"] if i["name"] != "img001.jpg"]
        stats = _compute_stats(images)
        assert stats["total"] == 4
        assert stats["captions_found"] == 0


class TestBuildPairs:
    def test_matched_pairs(self, pairs_dirs):
        target, control = pairs_dirs
        pairs = _build_pairs(target, control, recursive=False)
        # a, b, c should match; orphan has no match
        assert len(pairs) == 3

    def test_pair_structure(self, pairs_dirs):
        target, control = pairs_dirs
        pairs = _build_pairs(target, control, recursive=False)
        pair_a = next(p for p in pairs if p["stem"] == "a")
        assert "target" in pair_a
        assert "control" in pair_a
        assert pair_a["has_target_caption"] is True
        assert pair_a["has_control_caption"] is True


# ── Server API tests ──────────────────────────────────────────────────────────


class TestGalleryAPI:
    def test_html_root(self, server):
        base, _ = server
        status, body, ctype = _get_raw(base, "/")
        assert status == 200
        assert "text/html" in ctype

    def test_stats(self, server):
        base, _ = server
        status, data = _get(base, "/api/stats")
        assert status == 200
        assert data["total"] == 5

    def test_images(self, server):
        base, _ = server
        status, data = _get(base, "/api/images")
        assert status == 200
        assert len(data["images"]) == 5
        assert "total" in data

    def test_mode_no_pairs(self, server):
        base, _ = server
        status, data = _get(base, "/api/mode")
        assert status == 200
        assert data["has_pairs"] is False

    def test_filter_orientation(self, server):
        base, _ = server
        status, data = _get(base, "/api/images?orientation=portrait")
        assert status == 200
        assert len(data["images"]) == 1

    def test_filter_caption(self, server):
        base, _ = server
        status, data = _get(base, "/api/images?has_caption=true")
        assert status == 200
        assert len(data["images"]) == 1

    def test_filter_format(self, server):
        base, _ = server
        status, data = _get(base, "/api/images?format=jpg")
        assert status == 200
        assert len(data["images"]) == 5

    def test_search_by_name(self, server):
        base, _ = server
        status, data = _get(base, "/api/images?search=portrait")
        assert status == 200
        assert len(data["images"]) == 1
        assert data["images"][0]["name"] == "portrait.jpg"

    def test_serve_image(self, server):
        base, d = server
        p = urllib.parse.quote(str(d / "img001.jpg"))
        status, body, ctype = _get_raw(base, f"/api/image?path={p}")
        assert status == 200
        assert "image" in ctype

    def test_serve_image_access_denied(self, server):
        base, _ = server
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get_raw(base, "/api/image?path=/etc/passwd")
        assert exc.value.code == 403

    def test_thumbnail(self, server):
        base, d = server
        p = urllib.parse.quote(str(d / "img001.jpg"))
        status, body, ctype = _get_raw(base, f"/api/thumbnail?path={p}")
        assert status == 200
        assert "image" in ctype

    def test_image_info(self, server):
        base, d = server
        p = urllib.parse.quote(str(d / "img001.jpg"))
        status, data = _get(base, f"/api/image/info?path={p}")
        assert status == 200
        assert data["width"] == 512
        assert data["height"] == 512
        assert data["has_caption"] is True

    def test_read_caption_exists(self, server):
        base, d = server
        p = urllib.parse.quote(str(d / "img001.txt"))
        status, data = _get(base, f"/api/caption?path={p}")
        assert status == 200
        assert data["exists"] is True
        assert data["text"] == "A test caption"

    def test_read_caption_missing(self, server):
        base, d = server
        p = urllib.parse.quote(str(d / "img002.txt"))
        status, data = _get(base, f"/api/caption?path={p}")
        assert status == 200
        assert data["exists"] is False

    def test_save_caption(self, server):
        base, d = server
        status, data = _post(base, "/api/caption", {"path": str(d / "img002.txt"), "text": "hello"})
        assert status == 200
        assert data["ok"] is True
        assert (d / "img002.txt").read_text() == "hello"

    def test_save_caption_non_txt_denied(self, server):
        base, d = server
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(base, "/api/caption", {"path": str(d / "img002.jpg"), "text": "hack"})
        assert exc.value.code == 403

    def test_delete_image(self, server):
        base, d = server
        assert (d / "img003.jpg").exists()
        status, data = _post(base, "/api/delete", {"paths": [str(d / "img003.jpg")]})
        assert status == 200
        assert data["count"] >= 1
        assert not (d / "img003.jpg").exists()

    def test_delete_updates_stats(self, server):
        base, d = server
        _, before = _get(base, "/api/stats")
        _post(base, "/api/delete", {"paths": [str(d / "img002.jpg")]})
        _, after = _get(base, "/api/stats")
        assert after["total"] == before["total"] - 1

    def test_duplicates_endpoint(self, server):
        base, _ = server
        status, data = _get(base, "/api/duplicates")
        assert status == 200
        assert "groups" in data
        assert "total_groups" in data


class TestCommandSchemas:
    def test_all_commands_present(self):
        expected = {
            "resize",
            "caption",
            "align",
            "inspect",
            "filter",
            "degrade",
            "shuffle",
            "mask",
            "synthetic",
            "train",
            "audio",
        }
        assert expected == set(COMMAND_SCHEMAS.keys())

    def test_each_has_description_and_args(self):
        for key, schema in COMMAND_SCHEMAS.items():
            assert "description" in schema, key
            assert "args" in schema, key
            assert isinstance(schema["args"], list), key

    def test_args_to_argv_str(self):
        argv = _args_to_argv("resize", {"input": "/in", "output": "/out", "recursive": False})
        assert "--input" in argv
        assert "/in" in argv
        assert "--recursive" not in argv

    def test_args_to_argv_bool(self):
        argv = _args_to_argv("resize", {"recursive": True, "dry_run": True})
        assert "--recursive" in argv
        assert "--dry-run" in argv

    def test_args_to_argv_skip_empty(self):
        argv = _args_to_argv("resize", {"input": "", "resolution": ""})
        assert argv == []


class TestRunAPI:
    def test_get_commands(self, server):
        base, _ = server
        status, data = _get(base, "/api/commands")
        assert status == 200
        assert "resize" in data
        assert "train" in data

    def test_get_jobs_empty(self, server):
        base, _ = server
        status, data = _get(base, "/api/jobs")
        assert status == 200
        assert "jobs" in data

    def test_run_inspect(self, server):
        base, d = server
        status, data = _post(
            base,
            "/api/run",
            {
                "command": "inspect",
                "args": {"input": str(d)},
            },
        )
        assert status == 200
        assert "job_id" in data

    def test_run_job_polling(self, server):
        import time as _time

        base, d = server
        _, r = _post(base, "/api/run", {"command": "inspect", "args": {"input": str(d)}})
        jid = r["job_id"]
        # poll until done (max 10s)
        for _ in range(20):
            _time.sleep(0.5)
            _, j = _get(base, f"/api/job?id={jid}")
            if j["status"] in ("done", "failed"):
                break
        assert j["status"] == "done"
        assert len(j["output"]) > 0

    def test_run_unknown_command(self, server):
        base, _ = server
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(base, "/api/run", {"command": "notacommand", "args": {}})
        assert exc.value.code == 400

    def test_get_job_not_found(self, server):
        base, _ = server
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(base, "/api/job?id=doesnotexist")
        assert exc.value.code == 404


class TestPairsAPI:
    def test_mode_has_pairs(self, pairs_server):
        base, target, control = pairs_server
        status, data = _get(base, "/api/mode")
        assert status == 200
        assert data["has_pairs"] is True
        assert data["pairs_count"] == 3

    def test_get_pairs(self, pairs_server):
        base, target, control = pairs_server
        status, data = _get(base, "/api/pairs")
        assert status == 200
        assert isinstance(data, list)
        assert len(data) == 3

    def test_pair_structure(self, pairs_server):
        base, target, control = pairs_server
        _, pairs = _get(base, "/api/pairs")
        pair_a = next(p for p in pairs if p["stem"] == "a")
        assert pair_a["has_target_caption"] is True
        assert pair_a["has_control_caption"] is True

    def test_delete_pair(self, pairs_server):
        base, target, control = pairs_server
        assert (target / "b.jpg").exists()
        assert (control / "b.jpg").exists()

        status, data = _post(base, "/api/pair/delete", {"stem": "b"})
        assert status == 200
        assert data["count"] >= 2
        assert not (target / "b.jpg").exists()
        assert not (control / "b.jpg").exists()

    def test_delete_pair_updates_gallery(self, pairs_server):
        base, target, control = pairs_server
        _, before = _get(base, "/api/stats")
        _post(base, "/api/pair/delete", {"stem": "c"})
        _, after = _get(base, "/api/stats")
        assert after["total"] == before["total"] - 1

    def test_delete_pair_removes_captions(self, pairs_server):
        base, target, control = pairs_server
        (target / "b.txt").write_text("target b")
        (control / "b.txt").write_text("control b")
        _post(base, "/api/pair/delete", {"stem": "b"})
        assert not (target / "b.txt").exists()
        assert not (control / "b.txt").exists()
