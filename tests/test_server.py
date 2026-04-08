"""Tests for the datasety REST API server."""

import base64
import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer

import pytest
from PIL import Image

from datasety.server import _DATASETS, _JOBS, _STATE_LOCK, APIHandler


def make_image(path, width, height, color=(255, 0, 0)):
    img = Image.new("RGB", (width, height), color=color)
    img.save(path)


@pytest.fixture(autouse=True)
def reset_state():
    """Clear global state before each test."""
    with _STATE_LOCK:
        _DATASETS.clear()
        _JOBS.clear()


@pytest.fixture
def api_server(tmp_path):
    """Spin up the API server for testing."""
    srv = HTTPServer(("127.0.0.1", 0), APIHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    base_url = f"http://127.0.0.1:{port}"
    yield base_url, tmp_path
    srv.shutdown()


def _request(method, url, data=None):
    req = urllib.request.Request(url, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(data).encode("utf-8")
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def _request_binary(method, url, data, content_type="application/octet-stream"):
    req = urllib.request.Request(url, method=method, data=data)
    req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


class TestDatasetAPI:
    def test_create_and_list_dataset(self, api_server):
        base_url, tmp = api_server
        ds_dir = tmp / "images"
        ds_dir.mkdir()
        make_image(ds_dir / "test.jpg", 100, 100)

        status, data = _request("POST", f"{base_url}/v1/datasets", {"path": str(ds_dir)})
        assert status == 201
        assert "id" in data
        assert data["type"] == "image"
        ds_id = data["id"]

        status, data = _request("GET", f"{base_url}/v1/datasets")
        assert status == 200
        assert len(data["datasets"]) == 1
        assert data["datasets"][0]["id"] == ds_id

    def test_modify_and_delete_dataset(self, api_server):
        base_url, tmp = api_server
        ds_dir = tmp / "audio"
        ds_dir.mkdir()
        (ds_dir / "test.wav").touch()

        _, ds = _request("POST", f"{base_url}/v1/datasets", {"path": str(ds_dir)})
        ds_id = ds["id"]

        status, data = _request("PATCH", f"{base_url}/v1/datasets/{ds_id}", {"name": "New Audio"})
        assert status == 200
        assert data["name"] == "New Audio"

        status, data = _request("DELETE", f"{base_url}/v1/datasets/{ds_id}")
        assert status == 200

        status, _ = _request("GET", f"{base_url}/v1/datasets/{ds_id}")
        assert status == 404

    def test_list_files_and_serve(self, api_server):
        base_url, tmp = api_server
        ds_dir = tmp / "images"
        ds_dir.mkdir()
        img_path = ds_dir / "test.jpg"
        make_image(img_path, 10, 10)

        _, ds = _request("POST", f"{base_url}/v1/datasets", {"path": str(ds_dir)})
        ds_id = ds["id"]

        status, data = _request("GET", f"{base_url}/v1/datasets/{ds_id}/files")
        assert status == 200
        assert len(data["files"]) == 1
        assert data["files"][0]["name"] == "test.jpg"

        req = urllib.request.Request(f"{base_url}/v1/datasets/{ds_id}/files/test.jpg")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            assert resp.headers.get("Content-Type") == "image/jpeg"
            assert len(resp.read()) > 0

    def test_path_traversal_prevention(self, api_server):
        base_url, tmp = api_server
        ds_dir = tmp / "images"
        ds_dir.mkdir()

        _, ds = _request("POST", f"{base_url}/v1/datasets", {"path": str(ds_dir)})
        ds_id = ds["id"]

        req = urllib.request.Request(f"{base_url}/v1/datasets/{ds_id}/files/../etc/passwd")
        try:
            urllib.request.urlopen(req)
            assert False, "Should have raised HTTPError"
        except urllib.error.HTTPError as e:
            assert e.code == 403


class TestFilesCRUD:
    def test_create_file_binary(self, api_server):
        base_url, tmp = api_server
        ds_dir = tmp / "images"
        ds_dir.mkdir()

        _, ds = _request("POST", f"{base_url}/v1/datasets", {"path": str(ds_dir)})
        ds_id = ds["id"]

        img = Image.new("RGB", (10, 10), color=(0, 255, 0))
        import io

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        status, data = _request_binary(
            "POST",
            f"{base_url}/v1/datasets/{ds_id}/files/new_image.png",
            png_bytes,
            content_type="image/png",
        )
        assert status == 201
        assert data["status"] == "created"
        assert data["path"] == "new_image.png"

        req = urllib.request.Request(f"{base_url}/v1/datasets/{ds_id}/files/new_image.png")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            assert len(resp.read()) == len(png_bytes)

    def test_create_file_base64(self, api_server):
        base_url, tmp = api_server
        ds_dir = tmp / "images"
        ds_dir.mkdir()

        _, ds = _request("POST", f"{base_url}/v1/datasets", {"path": str(ds_dir)})
        ds_id = ds["id"]

        img = Image.new("RGB", (10, 10), color=(0, 0, 255))
        import io

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        status, data = _request(
            "POST",
            f"{base_url}/v1/datasets/{ds_id}/files/mask.png",
            {"data": b64},
        )
        assert status == 201
        assert data["status"] == "created"
        assert data["path"] == "mask.png"

    def test_create_file_with_caption(self, api_server):
        base_url, tmp = api_server
        ds_dir = tmp / "images"
        ds_dir.mkdir()

        _, ds = _request("POST", f"{base_url}/v1/datasets", {"path": str(ds_dir)})
        ds_id = ds["id"]

        img = Image.new("RGB", (10, 10))
        import io

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        status, data = _request(
            "POST",
            f"{base_url}/v1/datasets/{ds_id}/files/photo.png",
            {"data": b64, "caption": "a red square"},
        )
        assert status == 201
        assert data["status"] == "created"
        assert "caption_path" in data

        caption_path = ds_dir / "photo.txt"
        assert caption_path.exists()
        assert caption_path.read_text() == "a red square"

    def test_get_file_info(self, api_server):
        base_url, tmp = api_server
        ds_dir = tmp / "images"
        ds_dir.mkdir()
        make_image(ds_dir / "test.jpg", 10, 10)
        (ds_dir / "test.txt").write_text("hello world")

        _, ds = _request("POST", f"{base_url}/v1/datasets", {"path": str(ds_dir)})
        ds_id = ds["id"]

        status, data = _request("GET", f"{base_url}/v1/datasets/{ds_id}/files/test.jpg?info=true")
        assert status == 200
        assert data["caption"] == "hello world"
        assert data["caption_path"] == "test.txt"
        assert data["size_bytes"] > 0

    def test_update_caption_via_put(self, api_server):
        base_url, tmp = api_server
        ds_dir = tmp / "images"
        ds_dir.mkdir()
        make_image(ds_dir / "photo.jpg", 10, 10)

        _, ds = _request("POST", f"{base_url}/v1/datasets", {"path": str(ds_dir)})
        ds_id = ds["id"]

        status, data = _request(
            "PUT",
            f"{base_url}/v1/datasets/{ds_id}/files/photo.jpg",
            {"caption": "updated caption"},
        )
        assert status == 200
        assert data["status"] == "saved"
        assert "caption_path" in data

        assert (ds_dir / "photo.txt").read_text() == "updated caption"

    def test_update_metadata_via_put(self, api_server):
        base_url, tmp = api_server
        ds_dir = tmp / "audio"
        ds_dir.mkdir()
        (ds_dir / "clip.wav").touch()
        (ds_dir / "metadata.csv").write_text("clip|old text\n")

        _, ds = _request("POST", f"{base_url}/v1/datasets", {"path": str(ds_dir)})
        ds_id = ds["id"]

        status, data = _request(
            "PUT",
            f"{base_url}/v1/datasets/{ds_id}/files/clip.wav",
            {"metadata": "new text"},
        )
        assert status == 200
        assert data["status"] == "saved"
        assert "metadata_path" in data

        assert "new text" in (ds_dir / "metadata.csv").read_text()

    def test_update_file_data_via_put(self, api_server):
        base_url, tmp = api_server
        ds_dir = tmp / "images"
        ds_dir.mkdir()
        make_image(ds_dir / "test.jpg", 10, 10)

        _, ds = _request("POST", f"{base_url}/v1/datasets", {"path": str(ds_dir)})
        ds_id = ds["id"]

        img = Image.new("RGB", (20, 20))
        import io

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        status, data = _request(
            "PUT",
            f"{base_url}/v1/datasets/{ds_id}/files/test.jpg",
            {"data": b64, "caption": "replaced image"},
        )
        assert status == 200
        assert data["status"] == "saved"
        assert "caption_path" in data

    def test_put_file_not_found(self, api_server):
        base_url, tmp = api_server
        ds_dir = tmp / "images"
        ds_dir.mkdir()

        _, ds = _request("POST", f"{base_url}/v1/datasets", {"path": str(ds_dir)})
        ds_id = ds["id"]

        status, _ = _request(
            "PUT",
            f"{base_url}/v1/datasets/{ds_id}/files/nonexistent.jpg",
            {"caption": "test"},
        )
        assert status == 404

    def test_delete_file(self, api_server):
        base_url, tmp = api_server
        ds_dir = tmp / "images"
        ds_dir.mkdir()
        make_image(ds_dir / "test.jpg", 10, 10)

        _, ds = _request("POST", f"{base_url}/v1/datasets", {"path": str(ds_dir)})
        ds_id = ds["id"]

        assert (ds_dir / "test.jpg").exists()
        status, data = _request("DELETE", f"{base_url}/v1/datasets/{ds_id}/files/test.jpg")
        assert status == 200
        assert data["status"] == "deleted"
        assert not (ds_dir / "test.jpg").exists()

    def test_delete_file_with_caption(self, api_server):
        base_url, tmp = api_server
        ds_dir = tmp / "images"
        ds_dir.mkdir()
        make_image(ds_dir / "photo.jpg", 10, 10)
        (ds_dir / "photo.txt").write_text("a caption")

        _, ds = _request("POST", f"{base_url}/v1/datasets", {"path": str(ds_dir)})
        ds_id = ds["id"]

        status, data = _request(
            "DELETE", f"{base_url}/v1/datasets/{ds_id}/files/photo.jpg?caption=true"
        )
        assert status == 200
        assert not (ds_dir / "photo.jpg").exists()
        assert not (ds_dir / "photo.txt").exists()

    def test_delete_file_not_found(self, api_server):
        base_url, tmp = api_server
        ds_dir = tmp / "images"
        ds_dir.mkdir()

        _, ds = _request("POST", f"{base_url}/v1/datasets", {"path": str(ds_dir)})
        ds_id = ds["id"]

        status, _ = _request("DELETE", f"{base_url}/v1/datasets/{ds_id}/files/nope.jpg")
        assert status == 404

    def test_create_file_in_subdirectory(self, api_server):
        base_url, tmp = api_server
        ds_dir = tmp / "images"
        ds_dir.mkdir()

        _, ds = _request("POST", f"{base_url}/v1/datasets", {"path": str(ds_dir)})
        ds_id = ds["id"]

        b64 = base64.b64encode(b"fake-png-data").decode()
        status, data = _request(
            "POST",
            f"{base_url}/v1/datasets/{ds_id}/files/masks/photo_mask.png",
            {"data": b64},
        )
        assert status == 201
        assert data["path"] == "masks/photo_mask.png"
        assert (ds_dir / "masks" / "photo_mask.png").exists()

    def test_post_sidecar_only_no_data(self, api_server):
        base_url, tmp = api_server
        ds_dir = tmp / "images"
        ds_dir.mkdir()
        make_image(ds_dir / "photo.jpg", 10, 10)

        _, ds = _request("POST", f"{base_url}/v1/datasets", {"path": str(ds_dir)})
        ds_id = ds["id"]

        status, data = _request(
            "POST",
            f"{base_url}/v1/datasets/{ds_id}/files/photo.jpg",
            {"caption": "sidecar only"},
        )
        assert status == 201
        assert data["status"] == "created"
        assert (ds_dir / "photo.txt").read_text() == "sidecar only"


class TestFileTypeDetection:
    def _register_dataset(self, api_server, ds_dir):
        base_url, _ = api_server
        _, ds = _request("POST", f"{base_url}/v1/datasets", {"path": str(ds_dir)})
        return base_url, ds["id"]

    def test_control_suffix_detected_as_input(self, api_server):
        base_url, tmp = api_server
        ds_dir = tmp / "images"
        ds_dir.mkdir()
        make_image(ds_dir / "photo_control.png", 10, 10)

        base_url, ds_id = self._register_dataset(api_server, ds_dir)
        status, data = _request("GET", f"{base_url}/v1/datasets/{ds_id}/files?group=true")
        assert status == 200
        pair = data["pairs"][0]
        assert pair["input"] == "photo_control.png"
        assert pair["base"] == "photo"

    def test_canny_suffix(self, api_server):
        base_url, tmp = api_server
        ds_dir = tmp / "images"
        ds_dir.mkdir()
        make_image(ds_dir / "photo_input.png", 10, 10)
        make_image(ds_dir / "photo_canny.png", 10, 10)

        base_url, ds_id = self._register_dataset(api_server, ds_dir)
        status, data = _request("GET", f"{base_url}/v1/datasets/{ds_id}/files?group=true")
        assert status == 200
        pair = data["pairs"][0]
        assert pair["canny"] == "photo_canny.png"
        assert pair["input"] == "photo_input.png"

    def test_pose_suffix(self, api_server):
        base_url, tmp = api_server
        ds_dir = tmp / "images"
        ds_dir.mkdir()
        make_image(ds_dir / "photo_pose.png", 10, 10)

        base_url, ds_id = self._register_dataset(api_server, ds_dir)
        status, data = _request("GET", f"{base_url}/v1/datasets/{ds_id}/files?group=true")
        assert status == 200
        pair = data["pairs"][0]
        assert pair["pose"] == "photo_pose.png"

    def test_seg_suffix(self, api_server):
        base_url, tmp = api_server
        ds_dir = tmp / "images"
        ds_dir.mkdir()
        make_image(ds_dir / "photo_seg.png", 10, 10)

        base_url, ds_id = self._register_dataset(api_server, ds_dir)
        status, data = _request("GET", f"{base_url}/v1/datasets/{ds_id}/files?group=true")
        assert status == 200
        pair = data["pairs"][0]
        assert pair["seg"] == "photo_seg.png"

    def test_depth_suffix(self, api_server):
        base_url, tmp = api_server
        ds_dir = tmp / "images"
        ds_dir.mkdir()
        make_image(ds_dir / "photo_depth.png", 10, 10)

        base_url, ds_id = self._register_dataset(api_server, ds_dir)
        status, data = _request("GET", f"{base_url}/v1/datasets/{ds_id}/files?group=true")
        assert status == 200
        pair = data["pairs"][0]
        assert pair["depth"] == "photo_depth.png"

    def test_normal_suffix(self, api_server):
        base_url, tmp = api_server
        ds_dir = tmp / "images"
        ds_dir.mkdir()
        make_image(ds_dir / "photo_normal.png", 10, 10)

        base_url, ds_id = self._register_dataset(api_server, ds_dir)
        status, data = _request("GET", f"{base_url}/v1/datasets/{ds_id}/files?group=true")
        assert status == 200
        pair = data["pairs"][0]
        assert pair["normal"] == "photo_normal.png"

    def test_canny_folder_detection(self, api_server):
        base_url, tmp = api_server
        ds_dir = tmp / "images"
        canny_dir = ds_dir / "canny"
        canny_dir.mkdir(parents=True)
        make_image(canny_dir / "photo.png", 10, 10)

        base_url, ds_id = self._register_dataset(api_server, ds_dir)
        status, data = _request("GET", f"{base_url}/v1/datasets/{ds_id}/files")
        assert status == 200
        found = False
        for f_entry in data["files"]:
            for f in f_entry["files"]:
                if "photo" in f["path"]:
                    assert f["file_type"] == "canny"
                    found = True
        assert found

    def test_all_control_types_grouped(self, api_server):
        base_url, tmp = api_server
        ds_dir = tmp / "images"
        ds_dir.mkdir()
        make_image(ds_dir / "girl_input.png", 10, 10)
        make_image(ds_dir / "girl_target.png", 10, 10)
        make_image(ds_dir / "girl_mask.png", 10, 10)
        make_image(ds_dir / "girl_canny.png", 10, 10)
        make_image(ds_dir / "girl_pose.png", 10, 10)
        make_image(ds_dir / "girl_seg.png", 10, 10)
        make_image(ds_dir / "girl_depth.png", 10, 10)
        make_image(ds_dir / "girl_normal.png", 10, 10)

        base_url, ds_id = self._register_dataset(api_server, ds_dir)
        status, data = _request("GET", f"{base_url}/v1/datasets/{ds_id}/files?group=true")
        assert status == 200
        assert len(data["pairs"]) == 1
        pair = data["pairs"][0]
        assert pair["input"] == "girl_input.png"
        assert pair["target"] == "girl_target.png"
        assert pair["mask"] == "girl_mask.png"
        assert pair["canny"] == "girl_canny.png"
        assert pair["pose"] == "girl_pose.png"
        assert pair["seg"] == "girl_seg.png"
        assert pair["depth"] == "girl_depth.png"
        assert pair["normal"] == "girl_normal.png"

    def test_depth_folder_detection(self, api_server):
        base_url, tmp = api_server
        ds_dir = tmp / "images"
        depth_dir = ds_dir / "depth"
        depth_dir.mkdir(parents=True)
        make_image(depth_dir / "photo.png", 10, 10)

        base_url, ds_id = self._register_dataset(api_server, ds_dir)
        status, data = _request("GET", f"{base_url}/v1/datasets/{ds_id}/files")
        assert status == 200
        found = False
        for f_entry in data["files"]:
            for f in f_entry["files"]:
                if "photo" in f["path"]:
                    assert f["file_type"] == "depth"
                    found = True
        assert found


class TestJobsAPI:
    def test_execute_and_poll_job(self, api_server):
        import time

        base_url, tmp = api_server

        payload = {
            "command": "resize",
            "args": {
                "input": str(tmp),
                "output": str(tmp / "out"),
                "resolution": "512x512",
                "dry-run": True,
            },
        }

        status, data = _request("POST", f"{base_url}/v1/jobs", payload)
        assert status == 202
        assert "id" in data
        job_id = data["id"]

        job_data = None
        for _ in range(20):
            time.sleep(0.2)
            status, job_data = _request("GET", f"{base_url}/v1/jobs/{job_id}")
            if job_data["status"] in ("done", "failed"):
                break

        assert job_data is not None
        assert job_data["status"] == "done"
        assert job_data["exit_code"] == 0

    def test_cancel_job(self, api_server):
        base_url, tmp = api_server

        payload = {
            "command": "resize",
            "args": {
                "input": str(tmp / "input"),
                "output": str(tmp / "output"),
                "resolution": "512x512",
                "dry-run": True,
            },
        }
        _, data = _request("POST", f"{base_url}/v1/jobs", payload)
        job_id = data["id"]

        import time

        time.sleep(0.1)

        status, _ = _request("DELETE", f"{base_url}/v1/jobs/{job_id}")
        assert status in [200, 400]

        _, job_data = _request("GET", f"{base_url}/v1/jobs/{job_id}")
        assert job_data["status"] in ["cancelled", "done", "failed"]
