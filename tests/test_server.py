"""Tests for the datasety REST API server."""

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
