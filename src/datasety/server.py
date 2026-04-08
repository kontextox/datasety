"""REST API Server for datasety dataset management and job execution."""

import argparse
import csv
import json
import mimetypes
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}

_DATASETS = {}
_JOBS = {}
_STATE_LOCK = threading.Lock()


def _get_command_schemas() -> dict:
    """Dynamically get all command schemas from the CLI."""
    from datasety import (
        align,
        caption,
        degrade,
        filter,
        mask,
        resize,
        shuffle,
        synthetic,
    )
    from datasety import server as server_mod

    modules = [resize, caption, align, shuffle, synthetic, mask, filter, degrade, server_mod]

    try:
        from datasety import character

        modules.append(character)
    except ImportError:
        pass
    try:
        from datasety import workflow

        modules.append(workflow)
    except ImportError:
        pass
    try:
        from datasety import sweep

        modules.append(sweep)
    except ImportError:
        pass
    try:
        from datasety import train

        modules.append(train)
    except ImportError:
        pass
    try:
        from datasety import audio

        modules.append(audio)
    except ImportError:
        pass
    try:
        from datasety import upload

        modules.append(upload)
    except ImportError:
        pass

    commands = {}
    for mod in modules:
        if not hasattr(mod, "register_parser"):
            continue
        parser = argparse.ArgumentParser(prog="datasety")
        subparsers = parser.add_subparsers(dest="command", required=True)
        mod.register_parser(subparsers)

        for subparser in subparsers.choices.values():
            cmd_name = subparser.prog.split(" ")[-1]
            params = []
            for action in subparser._actions:
                if action.dest == "command" or action.dest == "help":
                    continue
                param_info = {
                    "name": action.dest,
                    "help": action.help or "",
                    "required": action.required,
                }
                if action.choices:
                    param_info["choices"] = action.choices
                if action.default and action.default != argparse.SUPPRESS:
                    param_info["default"] = (
                        str(action.default)
                        if not isinstance(action.default, bool)
                        else action.default
                    )
                if action.nargs in (None, "+", "*", "?"):
                    param_info["nargs"] = action.nargs
                param_type = action.type
                if param_type in (int, float, str):
                    param_info["type"] = param_type.__name__
                elif callable(param_type):
                    param_info["type"] = getattr(
                        param_type, "__name__", str(param_type).__class__.__name__
                    )

                params.append(param_info)

            commands[cmd_name] = {
                "description": subparser.description or "",
                "params": params,
            }

    return commands


def _detect_type(path: Path) -> str:
    """Detect dataset type from file extensions."""
    path = Path(path)
    if not path.is_dir():
        return "generic"

    exts = {p.suffix.lower() for p in path.rglob("*") if p.is_file()}

    if exts & _AUDIO_EXTENSIONS:
        return "audio"
    if exts & _IMAGE_EXTENSIONS:
        return "image"
    if exts & _VIDEO_EXTENSIONS:
        return "video"

    return "generic"


def _args_to_argv(command: str, args_dict: dict) -> list[str]:
    """Dynamically convert a JSON dictionary to datasety CLI arguments."""
    argv = [sys.executable, "-m", "datasety", command]
    for key, value in args_dict.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        elif isinstance(value, list):
            for item in value:
                argv.extend([flag, str(item)])
        else:
            argv.extend([flag, str(value)])
    return argv


class APIHandler(BaseHTTPRequestHandler):
    def send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status, message):
        self.send_json({"error": message}, status=status)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, PATCH, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def parse_body(self):
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(content_length))
        except json.JSONDecodeError:
            return None

    def _resolve_dataset_file(self, path, require_dataset=True):
        """Parse a /v1/datasets/<id>/files/<filepath> URL.

        Returns (ds_id, ds, ds_path, target_path) or sends an error and
        returns None.
        """
        match = re.match(r"^/v1/datasets/([a-zA-Z0-9-]+)/files/(.+)$", path)
        if not match:
            return None
        ds_id, filepath = match.groups()
        filepath = urllib.parse.unquote(filepath)
        with _STATE_LOCK:
            ds = _DATASETS.get(ds_id)
        if not ds:
            self.send_error_json(404, "Dataset not found")
            return None
        ds_path = Path(ds["path"]).resolve()
        target_path = (ds_path / filepath).resolve()
        try:
            target_path.relative_to(ds_path)
        except ValueError:
            self.send_error_json(403, "Access denied: Path traversal detected")
            return None
        return ds_id, ds, ds_path, target_path

    @staticmethod
    def _write_caption_sidecar(ds_path, file_path, caption_text):
        """Write a .txt sidecar caption file next to *file_path*."""
        base = file_path.stem
        txt_path = file_path.parent / f"{base}.txt"
        txt_path.write_text(caption_text)
        return str(txt_path.relative_to(ds_path))

    @staticmethod
    def _write_metadata_csv(ds_path, filepath, text):
        """Update a row in metadata.csv (Piper / LJSpeech format)."""
        metadata_csv = ds_path / "metadata.csv"
        base = Path(filepath).stem
        rows = []
        if metadata_csv.exists():
            try:
                with open(metadata_csv, "r", encoding="utf-8") as f:
                    reader = csv.reader(f, delimiter="|")
                    for row in reader:
                        if len(row) >= 2 and row[0].strip() == base:
                            row[1] = text
                        rows.append(row)
            except Exception:
                pass
        with open(metadata_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerows(rows)
        return "metadata.csv"

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == "/v1/datasets":
            with _STATE_LOCK:
                return self.send_json({"datasets": list(_DATASETS.values())})

        if path == "/v1/commands":
            commands = _get_command_schemas()
            return self.send_json({"commands": commands})

        match = re.match(r"^/v1/datasets/([a-zA-Z0-9-]+)$", path)
        if match:
            ds_id = match.group(1)
            with _STATE_LOCK:
                ds = _DATASETS.get(ds_id)
            if not ds:
                return self.send_error_json(404, "Dataset not found")
            return self.send_json(ds)

        match = re.match(r"^/v1/datasets/([a-zA-Z0-9-]+)/files$", path)
        if match:
            ds_id = match.group(1)
            with _STATE_LOCK:
                ds = _DATASETS.get(ds_id)
            if not ds:
                return self.send_error_json(404, "Dataset not found")

            ds_path = Path(ds["path"])
            if not ds_path.exists():
                return self.send_error_json(404, "Dataset path no longer exists on disk")

            query_params = urllib.parse.parse_qs(parsed_url.query)
            folder = query_params.get("folder", [""])[0]
            group = query_params.get("group", ["false"])[0] == "true"

            files = []
            search_path = ds_path
            if folder:
                search_path = ds_path / folder
                if not search_path.exists():
                    return self.send_error_json(404, f"Folder not found: {folder}")

            file_map = {}
            metadata_map = {}

            metadata_csv = search_path / "metadata.csv"
            if metadata_csv.exists():
                try:
                    with open(metadata_csv, "r", encoding="utf-8") as f:
                        reader = csv.reader(f, delimiter="|")
                        for row in reader:
                            if len(row) >= 2:
                                utt_id = row[0].strip()
                                text = row[1].strip()
                                metadata_map[utt_id] = text
                except Exception:
                    pass

            for p in search_path.rglob("*"):
                if not p.is_file():
                    continue

                rel_path = str(p.relative_to(ds_path))
                name = p.name
                ext = p.suffix.lower()
                base = p.stem
                parent_name = p.parent.name.lower()

                file_type = "other"
                if parent_name in ("input", "control"):
                    file_type = "input"
                elif parent_name in ("target", "output"):
                    file_type = "target"
                elif parent_name == "mask":
                    file_type = "mask"
                elif parent_name == "canny":
                    file_type = "canny"
                elif parent_name == "pose":
                    file_type = "pose"
                elif parent_name == "seg":
                    file_type = "seg"
                elif parent_name == "depth":
                    file_type = "depth"
                elif parent_name == "normal":
                    file_type = "normal"
                elif "_input" in base or "_start" in base or "_control" in base:
                    file_type = "input"
                    base = base.replace("_input", "").replace("_start", "").replace("_control", "")
                elif "_target" in base or "_end" in base:
                    file_type = "target"
                    base = base.replace("_target", "").replace("_end", "")
                elif "_mask" in base:
                    file_type = "mask"
                    base = base.replace("_mask", "")
                elif "_canny" in base:
                    file_type = "canny"
                    base = base.replace("_canny", "")
                elif "_pose" in base:
                    file_type = "pose"
                    base = base.replace("_pose", "")
                elif "_seg" in base:
                    file_type = "seg"
                    base = base.replace("_seg", "")
                elif "_depth" in base:
                    file_type = "depth"
                    base = base.replace("_depth", "")
                elif "_normal" in base:
                    file_type = "normal"
                    base = base.replace("_normal", "")
                elif ext in [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"]:
                    file_type = "image"

                if base not in file_map:
                    file_map[base] = {"base": base, "files": []}

                file_map[base]["files"].append(
                    {
                        "path": rel_path,
                        "name": name,
                        "size_bytes": p.stat().st_size,
                        "extension": ext,
                        "file_type": file_type,
                    }
                )

                if ext == ".txt":
                    file_map[base]["caption_path"] = rel_path
                    file_map[base]["caption"] = p.read_text()
                elif ext in _AUDIO_EXTENSIONS and base in metadata_map:
                    file_map[base]["caption"] = metadata_map[base]
                    file_map[base]["caption_source"] = "metadata.csv"

            pairs = []
            for base, data in file_map.items():
                files.append(
                    {
                        "base": base,
                        "name": data["files"][0]["name"],
                        "path": data["files"][0]["path"].rsplit("/", 1)[0]
                        if "/" in data["files"][0]["path"]
                        else "",
                        "files": data["files"],
                        "caption": data.get("caption", ""),
                        "caption_path": data.get("caption_path", ""),
                    }
                )
                pairs.append(
                    {
                        "base": base,
                        "input": next(
                            (f["path"] for f in data["files"] if f["file_type"] == "input"), None
                        ),
                        "target": next(
                            (f["path"] for f in data["files"] if f["file_type"] == "target"), None
                        ),
                        "mask": next(
                            (f["path"] for f in data["files"] if f["file_type"] == "mask"), None
                        ),
                        "canny": next(
                            (f["path"] for f in data["files"] if f["file_type"] == "canny"), None
                        ),
                        "pose": next(
                            (f["path"] for f in data["files"] if f["file_type"] == "pose"), None
                        ),
                        "seg": next(
                            (f["path"] for f in data["files"] if f["file_type"] == "seg"), None
                        ),
                        "depth": next(
                            (f["path"] for f in data["files"] if f["file_type"] == "depth"), None
                        ),
                        "normal": next(
                            (f["path"] for f in data["files"] if f["file_type"] == "normal"), None
                        ),
                        "image": next(
                            (f["path"] for f in data["files"] if f["file_type"] == "image"), None
                        ),
                        "caption": data.get("caption", ""),
                        "all_files": data["files"],
                    }
                )

            if group:
                return self.send_json({"pairs": pairs})
            return self.send_json({"files": files})

        result = self._resolve_dataset_file(path)
        if result is not None:
            ds_id, ds, ds_path, target_path = result

            if not target_path.is_file():
                return self.send_error_json(404, "File not found")

            query_params = urllib.parse.parse_qs(parsed_url.query)
            info = query_params.get("info", ["false"])[0] == "true"

            if info:
                resp = {
                    "path": str(target_path.relative_to(ds_path)),
                    "size_bytes": target_path.stat().st_size,
                }
                base = target_path.stem
                ext = target_path.suffix.lower()
                txt_path = target_path.parent / f"{base}.txt"
                if txt_path.exists():
                    resp["caption"] = txt_path.read_text()
                    resp["caption_path"] = str(txt_path.relative_to(ds_path))
                metadata_csv = ds_path / "metadata.csv"
                if metadata_csv.exists() and ext in _AUDIO_EXTENSIONS:
                    try:
                        with open(metadata_csv, "r", encoding="utf-8") as f:
                            reader = csv.reader(f, delimiter="|")
                            for row in reader:
                                if len(row) >= 2 and row[0].strip() == base:
                                    resp["metadata"] = row[1].strip()
                                    break
                    except Exception:
                        pass
                return self.send_json(resp)

            mime = mimetypes.guess_type(str(target_path))[0] or "application/octet-stream"
            file_size = target_path.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with open(target_path, "rb") as f:
                shutil.copyfileobj(f, self.wfile, length=8192)
            return

        if path == "/v1/jobs":
            with _STATE_LOCK:
                safe_jobs = [{k: v for k, v in j.items() if k != "proc"} for j in _JOBS.values()]
            return self.send_json({"jobs": safe_jobs})

        match = re.match(r"^/v1/jobs/([a-zA-Z0-9-]+)$", path)
        if match:
            job_id = match.group(1)
            with _STATE_LOCK:
                job = _JOBS.get(job_id)
            if not job:
                return self.send_error_json(404, "Job not found")
            safe_job = {
                k: list(v) if isinstance(v, deque) else v for k, v in job.items() if k != "proc"
            }
            return self.send_json(safe_job)

        self.send_error_json(404, "Endpoint not found")

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        content_type = self.headers.get("Content-Type", "")
        body = None
        if content_type == "application/json":
            body = self.parse_body()
            if body is None:
                return self.send_error_json(400, "Invalid JSON payload")

        if path == "/v1/datasets":
            if body is None:
                return self.send_error_json(400, "Invalid JSON payload")
            ds_path_str = body.get("path")
            if not ds_path_str:
                return self.send_error_json(400, "Missing 'path' in payload")

            ds_path = Path(ds_path_str).resolve()
            if not ds_path.is_dir():
                return self.send_error_json(
                    400, f"Provided path does not exist or is not a directory: {ds_path_str}"
                )

            ds_type = _detect_type(ds_path)
            ds_id = str(uuid.uuid4())
            folders = body.get("folders", [])
            if not folders:
                subdirs = [p.name for p in ds_path.iterdir() if p.is_dir()]
                folders = subdirs[:4]

            ds_obj = {
                "id": ds_id,
                "name": body.get("name", ds_path.name),
                "path": str(ds_path),
                "type": ds_type,
                "created_at": time.time(),
                "folders": folders,
            }
            with _STATE_LOCK:
                _DATASETS[ds_id] = ds_obj
            return self.send_json(ds_obj, status=201)

        if path == "/v1/jobs":
            if body is None:
                return self.send_error_json(400, "Invalid JSON payload")
            command = body.get("command")
            args_dict = body.get("args", {})
            if not command:
                return self.send_error_json(400, "Missing 'command' in payload")

            argv = _args_to_argv(command, args_dict)
            job_id = str(uuid.uuid4())

            job_obj = {
                "id": job_id,
                "command": command,
                "argv": argv,
                "status": "running",
                "output": deque(maxlen=1000),
                "exit_code": None,
                "started_at": time.time(),
                "ended_at": None,
                "proc": None,
            }

            try:
                proc = subprocess.Popen(
                    argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
                )
                job_obj["proc"] = proc
                with _STATE_LOCK:
                    _JOBS[job_id] = job_obj

                def _reader(jid, p):
                    for line in p.stdout:
                        with _STATE_LOCK:
                            if jid in _JOBS:
                                _JOBS[jid]["output"].append(line.rstrip())
                    p.wait()
                    with _STATE_LOCK:
                        if jid in _JOBS:
                            _JOBS[jid]["status"] = "done" if p.returncode == 0 else "failed"
                            _JOBS[jid]["exit_code"] = p.returncode
                            _JOBS[jid]["ended_at"] = time.time()

                threading.Thread(target=_reader, args=(job_id, proc), daemon=True).start()
                return self.send_json({"id": job_id, "status": "started"}, status=202)
            except Exception as e:
                return self.send_error_json(500, f"Failed to start job: {str(e)}")

        result = self._resolve_dataset_file(path)
        if result is not None:
            ds_id, ds, ds_path, target_path = result
            content_length = int(self.headers.get("Content-Length", 0))

            sidecar_results = {}

            if content_type == "application/json":
                if body is None:
                    return self.send_error_json(400, "Invalid JSON payload")
                data = body.get("data", "")
                if isinstance(data, str) and data:
                    import base64

                    try:
                        file_data = base64.b64decode(data)
                    except Exception:
                        return self.send_error_json(400, "Invalid base64 data")
                else:
                    file_data = None

                if "caption" in body:
                    rel = self._write_caption_sidecar(ds_path, target_path, body["caption"])
                    sidecar_results["caption_path"] = rel
                if "metadata" in body:
                    rel = self._write_metadata_csv(
                        ds_path, str(target_path.relative_to(ds_path)), body["metadata"]
                    )
                    sidecar_results["metadata_path"] = rel

                if file_data is not None:
                    try:
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        target_path.write_bytes(file_data)
                    except Exception as e:
                        return self.send_error_json(500, f"Failed to save file: {str(e)}")

                result_resp = {"status": "created", "path": str(target_path.relative_to(ds_path))}
                result_resp.update(sidecar_results)
                return self.send_json(result_resp, status=201)
            else:
                file_data = self.rfile.read(content_length)
                try:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    target_path.write_bytes(file_data)
                    return self.send_json(
                        {"status": "created", "path": str(target_path.relative_to(ds_path))},
                        status=201,
                    )
                except Exception as e:
                    return self.send_error_json(500, f"Failed to save file: {str(e)}")

        self.send_error_json(404, "Endpoint not found")

    def do_PUT(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        body = self.parse_body()

        if body is None:
            return self.send_error_json(400, "Invalid JSON payload")

        result = self._resolve_dataset_file(path)
        if result is not None:
            ds_id, ds, ds_path, target_path = result

            if not target_path.is_file():
                return self.send_error_json(404, "File not found")

            sidecar_results = {}

            if "caption" in body:
                rel = self._write_caption_sidecar(ds_path, target_path, body["caption"])
                sidecar_results["caption_path"] = rel
            if "metadata" in body:
                rel = self._write_metadata_csv(
                    ds_path, str(target_path.relative_to(ds_path)), body["metadata"]
                )
                sidecar_results["metadata_path"] = rel

            content_type = self.headers.get("Content-Type", "")
            if content_type == "application/json":
                data = body.get("data", "")
                if isinstance(data, str) and data:
                    import base64

                    try:
                        file_data = base64.b64decode(data)
                        target_path.write_bytes(file_data)
                    except Exception as e:
                        return self.send_error_json(500, f"Failed to save file: {str(e)}")

            result_resp = {"status": "saved", "path": str(target_path.relative_to(ds_path))}
            result_resp.update(sidecar_results)
            return self.send_json(result_resp)

        self.send_error_json(404, "Endpoint not found")

    def do_PATCH(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        body = self.parse_body()

        if body is None:
            return self.send_error_json(400, "Invalid JSON payload")

        match = re.match(r"^/v1/datasets/([a-zA-Z0-9-]+)$", path)
        if match:
            ds_id = match.group(1)
            with _STATE_LOCK:
                if ds_id not in _DATASETS:
                    return self.send_error_json(404, "Dataset not found")
                if "name" in body:
                    _DATASETS[ds_id]["name"] = body["name"]
                if "folders" in body:
                    _DATASETS[ds_id]["folders"] = body["folders"]
                return self.send_json(_DATASETS[ds_id])

        self.send_error_json(404, "Endpoint not found")

    def do_DELETE(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        match = re.match(r"^/v1/datasets/([a-zA-Z0-9-]+)$", path)
        if match:
            ds_id = match.group(1)
            with _STATE_LOCK:
                if ds_id in _DATASETS:
                    del _DATASETS[ds_id]
                    return self.send_json({"status": "deleted"})
            return self.send_error_json(404, "Dataset not found")

        result = self._resolve_dataset_file(path)
        if result is not None:
            ds_id, ds, ds_path, target_path = result

            if not target_path.is_file():
                return self.send_error_json(404, "File not found")

            try:
                target_path.unlink()
            except Exception as e:
                return self.send_error_json(500, f"Failed to delete file: {str(e)}")

            query_params = urllib.parse.parse_qs(parsed_url.query)
            remove_caption = query_params.get("caption", ["false"])[0] == "true"
            caption_path = target_path.parent / f"{target_path.stem}.txt"
            if remove_caption and caption_path.exists():
                try:
                    caption_path.unlink()
                except Exception:
                    pass

            return self.send_json(
                {"status": "deleted", "path": str(target_path.relative_to(ds_path))}
            )

        match = re.match(r"^/v1/jobs/([a-zA-Z0-9-]+)$", path)
        if match:
            job_id = match.group(1)
            with _STATE_LOCK:
                job = _JOBS.get(job_id)
                if not job:
                    return self.send_error_json(404, "Job not found")
                if job["status"] == "running" and job["proc"]:
                    job["proc"].terminate()
                    job["status"] = "cancelled"
                    job["ended_at"] = time.time()
                    return self.send_json({"status": "cancelled"})
                return self.send_error_json(400, "Job is not running")

        self.send_error_json(404, "Endpoint not found")


def cmd_server(args):
    """Start the REST API server."""
    port = args.port
    max_port_attempts = 10
    server = None

    for attempt in range(max_port_attempts):
        try:
            server = ThreadingHTTPServer(("0.0.0.0", port), APIHandler)
            break
        except OSError as e:
            if attempt < max_port_attempts - 1 and e.errno == 48:
                port += 1
                print(f"Port {port - 1} in use, trying {port}...")
                continue
            raise

    if server is None:
        raise RuntimeError("Failed to start server")

    print(f"datasety REST API running at http://localhost:{port}/v1/")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


def register_parser(subparsers):
    """Register the server subcommand."""
    p = subparsers.add_parser(
        "server",
        help="Start the datasety REST API server for remote dataset and job management",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the API server (default: 8080)",
    )
    p.set_defaults(func=cmd_server)
