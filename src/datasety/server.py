"""Universal web server for dataset management — datasety server."""

import io
import json
import mimetypes
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
import wave as _wave
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.resources import files
from pathlib import Path

from PIL import Image

from datasety.common import get_image_files, hamming_distance, image_phash

IMAGE_FORMATS = ["jpg", "jpeg", "png", "webp", "bmp", "tiff"]
AUDIO_FORMATS = ["wav"]

# Pattern to extract <script>…</script> block from page templates
_SCRIPT_RE = re.compile(r"<script>(.*?)</script>", re.DOTALL)


# ── Template rendering ─────────────────────────────────────────────────────────


def _load_template(name: str) -> str:
    """Load a template file from the templates package."""
    return files("datasety.templates").joinpath(name).read_text()


def _render_template(name: str, **context) -> str:
    """Load and render a template with string substitution."""
    html = _load_template(name)
    for key, value in context.items():
        html = html.replace("{{ " + key + " }}", str(value))
    return html


def _render_page(page: str, has_pairs: bool = False) -> str:
    """Render a full page by combining base template with page content."""
    # Determine active nav
    nav_active = {
        "index": " active" if page == "index" else "",
        "gallery": " active" if page == "gallery" else "",
        "compare": " active" if page == "compare" else "",
        "pairs": " active" if page == "pairs" else "",
        "run": " active" if page == "run" else "",
        "audio": " active" if page == "audio" else "",
    }

    # Load page-specific content; extract any <script> block to inject via
    # extra_scripts so it runs after base.html's globals (api, loadDatasetInfo…)
    page_content = _load_template(f"{page}.html")
    page_scripts = ""
    script_match = _SCRIPT_RE.search(page_content)
    if script_match:
        page_scripts = script_match.group(1)
        page_content = _SCRIPT_RE.sub("", page_content)

    # Page-specific header extras
    header_extra = ""
    if page == "gallery":
        header_extra = '<input type="text" id="search" style="width:190px;padding:5px 10px" placeholder="Search..." autocomplete="off">'

    # Render base template
    base = _load_template("base.html")
    html = base
    html = html.replace("{{ page_title }}", page.title())
    html = html.replace("{{ nav_active_index }}", nav_active["index"])
    html = html.replace("{{ nav_active_gallery }}", nav_active["gallery"])
    html = html.replace("{{ nav_active_compare }}", nav_active["compare"])
    html = html.replace("{{ nav_active_pairs }}", nav_active["pairs"])
    html = html.replace("{{ nav_active_run }}", nav_active["run"])
    html = html.replace("{{ nav_active_audio }}", nav_active["audio"])
    html = html.replace("{{ header_extra }}", header_extra)
    html = html.replace("{{ content }}", page_content)
    html = html.replace("{{ extra_styles }}", "")
    html = html.replace(
        "{{ extra_scripts }}",
        '<script>\ndocument.addEventListener("DOMContentLoaded",function(){\n'
        + page_scripts
        + "\n});\n</script>"
        if page_scripts
        else "",
    )
    return html


# ── Shared helpers ────────────────────────────────────────────────────────────


def _hamming(h1, h2):
    """Alias for hamming_distance (kept for internal use)."""
    return hamming_distance(h1, h2)


def _orientation(w, h):
    r = w / h
    if r > 1.4:
        return "landscape"
    if r < 0.7:
        return "portrait"
    return "square"


def _compute_stats(images):
    """Recompute stats from the in-memory images list (called after mutations)."""
    valid = [i for i in images if "error" not in i]
    total = len(images)
    captions_found = sum(1 for i in valid if i.get("has_caption"))
    total_bytes = sum(i.get("size_bytes", 0) for i in valid)
    res = Counter(i["resolution"] for i in valid if "resolution" in i)
    fmt = Counter(i["format"] for i in valid if "format" in i)
    orient = Counter(i["orientation"] for i in valid if "orientation" in i)
    return {
        "total": total,
        "total_size_mb": round(total_bytes / 1_048_576, 1) if total_bytes else 0,
        "captions_found": captions_found,
        "captions_missing": len(valid) - captions_found,
        "resolutions": dict(res.most_common()),
        "formats": dict(fmt),
        "orientations": dict(orient),
    }


def _scan(input_dir: Path, recursive: bool, compute_hashes: bool):
    """Scan dataset directory and build the in-memory image list."""
    image_files = get_image_files(input_dir, IMAGE_FORMATS, recursive=recursive)
    images = []
    hashes = {}
    res_ctr: Counter = Counter()
    fmt_ctr: Counter = Counter()
    orient_ctr: Counter = Counter()
    total_bytes = 0
    captions_found = 0

    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                fmt = img_path.suffix.lower().lstrip(".")
                size_bytes = img_path.stat().st_size
                o = _orientation(w, h)
                cap_path = img_path.with_suffix(".txt")
                has_cap = cap_path.exists()
                cap_text = ""
                if has_cap:
                    captions_found += 1
                    cap_text = cap_path.read_text(encoding="utf-8", errors="replace").strip()
                phash = ""
                if compute_hashes:
                    phash = image_phash(img)
                    hashes.setdefault(phash, []).append(str(img_path))
                images.append(
                    {
                        "path": str(img_path),
                        "name": img_path.name,
                        "stem": img_path.stem,
                        "width": w,
                        "height": h,
                        "resolution": f"{w}x{h}",
                        "format": fmt,
                        "size_bytes": size_bytes,
                        "size_kb": round(size_bytes / 1024, 1),
                        "orientation": o,
                        "has_caption": has_cap,
                        "caption": cap_text,
                        "phash": phash,
                    }
                )
                res_ctr[f"{w}x{h}"] += 1
                fmt_ctr[fmt] += 1
                orient_ctr[o] += 1
                total_bytes += size_bytes
        except Exception as e:
            images.append(
                {
                    "path": str(img_path),
                    "name": img_path.name,
                    "stem": img_path.stem,
                    "error": str(e),
                }
            )

    stats = {
        "total": len(images),
        "total_size_mb": round(total_bytes / 1_048_576, 1) if total_bytes else 0,
        "captions_found": captions_found,
        "captions_missing": len(image_files) - captions_found,
        "resolutions": dict(res_ctr.most_common()),
        "formats": dict(fmt_ctr),
        "orientations": dict(orient_ctr),
    }
    return {"images": images, "hashes": hashes, "stats": stats}


def _build_pairs(input_dir: Path, control_dir: Path, recursive: bool):
    """Match target/control image files by stem, return pair dicts."""
    target_files = get_image_files(input_dir, IMAGE_FORMATS, recursive=recursive)
    control_files = get_image_files(control_dir, IMAGE_FORMATS, recursive=recursive)
    control_by_stem = {f.stem: f for f in control_files}
    pairs = []
    for tf in sorted(target_files, key=lambda p: p.stem):
        cf = control_by_stem.get(tf.stem)
        if cf is None:
            continue
        pairs.append(
            {
                "stem": tf.stem,
                "target": str(tf),
                "control": str(cf),
                "target_caption": str(tf.with_suffix(".txt")),
                "control_caption": str(cf.with_suffix(".txt")),
                "has_target_caption": tf.with_suffix(".txt").exists(),
                "has_control_caption": cf.with_suffix(".txt").exists(),
            }
        )
    return pairs


def _scan_audio(input_dir: Path, recursive: bool):
    """Scan dataset directory for audio files and build the in-memory audio list.

    Supports LJSpeech format (input_dir/wavs/ + metadata.csv) and flat directories.
    """
    # Determine audio root: prefer input_dir/wavs/ for LJSpeech, else input_dir
    wavs_dir = input_dir / "wavs"
    audio_root = wavs_dir if wavs_dir.exists() else input_dir

    audio_files: list[Path] = []
    if recursive:
        for fmt in AUDIO_FORMATS:
            audio_files.extend(audio_root.rglob(f"*.{fmt}"))
    else:
        for fmt in AUDIO_FORMATS:
            audio_files.extend(audio_root.glob(f"*.{fmt}"))

    audio_files = sorted(audio_files)

    # Load LJSpeech metadata.csv if present
    metadata: dict[str, str] = {}
    csv_path = input_dir / "metadata.csv"
    if csv_path.exists():
        for line in csv_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            # LJSpeech format: filename|text
            if "|" in line:
                parts = line.split("|", 1)
                metadata[parts[0]] = parts[1]
            else:
                metadata[line] = ""

    audio_list = []
    total_bytes = 0
    total_duration = 0.0
    transcriptions_found = 0

    for af in audio_files:
        try:
            size_bytes = af.stat().st_size
            duration = 0.0
            sample_rate = 22050
            try:
                with _wave.open(str(af), "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    duration = frames / float(rate) if rate > 0 else 0.0
                    sample_rate = rate
            except Exception:
                pass

            txt_path = af.with_suffix(".txt")
            has_txt = txt_path.exists()
            txt_text = metadata.get(af.name, "")
            if not txt_text and has_txt:
                txt_text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
            if txt_text:
                transcriptions_found += 1
                has_txt = True

            audio_list.append(
                {
                    "path": str(af),
                    "name": af.name,
                    "stem": af.stem,
                    "duration": round(duration, 2),
                    "sample_rate": sample_rate,
                    "size_bytes": size_bytes,
                    "size_kb": round(size_bytes / 1024, 1),
                    "has_transcription": has_txt,
                    "transcription": txt_text,
                }
            )
            total_bytes += size_bytes
            total_duration += duration
        except Exception as e:
            audio_list.append(
                {
                    "path": str(af),
                    "name": af.name,
                    "stem": af.stem,
                    "error": str(e),
                }
            )

    audio_stats = {
        "total": len(audio_list),
        "total_size_mb": round(total_bytes / 1_048_576, 1) if total_bytes else 0,
        "total_duration_s": round(total_duration, 1),
        "transcriptions_found": transcriptions_found,
        "transcriptions_missing": len(audio_list) - transcriptions_found,
    }
    return {"audio": audio_list, "stats": audio_stats}


def _dup_groups(hashes: dict, threshold: int = 4):
    groups = []
    items = list(hashes.items())
    seen: set = set()
    for i, (h1, p1) in enumerate(items):
        if h1 in seen:
            continue
        if len(p1) > 1:
            groups.append(p1)
            seen.add(h1)
            continue
        group = list(p1)
        for j in range(i + 1, len(items)):
            h2, p2 = items[j]
            if h2 in seen:
                continue
            if _hamming(h1, h2) <= threshold:
                group.extend(p2)
                seen.add(h2)
        if len(group) > 1:
            groups.append(group)
            seen.add(h1)
    return groups


# ── Command runner ────────────────────────────────────────────────────────────

COMMAND_SCHEMAS: dict = {
    "resize": {
        "description": "Resize and crop images to a target resolution",
        "args": [
            {
                "id": "input",
                "flag": "--input",
                "label": "Input dir",
                "type": "str",
                "required": True,
            },
            {
                "id": "output",
                "flag": "--output",
                "label": "Output dir",
                "type": "str",
                "required": True,
            },
            {
                "id": "resolution",
                "flag": "--resolution",
                "label": "Resolution",
                "type": "str",
                "placeholder": "512x512",
            },
            {
                "id": "crop_position",
                "flag": "--crop-position",
                "label": "Crop position",
                "type": "select",
                "options": ["center", "top", "bottom", "left", "right", "random"],
                "default": "center",
            },
            {
                "id": "output_format",
                "flag": "--output-format",
                "label": "Output format",
                "type": "select",
                "options": ["", "jpg", "png", "webp"],
                "default": "",
            },
            {
                "id": "quality",
                "flag": "--quality",
                "label": "Quality (jpg/webp)",
                "type": "str",
                "placeholder": "95",
            },
            {"id": "recursive", "flag": "--recursive", "label": "Recursive", "type": "bool"},
            {"id": "dry_run", "flag": "--dry-run", "label": "Dry run (preview)", "type": "bool"},
        ],
    },
    "caption": {
        "description": "Generate captions using Florence-2 (requires caption extras + GPU)",
        "args": [
            {
                "id": "input",
                "flag": "--input",
                "label": "Input dir",
                "type": "str",
                "required": True,
            },
            {"id": "output", "flag": "--output", "label": "Output dir", "type": "str"},
            {
                "id": "trigger_word",
                "flag": "--trigger-word",
                "label": "Trigger word",
                "type": "str",
                "placeholder": "ohwx person",
            },
            {"id": "recursive", "flag": "--recursive", "label": "Recursive", "type": "bool"},
            {
                "id": "overwrite",
                "flag": "--overwrite",
                "label": "Overwrite existing",
                "type": "bool",
            },
        ],
    },
    "align": {
        "description": "Align control/target pairs to matching dimensions (multiples of 32)",
        "args": [
            {
                "id": "target",
                "flag": "--target",
                "label": "Target dir",
                "type": "str",
                "required": True,
            },
            {
                "id": "control",
                "flag": "--control",
                "label": "Control dir",
                "type": "str",
                "required": True,
            },
            {
                "id": "multiple_of",
                "flag": "--multiple-of",
                "label": "Multiple of",
                "type": "str",
                "placeholder": "32",
                "default": "32",
            },
            {
                "id": "output_format",
                "flag": "--output-format",
                "label": "Output format",
                "type": "select",
                "options": ["", "jpg", "png", "webp"],
                "default": "",
            },
            {"id": "recursive", "flag": "--recursive", "label": "Recursive", "type": "bool"},
            {"id": "dry_run", "flag": "--dry-run", "label": "Dry run (preview)", "type": "bool"},
        ],
    },
    "inspect": {
        "description": "Show dataset statistics, caption coverage, and detect duplicates",
        "args": [
            {
                "id": "input",
                "flag": "--input",
                "label": "Input dir",
                "type": "str",
                "required": True,
            },
            {
                "id": "duplicates",
                "flag": "--duplicates",
                "label": "Detect duplicates",
                "type": "bool",
            },
            {
                "id": "json",
                "flag": "--json",
                "label": "Export JSON path",
                "type": "str",
                "placeholder": "report.json",
            },
            {
                "id": "csv",
                "flag": "--csv",
                "label": "Export CSV path",
                "type": "str",
                "placeholder": "images.csv",
            },
            {"id": "recursive", "flag": "--recursive", "label": "Recursive", "type": "bool"},
        ],
    },
    "filter": {
        "description": "Move/delete images matching a content query using CLIP",
        "args": [
            {
                "id": "input",
                "flag": "--input",
                "label": "Input dir",
                "type": "str",
                "required": True,
            },
            {
                "id": "output",
                "flag": "--output",
                "label": "Output dir",
                "type": "str",
                "required": True,
            },
            {
                "id": "query",
                "flag": "--query",
                "label": "Query",
                "type": "str",
                "required": True,
                "placeholder": "male face, legs",
            },
            {
                "id": "action",
                "flag": "--action",
                "label": "Action",
                "type": "select",
                "options": ["move", "copy", "delete"],
                "default": "move",
            },
            {
                "id": "threshold",
                "flag": "--threshold",
                "label": "Threshold (0–1)",
                "type": "str",
                "placeholder": "0.5",
            },
            {"id": "recursive", "flag": "--recursive", "label": "Recursive", "type": "bool"},
        ],
    },
    "degrade": {
        "description": "Create degraded image versions for upscale/enhance training",
        "args": [
            {
                "id": "input",
                "flag": "--input",
                "label": "Input dir",
                "type": "str",
                "required": True,
            },
            {
                "id": "output",
                "flag": "--output",
                "label": "Output dir",
                "type": "str",
                "required": True,
            },
            {
                "id": "type",
                "flag": "--type",
                "label": "Degradation type",
                "type": "select",
                "options": ["jpeg", "blur", "noise", "downscale", "random"],
                "default": "jpeg",
            },
            {
                "id": "intensity",
                "flag": "--intensity",
                "label": "Intensity (0–1)",
                "type": "str",
                "placeholder": "0.5",
            },
            {"id": "paired", "flag": "--paired", "label": "Output LR/HR pairs", "type": "bool"},
            {"id": "recursive", "flag": "--recursive", "label": "Recursive", "type": "bool"},
        ],
    },
    "shuffle": {
        "description": "Shuffle and rename image/caption file pairs",
        "args": [
            {
                "id": "input",
                "flag": "--input",
                "label": "Input dir",
                "type": "str",
                "required": True,
            },
            {
                "id": "output",
                "flag": "--output",
                "label": "Output dir",
                "type": "str",
                "required": True,
            },
            {"id": "recursive", "flag": "--recursive", "label": "Recursive", "type": "bool"},
        ],
    },
    "mask": {
        "description": "Generate segmentation masks using SAM / CLIPSeg",
        "args": [
            {
                "id": "input",
                "flag": "--input",
                "label": "Input dir",
                "type": "str",
                "required": True,
            },
            {
                "id": "output",
                "flag": "--output",
                "label": "Output dir",
                "type": "str",
                "required": True,
            },
            {
                "id": "keywords",
                "flag": "--keywords",
                "label": "Keywords",
                "type": "str",
                "placeholder": "face,hair,shirt",
            },
            {"id": "recursive", "flag": "--recursive", "label": "Recursive", "type": "bool"},
        ],
    },
    "synthetic": {
        "description": "Generate synthetic image variations using diffusion (requires synthetic extras + GPU)",
        "args": [
            {
                "id": "input",
                "flag": "--input",
                "label": "Input dir",
                "type": "str",
                "required": True,
            },
            {
                "id": "output",
                "flag": "--output",
                "label": "Output dir",
                "type": "str",
                "required": True,
            },
            {
                "id": "prompt",
                "flag": "--prompt",
                "label": "Prompt",
                "type": "str",
                "required": True,
                "placeholder": "a photo of a person wearing a hat",
            },
            {
                "id": "count",
                "flag": "--count",
                "label": "Images per input",
                "type": "str",
                "placeholder": "4",
            },
            {"id": "recursive", "flag": "--recursive", "label": "Recursive", "type": "bool"},
        ],
    },
    "train": {
        "description": "Fine-tune a LoRA model on the dataset (requires train extras + GPU)",
        "args": [
            {
                "id": "config",
                "flag": "--config",
                "label": "Config YAML",
                "type": "str",
                "placeholder": "train.yaml",
            },
            {"id": "input", "flag": "--input", "label": "Dataset dir", "type": "str"},
            {"id": "output", "flag": "--output", "label": "Output dir", "type": "str"},
            {
                "id": "steps",
                "flag": "--steps",
                "label": "Training steps",
                "type": "str",
                "placeholder": "1000",
            },
            {
                "id": "rank",
                "flag": "--rank",
                "label": "LoRA rank",
                "type": "str",
                "placeholder": "16",
            },
            {
                "id": "lr",
                "flag": "--lr",
                "label": "Learning rate",
                "type": "str",
                "placeholder": "1e-4",
            },
            {
                "id": "batch_size",
                "flag": "--batch-size",
                "label": "Batch size",
                "type": "str",
                "placeholder": "1",
            },
        ],
    },
    "audio": {
        "description": "Build TTS audio dataset from video/audio (YouTube, URL, or local file)",
        "args": [
            {
                "id": "input",
                "flag": "--input",
                "label": "Input source",
                "type": "str",
                "required": True,
                "placeholder": "video.mp4 or YouTube URL",
            },
            {
                "id": "output",
                "flag": "--output",
                "label": "Output directory",
                "type": "str",
                "required": True,
            },
            {
                "id": "sample_rate",
                "flag": "--sample-rate",
                "label": "Sample rate (Hz)",
                "type": "str",
                "placeholder": "22050",
            },
            {
                "id": "whisper_model",
                "flag": "--whisper-model",
                "label": "Whisper model",
                "type": "select",
                "options": ["tiny", "base", "small", "medium", "large-v3"],
                "default": "base",
            },
            {
                "id": "language",
                "flag": "--language",
                "label": "Language code",
                "type": "str",
                "placeholder": "en (auto-detect if empty)",
            },
            {
                "id": "min_duration",
                "flag": "--min-duration",
                "label": "Min segment (s)",
                "type": "str",
                "placeholder": "1.5",
            },
            {
                "id": "max_duration",
                "flag": "--max-duration",
                "label": "Max segment (s)",
                "type": "str",
                "placeholder": "10.0",
            },
            {
                "id": "normalize_numbers",
                "flag": "--normalize-numbers",
                "label": "Expand numbers to words",
                "type": "bool",
            },
            {
                "id": "demucs",
                "flag": "--demucs",
                "label": "Isolate vocals (Demucs)",
                "type": "bool",
            },
            {"id": "dry_run", "flag": "--dry-run", "label": "Dry run (preview)", "type": "bool"},
        ],
    },
}

_jobs: dict = {}
_jobs_lock = threading.Lock()


def _args_to_argv(command: str, args: dict) -> list[str]:
    """Convert form args dict to datasety CLI argv list."""
    schema = COMMAND_SCHEMAS.get(command, {}).get("args", [])
    argv = []
    for adef in schema:
        val = args.get(adef["id"])
        if val is None or val == "" or val is False:
            continue
        if adef["type"] == "bool":
            if val:
                argv.append(adef["flag"])
        else:
            argv += [adef["flag"], str(val)]
    return argv


def _start_job(command: str, argv: list[str]) -> str:
    """Spawn a datasety subprocess and return its job ID."""
    jid = uuid.uuid4().hex[:12]
    cmd = [sys.executable, "-m", "datasety.cli"] + [command] + argv
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as exc:
        with _jobs_lock:
            _jobs[jid] = {
                "id": jid,
                "command": command,
                "argv": argv,
                "proc": None,
                "output": [f"Error: {exc}"],
                "status": "failed",
                "started_at": time.time(),
                "ended_at": time.time(),
                "exit_code": -1,
            }
        return jid

    entry: dict = {
        "id": jid,
        "command": command,
        "argv": argv,
        "proc": proc,
        "output": [],
        "status": "running",
        "started_at": time.time(),
        "ended_at": None,
        "exit_code": None,
    }
    with _jobs_lock:
        _jobs[jid] = entry
        done = sorted(
            [j for j in _jobs.values() if j["status"] != "running"],
            key=lambda j: j["started_at"],
        )
        for j in done[:-19]:
            del _jobs[j["id"]]

    def _reader(jid: str, proc: subprocess.Popen) -> None:
        for line in proc.stdout:  # type: ignore[union-attr]
            with _jobs_lock:
                if jid in _jobs:
                    _jobs[jid]["output"].append(line.rstrip())
        proc.wait()
        with _jobs_lock:
            if jid in _jobs:
                _jobs[jid]["status"] = "done" if proc.returncode == 0 else "failed"
                _jobs[jid]["exit_code"] = proc.returncode
                _jobs[jid]["ended_at"] = time.time()

    threading.Thread(target=_reader, args=(jid, proc), daemon=True).start()
    return jid


# ── HTTP handler ──────────────────────────────────────────────────────────────


def _make_handler(
    input_dir: Path,
    control_dir: Path | None,
    recursive: bool,
    compute_hashes: bool,
):
    # Resolve to absolute paths so relative_to() works in _in_allowed
    input_dir = input_dir.resolve()
    if control_dir:
        control_dir = control_dir.resolve()
    print("Scanning dataset…")
    dataset = _scan(input_dir, recursive, compute_hashes)
    audio_data = _scan_audio(input_dir, recursive)
    pairs: list = _build_pairs(input_dir, control_dir, recursive) if control_dir else []
    thumb_cache: dict = {}
    print(
        f"Found {dataset['stats']['total']} images"
        + (f", {len(pairs)} matched pairs." if control_dir else ".")
        + f", {audio_data['stats']['total']} audio files."
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(f"[server] {self.client_address[0]} {fmt % args}")

        def _json(self, data, status=200):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _err(self, status, msg):
            self._json({"error": msg}, status)

        def _body(self):
            n = int(self.headers.get("Content-Length", 0))
            return self.rfile.read(n) if n else b""

        def _in_allowed(self, p: Path) -> bool:
            """Check path is under input_dir or (when set) control_dir."""
            try:
                p.resolve().relative_to(input_dir.resolve())
                return True
            except ValueError:
                pass
            if control_dir:
                try:
                    p.resolve().relative_to(control_dir.resolve())
                    return True
                except ValueError:
                    pass
            return False

        def _in_input(self, p: Path) -> bool:
            try:
                p.resolve().relative_to(input_dir.resolve())
                return True
            except ValueError:
                return False

        # ── Routing ──────────────────────────────────────────────────────────

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            params = dict(urllib.parse.parse_qsl(parsed.query))

            # Page routes
            if path == "/":
                self._serve_page("index")
            elif path == "/gallery":
                self._serve_page("gallery")
            elif path == "/compare":
                self._serve_page("compare")
            elif path == "/pairs":
                self._serve_page("pairs")
            elif path == "/run":
                self._serve_page("run")
            elif path == "/audio":
                self._serve_page("audio")
            # API routes
            elif path == "/api/mode":
                self._json(
                    {
                        "has_pairs": bool(control_dir),
                        "control_dir": str(control_dir) if control_dir else None,
                        "pairs_count": len(pairs),
                    }
                )
            elif path == "/api/dataset":
                self._json(
                    {
                        "input_dir": str(input_dir),
                        "total": dataset["stats"]["total"],
                    }
                )
            elif path == "/api/stats":
                self._json(dataset["stats"])
            elif path == "/api/images":
                self._images(params)
            elif path == "/api/image":
                self._image(params)
            elif path == "/api/thumbnail":
                self._thumbnail(params)
            elif path == "/api/image/info":
                self._image_info(params)
            elif path == "/api/caption":
                self._caption(params)
            elif path == "/api/pairs":
                self._json(pairs)
            elif path == "/api/duplicates":
                self._duplicates()
            elif path == "/api/commands":
                self._json(COMMAND_SCHEMAS)
            elif path == "/api/job":
                self._get_job(params)
            elif path == "/api/jobs":
                self._get_jobs()
            elif path == "/api/audio":
                self._audio_list(params)
            elif path == "/api/audio/file":
                self._audio_file(params)
            elif path == "/api/audio/stats":
                self._json(audio_data["stats"])
            elif path == "/api/audio/transcription":
                self._audio_transcription()
            else:
                self._err(404, "Not found")

        def do_POST(self):
            path = urllib.parse.urlparse(self.path).path
            if path == "/api/caption":
                self._save_caption()
            elif path == "/api/delete":
                self._delete()
            elif path == "/api/pair/delete":
                self._delete_pair()
            elif path == "/api/upload":
                self._upload()
            elif path == "/api/run":
                self._run_command()
            elif path == "/api/job/cancel":
                self._cancel_job()
            elif path == "/api/audio/transcription":
                self._audio_transcription()
            elif path == "/api/audio/delete":
                self._audio_delete()
            else:
                self._err(404, "Not found")

        # ── Handlers ─────────────────────────────────────────────────────────

        def _serve_page(self, page: str):
            """Serve an HTML page from templates."""
            try:
                body = _render_page(page, has_pairs=bool(control_dir)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self._err(500, f"Template error: {e}")

        def _images(self, params):
            imgs = list(dataset["images"])
            if fmt := params.get("format"):
                imgs = [i for i in imgs if i.get("format") == fmt.lower()]
            if o := params.get("orientation"):
                imgs = [i for i in imgs if i.get("orientation") == o]
            if res := params.get("resolution"):
                imgs = [i for i in imgs if i.get("resolution") == res]
            if hc := params.get("has_caption"):
                imgs = [i for i in imgs if i.get("has_caption") == (hc == "true")]
            if q := params.get("search"):
                ql = q.lower()
                imgs = [
                    i
                    for i in imgs
                    if ql in i.get("name", "").lower() or ql in i.get("caption", "").lower()
                ]
            by = params.get("sort", "name")
            rev = params.get("order", "asc") == "desc"
            key_fns = {
                "size": lambda i: i.get("size_bytes", 0),
                "resolution": lambda i: i.get("width", 0) * i.get("height", 0),
                "format": lambda i: i.get("format", ""),
            }
            imgs = sorted(
                imgs,
                key=key_fns.get(by, lambda i: i.get("name", "")),
                reverse=rev,
            )
            self._json({"images": imgs, "total": len(imgs)})

        def _image(self, params):
            fp = params.get("path")
            if not fp:
                self._err(400, "Missing path")
                return
            p = Path(fp)
            if not p.exists():
                self._err(404, "Not found")
                return
            if not self._in_allowed(p):
                self._err(403, "Access denied")
                return
            data = p.read_bytes()
            mime = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            self.wfile.write(data)

        def _thumbnail(self, params):
            fp = params.get("path")
            size = min(int(params.get("size", "200")), 400)
            if not fp:
                self._err(400, "Missing path")
                return
            p = Path(fp)
            if not p.exists():
                self._err(404, "Not found")
                return
            if not self._in_allowed(p):
                self._err(403, "Access denied")
                return
            cache_key = f"{fp}:{size}"
            if cache_key in thumb_cache:
                data, mime = thumb_cache[cache_key]
            else:
                try:
                    with Image.open(p) as img:
                        img.thumbnail((size, size), Image.LANCZOS)
                        if img.mode not in ("RGB", "RGBA", "L"):
                            img = img.convert("RGB")
                        fmt = "JPEG" if img.mode in ("RGB", "L") else "PNG"
                        buf = io.BytesIO()
                        img.save(buf, format=fmt, quality=85)
                        data = buf.getvalue()
                        mime = "image/jpeg" if fmt == "JPEG" else "image/png"
                    if len(thumb_cache) < 1000:
                        thumb_cache[cache_key] = (data, mime)
                except Exception:
                    self._err(500, "Thumbnail failed")
                    return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            self.wfile.write(data)

        def _image_info(self, params):
            fp = params.get("path")
            if not fp:
                self._err(400, "Missing path")
                return
            for img in dataset["images"]:
                if img.get("path") == fp:
                    self._json(img)
                    return
            self._err(404, "Not found")

        def _caption(self, params):
            fp = params.get("path")
            if not fp:
                self._err(400, "Missing path")
                return
            p = Path(fp)
            if not p.exists():
                self._json({"text": "", "exists": False})
                return
            self._json(
                {
                    "text": p.read_text(encoding="utf-8", errors="replace"),
                    "exists": True,
                }
            )

        def _save_caption(self):
            try:
                data = json.loads(self._body())
            except json.JSONDecodeError:
                self._err(400, "Invalid JSON")
                return
            fp = data.get("path", "")
            text = data.get("text", "")
            if not fp:
                self._err(400, "Missing path")
                return
            p = Path(fp)
            if p.suffix != ".txt":
                self._err(403, "Only .txt files allowed")
                return
            if not self._in_allowed(p):
                self._err(403, "Access denied")
                return
            if text.strip():
                p.write_text(text, encoding="utf-8")
            elif p.exists():
                p.unlink()
            # Update cached gallery entry (input_dir images only)
            for img in dataset["images"]:
                img_p = Path(img.get("path", ""))
                if img_p.stem == p.stem and img_p.parent == p.parent:
                    img["has_caption"] = bool(text.strip())
                    img["caption"] = text.strip()
                    break
            # Update pairs entry if applicable
            for pair in pairs:
                if pair["target_caption"] == fp:
                    pair["has_target_caption"] = bool(text.strip())
                elif pair["control_caption"] == fp:
                    pair["has_control_caption"] = bool(text.strip())
            dataset["stats"] = _compute_stats(dataset["images"])
            self._json({"ok": True})

        def _delete(self):
            try:
                data = json.loads(self._body())
            except json.JSONDecodeError:
                self._err(400, "Invalid JSON")
                return
            paths = data.get("paths", [])
            if not paths:
                self._err(400, "Missing paths")
                return
            deleted = []
            for ps in paths:
                p = Path(ps)
                if not self._in_allowed(p):
                    continue
                if p.exists() and p.suffix.lower().lstrip(".") in IMAGE_FORMATS:
                    p.unlink()
                    deleted.append(str(p))
                cap = p.with_suffix(".txt")
                if cap.exists():
                    cap.unlink()
                    deleted.append(str(cap))
            dead = set(deleted)
            dataset["images"] = [i for i in dataset["images"] if i.get("path") not in dead]
            for h, pl in list(dataset["hashes"].items()):
                dataset["hashes"][h] = [x for x in pl if x not in dead]
                if not dataset["hashes"][h]:
                    del dataset["hashes"][h]
            # Purge thumbnails for deleted files
            for key in [k for k in thumb_cache if k.split(":")[0] in dead]:
                del thumb_cache[key]
            dataset["stats"] = _compute_stats(dataset["images"])
            self._json({"deleted": deleted, "count": len(deleted)})

        def _delete_pair(self):
            try:
                data = json.loads(self._body())
            except json.JSONDecodeError:
                self._err(400, "Invalid JSON")
                return
            stem = data.get("stem")
            if not stem:
                self._err(400, "Missing stem")
                return
            deleted = []
            for pair in pairs:
                if pair["stem"] != stem:
                    continue
                for fp_str in [pair["target"], pair["control"]]:
                    p = Path(fp_str)
                    ext = p.suffix.lower().lstrip(".")
                    if p.exists() and ext in IMAGE_FORMATS:
                        p.unlink()
                        deleted.append(fp_str)
                    cap = p.with_suffix(".txt")
                    if cap.exists():
                        cap.unlink()
                        deleted.append(str(cap))
                break
            # Remove pair from list and gallery cache
            dead = set(deleted)
            pairs[:] = [p for p in pairs if p["stem"] != stem]
            dataset["images"] = [i for i in dataset["images"] if i.get("path") not in dead]
            # Purge thumbnails
            for key in [k for k in thumb_cache if k.split(":")[0] in dead]:
                del thumb_cache[key]
            dataset["stats"] = _compute_stats(dataset["images"])
            self._json({"deleted": deleted, "count": len(deleted)})

        def _upload(self):
            ct = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in ct:
                self._err(400, "Expected multipart/form-data")
                return
            boundary = ""
            for part in ct.split(";"):
                part = part.strip()
                if part.startswith("boundary="):
                    boundary = part[9:].strip().strip('"')
            if not boundary:
                self._err(400, "Missing boundary")
                return
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n)
            delim = f"--{boundary}".encode()
            raw_parts = body.split(delim)
            uploaded = []
            for raw in raw_parts[1:]:
                if raw.startswith(b"--"):
                    break
                sep = raw.find(b"\r\n\r\n")
                if sep < 0:
                    continue
                raw_hdrs = raw[:sep].lstrip(b"\r\n")
                content = raw[sep + 4 :]
                if content.endswith(b"\r\n"):
                    content = content[:-2]
                if not content:
                    continue
                hdrs = {}
                for line in raw_hdrs.decode("utf-8", errors="replace").splitlines():
                    if ":" in line:
                        k, _, v = line.partition(":")
                        hdrs[k.strip().lower()] = v.strip()
                disp = hdrs.get("content-disposition", "")
                filename = None
                for token in disp.split(";"):
                    token = token.strip()
                    if token.startswith("filename="):
                        filename = token[9:].strip('"').strip("'")
                if not filename:
                    continue
                filename = Path(filename).name
                ext = Path(filename).suffix.lower().lstrip(".")
                if ext not in IMAGE_FORMATS:
                    continue
                dest = input_dir / filename
                counter = 1
                while dest.exists():
                    stem = Path(filename).stem
                    dest = input_dir / f"{stem}_{counter}{Path(filename).suffix}"
                    counter += 1
                try:
                    dest.write_bytes(content)
                    with Image.open(dest) as img:
                        w, h = img.size
                        o = _orientation(w, h)
                        size_bytes = dest.stat().st_size
                        dataset["images"].append(
                            {
                                "path": str(dest),
                                "name": dest.name,
                                "stem": dest.stem,
                                "width": w,
                                "height": h,
                                "resolution": f"{w}x{h}",
                                "format": ext,
                                "size_bytes": size_bytes,
                                "size_kb": round(size_bytes / 1024, 1),
                                "orientation": o,
                                "has_caption": False,
                                "caption": "",
                                "phash": "",
                            }
                        )
                        uploaded.append(dest.name)
                except Exception:
                    dest.unlink(missing_ok=True)
            dataset["stats"] = _compute_stats(dataset["images"])
            self._json({"uploaded": uploaded, "count": len(uploaded)})

        def _duplicates(self):
            if not dataset["hashes"]:
                self._json({"groups": [], "total_groups": 0, "total_images": 0})
                return
            groups = _dup_groups(dataset["hashes"])
            self._json(
                {
                    "groups": groups,
                    "total_groups": len(groups),
                    "total_images": sum(len(g) for g in groups),
                }
            )

        def _run_command(self):
            try:
                data = json.loads(self._body())
            except json.JSONDecodeError:
                self._err(400, "Invalid JSON")
                return
            command = data.get("command", "")
            args = data.get("args", {})
            if command not in COMMAND_SCHEMAS:
                self._err(400, f"Unknown command: {command}")
                return
            argv = _args_to_argv(command, args)
            jid = _start_job(command, argv)
            self._json({"job_id": jid})

        def _get_job(self, params):
            jid = params.get("id", "")
            with _jobs_lock:
                j = _jobs.get(jid)
            if not j:
                self._err(404, "Job not found")
                return
            self._json(
                {
                    "id": j["id"],
                    "command": j["command"],
                    "argv": j["argv"],
                    "status": j["status"],
                    "output": j["output"],
                    "exit_code": j["exit_code"],
                    "started_at": j["started_at"],
                    "ended_at": j["ended_at"],
                }
            )

        def _get_jobs(self):
            with _jobs_lock:
                jobs = sorted(_jobs.values(), key=lambda j: j["started_at"], reverse=True)
                result = [
                    {
                        "id": j["id"],
                        "command": j["command"],
                        "status": j["status"],
                        "started_at": j["started_at"],
                        "ended_at": j["ended_at"],
                        "exit_code": j["exit_code"],
                    }
                    for j in jobs
                ]
            self._json({"jobs": result})

        def _cancel_job(self):
            try:
                data = json.loads(self._body())
            except json.JSONDecodeError:
                self._err(400, "Invalid JSON")
                return
            jid = data.get("id", "")
            with _jobs_lock:
                j = _jobs.get(jid)
            if not j:
                self._err(404, "Job not found")
                return
            if j["status"] == "running" and j["proc"]:
                j["proc"].terminate()
                j["status"] = "cancelled"
                j["ended_at"] = time.time()
            self._json({"ok": True})

        def _audio_list(self, params):
            """List audio files with optional filters."""
            audio = list(audio_data["audio"])
            if hc := params.get("has_transcription"):
                audio = [a for a in audio if a.get("has_transcription") == (hc == "true")]
            if q := params.get("search"):
                ql = q.lower()
                audio = [
                    a
                    for a in audio
                    if ql in a.get("name", "").lower() or ql in a.get("transcription", "").lower()
                ]
            by = params.get("sort", "name")
            rev = params.get("order", "asc") == "desc"
            key_fns = {
                "size": lambda a: a.get("size_bytes", 0),
                "duration": lambda a: a.get("duration", 0),
            }
            audio = sorted(
                audio,
                key=key_fns.get(by, lambda a: a.get("name", "")),
                reverse=rev,
            )
            self._json({"audio": audio, "total": len(audio)})

        def _audio_file(self, params):
            """Serve an audio file."""
            fp = params.get("path")
            if not fp:
                self._err(400, "Missing path")
                return
            p = Path(fp)
            if not p.exists():
                self._err(404, "Not found")
                return
            if not self._in_allowed(p):
                self._err(403, "Access denied")
                return
            data = p.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            self.wfile.write(data)

        def _audio_transcription(self):
            """Read or save transcription for an audio file."""
            if self.command == "GET":
                parsed = urllib.parse.urlparse(self.path)
                params = dict(urllib.parse.parse_qsl(parsed.query))
                fp = params.get("path")
                if not fp:
                    self._err(400, "Missing path")
                    return
                # Look up cached audio entry to get transcription (from metadata.csv or .txt)
                audio_p = Path(fp)
                txt_p = audio_p.with_suffix(".txt")
                cached = None
                for au in audio_data["audio"]:
                    if au.get("path") == str(audio_p):
                        cached = au
                        break
                if cached:
                    self._json(
                        {
                            "text": cached.get("transcription", ""),
                            "exists": cached.get("has_transcription", False),
                        }
                    )
                elif txt_p.exists():
                    self._json(
                        {
                            "text": txt_p.read_text(encoding="utf-8", errors="replace"),
                            "exists": True,
                        }
                    )
                else:
                    self._json({"text": "", "exists": False})
                return

            # POST — save transcription
            try:
                data = json.loads(self._body())
            except json.JSONDecodeError:
                self._err(400, "Invalid JSON")
                return
            fp = data.get("path", "")
            text = data.get("text", "")
            if not fp:
                self._err(400, "Missing path")
                return
            audio_p = Path(fp)
            txt_p = audio_p.with_suffix(".txt")
            if not self._in_allowed(txt_p):
                self._err(403, "Access denied")
                return
            if text.strip():
                txt_p.write_text(text, encoding="utf-8")
            elif txt_p.exists():
                txt_p.unlink()
            # Update cached audio entry
            for au in audio_data["audio"]:
                au_p = Path(au.get("path", ""))
                if au_p.stem == txt_p.stem and au_p.parent == txt_p.parent:
                    au["has_transcription"] = bool(text.strip())
                    au["transcription"] = text.strip()
                    break
            audio_data["stats"] = {
                "total": len(audio_data["audio"]),
                "total_size_mb": audio_data["stats"].get("total_size_mb", 0),
                "total_duration_s": audio_data["stats"].get("total_duration_s", 0),
                "transcriptions_found": sum(
                    1 for a in audio_data["audio"] if a.get("has_transcription")
                ),
                "transcriptions_missing": sum(
                    1 for a in audio_data["audio"] if not a.get("has_transcription")
                ),
            }
            self._json({"ok": True})

        def _audio_delete(self):
            """Delete audio files and their transcriptions."""
            try:
                data = json.loads(self._body())
            except json.JSONDecodeError:
                self._err(400, "Invalid JSON")
                return
            paths = data.get("paths", [])
            if not paths:
                self._err(400, "Missing paths")
                return
            deleted = []
            for ps in paths:
                p = Path(ps)
                if not self._in_allowed(p):
                    continue
                if p.exists() and p.suffix.lower().lstrip(".") in AUDIO_FORMATS:
                    p.unlink()
                    deleted.append(str(p))
                txt = p.with_suffix(".txt")
                if txt.exists():
                    txt.unlink()
                    deleted.append(str(txt))
            dead = set(deleted)
            audio_data["audio"] = [a for a in audio_data["audio"] if a.get("path") not in dead]
            audio_data["stats"]["total"] = len(audio_data["audio"])
            audio_data["stats"]["transcriptions_found"] = sum(
                1 for a in audio_data["audio"] if a.get("has_transcription")
            )
            audio_data["stats"]["transcriptions_missing"] = (
                audio_data["stats"]["total"] - audio_data["stats"]["transcriptions_found"]
            )
            self._json({"deleted": deleted, "count": len(deleted)})

    return Handler


# ── CLI entry point ───────────────────────────────────────────────────────────


def cmd_server(args):
    """Start the universal dataset management server."""
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Directory '{input_dir}' does not exist.")
        sys.exit(1)
    control_dir: Path | None = None
    if getattr(args, "control", None):
        control_dir = Path(args.control)
        if not control_dir.exists():
            print(f"Error: Control directory '{control_dir}' does not exist.")
            sys.exit(1)

    print(f"Input:      {input_dir}")
    if control_dir:
        print(f"Control:    {control_dir}")
    print(f"Recursive:  {args.recursive}")
    print(f"Duplicates: {args.duplicates}")
    # Find an available port
    port = args.port
    max_port_attempts = 10
    for attempt in range(max_port_attempts):
        try:
            handler = _make_handler(input_dir, control_dir, args.recursive, args.duplicates)
            server = HTTPServer(("0.0.0.0", port), handler)
            break
        except OSError as e:
            if attempt < max_port_attempts - 1 and e.errno == 48:  # Address in use
                port += 1
                print(f"Port {port - 1} in use, trying {port}...")
                continue
            raise

    print(f"Server:     http://localhost:{port}")
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
        help="Start the dataset management web dashboard (universal UI)",
    )
    p.add_argument("--input", "-i", required=True, help="Dataset directory to manage")
    p.add_argument(
        "--control",
        "-c",
        default="",
        help="Control images directory for pairs comparison mode (align workflow)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the web server (default: 8080)",
    )
    p.add_argument(
        "--recursive",
        "-R",
        action="store_true",
        help="Search directories recursively for images",
    )
    p.add_argument(
        "--duplicates",
        action="store_true",
        help="Pre-compute perceptual hashes for duplicate detection",
    )
    p.set_defaults(func=cmd_server)
