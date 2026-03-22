"""Web server for comparing aligned control/target image pairs."""

import json
import mimetypes
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from datasety.common import get_image_files

IMAGE_FORMATS = ["jpg", "jpeg", "png", "webp", "bmp", "tiff"]


def _build_pairs(target_dir: Path, control_dir: Path, recursive: bool):
    """Build list of matched image pairs with optional caption files."""
    target_files = get_image_files(target_dir, IMAGE_FORMATS, recursive=recursive)
    control_files = get_image_files(control_dir, IMAGE_FORMATS, recursive=recursive)
    control_by_stem = {f.stem: f for f in control_files}

    pairs = []
    for tf in sorted(target_files, key=lambda p: p.stem):
        cf = control_by_stem.get(tf.stem)
        if cf is None:
            continue
        pairs.append({
            "stem": tf.stem,
            "target": str(tf),
            "control": str(cf),
            "target_caption": str(tf.with_suffix(".txt")),
            "control_caption": str(cf.with_suffix(".txt")),
            "has_target_caption": tf.with_suffix(".txt").exists(),
            "has_control_caption": cf.with_suffix(".txt").exists(),
        })
    return pairs


def _make_handler(target_dir: Path, control_dir: Path, recursive: bool):
    """Create a request handler class with the given directories."""

    class AlignHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            msg = format % args
            print(f"[server] {self.client_address[0]} {msg}")

        def _send_json(self, data, status=200):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_error(self, status, message):
            self._send_json({"error": message}, status)

        def _read_body(self):
            length = int(self.headers.get("Content-Length", 0))
            return self.rfile.read(length) if length else b""

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            params = dict(urllib.parse.parse_qsl(parsed.query))

            if path == "/":
                self._serve_html()
            elif path == "/api/pairs":
                pairs = _build_pairs(target_dir, control_dir, recursive)
                self._send_json(pairs)
            elif path == "/api/image":
                self._serve_image(params)
            elif path == "/api/caption":
                self._serve_caption(params)
            else:
                self._send_error(404, "Not found")

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path

            if path == "/api/caption":
                self._save_caption()
            else:
                self._send_error(404, "Not found")

        def do_DELETE(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            params = dict(urllib.parse.parse_qsl(parsed.query))

            if path == "/api/pair":
                self._delete_pair(params)
            else:
                self._send_error(404, "Not found")

        def _serve_html(self):
            html = _get_html()
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_image(self, params):
            file_path = params.get("path")
            if not file_path:
                self._send_error(400, "Missing path parameter")
                return
            p = Path(file_path)
            if not p.exists():
                self._send_error(404, "Image not found")
                return
            # Security: only serve files under target or control dirs
            try:
                p.resolve().relative_to(target_dir.resolve())
            except ValueError:
                try:
                    p.resolve().relative_to(control_dir.resolve())
                except ValueError:
                    self._send_error(403, "Access denied")
                    return
            mime = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
            data = p.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

        def _serve_caption(self, params):
            file_path = params.get("path")
            if not file_path:
                self._send_error(400, "Missing path parameter")
                return
            p = Path(file_path)
            if not p.exists():
                self._send_json({"text": "", "exists": False})
                return
            text = p.read_text(encoding="utf-8", errors="replace")
            self._send_json({"text": text, "exists": True})

        def _save_caption(self):
            body = self._read_body()
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                self._send_error(400, "Invalid JSON")
                return
            file_path = data.get("path")
            text = data.get("text", "")
            if not file_path:
                self._send_error(400, "Missing path")
                return
            p = Path(file_path)
            # Security: only allow writing .txt under target or control dirs
            if p.suffix != ".txt":
                self._send_error(403, "Only .txt files allowed")
                return
            try:
                p.resolve().relative_to(target_dir.resolve())
            except ValueError:
                try:
                    p.resolve().relative_to(control_dir.resolve())
                except ValueError:
                    self._send_error(403, "Access denied")
                    return
            if text.strip():
                p.write_text(text, encoding="utf-8")
            elif p.exists():
                p.unlink()
            self._send_json({"ok": True})

        def _delete_pair(self, params):
            stem = params.get("stem")
            if not stem:
                self._send_error(400, "Missing stem parameter")
                return
            deleted = []
            for d in [target_dir, control_dir]:
                for ext in IMAGE_FORMATS + ["txt"]:
                    candidate = d / f"{stem}.{ext}"
                    if candidate.exists():
                        candidate.unlink()
                        deleted.append(str(candidate))
            self._send_json({"deleted": deleted})

    return AlignHandler


def cmd_align_server(target_dir: Path, control_dir: Path, port: int, recursive: bool):
    """Start the align comparison web server."""
    if not target_dir.exists():
        print(f"Error: Target directory '{target_dir}' does not exist.")
        sys.exit(1)
    if not control_dir.exists():
        print(f"Error: Control directory '{control_dir}' does not exist.")
        sys.exit(1)

    pairs = _build_pairs(target_dir, control_dir, recursive)
    print(f"Target: {target_dir}")
    print(f"Control: {control_dir}")
    print(f"Matched pairs: {len(pairs)}")
    print(f"Server: http://localhost:{port}")
    print("Press Ctrl+C to stop.")

    handler = _make_handler(target_dir, control_dir, recursive)
    server = HTTPServer(("0.0.0.0", port), handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


def _get_html():
    return '''<!DOCTYPE html>
<html lang="en" data-theme="auto">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
<title>datasety align</title>
<style>
*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

:root, [data-theme="light"] {
  --bg: #f5f5f5;
  --bg2: #fff;
  --bg3: #e8e8e8;
  --fg: #1a1a1a;
  --fg2: #666;
  --border: #d0d0d0;
  --accent: #888;
  --danger: #a33;
  --danger-hover: #c44;
  --overlay: rgba(0,0,0,0.5);
  --modal-bg: #fff;
  --kbd-bg: #e8e8e8;
  --kbd-border: #ccc;
}

[data-theme="dark"] {
  --bg: #111;
  --bg2: #1a1a1a;
  --bg3: #252525;
  --fg: #e0e0e0;
  --fg2: #888;
  --border: #333;
  --accent: #777;
  --danger: #a44;
  --danger-hover: #c55;
  --overlay: rgba(0,0,0,0.65);
  --modal-bg: #1e1e1e;
  --kbd-bg: #333;
  --kbd-border: #555;
}

@media (prefers-color-scheme: dark) {
  [data-theme="auto"] {
    --bg: #111;
    --bg2: #1a1a1a;
    --bg3: #252525;
    --fg: #e0e0e0;
    --fg2: #888;
    --border: #333;
    --accent: #777;
    --danger: #a44;
    --danger-hover: #c55;
    --overlay: rgba(0,0,0,0.65);
    --modal-bg: #1e1e1e;
    --kbd-bg: #333;
    --kbd-border: #555;
  }
}

html, body { height: 100%; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  background: var(--bg);
  color: var(--fg);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  -webkit-user-select: none;
  user-select: none;
  padding: 8px;
}

@media (min-width: 601px) {
  body { padding: 16px; }
}

/* ── Header ── */
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  background: var(--bg2);
  border: 1px solid var(--border);
  flex-shrink: 0;
  min-height: 44px;
  gap: 8px;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
}

.header-title {
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.02em;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  min-width: 0;
}

.header-nav {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-shrink: 0;
}

.header-counter {
  font-size: 12px;
  color: var(--fg2);
  font-variant-numeric: tabular-nums;
  min-width: 3em;
  text-align: center;
}

/* ── Buttons ── */
button {
  font-family: inherit;
  font-size: 12px;
  font-weight: 500;
  background: var(--bg3);
  color: var(--fg);
  border: 1px solid var(--border);
  padding: 6px 12px;
  cursor: pointer;
  border-radius: 0;
  line-height: 1;
  transition: background 0.1s;
}

button:hover { background: var(--border); }
button:active { background: var(--accent); }
button:disabled { opacity: 0.35; cursor: default; }
button:disabled:hover { background: var(--bg3); }

button.danger { color: var(--danger); }
button.danger:hover { background: var(--danger-hover); color: #fff; }

button.icon-btn {
  padding: 6px 8px;
  font-size: 14px;
  line-height: 1;
}

/* ── Content area ── */
.content {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
  border: 1px solid var(--border);
  border-top: none;
}

/* ── Compare container ── */
.compare-wrap {
  flex: 1;
  position: relative;
  overflow: hidden;
  background: var(--bg);
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 0;
}

.compare-container {
  position: relative;
  max-width: 100%;
  max-height: 100%;
}

.compare-container img {
  display: block;
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}

.compare-target {
  display: block;
}

.compare-control-clip {
  position: absolute;
  top: 0;
  left: 0;
  width: 50%;
  height: 100%;
  overflow: hidden;
}

.compare-control-clip img {
  display: block;
  width: var(--full-w);
  max-width: none;
  height: var(--full-h);
}

.compare-divider {
  position: absolute;
  top: 0;
  left: 50%;
  width: 2px;
  height: 100%;
  background: var(--fg);
  opacity: 0.7;
  pointer-events: none;
  transform: translateX(-1px);
}

.compare-label {
  position: absolute;
  top: 8px;
  font-size: 11px;
  font-weight: 600;
  padding: 3px 8px;
  background: rgba(0,0,0,0.55);
  color: #fff;
  pointer-events: none;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.compare-label.left { left: 8px; }
.compare-label.right { right: 8px; }

/* ── Caption panels ── */
.caption-panel {
  display: flex;
  flex-direction: column;
  background: var(--bg2);
  min-width: 0;
  min-height: 0;
}

.caption-panel-label {
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--fg2);
  padding: 4px 8px;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}

.caption-panel textarea {
  flex: 1;
  border: none;
  outline: none;
  resize: none;
  padding: 8px;
  font-family: inherit;
  font-size: 13px;
  line-height: 1.5;
  background: transparent;
  color: var(--fg);
  min-height: 50px;
}

.caption-panel textarea::placeholder { color: var(--fg2); }

/* ── Save bar ── */
.save-bar {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  background: var(--bg2);
  border: 1px solid var(--border);
  border-top: none;
  flex-shrink: 0;
  justify-content: flex-end;
}

.save-bar .save-status {
  font-size: 11px;
  color: var(--fg2);
}

/* ── Narrow / default layout (< 1024px) ── */
/* Compare on top, captions in a row below */
.content {
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr auto;
}

.compare-wrap {
  grid-column: 1 / -1;
  grid-row: 1;
}

.caption-panel.control-panel {
  grid-column: 1;
  grid-row: 2;
  border-top: 1px solid var(--border);
}

.caption-panel.target-panel {
  grid-column: 2;
  grid-row: 2;
  border-top: 1px solid var(--border);
  border-left: 1px solid var(--border);
}

/* ── Mobile (< 600px): captions stack vertically ── */
@media (max-width: 600px) {
  .header { padding: 6px 8px; }
  .header-title { font-size: 12px; }
  button { padding: 8px 10px; font-size: 11px; }
  button.icon-btn { padding: 8px; }
  .compare-label { font-size: 10px; padding: 2px 6px; }

  .content {
    grid-template-columns: 1fr;
    grid-template-rows: 1fr auto auto;
  }
  .caption-panel.control-panel { grid-column: 1; grid-row: 2; }
  .caption-panel.target-panel {
    grid-column: 1;
    grid-row: 3;
    border-left: none;
  }
}

/* ── Wide desktop (>= 1024px): captions on sides ── */
@media (min-width: 1024px) {
  .content {
    grid-template-columns: 1fr auto 1fr;
    grid-template-rows: 1fr;
  }
  .caption-panel.control-panel {
    grid-column: 1;
    grid-row: 1;
    border-top: none;
    border-right: 1px solid var(--border);
  }
  .compare-wrap {
    grid-column: 2;
    grid-row: 1;
  }
  .caption-panel.target-panel {
    grid-column: 3;
    grid-row: 1;
    border-top: none;
    border-left: 1px solid var(--border);
  }
}

.hidden { display: none !important; }

/* ── Empty state ── */
.empty-state {
  display: flex;
  align-items: center;
  justify-content: center;
  flex: 1;
  color: var(--fg2);
  font-size: 14px;
}

/* ── Modal ── */
.modal-overlay {
  position: fixed;
  inset: 0;
  background: var(--overlay);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
  padding: 16px;
}

.modal {
  background: var(--modal-bg);
  border: 1px solid var(--border);
  max-width: 460px;
  width: 100%;
  max-height: 80vh;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
}

.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
}

.modal-header h2 {
  font-size: 14px;
  font-weight: 600;
}

.modal-body {
  padding: 16px;
}

.modal-body table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.modal-body td {
  padding: 6px 0;
  vertical-align: top;
}

.modal-body td:first-child {
  white-space: nowrap;
  padding-right: 16px;
  width: 1%;
}

kbd {
  display: inline-block;
  background: var(--kbd-bg);
  border: 1px solid var(--kbd-border);
  padding: 1px 6px;
  font-family: inherit;
  font-size: 11px;
  line-height: 1.6;
  min-width: 22px;
  text-align: center;
}
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <button id="btn-help" class="icon-btn" title="Help (?)" aria-label="Help">?</button>
    <button id="btn-theme" class="icon-btn" title="Toggle theme"
      aria-label="Toggle theme">&#9788;</button>
    <span class="header-title" id="title">datasety align</span>
  </div>
  <div class="header-nav">
    <button id="btn-prev" title="Previous">Prev</button>
    <span class="header-counter" id="counter">-</span>
    <button id="btn-next" title="Next">Next</button>
    <button id="btn-delete" class="danger" title="Delete this pair">Delete</button>
  </div>
</div>

<div class="content" id="content">
  <div class="caption-panel control-panel">
    <div class="caption-panel-label">Control</div>
    <textarea id="caption-control" placeholder="No caption file"></textarea>
  </div>

  <div class="compare-wrap" id="compare-wrap">
    <div class="compare-container" id="compare-container">
      <img class="compare-target" id="img-target" draggable="false">
      <div class="compare-control-clip" id="control-clip">
        <img id="img-control" draggable="false">
      </div>
      <div class="compare-divider" id="divider"></div>
      <div class="compare-label left">Control</div>
      <div class="compare-label right">Target</div>
    </div>
  </div>

  <div class="caption-panel target-panel">
    <div class="caption-panel-label">Target</div>
    <textarea id="caption-target" placeholder="No caption file"></textarea>
  </div>
</div>

<div class="save-bar" id="save-bar">
  <span class="save-status" id="save-status"></span>
  <button id="btn-save-captions">Save captions</button>
</div>

<div class="empty-state hidden" id="empty-state">No matched pairs found.</div>

<script>
(function() {
  let pairs = [];
  let idx = 0;

  const $ = id => document.getElementById(id);
  const imgTarget = $("img-target");
  const imgControl = $("img-control");
  const controlClip = $("control-clip");
  const divider = $("divider");
  const container = $("compare-container");
  const compareWrap = $("compare-wrap");
  const counter = $("counter");
  const title = $("title");
  const captionControl = $("caption-control");
  const captionTarget = $("caption-target");
  const contentEl = $("content");
  const saveBar = $("save-bar");
  const saveStatus = $("save-status");
  const emptyState = $("empty-state");

  // ── Theme ──
  const THEMES = ["light", "dark"];
  const THEME_ICONS = { auto: "\\u263C", light: "\\u2600", dark: "\\u263E" };
  let themeIdx = 0;

  function applyTheme() {
    const t = THEMES[themeIdx];
    document.documentElement.setAttribute("data-theme", t);
    $("btn-theme").textContent = THEME_ICONS[t];
    try { localStorage.setItem("datasety-theme", t); } catch {}
  }

  (function initTheme() {
    try {
      const saved = localStorage.getItem("datasety-theme");
      if (saved) { const i = THEMES.indexOf(saved); if (i >= 0) themeIdx = i; }
    } catch {}
    applyTheme();
  })();

  $("btn-theme").addEventListener("click", () => {
    themeIdx = (themeIdx + 1) % THEMES.length;
    applyTheme();
  });

  // ── Help modal ──
  function showHelp() {
    if (document.querySelector(".modal-overlay")) return;
    const overlay = document.createElement("div");
    overlay.className = "modal-overlay";
    overlay.innerHTML = `
      <div class="modal">
        <div class="modal-header">
          <h2>Keyboard shortcuts</h2>
          <button id="btn-close-help" class="icon-btn" aria-label="Close">&times;</button>
        </div>
        <div class="modal-body">
          <table>
            <tr><td><kbd>\\u2190</kbd> <kbd>\\u2192</kbd></td><td>Previous / Next pair</td></tr>
            <tr><td><kbd>[</kbd> <kbd>]</kbd></td><td>Move slider left / right</td></tr>
            <tr><td><kbd>Shift</kbd>+<kbd>\\u2190</kbd> <kbd>\\u2192</kbd></td>
            <td>Move slider in fine steps</td></tr>
            <tr><td><kbd>Delete</kbd></td><td>Delete current pair</td></tr>
            <tr><td><kbd>Ctrl</kbd>+<kbd>S</kbd></td><td>Save captions</td></tr>
            <tr><td><kbd>?</kbd></td><td>Toggle this help</td></tr>
          </table>
          <p style="margin-top:14px;font-size:12px;color:var(--fg2)">
            On mobile, swipe left/right on the image to navigate.
            Drag on the image to move the compare slider.
          </p>
        </div>
      </div>`;
    document.body.appendChild(overlay);
    overlay.querySelector("#btn-close-help").addEventListener("click", closeHelp);
    overlay.addEventListener("click", e => { if (e.target === overlay) closeHelp(); });
  }

  function closeHelp() {
    const o = document.querySelector(".modal-overlay");
    if (o) o.remove();
  }

  $("btn-help").addEventListener("click", () => {
    document.querySelector(".modal-overlay") ? closeHelp() : showHelp();
  });

  // ── Slider logic ──
  let dragging = false;
  let sliderX = 0.5;

  function setSlider(fraction) {
    sliderX = Math.max(0, Math.min(1, fraction));
    const pct = (sliderX * 100).toFixed(2) + "%";
    controlClip.style.width = pct;
    divider.style.left = pct;
  }

  function getPointerFraction(e) {
    const rect = container.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    return (clientX - rect.left) / rect.width;
  }

  container.addEventListener("mousedown", e => {
    dragging = true; setSlider(getPointerFraction(e));
  });
  container.addEventListener("touchstart", e => {
    dragging = true; setSlider(getPointerFraction(e));
  }, { passive: true });
  window.addEventListener("mousemove", e => {
    if (dragging) setSlider(getPointerFraction(e));
  });
  window.addEventListener("touchmove", e => {
    if (dragging) setSlider(getPointerFraction(e));
  }, { passive: true });
  window.addEventListener("mouseup", () => { dragging = false; });
  window.addEventListener("touchend", () => { dragging = false; });

  // ── Sync control image width to match target rendered size ──
  function syncSizes() {
    const nw = imgTarget.naturalWidth;
    const nh = imgTarget.naturalHeight;
    if (!nw || !nh) return;
    const wrapW = compareWrap.clientWidth;
    const wrapH = compareWrap.clientHeight;
    const scale = Math.min(wrapW / nw, wrapH / nh, 1);
    const w = Math.round(nw * scale);
    const h = Math.round(nh * scale);
    container.style.width = w + "px";
    container.style.height = h + "px";
    imgTarget.style.width = w + "px";
    imgTarget.style.height = h + "px";
    controlClip.style.setProperty("--full-w", w + "px");
    controlClip.style.setProperty("--full-h", h + "px");
    setSlider(sliderX);
  }

  imgTarget.addEventListener("load", syncSizes);
  window.addEventListener("resize", syncSizes);

  // ── Navigation ──
  function go(i) {
    if (pairs.length === 0) return;
    idx = ((i % pairs.length) + pairs.length) % pairs.length;
    const p = pairs[idx];
    const ts = Date.now();
    imgTarget.src = "/api/image?path=" + encodeURIComponent(p.target) + "&t=" + ts;
    imgControl.src = "/api/image?path=" + encodeURIComponent(p.control) + "&t=" + ts;
    counter.textContent = (idx + 1) + "/" + pairs.length;
    title.textContent = p.stem;
    loadCaptions(p);
    saveStatus.textContent = "";
  }

  $("btn-prev").addEventListener("click", () => go(idx - 1));
  $("btn-next").addEventListener("click", () => go(idx + 1));

  document.addEventListener("keydown", e => {
    // Allow Ctrl+S everywhere
    if ((e.ctrlKey || e.metaKey) && e.key === "s") {
      e.preventDefault();
      $("btn-save-captions").click();
      return;
    }
    // Skip shortcuts when typing in textarea (except Escape)
    if (e.target.tagName === "TEXTAREA" && e.key !== "Escape") return;
    // Close modal on Escape
    if (e.key === "Escape") { closeHelp(); return; }
    if (e.key === "ArrowLeft" && e.shiftKey) { e.preventDefault(); setSlider(sliderX - 0.02); }
    else if (e.key === "ArrowRight" && e.shiftKey) {
      e.preventDefault(); setSlider(sliderX + 0.02);
    }
    else if (e.key === "ArrowLeft") { e.preventDefault(); go(idx - 1); }
    else if (e.key === "ArrowRight") { e.preventDefault(); go(idx + 1); }
    else if (e.key === "Delete" || (e.key === "Backspace" && e.target.tagName !== "TEXTAREA")) {
      e.preventDefault(); deletePair();
    }
    else if (e.key === "[") { e.preventDefault(); setSlider(sliderX - 0.05); }
    else if (e.key === "]") { e.preventDefault(); setSlider(sliderX + 0.05); }
    else if (e.key === "?") {
      e.preventDefault();
      document.querySelector(".modal-overlay") ? closeHelp() : showHelp();
    }
  });

  // Swipe support
  let touchStartX = 0;
  compareWrap.addEventListener("touchstart", e => {
    touchStartX = e.touches[0].clientX;
  }, { passive: true });
  compareWrap.addEventListener("touchend", e => {
    const dx = e.changedTouches[0].clientX - touchStartX;
    if (!dragging && Math.abs(dx) > 60) {
      if (dx > 0) go(idx - 1);
      else go(idx + 1);
    }
  });

  // ── Captions ──
  async function loadCaptions(p) {
    const [cc, tc] = await Promise.all([
      fetch("/api/caption?path=" + encodeURIComponent(p.control_caption)).then(r => r.json()),
      fetch("/api/caption?path=" + encodeURIComponent(p.target_caption)).then(r => r.json()),
    ]);
    captionControl.value = cc.text;
    captionTarget.value = tc.text;
  }

  $("btn-save-captions").addEventListener("click", async () => {
    const p = pairs[idx];
    if (!p) return;
    saveStatus.textContent = "Saving...";
    try {
      await Promise.all([
        fetch("/api/caption", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ path: p.control_caption, text: captionControl.value }),
        }),
        fetch("/api/caption", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ path: p.target_caption, text: captionTarget.value }),
        }),
      ]);
      saveStatus.textContent = "Saved";
    } catch (e) {
      saveStatus.textContent = "Error saving";
    }
  });

  // ── Delete ──
  $("btn-delete").addEventListener("click", deletePair);

  async function deletePair() {
    const p = pairs[idx];
    if (!p) return;
    if (!confirm("Delete pair \\\"" + p.stem + "\\\" and associated caption files?")) return;
    await fetch("/api/pair?stem=" + encodeURIComponent(p.stem), { method: "DELETE" });
    pairs.splice(idx, 1);
    if (pairs.length === 0) {
      showEmpty();
    } else {
      go(Math.min(idx, pairs.length - 1));
    }
  }

  // ── Init ──
  function showEmpty() {
    contentEl.classList.add("hidden");
    saveBar.classList.add("hidden");
    emptyState.classList.remove("hidden");
    counter.textContent = "0/0";
    title.textContent = "datasety align";
    $("btn-prev").disabled = true;
    $("btn-next").disabled = true;
    $("btn-delete").disabled = true;
  }

  async function init() {
    const res = await fetch("/api/pairs");
    pairs = await res.json();
    if (pairs.length === 0) {
      showEmpty();
      return;
    }
    go(0);
  }

  init();
})();
</script>
</body>
</html>'''
