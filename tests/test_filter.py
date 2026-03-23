"""Tests for the filter command."""

import subprocess
import sys

from PIL import Image


def run_filter(*args):
    """Run datasety filter and return the result."""
    return subprocess.run(
        [sys.executable, "-m", "datasety", "filter", *args],
        capture_output=True,
        text=True,
    )


def make_image(path, width=100, height=100, color=(255, 0, 0)):
    """Create a solid color test image."""
    img = Image.new("RGB", (width, height), color=color)
    img.save(path)
    return path


# ── CLI argument tests (no models) ──


class TestFilterCLI:
    """Test filter CLI argument parsing."""

    def test_help(self):
        result = run_filter("--help")
        assert result.returncode == 0
        assert "--query" in result.stdout
        assert "--labels" in result.stdout
        assert "--model" in result.stdout
        assert "--action" in result.stdout
        assert "--threshold" in result.stdout
        assert "--confirm" in result.stdout
        assert "--dry-run" in result.stdout
        assert "--recursive" in result.stdout
        assert "--preserve-structure" in result.stdout
        assert "--log" in result.stdout

    def test_model_choices(self):
        result = run_filter("--help")
        assert "clip" in result.stdout
        assert "nudenet" in result.stdout

    def test_action_choices(self):
        result = run_filter("--help")
        assert "move" in result.stdout
        assert "copy" in result.stdout
        assert "delete" in result.stdout
        assert "keep" in result.stdout

    def test_missing_input(self):
        result = run_filter("--query", "face")
        assert result.returncode != 0

    def test_missing_query_and_labels(self, tmp_path):
        inp = tmp_path / "in"
        inp.mkdir()
        make_image(inp / "1.jpg")
        result = run_filter("-i", str(inp), "-o", str(tmp_path / "out"))
        assert result.returncode != 0
        assert "required" in result.stderr or "required" in result.stdout

    def test_delete_requires_confirm(self, tmp_path):
        inp = tmp_path / "in"
        inp.mkdir()
        make_image(inp / "1.jpg")
        result = run_filter(
            "-i", str(inp), "--query", "face", "--action", "delete"
        )
        assert result.returncode != 0
        assert "confirm" in result.stdout.lower() or "confirm" in result.stderr.lower()

    def test_move_requires_output(self, tmp_path):
        inp = tmp_path / "in"
        inp.mkdir()
        make_image(inp / "1.jpg")
        result = run_filter(
            "-i", str(inp), "--query", "face", "--action", "move"
        )
        assert result.returncode != 0
        assert "output" in result.stdout.lower() or "output" in result.stderr.lower()

    def test_labels_requires_nudenet(self, tmp_path):
        inp = tmp_path / "in"
        inp.mkdir()
        make_image(inp / "1.jpg")
        result = run_filter(
            "-i", str(inp), "-o", str(tmp_path / "out"),
            "--labels", "FACE_FEMALE", "--model", "clip"
        )
        assert result.returncode != 0

    def test_query_rejects_nudenet(self, tmp_path):
        inp = tmp_path / "in"
        inp.mkdir()
        make_image(inp / "1.jpg")
        result = run_filter(
            "-i", str(inp), "-o", str(tmp_path / "out"),
            "--query", "face", "--model", "nudenet"
        )
        assert result.returncode != 0

    def test_nonexistent_input(self, tmp_path):
        result = run_filter(
            "-i", str(tmp_path / "nonexistent"),
            "-o", str(tmp_path / "out"),
            "--query", "face",
        )
        assert result.returncode != 0


# ── Companion file detection tests ──


class TestCompanionFiles:
    """Test companion file detection logic."""

    def test_finds_txt_companion(self, tmp_path):
        from datasety.filter import _find_companions

        img = tmp_path / "photo.jpg"
        txt = tmp_path / "photo.txt"
        img.touch()
        txt.write_text("a caption")
        companions = _find_companions(img)
        assert txt in companions

    def test_finds_multiple_companions(self, tmp_path):
        from datasety.filter import _find_companions

        img = tmp_path / "photo.jpg"
        txt = tmp_path / "photo.txt"
        json_f = tmp_path / "photo.json"
        img.touch()
        txt.write_text("caption")
        json_f.write_text("{}")
        companions = _find_companions(img)
        assert txt in companions
        assert json_f in companions

    def test_no_companions(self, tmp_path):
        from datasety.filter import _find_companions

        img = tmp_path / "photo.jpg"
        img.touch()
        companions = _find_companions(img)
        assert companions == []


# ── File action tests ──


class TestFileActions:
    """Test move/copy/delete actions."""

    def test_move(self, tmp_path):
        from datasety.filter import _act_on_file

        src = tmp_path / "in" / "photo.jpg"
        out = tmp_path / "out"
        src.parent.mkdir()
        out.mkdir()
        make_image(src)
        _act_on_file(src, "move", out, src.parent, False)
        assert (out / "photo.jpg").exists()
        assert not src.exists()

    def test_copy(self, tmp_path):
        from datasety.filter import _act_on_file

        src = tmp_path / "in" / "photo.jpg"
        out = tmp_path / "out"
        src.parent.mkdir()
        out.mkdir()
        make_image(src)
        _act_on_file(src, "copy", out, src.parent, False)
        assert (out / "photo.jpg").exists()
        assert src.exists()

    def test_delete(self, tmp_path):
        from datasety.filter import _act_on_file

        src = tmp_path / "photo.jpg"
        make_image(src)
        _act_on_file(src, "delete", None, None, False)
        assert not src.exists()

    def test_preserve_structure(self, tmp_path):
        from datasety.filter import _act_on_file

        root = tmp_path / "in"
        sub = root / "sub"
        sub.mkdir(parents=True)
        src = sub / "photo.jpg"
        out = tmp_path / "out"
        make_image(src)
        _act_on_file(src, "copy", out, root, True)
        assert (out / "sub" / "photo.jpg").exists()
