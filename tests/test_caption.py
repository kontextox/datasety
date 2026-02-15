"""Tests for the caption command."""

import subprocess
import sys

from PIL import Image


def run_caption(*args):
    """Run datasety caption and return the result."""
    return subprocess.run(
        [sys.executable, "-m", "datasety", "caption", *args],
        capture_output=True, text=True,
    )


def make_image(path, width, height, color=(255, 0, 0)):
    """Create a solid color test image."""
    img = Image.new("RGB", (width, height), color=color)
    img.save(path)


# ── CLI argument tests (no models) ──


class TestCaptionCLI:
    """Test caption CLI argument parsing."""

    def test_help(self):
        result = run_caption("--help")
        assert result.returncode == 0
        assert "--trigger-word" in result.stdout
        assert "--florence-2-base" in result.stdout
        assert "--florence-2-large" in result.stdout
        assert "--num-beams" in result.stdout
        assert "--device" in result.stdout

    def test_missing_input_dir(self, tmp_path):
        result = run_caption(
            "-i", str(tmp_path / "nonexistent"),
            "-o", str(tmp_path / "out"),
        )
        assert result.returncode != 0

    def test_default_device_is_auto(self):
        result = run_caption("--help")
        assert "auto" in result.stdout
