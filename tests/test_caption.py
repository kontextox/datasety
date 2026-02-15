"""Tests for the caption command."""

import subprocess
import sys

import pytest
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
        assert "does not exist" in result.stdout

    def test_default_device_is_auto(self):
        result = run_caption("--help")
        assert "auto" in result.stdout


# ── GPU-required tests ──


@pytest.mark.gpu
class TestCaptionWithModel:
    """Tests that load Florence-2. Run with: pytest -m gpu"""

    def test_florence2_generates_captions(self, tmp_path):
        input_dir = tmp_path / "images"
        output_dir = tmp_path / "captions"
        input_dir.mkdir()
        output_dir.mkdir()

        make_image(input_dir / "001.jpg", 256, 256)
        make_image(input_dir / "002.jpg", 256, 256)

        result = run_caption(
            "-i", str(input_dir),
            "-o", str(output_dir),
            "--device", "cpu",
            "--florence-2-base",
        )
        assert result.returncode == 0
        captions = list(output_dir.glob("*.txt"))
        assert len(captions) == 2
        for txt in captions:
            content = txt.read_text()
            assert len(content) > 0

    def test_trigger_word_prepended(self, tmp_path):
        input_dir = tmp_path / "images"
        output_dir = tmp_path / "captions"
        input_dir.mkdir()
        output_dir.mkdir()

        make_image(input_dir / "001.jpg", 256, 256)

        result = run_caption(
            "-i", str(input_dir),
            "-o", str(output_dir),
            "--device", "cpu",
            "--florence-2-base",
            "--trigger-word", "sks person,",
        )
        assert result.returncode == 0
        caption = (output_dir / "001.txt").read_text()
        assert caption.startswith("sks person,")

    def test_greedy_decoding(self, tmp_path):
        input_dir = tmp_path / "images"
        output_dir = tmp_path / "captions"
        input_dir.mkdir()
        output_dir.mkdir()

        make_image(input_dir / "001.jpg", 256, 256)

        result = run_caption(
            "-i", str(input_dir),
            "-o", str(output_dir),
            "--device", "cpu",
            "--florence-2-base",
            "--num-beams", "1",
        )
        assert result.returncode == 0
        assert (output_dir / "001.txt").exists()
