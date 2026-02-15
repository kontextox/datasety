"""Tests for the resize command (end-to-end CLI tests)."""

import subprocess
import sys

from PIL import Image


def run_resize(*args):
    """Run datasety resize and return the result."""
    return subprocess.run(
        [sys.executable, "-m", "datasety", "resize", *args],
        capture_output=True, text=True,
    )


def make_image(path, width, height, color=(255, 0, 0)):
    img = Image.new("RGB", (width, height), color=color)
    img.save(path)


class TestResizeCommand:
    """Test resize command end-to-end."""

    def test_basic_resize(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        make_image(input_dir / "001.jpg", 2000, 1500)

        result = run_resize(
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-r", "1024x1024",
        )
        assert result.returncode == 0
        assert "Processed: 1" in result.stdout

        with Image.open(output_dir / "001.jpg") as img:
            assert img.size == (1024, 1024)

    def test_skip_small_images(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        make_image(input_dir / "tiny.jpg", 100, 100)

        result = run_resize(
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-r", "1024x1024",
        )
        assert result.returncode == 0
        assert "Skipped: 1" in result.stdout
        assert len(list(output_dir.glob("*"))) == 0

    def test_output_format_png(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        make_image(input_dir / "001.jpg", 2000, 2000)

        result = run_resize(
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-r", "512x512",
            "--output-format", "png",
        )
        assert result.returncode == 0
        assert (output_dir / "001.png").exists()

    def test_numbered_output(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        for i in range(3):
            make_image(input_dir / f"img_{i}.jpg", 2000, 2000)

        result = run_resize(
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-r", "512x512",
            "--output-name-numbers",
        )
        assert result.returncode == 0
        assert (output_dir / "1.jpg").exists()
        assert (output_dir / "2.jpg").exists()
        assert (output_dir / "3.jpg").exists()

    def test_crop_position_top(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        make_image(input_dir / "001.jpg", 1500, 3000)

        result = run_resize(
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-r", "1024x1024",
            "--crop-position", "top",
        )
        assert result.returncode == 0
        with Image.open(output_dir / "001.jpg") as img:
            assert img.size == (1024, 1024)

    def test_non_square_resolution(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        make_image(input_dir / "001.jpg", 2000, 1500)

        result = run_resize(
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-r", "768x1024",
        )
        assert result.returncode == 0
        with Image.open(output_dir / "001.jpg") as img:
            assert img.size == (768, 1024)

    def test_missing_input_dir(self, tmp_path):
        result = run_resize(
            "-i", str(tmp_path / "nonexistent"),
            "-o", str(tmp_path / "out"),
            "-r", "512x512",
        )
        assert result.returncode != 0

    def test_invalid_resolution(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        make_image(input_dir / "001.jpg", 2000, 2000)

        result = run_resize(
            "-i", str(input_dir),
            "-o", str(tmp_path / "out"),
            "-r", "invalid",
        )
        assert result.returncode != 0

    def test_no_images(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        (input_dir / "readme.txt").write_text("not an image")

        result = run_resize(
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-r", "512x512",
        )
        assert result.returncode == 0
        assert "No images found" in result.stdout

    def test_input_format_filter(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        make_image(input_dir / "001.jpg", 2000, 2000)
        make_image(input_dir / "002.png", 2000, 2000)

        result = run_resize(
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-r", "512x512",
            "--input-format", "png",
        )
        assert result.returncode == 0
        assert "Processed: 1" in result.stdout
