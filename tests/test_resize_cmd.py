"""Tests for the resize command (end-to-end CLI tests)."""

import subprocess
import sys

from PIL import Image


def run_resize(*args):
    """Run datasety resize and return the result."""
    return subprocess.run(
        [sys.executable, "-m", "datasety", "resize", *args],
        capture_output=True,
        text=True,
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
            "-i",
            str(input_dir),
            "-o",
            str(output_dir),
            "-r",
            "1024x1024",
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
            "-i",
            str(input_dir),
            "-o",
            str(output_dir),
            "-r",
            "1024x1024",
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
            "-i",
            str(input_dir),
            "-o",
            str(output_dir),
            "-r",
            "512x512",
            "--output-format",
            "png",
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
            "-i",
            str(input_dir),
            "-o",
            str(output_dir),
            "-r",
            "512x512",
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
            "-i",
            str(input_dir),
            "-o",
            str(output_dir),
            "-r",
            "1024x1024",
            "--crop-position",
            "top",
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
            "-i",
            str(input_dir),
            "-o",
            str(output_dir),
            "-r",
            "768x1024",
        )
        assert result.returncode == 0
        with Image.open(output_dir / "001.jpg") as img:
            assert img.size == (768, 1024)

    def test_missing_input_dir(self, tmp_path):
        result = run_resize(
            "-i",
            str(tmp_path / "nonexistent"),
            "-o",
            str(tmp_path / "out"),
            "-r",
            "512x512",
        )
        assert result.returncode != 0

    def test_invalid_resolution(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        make_image(input_dir / "001.jpg", 2000, 2000)

        result = run_resize(
            "-i",
            str(input_dir),
            "-o",
            str(tmp_path / "out"),
            "-r",
            "invalid",
        )
        assert result.returncode != 0

    def test_no_images(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        (input_dir / "readme.txt").write_text("not an image")

        result = run_resize(
            "-i",
            str(input_dir),
            "-o",
            str(output_dir),
            "-r",
            "512x512",
        )
        assert result.returncode == 0
        assert "No images found" in result.stdout

    def test_megapixel_square(self, tmp_path):
        """--megapixel 0.5 --aspect-ratio 1:1 produces ~704x704."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        make_image(input_dir / "001.jpg", 2000, 2000)

        result = run_resize(
            "-i",
            str(input_dir),
            "-o",
            str(output_dir),
            "--megapixel",
            "0.5",
            "--aspect-ratio",
            "1:1",
        )
        assert result.returncode == 0
        with Image.open(output_dir / "001.jpg") as img:
            assert img.size == (704, 704)

    def test_megapixel_landscape(self, tmp_path):
        """--megapixel 1.0 --aspect-ratio 16:9 produces landscape."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        make_image(input_dir / "001.jpg", 2000, 2000)

        result = run_resize(
            "-i",
            str(input_dir),
            "-o",
            str(output_dir),
            "--megapixel",
            "1.0",
            "--aspect-ratio",
            "16:9",
        )
        assert result.returncode == 0
        with Image.open(output_dir / "001.jpg") as img:
            w, h = img.size
            assert w > h
            assert w % 8 == 0
            assert h % 8 == 0

    def test_megapixel_without_aspect_ratio(self, tmp_path):
        """--megapixel without --aspect-ratio preserves native aspect ratio."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "out"
        input_dir.mkdir()
        make_image(input_dir / "001.jpg", 2000, 1500)  # 4:3 aspect ratio

        result = run_resize(
            "-i",
            str(input_dir),
            "-o",
            str(output_dir),
            "--megapixel",
            "0.5",
        )
        assert result.returncode == 0
        assert "Processed: 1" in result.stdout
        with Image.open(output_dir / "001.jpg") as img:
            w, h = img.size
            assert w % 8 == 0
            assert h % 8 == 0
            # Should be close to 0.5 megapixels
            assert abs(w * h - 500_000) < 10_000

    def test_megapixel_without_aspect_ratio_mixed_orientations(self, tmp_path):
        """Each image keeps its own aspect ratio when --aspect-ratio is omitted."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "out"
        input_dir.mkdir()
        make_image(input_dir / "landscape.jpg", 3000, 2000)  # 3:2
        make_image(input_dir / "portrait.jpg", 2000, 3000)  # 2:3

        result = run_resize(
            "-i",
            str(input_dir),
            "-o",
            str(output_dir),
            "--megapixel",
            "1.0",
        )
        assert result.returncode == 0
        assert "Processed: 2" in result.stdout
        with Image.open(output_dir / "landscape.jpg") as img:
            w, h = img.size
            assert w > h  # still landscape
            assert w % 8 == 0 and h % 8 == 0
        with Image.open(output_dir / "portrait.jpg") as img:
            w, h = img.size
            assert h > w  # still portrait
            assert w % 8 == 0 and h % 8 == 0

    def test_megapixel_with_resolution_conflict(self, tmp_path):
        """--megapixel and --resolution together should error."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        make_image(input_dir / "001.jpg", 2000, 2000)

        result = run_resize(
            "-i",
            str(input_dir),
            "-o",
            str(tmp_path / "out"),
            "--megapixel",
            "0.5",
            "--aspect-ratio",
            "1:1",
            "-r",
            "512x512",
        )
        assert result.returncode != 0

    def test_no_resolution_or_megapixel(self, tmp_path):
        """Neither --resolution nor --megapixel should error."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        make_image(input_dir / "001.jpg", 2000, 2000)

        result = run_resize(
            "-i",
            str(input_dir),
            "-o",
            str(tmp_path / "out"),
        )
        assert result.returncode != 0

    def test_dry_run(self, tmp_path):
        """--dry-run should not write output files."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        make_image(input_dir / "001.jpg", 2000, 1500)

        result = run_resize(
            "-i",
            str(input_dir),
            "-o",
            str(output_dir),
            "-r",
            "1024x1024",
            "--dry-run",
        )
        assert result.returncode == 0
        assert "DRY RUN" in result.stdout
        assert "001.jpg" in result.stdout
        assert len(list(output_dir.glob("*.jpg"))) == 0

    def test_input_format_filter(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        make_image(input_dir / "001.jpg", 2000, 2000)
        make_image(input_dir / "002.png", 2000, 2000)

        result = run_resize(
            "-i",
            str(input_dir),
            "-o",
            str(output_dir),
            "-r",
            "512x512",
            "--input-format",
            "png",
        )
        assert result.returncode == 0
        assert "Processed: 1" in result.stdout
