"""Tests for the align command."""

import subprocess
import sys

import pytest
from PIL import Image


@pytest.fixture
def setup_dirs(tmp_path):
    """Create target and control directories with test images."""
    target_dir = tmp_path / "target"
    control_dir = tmp_path / "control"
    target_dir.mkdir()
    control_dir.mkdir()
    return target_dir, control_dir


def make_image(path, width, height):
    """Create a solid color test image."""
    img = Image.new("RGB", (width, height), color=(255, 0, 0))
    img.save(path)


def run_align(*args):
    """Run datasety align and return the result."""
    return subprocess.run(
        [sys.executable, "-m", "datasety", "align", *args],
        capture_output=True, text=True,
    )


class TestAlignDryRun:
    """Test dry-run mode doesn't modify files."""

    def test_dry_run_no_changes(self, setup_dirs):
        target_dir, control_dir = setup_dirs
        make_image(target_dir / "001.jpg", 1050, 1580)
        make_image(control_dir / "001.jpg", 1050, 1580)

        result = run_align("-t", str(target_dir), "-c", str(control_dir), "--dry-run")
        assert result.returncode == 0
        assert "DRY RUN" in result.stdout
        assert "Fixed: 1" in result.stdout

        # File should be unchanged
        with Image.open(target_dir / "001.jpg") as img:
            assert img.size == (1050, 1580)


class TestAlignDimensions:
    """Test dimension alignment to multiple of 32."""

    def test_aligns_to_multiple_of_32(self, setup_dirs):
        target_dir, control_dir = setup_dirs
        make_image(target_dir / "001.jpg", 1050, 1580)
        make_image(control_dir / "001.jpg", 1050, 1580)

        result = run_align("-t", str(target_dir), "-c", str(control_dir))
        assert result.returncode == 0

        with Image.open(target_dir / "001.jpg") as img:
            w, h = img.size
            assert w % 32 == 0
            assert h % 32 == 0
            assert w == 1024
            assert h == 1568

        with Image.open(control_dir / "001.jpg") as img:
            assert img.size == (1024, 1568)

    def test_already_aligned(self, setup_dirs):
        target_dir, control_dir = setup_dirs
        make_image(target_dir / "001.jpg", 1024, 1536)
        make_image(control_dir / "001.jpg", 1024, 1536)

        result = run_align("-t", str(target_dir), "-c", str(control_dir))
        assert result.returncode == 0
        assert "Already OK: 1" in result.stdout

    def test_control_resized_to_match_target(self, setup_dirs):
        target_dir, control_dir = setup_dirs
        make_image(target_dir / "001.jpg", 1024, 1536)
        make_image(control_dir / "001.jpg", 800, 1200)

        result = run_align("-t", str(target_dir), "-c", str(control_dir))
        assert result.returncode == 0

        with Image.open(control_dir / "001.jpg") as img:
            assert img.size == (1024, 1536)

    def test_custom_multiple(self, setup_dirs):
        target_dir, control_dir = setup_dirs
        make_image(target_dir / "001.jpg", 100, 100)
        make_image(control_dir / "001.jpg", 100, 100)

        result = run_align(
            "-t", str(target_dir), "-c", str(control_dir), "--multiple-of", "64"
        )
        assert result.returncode == 0

        with Image.open(target_dir / "001.jpg") as img:
            w, h = img.size
            assert w % 64 == 0
            assert h % 64 == 0


class TestAlignPairing:
    """Test file pairing logic."""

    def test_missing_control(self, setup_dirs):
        target_dir, control_dir = setup_dirs
        make_image(target_dir / "001.jpg", 1024, 1536)
        # No matching control image

        result = run_align("-t", str(target_dir), "-c", str(control_dir))
        assert result.returncode == 0
        assert "Missing control: 1" in result.stdout

    def test_orphan_control(self, setup_dirs):
        target_dir, control_dir = setup_dirs
        make_image(target_dir / "001.jpg", 1024, 1536)
        make_image(control_dir / "001.jpg", 1024, 1536)
        make_image(control_dir / "999.jpg", 1024, 1536)

        result = run_align("-t", str(target_dir), "-c", str(control_dir))
        assert result.returncode == 0
        assert "ORPHAN" in result.stdout

    def test_cross_format_matching(self, setup_dirs):
        target_dir, control_dir = setup_dirs
        make_image(target_dir / "001.jpg", 1050, 1580)
        make_image(control_dir / "001.png", 1050, 1580)

        result = run_align("-t", str(target_dir), "-c", str(control_dir))
        assert result.returncode == 0
        assert "Fixed: 1" in result.stdout

    def test_multiple_pairs(self, setup_dirs):
        target_dir, control_dir = setup_dirs
        for i in range(5):
            make_image(target_dir / f"{i:03d}.jpg", 1050, 1580)
            make_image(control_dir / f"{i:03d}.jpg", 1050, 1580)

        result = run_align("-t", str(target_dir), "-c", str(control_dir))
        assert result.returncode == 0
        assert "Fixed: 5" in result.stdout


class TestAlignFormatConversion:
    """Test format conversion."""

    def test_convert_to_png(self, setup_dirs):
        target_dir, control_dir = setup_dirs
        make_image(target_dir / "001.jpg", 1024, 1536)
        make_image(control_dir / "001.jpg", 1024, 1536)

        result = run_align(
            "-t", str(target_dir), "-c", str(control_dir), "--output-format", "png"
        )
        assert result.returncode == 0
        assert (target_dir / "001.png").exists()
        assert (control_dir / "001.png").exists()
        assert not (target_dir / "001.jpg").exists()
        assert not (control_dir / "001.jpg").exists()
