"""Tests for the mask command."""

import subprocess
import sys

import numpy as np
import pytest
from PIL import Image, ImageFilter


def run_mask(*args):
    """Run datasety mask and return the result."""
    return subprocess.run(
        [sys.executable, "-m", "datasety", "mask", *args],
        capture_output=True, text=True,
    )


def make_image(path, width, height, color=(255, 0, 0)):
    """Create a solid color test image."""
    img = Image.new("RGB", (width, height), color=color)
    img.save(path)
    return path


# ── CLI argument tests (no models) ──


class TestMaskCLI:
    """Test mask CLI argument parsing."""

    def test_help(self):
        result = run_mask("--help")
        assert result.returncode == 0
        assert "--keywords" in result.stdout
        assert "--threshold" in result.stdout
        assert "--padding" in result.stdout
        assert "--blur" in result.stdout
        assert "--invert" in result.stdout
        assert "--naming" in result.stdout
        assert "--dry-run" in result.stdout

    def test_missing_input(self, tmp_path):
        result = run_mask(
            "-i", str(tmp_path / "nonexistent"),
            "-o", str(tmp_path / "out"),
            "-k", "face",
        )
        assert result.returncode != 0
        assert "does not exist" in result.stdout

    def test_model_choices(self):
        result = run_mask("--help")
        assert "sam3" in result.stdout
        assert "grounded-sam2" in result.stdout
        assert "clipseg" in result.stdout


# ── Post-processing tests (no models) ──


class TestMaskPostProcessing:
    """Test mask post-processing: padding, blur, invert."""

    def test_padding_expands_mask(self):
        """Padding (dilation) should expand white regions."""
        # Create mask with a small white dot in center
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 50] = 255

        mask_img = Image.fromarray(mask, mode="L")
        padded = mask_img.filter(ImageFilter.MaxFilter(size=5))
        padded_arr = np.array(padded)

        # The dot should have expanded
        assert np.sum(padded_arr > 0) > np.sum(mask > 0)

    def test_blur_softens_edges(self):
        """Blur should create intermediate values at edges."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 255

        mask_img = Image.fromarray(mask, mode="L")
        blurred = mask_img.filter(ImageFilter.GaussianBlur(radius=5))
        blurred_arr = np.array(blurred)

        # Should have values between 0 and 255 at edges
        unique = np.unique(blurred_arr)
        assert len(unique) > 2  # More than just 0 and 255

    def test_invert_flips_mask(self):
        """Invert should swap black and white."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 255

        inverted = 255 - mask
        assert inverted[50, 50] == 0  # Was white, now black
        assert inverted[0, 0] == 255  # Was black, now white

    def test_padding_then_blur_order(self):
        """Padding should be applied before blur (matches cmd_mask logic)."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 50] = 255

        # Pad first
        mask_img = Image.fromarray(mask, mode="L")
        padded = mask_img.filter(ImageFilter.MaxFilter(size=5))
        # Then blur
        result = padded.filter(ImageFilter.GaussianBlur(radius=3))
        result_arr = np.array(result)

        # Should have expanded and softened
        assert np.sum(result_arr > 0) > np.sum(mask > 0)
        assert len(np.unique(result_arr)) > 2


class TestMaskCoverage:
    """Test mask coverage calculation."""

    def test_full_mask_100_percent(self):
        mask = np.full((100, 100), 255, dtype=np.uint8)
        pixel_count = int(np.sum(mask > 127))
        coverage = pixel_count / (100 * 100) * 100
        assert coverage == 100.0

    def test_empty_mask_0_percent(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        pixel_count = int(np.sum(mask > 127))
        coverage = pixel_count / (100 * 100) * 100
        assert coverage == 0.0

    def test_half_mask_50_percent(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:50, :] = 255
        pixel_count = int(np.sum(mask > 127))
        coverage = pixel_count / (100 * 100) * 100
        assert coverage == 50.0


class TestMaskOutputNaming:
    """Test folder vs suffix naming modes."""

    def test_folder_naming(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "masks"
        input_dir.mkdir()
        output_dir.mkdir()

        # Simulate the naming logic from cmd_mask
        img_stem = "photo001"
        out_fmt = "png"
        out_path = output_dir / f"{img_stem}.{out_fmt}"
        assert out_path == output_dir / "photo001.png"

    def test_suffix_naming(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        img_stem = "photo001"
        out_fmt = "png"
        out_path = input_dir / f"{img_stem}_mask.{out_fmt}"
        assert out_path == input_dir / "photo001_mask.png"


class TestMaskKeywordParsing:
    """Test keyword parsing logic."""

    def test_single_keyword(self):
        keywords = [k.strip() for k in "face".split(",") if k.strip()]
        assert keywords == ["face"]

    def test_multiple_keywords(self):
        keywords = [k.strip() for k in "face,hair,hat".split(",") if k.strip()]
        assert keywords == ["face", "hair", "hat"]

    def test_keywords_with_spaces(self):
        keywords = [k.strip() for k in " face , hair , hat ".split(",") if k.strip()]
        assert keywords == ["face", "hair", "hat"]

    def test_empty_keywords(self):
        keywords = [k.strip() for k in ",,".split(",") if k.strip()]
        assert keywords == []

    def test_trailing_comma(self):
        keywords = [k.strip() for k in "face,hair,".split(",") if k.strip()]
        assert keywords == ["face", "hair"]


# ── GPU-required tests ──


@pytest.mark.gpu
class TestMaskWithCLIPSeg:
    """Tests that load CLIPSeg model. Run with: pytest -m gpu"""

    def test_clipseg_generates_masks(self, tmp_path):
        input_dir = tmp_path / "images"
        output_dir = tmp_path / "masks"
        input_dir.mkdir()

        # Create test images
        make_image(input_dir / "001.jpg", 256, 256)
        make_image(input_dir / "002.jpg", 256, 256)

        result = run_mask(
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-k", "red",
            "--model", "clipseg",
            "--device", "cpu",
        )
        assert result.returncode == 0
        masks = list(output_dir.glob("*.png"))
        assert len(masks) == 2

        # Masks should be same dimensions as input
        with Image.open(masks[0]) as mask:
            assert mask.size == (256, 256)
            assert mask.mode == "L"

    def test_clipseg_dry_run(self, tmp_path):
        input_dir = tmp_path / "images"
        output_dir = tmp_path / "masks"
        input_dir.mkdir()
        make_image(input_dir / "001.jpg", 128, 128)

        result = run_mask(
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-k", "red",
            "--model", "clipseg",
            "--device", "cpu",
            "--dry-run",
        )
        assert result.returncode == 0
        assert "DRY RUN" in result.stdout
        # Output dir should not be created or should be empty
        masks = list(output_dir.glob("*.png")) if output_dir.exists() else []
        assert len(masks) == 0

    def test_clipseg_suffix_naming(self, tmp_path):
        input_dir = tmp_path / "images"
        input_dir.mkdir()
        make_image(input_dir / "photo.jpg", 128, 128)

        result = run_mask(
            "-i", str(input_dir),
            "-o", str(input_dir),
            "-k", "red",
            "--model", "clipseg",
            "--device", "cpu",
            "--naming", "suffix",
        )
        assert result.returncode == 0
        assert (input_dir / "photo_mask.png").exists()

    def test_clipseg_invert(self, tmp_path):
        input_dir = tmp_path / "images"
        output_dir = tmp_path / "masks"
        output_dir_inv = tmp_path / "masks_inv"
        input_dir.mkdir()
        make_image(input_dir / "001.jpg", 128, 128, color=(255, 0, 0))

        # Normal
        run_mask(
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-k", "red",
            "--model", "clipseg",
            "--device", "cpu",
        )
        # Inverted
        run_mask(
            "-i", str(input_dir),
            "-o", str(output_dir_inv),
            "-k", "red",
            "--model", "clipseg",
            "--device", "cpu",
            "--invert",
        )

        normal = np.array(Image.open(output_dir / "001.png"))
        inverted = np.array(Image.open(output_dir_inv / "001.png"))
        # They should sum to 255 at every pixel
        assert np.allclose(normal.astype(int) + inverted.astype(int), 255)

    def test_clipseg_with_padding_and_blur(self, tmp_path):
        input_dir = tmp_path / "images"
        output_dir = tmp_path / "masks"
        input_dir.mkdir()
        make_image(input_dir / "001.jpg", 128, 128)

        result = run_mask(
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-k", "red",
            "--model", "clipseg",
            "--device", "cpu",
            "--padding", "3",
            "--blur", "2",
        )
        assert result.returncode == 0
        assert len(list(output_dir.glob("*.png"))) == 1


@pytest.mark.gpu
class TestMaskWithSAM3:
    """Tests that load SAM 3 model. Run with: pytest -m gpu"""

    def test_sam3_generates_masks(self, tmp_path):
        input_dir = tmp_path / "images"
        output_dir = tmp_path / "masks"
        input_dir.mkdir()
        make_image(input_dir / "001.jpg", 256, 256)

        result = run_mask(
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-k", "red object",
            "--model", "sam3",
            "--device", "cpu",
        )
        assert result.returncode == 0
        assert len(list(output_dir.glob("*.png"))) == 1
