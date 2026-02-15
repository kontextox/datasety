"""Tests for the mask command."""

import subprocess
import sys

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


# ── Post-processing tests (no models, no numpy) ──


class TestMaskPostProcessing:
    """Test mask post-processing: padding, blur, invert using only Pillow."""

    def test_padding_expands_mask(self):
        """Padding (dilation) should expand white regions."""
        # Create mask with a small white dot in center
        mask_img = Image.new("L", (100, 100), 0)
        mask_img.putpixel((50, 50), 255)

        padded = mask_img.filter(ImageFilter.MaxFilter(size=5))

        # Count white pixels — should have expanded
        orig_white = sum(1 for p in mask_img.getdata() if p > 0)
        padded_white = sum(1 for p in padded.getdata() if p > 0)
        assert padded_white > orig_white

    def test_blur_softens_edges(self):
        """Blur should create intermediate values at edges."""
        mask_img = Image.new("L", (100, 100), 0)
        mask_img.paste(255, (30, 30, 70, 70))

        blurred = mask_img.filter(ImageFilter.GaussianBlur(radius=5))

        # Should have values between 0 and 255 at edges
        unique = set(blurred.getdata())
        assert len(unique) > 2  # More than just 0 and 255

    def test_invert_flips_mask(self):
        """Invert should swap black and white."""
        mask_img = Image.new("L", (100, 100), 0)
        mask_img.paste(255, (30, 30, 70, 70))

        # Simulate the invert logic from cmd_mask: 255 - array
        inverted = Image.eval(mask_img, lambda x: 255 - x)
        assert inverted.getpixel((50, 50)) == 0   # Was white, now black
        assert inverted.getpixel((0, 0)) == 255    # Was black, now white

    def test_padding_then_blur_order(self):
        """Padding should be applied before blur (matches cmd_mask logic)."""
        mask_img = Image.new("L", (100, 100), 0)
        mask_img.putpixel((50, 50), 255)

        # Pad first
        padded = mask_img.filter(ImageFilter.MaxFilter(size=5))
        # Then blur
        result = padded.filter(ImageFilter.GaussianBlur(radius=3))

        # Should have expanded and softened
        orig_white = sum(1 for p in mask_img.getdata() if p > 0)
        result_white = sum(1 for p in result.getdata() if p > 0)
        assert result_white > orig_white
        assert len(set(result.getdata())) > 2


class TestMaskCoverage:
    """Test mask coverage calculation."""

    def test_full_mask_100_percent(self):
        mask_img = Image.new("L", (100, 100), 255)
        pixel_count = sum(1 for p in mask_img.getdata() if p > 127)
        coverage = pixel_count / (100 * 100) * 100
        assert coverage == 100.0

    def test_empty_mask_0_percent(self):
        mask_img = Image.new("L", (100, 100), 0)
        pixel_count = sum(1 for p in mask_img.getdata() if p > 127)
        coverage = pixel_count / (100 * 100) * 100
        assert coverage == 0.0

    def test_half_mask_50_percent(self):
        mask_img = Image.new("L", (100, 100), 0)
        mask_img.paste(255, (0, 0, 100, 50))
        pixel_count = sum(1 for p in mask_img.getdata() if p > 127)
        coverage = pixel_count / (100 * 100) * 100
        assert coverage == 50.0


class TestMaskOutputNaming:
    """Test folder vs suffix naming modes."""

    def test_folder_naming(self, tmp_path):
        output_dir = tmp_path / "masks"
        output_dir.mkdir()

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
        import numpy as np

        input_dir = tmp_path / "images"
        output_dir = tmp_path / "masks"
        output_dir_inv = tmp_path / "masks_inv"
        input_dir.mkdir()
        make_image(input_dir / "001.jpg", 128, 128, color=(255, 0, 0))

        run_mask(
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-k", "red",
            "--model", "clipseg",
            "--device", "cpu",
        )
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
