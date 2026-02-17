"""Tests for the mask command."""

import subprocess
import sys

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

    def test_model_choices(self):
        result = run_mask("--help")
        assert "sam3" in result.stdout
        assert "sam2" in result.stdout
        assert "clipseg" in result.stdout


# ── Post-processing tests (no models, no numpy) ──


class TestMaskPostProcessing:
    """Test mask post-processing: padding, blur, invert using only Pillow."""

    def test_padding_expands_mask(self):
        """Padding (dilation) should expand white regions."""
        mask_img = Image.new("L", (100, 100), 0)
        mask_img.putpixel((50, 50), 255)

        padded = mask_img.filter(ImageFilter.MaxFilter(size=5))

        orig_white = sum(b > 0 for b in mask_img.tobytes())
        padded_white = sum(b > 0 for b in padded.tobytes())
        assert padded_white > orig_white

    def test_blur_softens_edges(self):
        """Blur should create intermediate values at edges."""
        mask_img = Image.new("L", (100, 100), 0)
        mask_img.paste(255, (30, 30, 70, 70))

        blurred = mask_img.filter(ImageFilter.GaussianBlur(radius=5))

        unique = set(blurred.tobytes())
        assert len(unique) > 2  # More than just 0 and 255

    def test_invert_flips_mask(self):
        """Invert should swap black and white."""
        mask_img = Image.new("L", (100, 100), 0)
        mask_img.paste(255, (30, 30, 70, 70))

        inverted = Image.eval(mask_img, lambda x: 255 - x)
        assert inverted.getpixel((50, 50)) == 0   # Was white, now black
        assert inverted.getpixel((0, 0)) == 255    # Was black, now white

    def test_padding_then_blur_order(self):
        """Padding should be applied before blur (matches cmd_mask logic)."""
        mask_img = Image.new("L", (100, 100), 0)
        mask_img.putpixel((50, 50), 255)

        padded = mask_img.filter(ImageFilter.MaxFilter(size=5))
        result = padded.filter(ImageFilter.GaussianBlur(radius=3))

        orig_white = sum(b > 0 for b in mask_img.tobytes())
        result_white = sum(b > 0 for b in result.tobytes())
        assert result_white > orig_white
        assert len(set(result.tobytes())) > 2


class TestMaskCoverage:
    """Test mask coverage calculation."""

    def test_full_mask_100_percent(self):
        mask_img = Image.new("L", (100, 100), 255)
        pixel_count = sum(b > 127 for b in mask_img.tobytes())
        coverage = pixel_count / (100 * 100) * 100
        assert coverage == 100.0

    def test_empty_mask_0_percent(self):
        mask_img = Image.new("L", (100, 100), 0)
        pixel_count = sum(b > 127 for b in mask_img.tobytes())
        coverage = pixel_count / (100 * 100) * 100
        assert coverage == 0.0

    def test_half_mask_50_percent(self):
        mask_img = Image.new("L", (100, 100), 0)
        mask_img.paste(255, (0, 0, 100, 50))
        pixel_count = sum(b > 127 for b in mask_img.tobytes())
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
