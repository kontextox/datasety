"""Tests for the degrade command."""

import subprocess
import sys

import pytest
from PIL import Image

from datasety.degrade import (
    DEGRADATION_TYPES,
    _degrade_blur,
    _degrade_color_bands,
    _degrade_jpeg,
    _degrade_lowres,
    _degrade_motion_blur,
    _degrade_noise,
    _degrade_oversharpen,
    _degrade_pixelate,
    _degrade_upscale_sim,
    apply_degradations,
)


@pytest.fixture
def sample_image():
    """Create a 100x100 test image with varied content."""
    img = Image.new("RGB", (100, 100))
    pixels = img.load()
    for x in range(100):
        for y in range(100):
            pixels[x, y] = (x * 2 % 256, y * 2 % 256, (x + y) % 256)
    return img


# ── Unit tests for each degradation function ──


class TestDegradationFunctions:
    def test_lowres(self, sample_image):
        result = _degrade_lowres(sample_image, 0.5)
        assert result.size == sample_image.size
        assert result.tobytes() != sample_image.tobytes()

    def test_lowres_full_intensity(self, sample_image):
        result = _degrade_lowres(sample_image, 1.0)
        assert result.size == sample_image.size

    def test_oversharpen(self, sample_image):
        result = _degrade_oversharpen(sample_image, 0.5)
        assert result.size == sample_image.size
        assert result.tobytes() != sample_image.tobytes()

    def test_noise(self, sample_image):
        import random

        random.seed(42)
        result = _degrade_noise(sample_image, 0.5)
        assert result.size == sample_image.size
        assert result.tobytes() != sample_image.tobytes()

    def test_blur(self, sample_image):
        result = _degrade_blur(sample_image, 0.5)
        assert result.size == sample_image.size
        assert result.tobytes() != sample_image.tobytes()

    def test_jpeg(self, sample_image):
        result = _degrade_jpeg(sample_image, 0.5)
        assert result.size == sample_image.size
        assert result.tobytes() != sample_image.tobytes()

    def test_motion_blur(self, sample_image):
        result = _degrade_motion_blur(sample_image, 0.5)
        assert result.size == sample_image.size
        assert result.tobytes() != sample_image.tobytes()

    def test_pixelate(self, sample_image):
        result = _degrade_pixelate(sample_image, 0.5)
        assert result.size == sample_image.size
        assert result.tobytes() != sample_image.tobytes()

    def test_color_bands(self, sample_image):
        result = _degrade_color_bands(sample_image, 0.5)
        assert result.size == sample_image.size
        assert result.tobytes() != sample_image.tobytes()

    def test_upscale_sim(self, sample_image):
        result = _degrade_upscale_sim(sample_image, 0.5)
        assert result.size == sample_image.size
        assert result.tobytes() != sample_image.tobytes()

    def test_all_types_at_zero_intensity(self, sample_image):
        """All degradation functions should work at intensity=0."""
        for name in DEGRADATION_TYPES:
            from datasety.degrade import _DEGRADATION_FUNCS

            result = _DEGRADATION_FUNCS[name](sample_image, 0.0)
            assert result.size == sample_image.size

    def test_all_types_at_full_intensity(self, sample_image):
        """All degradation functions should work at intensity=1."""
        import random

        random.seed(42)
        for name in DEGRADATION_TYPES:
            from datasety.degrade import _DEGRADATION_FUNCS

            result = _DEGRADATION_FUNCS[name](sample_image, 1.0)
            assert result.size == sample_image.size


# ── apply_degradations tests ──


class TestDegradePipeline:
    def test_single_type(self, sample_image):
        result, steps = apply_degradations(sample_image, ["blur"], 0.5, chain=False)
        assert result.size == sample_image.size
        assert len(steps) == 1
        assert steps[0][0] == "blur"
        assert steps[0][1] == 0.5

    def test_chained(self, sample_image):
        import random

        random.seed(42)
        result, steps = apply_degradations(sample_image, ["blur", "noise", "jpeg"], 0.5, chain=True)
        assert result.size == sample_image.size
        assert result.tobytes() != sample_image.tobytes()
        assert len(steps) == 3
        assert [s[0] for s in steps] == ["blur", "noise", "jpeg"]

    def test_random_type(self, sample_image):
        import random

        random.seed(42)
        result, steps = apply_degradations(
            sample_image,
            ["random"],
            0.5,
            chain=False,
        )
        assert result.size == sample_image.size
        assert len(steps) == 1
        assert steps[0][0] in DEGRADATION_TYPES

    def test_chain_false_picks_one(self, sample_image):
        """With chain=False, only one degradation should be applied."""
        import random

        random.seed(42)
        result, steps = apply_degradations(sample_image, ["blur", "jpeg"], 0.5, chain=False)
        assert result.size == sample_image.size
        assert len(steps) == 1

    def test_unknown_type_raises(self, sample_image):
        with pytest.raises(ValueError, match="Unknown degradation type"):
            apply_degradations(sample_image, ["nonexistent"], 0.5)

    def test_does_not_mutate_input(self, sample_image):
        original_bytes = sample_image.tobytes()
        apply_degradations(sample_image, ["blur"], 0.5)
        assert sample_image.tobytes() == original_bytes

    def test_intensity_range_per_step(self, sample_image):
        """Each step in a chain should get independent random intensity."""
        import random

        random.seed(42)
        result, steps = apply_degradations(
            sample_image,
            ["blur", "jpeg", "noise"],
            chain=True,
            intensity_range=(0.2, 0.8),
        )
        assert len(steps) == 3
        intensities = [s[1] for s in steps]
        # Each intensity should be within range
        for i in intensities:
            assert 0.2 <= i <= 0.8
        # With random seed, intensities should differ from each other
        assert len(set(f"{i:.4f}" for i in intensities)) > 1

    def test_steps_return_format(self, sample_image):
        """Steps should be list of (name, intensity) tuples."""
        result, steps = apply_degradations(
            sample_image,
            ["blur", "jpeg"],
            0.7,
            chain=True,
        )
        for name, intensity in steps:
            assert isinstance(name, str)
            assert isinstance(intensity, float)


# ── Intensity range tests ──


class TestIntensityRange:
    def test_intensity_clamped_low(self, sample_image):
        """Intensity at 0 should still produce a valid image."""
        result, steps = apply_degradations(sample_image, ["blur"], 0.0)
        assert result.size == sample_image.size

    def test_intensity_clamped_high(self, sample_image):
        """Intensity at 1 should still produce a valid image."""
        import random

        random.seed(42)
        result, steps = apply_degradations(sample_image, ["noise"], 1.0)
        assert result.size == sample_image.size


# ── CLI tests ──


class TestDegradeCLI:
    def test_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "datasety", "degrade", "--help"], capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "degrade" in result.stdout.lower()
        assert "--type" in result.stdout
        assert "--intensity" in result.stdout

    def test_missing_input(self):
        result = subprocess.run(
            [sys.executable, "-m", "datasety", "degrade"], capture_output=True, text=True
        )
        assert result.returncode != 0

    def test_single_image(self, tmp_path):
        # Create a test image
        img = Image.new("RGB", (100, 100), (128, 128, 128))
        in_path = tmp_path / "test.png"
        img.save(in_path)

        out_path = tmp_path / "degraded.png"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "datasety",
                "degrade",
                "--input-image",
                str(in_path),
                "--output-image",
                str(out_path),
                "--type",
                "blur",
                "--intensity",
                "0.5",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert out_path.exists()

    def test_directory_mode(self, tmp_path):
        in_dir = tmp_path / "input"
        out_dir = tmp_path / "output"
        in_dir.mkdir()

        for i in range(3):
            Image.new("RGB", (50, 50), (i * 50, i * 50, i * 50)).save(in_dir / f"img_{i}.png")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "datasety",
                "degrade",
                "-i",
                str(in_dir),
                "-o",
                str(out_dir),
                "--type",
                "jpeg",
                "--intensity",
                "0.5",
                "--seed",
                "42",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert out_dir.exists()
        assert len(list(out_dir.glob("*.png"))) == 3

    def test_paired_mode(self, tmp_path):
        in_dir = tmp_path / "input"
        out_dir = tmp_path / "output"
        in_dir.mkdir()

        Image.new("RGB", (50, 50), (100, 100, 100)).save(in_dir / "photo.png")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "datasety",
                "degrade",
                "-i",
                str(in_dir),
                "-o",
                str(out_dir),
                "--type",
                "blur",
                "--paired",
                "--seed",
                "42",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert (out_dir / "control").exists()
        assert (out_dir / "target").exists()
        assert (out_dir / "control" / "photo.png").exists()
        assert (out_dir / "target" / "photo.png").exists()

    def test_chain_mode(self, tmp_path):
        img = Image.new("RGB", (50, 50), (128, 128, 128))
        in_path = tmp_path / "test.png"
        img.save(in_path)
        out_path = tmp_path / "out.png"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "datasety",
                "degrade",
                "--input-image",
                str(in_path),
                "--output-image",
                str(out_path),
                "--type",
                "blur",
                "--type",
                "jpeg",
                "--chain",
                "--intensity",
                "0.5",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert out_path.exists()

    def test_num_variants(self, tmp_path):
        in_dir = tmp_path / "input"
        out_dir = tmp_path / "output"
        in_dir.mkdir()

        Image.new("RGB", (50, 50), (100, 100, 100)).save(in_dir / "photo.png")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "datasety",
                "degrade",
                "-i",
                str(in_dir),
                "-o",
                str(out_dir),
                "--type",
                "random",
                "--num-variants",
                "3",
                "--seed",
                "42",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        files = sorted(out_dir.glob("*.png"))
        assert len(files) == 3
        assert files[0].stem == "photo_1"
        assert files[1].stem == "photo_2"
        assert files[2].stem == "photo_3"

    def test_num_variants_paired(self, tmp_path):
        in_dir = tmp_path / "input"
        out_dir = tmp_path / "output"
        in_dir.mkdir()

        Image.new("RGB", (50, 50), (100, 100, 100)).save(in_dir / "face.png")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "datasety",
                "degrade",
                "-i",
                str(in_dir),
                "-o",
                str(out_dir),
                "--type",
                "random",
                "--type",
                "random",
                "--chain",
                "--num-variants",
                "3",
                "--paired",
                "--intensity-range",
                "0.3-0.8",
                "--seed",
                "42",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        ctrl = sorted((out_dir / "control").glob("*.png"))
        tgt = sorted((out_dir / "target").glob("*.png"))
        assert len(ctrl) == 3
        assert len(tgt) == 3
        # Each variant has a different degradation (files should differ)
        ctrl_bytes = [Image.open(f).tobytes() for f in ctrl]
        assert ctrl_bytes[0] != ctrl_bytes[1] or ctrl_bytes[1] != ctrl_bytes[2]
        # Target copies are all identical (same original)
        tgt_bytes = [Image.open(f).tobytes() for f in tgt]
        assert tgt_bytes[0] == tgt_bytes[1] == tgt_bytes[2]

    def test_log_output_single(self, tmp_path):
        """Single variant should show degradation steps inline."""
        img = Image.new("RGB", (50, 50), (128, 128, 128))
        in_path = tmp_path / "test.png"
        img.save(in_path)
        out_path = tmp_path / "out.png"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "datasety",
                "degrade",
                "--input-image",
                str(in_path),
                "--output-image",
                str(out_path),
                "--type",
                "blur",
                "--type",
                "jpeg",
                "--chain",
                "--intensity",
                "0.5",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        # Should show: [OK] test.png -> out.png (blur:0.50 > jpeg:0.50)
        assert "blur:0.50" in result.stdout
        assert "jpeg:0.50" in result.stdout
        assert ">" in result.stdout

    def test_log_output_variants(self, tmp_path):
        """Multiple variants should list each with its own steps."""
        in_dir = tmp_path / "input"
        out_dir = tmp_path / "output"
        in_dir.mkdir()

        Image.new("RGB", (50, 50), (100, 100, 100)).save(in_dir / "photo.png")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "datasety",
                "degrade",
                "-i",
                str(in_dir),
                "-o",
                str(out_dir),
                "--type",
                "random",
                "--type",
                "random",
                "--chain",
                "--num-variants",
                "3",
                "--intensity-range",
                "0.3-0.8",
                "--seed",
                "42",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        # Should show variant lines indented
        assert "photo_1.png" in result.stdout
        assert "photo_2.png" in result.stdout
        assert "photo_3.png" in result.stdout
        assert "3 variants" in result.stdout
        # Each line should have the step chain with >
        lines = [ln for ln in result.stdout.splitlines() if "photo_" in ln]
        assert len(lines) == 3
        for line in lines:
            assert ">" in line  # chained steps

    def test_intensity_range(self, tmp_path):
        in_dir = tmp_path / "input"
        out_dir = tmp_path / "output"
        in_dir.mkdir()

        for i in range(5):
            Image.new("RGB", (50, 50)).save(in_dir / f"img_{i}.png")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "datasety",
                "degrade",
                "-i",
                str(in_dir),
                "-o",
                str(out_dir),
                "--type",
                "blur",
                "--intensity-range",
                "0.2-0.8",
                "--seed",
                "42",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert len(list(out_dir.glob("*.png"))) == 5
