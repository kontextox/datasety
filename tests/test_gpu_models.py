"""
GPU model tests — run on a server with 24-32 GB VRAM.

    pytest -m gpu -v                    # run all GPU tests
    pytest -m gpu -v -k caption         # only caption
    pytest -m gpu -v -k synthetic       # all synthetic families
    pytest -m gpu -v -k qwen            # only Qwen synthetic
    pytest -m gpu -v -k mask            # all mask models
    pytest -m gpu -v -k clipseg         # only CLIPSeg mask

VRAM budget (peak per test, bf16/fp16):

    Caption  | Florence-2-base       |  ~1 GB
    Synthetic| Qwen-Image-Edit       | ~32 GB  (auto sequential cpu-offload)
    Synthetic| FLUX.1-Kontext-dev    | ~33 GB  (auto cpu-offload, gated model)
    Synthetic| FLUX.2-klein-4B       |  ~8 GB
    Synthetic| FLUX.2-dev            | ~24 GB
    Synthetic| LongCat               | ~18 GB
    Synthetic| SDXL base             |  ~7 GB
    Synthetic| HunyuanImage          |  SKIPPED (needs 48 GB)
    Mask     | CLIPSeg               |  ~0.5 GB
    Mask     | SAM 2        |  ~6 GB
    Mask     | SAM 3                 |  ~5 GB  (gated model)

Gated models (FLUX Kontext, SAM 3) require:
    hf auth login
    # then accept the license on the model page
"""

import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

pytestmark = pytest.mark.gpu


# ── Helpers ──


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "datasety", *args],
        capture_output=True, text=True,
        timeout=600,
    )


def make_image(path, width, height, color=(255, 0, 0)):
    img = Image.new("RGB", (width, height), color=color)
    img.save(path)


def make_test_images(tmp_path, n=2, size=256):
    """Create input dir with n test images, return (input_dir, output_dir)."""
    input_dir = tmp_path / "images"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    for i in range(n):
        make_image(input_dir / f"{i:03d}.jpg", size, size, color=(i * 80, 50, 50))
    return input_dir, output_dir


# ═══════════════════════════════════════════════════════════════════════════════
#  CAPTION — Florence-2
# ═══════════════════════════════════════════════════════════════════════════════


class TestCaptionFlorence2:
    """Florence-2-base (~1 GB VRAM). Runs on CPU too."""

    def test_generates_captions_for_multiple_images(self, tmp_path):
        input_dir, output_dir = make_test_images(tmp_path, n=3)

        result = run_cli(
            "caption",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "--device", "auto",
            "--florence-2-base",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        captions = list(output_dir.glob("*.txt"))
        assert len(captions) == 3
        for txt in captions:
            assert len(txt.read_text().strip()) > 0

    def test_trigger_word_is_prepended(self, tmp_path):
        input_dir, output_dir = make_test_images(tmp_path, n=1)

        result = run_cli(
            "caption",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "--device", "auto",
            "--florence-2-base",
            "--trigger-word", "sks person,",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        caption = (output_dir / "000.txt").read_text()
        assert caption.startswith("sks person,")

    def test_greedy_decoding(self, tmp_path):
        input_dir, output_dir = make_test_images(tmp_path, n=1)

        result = run_cli(
            "caption",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "--device", "auto",
            "--florence-2-base",
            "--num-beams", "1",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert (output_dir / "000.txt").exists()

    def test_detailed_caption_prompt(self, tmp_path):
        input_dir, output_dir = make_test_images(tmp_path, n=1)

        result = run_cli(
            "caption",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "--device", "auto",
            "--florence-2-base",
            "--prompt", "<DETAILED_CAPTION>",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert len((output_dir / "000.txt").read_text().strip()) > 0


# ═══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC — Qwen (~32 GB VRAM)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSyntheticQwen:
    """Qwen/Qwen-Image-Edit-2511 (~32 GB bf16, auto sequential cpu-offload)."""

    def test_qwen_generates_image(self, tmp_path):
        input_dir, output_dir = make_test_images(tmp_path, n=1)

        result = run_cli(
            "synthetic",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-p", "add a small red hat",
            "--model", "Qwen/Qwen-Image-Edit-2511",
            "--device", "auto",
            "--steps", "2",
            "--seed", "42",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "family: qwen" in result.stdout
        outputs = list(output_dir.glob("*.png"))
        assert len(outputs) == 1

    def test_qwen_output_format_jpg(self, tmp_path):
        input_dir, output_dir = make_test_images(tmp_path, n=1)

        result = run_cli(
            "synthetic",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-p", "make it brighter",
            "--model", "Qwen/Qwen-Image-Edit-2511",
            "--device", "auto",
            "--steps", "2",
            "--seed", "42",
            "--output-format", "jpg",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(list(output_dir.glob("*.jpg"))) == 1

    def test_qwen_multiple_outputs(self, tmp_path):
        input_dir, output_dir = make_test_images(tmp_path, n=1)

        result = run_cli(
            "synthetic",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-p", "add sunglasses",
            "--model", "Qwen/Qwen-Image-Edit-2511",
            "--device", "auto",
            "--steps", "2",
            "--seed", "42",
            "--num-images", "2",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        outputs = list(output_dir.glob("*.png"))
        assert len(outputs) == 2

    def test_qwen_true_cfg_reported(self, tmp_path):
        input_dir, output_dir = make_test_images(tmp_path, n=1)

        result = run_cli(
            "synthetic",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-p", "test",
            "--model", "Qwen/Qwen-Image-Edit-2511",
            "--device", "auto",
            "--steps", "2",
            "--true-cfg-scale", "3.0",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "True CFG: 3.0" in result.stdout


# ═══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC — FLUX Kontext (~33 GB VRAM, gated model)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSyntheticFluxKontext:
    """black-forest-labs/FLUX.1-Kontext-dev (~33 GB bf16, auto cpu-offload).

    This model is GATED — you must accept the license at
    https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
    and run: hf auth login
    """

    def test_flux_kontext_generates_image(self, tmp_path):
        input_dir, output_dir = make_test_images(tmp_path, n=1)

        result = run_cli(
            "synthetic",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-p", "add a blue scarf",
            "--model", "black-forest-labs/FLUX.1-Kontext-dev",
            "--device", "auto",
            "--steps", "2",
            "--cfg-scale", "2.5",
            "--seed", "42",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "family: flux-kontext" in result.stdout
        assert len(list(output_dir.glob("*.png"))) == 1

    def test_flux_kontext_no_true_cfg_in_output(self, tmp_path):
        input_dir, output_dir = make_test_images(tmp_path, n=1)

        result = run_cli(
            "synthetic",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-p", "test",
            "--model", "black-forest-labs/FLUX.1-Kontext-dev",
            "--device", "auto",
            "--steps", "2",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        # true_cfg_scale is Qwen-only, should not be printed
        assert "True CFG" not in result.stdout


# ═══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC — FLUX.2 klein 4B (~8 GB VRAM)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSyntheticFlux2Klein:
    """black-forest-labs/FLUX.2-klein-4B (~8 GB bf16)."""

    def test_flux2_klein_generates_image(self, tmp_path):
        input_dir, output_dir = make_test_images(tmp_path, n=1)

        result = run_cli(
            "synthetic",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-p", "a painting in watercolor style",
            "--model", "black-forest-labs/FLUX.2-klein-4B",
            "--device", "auto",
            "--steps", "2",
            "--strength", "0.6",
            "--seed", "42",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "family: flux2-klein" in result.stdout
        assert len(list(output_dir.glob("*.png"))) == 1

    def test_flux2_klein_different_strength(self, tmp_path):
        input_dir, output_dir = make_test_images(tmp_path, n=1)

        result = run_cli(
            "synthetic",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-p", "enhance colors",
            "--model", "black-forest-labs/FLUX.2-klein-4B",
            "--device", "auto",
            "--steps", "2",
            "--strength", "0.3",
            "--seed", "42",
        )
        assert result.returncode == 0, result.stdout + result.stderr


# ═══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC — SDXL (~7 GB VRAM)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSyntheticSDXL:
    """stabilityai/stable-diffusion-xl-base-1.0 (~7 GB fp16)."""

    def test_sdxl_generates_image(self, tmp_path):
        input_dir, output_dir = make_test_images(tmp_path, n=1)

        result = run_cli(
            "synthetic",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-p", "a photo of a landscape, high quality, 4k",
            "--model", "stabilityai/stable-diffusion-xl-base-1.0",
            "--device", "auto",
            "--steps", "2",
            "--cfg-scale", "7.5",
            "--strength", "0.7",
            "--seed", "42",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "family: sdxl" in result.stdout
        assert len(list(output_dir.glob("*.png"))) == 1

    def test_sdxl_with_negative_prompt(self, tmp_path):
        input_dir, output_dir = make_test_images(tmp_path, n=1)

        result = run_cli(
            "synthetic",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-p", "a beautiful scene",
            "--model", "stabilityai/stable-diffusion-xl-base-1.0",
            "--device", "auto",
            "--steps", "2",
            "--strength", "0.5",
            "--negative-prompt", "ugly, blurry, low quality",
            "--seed", "42",
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_sdxl_multiple_input_images(self, tmp_path):
        input_dir, output_dir = make_test_images(tmp_path, n=3)

        result = run_cli(
            "synthetic",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-p", "add warm lighting",
            "--model", "stabilityai/stable-diffusion-xl-base-1.0",
            "--device", "auto",
            "--steps", "2",
            "--strength", "0.5",
            "--seed", "42",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Processed: 3 images" in result.stdout
        assert len(list(output_dir.glob("*.png"))) == 3

    def test_sdxl_webp_output(self, tmp_path):
        input_dir, output_dir = make_test_images(tmp_path, n=1)

        result = run_cli(
            "synthetic",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-p", "test",
            "--model", "stabilityai/stable-diffusion-xl-base-1.0",
            "--device", "auto",
            "--steps", "2",
            "--strength", "0.5",
            "--output-format", "webp",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(list(output_dir.glob("*.webp"))) == 1


# ═══════════════════════════════════════════════════════════════════════════════
#  MASK — CLIPSeg (~0.5 GB VRAM)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMaskCLIPSeg:
    """CIDAS/clipseg-rd64-refined (~0.5 GB). Lightweight, runs on CPU too."""

    def test_clipseg_generates_masks(self, tmp_path):
        input_dir, output_dir = make_test_images(tmp_path, n=2)

        result = run_cli(
            "mask",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-k", "red",
            "--model", "clipseg",
            "--device", "auto",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        masks = list(output_dir.glob("*.png"))
        assert len(masks) == 2
        with Image.open(masks[0]) as m:
            assert m.size == (256, 256)
            assert m.mode == "L"

    def test_clipseg_multiple_keywords(self, tmp_path):
        input_dir, output_dir = make_test_images(tmp_path, n=1)

        result = run_cli(
            "mask",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-k", "red,dark,square",
            "--model", "clipseg",
            "--device", "auto",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "3 keywords" in result.stdout

    def test_clipseg_dry_run(self, tmp_path):
        input_dir, output_dir = make_test_images(tmp_path, n=1)

        result = run_cli(
            "mask",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-k", "red",
            "--model", "clipseg",
            "--device", "auto",
            "--dry-run",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "DRY RUN" in result.stdout
        masks = list(output_dir.glob("*.png"))
        assert len(masks) == 0

    def test_clipseg_suffix_naming(self, tmp_path):
        input_dir, output_dir = make_test_images(tmp_path, n=1)

        result = run_cli(
            "mask",
            "-i", str(input_dir),
            "-o", str(input_dir),
            "-k", "red",
            "--model", "clipseg",
            "--device", "auto",
            "--naming", "suffix",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert (input_dir / "000_mask.png").exists()

    def test_clipseg_invert(self, tmp_path):
        input_dir = tmp_path / "images"
        out_normal = tmp_path / "normal"
        out_inverted = tmp_path / "inverted"
        input_dir.mkdir()
        make_image(input_dir / "001.jpg", 128, 128, color=(255, 0, 0))

        run_cli(
            "mask", "-i", str(input_dir), "-o", str(out_normal),
            "-k", "red", "--model", "clipseg", "--device", "auto",
        )
        run_cli(
            "mask", "-i", str(input_dir), "-o", str(out_inverted),
            "-k", "red", "--model", "clipseg", "--device", "auto",
            "--invert",
        )

        normal = np.array(Image.open(out_normal / "001.png"))
        inverted = np.array(Image.open(out_inverted / "001.png"))
        assert np.allclose(normal.astype(int) + inverted.astype(int), 255)

    def test_clipseg_padding_and_blur(self, tmp_path):
        input_dir, output_dir = make_test_images(tmp_path, n=1)

        result = run_cli(
            "mask",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-k", "red",
            "--model", "clipseg",
            "--device", "auto",
            "--padding", "5",
            "--blur", "3",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(list(output_dir.glob("*.png"))) == 1

    def test_clipseg_threshold(self, tmp_path):
        input_dir = tmp_path / "images"
        out_low = tmp_path / "low"
        out_high = tmp_path / "high"
        input_dir.mkdir()
        make_image(input_dir / "001.jpg", 128, 128, color=(255, 0, 0))

        run_cli(
            "mask", "-i", str(input_dir), "-o", str(out_low),
            "-k", "red", "--model", "clipseg", "--device", "auto",
            "--threshold", "0.1",
        )
        run_cli(
            "mask", "-i", str(input_dir), "-o", str(out_high),
            "-k", "red", "--model", "clipseg", "--device", "auto",
            "--threshold", "0.9",
        )

        low_mask = np.array(Image.open(out_low / "001.png"))
        high_mask = np.array(Image.open(out_high / "001.png"))
        # Lower threshold = more white pixels
        assert np.sum(low_mask > 127) >= np.sum(high_mask > 127)


# ═══════════════════════════════════════════════════════════════════════════════
#  MASK — SAM 2 (~6 GB VRAM)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMaskSAM2:
    """IDEA-Research/grounding-dino-base + facebook/sam2-hiera-large (~6 GB)."""

    def test_sam2_generates_mask(self, tmp_path):
        input_dir, output_dir = make_test_images(tmp_path, n=1, size=512)

        result = run_cli(
            "mask",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-k", "colored square",
            "--model", "sam2",
            "--device", "auto",
            "--threshold", "0.2",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        masks = list(output_dir.glob("*.png"))
        assert len(masks) == 1
        with Image.open(masks[0]) as m:
            assert m.size == (512, 512)
            assert m.mode == "L"

    def test_sam2_multiple_images(self, tmp_path):
        input_dir, output_dir = make_test_images(tmp_path, n=3, size=256)

        result = run_cli(
            "mask",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-k", "object",
            "--model", "sam2",
            "--device", "auto",
            "--threshold", "0.2",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Processed: 3 images" in result.stdout


# ═══════════════════════════════════════════════════════════════════════════════
#  MASK — SAM 3 (~5 GB VRAM, gated model)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMaskSAM3:
    """facebook/sam3 (~5 GB). Gated — requires Meta access approval.

    Request access at: https://huggingface.co/facebook/sam3
    Then: hf auth login
    """

    def test_sam3_generates_mask(self, tmp_path):
        input_dir, output_dir = make_test_images(tmp_path, n=1, size=256)

        result = run_cli(
            "mask",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-k", "red object",
            "--model", "sam3",
            "--device", "auto",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        masks = list(output_dir.glob("*.png"))
        assert len(masks) == 1

    def test_sam3_multiple_keywords(self, tmp_path):
        input_dir, output_dir = make_test_images(tmp_path, n=1, size=256)

        result = run_cli(
            "mask",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-k", "red,dark",
            "--model", "sam3",
            "--device", "auto",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "2 keywords" in result.stdout
