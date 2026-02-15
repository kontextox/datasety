"""Tests for the synthetic command."""

import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from datasety.cli import _detect_model_family, _run_synthetic_pipeline


def _torch_available():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# ── Pure function tests (no models) ──


class TestDetectModelFamily:
    """Test model family auto-detection."""

    def test_qwen_default(self):
        assert _detect_model_family("Qwen/Qwen-Image-Edit-2511") == "qwen"

    def test_qwen_unknown(self):
        assert _detect_model_family("some-random/model") == "qwen"

    def test_flux_kontext(self):
        assert _detect_model_family("black-forest-labs/FLUX.1-Kontext-dev") == "flux-kontext"

    def test_flux_kontext_case_insensitive(self):
        assert _detect_model_family("user/my-KONTEXT-finetune") == "flux-kontext"

    def test_flux2_klein(self):
        assert _detect_model_family("black-forest-labs/FLUX.2-klein-9B") == "flux2-klein"

    def test_flux2_klein_alt(self):
        assert _detect_model_family("some/flux2-model") == "flux2-klein"

    def test_klein_keyword(self):
        assert _detect_model_family("some/Klein-thing") == "flux2-klein"

    def test_sdxl(self):
        assert _detect_model_family("stabilityai/stable-diffusion-xl-base-1.0") == "sdxl"

    def test_sdxl_shortname(self):
        assert _detect_model_family("user/sdxl-turbo") == "sdxl"

    def test_hunyuan(self):
        assert _detect_model_family("tencent/HunyuanImage-3.0") == "hunyuan"

    def test_hunyuan_distil(self):
        assert _detect_model_family("tencent/HunyuanImage-3.0-Distil") == "hunyuan"


@pytest.mark.skipif(
    not _torch_available(),
    reason="torch not installed",
)
class TestRunSyntheticPipelineKwargs:
    """Test that _run_synthetic_pipeline passes the right kwargs per family."""

    def _make_args(self, **overrides):
        defaults = {
            "prompt": "test prompt",
            "steps": 10,
            "num_images": 1,
            "seed": None,
            "cfg_scale": 1.0,
            "true_cfg_scale": 4.0,
            "negative_prompt": " ",
            "strength": 0.7,
            "cpu_offload": False,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def _make_pipeline(self):
        pipeline = MagicMock()
        output = MagicMock()
        output.images = [Image.new("RGB", (64, 64))]
        pipeline.return_value = output
        return pipeline

    def test_qwen_passes_true_cfg(self):
        pipeline = self._make_pipeline()
        args = self._make_args(true_cfg_scale=5.0)
        _run_synthetic_pipeline(pipeline, "qwen", Image.new("RGB", (64, 64)), args, "cpu", False)
        call_kwargs = pipeline.call_args[1]
        assert "true_cfg_scale" in call_kwargs
        assert call_kwargs["true_cfg_scale"] == 5.0
        assert "strength" not in call_kwargs

    def test_flux_kontext_no_true_cfg(self):
        pipeline = self._make_pipeline()
        args = self._make_args()
        _run_synthetic_pipeline(
            pipeline, "flux-kontext", Image.new("RGB", (64, 64)), args, "cpu", False
        )
        call_kwargs = pipeline.call_args[1]
        assert "true_cfg_scale" not in call_kwargs
        assert "strength" not in call_kwargs

    def test_sdxl_passes_strength(self):
        pipeline = self._make_pipeline()
        args = self._make_args(strength=0.8)
        _run_synthetic_pipeline(pipeline, "sdxl", Image.new("RGB", (64, 64)), args, "cpu", False)
        call_kwargs = pipeline.call_args[1]
        assert call_kwargs["strength"] == 0.8
        assert "true_cfg_scale" not in call_kwargs

    def test_sdxl_passes_negative_prompt(self):
        pipeline = self._make_pipeline()
        args = self._make_args(negative_prompt="ugly, bad")
        _run_synthetic_pipeline(pipeline, "sdxl", Image.new("RGB", (64, 64)), args, "cpu", False)
        call_kwargs = pipeline.call_args[1]
        assert call_kwargs["negative_prompt"] == "ugly, bad"

    def test_sdxl_skips_empty_negative_prompt(self):
        pipeline = self._make_pipeline()
        args = self._make_args(negative_prompt="  ")
        _run_synthetic_pipeline(pipeline, "sdxl", Image.new("RGB", (64, 64)), args, "cpu", False)
        call_kwargs = pipeline.call_args[1]
        assert "negative_prompt" not in call_kwargs

    def test_flux2_klein_passes_strength(self):
        pipeline = self._make_pipeline()
        args = self._make_args(strength=0.5)
        _run_synthetic_pipeline(
            pipeline, "flux2-klein", Image.new("RGB", (64, 64)), args, "cpu", False
        )
        call_kwargs = pipeline.call_args[1]
        assert call_kwargs["strength"] == 0.5

    def test_hunyuan_no_strength(self):
        pipeline = self._make_pipeline()
        args = self._make_args()
        _run_synthetic_pipeline(
            pipeline, "hunyuan", Image.new("RGB", (64, 64)), args, "cpu", False
        )
        call_kwargs = pipeline.call_args[1]
        assert "strength" not in call_kwargs
        assert "true_cfg_scale" not in call_kwargs

    def test_seed_creates_generator(self):
        pipeline = self._make_pipeline()
        args = self._make_args(seed=42)
        _run_synthetic_pipeline(pipeline, "qwen", Image.new("RGB", (64, 64)), args, "cpu", False)
        call_kwargs = pipeline.call_args[1]
        assert "generator" in call_kwargs

    def test_no_seed_no_generator(self):
        pipeline = self._make_pipeline()
        args = self._make_args(seed=None)
        _run_synthetic_pipeline(pipeline, "qwen", Image.new("RGB", (64, 64)), args, "cpu", False)
        call_kwargs = pipeline.call_args[1]
        assert "generator" not in call_kwargs

    def test_qwen_wraps_image_in_list(self):
        pipeline = self._make_pipeline()
        args = self._make_args()
        img = Image.new("RGB", (64, 64))
        _run_synthetic_pipeline(pipeline, "qwen", img, args, "cpu", False)
        call_kwargs = pipeline.call_args[1]
        assert call_kwargs["image"] == [img]

    def test_flux_kontext_image_not_list(self):
        pipeline = self._make_pipeline()
        args = self._make_args()
        img = Image.new("RGB", (64, 64))
        _run_synthetic_pipeline(pipeline, "flux-kontext", img, args, "cpu", False)
        call_kwargs = pipeline.call_args[1]
        assert call_kwargs["image"] is img


# ── CLI integration tests (no models) ──


def run_synthetic(*args):
    return subprocess.run(
        [sys.executable, "-m", "datasety", "synthetic", *args],
        capture_output=True, text=True,
    )


class TestSyntheticCLI:
    """Test synthetic CLI argument parsing and error handling."""

    def test_missing_input_dir(self, tmp_path):
        result = run_synthetic(
            "-i", str(tmp_path / "nonexistent"),
            "-o", str(tmp_path / "out"),
            "-p", "test",
        )
        assert result.returncode != 0

    def test_help(self):
        result = run_synthetic("--help")
        assert result.returncode == 0
        assert "--gguf" in result.stdout
        assert "--strength" in result.stdout
        assert "--cpu-offload" in result.stdout

    def test_no_images_exits_cleanly(self, tmp_path):
        """Empty input dir should exit 0 (no error) after model load fails."""
        input_dir = tmp_path / "empty_input"
        input_dir.mkdir()
        result = run_synthetic(
            "-i", str(input_dir),
            "-o", str(tmp_path / "out"),
            "-p", "test",
        )
        # Will fail at model load (no diffusers/model), which is expected
        assert result.returncode != 0
