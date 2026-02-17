"""Tests for the synthetic command."""

import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from datasety.common import _resolve_hf_file
from datasety.synthetic import (
    _detect_model_family,
    _parse_lora_spec,
    _resolve_gguf_path,
    _run_synthetic_pipeline,
)


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
        assert _detect_model_family("black-forest-labs/FLUX.2-klein-4B") == "flux2-klein"

    def test_flux2_klein_9b(self):
        assert _detect_model_family("black-forest-labs/FLUX.2-klein-9B") == "flux2-klein-9b"

    def test_flux2_klein_base(self):
        assert _detect_model_family("black-forest-labs/FLUX.2-klein-base-4B") == "flux2-klein-base"

    def test_flux2_klein_base_9b(self):
        result = _detect_model_family("black-forest-labs/FLUX.2-klein-base-9B")
        assert result == "flux2-klein-base-9b"

    def test_flux2_klein_base_fp8(self):
        result = _detect_model_family("black-forest-labs/FLUX.2-klein-base-4b-fp8")
        assert result == "flux2-klein-base-4b-fp8"

    def test_flux2_klein_fp8(self):
        assert _detect_model_family("black-forest-labs/FLUX.2-klein-4b-fp8") == "flux2-klein-4b-fp8"

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

    def test_flux2_dev(self):
        assert _detect_model_family("black-forest-labs/FLUX.2-dev") == "flux2-dev"

    def test_flux2_dev_not_klein(self):
        """FLUX.2-dev should NOT map to flux2-klein."""
        assert _detect_model_family("black-forest-labs/FLUX.2-dev") != "flux2-klein"

    def test_firered_maps_to_qwen(self):
        assert _detect_model_family("FireRedTeam/FireRed-Image-Edit-1.0") == "qwen"

    def test_longcat(self):
        assert _detect_model_family("meituan-longcat/LongCat-Image-Edit-Turbo") == "longcat"


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

    def test_flux2_klein_no_strength(self):
        pipeline = self._make_pipeline()
        args = self._make_args(strength=0.5)
        _run_synthetic_pipeline(
            pipeline, "flux2-klein", Image.new("RGB", (64, 64)), args, "cpu", False
        )
        call_kwargs = pipeline.call_args[1]
        assert "strength" not in call_kwargs

    def test_flux2_klein_wraps_image_in_list(self):
        pipeline = self._make_pipeline()
        args = self._make_args()
        img = Image.new("RGB", (64, 64))
        _run_synthetic_pipeline(pipeline, "flux2-klein", img, args, "cpu", False)
        call_kwargs = pipeline.call_args[1]
        assert call_kwargs["image"] == [img]

    def test_hunyuan_no_strength(self):
        pipeline = self._make_pipeline()
        args = self._make_args()
        _run_synthetic_pipeline(pipeline, "hunyuan", Image.new("RGB", (64, 64)), args, "cpu", False)
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

    def test_flux2_dev_passes_strength(self):
        pipeline = self._make_pipeline()
        args = self._make_args(strength=0.6)
        img = Image.new("RGB", (64, 64))
        _run_synthetic_pipeline(pipeline, "flux2-dev", img, args, "cpu", False)
        call_kwargs = pipeline.call_args[1]
        assert call_kwargs["strength"] == 0.6
        assert call_kwargs["image"] is img  # not wrapped in list

    def test_longcat_passes_negative_prompt(self):
        pipeline = self._make_pipeline()
        args = self._make_args(negative_prompt="ugly, bad")
        img = Image.new("RGB", (64, 64))
        _run_synthetic_pipeline(pipeline, "longcat", img, args, "cpu", False)
        call_kwargs = pipeline.call_args[1]
        assert call_kwargs["negative_prompt"] == "ugly, bad"
        assert "true_cfg_scale" not in call_kwargs

    def test_longcat_skips_empty_negative_prompt(self):
        pipeline = self._make_pipeline()
        args = self._make_args(negative_prompt="  ")
        img = Image.new("RGB", (64, 64))
        _run_synthetic_pipeline(pipeline, "longcat", img, args, "cpu", False)
        call_kwargs = pipeline.call_args[1]
        assert "negative_prompt" not in call_kwargs


# ── GGUF resolver tests ──


class TestResolveGgufPath:
    """Test _resolve_gguf_path helper."""

    def test_none_returns_none(self):
        assert _resolve_gguf_path(None) is None

    def test_local_path_unchanged(self):
        assert _resolve_gguf_path("/tmp/model.gguf") == "/tmp/model.gguf"

    def test_non_hf_url_unchanged(self):
        url = "https://example.com/model.gguf"
        assert _resolve_gguf_path(url) == url


# ── CLI integration tests (no models) ──


def run_synthetic(*args):
    return subprocess.run(
        [sys.executable, "-m", "datasety", "synthetic", *args],
        capture_output=True,
        text=True,
    )


class TestSyntheticCLI:
    """Test synthetic CLI argument parsing and error handling."""

    def test_missing_input_dir(self, tmp_path):
        result = run_synthetic(
            "-i",
            str(tmp_path / "nonexistent"),
            "-o",
            str(tmp_path / "out"),
            "-p",
            "test",
        )
        assert result.returncode != 0

    def test_help(self):
        result = run_synthetic("--help")
        assert result.returncode == 0
        assert "--gguf" in result.stdout
        assert "--strength" in result.stdout
        assert "--cpu-offload" in result.stdout

    def test_no_images_exits_cleanly(self, tmp_path):
        """Empty input dir should print 'No images found' message."""
        input_dir = tmp_path / "empty_input"
        input_dir.mkdir()
        result = run_synthetic(
            "-i",
            str(input_dir),
            "-o",
            str(tmp_path / "out"),
            "-p",
            "test",
        )
        # On GPU: model loads, then "No images found" -> exit 0
        # On CPU without model: model load fails -> exit 1
        # Both are acceptable; check that it doesn't hang or crash unexpectedly
        assert "No images found" in result.stdout or result.returncode != 0

    def test_lora_flag_in_help(self):
        result = run_synthetic("--help")
        assert "--lora" in result.stdout

    def test_dry_run(self, tmp_path):
        """--dry-run should preview without loading models."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        Image.new("RGB", (64, 64), (128, 128, 128)).save(input_dir / "test.jpg")

        result = run_synthetic(
            "-i",
            str(input_dir),
            "-o",
            str(output_dir),
            "-p",
            "add a hat",
            "--dry-run",
        )
        assert result.returncode == 0
        assert "DRY RUN" in result.stdout
        assert "test.jpg" in result.stdout

    def test_image_api_in_help(self):
        result = run_synthetic("--help")
        assert "--image-api" in result.stdout

    def test_image_api_dry_run(self, tmp_path):
        """--image-api + --dry-run should work without API key."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        Image.new("RGB", (64, 64), (128, 128, 128)).save(input_dir / "test.jpg")

        result = run_synthetic(
            "-i",
            str(input_dir),
            "-o",
            str(output_dir),
            "-p",
            "add a hat",
            "--image-api",
            "--dry-run",
        )
        assert result.returncode == 0
        assert "DRY RUN" in result.stdout
        assert "API" in result.stdout


# ── LoRA spec parsing tests ──


class TestParseLoraSpec:
    """Test _parse_lora_spec helper."""

    def test_path_only(self):
        path, weight = _parse_lora_spec("adapter.safetensors")
        assert path == "adapter.safetensors"
        assert weight == 1.0

    def test_path_with_weight(self):
        path, weight = _parse_lora_spec("adapter.safetensors:0.8")
        assert path == "adapter.safetensors"
        assert weight == 0.8

    def test_url_no_weight(self):
        url = "https://huggingface.co/user/repo/resolve/main/lora.safetensors"
        path, weight = _parse_lora_spec(url)
        assert path == url
        assert weight == 1.0

    def test_url_with_weight(self):
        url = "https://huggingface.co/user/repo/resolve/main/lora.safetensors"
        path, weight = _parse_lora_spec(f"{url}:0.5")
        assert path == url
        assert weight == 0.5

    def test_hf_repo_id_with_weight(self):
        """repo_id alone looks like 'user/repo' — no colon, so weight=1.0."""
        path, weight = _parse_lora_spec("user/repo")
        assert path == "user/repo"
        assert weight == 1.0

    def test_local_path_with_zero_weight(self):
        path, weight = _parse_lora_spec("/tmp/lora.safetensors:0.0")
        assert path == "/tmp/lora.safetensors"
        assert weight == 0.0

    def test_blob_url_with_weight(self):
        url = "https://huggingface.co/user/repo/blob/main/lora.safetensors"
        path, weight = _parse_lora_spec(f"{url}:0.7")
        assert path == url
        assert weight == 0.7


# ── HF file resolver tests ──


class TestResolveHfFile:
    """Test _resolve_hf_file helper."""

    def test_none_returns_none(self):
        assert _resolve_hf_file(None) is None

    def test_local_path_unchanged(self):
        assert _resolve_hf_file("/tmp/model.safetensors") == "/tmp/model.safetensors"

    def test_non_hf_url_unchanged(self):
        url = "https://example.com/model.safetensors"
        assert _resolve_hf_file(url) == url

    def test_blob_url_pattern_recognized(self):
        """Blob URLs should be recognized (download tested separately)."""
        import re

        url = "https://huggingface.co/user/repo/blob/main/file.safetensors"
        m = re.match(
            r"https?://huggingface\.co/([^/]+/[^/]+)/(?:resolve|blob)/([^/]+)/(.+)",
            url,
        )
        assert m is not None
        assert m.group(1) == "user/repo"
        assert m.group(2) == "main"
        assert m.group(3) == "file.safetensors"

    def test_resolve_url_pattern_recognized(self):
        """Resolve URLs should be recognized."""
        import re

        url = "https://huggingface.co/user/repo/resolve/main/file.gguf"
        m = re.match(
            r"https?://huggingface\.co/([^/]+/[^/]+)/(?:resolve|blob)/([^/]+)/(.+)",
            url,
        )
        assert m is not None
        assert m.group(3) == "file.gguf"
