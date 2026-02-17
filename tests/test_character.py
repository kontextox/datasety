"""Tests for the character command."""

import base64
import io
import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from datasety.character import _build_user_prompt, _parse_prompts
from datasety.llm import (
    _create_llm_backend,
    _generate_image_via_api,
    _OllamaBackend,
    _OpenAIBackend,
)

# ── Prompt parsing tests ──


class TestParsePrompts:
    def test_basic_lines(self):
        text = "A portrait in a garden\nA headshot with sunset\nWalking on the beach"
        prompts = _parse_prompts(text)
        assert len(prompts) == 3
        assert prompts[0] == "A portrait in a garden"

    def test_numbered_lines(self):
        text = "1. A portrait in a garden\n2. A headshot with sunset\n3. Walking"
        prompts = _parse_prompts(text)
        assert len(prompts) == 3
        assert prompts[0] == "A portrait in a garden"

    def test_numbered_paren(self):
        text = "1) First prompt\n2) Second prompt"
        prompts = _parse_prompts(text)
        assert len(prompts) == 2
        assert prompts[0] == "First prompt"

    def test_empty_lines_skipped(self):
        text = "Line one\n\n\nLine two\n\n"
        prompts = _parse_prompts(text)
        assert len(prompts) == 2

    def test_quoted_prompts(self):
        text = '"A portrait"\n"Another scene"'
        prompts = _parse_prompts(text)
        assert prompts[0] == "A portrait"

    def test_empty_input(self):
        assert _parse_prompts("") == []
        assert _parse_prompts("   \n\n  ") == []


# ── User prompt building ──


class TestBuildUserPrompt:
    def test_basic(self):
        prompt = _build_user_prompt("a woman with red hair", "photorealistic", 5)
        assert "5" in prompt
        assert "red hair" in prompt
        assert "photorealistic" in prompt

    def test_no_description(self):
        prompt = _build_user_prompt("", "anime", 10)
        assert "10" in prompt
        assert "anime" in prompt

    def test_no_style(self):
        prompt = _build_user_prompt("a man", "", 3)
        assert "3" in prompt
        assert "a man" in prompt


# ── Backend factory ──


class TestLLMBackendFactory:
    def test_api_backend(self):
        args = SimpleNamespace(llm_api=True, llm_ollama="", llm_gguf="", llm_model="")
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            backend = _create_llm_backend(args)
        assert isinstance(backend, _OpenAIBackend)

    def test_ollama_backend(self):
        args = SimpleNamespace(llm_api=False, llm_ollama="llama3.2", llm_gguf="", llm_model="")
        backend = _create_llm_backend(args)
        assert isinstance(backend, _OllamaBackend)
        assert backend.model == "llama3.2"

    def test_no_backend(self):
        args = SimpleNamespace(llm_api=False, llm_ollama="", llm_gguf="", llm_model="")
        backend = _create_llm_backend(args)
        assert backend is None

    def test_api_requires_key(self):
        """OpenAI backend should exit if no API key is set."""
        with patch.dict("os.environ", {}, clear=True):
            # Remove key if present
            import os

            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("OPENAI_BASE_URL", None)
            os.environ.pop("OPENAI_MODEL", None)
            with pytest.raises(SystemExit):
                _OpenAIBackend()


# ── Mock HTTP backend tests ──


class TestOpenAIBackend:
    def test_generate(self):
        backend = _OpenAIBackend.__new__(_OpenAIBackend)
        backend.base_url = "https://api.example.com/v1"
        backend.model = "test-model"
        backend.api_key = "test-key"

        response_data = {"choices": [{"message": {"content": "prompt 1\nprompt 2"}}]}

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            result = backend.generate("system", "user")

        assert result == "prompt 1\nprompt 2"
        mock_open.assert_called_once()

        # Verify request format
        call_args = mock_open.call_args
        req = call_args[0][0]
        body = json.loads(req.data.decode())
        assert body["model"] == "test-model"
        assert len(body["messages"]) == 2


class TestOllamaBackend:
    def test_generate(self):
        backend = _OllamaBackend("llama3.2")

        response_data = {"message": {"content": "prompt 1\nprompt 2"}}

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            result = backend.generate("system", "user")

        assert result == "prompt 1\nprompt 2"

        # Verify Ollama API format
        call_args = mock_open.call_args
        req = call_args[0][0]
        body = json.loads(req.data.decode())
        assert body["model"] == "llama3.2"
        assert body["stream"] is False


# ── CLI tests ──


class TestCharacterCLI:
    def test_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "datasety", "character", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--reference" in result.stdout
        assert "--llm-api" in result.stdout
        assert "--dry-run" in result.stdout
        assert "--gguf" in result.stdout
        assert "--height" in result.stdout
        assert "--width" in result.stdout

    def test_image_api_in_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "datasety", "character", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--image-api" in result.stdout

    def test_no_llm_backend_errors(self):
        """Should error when no LLM backend and no --prompts-file."""
        result = subprocess.run(
            [sys.executable, "-m", "datasety", "character", "-o", "/tmp/test_char"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_reference_optional(self, tmp_path):
        """--reference should be optional when using --prompts-file."""
        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text("A portrait outdoors\n")

        out_dir = tmp_path / "output"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "datasety",
                "character",
                "-o",
                str(out_dir),
                "--prompts-file",
                str(prompts_file),
                "--prompts-only",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_prompts_only_with_file(self, tmp_path):
        """Should work with --prompts-file and --prompts-only."""
        ref_img = tmp_path / "face.png"
        from PIL import Image

        Image.new("RGB", (64, 64), (128, 128, 128)).save(ref_img)

        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text("A portrait outdoors\nA headshot indoors\nWalking in rain\n")

        out_dir = tmp_path / "output"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "datasety",
                "character",
                "--reference",
                str(ref_img),
                "-o",
                str(out_dir),
                "--prompts-file",
                str(prompts_file),
                "--prompts-only",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert (out_dir / "prompts.txt").exists()
        saved = (out_dir / "prompts.txt").read_text().splitlines()
        assert len(saved) == 3

    def test_dry_run_with_prompts_file(self, tmp_path):
        """--dry-run should preview without generating images."""
        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text("A portrait outdoors\nA headshot indoors\n")

        out_dir = tmp_path / "output"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "datasety",
                "character",
                "-o",
                str(out_dir),
                "--prompts-file",
                str(prompts_file),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "DRY RUN" in result.stdout
        # prompts.txt should NOT be written in dry-run mode
        assert not (out_dir / "prompts.txt").exists()


# ── Image generation API tests ──


def _make_tiny_png_b64():
    """Create a tiny PNG and return its base64 string."""
    img = Image.new("RGB", (4, 4), (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class TestGenerateImageViaApi:
    def test_structured_response(self):
        """Test parsing structured image response (images array)."""
        b64 = _make_tiny_png_b64()
        response_data = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "images": [{"image_url": {"url": f"data:image/png;base64,{b64}"}}],
                    }
                }
            ]
        }

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = _generate_image_via_api(
                "a cat", "test-key", "https://api.example.com/v1", "test-model"
            )

        assert isinstance(result, Image.Image)
        assert result.size == (4, 4)

    def test_markdown_fallback(self):
        """Test parsing markdown inline image fallback."""
        b64 = _make_tiny_png_b64()
        response_data = {
            "choices": [
                {"message": {"content": f"Here is your image: ![](data:image/png;base64,{b64})"}}
            ]
        }

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = _generate_image_via_api(
                "a cat", "test-key", "https://api.example.com/v1", "test-model"
            )

        assert isinstance(result, Image.Image)

    def test_sends_modalities(self):
        """Test that request includes modalities: [image]."""
        b64 = _make_tiny_png_b64()
        response_data = {
            "choices": [
                {"message": {"images": [{"image_url": {"url": f"data:image/png;base64,{b64}"}}]}}
            ]
        }

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            _generate_image_via_api("a cat", "test-key", "https://api.example.com/v1", "test-model")

        req = mock_open.call_args[0][0]
        body = json.loads(req.data.decode())
        assert body["modalities"] == ["image"]
        assert body["model"] == "test-model"

    def test_with_input_image(self):
        """Test image-to-image sends input as data URL."""
        b64 = _make_tiny_png_b64()
        response_data = {
            "choices": [
                {"message": {"images": [{"image_url": {"url": f"data:image/png;base64,{b64}"}}]}}
            ]
        }

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        input_img = Image.new("RGB", (8, 8), (0, 255, 0))

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            _generate_image_via_api(
                "edit this",
                "test-key",
                "https://api.example.com/v1",
                "test-model",
                input_image=input_img,
            )

        req = mock_open.call_args[0][0]
        body = json.loads(req.data.decode())
        content = body["messages"][0]["content"]
        # Should have image_url + text
        assert len(content) == 2
        assert content[0]["type"] == "image_url"
        assert content[1]["type"] == "text"

    def test_no_image_raises(self):
        """Test that missing image in response raises RuntimeError."""
        response_data = {"choices": [{"message": {"content": "No image here"}}]}

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="No image found"):
                _generate_image_via_api(
                    "a cat", "test-key", "https://api.example.com/v1", "test-model"
                )
