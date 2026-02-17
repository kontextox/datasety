"""Tests for the caption command."""

import json
import subprocess
import sys
from unittest.mock import MagicMock, patch

from PIL import Image

from datasety.caption import _caption_via_api, _image_to_data_url


def run_caption(*args):
    """Run datasety caption and return the result."""
    return subprocess.run(
        [sys.executable, "-m", "datasety", "caption", *args],
        capture_output=True, text=True,
    )


def make_image(path, width, height, color=(255, 0, 0)):
    """Create a solid color test image."""
    img = Image.new("RGB", (width, height), color=color)
    img.save(path)


# ── CLI argument tests (no models) ──


class TestCaptionCLI:
    """Test caption CLI argument parsing."""

    def test_help(self):
        result = run_caption("--help")
        assert result.returncode == 0
        assert "--trigger-word" in result.stdout
        assert "--florence-2-base" in result.stdout
        assert "--florence-2-large" in result.stdout
        assert "--num-beams" in result.stdout
        assert "--device" in result.stdout

    def test_help_shows_llm_api(self):
        result = run_caption("--help")
        assert "--llm-api" in result.stdout
        assert "--max-tokens" in result.stdout
        assert "--temperature" in result.stdout

    def test_missing_input_dir(self, tmp_path):
        result = run_caption(
            "-i", str(tmp_path / "nonexistent"),
            "-o", str(tmp_path / "out"),
        )
        assert result.returncode != 0

    def test_default_device_is_auto(self):
        result = run_caption("--help")
        assert "auto" in result.stdout

    def test_llm_api_requires_key(self, tmp_path):
        """--llm-api should error if OPENAI_API_KEY is not set."""
        in_dir = tmp_path / "input"
        out_dir = tmp_path / "output"
        in_dir.mkdir()
        make_image(in_dir / "test.jpg", 50, 50)

        env = {k: v for k, v in __import__("os").environ.items()}
        env.pop("OPENAI_API_KEY", None)
        env.pop("OPENAI_BASE_URL", None)
        env.pop("OPENAI_API_BASE", None)

        result = subprocess.run(
            [sys.executable, "-m", "datasety", "caption",
             "-i", str(in_dir), "-o", str(out_dir),
             "--llm-api", "--model", "test-model",
             "--prompt", "Describe this image."],
            capture_output=True, text=True, env=env,
        )
        assert result.returncode != 0
        assert "OPENAI_API_KEY" in result.stdout


# ── LLM API unit tests ──


class TestImageToDataUrl:
    def test_jpeg(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        make_image(img_path, 10, 10)
        url = _image_to_data_url(img_path)
        assert url.startswith("data:image/jpeg;base64,")

    def test_png(self, tmp_path):
        img_path = tmp_path / "test.png"
        make_image(img_path, 10, 10)
        url = _image_to_data_url(img_path)
        assert url.startswith("data:image/png;base64,")

    def test_roundtrip(self, tmp_path):
        """Base64 data should decode back to valid image bytes."""
        import base64
        img_path = tmp_path / "test.png"
        make_image(img_path, 10, 10)
        url = _image_to_data_url(img_path)
        # Extract base64 part
        b64_data = url.split(",", 1)[1]
        decoded = base64.b64decode(b64_data)
        assert decoded == img_path.read_bytes()


class TestCaptionViaApi:
    def test_sends_correct_request(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        make_image(img_path, 10, 10)

        response_data = {
            "choices": [{"message": {"content": "A red square image."}}]
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            result = _caption_via_api(
                img_path, "Describe this image.",
                "test-key", "https://api.example.com/v1",
                "test-model", 300, 0.3,
            )

        assert result == "A red square image."

        # Verify request structure
        call_args = mock_open.call_args
        req = call_args[0][0]
        body = json.loads(req.data.decode())
        assert body["model"] == "test-model"
        assert body["max_tokens"] == 300
        assert body["temperature"] == 0.3
        assert len(body["messages"]) == 1
        msg = body["messages"][0]
        assert msg["role"] == "user"
        assert len(msg["content"]) == 2
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][0]["text"] == "Describe this image."
        assert msg["content"][1]["type"] == "image_url"
        assert msg["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

        # Verify auth header
        assert req.get_header("Authorization") == "Bearer test-key"

    def test_custom_base_url(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        make_image(img_path, 10, 10)

        response_data = {
            "choices": [{"message": {"content": "Caption."}}]
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            _caption_via_api(
                img_path, "Describe.",
                "key", "https://openrouter.ai/api/v1",
                "x-ai/grok-4.1-fast", 300, 0.3,
            )

        req = mock_open.call_args[0][0]
        assert req.full_url == "https://openrouter.ai/api/v1/chat/completions"
        body = json.loads(req.data.decode())
        assert body["model"] == "x-ai/grok-4.1-fast"


class TestCmdCaptionLlmApi:
    """In-process tests for the LLM API captioning flow."""

    def _make_mock_urlopen(self, content="A test caption."):
        response_data = {
            "choices": [{"message": {"content": content}}]
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_single_image(self, tmp_path):
        from types import SimpleNamespace

        from datasety.caption import _cmd_caption_llm_api

        img_path = tmp_path / "photo.jpg"
        make_image(img_path, 50, 50)
        out_path = tmp_path / "photo.txt"

        args = SimpleNamespace(
            model="gpt-4o-mini", prompt="Describe.",
            trigger_word="", max_tokens=300, temperature=0.3,
        )
        mock_resp = self._make_mock_urlopen("This photo shows a red square.")

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                _cmd_caption_llm_api(
                    args, [img_path], out_path,
                    tmp_path, True,
                )

        assert out_path.exists()
        assert out_path.read_text() == "This photo shows a red square."

    def test_directory_mode(self, tmp_path):
        from types import SimpleNamespace

        from datasety.caption import _cmd_caption_llm_api

        in_dir = tmp_path / "input"
        out_dir = tmp_path / "output"
        in_dir.mkdir()
        out_dir.mkdir()

        for i in range(3):
            make_image(in_dir / f"img_{i}.jpg", 20, 20)

        image_files = sorted(in_dir.glob("*.jpg"))

        args = SimpleNamespace(
            model="x-ai/grok-4.1-fast",
            prompt="Describe in 3 sentences.",
            trigger_word="", max_tokens=300, temperature=0.3,
        )
        mock_resp = self._make_mock_urlopen("Caption for image.")

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                _cmd_caption_llm_api(
                    args, image_files, None, out_dir, False,
                )

        txt_files = list(out_dir.glob("*.txt"))
        assert len(txt_files) == 3

    def test_trigger_word(self, tmp_path):
        from types import SimpleNamespace

        from datasety.caption import _cmd_caption_llm_api

        img_path = tmp_path / "photo.jpg"
        make_image(img_path, 50, 50)
        out_path = tmp_path / "photo.txt"

        args = SimpleNamespace(
            model="gpt-4o-mini", prompt="Describe.",
            trigger_word="[photo]", max_tokens=300, temperature=0.3,
        )
        mock_resp = self._make_mock_urlopen("A scenic landscape.")

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                _cmd_caption_llm_api(
                    args, [img_path], out_path, tmp_path, True,
                )

        assert out_path.read_text() == "[photo] A scenic landscape."

    def test_base_url_env_vars(self, tmp_path):
        """OPENAI_BASE_URL / OPENAI_API_BASE should set the endpoint."""
        from types import SimpleNamespace

        from datasety.caption import _cmd_caption_llm_api

        img_path = tmp_path / "photo.jpg"
        make_image(img_path, 10, 10)
        out_path = tmp_path / "photo.txt"

        args = SimpleNamespace(
            model="test-model", prompt="Describe.",
            trigger_word="", max_tokens=300, temperature=0.3,
        )
        mock_resp = self._make_mock_urlopen("Caption.")

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            with patch.dict("os.environ", {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_BASE_URL": "https://openrouter.ai/api/v1",
            }):
                _cmd_caption_llm_api(
                    args, [img_path], out_path, tmp_path, True,
                )

        req = mock_open.call_args[0][0]
        assert "openrouter.ai" in req.full_url

    def test_openai_api_base_fallback(self, tmp_path):
        """OPENAI_API_BASE (legacy) should also work."""
        from types import SimpleNamespace

        from datasety.caption import _cmd_caption_llm_api

        img_path = tmp_path / "photo.jpg"
        make_image(img_path, 10, 10)
        out_path = tmp_path / "photo.txt"

        args = SimpleNamespace(
            model="test-model", prompt="Describe.",
            trigger_word="", max_tokens=300, temperature=0.3,
        )
        mock_resp = self._make_mock_urlopen("Caption.")

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            with patch.dict("os.environ", {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_API_BASE": "https://custom.api.com/v1",
            }, clear=False):
                # Make sure OPENAI_BASE_URL is not set
                import os
                os.environ.pop("OPENAI_BASE_URL", None)
                _cmd_caption_llm_api(
                    args, [img_path], out_path, tmp_path, True,
                )

        req = mock_open.call_args[0][0]
        assert "custom.api.com" in req.full_url

    def test_openai_model_env_var(self, tmp_path):
        """OPENAI_MODEL env var should be used when --model is not provided."""
        from types import SimpleNamespace

        from datasety.caption import _cmd_caption_llm_api

        img_path = tmp_path / "photo.jpg"
        make_image(img_path, 10, 10)
        out_path = tmp_path / "photo.txt"

        args = SimpleNamespace(
            model="", prompt="Describe.",
            trigger_word="", max_tokens=300, temperature=0.3,
        )
        mock_resp = self._make_mock_urlopen("Caption.")

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            with patch.dict("os.environ", {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_MODEL": "gpt-4o",
            }, clear=False):
                import os
                os.environ.pop("OPENAI_BASE_URL", None)
                os.environ.pop("OPENAI_API_BASE", None)
                _cmd_caption_llm_api(
                    args, [img_path], out_path, tmp_path, True,
                )

        req = mock_open.call_args[0][0]
        body = json.loads(req.data.decode())
        assert body["model"] == "gpt-4o"

    def test_model_arg_overrides_env_var(self, tmp_path):
        """Explicit --model should override OPENAI_MODEL env var."""
        from types import SimpleNamespace

        from datasety.caption import _cmd_caption_llm_api

        img_path = tmp_path / "photo.jpg"
        make_image(img_path, 10, 10)
        out_path = tmp_path / "photo.txt"

        args = SimpleNamespace(
            model="explicit-model", prompt="Describe.",
            trigger_word="", max_tokens=300, temperature=0.3,
        )
        mock_resp = self._make_mock_urlopen("Caption.")

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            with patch.dict("os.environ", {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_MODEL": "should-not-use",
            }, clear=False):
                _cmd_caption_llm_api(
                    args, [img_path], out_path, tmp_path, True,
                )

        req = mock_open.call_args[0][0]
        body = json.loads(req.data.decode())
        assert body["model"] == "explicit-model"
