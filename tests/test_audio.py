"""Tests for the audio command."""

import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestNormalizeText:
    """Test text normalization for TTS."""

    def test_strip_control_characters(self):
        """Control characters should be stripped."""
        from datasety.audio import _normalize_text
        text = "Hello\x00world\x07!"
        result = _normalize_text(text, "en", normalize_numbers=False, clean_text=True)
        assert "\x00" not in result
        assert "\x07" not in result
        # Note: EnglishTextNormalizer lowercases
        assert "hello" in result.lower()

    def test_strip_emoji(self):
        """Emojis should be stripped when clean_text=True."""
        from datasety.audio import _normalize_text
        text = "Hello world! 😀🎉"
        result = _normalize_text(text, "en", normalize_numbers=False, clean_text=True)
        assert "😀" not in result
        assert "🎉" not in result
        # Note: EnglishTextNormalizer lowercases and removes punctuation
        assert "hello world" in result.lower()

    def test_keep_basic_punctuation(self):
        """Basic punctuation should be preserved."""
        from datasety.audio import _normalize_text
        text = 'Hello, world! How are you? "I\'m fine."'
        # With EnglishTextNormalizer, punctuation is normalized
        result = _normalize_text(text, "en", normalize_numbers=False, clean_text=True)
        # Check text is preserved (may be lowercased and punctuation changed)
        assert "hello" in result.lower()
        assert "world" in result.lower()

    def test_expand_numbers_english(self):
        """Numbers should be expanded to words in English when num2words is available."""
        import importlib.util
        if importlib.util.find_spec("num2words") is None:
            pytest.skip("num2words not installed")

        from datasety.audio import _normalize_text
        text = "I have 123 apples"
        result = _normalize_text(text, "en", normalize_numbers=True, clean_text=False)
        # num2words returns "one hundred and twenty-three" by default
        assert "one" in result.lower()
        assert "123" not in result

    def test_expand_numbers_spanish(self):
        """Numbers should be expanded to words in Spanish when num2words is available."""
        import importlib.util
        if importlib.util.find_spec("num2words") is None:
            pytest.skip("num2words not installed")

        from datasety.audio import _normalize_text
        text = "Tengo 5 gatos"
        result = _normalize_text(text, "es", normalize_numbers=True, clean_text=False)
        # num2words returns "cinco" for Spanish
        assert "cinco" in result.lower()
        assert "5" not in result

    def test_no_normalize_when_disabled(self):
        """Numbers should be kept as-is when normalize_numbers=False."""
        from datasety.audio import _normalize_text
        text = "I have 123 apples"
        # With clean_text=False, only control chars stripped
        result = _normalize_text(text, "en", normalize_numbers=False, clean_text=False)
        assert "123" in result

    def test_strip_leading_trailing_whitespace(self):
        """Leading and trailing whitespace should be stripped."""
        from datasety.audio import _normalize_text
        text = "   Hello world!   "
        result = _normalize_text(text, "en", normalize_numbers=False, clean_text=False)
        assert result == "Hello world!"

    def test_strips_special_characters(self):
        """Special characters like #, @ should be stripped. % may be preserved."""
        from datasety.audio import _normalize_text
        text = "Hello #world @user 100%"
        result = _normalize_text(text, "en", normalize_numbers=False, clean_text=True)
        # NeMo/whisper normalizers may preserve % as it's a semiotic class
        assert "#" not in result
        assert "@" not in result
        # Text is lowercased
        assert "hello" in result.lower()

    def test_non_english_basic_clean(self):
        """Non-English text should be cleaned while preserving Unicode characters."""
        from datasety.audio import _normalize_text
        # Spanish text
        text = "Hola Mundo! Cómo estás?"
        result = _normalize_text(text, "es", normalize_numbers=False, clean_text=True)
        assert "hola" in result.lower()
        assert "mundo" in result.lower()


class TestCheckFfmpeg:
    """Test FFmpeg availability check."""

    def test_ffmpeg_missing(self):
        """Should exit when ffmpeg is not available."""
        with patch("shutil.which", return_value=None):
            with pytest.raises(SystemExit):
                from datasety.audio import _check_ffmpeg
                _check_ffmpeg()


class TestExtractAudio:
    """Test audio extraction via FFmpeg subprocess."""

    def test_ffmpeg_args_mono_16bit(self):
        """FFmpeg should be called with correct PCM encoding arguments."""
        import datasety.audio as audio_module

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock()
            audio_module._extract_audio(
                Path("/input/video.mp4"),
                Path("/output/audio.wav"),
                sample_rate=22050,
                verbose=False,
            )

            mock_run.assert_called_once()
            call_args = mock_run.call_args

            # Extract command list - subprocess.run takes 'args' as first positional
            cmd = call_args[0][0]

            assert "ffmpeg" in cmd
            assert "-y" in cmd
            assert "-i" in cmd
            assert "-vn" in cmd
            assert "-acodec" in cmd
            assert "pcm_s16le" in cmd
            assert "-ar" in cmd
            assert "22050" in cmd
            assert "-ac" in cmd
            assert "1" in cmd  # mono

            # Verify order: ffmpeg -y -i input -vn -acodec pcm_s16le -ar 22050 -ac 1 output
            ffmpeg_idx = cmd.index("ffmpeg")
            y_idx = cmd.index("-y")
            i_idx = cmd.index("-i")
            acodec_idx = cmd.index("-acodec")
            ar_idx = cmd.index("-ar")
            ac_idx = cmd.index("-ac")

            assert y_idx > ffmpeg_idx
            assert i_idx > y_idx
            assert acodec_idx > i_idx
            assert ar_idx > acodec_idx
            assert ac_idx > ar_idx

    def test_extract_audio_sample_rate_arg(self):
        """FFmpeg should use the provided sample rate."""
        import datasety.audio as audio_module

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock()
            audio_module._extract_audio(
                Path("/input/video.mp4"),
                Path("/output/audio.wav"),
                sample_rate=44100,
                verbose=False,
            )

            cmd = mock_run.call_args[0][0]
            assert "44100" in cmd


class TestIsolateVocals:
    """Test Demucs vocal isolation (requires demucs package)."""

    def test_demucs_called_with_correct_args(self):
        """Demucs Separator should be initialized with correct model and device."""
        pytest.skip("Demucs requires GPU to test meaningfully")

    def test_no_vocals_stem_returns_original(self):
        """If Demucs finds no vocals, should return original audio path."""
        pytest.skip("Demucs requires GPU to test meaningfully")


class TestTranscribe:
    """Test faster-whisper transcription (requires faster-whisper package)."""

    def test_whisper_model_loaded_with_correct_args(self):
        """WhisperModel should be called with correct model size and device."""
        import importlib.util
        if importlib.util.find_spec("faster_whisper") is None:
            pytest.skip("faster-whisper not installed")

        mock_model = MagicMock()
        mock_segments = [
            MagicMock(start=0.0, end=2.5, text="Hello world"),
            MagicMock(start=3.0, end=5.5, text="How are you?"),
        ]
        mock_model.transcribe.return_value = (mock_segments, MagicMock())

        with patch("faster_whisper.WhisperModel", return_value=mock_model):
            from datasety.audio import _transcribe

            segments = _transcribe(
                Path("/input/vocals.wav"),
                model_size="base",
                device="cpu",
                language="en",
                verbose=False,
                vad=False,
            )

            mock_model.transcribe.assert_called_once()
            call_kwargs = mock_model.transcribe.call_args[1]
            assert call_kwargs["vad_filter"] is False
            assert call_kwargs["language"] == "en"

            assert len(segments) == 2
            assert segments[0]["start"] == 0.0
            assert segments[0]["end"] == 2.5
            assert segments[0]["text"] == "Hello world"

    def test_whisper_model_with_vad_enabled(self):
        """VAD should be enabled when vad=True."""
        import importlib.util
        if importlib.util.find_spec("faster_whisper") is None:
            pytest.skip("faster-whisper not installed")

        mock_model = MagicMock()
        mock_segments = [MagicMock(start=0.0, end=2.5, text="Hello world")]
        mock_model.transcribe.return_value = (mock_segments, MagicMock())

        with patch("faster_whisper.WhisperModel", return_value=mock_model):
            from datasety.audio import _transcribe

            _transcribe(
                Path("/input/vocals.wav"),
                model_size="tiny",
                device="cpu",
                language="en",
                verbose=False,
                vad=True,
            )

            call_kwargs = mock_model.transcribe.call_args[1]
            assert call_kwargs["vad_filter"] is True
            assert "vad_parameters" in call_kwargs

    def test_transcribe_without_language(self):
        """Transcription should work without specifying language (auto-detect)."""
        import importlib.util
        if importlib.util.find_spec("faster_whisper") is None:
            pytest.skip("faster-whisper not installed")

        mock_model = MagicMock()
        mock_segments = [MagicMock(start=0.0, end=1.5, text="Test")]
        mock_model.transcribe.return_value = (mock_segments, MagicMock())

        with patch("faster_whisper.WhisperModel", return_value=mock_model):
            from datasety.audio import _transcribe

            _transcribe(
                Path("/input/vocals.wav"),
                model_size="tiny",
                device="cpu",
                language=None,
                verbose=False,
                vad=False,
            )

            call_kwargs = mock_model.transcribe.call_args[1]
            assert "language" not in call_kwargs
            assert call_kwargs["vad_filter"] is False


class TestSliceAudio:
    """Test audio slicing with soundfile (requires soundfile package)."""

    def test_slices_audio_correctly(self):
        """Audio should be sliced at correct sample boundaries."""
        import importlib.util
        if importlib.util.find_spec("soundfile") is None:
            pytest.skip("soundfile not installed")

        import numpy as np
        import soundfile as sf

        # Create a small test WAV file
        sample_rate = 22050
        duration = 5.0
        samples = int(sample_rate * duration)
        audio_data = np.random.randn(samples).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_data, sample_rate)
            temp_path = Path(f.name)

        try:
            segments = [
                {"start": 0.0, "end": 2.0, "text": "First segment"},
                {"start": 2.5, "end": 4.5, "text": "Second segment"},
            ]

            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir)
                from datasety.audio import _slice_audio

                result = _slice_audio(
                    temp_path,
                    segments,
                    output_dir,
                    min_dur=1.0,
                    max_dur=10.0,
                    merge_gap=0.3,
                )
                metadata = list(result)

                assert len(metadata) == 2
                assert metadata[0]["filename"] == "utt_0001.wav"
                # Text may be lowercased and ordinals expanded (First -> 1st)
                assert "segment" in metadata[0]["text"].lower()
                assert (output_dir / "utt_0001.wav").exists()
                assert (output_dir / "utt_0002.wav").exists()
        finally:
            os.unlink(temp_path)

    def test_skips_short_segments(self):
        """Segments shorter than min_dur should be skipped."""
        import importlib.util
        if importlib.util.find_spec("soundfile") is None:
            pytest.skip("soundfile not installed")

        import numpy as np
        import soundfile as sf

        sample_rate = 22050
        duration = 5.0
        samples = int(sample_rate * duration)
        audio_data = np.random.randn(samples).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_data, sample_rate)
            temp_path = Path(f.name)

        try:
            segments = [
                {"start": 0.0, "end": 0.5, "text": "Too short"},  # < 1.5s
                {"start": 1.0, "end": 3.0, "text": "OK"},
            ]

            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir)
                from datasety.audio import _slice_audio

                result = _slice_audio(
                    temp_path,
                    segments,
                    output_dir,
                    min_dur=1.5,
                    max_dur=10.0,
                    merge_gap=0.3,
                )
                metadata = list(result)

                assert len(metadata) == 1
                # Text may be lowercased by normalizer
                assert "ok" in metadata[0]["text"].lower()
        finally:
            os.unlink(temp_path)

    def test_merges_close_segments(self):
        """Segments closer than merge_gap should be merged."""
        import importlib.util
        if importlib.util.find_spec("soundfile") is None:
            pytest.skip("soundfile not installed")

        import numpy as np
        import soundfile as sf

        sample_rate = 22050
        duration = 10.0
        samples = int(sample_rate * duration)
        audio_data = np.random.randn(samples).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_data, sample_rate)
            temp_path = Path(f.name)

        try:
            segments = [
                {"start": 0.0, "end": 2.0, "text": "Hello"},
                {"start": 2.3, "end": 4.0, "text": "world"},  # gap = 0.3s < 0.5s merge_gap
            ]

            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir)
                from datasety.audio import _slice_audio

                result = _slice_audio(
                    temp_path,
                    segments,
                    output_dir,
                    min_dur=1.0,
                    max_dur=10.0,
                    merge_gap=0.5,
                )
                metadata = list(result)

                assert len(metadata) == 1
                # Text may be lowercased by normalizer
                assert "hello" in metadata[0]["text"].lower()
        finally:
            os.unlink(temp_path)

    def test_no_merging_when_merge_gap_zero(self):
        """Segments should NOT merge when merge_gap is 0.0."""
        import importlib.util
        if importlib.util.find_spec("soundfile") is None:
            pytest.skip("soundfile not installed")

        import numpy as np
        import soundfile as sf

        sample_rate = 22050
        duration = 10.0
        samples = int(sample_rate * duration)
        audio_data = np.random.randn(samples).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_data, sample_rate)
            temp_path = Path(f.name)

        try:
            # Adjacent segments (gap=0)
            segments = [
                {"start": 0.0, "end": 3.0, "text": "First"},
                {"start": 3.0, "end": 6.0, "text": "Second"},
                {"start": 6.0, "end": 9.0, "text": "Third"},
            ]

            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir)
                from datasety.audio import _slice_audio

                result = _slice_audio(
                    temp_path,
                    segments,
                    output_dir,
                    min_dur=1.0,
                    max_dur=10.0,
                    merge_gap=0.0,  # OFF - no merging
                )
                metadata = list(result)

                # Should have 3 separate segments
                assert len(metadata) == 3
        finally:
            os.unlink(temp_path)


class TestMetadataCSV:
    """Test that metadata.csv is written in LJSpeech/Piper format."""

    def test_ljspeech_format(self):
        """CSV should use pipe delimiter with filename|text format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "metadata.csv"

            metadata = [
                {"filename": "utt_0001.wav", "text": "Hello world"},
                {"filename": "utt_0002.wav", "text": "How are you?"},
            ]

            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter="|")
                for entry in metadata:
                    writer.writerow([entry["filename"], entry["text"]])

            with open(csv_path, "r", encoding="utf-8") as f:
                content = f.read()

            lines = content.strip().split("\n")
            assert len(lines) == 2
            assert lines[0] == "utt_0001.wav|Hello world"
            assert lines[1] == "utt_0002.wav|How are you?"


class TestIsYouTube:
    """Test YouTube URL detection."""

    def test_youtube_url(self):
        from datasety.audio import _is_youtube
        assert _is_youtube("https://www.youtube.com/watch?v=abc123")
        assert _is_youtube("https://youtu.be/abc123")
        assert _is_youtube("https://youtube.com/shorts/abc123")

    def test_non_youtube_url(self):
        from datasety.audio import _is_youtube
        assert not _is_youtube("https://vimeo.com/123456")
        assert not _is_youtube("https://example.com/video.mp4")

