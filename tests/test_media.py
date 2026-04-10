"""Tests for shared media processing utilities."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestFormatTimestamp:
    """Test _format_timestamp converting seconds to HHMMSS string."""

    def test_zero(self):
        from datasety.media import _format_timestamp

        assert _format_timestamp(0.0) == "000000"

    def test_single_seconds(self):
        from datasety.media import _format_timestamp

        assert _format_timestamp(3.5) == "000003"

    def test_over_one_minute(self):
        from datasety.media import _format_timestamp

        assert _format_timestamp(65.0) == "000105"

    def test_over_one_hour(self):
        from datasety.media import _format_timestamp

        # 3745s = 1h 2m 25s
        assert _format_timestamp(3745.0) == "010225"

    def test_exact_minute(self):
        from datasety.media import _format_timestamp

        assert _format_timestamp(60.0) == "000100"

    def test_exact_hour(self):
        from datasety.media import _format_timestamp

        assert _format_timestamp(3600.0) == "010000"

    def test_large_value(self):
        from datasety.media import _format_timestamp

        assert _format_timestamp(86400.0) == "240000"


class TestMakeSegmentName:
    """Test _make_segment_name building filename stems."""

    def test_single_source_no_prefix(self):
        from datasety.media import _make_segment_name

        result = _make_segment_name(None, 0.0, 3.5)
        assert result == "000000-000003"

    def test_empty_source_name(self):
        from datasety.media import _make_segment_name

        result = _make_segment_name("", 0.0, 3.5)
        assert result == "000000-000003"

    def test_with_source_name(self):
        from datasety.media import _make_segment_name

        result = _make_segment_name("clip23", 0.0, 3.0)
        assert result == "clip23-000000-000003"

    def test_youtube_id_source(self):
        from datasety.media import _make_segment_name

        result = _make_segment_name("dQw4w9WgXcQ", 75.0, 87.0)
        assert result == "dQw4w9WgXcQ-000115-000127"

    def test_timestamp_ranges(self):
        from datasety.media import _make_segment_name

        # 3723s = 1h 2m 3s, 3727s = 1h 2m 7s
        result = _make_segment_name("vid", 3723.0, 3727.0)
        assert result == "vid-010203-010207"


class TestGetSourceName:
    """Test _get_source_name extracting source identifiers."""

    def test_local_file(self):
        from datasety.media import _get_source_name

        item = {
            "source": "/path/to/clip23.mp4",
            "name": "clip23.mp4",
            "is_url": False,
            "is_youtube": False,
        }
        assert _get_source_name(item) == "clip23"

    def test_youtube_url(self):
        from datasety.media import _get_source_name

        item = {
            "source": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "is_url": True,
            "is_youtube": True,
        }
        assert _get_source_name(item) == "dQw4w9WgXcQ"

    def test_youtube_short_url(self):
        from datasety.media import _get_source_name

        item = {"source": "https://youtu.be/abc123", "is_url": True, "is_youtube": True}
        assert _get_source_name(item) == "abc123"

    def test_non_youtube_url(self):
        from datasety.media import _get_source_name

        item = {"source": "https://example.com/video.mp4", "is_url": True, "is_youtube": False}
        result = _get_source_name(item)
        assert len(result) == 12
        assert result.isalnum()

    def test_local_file_with_dots(self):
        from datasety.media import _get_source_name

        item = {
            "source": "/path/my.video.file.mp4",
            "name": "my.video.file.mp4",
            "is_url": False,
            "is_youtube": False,
        }
        assert _get_source_name(item) == "my.video.file"


class TestGetMediaFiles:
    """Test directory scanning for audio/video files."""

    def test_finds_audio_extensions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = Path(tmpdir)
            (audio_dir / "1.mp3").touch()
            (audio_dir / "2.wav").touch()
            (audio_dir / "3.flac").touch()
            (audio_dir / "video.mp4").touch()
            (audio_dir / "video.mkv").touch()

            from datasety.media import _get_media_files

            files = _get_media_files(audio_dir)

            names = [p.name for p in files]
            assert "1.mp3" in names
            assert "2.wav" in names
            assert "3.flac" in names
            assert "video.mp4" in names
            assert "video.mkv" in names

    def test_ignores_non_media_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = Path(tmpdir)
            (audio_dir / "1.mp3").touch()
            (audio_dir / "readme.txt").touch()
            (audio_dir / "script.py").touch()

            from datasety.media import _get_media_files

            files = _get_media_files(audio_dir)

            assert len(files) == 1
            assert files[0].name == "1.mp3"

    def test_sorts_numerically(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = Path(tmpdir)
            (audio_dir / "10.mp3").touch()
            (audio_dir / "2.mp3").touch()
            (audio_dir / "1.mp3").touch()
            (audio_dir / "3.mp3").touch()

            from datasety.media import _get_media_files

            files = _get_media_files(audio_dir)

            names = [p.name for p in files]
            assert names == ["1.mp3", "2.mp3", "3.mp3", "10.mp3"]

    def test_case_insensitive_sort(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = Path(tmpdir)
            (audio_dir / "B.mp3").touch()
            (audio_dir / "a.mp3").touch()

            from datasety.media import _get_media_files

            files = _get_media_files(audio_dir)

            names = [p.name for p in files]
            assert names == ["a.mp3", "B.mp3"]

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = Path(tmpdir)
            (audio_dir / "readme.txt").touch()

            from datasety.media import _get_media_files

            files = _get_media_files(audio_dir)

            assert files == []

    def test_no_subdirectory_scan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = Path(tmpdir)
            sub_dir = audio_dir / "subdir"
            sub_dir.mkdir()
            (audio_dir / "1.mp3").touch()
            (sub_dir / "2.mp3").touch()

            from datasety.media import _get_media_files

            files = _get_media_files(audio_dir)

            assert len(files) == 1
            assert files[0].name == "1.mp3"


class TestNormalizeText:
    """Test text normalization for TTS."""

    def test_strip_control_characters(self):
        from datasety.media import _normalize_text

        text = "Hello\x00world\x07!"
        result = _normalize_text(text, "en", normalize_numbers=False, clean_text=True)
        assert "\x00" not in result
        assert "\x07" not in result
        assert "hello" in result.lower()

    def test_strip_emoji(self):
        from datasety.media import _normalize_text

        text = "Hello world! 😀🎉"
        result = _normalize_text(text, "en", normalize_numbers=False, clean_text=True)
        assert "😀" not in result
        assert "🎉" not in result
        assert "hello world" in result.lower()

    def test_expand_numbers_english(self):
        import importlib.util

        if importlib.util.find_spec("num2words") is None:
            pytest.skip("num2words not installed")

        from datasety.media import _normalize_text

        text = "I have 123 apples"
        result = _normalize_text(text, "en", normalize_numbers=True, clean_text=False)
        assert "one" in result.lower()
        assert "123" not in result

    def test_expand_numbers_spanish(self):
        import importlib.util

        if importlib.util.find_spec("num2words") is None:
            pytest.skip("num2words not installed")

        from datasety.media import _normalize_text

        text = "Tengo 5 gatos"
        result = _normalize_text(text, "es", normalize_numbers=True, clean_text=False)
        assert "cinco" in result.lower()
        assert "5" not in result

    def test_no_normalize_when_disabled(self):
        from datasety.media import _normalize_text

        text = "I have 123 apples"
        result = _normalize_text(text, "en", normalize_numbers=False, clean_text=False)
        assert "123" in result

    def test_strip_leading_trailing_whitespace(self):
        from datasety.media import _normalize_text

        text = "   Hello world!   "
        result = _normalize_text(text, "en", normalize_numbers=False, clean_text=False)
        assert result == "Hello world!"

    def test_non_english_basic_clean(self):
        from datasety.media import _normalize_text

        text = "Hola Mundo! Cómo estás?"
        result = _normalize_text(text, "es", normalize_numbers=False, clean_text=True)
        assert "hola" in result.lower()
        assert "mundo" in result.lower()


class TestCheckFfmpeg:
    """Test FFmpeg availability check."""

    def test_ffmpeg_missing(self):
        with patch("shutil.which", return_value=None):
            with pytest.raises(SystemExit):
                from datasety.media import _check_ffmpeg

                _check_ffmpeg()


class TestIsYouTube:
    """Test YouTube URL detection."""

    def test_youtube_url(self):
        from datasety.media import _is_youtube

        assert _is_youtube("https://www.youtube.com/watch?v=abc123")
        assert _is_youtube("https://youtu.be/abc123")
        assert _is_youtube("https://youtube.com/shorts/abc123")

    def test_non_youtube_url(self):
        from datasety.media import _is_youtube

        assert not _is_youtube("https://vimeo.com/123456")
        assert not _is_youtube("https://example.com/video.mp4")


class TestBuildMediaItems:
    """Test build_media_items for parsing input sources."""

    def test_single_local_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "test.mp4"
            f.touch()
            from datasety.media import build_media_items

            items = build_media_items(str(f))
            assert len(items) == 1
            assert items[0]["name"] == "test.mp4"
            assert items[0]["is_url"] is False
            assert items[0]["is_youtube"] is False

    def test_directory_of_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "1.mp3").touch()
            (d / "2.wav").touch()
            (d / "readme.txt").touch()
            from datasety.media import build_media_items

            items = build_media_items(str(d))
            assert len(items) == 2

    def test_text_file_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = Path(tmpdir) / "a.mp3"
            f2 = Path(tmpdir) / "b.wav"
            f1.touch()
            f2.touch()
            list_file = Path(tmpdir) / "sources.txt"
            list_file.write_text(f"{f1}\n{f2}\n", encoding="utf-8")
            from datasety.media import build_media_items

            items = build_media_items(str(list_file))
            assert len(items) == 2

    def test_youtube_url(self):
        from datasety.media import build_media_items

        items = build_media_items("https://www.youtube.com/watch?v=abc123")
        assert len(items) == 1
        assert items[0]["is_youtube"] is True

    def test_url_with_time_slicing(self):
        from datasety.media import build_media_items

        items = build_media_items("https://www.youtube.com/watch?v=abc123&start=50&end=90")
        assert len(items) == 1
        assert items[0]["start"] == 50.0
        assert items[0]["end"] == 90.0


class TestCleanTtsText:
    """Test TTS text cleaning."""

    def test_em_dash_to_hyphen(self):
        from datasety.media import _clean_tts_text

        assert _clean_tts_text("hello—world") == "hello-world"
        assert _clean_tts_text("hello–world") == "hello-world"

    def test_connect_hyphen(self):
        from datasety.media import _clean_tts_text

        assert _clean_tts_text("some - thing") == "some-thing"

    def test_connect_apostrophe(self):
        from datasety.media import _clean_tts_text

        assert _clean_tts_text("ім 'я") == "ім'я"

    def test_hallucination_loop(self):
        from datasety.media import _clean_tts_text

        repeated = "abc1234567890ab" * 3
        text = f"hello {repeated}"
        result = _clean_tts_text(text)
        assert result.count("abc1234567890ab") == 1

    def test_fix_punctuation_spacing(self):
        from datasety.media import _clean_tts_text

        assert _clean_tts_text("hello , world !") == "hello, world!"


class TestIsValidByPhonemes:
    def test_empty_valid_chars(self):
        from datasety.media import _is_valid_by_phonemes

        assert _is_valid_by_phonemes("any text", set()) is True

    def test_valid_text(self):
        from datasety.media import _is_valid_by_phonemes

        valid = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p"}
        assert _is_valid_by_phonemes("hello", valid) is True

    def test_invalid_char(self):
        from datasety.media import _is_valid_by_phonemes

        valid = {"a", "b", "c"}
        assert _is_valid_by_phonemes("hello", valid) is False

    def test_whitespace_allowed(self):
        from datasety.media import _is_valid_by_phonemes

        valid = {"h", "e", "l", "o"}
        assert _is_valid_by_phonemes("h e l l o", valid) is True


class TestApplyTemplate:
    def test_template_with_caption_placeholder(self):
        from datasety.media import _apply_template

        result = _apply_template("[trigger] {{caption}}", "a cat sitting")
        assert result == "[trigger] a cat sitting"

    def test_template_with_transcript_placeholder(self):
        from datasety.media import _apply_template

        result = _apply_template("sks person says: {{transcript}}", "hello world")
        assert result == "sks person says: hello world"

    def test_template_without_placeholder_prepends(self):
        from datasety.media import _apply_template

        result = _apply_template("photo of sks person,", "a woman in red")
        assert result == "photo of sks person, a woman in red"

    def test_template_none_returns_text(self):
        from datasety.media import _apply_template

        assert _apply_template(None, "hello world") == "hello world"

    def test_template_empty_string_returns_text(self):
        from datasety.media import _apply_template

        assert _apply_template("", "hello world") == "hello world"

    def test_template_with_multiple_placeholders(self):
        from datasety.media import _apply_template

        result = _apply_template("Q: {{caption}} A: {{caption}}", "test")
        assert result == "Q: test A: test"

    def test_template_with_transcript_and_caption_both(self):
        from datasety.media import _apply_template

        result = _apply_template("{{transcript}} - {{caption}}", "hello")
        assert result == "hello - hello"

    def test_prepend_no_placeholder(self):
        from datasety.media import _apply_template

        result = _apply_template("ohwx person,", "a photo of a face")
        assert result == "ohwx person, a photo of a face"
