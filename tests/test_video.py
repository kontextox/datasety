"""Tests for the video command."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestExtractVideoSegment:
    """Test video segment extraction via FFmpeg."""

    def test_stream_copy_args(self):
        from datasety.video import _extract_video_segment

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock()
            _extract_video_segment(
                Path("/input/video.mp4"),
                Path("/output/seg.mp4"),
                start=5.0,
                end=10.0,
                re_encode=False,
                verbose=False,
            )

            cmd = mock_run.call_args[0][0]
            assert "ffmpeg" in cmd
            assert "-y" in cmd
            assert "-ss" in cmd
            assert "-to" in cmd
            assert "-c" in cmd
            assert "copy" in cmd
            assert "5.0" in cmd
            assert "10.0" in cmd

    def test_re_encode_args(self):
        from datasety.video import _extract_video_segment

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock()
            _extract_video_segment(
                Path("/input/video.mkv"),
                Path("/output/seg.mp4"),
                start=5.0,
                end=10.0,
                re_encode=True,
                verbose=False,
            )

            cmd = mock_run.call_args[0][0]
            assert "ffmpeg" in cmd
            assert "-c:v" in cmd
            assert "libx264" in cmd
            assert "-c:a" in cmd
            assert "aac" in cmd

    def test_stream_copy_no_encode_args(self):
        from datasety.video import _extract_video_segment

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock()
            _extract_video_segment(
                Path("/input/video.mp4"),
                Path("/output/seg.mp4"),
                start=0.0,
                end=5.0,
                re_encode=False,
            )

            cmd = mock_run.call_args[0][0]
            assert "-c" in cmd
            assert "copy" in cmd
            assert "libx264" not in cmd


class TestSliceVideo:
    """Test video slicing logic."""

    def test_slice_video_creates_pairs(self):
        from datasety.video import _slice_video

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            media_path = Path(tmpdir) / "source.mp4"
            media_path.touch()

            segments = [
                {"start": 0.0, "end": 3.0, "text": "Hello world"},
                {"start": 5.0, "end": 10.0, "text": "Second segment"},
            ]

            with patch("datasety.video._extract_video_segment"):
                result = list(
                    _slice_video(
                        media_path,
                        segments,
                        output_dir,
                        local_skip=0,
                        min_dur=1.0,
                        max_dur=30.0,
                        merge_gap=0.0,
                        re_encode=False,
                        clean_text=False,
                        source_name=None,
                    )
                )

                assert len(result) == 2
                assert result[0][2]["filename"] == "000000-000003.mp4"
                assert result[1][2]["filename"] == "000005-000010.mp4"
                assert (output_dir / "000000-000003.txt").exists()
                assert (output_dir / "000005-000010.txt").exists()

    def test_slice_video_with_source_name(self):
        from datasety.video import _slice_video

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            media_path = Path(tmpdir) / "source.mp4"
            media_path.touch()

            segments = [
                {"start": 0.0, "end": 3.0, "text": "Hello"},
            ]

            with patch("datasety.video._extract_video_segment"):
                result = list(
                    _slice_video(
                        media_path,
                        segments,
                        output_dir,
                        local_skip=0,
                        min_dur=1.0,
                        max_dur=30.0,
                        merge_gap=0.0,
                        re_encode=False,
                        clean_text=False,
                        source_name="clip23",
                    )
                )

                assert len(result) == 1
                assert result[0][2]["filename"] == "clip23-000000-000003.mp4"
                assert (output_dir / "clip23-000000-000003.txt").exists()

    def test_skips_short_segments(self):
        from datasety.video import _slice_video

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            media_path = Path(tmpdir) / "source.mp4"
            media_path.touch()

            segments = [
                {"start": 0.0, "end": 0.5, "text": "Too short"},
                {"start": 1.0, "end": 5.0, "text": "OK"},
            ]

            with patch("datasety.video._extract_video_segment"):
                result = list(
                    _slice_video(
                        media_path,
                        segments,
                        output_dir,
                        local_skip=0,
                        min_dur=1.5,
                        max_dur=30.0,
                        merge_gap=0.0,
                        re_encode=False,
                        clean_text=False,
                        source_name=None,
                    )
                )

                assert len(result) == 1

    def test_merges_close_segments(self):
        from datasety.video import _slice_video

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            media_path = Path(tmpdir) / "source.mp4"
            media_path.touch()

            segments = [
                {"start": 0.0, "end": 2.0, "text": "Hello"},
                {"start": 2.3, "end": 4.0, "text": "world"},
            ]

            with patch("datasety.video._extract_video_segment"):
                result = list(
                    _slice_video(
                        media_path,
                        segments,
                        output_dir,
                        local_skip=0,
                        min_dur=1.0,
                        max_dur=30.0,
                        merge_gap=0.5,
                        re_encode=False,
                        clean_text=False,
                        source_name=None,
                    )
                )

                assert len(result) == 1
                assert "000000-000004.mp4" == result[0][2]["filename"]

    def test_txt_content(self):
        from datasety.video import _slice_video

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            media_path = Path(tmpdir) / "source.mp4"
            media_path.touch()

            segments = [
                {"start": 0.0, "end": 3.0, "text": "Hello world"},
            ]

            with patch("datasety.video._extract_video_segment"):
                list(
                    _slice_video(
                        media_path,
                        segments,
                        output_dir,
                        local_skip=0,
                        min_dur=1.0,
                        max_dur=30.0,
                        merge_gap=0.0,
                        re_encode=False,
                        clean_text=False,
                        source_name=None,
                    )
                )

                txt_path = output_dir / "000000-000003.txt"
                assert txt_path.exists()
                content = txt_path.read_text(encoding="utf-8")
                assert "Hello world" in content

    def test_preserves_video_extension(self):
        from datasety.video import _slice_video

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            media_path = Path(tmpdir) / "source.mkv"
            media_path.touch()

            segments = [
                {"start": 0.0, "end": 3.0, "text": "Hello"},
            ]

            with patch("datasety.video._extract_video_segment"):
                result = list(
                    _slice_video(
                        media_path,
                        segments,
                        output_dir,
                        local_skip=0,
                        min_dur=1.0,
                        max_dur=30.0,
                        merge_gap=0.0,
                        re_encode=False,
                        clean_text=False,
                        source_name=None,
                    )
                )

                assert result[0][2]["filename"] == "000000-000003.mkv"

    def test_non_video_extension_defaults_to_mp4(self):
        from datasety.video import _slice_video

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            media_path = Path(tmpdir) / "source.wav"
            media_path.touch()

            segments = [
                {"start": 0.0, "end": 3.0, "text": "Hello"},
            ]

            with patch("datasety.video._extract_video_segment"):
                result = list(
                    _slice_video(
                        media_path,
                        segments,
                        output_dir,
                        local_skip=0,
                        min_dur=1.0,
                        max_dur=30.0,
                        merge_gap=0.0,
                        re_encode=False,
                        clean_text=False,
                        source_name=None,
                    )
                )

                assert result[0][2]["filename"] == "000000-000003.mp4"


class TestDeduplicatePairs:
    """Test consecutive duplicate removal for video pairs."""

    def test_removes_consecutive_duplicate_pairs(self):
        from datasety.video import _deduplicate_pairs

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            (output_dir / "000000-000002.mp4").touch()
            (output_dir / "000000-000002.txt").write_text("Hello", encoding="utf-8")
            (output_dir / "000003-000005.mp4").touch()
            (output_dir / "000003-000005.txt").write_text("Hello", encoding="utf-8")
            (output_dir / "000006-000008.mp4").touch()
            (output_dir / "000006-000008.txt").write_text("World", encoding="utf-8")

            _deduplicate_pairs(output_dir)

            assert (output_dir / "000000-000002.mp4").exists()
            assert (output_dir / "000000-000002.txt").exists()
            assert not (output_dir / "000003-000005.mp4").exists()
            assert not (output_dir / "000003-000005.txt").exists()
            assert (output_dir / "000006-000008.mp4").exists()
            assert (output_dir / "000006-000008.txt").exists()
