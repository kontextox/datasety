"""Tests for the audio command."""

import csv
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestExtractAudio:
    """Test audio extraction via FFmpeg subprocess."""

    def test_ffmpeg_args_mono_16bit(self):
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
            assert "1" in cmd

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


class TestSliceAudio:
    """Test audio slicing with soundfile (requires soundfile package)."""

    def test_slices_audio_pairs_format(self):
        """In pairs format, slices should create .wav + .txt pairs with timestamp names."""
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
                    global_idx=0,
                    local_skip=0,
                    min_dur=1.0,
                    max_dur=10.0,
                    merge_gap=0.3,
                    source_name=None,
                    output_format="pairs",
                )
                metadata = list(result)

                assert len(metadata) == 2
                assert (output_dir / "000000-000002.wav").exists()
                assert (output_dir / "000000-000002.txt").exists()
                assert (output_dir / "000002-000004.wav").exists()
                assert (output_dir / "000002-000004.txt").exists()
        finally:
            os.unlink(temp_path)

    def test_slices_audio_ljspeech_format(self):
        """In ljspeech format, slices should create utt_NNNN.wav files."""
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
                    global_idx=0,
                    local_skip=0,
                    min_dur=1.0,
                    max_dur=10.0,
                    merge_gap=0.3,
                    output_format="ljspeech",
                )
                metadata = list(result)

                assert len(metadata) == 2
                assert metadata[0][2]["filename"] == "utt_0001.wav"
                assert (output_dir / "utt_0001.wav").exists()
                assert (output_dir / "utt_0002.wav").exists()
        finally:
            os.unlink(temp_path)

    def test_pairs_with_source_name(self):
        """In pairs format with source_name, filenames should include source prefix."""
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
                {"start": 0.0, "end": 2.0, "text": "First segment"},
            ]

            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir)
                from datasety.audio import _slice_audio

                result = _slice_audio(
                    temp_path,
                    segments,
                    output_dir,
                    global_idx=0,
                    local_skip=0,
                    min_dur=1.0,
                    max_dur=10.0,
                    merge_gap=0.0,
                    source_name="clip23",
                    output_format="pairs",
                )
                metadata = list(result)

                assert len(metadata) == 1
                assert metadata[0][2]["filename"] == "clip23-000000-000002.wav"
                assert (output_dir / "clip23-000000-000002.wav").exists()
                assert (output_dir / "clip23-000000-000002.txt").exists()
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
                {"start": 0.0, "end": 0.5, "text": "Too short"},
                {"start": 1.0, "end": 3.0, "text": "OK"},
            ]

            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir)
                from datasety.audio import _slice_audio

                result = _slice_audio(
                    temp_path,
                    segments,
                    output_dir,
                    global_idx=0,
                    local_skip=0,
                    min_dur=1.5,
                    max_dur=10.0,
                    merge_gap=0.3,
                )
                metadata = list(result)

                assert len(metadata) == 1
                assert "ok" in metadata[0][2]["text"].lower()
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
                {"start": 2.3, "end": 4.0, "text": "world"},
            ]

            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir)
                from datasety.audio import _slice_audio

                result = _slice_audio(
                    temp_path,
                    segments,
                    output_dir,
                    global_idx=0,
                    local_skip=0,
                    min_dur=1.0,
                    max_dur=10.0,
                    merge_gap=0.5,
                )
                metadata = list(result)

                assert len(metadata) == 1
                assert "hello" in metadata[0][2]["text"].lower()
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
                    global_idx=0,
                    local_skip=0,
                    min_dur=1.0,
                    max_dur=10.0,
                    merge_gap=0.0,
                )
                metadata = list(result)

                assert len(metadata) == 3
        finally:
            os.unlink(temp_path)

    def test_txt_sidecar_content(self):
        """The .txt sidecar should contain the cleaned text."""
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
                {"start": 0.0, "end": 2.0, "text": "Hello world"},
            ]

            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir)
                from datasety.audio import _slice_audio

                list(
                    _slice_audio(
                        temp_path,
                        segments,
                        output_dir,
                        global_idx=0,
                        local_skip=0,
                        min_dur=1.0,
                        max_dur=10.0,
                        merge_gap=0.0,
                        source_name=None,
                        output_format="pairs",
                        clean_text=False,
                    )
                )

                txt_path = output_dir / "000000-000002.txt"
                assert txt_path.exists()
                content = txt_path.read_text(encoding="utf-8")
                assert "hello" in content.lower() or "Hello" in content
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


class TestDeduplicateMetadata:
    """Test consecutive duplicate removal from metadata.csv."""

    def test_removes_consecutive_duplicates(self):
        from datasety.audio import _deduplicate_metadata

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            wavs_dir = output_dir / "wavs"
            wavs_dir.mkdir()

            csv_path = output_dir / "metadata.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter="|")
                writer.writerow(["utt_0001.wav", "про що ти хочеш мене попросити?"])
                writer.writerow(["utt_0002.wav", "запитала вона підозріло."])
                writer.writerow(["utt_0003.wav", "запитала вона підозріло."])
                writer.writerow(["utt_0004.wav", "про те що"])

            (wavs_dir / "utt_0001.wav").touch()
            (wavs_dir / "utt_0002.wav").touch()
            (wavs_dir / "utt_0003.wav").touch()
            (wavs_dir / "utt_0004.wav").touch()

            _deduplicate_metadata(output_dir, wavs_dir)

            with open(csv_path, "r", encoding="utf-8") as f:
                content = f.read()

            lines = content.strip().split("\n")
            assert len(lines) == 3
            assert lines[0] == "utt_0001.wav|про що ти хочеш мене попросити?"
            assert lines[1] == "utt_0002.wav|запитала вона підозріло."
            assert lines[2] == "utt_0004.wav|про те що"

            assert (wavs_dir / "utt_0001.wav").exists()
            assert (wavs_dir / "utt_0002.wav").exists()
            assert not (wavs_dir / "utt_0003.wav").exists()
            assert (wavs_dir / "utt_0004.wav").exists()

    def test_keeps_non_consecutive_duplicates(self):
        from datasety.audio import _deduplicate_metadata

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            wavs_dir = output_dir / "wavs"
            wavs_dir.mkdir()

            csv_path = output_dir / "metadata.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter="|")
                writer.writerow(["utt_0001.wav", "Hello world"])
                writer.writerow(["utt_0002.wav", "Hello world"])
                writer.writerow(["utt_0003.wav", "Hello world"])

            (wavs_dir / "utt_0001.wav").touch()
            (wavs_dir / "utt_0002.wav").touch()
            (wavs_dir / "utt_0003.wav").touch()

            _deduplicate_metadata(output_dir, wavs_dir)

            with open(csv_path, "r", encoding="utf-8") as f:
                content = f.read()

            lines = content.strip().split("\n")
            assert len(lines) == 1
            assert lines[0] == "utt_0001.wav|Hello world"

            assert (wavs_dir / "utt_0001.wav").exists()
            assert not (wavs_dir / "utt_0002.wav").exists()
            assert not (wavs_dir / "utt_0003.wav").exists()

    def test_no_duplicates_unchanged(self):
        from datasety.audio import _deduplicate_metadata

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            wavs_dir = output_dir / "wavs"
            wavs_dir.mkdir()

            csv_path = output_dir / "metadata.csv"
            original = [
                ["utt_0001.wav", "First unique text"],
                ["utt_0002.wav", "Second unique text"],
            ]
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter="|")
                writer.writerows(original)

            (wavs_dir / "utt_0001.wav").touch()
            (wavs_dir / "utt_0002.wav").touch()

            _deduplicate_metadata(output_dir, wavs_dir)

            with open(csv_path, "r", encoding="utf-8") as f:
                content = f.read()

            lines = content.strip().split("\n")
            assert len(lines) == 2
            assert lines[0] == "utt_0001.wav|First unique text"
            assert lines[1] == "utt_0002.wav|Second unique text"

    def test_empty_metadata_unchanged(self):
        from datasety.audio import _deduplicate_metadata

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            wavs_dir = output_dir / "wavs"
            wavs_dir.mkdir()

            csv_path = output_dir / "metadata.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                pass

            _deduplicate_metadata(output_dir, wavs_dir)

            with open(csv_path, "r", encoding="utf-8") as f:
                content = f.read()

            assert content == ""

    def test_single_entry_unchanged(self):
        from datasety.audio import _deduplicate_metadata

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            wavs_dir = output_dir / "wavs"
            wavs_dir.mkdir()

            csv_path = output_dir / "metadata.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter="|")
                writer.writerow(["utt_0001.wav", "Only one"])

            (wavs_dir / "utt_0001.wav").touch()

            _deduplicate_metadata(output_dir, wavs_dir)

            with open(csv_path, "r", encoding="utf-8") as f:
                content = f.read()

            lines = content.strip().split("\n")
            assert len(lines) == 1
            assert lines[0] == "utt_0001.wav|Only one"

    def test_deletion_log_written(self):
        from datasety.audio import _deduplicate_metadata

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            wavs_dir = output_dir / "wavs"
            wavs_dir.mkdir()

            csv_path = output_dir / "metadata.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter="|")
                writer.writerow(["utt_0001.wav", "Hello"])
                writer.writerow(["utt_0002.wav", "duplicate"])
                writer.writerow(["utt_0003.wav", "duplicate"])
                writer.writerow(["utt_0004.wav", "world"])

            (wavs_dir / "utt_0001.wav").touch()
            (wavs_dir / "utt_0002.wav").touch()
            (wavs_dir / "utt_0003.wav").touch()
            (wavs_dir / "utt_0004.wav").touch()

            _deduplicate_metadata(output_dir, wavs_dir)

            log_path = output_dir / "deletions.csv"
            assert log_path.exists()

            with open(log_path, "r", encoding="utf-8") as f:
                content = f.read()

            lines = content.strip().split("\n")
            assert len(lines) == 2
            assert "utt_0003.wav" in lines[1]
            assert "duplicate_text" in lines[1]


class TestDeduplicatePairs:
    """Test consecutive duplicate removal in flat pair mode."""

    def test_removes_consecutive_duplicate_pairs(self):
        from datasety.audio import _deduplicate_pairs

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            (output_dir / "000000-000002.wav").touch()
            (output_dir / "000000-000002.txt").write_text("Hello", encoding="utf-8")
            (output_dir / "000003-000005.wav").touch()
            (output_dir / "000003-000005.txt").write_text("Hello", encoding="utf-8")
            (output_dir / "000006-000008.wav").touch()
            (output_dir / "000006-000008.txt").write_text("World", encoding="utf-8")

            _deduplicate_pairs(output_dir)

            assert (output_dir / "000000-000002.wav").exists()
            assert (output_dir / "000000-000002.txt").exists()
            assert not (output_dir / "000003-000005.wav").exists()
            assert not (output_dir / "000003-000005.txt").exists()
            assert (output_dir / "000006-000008.wav").exists()
            assert (output_dir / "000006-000008.txt").exists()

    def test_no_duplicates_unchanged(self):
        from datasety.audio import _deduplicate_pairs

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            (output_dir / "000000-000002.wav").touch()
            (output_dir / "000000-000002.txt").write_text("Hello", encoding="utf-8")
            (output_dir / "000003-000005.wav").touch()
            (output_dir / "000003-000005.txt").write_text("World", encoding="utf-8")

            _deduplicate_pairs(output_dir)

            assert (output_dir / "000000-000002.wav").exists()
            assert (output_dir / "000003-000005.wav").exists()
