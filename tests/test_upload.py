"""Tests for the upload command."""

import csv
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class TestDetectDatasetType:
    """Test auto-detection of dataset types from directory structure."""

    def test_audio_folder_structure(self):
        """HF AudioFolder (wavs/ + metadata.csv) should be detected as audio."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "wavs").mkdir()
            (d / "metadata.csv").touch()

            from datasety.upload import detect_dataset_type

            assert detect_dataset_type(d) == "audio"

    def test_audio_by_extension(self):
        """Directory with .wav files should be detected as audio."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "test.wav").touch()
            (d / "clip.mp3").touch()

            from datasety.upload import detect_dataset_type

            assert detect_dataset_type(d) == "audio"

    def test_audio_flac(self):
        """Directory with .flac files should be detected as audio."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "audio.flac").touch()

            from datasety.upload import detect_dataset_type

            assert detect_dataset_type(d) == "audio"

    def test_image_with_subdirs(self):
        """Directory with image files AND subdirectories should be detected as image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "train").mkdir()
            (d / "train" / "img.jpg").touch()
            (d / "test").mkdir()
            (d / "test" / "img2.png").touch()

            from datasety.upload import detect_dataset_type

            assert detect_dataset_type(d) == "image"

    def test_image_without_subdirs_is_generic(self):
        """Directory with images but no subdirs falls through to generic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "photo.jpg").touch()

            from datasety.upload import detect_dataset_type

            # No subdirs, so no ImageFolder structure — falls through
            assert detect_dataset_type(d) == "generic"

    def test_video_by_extension(self):
        """Directory with video files should be detected as video."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "video.mp4").touch()
            (d / "clip.mkv").touch()

            from datasety.upload import detect_dataset_type

            assert detect_dataset_type(d) == "video"

    def test_document_pdf(self):
        """Directory with PDF files should be detected as document."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "doc.pdf").touch()

            from datasety.upload import detect_dataset_type

            assert detect_dataset_type(d) == "document"

    def test_model_safetensors(self):
        """Directory with .safetensors files should be detected as model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "lora.safetensors").touch()

            from datasety.upload import detect_dataset_type

            assert detect_dataset_type(d) == "model"

    def test_model_bin(self):
        """.bin files should be detected as model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "model.bin").touch()

            from datasety.upload import detect_dataset_type

            assert detect_dataset_type(d) == "model"

    def test_model_gguf(self):
        """.gguf files should be detected as model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "model.gguf").touch()

            from datasety.upload import detect_dataset_type

            assert detect_dataset_type(d) == "model"

    def test_generic_csv(self):
        """Directory with CSV but no audio/image/video falls to generic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "data.csv").touch()

            from datasety.upload import detect_dataset_type

            assert detect_dataset_type(d) == "generic"

    def test_generic_json(self):
        """.json files should be detected as generic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "data.json").touch()

            from datasety.upload import detect_dataset_type

            assert detect_dataset_type(d) == "generic"

    def test_nonexistent_path_returns_generic(self):
        """Non-existent path should return generic (no files found)."""
        from datasety.upload import detect_dataset_type

        assert detect_dataset_type(Path("/nonexistent/path")) == "generic"

    def test_single_file_detection(self):
        """Single file (not directory) should be detected by its extension."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake audio")
            p = Path(f.name)

        try:
            from datasety.upload import detect_dataset_type

            assert detect_dataset_type(p) == "audio"
        finally:
            p.unlink()


class TestGenerateDatasetCard:
    """Test README dataset card generation."""

    def _make_audio_dataset(self, tmpdir: Path) -> Path:
        """Create a minimal audio dataset with wavs/ and metadata.csv."""
        wavs = tmpdir / "wavs"
        wavs.mkdir()
        (wavs / "utt_0001.wav").touch()
        (wavs / "utt_0002.wav").touch()
        meta = tmpdir / "metadata.csv"
        with open(meta, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, delimiter="|", fieldnames=["filename", "text"])
            writer.writeheader()
            writer.writerow({"filename": "utt_0001.wav", "text": "Hello world"})
            writer.writerow({"filename": "utt_0002.wav", "text": "How are you"})
        return tmpdir

    def test_audio_card_has_yaml_frontmatter(self):
        """Generated card should have YAML frontmatter with required fields."""
        from datasety.upload import generate_dataset_card

        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_audio_dataset(Path(tmpdir))
            card = generate_dataset_card(Path(tmpdir), "audio")

            assert card.startswith("---")
            lines = card.split("\n")
            end_idx = lines.index("---", 1)
            frontmatter = "\n".join(lines[: end_idx + 1])

            # Required frontmatter fields
            assert "annotations_creators:" in frontmatter
            assert "language:" in frontmatter
            assert "license:" in frontmatter
            assert "size_categories:" in frontmatter
            assert "task_categories:" in frontmatter
            assert "dataset_modality:" in frontmatter

    def test_audio_card_has_wavs_structure(self):
        """Audio card should show wavs/ structure in Dataset Structure section."""
        from datasety.upload import generate_dataset_card

        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_audio_dataset(Path(tmpdir))
            card = generate_dataset_card(Path(tmpdir), "audio")

            assert "wavs/" in card
            assert "metadata.csv" in card

    def test_audio_card_has_tts_task(self):
        """Audio card should list text-to-speech as a supported task."""
        from datasety.upload import generate_dataset_card

        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_audio_dataset(Path(tmpdir))
            card = generate_dataset_card(Path(tmpdir), "audio")

            assert "Text-to-Speech" in card

    def test_image_card_structure(self):
        """Image card should show train/test structure."""
        from datasety.upload import generate_dataset_card

        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "train").mkdir()
            (d / "test").mkdir()
            card = generate_dataset_card(d, "image")

            assert "train/" in card
            assert "test/" in card
            assert "Image Captioning" in card

    def test_video_card_has_video_modality(self):
        """Video card should list video modality."""
        from datasety.upload import generate_dataset_card

        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "video.mp4").touch()
            card = generate_dataset_card(d, "video")

            assert "Video Captioning" in card

    def test_extra_metadata_license(self):
        """--metadata license should override default mit license."""
        from datasety.upload import generate_dataset_card

        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            card = generate_dataset_card(d, "generic", extra_metadata={"license": "cc-by-4.0"})

            assert "cc-by-4.0" in card

    def test_extra_metadata_list_values(self):
        """List values in extra_metadata should render as YAML lists."""
        from datasety.upload import generate_dataset_card

        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            card = generate_dataset_card(d, "generic", extra_metadata={"language": ["en", "fr"]})

            # Should contain YAML list format
            assert "  - en" in card
            assert "  - fr" in card

    def test_size_category_small(self):
        """Datasets with < 1000 examples should use n<N format."""
        from datasety.upload import _size_category

        assert _size_category(100) == "n<100"
        assert _size_category(999) == "n<999"

    def test_size_category_thousands(self):
        """Datasets in 1k-10k range should use k-k+1 format."""
        from datasety.upload import _size_category

        assert _size_category(1000) == "1k-2k"
        assert _size_category(5500) == "5k-6k"
        assert _size_category(9999) == "9k-10k"

    def test_size_category_large(self):
        """Datasets > 100k should use n>N format."""
        from datasety.upload import _size_category

        assert _size_category(100000) == "n>100000"


class TestValidateDatasetStructure:
    """Test structure validation warnings."""

    def test_audio_missing_wavs(self):
        """Should warn when audio dataset lacks wavs/ directory."""
        from datasety.upload import validate_dataset_structure

        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "metadata.csv").touch()
            warnings = validate_dataset_structure(d, "audio")

            assert any("wavs" in w.lower() for w in warnings)

    def test_audio_missing_metadata(self):
        """Should warn when audio dataset lacks metadata.csv."""
        from datasety.upload import validate_dataset_structure

        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "wavs").mkdir()
            warnings = validate_dataset_structure(d, "audio")

            assert any("metadata" in w.lower() for w in warnings)

    def test_audio_complete_no_warnings(self):
        """Complete audio dataset should produce no warnings."""
        from datasety.upload import validate_dataset_structure

        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "wavs").mkdir()
            (d / "metadata.csv").touch()
            warnings = validate_dataset_structure(d, "audio")

            assert warnings == []

    def test_image_missing_subdirs(self):
        """Should warn when image dataset lacks class subdirs."""
        from datasety.upload import validate_dataset_structure

        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "photo.jpg").touch()
            warnings = validate_dataset_structure(d, "image")

            assert any("subdirectories" in w.lower() or "class" in w.lower() for w in warnings)

    def test_image_with_subdirs_no_warnings(self):
        """Image dataset with train/test subdirs should have no warnings."""
        from datasety.upload import validate_dataset_structure

        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "train").mkdir()
            (d / "test").mkdir()
            warnings = validate_dataset_structure(d, "image")

            assert warnings == []

    def test_generic_no_data_files(self):
        """Should warn when generic dataset has no common data files."""
        from datasety.upload import validate_dataset_structure

        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            # Use .md which is not in GENERIC_EXTENSIONS
            (d / "readme.md").touch()
            warnings = validate_dataset_structure(d, "generic")

            assert any("csv" in w.lower() or "json" in w.lower() for w in warnings)


class TestCmdUpload:
    """Test the main upload CLI command."""

    def test_dry_run_prints_summary(self, monkeypatch):
        """--dry-run should print summary without uploading."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "test.wav").touch()

            mock_args = MagicMock()
            mock_args.path = str(d)
            mock_args.repo_id = "test/user-dataset"
            mock_args.type = "auto"
            mock_args.private = False
            mock_args.dry_run = True
            mock_args.verbose = False
            mock_args.force = False
            mock_args.metadata = None
            mock_args.yes = True
            mock_args.token = None

            from datasety.upload import cmd_upload

            printed = []
            monkeypatch.setattr("sys.stdout.write", lambda s: printed.append(s))
            monkeypatch.setattr("sys.stderr", MagicMock())

            cmd_upload(mock_args)

            output = "".join(printed)
            assert "DRY RUN" in output or "dry-run" in output.lower()
            assert "test/user-dataset" in output

    def test_detect_type_auto(self, monkeypatch):
        """When type=auto, should detect from structure and print message."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "wavs").mkdir()
            (d / "metadata.csv").touch()

            mock_args = MagicMock()
            mock_args.path = str(d)
            mock_args.repo_id = "test/audio-dataset"
            mock_args.type = "auto"
            mock_args.private = False
            mock_args.dry_run = True
            mock_args.verbose = True
            mock_args.force = False
            mock_args.metadata = None
            mock_args.yes = True
            mock_args.token = None

            from datasety.upload import cmd_upload

            printed = []
            monkeypatch.setattr("sys.stdout.write", lambda s: printed.append(s))
            monkeypatch.setattr("sys.stderr", MagicMock())

            cmd_upload(mock_args)

            output = "".join(printed)
            assert "audio" in output.lower()

    def test_unknown_type_exits(self, monkeypatch):
        """Unknown --type should exit with error."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "test.wav").touch()

            mock_args = MagicMock()
            mock_args.path = str(d)
            mock_args.repo_id = "test/dataset"
            mock_args.type = "not_a_type"
            mock_args.private = False
            mock_args.dry_run = False
            mock_args.verbose = False
            mock_args.force = False
            mock_args.metadata = None
            mock_args.yes = True
            mock_args.token = None

            from datasety.upload import cmd_upload

            def mock_exit(code):
                raise SystemExit(code)

            monkeypatch.setattr("sys.exit", mock_exit)
            monkeypatch.setattr("sys.stderr", MagicMock())

            with pytest.raises(SystemExit) as exc_info:
                cmd_upload(mock_args)

            assert exc_info.value.code == 1

    def test_nonexistent_path_exits(self, monkeypatch):
        """Non-existent --path should exit with error."""
        mock_args = MagicMock()
        mock_args.path = "/this/path/does/not/exist/12345"
        mock_args.repo_id = "test/fake"
        mock_args.type = "auto"
        mock_args.private = False
        mock_args.dry_run = False
        mock_args.verbose = False
        mock_args.force = False
        mock_args.metadata = None
        mock_args.yes = True
        mock_args.token = None

        from datasety.upload import cmd_upload

        def mock_exit(code):
            raise SystemExit(code)

        monkeypatch.setattr("sys.exit", mock_exit)
        monkeypatch.setattr("sys.stderr", MagicMock())

        with pytest.raises(SystemExit) as exc_info:
            cmd_upload(mock_args)

        assert exc_info.value.code == 1

    def test_repo_id_derived_from_path(self, monkeypatch):
        """When --repo-id omitted, should derive from path name."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir) / "my-cool-dataset"
            d.mkdir()
            (d / "wavs").mkdir()
            (d / "metadata.csv").touch()

            mock_args = MagicMock()
            mock_args.path = str(d)
            mock_args.repo_id = None
            mock_args.type = "audio"
            mock_args.private = False
            mock_args.dry_run = True
            mock_args.verbose = False
            mock_args.force = False
            mock_args.metadata = None
            mock_args.yes = True
            mock_args.token = None

            from datasety.upload import cmd_upload

            printed = []
            monkeypatch.setattr("sys.stdout.write", lambda s: printed.append(s))
            monkeypatch.setattr("sys.stderr", MagicMock())

            cmd_upload(mock_args)

            output = "".join(printed)
            # Slugified path name
            assert "my-cool-dataset" in output or "my-cool" in output or "my" in output.lower()

    def test_metadata_yaml_parse_error(self, monkeypatch):
        """Invalid YAML in --metadata should exit with error."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "test.wav").touch()

            mock_args = MagicMock()
            mock_args.path = str(d)
            mock_args.repo_id = "test/dataset"
            mock_args.type = "audio"
            mock_args.private = False
            mock_args.dry_run = False
            mock_args.verbose = False
            mock_args.force = False
            mock_args.metadata = "not: valid: yaml: [broken"
            mock_args.yes = True
            mock_args.token = None

            from datasety.upload import cmd_upload

            exited = []
            monkeypatch.setattr("sys.exit", lambda c: exited.append(c))
            monkeypatch.setattr("sys.stderr", MagicMock())

            cmd_upload(mock_args)

            assert exited and exited[0] == 1

    def test_generates_readme(self, monkeypatch):
        """Should generate README.md in the dataset directory."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "wavs").mkdir()
            (d / "metadata.csv").touch()

            mock_args = MagicMock()
            mock_args.path = str(d)
            mock_args.repo_id = "test/audio-dataset"
            mock_args.type = "audio"
            mock_args.private = False
            mock_args.dry_run = True
            mock_args.verbose = False
            mock_args.force = False
            mock_args.metadata = None
            mock_args.yes = True
            mock_args.token = None

            from datasety.upload import cmd_upload

            monkeypatch.setattr("sys.stdout", MagicMock())
            monkeypatch.setattr("sys.stderr", MagicMock())

            cmd_upload(mock_args)

            readme = d / "README.md"
            assert readme.exists()
            content = readme.read_text()
            assert "---" in content  # YAML frontmatter
