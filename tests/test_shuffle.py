"""Tests for the shuffle command."""

import subprocess
import sys

import pytest
from PIL import Image


@pytest.fixture
def setup_dirs(tmp_path):
    """Create input directory with test images."""
    input_dir = tmp_path / "images"
    output_dir = tmp_path / "captions"
    input_dir.mkdir()
    output_dir.mkdir()

    for i in range(10):
        img = Image.new("RGB", (100, 100), color=(i * 25, 0, 0))
        img.save(input_dir / f"{i:03d}.jpg")

    return input_dir, output_dir


def run_shuffle(*args):
    """Run datasety shuffle and return the result."""
    return subprocess.run(
        [sys.executable, "-m", "datasety", "shuffle", *args],
        capture_output=True, text=True,
    )


class TestShuffleBasic:
    """Test basic shuffle functionality."""

    def test_generates_caption_files(self, setup_dirs):
        input_dir, output_dir = setup_dirs
        result = run_shuffle(
            "-i", str(input_dir), "-o", str(output_dir),
            "--group", "Hello.|Hey!", "--group", "World.|Earth!",
        )
        assert result.returncode == 0
        assert "10 captions generated" in result.stdout

        txt_files = list(output_dir.glob("*.txt"))
        assert len(txt_files) == 10

    def test_caption_content_from_groups(self, setup_dirs):
        input_dir, output_dir = setup_dirs
        result = run_shuffle(
            "-i", str(input_dir), "-o", str(output_dir),
            "--group", "A|B", "--group", "X|Y",
            "--seed", "42",
        )
        assert result.returncode == 0

        for txt_file in output_dir.glob("*.txt"):
            caption = txt_file.read_text()
            parts = caption.split(" ")
            assert len(parts) == 2
            assert parts[0] in ("A", "B")
            assert parts[1] in ("X", "Y")

    def test_single_group(self, setup_dirs):
        input_dir, output_dir = setup_dirs
        result = run_shuffle(
            "-i", str(input_dir), "-o", str(output_dir),
            "--group", "Only option",
        )
        assert result.returncode == 0

        for txt_file in output_dir.glob("*.txt"):
            assert txt_file.read_text() == "Only option"


class TestShuffleSeed:
    """Test seed reproducibility."""

    def test_same_seed_same_results(self, setup_dirs):
        input_dir, output_dir = setup_dirs
        output_dir2 = input_dir.parent / "captions2"
        output_dir2.mkdir()

        common_args = [
            "-i", str(input_dir),
            "--group", "A|B|C", "--group", "X|Y|Z",
            "--seed", "123",
        ]

        run_shuffle(*common_args, "-o", str(output_dir))
        run_shuffle(*common_args, "-o", str(output_dir2))

        for txt_file in sorted(output_dir.glob("*.txt")):
            txt_file2 = output_dir2 / txt_file.name
            assert txt_file.read_text() == txt_file2.read_text()

    def test_different_seed_different_results(self, setup_dirs):
        input_dir, output_dir = setup_dirs
        output_dir2 = input_dir.parent / "captions2"
        output_dir2.mkdir()

        common_args = [
            "-i", str(input_dir),
            "--group", "A|B|C|D|E|F|G|H",
            "--group", "X|Y|Z|W|V|U|T|S",
        ]

        run_shuffle(*common_args, "-o", str(output_dir), "--seed", "1")
        run_shuffle(*common_args, "-o", str(output_dir2), "--seed", "2")

        results1 = [f.read_text() for f in sorted(output_dir.glob("*.txt"))]
        results2 = [f.read_text() for f in sorted(output_dir2.glob("*.txt"))]
        assert results1 != results2


class TestShuffleDryRun:
    """Test dry-run mode."""

    def test_dry_run_no_files(self, setup_dirs):
        input_dir, output_dir = setup_dirs
        result = run_shuffle(
            "-i", str(input_dir), "-o", str(output_dir),
            "--group", "A|B", "--dry-run",
        )
        assert result.returncode == 0
        assert "DRY RUN" in result.stdout
        assert len(list(output_dir.glob("*.txt"))) == 0


class TestShuffleSeparator:
    """Test custom separator."""

    def test_custom_separator(self, setup_dirs):
        input_dir, output_dir = setup_dirs
        result = run_shuffle(
            "-i", str(input_dir), "-o", str(output_dir),
            "--group", "Hello", "--group", "World",
            "--separator", ", ",
        )
        assert result.returncode == 0

        for txt_file in output_dir.glob("*.txt"):
            assert txt_file.read_text() == "Hello, World"


class TestShuffleDistribution:
    """Test show-distribution flag."""

    def test_show_distribution(self, setup_dirs):
        input_dir, output_dir = setup_dirs
        result = run_shuffle(
            "-i", str(input_dir), "-o", str(output_dir),
            "--group", "A|B", "--group", "X",
            "--show-distribution", "--seed", "42",
        )
        assert result.returncode == 0
        assert "Caption distribution:" in result.stdout
        assert "x:" in result.stdout


class TestShuffleErrors:
    """Test error handling."""

    def test_no_groups(self, setup_dirs):
        input_dir, output_dir = setup_dirs
        result = run_shuffle(
            "-i", str(input_dir), "-o", str(output_dir),
        )
        assert result.returncode != 0

    def test_missing_input_dir(self, tmp_path):
        result = run_shuffle(
            "-i", str(tmp_path / "nonexistent"), "-o", str(tmp_path),
            "--group", "A|B",
        )
        assert result.returncode != 0
