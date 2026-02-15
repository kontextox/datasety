"""Shared test configuration and fixtures."""

import subprocess
import sys

import pytest
from PIL import Image


def run_cli(*args):
    """Run datasety CLI command and return the result."""
    return subprocess.run(
        [sys.executable, "-m", "datasety", *args],
        capture_output=True, text=True,
    )


def make_image(path, width, height, color=(255, 0, 0)):
    """Create a solid color test image."""
    img = Image.new("RGB", (width, height), color=color)
    img.save(path)
    return path


@pytest.fixture
def image_dir(tmp_path):
    """Create a temporary directory with test images."""
    d = tmp_path / "images"
    d.mkdir()
    for i in range(3):
        make_image(d / f"{i:03d}.jpg", 512, 512, color=(i * 80, 0, 0))
    return d


@pytest.fixture
def output_dir(tmp_path):
    """Create a temporary output directory."""
    d = tmp_path / "output"
    d.mkdir()
    return d
