"""Tests for the resize command."""

import pytest
from datasety.cli import calculate_resize_and_crop, get_image_files
from pathlib import Path


class TestCalculateResizeAndCrop:
    """Test resize and crop calculations."""

    def test_wider_image_center_crop(self):
        """Test cropping a wider image from center."""
        # 2000x1000 image -> 1024x1024 target
        (new_w, new_h), (left, top, right, bottom) = calculate_resize_and_crop(
            2000, 1000, 1024, 1024, "center"
        )
        assert new_h == 1024
        assert new_w == 2048  # maintains aspect ratio
        assert top == 0
        assert bottom == 1024
        assert left == (2048 - 1024) // 2
        assert right == left + 1024

    def test_taller_image_center_crop(self):
        """Test cropping a taller image from center."""
        # 1000x2000 image -> 1024x1024 target
        (new_w, new_h), (left, top, right, bottom) = calculate_resize_and_crop(
            1000, 2000, 1024, 1024, "center"
        )
        assert new_w == 1024
        assert new_h == 2048
        assert left == 0
        assert right == 1024
        assert top == (2048 - 1024) // 2
        assert bottom == top + 1024

    def test_top_crop(self):
        """Test cropping from top."""
        (new_w, new_h), (left, top, right, bottom) = calculate_resize_and_crop(
            1000, 2000, 1024, 1024, "top"
        )
        assert top == 0
        assert bottom == 1024

    def test_bottom_crop(self):
        """Test cropping from bottom."""
        (new_w, new_h), (left, top, right, bottom) = calculate_resize_and_crop(
            1000, 2000, 1024, 1024, "bottom"
        )
        assert bottom == new_h
        assert top == new_h - 1024

    def test_left_crop(self):
        """Test cropping from left."""
        (new_w, new_h), (left, top, right, bottom) = calculate_resize_and_crop(
            2000, 1000, 1024, 1024, "left"
        )
        assert left == 0
        assert right == 1024

    def test_right_crop(self):
        """Test cropping from right."""
        (new_w, new_h), (left, top, right, bottom) = calculate_resize_and_crop(
            2000, 1000, 1024, 1024, "right"
        )
        assert right == new_w
        assert left == new_w - 1024

    def test_non_square_target(self):
        """Test with non-square target resolution."""
        # 2000x1500 image -> 768x1024 target (portrait)
        # orig_ratio=1.33 > target_ratio=0.75, so resize by height
        (new_w, new_h), (left, top, right, bottom) = calculate_resize_and_crop(
            2000, 1500, 768, 1024, "center"
        )
        assert new_h == 1024
        assert new_w == int(2000 * (1024 / 1500))  # 1365
        assert right - left == 768
        assert bottom - top == 1024

    def test_invalid_crop_position(self):
        """Test that invalid crop position raises error."""
        with pytest.raises(ValueError):
            calculate_resize_and_crop(1000, 1000, 512, 512, "invalid")
