"""Tests for the resize command."""

import pytest

from datasety.resize import _resolution_from_megapixel, calculate_resize_and_crop


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


class TestResolutionFromMegapixel:
    """Test megapixel + aspect ratio resolution calculation."""

    def test_1mp_square(self):
        """1.0 MP at 1:1 should be 1000x1000 (already a multiple of 8)."""
        w, h = _resolution_from_megapixel(1.0, "1:1")
        assert w == 1000
        assert h == 1000
        assert w % 8 == 0
        assert h % 8 == 0

    def test_05mp_square(self):
        """0.5 MP at 1:1 -> ~707x707 -> rounded to 704x704."""
        w, h = _resolution_from_megapixel(0.5, "1:1")
        assert w == h
        assert w == 704  # round(707.1/8)*8 = 704
        assert w % 8 == 0

    def test_05mp_16_9(self):
        """0.5 MP at 16:9."""
        w, h = _resolution_from_megapixel(0.5, "16:9")
        assert w > h  # landscape
        assert w % 8 == 0
        assert h % 8 == 0
        # Check approximately right: w*h ~ 500000
        assert abs(w * h - 500_000) < 10_000

    def test_05mp_9_16(self):
        """0.5 MP at 9:16 should swap dimensions vs 16:9."""
        w1, h1 = _resolution_from_megapixel(0.5, "16:9")
        w2, h2 = _resolution_from_megapixel(0.5, "9:16")
        assert w2 == h1
        assert h2 == w1

    def test_1mp_3_2(self):
        """1.0 MP at 3:2."""
        w, h = _resolution_from_megapixel(1.0, "3:2")
        assert w > h
        assert w % 8 == 0
        assert h % 8 == 0
        assert abs(w * h - 1_000_000) < 10_000

    def test_results_are_multiples_of_8(self):
        """All results should be multiples of 8."""
        for mp in [0.25, 0.5, 1.0, 2.0]:
            for ratio in ["1:1", "16:9", "4:3", "3:2"]:
                w, h = _resolution_from_megapixel(mp, ratio)
                assert w % 8 == 0, f"width {w} not multiple of 8 for {mp}MP {ratio}"
                assert h % 8 == 0, f"height {h} not multiple of 8 for {mp}MP {ratio}"
