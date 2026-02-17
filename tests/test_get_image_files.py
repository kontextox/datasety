"""Tests for the get_image_files utility function."""

from PIL import Image

from datasety.common import get_image_files


class TestGetImageFiles:
    """Test image file discovery."""

    def test_finds_jpg_files(self, tmp_path):
        for i in range(3):
            Image.new("RGB", (10, 10)).save(tmp_path / f"{i}.jpg")
        files = get_image_files(tmp_path, ["jpg"])
        assert len(files) == 3

    def test_finds_multiple_formats(self, tmp_path):
        Image.new("RGB", (10, 10)).save(tmp_path / "a.jpg")
        Image.new("RGB", (10, 10)).save(tmp_path / "b.png")
        Image.new("RGB", (10, 10)).save(tmp_path / "c.webp")
        files = get_image_files(tmp_path, ["jpg", "png", "webp"])
        assert len(files) == 3

    def test_returns_sorted(self, tmp_path):
        for name in ["c", "a", "b"]:
            Image.new("RGB", (10, 10)).save(tmp_path / f"{name}.jpg")
        files = get_image_files(tmp_path, ["jpg"])
        names = [f.stem for f in files]
        assert names == sorted(names)

    def test_no_duplicates(self, tmp_path):
        Image.new("RGB", (10, 10)).save(tmp_path / "a.jpg")
        files = get_image_files(tmp_path, ["jpg", "jpg"])
        assert len(files) == 1

    def test_empty_directory(self, tmp_path):
        files = get_image_files(tmp_path, ["jpg", "png"])
        assert files == []

    def test_ignores_non_matching_formats(self, tmp_path):
        Image.new("RGB", (10, 10)).save(tmp_path / "a.jpg")
        (tmp_path / "b.txt").write_text("not an image")
        files = get_image_files(tmp_path, ["jpg"])
        assert len(files) == 1

    def test_case_insensitive(self, tmp_path):
        Image.new("RGB", (10, 10)).save(tmp_path / "a.jpg")
        Image.new("RGB", (10, 10)).save(tmp_path / "b.JPG")
        files = get_image_files(tmp_path, ["jpg"])
        assert len(files) == 2

    def test_mixed_case_extensions(self, tmp_path):
        Image.new("RGB", (10, 10)).save(tmp_path / "a.jpg")
        Image.new("RGB", (10, 10)).save(tmp_path / "b.Jpg")
        Image.new("RGB", (10, 10)).save(tmp_path / "c.PNG")
        Image.new("RGB", (10, 10)).save(tmp_path / "d.Jpeg")
        files = get_image_files(tmp_path, ["jpg", "jpeg", "png"])
        assert len(files) == 4

    def test_recursive(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        Image.new("RGB", (10, 10)).save(tmp_path / "a.jpg")
        Image.new("RGB", (10, 10)).save(sub / "b.jpg")
        files = get_image_files(tmp_path, ["jpg"], recursive=False)
        assert len(files) == 1
        files = get_image_files(tmp_path, ["jpg"], recursive=True)
        assert len(files) == 2
