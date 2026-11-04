from PIL import Image

from defect_detection.data.mvtec import build_manifest
from defect_detection.data.preview import save_preview


def test_preview_shows_both_classes_and_aligned_mask(data_config, tmp_path):
    output = tmp_path / "preview.png"
    save_preview(data_config, build_manifest(data_config), output, limit=2)
    with Image.open(output) as image:
        assert image.size == (60, 92)
        assert image.getpixel((30, 36)) == (255, 255, 255)
        assert image.getpixel((50, 36)) == (227, 50, 40)
        assert image.getpixel((30, 82)) == (0, 0, 0)
