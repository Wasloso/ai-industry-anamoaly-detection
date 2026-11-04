import pytest
from PIL import Image, ImageDraw

from defect_detection.config import DataConfig


@pytest.fixture
def data_config(tmp_path):
    root = tmp_path / "mvtec"
    for folder in ("train/good", "test/good", "test/crack", "ground_truth/crack"):
        (root / "bottle" / folder).mkdir(parents=True)
    for index in range(12):
        Image.new("RGB", (40, 24), (index * 10, 80, 100)).save(
            root / "bottle/train/good" / f"{index:03}.png"
        )
    Image.new("RGB", (40, 24), (180, 100, 80)).save(root / "bottle/test/good/000.png")
    Image.new("RGB", (40, 24), (200, 100, 80)).save(root / "bottle/test/crack/000.png")
    mask = Image.new("L", (40, 24))
    ImageDraw.Draw(mask).rectangle((10, 6, 29, 17), fill=255)
    mask.save(root / "bottle/ground_truth/crack/000_mask.png")
    return DataConfig(
        root=root, category="bottle", manifest=tmp_path / "manifest.json", image_size=20
    )
