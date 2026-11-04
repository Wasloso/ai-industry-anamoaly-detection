from pathlib import Path

from PIL import Image, ImageDraw

from defect_detection.config import DataConfig


def save_preview(config: DataConfig, manifest: dict, output: Path, split="test", limit=8):
    if limit < 1:
        raise ValueError("Preview limit must be positive")
    samples = [sample for sample in manifest["samples"] if sample["split"] == split]
    if not samples:
        raise ValueError(f"No samples in split: {split}")
    examples = {}
    for sample in samples:
        examples.setdefault(Path(sample["image_path"]).parent.name, sample)
    selected = list(examples.values())
    selected += [sample for sample in samples if sample not in selected]
    selected = selected[:limit]

    size = config.image_size
    row_height = size + 26
    canvas = Image.new("RGB", (size * 3, row_height * len(selected)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, sample in enumerate(selected):
        with Image.open(config.root / sample["image_path"]) as image:
            image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        mask = Image.new("L", image.size)
        if sample["mask_path"]:
            with Image.open(config.root / sample["mask_path"]) as source:
                mask = source.convert("L").resize(image.size, Image.Resampling.NEAREST)
                mask = mask.point(lambda value: 255 if value else 0)
        tint = Image.new("RGB", image.size, "red")
        overlay = Image.composite(Image.blend(image, tint, 0.5), image, mask)
        y = index * row_height
        draw.text((4, y + 4), sample["image_path"], fill="black")
        for column, tile in enumerate((image, mask.convert("RGB"), overlay)):
            canvas.paste(tile, (column * size, y + 26))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
