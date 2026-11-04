import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor

from defect_detection.config import DataConfig

SPLITS = ("train", "validation", "calibration", "test")


@dataclass
class Sample:
    image_path: str
    mask_path: str | None
    split: str
    label: int
    width: int
    height: int
    image_hash: str
    mask_hash: str | None


def inspect_image(path: Path, mode: str) -> tuple[int, int, str]:
    try:
        with Image.open(path) as image:
            image = image.convert(mode)
            digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            return image.width, image.height, digest
    except OSError as error:
        raise ValueError(f"Cannot read image: {path}") from error


def build_manifest(config: DataConfig) -> dict:
    category_root = config.root / config.category
    normal_dir = category_root / "train" / "good"
    test_dir = category_root / "test"
    if not normal_dir.is_dir() or not test_dir.is_dir():
        raise ValueError(f"Expected train/good and test directories under {category_root}")
    unexpected = [p for p in (category_root / "train").iterdir() if p.is_dir() and p.name != "good"]
    if unexpected:
        raise ValueError("Training data must contain only the good category")

    train_paths = sorted(normal_dir.glob("*.png"))
    train_set = set(train_paths)
    test_paths = sorted(test_dir.glob("*/*.png"))
    if not train_paths or not test_paths:
        raise ValueError("Training and test directories must both contain PNG images")

    samples = []
    for path in train_paths + test_paths:
        is_train = path in train_set
        label = int(not is_train and path.parent.name != "good")
        width, height, image_hash = inspect_image(path, "RGB")
        mask_path = None
        mask_hash = None
        if label:
            mask = category_root / "ground_truth" / path.parent.name / f"{path.stem}_mask.png"
            if not mask.is_file():
                raise ValueError(f"Missing defect mask: {mask}")
            mask_width, mask_height, mask_hash = inspect_image(mask, "L")
            if (width, height) != (mask_width, mask_height):
                raise ValueError(f"Image and mask dimensions differ: {path}")
            with Image.open(mask) as image:
                values = {
                    value for value, count in enumerate(image.convert("L").histogram()) if count
                }
            if not values <= {0, 1, 255} or not values - {0}:
                raise ValueError(f"Expected a non-empty binary defect mask: {mask}")
            mask_path = mask.relative_to(config.root).as_posix()
        samples.append(
            Sample(
                image_path=path.relative_to(config.root).as_posix(),
                mask_path=mask_path,
                split="train" if is_train else "test",
                label=label,
                width=width,
                height=height,
                image_hash=image_hash,
                mask_hash=mask_hash,
            )
        )

    normal_groups = defaultdict(list)
    train_hashes = {sample.image_hash for sample in samples if sample.split == "train"}
    for sample in samples:
        if sample.split == "train":
            normal_groups[sample.image_hash].append(sample)
        elif sample.image_hash in train_hashes:
            raise ValueError(f"Image occurs in both training and test data: {sample.image_path}")

    hashes = sorted(normal_groups)
    random.Random(config.seed).shuffle(hashes)
    n_validation = max(1, int(len(hashes) * config.validation_ratio))
    n_calibration = max(1, int(len(hashes) * config.calibration_ratio))
    if n_validation + n_calibration >= len(hashes):
        raise ValueError("Not enough distinct normal images for train, validation, and calibration")
    for index, digest in enumerate(hashes):
        split = "train"
        if index < n_validation:
            split = "validation"
        elif index < n_validation + n_calibration:
            split = "calibration"
        for sample in normal_groups[digest]:
            sample.split = split

    return {
        "version": 1,
        "dataset": "mvtec_ad",
        "source": "https://www.mvtec.com/research-teaching/datasets/mvtec-ad",
        "category": config.category,
        "seed": config.seed,
        "validation_ratio": config.validation_ratio,
        "calibration_ratio": config.calibration_ratio,
        "samples": [asdict(sample) for sample in samples],
    }


def prepare_data(config: DataConfig) -> dict:
    manifest = build_manifest(config)
    if config.manifest.exists():
        saved = json.loads(config.manifest.read_text())
        if saved != manifest:
            raise ValueError("Data or split settings changed; choose a new manifest path")
    else:
        config.manifest.parent.mkdir(parents=True, exist_ok=True)
        config.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def summarize(manifest: dict) -> dict:
    return {
        split: dict(
            Counter(
                "defect" if s["label"] else "normal"
                for s in manifest["samples"]
                if s["split"] == split
            )
        )
        for split in SPLITS
    }


class MVTecDataset(Dataset):
    def __init__(self, config: DataConfig, manifest: dict, split: str):
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        self.root = config.root
        self.image_size = config.image_size
        self.samples = [sample for sample in manifest["samples"] if sample["split"] == split]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        size = (self.image_size, self.image_size)
        with Image.open(self.root / sample["image_path"]) as image:
            image = image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
            tensor = pil_to_tensor(image).float() / 255
        mask = torch.zeros(self.image_size, self.image_size, dtype=torch.bool)
        if sample["mask_path"]:
            with Image.open(self.root / sample["mask_path"]) as image:
                image = image.convert("L").resize(size, Image.Resampling.NEAREST)
                mask = pil_to_tensor(image).squeeze(0) > 0
        return {
            "image": tensor,
            "mask": mask,
            "label": sample["label"],
            "sample_id": sample["image_path"],
            "original_size": torch.tensor([sample["height"], sample["width"]]),
        }


def make_loader(config: DataConfig, manifest: dict, split: str) -> DataLoader:
    return DataLoader(
        MVTecDataset(config, manifest, split),
        batch_size=config.batch_size,
        shuffle=split == "train",
        num_workers=config.num_workers,
        generator=torch.Generator().manual_seed(config.seed),
    )
