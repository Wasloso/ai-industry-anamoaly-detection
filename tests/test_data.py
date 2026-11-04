import json
import shutil
from dataclasses import replace

import pytest
import torch
from PIL import Image

from defect_detection.data.mvtec import MVTecDataset, build_manifest, make_loader, prepare_data


def test_splits_are_reproducible_and_keep_test_images(data_config):
    manifest = prepare_data(data_config)
    assert prepare_data(data_config) == manifest
    assert json.loads(data_config.manifest.read_text()) == manifest
    assert len([s for s in manifest["samples"] if s["split"] == "train"]) == 10
    assert len([s for s in manifest["samples"] if s["split"] == "test"]) == 2
    split_hashes = {}
    for sample in manifest["samples"]:
        split_hashes.setdefault(sample["split"], set()).add(sample["image_hash"])
        if sample["split"] != "test":
            assert sample["label"] == 0
            assert sample["mask_path"] is None
    for left, hashes in split_hashes.items():
        for right, other in split_hashes.items():
            if left != right:
                assert hashes.isdisjoint(other)


def test_duplicate_normals_stay_in_one_partition(data_config):
    good = data_config.root / "bottle/train/good"
    shutil.copyfile(good / "000.png", good / "copy.png")
    samples = build_manifest(data_config)["samples"]
    duplicates = [
        s
        for s in samples
        if s["image_path"].endswith(("/000.png", "/copy.png")) and "/train/" in s["image_path"]
    ]
    assert len(duplicates) == 2
    assert len({s["split"] for s in duplicates}) == 1


def test_train_test_duplicate_is_rejected(data_config):
    shutil.copyfile(
        data_config.root / "bottle/train/good/000.png",
        data_config.root / "bottle/test/good/000.png",
    )
    with pytest.raises(ValueError, match="both training and test"):
        build_manifest(data_config)


@pytest.mark.parametrize("problem", ["missing", "wrong_size", "empty", "not_binary"])
def test_invalid_mask_is_rejected(data_config, defect_mask, problem):
    if problem == "missing":
        defect_mask.unlink()
    else:
        size = (10, 10) if problem == "wrong_size" else (40, 24)
        value = 0 if problem == "empty" else 128 if problem == "not_binary" else 255
        Image.new("L", size, value).save(defect_mask)
    with pytest.raises(ValueError):
        build_manifest(data_config)


def test_corrupt_image_is_rejected(data_config):
    (data_config.root / "bottle/train/good/000.png").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Cannot read image"):
        build_manifest(data_config)


def test_existing_manifest_cannot_silently_change(data_config):
    prepare_data(data_config)
    with pytest.raises(ValueError, match="choose a new manifest"):
        prepare_data(replace(data_config, seed=99))


def test_dataset_resizes_images_and_masks_without_normalizing(data_config):
    manifest = build_manifest(data_config)
    dataset = MVTecDataset(data_config, manifest, "test")
    defect = next(dataset[i] for i, s in enumerate(dataset.samples) if s["label"] == 1)
    assert defect["image"].shape == (3, 20, 20)
    assert defect["image"].dtype == torch.float32
    assert 0 <= defect["image"].min() <= defect["image"].max() <= 1
    assert defect["original_size"].tolist() == [24, 40]
    assert defect["mask"].dtype == torch.bool
    expected = torch.zeros(20, 20, dtype=torch.bool)
    expected[5:15, 5:15] = True
    assert torch.equal(defect["mask"], expected)
    normal = next(dataset[i] for i, s in enumerate(dataset.samples) if s["label"] == 0)
    assert not normal["mask"].any()


def test_loader_returns_repeatable_batches(data_config):
    manifest = build_manifest(data_config)
    first = next(iter(make_loader(data_config, manifest, "train")))
    second = next(iter(make_loader(data_config, manifest, "train")))
    assert first["image"].shape == (8, 3, 20, 20)
    assert first["mask"].shape == (8, 20, 20)
    assert first["sample_id"] == second["sample_id"]
