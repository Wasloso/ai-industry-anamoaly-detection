import json
import shutil

import pytest
import torch

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


def test_train_test_duplicate_is_rejected(data_config):
    shutil.copyfile(
        data_config.root / "bottle/train/good/000.png",
        data_config.root / "bottle/test/good/000.png",
    )
    with pytest.raises(ValueError, match="both training and test"):
        build_manifest(data_config)


def test_loader_preserves_image_and_mask_data(data_config):
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

    batch = next(iter(make_loader(data_config, manifest, "train")))
    assert batch["image"].shape == (8, 3, 20, 20)
    assert batch["mask"].shape == (8, 20, 20)
