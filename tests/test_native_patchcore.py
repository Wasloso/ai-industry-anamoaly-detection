import math
from dataclasses import replace

import torch
from torchvision.models import _api as model_api

from defect_detection.config import PatchCoreConfig
from defect_detection.data.mvtec import make_loader, prepare_data
from defect_detection.models.artifacts import load_artifact, save_artifact
from defect_detection.models.patchcore import PatchCore
from defect_detection.models.torch_patchcore import PatchCoreModel, select_coreset


def test_coreset_covers_separated_groups_and_keeps_original_features():
    torch.manual_seed(42)
    features = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0], [11.0, 0.0], [12.0, 0.0]]
    )
    selected = select_coreset(features, 0.5)
    assert selected.shape == (3, 2)
    assert len(selected.unique(dim=0)) == 3
    assert all((features == row).all(dim=1).any() for row in selected)
    assert torch.cdist(features, selected).min(dim=1).values.max() <= 2


def test_nearest_neighbors_and_reweighted_score_match_known_distances(monkeypatch):
    torch.set_num_threads(2)
    model = PatchCoreModel("resnet18", ["layer2", "layer3"], pretrained=False, num_neighbors=3)
    model.memory_bank = torch.tensor([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    queries = torch.tensor([[0.0, 4.0], [6.0, 8.0]]).repeat(600, 1)
    distances, indices = model.nearest_neighbors(queries)
    torch.testing.assert_close(distances[:, 0], torch.tensor([3.0, 0.0]).repeat(600))
    assert torch.equal(indices[:, 0], torch.tensor([1, 2]).repeat(600))
    features = torch.tensor([[[[0.0, 6.0]], [[4.0, 8.0]]]])
    monkeypatch.setattr(model, "embed", lambda _: features)
    scores, maps = model(torch.zeros(1, 3, 16, 16))
    expected = 3 * (1 - math.exp(3) / (math.exp(4) + math.exp(3) + math.exp(math.sqrt(52))))
    torch.testing.assert_close(scores, torch.tensor([expected]))
    assert maps.shape == (1, 1, 16, 16)
    assert torch.isfinite(maps).all()


def test_fitted_artifact_reloads_without_downloading_weights(data_config, tmp_path, monkeypatch):
    torch.set_num_threads(2)
    data = replace(data_config, image_size=32)
    manifest = prepare_data(data)
    model = PatchCore(PatchCoreConfig(coreset_ratio=0.5), 32, device="cpu", pretrained=False)
    report = model.fit(make_loader(data, manifest, "train"))
    images = next(iter(make_loader(data, manifest, "test")))["image"]
    expected_scores, expected_maps = model.predict(images)
    artifact = tmp_path / "model"
    save_artifact(model, artifact, data, manifest, report)

    def reject_download(*args, **kwargs):
        raise AssertionError("Reload must not download weights")

    monkeypatch.setattr(model_api, "load_state_dict_from_url", reject_download)
    restored, metadata = load_artifact(artifact, device="cpu")
    scores, maps = restored.predict(images)
    assert metadata["implementation"] == "project-pytorch"
    assert report["fitting_images"] == 10
    assert not any(parameter.requires_grad for parameter in restored.model.parameters())
    torch.testing.assert_close(scores, expected_scores)
    torch.testing.assert_close(maps, expected_maps)
