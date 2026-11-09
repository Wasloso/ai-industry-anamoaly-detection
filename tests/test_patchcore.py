import math

import torch

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
