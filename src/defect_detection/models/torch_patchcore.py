import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import get_model


def select_coreset(features, ratio):
    count = int(len(features) * ratio)
    if not 1 <= count <= len(features):
        raise ValueError("Not enough patches for this coreset_ratio")
    if count == len(features):
        return features.clone()
    projected = features
    if features.shape[1] > 128:
        projection = torch.randn(features.shape[1], 128, device=features.device) / math.sqrt(128)
        projected = features @ projection
    distances = torch.full((len(features),), float("inf"), device=features.device)
    selected = torch.empty(count, dtype=torch.long, device=features.device)
    current = torch.randint(len(features), (), device=features.device).item()
    for index in range(count):
        selected[index] = current
        candidate_distances = (projected - projected[current]).square().sum(dim=1)
        distances = torch.minimum(distances, candidate_distances)
        distances[current] = -1
        current = distances.argmax().item()
    return features[selected].clone()


class PatchCoreModel(nn.Module):
    def __init__(self, backbone, layers, pretrained=True, num_neighbors=9):
        super().__init__()
        available = ("layer1", "layer2", "layer3", "layer4")
        if not layers or len(set(layers)) != len(layers) or set(layers) - set(available):
            raise ValueError("Choose distinct ResNet layers from layer1 to layer4")
        if backbone not in ("resnet18", "resnet34", "resnet50", "wide_resnet50_2"):
            raise ValueError("Unsupported ResNet backbone")
        network = get_model(backbone, weights="DEFAULT" if pretrained else None)
        self.layers = set(layers)
        last = max(layers, key=available.index)
        self.feature_extractor = nn.ModuleDict()
        for name, module in network.named_children():
            self.feature_extractor[name] = module
            if name == last:
                break
        self.feature_extractor.requires_grad_(False)
        self.feature_extractor.eval()
        self.num_neighbors = num_neighbors
        self.register_buffer("memory_bank", torch.empty(0, 0))

    def embed(self, images):
        self.feature_extractor.eval()
        features = []
        for name, layer in self.feature_extractor.items():
            images = layer(images)
            if name in self.layers:
                features.append(F.avg_pool2d(images, kernel_size=3, stride=1, padding=1))
        size = features[0].shape[-2:]
        aligned = [
            F.interpolate(feature, size=size, mode="bilinear", align_corners=False)
            for feature in features
        ]
        return torch.cat(aligned, dim=1)

    def nearest_neighbors(self, queries, count=1):
        if not self.memory_bank.numel():
            raise ValueError("Fit or load a model before predicting")
        distances, indices = [], []
        count = min(count, len(self.memory_bank))
        for chunk in queries.split(1024):
            values, positions = torch.cdist(chunk, self.memory_bank).topk(count, largest=False)
            distances.append(values)
            indices.append(positions)
        return torch.cat(distances), torch.cat(indices)

    def forward(self, images):
        features = self.embed(images)
        batch, channels, height, width = features.shape
        patches = features.permute(0, 2, 3, 1).reshape(-1, channels)
        distances, neighbors = self.nearest_neighbors(patches)
        patch_scores = distances.reshape(batch, height * width)
        scores, worst = patch_scores.max(dim=1)
        count = min(self.num_neighbors, len(self.memory_bank))
        if count > 1:
            rows = torch.arange(batch, device=images.device)
            nearest = neighbors.reshape(batch, -1)[rows, worst]
            _, support = self.nearest_neighbors(self.memory_bank[nearest], count)
            queries = patches.reshape(batch, -1, channels)[rows, worst]
            support_distances = torch.linalg.vector_norm(
                queries[:, None] - self.memory_bank[support], dim=-1
            )
            weights = 1 - torch.exp(scores - torch.logsumexp(support_distances, dim=1))
            scores = scores * weights.clamp(0, 1)
        maps = F.interpolate(
            patch_scores.reshape(batch, 1, height, width),
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        axis = torch.arange(-16, 17, dtype=maps.dtype, device=maps.device)
        kernel = torch.exp(-axis.square() / (2 * 4**2))
        kernel = kernel / kernel.sum()
        maps = F.conv2d(F.pad(maps, (16, 16, 0, 0), mode="replicate"), kernel.view(1, 1, 1, -1))
        maps = F.conv2d(F.pad(maps, (0, 0, 16, 16), mode="replicate"), kernel.view(1, 1, -1, 1))
        return scores, maps
