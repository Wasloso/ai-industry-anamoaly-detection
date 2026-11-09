import json

import torch

from defect_detection.config import PatchCoreConfig
from defect_detection.models.patchcore import PatchCore

torch.manual_seed(42)
torch.set_num_threads(4)
model = PatchCore(PatchCoreConfig(coreset_ratio=0.1), image_size=64, pretrained=False)
images = torch.rand(2, 3, 64, 64)
report = model.fit([{"image": images, "label": torch.zeros(2, dtype=torch.long)}])
scores, maps = model.predict(images)
assert scores.shape == (2,) and maps.shape == (2, 1, 64, 64)
assert torch.isfinite(scores).all() and torch.isfinite(maps).all()
print(
    json.dumps({"check": "synthetic execution only; random backbone weights", **report}, indent=2)
)
