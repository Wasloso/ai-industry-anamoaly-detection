import json
import time

import torch
from anomalib.models import EfficientAd, Patchcore

torch.manual_seed(42)
torch.set_num_threads(4)
device = "cuda" if torch.cuda.is_available() else "cpu"

for name in ("patchcore", "efficientad"):
    model = (
        Patchcore(backbone="resnet18", coreset_sampling_ratio=0.01)
        if name == "patchcore"
        else EfficientAd(model_size="small")
    ).to(device)
    if name == "efficientad":
        model.prepare_pretrained_model()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    if name == "efficientad":
        model.model.train()
        optimizer = torch.optim.Adam(
            list(model.model.student.parameters()) + list(model.model.ae.parameters()), lr=1e-4
        )
        loss = sum(
            model.model(
                torch.rand(1, 3, 256, 256, device=device),
                torch.rand(1, 3, 256, 256, device=device),
            )
        )
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        del optimizer, loss
    with torch.no_grad():
        if name == "patchcore":
            model.model.train()
            model.model.feature_extractor.eval()
            model.model(torch.rand(2, 3, 256, 256, device=device))
            model.model.subsample_embedding(0.01)
        model.model.eval()
        result = model.model(torch.rand(1, 3, 256, 256, device=device))
    assert torch.isfinite(result.pred_score).all()
    assert torch.isfinite(result.anomaly_map).all()
    if device == "cuda":
        torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "check": "synthetic smoke test",
                "model": name,
                "device": device,
                "score_shape": list(result.pred_score.shape),
                "map_shape": list(result.anomaly_map.shape),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2)
                if device == "cuda"
                else None,
            }
        )
    )
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
