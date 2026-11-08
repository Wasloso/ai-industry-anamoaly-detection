from pathlib import Path

import torch

from defect_detection.config import load_training_config
from defect_detection.data.mvtec import make_loader, prepare_data
from defect_detection.models.artifacts import save_artifact
from defect_detection.models.patchcore import PatchCore


def train(config: Path, run_id: str, artifacts_dir=Path("artifacts"), device="auto") -> dict:
    if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run_id must be a single directory name")
    path = artifacts_dir / run_id
    if path.exists():
        raise ValueError(f"Run already exists: {path}. Choose a new run_id")
    data, settings = load_training_config(config)
    manifest = prepare_data(data)
    torch.manual_seed(data.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = PatchCore(settings, data.image_size, device)
    report = model.fit(make_loader(data, manifest, "train"))
    save_artifact(model, path, data, manifest, report)
    return {"artifact": str(path.resolve()), **report}
