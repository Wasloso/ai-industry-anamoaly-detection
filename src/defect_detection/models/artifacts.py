import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch

from defect_detection.config import DataConfig, PatchCoreConfig
from defect_detection.models.patchcore import MEAN, STD, PatchCore


def manifest_hash(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def save_artifact(model: PatchCore, path: Path, data: DataConfig, manifest: dict, report: dict):
    if not model.model.memory_bank.numel():
        raise ValueError("Cannot save an unfitted model")
    path.mkdir(parents=True, exist_ok=False)
    torch.save(
        {key: value.cpu() for key, value in model.model.state_dict().items()}, path / "model.pt"
    )
    metadata = {
        "format_version": 1,
        "model_type": "patchcore",
        "run_id": path.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_config": asdict(model.config),
        "preprocessing": {
            "image_size": model.image_size,
            "resize": "bilinear",
            "color": "RGB",
            "mean": list(MEAN),
            "std": list(STD),
        },
        "data_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(data).items()
        },
        "manifest_sha256": manifest_hash(manifest),
        "model_sha256": file_hash(path / "model.pt"),
        "versions": {
            name: version(name) for name in ("torch", "torchvision", "anomalib", "pillow")
        },
        "fit": report,
    }
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_artifact(path: Path, device="auto") -> tuple[PatchCore, dict]:
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata["format_version"] != 1 or metadata["model_type"] != "patchcore":
        raise ValueError("Unsupported model artifact")
    preprocessing = metadata["preprocessing"]
    if (
        preprocessing["mean"] != list(MEAN)
        or preprocessing["std"] != list(STD)
        or preprocessing["resize"] != "bilinear"
        or preprocessing["color"] != "RGB"
    ):
        raise ValueError("Unsupported preprocessing in the artifact")
    if file_hash(path / "model.pt") != metadata["model_sha256"]:
        raise ValueError("Model checkpoint does not match its metadata")
    model = PatchCore(
        PatchCoreConfig(**metadata["model_config"]),
        preprocessing["image_size"],
        device=device,
        pretrained=False,
    )
    state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    model.model.load_state_dict(state)
    model.model.eval()
    if not model.model.memory_bank.numel():
        raise ValueError("The artifact contains an empty memory bank")
    return model, metadata
