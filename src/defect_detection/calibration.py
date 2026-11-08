import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from defect_detection.config import load_data_config
from defect_detection.data.mvtec import make_loader, prepare_data
from defect_detection.models.artifacts import file_hash, load_artifact, manifest_hash


def calibrate(
    artifact: Path, config: Path, device="auto", false_alarm_rate=0.05, pixel_false_alarm_rate=0.005
) -> dict:
    if not 0 < false_alarm_rate < 1 or not 0 < pixel_false_alarm_rate < 1:
        raise ValueError("False-alarm rates must be between 0 and 1")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model, metadata = load_artifact(artifact, device)
    data = load_data_config(config)
    manifest = prepare_data(data)
    if manifest_hash(manifest) != metadata["manifest_sha256"]:
        raise ValueError("Calibration data must match the training split manifest")
    if data.image_size != model.image_size:
        raise ValueError("Calibration image_size must match the saved model")
    scores, maps = [], []
    for batch in make_loader(data, manifest, "calibration"):
        if torch.any(batch["label"] != 0):
            raise ValueError("Calibration requires held-out normal images only")
        batch_scores, batch_maps = model.predict(batch["image"])
        scores.append(batch_scores)
        maps.append(batch_maps)
    if not scores:
        raise ValueError("The calibration split is empty")
    scores = torch.cat(scores).numpy()
    maps = torch.cat(maps).numpy()
    image_threshold = float(np.quantile(scores, 1 - false_alarm_rate, method="higher"))
    pixel_threshold = float(np.quantile(maps, 1 - pixel_false_alarm_rate, method="higher"))
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_sha256": metadata["model_sha256"],
        "metadata_sha256": file_hash(artifact / "metadata.json"),
        "split": "calibration",
        "sample_count": len(scores),
        "sample_ids": [s["image_path"] for s in manifest["samples"] if s["split"] == "calibration"],
        "false_alarm_rate": false_alarm_rate,
        "pixel_false_alarm_rate": pixel_false_alarm_rate,
        "quantile_method": "higher",
        "image_threshold": image_threshold,
        "pixel_threshold": pixel_threshold,
        "observed_image_false_alarm_rate": float(np.mean(scores > image_threshold)),
        "observed_pixel_false_alarm_rate": float(np.mean(maps > pixel_threshold)),
        "heatmap_max": max(float(maps.max()), pixel_threshold * 1.5, 1e-12),
    }
    (artifact / "calibration.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def load_calibration(artifact: Path, metadata: dict) -> dict:
    path = artifact / "calibration.json"
    if not path.exists():
        raise ValueError("Calibrate the model before predicting accept/reject decisions")
    result = json.loads(path.read_text())
    if result["model_sha256"] != metadata["model_sha256"] or result["metadata_sha256"] != file_hash(
        artifact / "metadata.json"
    ):
        raise ValueError("Calibration does not match this model artifact")
    return result
