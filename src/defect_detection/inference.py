import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn.functional import interpolate
from torchvision.transforms.functional import pil_to_tensor

from defect_detection.calibration import load_calibration
from defect_detection.models.artifacts import file_hash, load_artifact

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def save_outputs(
    image: Image.Image, anomaly_map: torch.Tensor, folder: Path, name: str, calibration: dict
) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    values = anomaly_map.numpy()
    np.save(folder / f"{name}.map.npy", values)
    mask = Image.fromarray((values > calibration["pixel_threshold"]).astype(np.uint8) * 255)
    mask.save(folder / f"{name}.mask.png")
    scaled = (np.clip(values / calibration["heatmap_max"], 0, 1) * 255).astype(np.uint8)
    heatmap = ImageOps.colorize(Image.fromarray(scaled), black="navy", mid="yellow", white="red")
    heatmap.save(folder / f"{name}.heatmap.png")
    overlay = Image.composite(
        Image.blend(image, Image.new("RGB", image.size, "red"), 0.5), image, mask
    )
    overlay.save(folder / f"{name}.overlay.png")
    return {
        "map": f"{name}.map.npy",
        "mask": f"{name}.mask.png",
        "heatmap": f"{name}.heatmap.png",
        "overlay": f"{name}.overlay.png",
    }


def predict(artifact: Path, source: Path, output: Path, device="auto") -> dict:
    if not source.exists():
        raise ValueError(f"Input does not exist: {source}")
    if source.is_dir() and output.resolve().is_relative_to(source.resolve()):
        raise ValueError("The output directory must be outside the input folder")
    paths = (
        [source]
        if source.is_file()
        else sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    )
    if not paths:
        raise ValueError("No supported images found in the input folder")
    if output.exists():
        raise ValueError(f"Output already exists: {output}. Choose a new output folder")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model, metadata = load_artifact(artifact, device)
    calibration = load_calibration(artifact, metadata)
    calibration_hash = file_hash(artifact / "calibration.json")
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for path in paths:
        with Image.open(path) as original:
            image = original.convert("RGB")
        resized = image.resize((model.image_size, model.image_size), Image.Resampling.BILINEAR)
        inputs = pil_to_tensor(resized).float().unsqueeze(0) / 255
        started = time.perf_counter()
        scores, maps = model.predict(inputs)
        inference_ms = (time.perf_counter() - started) * 1000
        anomaly_map = interpolate(
            maps, size=(image.height, image.width), mode="bilinear", align_corners=False
        )[0, 0]
        relative = Path(path.name) if source.is_file() else path.relative_to(source)
        files = save_outputs(
            image, anomaly_map, output / relative.parent, relative.name, calibration
        )
        score = float(scores[0])
        results.append(
            {
                "input": str(path.resolve()),
                "run_id": metadata["run_id"],
                "model_sha256": metadata["model_sha256"],
                "calibration_sha256": calibration_hash,
                "category": metadata["data_config"]["category"],
                "score": score,
                "threshold": calibration["image_threshold"],
                "is_anomaly": score > calibration["image_threshold"],
                "pixel_threshold": calibration["pixel_threshold"],
                "width": image.width,
                "height": image.height,
                "inference_ms": round(inference_ms, 3),
                "outputs": {
                    key: (relative.parent / name).as_posix() for key, name in files.items()
                },
            }
        )
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    report = {
        "output": str(output.resolve()),
        "images": len(results),
        "anomalies": sum(result["is_anomaly"] for result in results),
    }
    if len(results) == 1:
        report.update(score=results[0]["score"], is_anomaly=results[0]["is_anomaly"])
    return report
