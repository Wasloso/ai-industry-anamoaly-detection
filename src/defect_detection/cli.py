import argparse
import json
import logging
import platform
from importlib.metadata import version
from pathlib import Path

import torch


def check_environment():
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "anomalib": version("anomalib"),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        device = torch.cuda.get_device_properties(0)
        report["gpu"] = device.name
        report["gpu_memory_gib"] = round(device.total_memory / 1024**3, 2)
        report["cuda_runtime"] = torch.version.cuda
        torch.ones(2, device="cuda").sum().item()
    print(json.dumps(report, indent=2))


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(prog="defect")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check-env")
    training = subparsers.add_parser("train")
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--run-id", required=True)
    training.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    training.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    data = subparsers.add_parser("data")
    commands = data.add_subparsers(dest="data_command", required=True)
    for name in ("validate", "preview"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name == "preview":
            command.add_argument(
                "--split", choices=("train", "validation", "calibration", "test"), default="test"
            )
            command.add_argument("--output", type=Path, default=Path("artifacts/preview.png"))
            command.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    if args.command == "check-env":
        check_environment()
        return

    import yaml

    from defect_detection.config import load_data_config
    from defect_detection.data.mvtec import make_loader, prepare_data, summarize
    from defect_detection.data.preview import save_preview

    try:
        if args.command == "train":
            from defect_detection.pipeline import train

            report = train(args.config, args.run_id, args.artifacts_dir, args.device)
            print(json.dumps(report, indent=2))
            return
        config = load_data_config(args.config)
        manifest = prepare_data(config)
        if args.data_command == "validate":
            batch = next(iter(make_loader(config, manifest, "train")))
            print(
                json.dumps(
                    {
                        "category": config.category,
                        "manifest": str(config.manifest),
                        "splits": summarize(manifest),
                        "batch_shape": list(batch["image"].shape),
                    },
                    indent=2,
                )
            )
        else:
            save_preview(config, manifest, args.output, args.split, args.limit)
            print(f"Preview saved to {args.output}")
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
