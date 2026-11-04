import argparse
import json
import platform
from importlib.metadata import version

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
    parser = argparse.ArgumentParser(prog="defect")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check-env")
    args = parser.parse_args()
    if args.command == "check-env":
        check_environment()
