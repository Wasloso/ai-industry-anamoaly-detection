# Wykrywanie defektów produktów z wykorzystaniem analizy obrazów.
## Algorytmy sztucznej inteligencji w Przemyśle 4.0

## Setup

Use Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked
uv run defect check-env
```

The lockfile includes PyTorch's CUDA 12.6 build on Linux and Windows. It also works
on CPU when no compatible GPU is available. macOS uses the standard PyTorch build.

To check that both model architectures can produce scores and maps:

```bash
uv run python scripts/check_models.py
```

This uses random images to check execution and GPU memory, not detection accuracy.
It fits a small PatchCore memory bank and runs one EfficientAD optimization step.
The first run downloads the ResNet-18 backbone and EfficientAD teacher weights.
Full training and ImageNette preparation come in the next stage.

```bash
uv run pytest
uv run ruff check .
```

## MVTec AD data loader

Download the `Bottle` archive from the [official download page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad/downloads)
and extract it under `data/mvtec_ad`. The expected layout is:

```text
data/mvtec_ad/bottle/
  train/good/*.png
  test/good/*.png
  test/<defect>/*.png
  ground_truth/<defect>/*_mask.png
```

MVTec AD is released under CC BY-NC-SA 4.0. Dataset files, downloaded weights,
manifests, and previews stay outside Git.

```bash
uv run defect data validate --config configs/mvtec-bottle.yaml
uv run defect data preview --config configs/mvtec-bottle.yaml
```

Validation checks image readability, defect masks, and duplicate images. It saves
a split manifest on the first run and refuses to silently change it later.
Ten percent of distinct normal training images go to validation and another ten
percent to calibration, rounded down with at least one group in each. Duplicate
normal images stay together. Official test images keep their original partition.
Use a new manifest path when intentionally changing the dataset or split settings.

Paths inside the YAML file are relative to that file. Change `root` or `category`
to use another extracted MVTec AD category. Adjust `batch_size`, `image_size`, and
`num_workers` there as needed. Images are RGB float tensors in `[0, 1]`, with no
model normalization. Binary masks use nearest-neighbor resizing; original image
dimensions are retained for mapping predictions back later.

The preview shows the source image, ground-truth mask, and red mask overlay.
It is a data check, not a model prediction. By default it is saved to
`artifacts/preview.png`; use `--output`, `--split`, or `--limit` to change it.

To load a batch from Python:

```python
from pathlib import Path
from defect_detection.config import load_data_config
from defect_detection.data.mvtec import make_loader, prepare_data

config = load_data_config(Path("configs/mvtec-bottle.yaml"))
manifest = prepare_data(config)
batch = next(iter(make_loader(config, manifest, "train")))
print(batch["image"].shape)
```
