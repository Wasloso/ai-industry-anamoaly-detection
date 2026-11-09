# Wykrywanie defektów produktów z wykorzystaniem analizy obrazów.
## Algorytmy sztucznej inteligencji w Przemyśle 4.0

## Implementation direction

The selected solution is our own PatchCore implementation in PyTorch, using a
torchvision pretrained ResNet only as the feature extractor. Feature aggregation,
coreset selection, the memory bank, neighbour search, scores and maps are
implemented in the project. EfficientAD and other methods remain
literature alternatives. Experiments will compare configurations of PatchCore.

The application commands use our PyTorch implementation. The original Anomalib
prototypes are preserved under [experiments](experiments/README.md) and require
the optional `experiments` dependency group. They are not imported by the application.

## Setup

Use Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked
uv run defect check-env
```

The lockfile includes PyTorch's CUDA 12.6 build on Linux and Windows. It also works
on CPU when no compatible GPU is available. macOS uses the standard PyTorch build.

To check the native model on small synthetic inputs:

```bash
uv run python scripts/check_patchcore.py
```

This checks execution with random backbone weights, not detection accuracy.
Real fitting downloads ImageNet weights on the first run. To preserve those
weights inside the ignored project cache, set `TORCH_HOME` to `.cache/torch`.

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

## PatchCore fitting

`configs/mvtec-bottle-patchcore.yaml` selects the dataset configuration, backbone,
feature layers, and memory-bank sampling ratio. Start with ResNet-18 and a 1%
coreset to keep memory use small on the development laptop.

```python
from pathlib import Path
import torch
from defect_detection.config import load_training_config
from defect_detection.data.mvtec import make_loader, prepare_data
from defect_detection.models.patchcore import PatchCore

data_config, model_config = load_training_config(Path("configs/mvtec-bottle-patchcore.yaml"))
manifest = prepare_data(data_config)
torch.manual_seed(data_config.seed)
model = PatchCore(model_config, image_size=data_config.image_size)
report = model.fit(make_loader(data_config, manifest, "train"))
scores, maps = model.predict(next(iter(make_loader(data_config, manifest, "validation")))["image"])
```

Fitting collects features from normal training images and selects a representative
memory bank; it does not optimize CNN weights. The model applies ImageNet
normalization once, after the loader's resize. Validation/calibration/test images
are excluded from fitting. Scores are raw anomaly distances, not probabilities.

The implementation pools local features with a 3 x 3 window, aligns feature maps,
and concatenates them. Greedy farthest-point selection uses a Gaussian projection
to 128 dimensions when needed; the bank retains the original features. Prediction
uses chunked Euclidean searches, support-neighbour image-score reweighting, and
bilinear maps smoothed with sigma 4 and replicated borders. `num_neighbors` controls
the reweighting support, not averaging the closest patch distances.

This is our implementation of the [PatchCore approach](https://arxiv.org/abs/2106.08265),
with a smaller backbone and explicit sampling/search choices. It does not claim
to reproduce the paper's published benchmark results.


## Saved models

```bash
uv run defect train --config configs/mvtec-bottle-patchcore.yaml --run-id bottle-pc-001
```

Use `--device cpu` or `--device cuda` to select a device explicitly; the default
is `auto`. Each run creates a new folder under `artifacts/`, with model weights
and the memory bank in `model.pt`, readable settings and provenance in
`metadata.json`, and a snapshot of the split manifest. Existing runs are not
overwritten. Input size and normalization are saved with the model.

```python
from defect_detection.models.artifacts import load_artifact

model, metadata = load_artifact(Path("artifacts/bottle-pc-001"), device="cpu")
scores, maps = model.predict(batch["image"])
```

Reloading uses saved weights without downloading a backbone, reading the training
dataset, or refitting the memory bank. Use the locked project environment when
loading models. A checkpoint checksum detects mismatched files.

Native checkpoints use artifact format 2 and record `project-pytorch` as their
implementation. Earlier Anomalib checkpoints are experimental format 1; the main
loader rejects them explicitly. Fit a new run before calibration and prediction.


## Calibration

```bash
uv run defect calibrate --artifact artifacts/bottle-pc-001 --config configs/mvtec-bottle.yaml
```

This scores only the reserved normal calibration images, then saves image and
pixel thresholds in `calibration.json`. It checks the split manifest and input
size against the training run. Default target false-alarm rates are 5% per image
and 0.5% per pixel, configurable with `--false-alarm-rate` and
`--pixel-false-alarm-rate`. Decisions use scores strictly greater than the
threshold. The saved thresholds are tied to the model and preprocessing metadata.

The reported false-alarm rates describe the calibration samples. With only 20
normal images, these estimates are coarse and do not guarantee future rates.
Test labels and defect masks are not used to choose thresholds. Recalibration
replaces the threshold file without changing fitted weights or the memory bank.

## Inference

```bash
uv run defect predict --artifact artifacts/bottle-pc-001 --input data/mvtec_ad/bottle/test/broken_large/000.png --output artifacts/prediction-001
uv run defect predict --artifact artifacts/bottle-pc-001 --input data/mvtec_ad/bottle/test --output artifacts/bottle-test-predictions
```

Calibrate the artifact first. Inference loads it once and processes a single image
or all supported images in a folder, including subfolders. It needs only the saved
artifact and input images; no dataset configuration, labels, or ground-truth masks
are needed. The configured product category comes from the training artifact.

Each output folder contains `results.json` with scores, thresholds, decisions,
model/calibration identities, input dimensions, timing, and output paths. Each
image produces a raw float map (`.map.npy`), heatmap, binary mask, and red defect
overlay. Maps and visualizations have the original image dimensions. The color
scale is fixed by calibration, so images remain visually comparable. Pixel masks
are suspected defect regions, not labels for named defect types.

Use a new output directory for each run, outside the input folder. Unsupported
file extensions are skipped during folder discovery; unreadable images stop the
command with an error. Inference time covers normalization, model execution, and
returning scores/maps to CPU; it excludes decoding, resizing, and writing images.
The first call also includes model warm-up, so these per-image timings are not a
standalone throughput benchmark. Existing output folders are not overwritten.
