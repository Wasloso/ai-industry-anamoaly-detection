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
