# Anomalib experiments

These files preserve the initial library-based feasibility work. They are
reference experiments, not the independent PatchCore implementation.

- `anomalib_patchcore.py` keeps the original fit/predict wrapper.
- `check_anomalib_models.py` checks PatchCore fitting and one EfficientAD
  optimization step on random images. It is not an accuracy benchmark.

Run the architecture check from the repository root:

```bash
uv run python experiments/check_anomalib_models.py
```

Large weights, datasets and experiment outputs stay outside version control.
