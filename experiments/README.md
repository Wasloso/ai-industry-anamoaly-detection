# Earlier experiments

These files preserve previous feasibility work. They are not the new two-model
Anomalib workflow or an accuracy comparison.

- `anomalib_patchcore.py`: the original Anomalib fit/predict wrapper.
- `check_anomalib_models.py`: a historical random-image check of PatchCore and one EfficientAD optimization step.
- `configs/mvtec-bottle-native.yaml`: configuration for the earlier own-PyTorch implementation, still stored in `src/defect_detection/models/` and accessed through `defect native ...`.

Anomalib is now a main dependency; no extra installation flag is needed:

```bash
uv run --locked python experiments/check_anomalib_models.py
```

For the native prototype, see [getting started](../docs/getting-started.md). Its
small synthetic execution check remains `uv run --locked python scripts/check_patchcore.py`.
Neither experiment is required to validate the initial Anomalib configurations.
Datasets, weights and generated artifacts stay outside version control.
