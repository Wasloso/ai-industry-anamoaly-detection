# Project guidance

Read `README.md`, `docs/getting-started.md` and `docs/implementation-roadmap.md` before changing the workflow or plan.

- The selected methods are PatchCore and PaDiM from Anomalib, now a main dependency. Initial YAML files and model-construction checks exist; their fitting/comparison workflow is not complete.
- Keep the project small: one local Streamlit application with manual image analysis, side-by-side results and downloads. Do not add streaming, Spark/Kafka, station simulation, telemetry, a separate API or database. The Big Data project is separate.
- `defect model check` constructs an unfitted model without pretrained weights or dataset access. Do not describe this as training, inference or an accuracy result. Future fitting must explicitly enable pretrained weights.
- Preserve fitting, validation, calibration and test separation. Check library preprocessing/postprocessing; do not normalize twice or choose thresholds with final test labels or masks.
- Earlier own-PyTorch work is kept under the explicit `defect native ...` commands and `experiments/`. Native format-2 and earlier format-1 artifacts are not interchangeable with future library-backed artifacts.
- Keep available commands separate from planned fitting, analysis, comparison and application features in the docs.
- For code changes run `uv run --locked pytest -q` and `uv run --locked ruff check .`. The dataset-free checks are `uv run --locked defect model check --config configs/mvtec-bottle-patchcore.yaml` and the analogous `padim` configuration.
- Keep datasets, downloaded weights, models and generated results outside Git. Record only measurements actually performed.

The companion report and research notes are in `Wasloso/isa-s2-projects-docs`, under the matching project directory. Keep implementation status consistent. The report uses 14 selected references, including four academic textbooks.
