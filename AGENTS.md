# Project guidance

Read `README.md`, `docs/getting-started.md` and `docs/implementation-roadmap.md` before changing the workflow or implementation plan.

- The application uses our PyTorch PatchCore implementation. Torchvision supplies the pretrained ResNet feature extractor; feature aggregation, coreset selection, neighbour search, scoring and maps belong to this project.
- Keep Anomalib imports inside `experiments/`. It is installed only through the optional `experiments` extra and is not required by the main CLI.
- EfficientAD, PaDiM and DRAEM are literature alternatives, not additional implementation milestones. Initial experiments compare configurations of our detector.
- Preserve fitting, validation, calibration and test separation. Do not choose thresholds using the final test labels or masks.
- Native artifacts use format 2. Preserve explicit rejection of legacy Anomalib artifacts rather than silently loading them with different semantics.
- Keep working commands separate from planned API, dashboard, evaluation and simulation commands in the docs.
- For code changes, run `uv run --locked pytest -q` and `uv run --locked ruff check .`. The dataset-free execution check is `uv run --locked python scripts/check_patchcore.py`; it is not an accuracy benchmark.
- Keep datasets, downloaded weights and generated artifacts outside Git. Record only measurements actually performed.

The companion report and research notes live in `Wasloso/isa-s2-projects-docs`, under the matching project directory. Update their implementation status when relevant; the report uses 14 selected references, including four academic textbooks.
