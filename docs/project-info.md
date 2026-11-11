*Course*: Projekt przejściowy
*Project title*: Wykrywanie defektów produktów z wykorzystaniem analizy obrazów.

*Dataset(s)*: MVTec AD, początkowo `bottle`; później kolejne kategorie i opcjonalnie VisA.

*Current scope*:
- Use Anomalib implementations of PatchCore and PaDiM with explicitly recorded configurations.
- Fit and calibrate on separate normal-data partitions, then compare image detection, localization, time and memory on the same final test samples.
- Provide one simple local Streamlit application: select images, invoke analysis, inspect both methods and download tables and visualizations.
- Export CSV/JSON and an archive of generated images; keep model artifacts and results in local files.

There is no streaming pipeline, Spark/Kafka, station simulation, telemetry, separate API or database in this project's scope. The initial code provides dependencies, model configuration checks and the existing data loader. Library-backed fitting, comparison and the application remain subsequent steps. Previous native PatchCore work is retained as an explicitly named experiment.
