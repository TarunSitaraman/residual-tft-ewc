# Data

The handover CSV datasets are **not stored in Git**. CSV files are ignored by `.gitignore`; only code, schema notes and small result files are versioned.

## Providing the data

The notebooks read every file from `DATA_ROOT`:

```python
DATA_ROOT = os.environ.get("DATA_ROOT", "data")   # relative paths resolve against the repo root
```

Either copy the CSVs into this directory, or mount/download them elsewhere and point `DATA_ROOT` at that location:

```bash
export DATA_ROOT=/mnt/project-data
# or
export DATA_ROOT=/workspace/datasets/batteries
```

Expected files:

| File | Content |
|------|---------|
| `cleaned_battery_data.csv` | Cleaned point-level cycling data (large; profiled in chunks) |
| `rpt_capacity_data.csv` | Extracted RPT capacity measurements |
| `DATA_DICTIONARY.csv` | Column definitions |
| `ml_training_dataset.csv` | SOH-labelled cycles; rows before the first RPT excluded |
| `final_ml_dataset.csv` | All cycles; SOH missing before the first RPT |
| `final_battery_health_dataset.csv` | RPT capacity and SOH progression per battery |
| `final_battery_cycling_features.csv` | Cycle-level cycling features |

## Revised handover package

The revised external handover may provide `ml_ready_dataset.csv` instead of
`ml_training_dataset.csv`. It contains extra `Test_Date` and temperature
missingness-flag columns and pre-imputed temperature values. It is not a drop-in
replacement for the canonical baseline input because its imputation scope must
be audited for train/test leakage first. See
`docs/REVISED_HANDOVER_COMPARISON.md`.

If `DATA_ROOT` or a file a notebook needs is missing, the notebook stops with a `DataFilesMissingError` listing the missing files. No substitute data is generated.

## Running the notebooks

From the repository root:

```bash
pip install -r requirements.txt
export DATA_ROOT=/path/to/handover-csvs
for nb in notebooks/0*.ipynb; do
  jupyter nbconvert --to notebook --execute "$nb" --output-dir /tmp/executed
done
```

| Notebook | Needs |
|----------|-------|
| `01_data_ingestion.ipynb` | all seven files |
| `02_eda_analysis.ipynb` | `ml_training`, `final_ml`, `final_battery_health`, `final_battery_cycling_features` |
| `03_feature_engineering.ipynb` | all except `cleaned_battery_data.csv` |
| `04_baseline_model.ipynb` | `ml_training_dataset.csv` |
| `05_reference_baselines.ipynb` | `ml_training_dataset.csv`, `final_battery_health_dataset.csv` |
| `06_continual_learning.ipynb` | `ml_training_dataset.csv` |
| `07_tft_ewc.ipynb` | `ml_training_dataset.csv` |
| `08_reuse_assessment.ipynb` | `ml_training_dataset.csv`, `final_battery_health_dataset.csv` |
| `09_missing_temperature.ipynb` | `ml_training_dataset.csv` |

Notebooks 03–09 write small JSON files to `results/` (`preprocessing_table.json`, `baseline_metrics.json`, `reference_baselines.json`, `cl_metrics.json`, `tft_ewc_metrics.json`, `reuse_assessment.json`, `missing_temperature.json`). Notebooks 05–09 accept `EXPERIMENT_PROFILE=smoke` for a fast code check; only the default `paper` profile produces reportable numbers. Executing into `--output-dir` keeps outputs derived from the data out of the committed notebooks.

See the root README for the SOH definition and temporal-alignment rules.
