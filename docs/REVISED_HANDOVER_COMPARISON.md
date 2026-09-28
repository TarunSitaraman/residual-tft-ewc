# Revised Handover Comparison

The package received as `ML_HANDOVER_FINAL (2)` was compared with the data
currently used by this repository.

## Findings

The following shared datasets are byte-identical between the two packages:

- `cleaned_battery_data.csv`
- `final_battery_cycling_features.csv`
- `final_battery_health_dataset.csv`
- `final_ml_dataset.csv`
- `rpt_capacity_data.csv`

The revised package does not contain `ml_training_dataset.csv`. Instead, it
contains `ml_ready_dataset.csv`, which has the usual cycle and health columns
plus:

- `Test_Date`
- `Temperature_Mean_Missing`
- `Temperature_Min_Missing`
- `Temperature_Max_Missing`

Its README also states that temperature values are pre-imputed using
battery-level medians, followed by an overall median fallback.

The revised package additionally contains a starter notebook, six graphs, and
four summary tables. These are useful handover artifacts, but they do not
replace the repository's validation notebooks or `src/data_pipeline.py`.

## Decision

`ml_training_dataset.csv` remains the canonical baseline input. The current
baseline fits imputation on the training split, which avoids using future
validation/test rows when calculating medians. The revised `ml_ready_dataset.csv`
is retained as an external candidate input until its battery-level imputation
scope is confirmed.

Do not silently rename `ml_ready_dataset.csv` to
`ml_training_dataset.csv`. Using it changes the schema and may introduce
preprocessing leakage if its medians were calculated over complete battery
histories. If it is adopted, it must receive a separate validation report and
its missingness flags must be treated as explicit model inputs.

The revised package can still be mounted for inspection using `DATA_ROOT`, but
the current notebooks expect the canonical `ml_training_dataset.csv` filename.

## Audit

When `ml_ready_dataset.csv` is placed in `DATA_ROOT` next to the canonical
files, notebook 03 §9 checks it against `ml_training_dataset.csv`: same rows,
identical shared values, flags matching the canonical missing temperatures, and
whether the imputed values equal full-history battery medians (leakage). See
decision D7 in `docs/DECISIONS.md`.

## Open item

This comparison does not state whether the revised package contains
`DATA_DICTIONARY.csv` or whether it changed. Notebooks 01 and 03 require it;
record the result of `cmp` between the two copies here.
