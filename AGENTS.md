# Project Agent Notes

This repository contains the continual-learning workflow for SOH estimation and second-life suitability assessment of heterogeneous EV batteries.

## Conventions

- Keep datasets outside Git under `data/`.
- Preserve battery-group identity and temporal ordering in evaluation.
- Document missing-temperature handling; do not fabricate values silently.
- Keep generated checkpoints and experiment outputs out of version control.
- Forecasting work (D16–D20) uses the C/20 labels in `data/rpt_health_c20.csv` and the features built from the raw archives (`data/README.md`). `ml_training_dataset.csv` is only the input of the superseded current-SOH notebooks 01–09.

## Invariants

- `SOH_Percent = Capacity_Ah / Reference_Capacity_Ah * 100`, where `Capacity_Ah` is the C/20 discharge capacity of the RPT, pulses excluded (`data/rpt_health_c20.csv`, decision D17)
- A cycle uses only the most recent RPT on or before its cycle date.
- Continual-learning evaluation reports adaptation and retention.
