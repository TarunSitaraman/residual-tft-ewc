# Data report: first look at the real handover data

Written after the first real-data runs (29 Sep 2026), from notebooks 01–04 and 10 executed on the data published at
`huggingface.co/datasets/TarunSitaraman/second-life-ev-battery-datasets` (SHA-256 of the two large files verified against the repository's LFS records).
Regenerate every number and figure by running the notebooks; figures are written to `results/figures/`.

## 1. What the data is

| Item | Value |
|---|---|
| Batteries | 6: g1, v4, v5, w8, w9, w10 |
| Period | Oct 2022 – Jan 2025 (~27 months); w9 cycling ends Mar 2024, but it has one more RPT on 1 Aug 2024 (RPT_16, after ~4 months without cycling) |
| RPTs | 19 per battery (w9: 16), roughly monthly; 111 in total |
| RPT gap | no RPTs between days ~510 and ~625; SOH rises at the next RPT for every battery (likely recovery after the long rest) |
| SOH range | 100 % → 95.1–97.5 % (total fade 2.5–4.9 pp) |
| Labelled cycle-level rows | 582 (`ml_training_dataset.csv`); 609 in `final_ml_dataset.csv` (27 before the first RPT) |
| Rows per RPT interval | ~6 |
| Point-level measurements | 223,973 rows (`cleaned_battery_data.csv`), 60 s sampling during protocol steps, 900 s during the duty cycle |
| Missing temperature | 0.5 % of labelled rows; 4.9 % of point-level rows |
| EIS | measured at 3.26 / 3.63 / 4.00 V on 12–20 dates per battery, but **the spectra are not in the handover** (only file references with capacity 0) |

**Provenance (confirmed 29 Sep 2026).** The handover is a processed subset of the public Stanford second-life dataset, file version 2 (Moy et al., *Data in Brief* 57, 111046, 2024; data doi:10.17605/OSF.IO/8JNR5, CC BY 4.0).

- All 337 file references in `rpt_capacity_data.csv` and all 103 cell × cycling-test blocks match the OSF archive.
- Full citations, licence, version, hashes and attribution duties: `docs/EXTERNAL_DATA.md`.
- Check script: `scripts/verify_external_data.py`.

**Heterogeneity is limited.** According to the source, the cells differ in first-life charge rate (C/4 to 3C) and second-life duty cycle (residential: v4, w9, w10; commercial: w8, v5, g1); see `docs/EXTERNAL_DATA.md` S1. All six are the same cell type, under the same temperature schedule (alternating 20 and 35 °C), with the same 1C discharge. They differ in the CC charge current in the duty cycle (1.46 vs 1.74 A) and in the depth of discharge (minimum voltage 2.6 V for v4 and w10 vs 3.1–3.3 V for the others). Their SOH trajectories are nearly identical, except v5, which fades faster.

## 2. Mismatches with the handover description (fixed in the pipeline)

| File | Handover description | Delivered |
|---|---|---|
| `final_battery_cycling_features.csv` | cycle-level features | **point-level**: one row per measurement (`Date_Time`, `Cycle_Count`), identical to `cleaned_battery_data.csv` (0 mismatches over 223,973 matched rows) |
| `rpt_capacity_data.csv` | `Battery_ID, RPT, Capacity_Ah, Test_Date` | `RPT, Battery, File, Sheet, Capacity_Ah`; battery and date are embedded in `Battery` (e.g. `20221114_g1_RPT`); 225 of 337 rows are EIS file references |
| Cycle-level "cycles" | charge/discharge cycles | segments of monthly cycling blocks (~2 days apart); capacity and energy columns are **running totals within the block** |

Notebooks 01–03 now handle the delivered schemas (`dp.normalise_rpt_capacity`, `Date_Time` parsing, point-level checks).

## 3. Label validation: all passed

- SOH formula holds on all 111 RPTs and all 582 labelled rows.
- The reference capacity is the first RPT for every battery.
- Every cycle carries the most recent RPT on or before it (no future labels).
- The training subset equals the labelled part of `final_ml_dataset.csv`.
- `rpt_capacity_data.csv` matches the health table exactly (capacity and date).
- Only 10 cycles fall on a date-only RPT day, so the D3 convention barely matters here.

## 4. Critical findings

### 4.1 The cycle-level features carry almost no health information
Within each battery, the Spearman correlation of the 18 model inputs with SOH has a median |ρ| of **0.09** (largest |ρ| ≈ 0.3), against ≈ **−0.99** for elapsed time (`results/figures/02_feature_soh_correlation.png`). Consistently, the notebook 04 baseline is **worse than persistence** on later-life test periods: test RMSE is 0.99 pp for persistence, 1.78 pp for gradient boosting and 2.37 pp for ridge, and R² is negative for all learned models (`results/baseline_metrics.json`).

### 4.2 Protocol leakage in the discharge current
Every monthly block starts with a standardised segment:

1. constant-current (CC) charge at 1.46 A;
2. constant-voltage (CV) charge;
3. rest;
4. 30-minute discharge;
5. rest.

The discharge current equals **0.9966 × the capacity measured at the previous RPT** for every battery, with a coefficient of variation of 0.14 %. The protocol set it to 1C of the last measured capacity, so the discharge current, discharged Ah and discharge energy encode the label. They must not be used as inputs (decision D14). The handover's `Current_Min` and discharge-capacity columns partly carry the same scaling.

### 4.3 Legitimate health indicators track SOH, but not better than time
Indicators from the same segment (CC-charge duration and Ah, resistance proxy, duty-cycle voltage envelope, and partial-charge ΔQ in a fixed voltage window) move in the physically expected direction. The best, ΔQ between 3.7 and 3.9 V during the CC charge, has a median within-battery ρ ≈ 0.76.

In leave-one-battery-out estimation of the SOH at the RPT closing each block (notebook 10; 93 standard blocks with a previous RPT; `results/health_indicators.json`):

| Predictor | Mean held-out RMSE (SOH pp) |
|---|---|
| Previous RPT's SOH (persistence) | 0.288 |
| Elapsed time | 0.505 |
| ΔQ(3.7–3.9 V) + time | 0.519 |
| All indicators + time | 0.559 |
| All indicators | 0.815 |
| ΔQ(3.7–3.9 V) | 1.218 |

ΔQ(3.7–3.9 V) exists only for blocks whose CC charge starts below 3.7 V (about half). In the other blocks it is imputed with the training median, which weakens it; restricted to blocks where it exists, it reached about 0.88 alone and about 0.50 with time in an exploratory run. Four blocks (v4 and w10, tests 17–18) used a non-standard 1.05 A charge and are flagged and excluded.

The limiting factor is the data, not the method. Total fade is only 3–5 pp along nearly identical trajectories, so time explains almost everything, and monthly RPTs make persistence very strong.

### 4.4 Consequences for the pre-registered plan

| Plan element | Status on this data |
|---|---|
| SOH estimation benchmark (E1–E9) | Current-SOH estimation from cycle features is not a meaningful benchmark: trivial baselines win |
| Continual learning across groups | Little to forget: groups are nearly identical |
| Reuse classes (D10) | Degenerate: every sample is class C0 (no SOH ≤ 95 %) |
| Missing-temperature robustness (E11) | Low relevance: 0.5 % missing in labelled data |
| TFT + EWC | Would mostly learn a time trend from uninformative inputs |

## 5. Raw diagnostic features (notebook 11, decision D16)

The raw RPT and EIS files from the public archive give one row of 48 features per (battery, RPT). See `docs/DECISIONS.md` D16 for definitions and selection rules. Summary: `results/diagnostic_features.json`.

- **Validation.** RPT 7 EIS was decoded from the aggregate files, and its R0 equals the authors' own high-frequency resistance for all 18 values. Pulse resistances (29–32 mΩ) and R0 (about 25–27 mΩ) match the ranges in the article.
- **Health tracking.** Almost every feature tracks SOH within each battery, with the same sign in all six cells:
  - the IC peak height near 4.07 V falls smoothly (median within-battery ρ = 0.97);
  - pulse and EIS resistances rise (|ρ| ≈ 0.85–0.94).
- **Forecasting relevance.** Against the next SOH change (the part a forecaster must predict beyond persistence), associations are weak: median |ρ| ≤ 0.3, with consistent signs in at most 5 of 6 cells.
  - The features describe the current state well, but the data offer little evidence that they predict the next increment.
  - With about 15 RPT pairs per cell, this test has low power.
- **Common-mode shifts.** EIS R0 moves for all cells together at some RPTs (e.g. RPT 11 and 15). Arc resistance and the charge below 3.26 V alternate from RPT to RPT, most strongly in the residential cells (v4, w9, w10). The likely cause is measurement or storage conditions (the preceding cycling batch alternates 20/35 °C), not ageing. This is not yet tested.
- **Data-quality findings:**
  - RPT 2 was run at about 20 °C.
  - Pulses were off protocol in RPT 1 (4.00 V) and for v4 and w10 in RPT 17–19.
  - Galvanostatic EIS was excluded for w8 in RPT 17 and 19.
  - EIS is missing for most cells at RPT 16 and for v4 and w10 at RPT 17–19.
- **Label issue (open).** The handover capacity includes the pulse charge. The v4 and w10 labels at RPT 17–19 are therefore about 0.17 pp lower than they would be under the standard pulses. The recommended fix is to label with the C/20 capacity (D16).

## 6. Options (team decision)

1. **Reframe the task as SOH forecasting.** Predict SOH several RPTs ahead from the RPT history, health indicators and usage. Persistence and linear extrapolation are then the baselines to beat, and this is the TFT's native task.
2. **Add public datasets with larger and more varied fade**, for example the first-life data of the same cells and other chemistries, so that continual learning, heterogeneity and reuse classes become meaningful. This is the strongest route to a novel, defensible paper.
3. **Use the EIS spectra and raw RPT curves from the public archive** (done: §5, notebook 11). They are not in the handover, but OSF version 2 contains 237 EIS files and 112 raw RPT files under the same licence (`docs/EXTERNAL_DATA.md` S1). Impedance and capacity-test curves are the strongest known health indicators.
4. **Any change to pre-registered choices** (reuse cut-points, task definition) must be reported as made after seeing the data.
