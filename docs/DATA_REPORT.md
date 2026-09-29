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

The discharge current equals **0.9998 × the C/20 capacity measured at the previous RPT** for every battery, with a coefficient of variation of 0.14 %. On the handover's capacity, which includes the pulse charge, the factor is 0.9966. The lab therefore set 1C from the C/20 capacity, which supports decision D17. The protocol set it to 1C of the last measured capacity, so the discharge current, discharged Ah and discharge energy encode the label. They must not be used as inputs (decision D14). The handover's `Current_Min` and discharge-capacity columns partly carry the same scaling.

### 4.3 Legitimate health indicators track SOH, but not better than time
Indicators from the same segment (CC-charge duration and Ah, resistance proxy, duty-cycle voltage envelope, and partial-charge ΔQ in a fixed voltage window) move in the physically expected direction. The best, ΔQ between 3.7 and 3.9 V during the CC charge, has a median within-battery ρ ≈ 0.76.

In leave-one-battery-out estimation of the SOH at the RPT closing each block (notebook 10; 93 standard blocks with a previous RPT; C/20 labels, D17; `results/health_indicators.json`):

| Predictor | Mean held-out RMSE (SOH pp) |
|---|---|
| Previous RPT's SOH (persistence) | 0.292 |
| Elapsed time | 0.521 |
| ΔQ(3.7–3.9 V) + time | 0.531 |
| All indicators + time | 0.581 |
| All indicators | 0.817 |
| ΔQ(3.7–3.9 V) | 1.227 |

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

The raw RPT and EIS files from the public archive give one row of 47 features per (battery, RPT). See `docs/DECISIONS.md` D16 for definitions and selection rules. Summary: `results/diagnostic_features.json`.

- **Validation.** RPT 7 EIS was decoded from the aggregate files, and its R0 equals the authors' own high-frequency resistance for all 18 values. Pulse resistances (29–32 mΩ) and R0 (about 25–27 mΩ) match the ranges in the article.
- **Health tracking.** Almost every feature tracks SOH within each battery, with the same sign in all six cells:
  - the IC peak height near 4.07 V falls smoothly (median within-battery ρ = 0.97);
  - pulse and EIS resistances rise (|ρ| ≈ 0.85–0.94).
- **Forecasting relevance.** Against the next SOH change (the part a forecaster must predict beyond persistence), most features are weak (median |ρ| ≤ 0.3). The exceptions are at the low-voltage end:
  - low-frequency Z′ at 3.26 V (0.1 and 0.01 Hz): ρ ≈ 0.42–0.44, same sign in all six cells;
  - the charge between 3.63 and 3.26 V: ρ ≈ −0.52, same sign in five cells.

  These are also the features that alternate from RPT to RPT with the 20/35 °C cycling batches, so the association may reflect the temperature schedule, not predictive ageing information. The forecasting study must test this, for example by adding the next batch's temperature as a known input.

  With about 15 RPT pairs per cell, this test has low power.
- **Common-mode shifts.** EIS R0 moves for all cells together at some RPTs (e.g. RPT 11 and 15). Arc resistance and the charge below 3.26 V alternate from RPT to RPT, most strongly in the residential cells (v4, w9, w10). The likely cause is measurement or storage conditions (the preceding cycling batch alternates 20/35 °C), not ageing. This is not yet tested.
- **Data-quality findings:**
  - RPT 2 was run at about 20 °C.
  - Pulses were off protocol in RPT 1 (4.00 V) and for v4 and w10 in RPT 17–19.
  - Galvanostatic EIS was excluded for w8 in RPT 17 and 19.
  - EIS is missing for most cells at RPT 16 and for v4 and w10 at RPT 17–19.
- **Label issue (resolved, D17).** The handover capacity includes the pulse charge, whose size changed with the protocol (RPT 1 for four cells; v4 and w10 in RPT 17–19). SOH labels now use the C/20 capacity (`data/rpt_health_c20.csv`). They differ from the handover labels by at most 0.089 pp.

## 6. Options (team decision)

1. **Reframe the task as SOH forecasting.** Predict SOH several RPTs ahead from the RPT history, health indicators and usage. Persistence and linear extrapolation are then the baselines to beat, and this is the TFT's native task.
2. **Add public datasets with larger and more varied fade**, for example the first-life data of the same cells and other chemistries, so that continual learning, heterogeneity and reuse classes become meaningful. This is the strongest route to a novel, defensible paper.
3. **Use the EIS spectra and raw RPT curves from the public archive** (done: §5, notebook 11). They are not in the handover, but OSF version 2 contains 237 EIS files and 112 raw RPT files under the same licence (`docs/EXTERNAL_DATA.md` S1). Impedance and capacity-test curves are the strongest known health indicators.
4. **Any change to pre-registered choices** (reuse cut-points, task definition) must be reported as made after seeing the data.

## 7. SOH forecasting benchmark (notebook 12, protocol D18)

All numbers are from `results/forecasting.json` (commit `d710cad`, clean). Errors are the RMSE of SOH(*k*+*h*) in pp, on the test splits, pooled over *h* = 1–3 and averaged over batteries.

**Tuning (validation only).**
- Learning rates: MLP 1e-3; TFT 3e-2, inside the grid after it was extended (D18 deviation).
- EWC λ: 10 for the MLP, 1000 for the TFT.
- Replay: full rehearsal (the task size is below every buffer value).

**P1, continual stream** (10 orders × 5 seeds; TFT + EWC vs each alternative, Holm-corrected):

| Method | Final avg. RMSE | Skill vs persistence | vs TFT + EWC |
|---|---|---|---|
| Ridge, schedule only | 0.450 | 0.29 | better by 0.012 (CI 0.007–0.018) |
| TFT, joint (upper bound) | 0.455 | 0.28 | not tested (upper bound) |
| **TFT + EWC (proposed)** | **0.462** | **0.27** | — |
| Gradient boosting, F1 | 0.466 | 0.26 | n.s. (p_Holm 0.18) |
| TFT fine-tuning | 0.467 | 0.26 | n.s. (p_Holm 0.28) |
| TFT + EWC, F0 (no diagnostics) | 0.472 | 0.25 | n.s. (p_Holm 0.08) |
| MLP replay / fine-tune / EWC | 0.475 / 0.492 / 0.497 | 0.25 / 0.22 / 0.21 | replay n.s.; fine-tune and EWC worse (p < 0.001) |
| Persistence | 0.632 | 0 | worse (p < 0.001) |
| Linear extrapolation | 0.726 | −0.15 | worse (p < 0.001) |

- **Forgetting is small for every method** (0.016–0.040 pp). TFT + EWC has the least (0.016) against TFT fine-tuning (0.026), but the difference is not significant after correction.
- **Why so little forgetting:** the six cells are nearly homogeneous, so there is little to forget. This protocol cannot show a benefit of EWC on S1.
- **Diagnostics:** F1 vs F0 improves TFT + EWC by 0.009 pp (CI 0.001–0.017), which is not significant after Holm correction. Diagnostic features add at most a small improvement over SOH history, static descriptors and schedule.
- **By horizon:** the schedule-only ridge is best at *h* = 1 (0.40). The joint TFT is best at *h* = 3 (0.42). Persistence and linear extrapolation degrade quickly with horizon (0.83 and 0.90 at *h* = 3).
- **Main message on S1:** fade over the next 1–3 RPTs is explained mostly by the known schedule (days at 35 °C, days at 20 °C, rest days). A three-feature ridge on the schedule is the strongest single model. The TFT is competitive but not better.

**P2, unseen battery** (leave one battery out; six held-out batteries):
- Gradient boosting on F1 is best: 0.219.
- MLP 0.285, TFT 0.292 and schedule-only ridge 0.301 are close; persistence is 0.545.
- With six pairs, no difference survives Holm correction; the smallest attainable p is 0.031 before correction.

**Implications.**
1. **For the paper**, the honest S1 result is that learned forecasters beat persistence and linear extrapolation by 25–29 % in RMSE, driven mainly by the known temperature and rest schedule. TFT + EWC is competitive (level with gradient boosting and TFT fine-tuning) but does not beat a schedule-aware linear model. S1 is too homogeneous and too small to show continual-learning benefits.
2. **This is why the second dataset is needed** (S6, D19). It has 20 cells in seven usage conditions, fade to 30–50 % SOH with knees, and about 2,400 forecasting samples. Continual learning, EWC and reuse classes can be tested meaningfully there.

