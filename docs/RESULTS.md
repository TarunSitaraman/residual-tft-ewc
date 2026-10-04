# Results

The single record of what the study found. Detail and derivations are in `docs/DATA_REPORT.md` (§5 diagnostics, §7 S1 forecasting, §8 S6 forecasting); protocol decisions and deviations are in `docs/DECISIONS.md` (D15–D20); data sources are in `docs/EXTERNAL_DATA.md`.

## Provenance

Every number below comes from a committed results file written on a clean commit. The tables at the end are generated from these files by `scripts/make_results_tables.py`, which refuses results written from a dirty tree.

| Results file | Content | Written at commit | Produced by |
|---|---|---|---|
| `results/diagnostic_features.json` | S1 diagnostic features and their association with SOH | `c643f1e` | `scripts/build_diagnostic_features.py`, notebook 11 |
| `results/forecasting.json` | S1 forecasting benchmark (P1, P2) | `d710cad` | notebook 12 |
| `results/forecasting_wheeler.json` | S6 forecasting benchmark (P1, P2) | `53ab76f` | notebook 13 |

Each file records Python and library versions (Python 3.11.15, numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1).

## Set-up in brief

- **Data.**
  - S1: six second-life NMC cells on grid duty cycles, OSF 8jnr5 version 2.
  - S6: 20 LFP cells in seven usage conditions (test cases), aged at 50 °C into second life, Wheeler et al. 2025, CC BY 4.0. It has 847 checks, of which 843 are labelled, and 2,409 forecasting samples.
- **Label.** SOH = Capacity_Ah / Reference_Capacity_Ah × 100, relative to the first valid RPT or check of each cell, not to nominal capacity.
  - S1: C/20 discharge capacity, pulses excluded (D17).
  - S6: C/25 discharge capacity (D20).
- **Task (D18).** Forecast ΔSOH from an origin RPT or check *k* to *k* + *h*, for *h* = 1–3. The error is the RMSE of SOH(*k* + *h*) in percentage points (pp).
  - F0 inputs: SOH history, static descriptors and planned usage.
  - F1 inputs: F0 plus diagnostic features.
  - Test currents are never inputs, because they are set from the previous capacity (D14, D16).
- **Splits.** Purged chronological splits within each cell: train ≤ 60 %, validation 60–80 %, test ≥ 80 %. Hyperparameters are tuned on validation data only.
- **Protocols.**
  - P1: a continual stream of tasks (S1: cells; S6: test cases), 10 orders × 5 seeds.
  - P2: leave one group out.
- **Statistics.** Paired Wilcoxon tests with Holm correction, rank-biserial effect sizes and bootstrap 95 % CIs.

## Key findings

1. **Label and leakage corrections (S1).** Two artefacts had to be corrected before any model could be trained.
   - The processed capacity label included pulse charge, which shifted some labels by up to 0.09 pp.
   - Test currents are set from the previous capacity and would leak the label.

   Our S1 capacities match the handover values for 112/112 RPTs, and our S6 parsing reproduces the authors' own capacities on all 843 checks.
2. **Diagnostics describe the current SOH, not its next change (S1).**
   - They track the current SOH closely: IC peak near 4.07 V, median within-cell ρ = 0.97; pulse and EIS resistances, |ρ| = 0.83–0.94.
   - They are weakly associated with the next ΔSOH (median |ρ| ≤ 0.3 for most features).
   - The exceptions are EIS Z′ at 3.26 V (ρ ≈ 0.43, same sign in 6/6 cells) and q_3.63_3.26 (ρ = −0.52, 5/6 cells). Both alternate with the 20/35 °C blocks, so the temperature schedule may drive them.
3. **The best forecasters beat persistence on both datasets, but not every method does.**
   - Best skill: 0.29 on S1 (usage-only ridge) and 0.79 on S6 (linear extrapolation, which is not a learned model).
   - On S1, ridge regression on F0 and linear extrapolation are worse than persistence.
4. **What drives the forecast differs by dataset.**
   - S1 (small, schedule-driven fade): a ridge regression on planned usage alone is the best P1 model, at 0.450 pp.
   - S6 (large, smooth fade): extrapolating the recent fade rate is best, at 0.343 pp, followed by gradient boosting on F1 at 0.372 pp.
5. **Forgetting depends on task heterogeneity.**
   - It is negligible across the homogeneous S1 cells (0.016–0.040 pp).
   - It is substantial across the S6 usage conditions (0.18–0.46 pp).
6. **EWC helps the TFT only on the heterogeneous stream.** *(The significance statements below did not reproduce; see "Reproducibility check".)*
   - S6: TFT + EWC (0.677 pp) beats TFT fine-tuning (0.883 pp) by 0.206 pp on average (CI 0.091–0.338, p_Holm = 0.005, r = 0.52). The median paired difference is only 0.062 pp: the mean is driven by unstable TFT fine-tuning runs (s.d. 0.54 pp).
   - S6 forgetting: TFT + EWC forgets 0.187 pp less on average (median 0.075 pp), but this is not significant after Holm correction (p_Holm = 0.0503).
   - S1: there is no significant difference in error or forgetting.
7. **TFT + EWC is not the most accurate forecaster on either dataset.**
   - S1: it is level with gradient boosting and TFT fine-tuning, and slightly behind the usage-only ridge (by 0.012 pp, CI 0.007–0.017).
   - S6: it is behind MLP fine-tuning and MLP + EWC (significant), gradient boosting and linear extrapolation. The joint upper bounds (MLP 0.414, TFT 0.427 pp) put the continual-learning cost for the TFT at about 0.25 pp.
   - The TFT is also sensitive to task order: on S6 its s.d. over runs is 0.26 pp with EWC and 0.54 pp without, against 0.15–0.18 pp for the MLP methods.
8. **Diagnostic features do not significantly improve TFT + EWC.** *(The S6 forgetting difference below did not reproduce; see "Reproducibility check".)*
   - S1: F1 improves on F0 by 0.009 pp, which is not significant after Holm correction (p_Holm = 0.083).
   - S6: F0 is 0.621 pp against 0.677 pp for F1 (not significant), and F0 forgets significantly less.
9. **Unseen groups (P2).**
   - Gradient boosting is best on both datasets: 0.219 pp on S1 and 0.710 pp on S6.
   - On S6, the 100 % depth-of-discharge case (case 7) is the hardest for every method except linear extrapolation. The ridge, gradient-boosting, MLP and TFT models reach 1.8–2.4 pp on it, and persistence reaches 4.3 pp.
   - With six or seven held-out groups, no difference against the TFT survives Holm correction.
10. **Reuse classes.**
    - S1: every sample stays in C0, so classes cannot be evaluated there.
    - S6 P1: every test period is in C3, so class agreement is trivially perfect.
    - S6 P2: forecast classes agree with the measured classes for 96–97 % of samples with gradient boosting, the MLP and the TFT (macro-F1 0.87–0.88), against 85 % (0.64) for persistence. Closeness to class boundaries was not quantified.

## Limitations to state with any result

- **Task change.** The task was changed from current-SOH estimation to forecasting after S1 had been inspected. The forecasting protocol was fixed before any model was trained. Its deviations, the S1 TFT learning-rate grid extension and test errors printed during smoke runs, are logged in D18.
- **Small datasets.** S1 has six cells, and S6 has two or three cells per condition, so P2 has little statistical power.
- **S6 conditions.** S6 is aged at 50 °C, and its "second life" means continued ageing below 80 % SOH under the same protocol, not a repurposed duty.
- **Relative SOH.** SOH is relative to the first measurement. In S1 that measurement is already an aged state, so SOH values must not be compared directly with literature results that use nominal capacity or other datasets.
- **No pooling.** The two datasets are not pooled, so no transfer between chemistries is claimed.
- **Conditional forecasts.** Forecasts are conditional on the planned usage. On S1, the TFT has more parameters than the MLP.
- **Excluded data.** The Cui/Relyion second-life LMO data (S5) is not used because it carries no licence.

## Reproducibility check (2 Oct 2026)

Notebook 12 (S1) was re-run in the pinned environment of `requirements.txt` (Python 3.11.15, numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, torch 2.14.1), on a different machine, from the C/20 data rebuilt from the raw archive. The torch version of the committed run was not recorded, so the two runs may differ in torch version as well as hardware.

- **Identical:** all baseline results (persistence, linear extrapolation, ridge, gradient boosting), the tuned hyperparameters and the ranking of methods. The MLP joint and P2 MLP results agree to within 1e-8 pp.
- **Neural models drift slightly:** P1 RMSE changes by at most 0.002 pp (TFT + EWC 0.4625 → 0.4605), P2 TFT by 0.0015 pp.
- **S1 significance is not robust to that drift.** Five Holm-corrected comparisons with TFT + EWC cross p = 0.05:

  | Comparison (S1, P1) | Mean difference (pp), committed → re-run | p_Holm, committed → re-run |
  |---|---|---|
  | final error, replay | 0.012 → 0.014 | 0.083 → 0.033 |
  | final error, TFT + EWC F0 (diagnostics) | 0.009 → 0.010 | 0.083 → 0.033 |
  | final error, TFT fine-tuning | 0.005 → 0.007 | 0.278 → 0.035 |
  | forgetting, MLP fine-tuning | 0.014 → 0.014 | 0.052 → 0.018 |
  | final error, gradient boosting F1 | — | 0.175 → 0.028 |

  These differences (0.005–0.014 pp) are an order of magnitude below the label shift corrected by D17 (up to 0.089 pp). With 50 runs of very low variance, the paired test detects differences that have no practical meaning, and whether they cross 0.05 depends on the numerical environment. **On S1, method comparisons should be reported as practically equivalent, with no claim of significance in either direction.**

Notebook 13 (S6) was re-run the same way (3 Oct 2026, commit `ce46b99`), on S6 data re-fetched and verified file by file against the committed SHA-256 manifest.

- **Identical or within 0.006 pp:** all baselines, both joint upper bounds, every P2 result (including reuse-class accuracy and macro-F1), and the backbone learning rates. The ranking of methods is unchanged except that ridge F1 and MLP replay swap places.
- **TFT + EWC tuning flips.** The validation-selected EWC strength for the TFT changes from λ = 100 to λ = 1000. In the re-run, validation error was 0.4057 pp at λ = 1000 against 0.4138 at λ = 100, a 0.008 pp gap estimated from only 3 tuning runs, while TFT results vary by about 0.25 pp across task orders. The selection is a near-tie.
- **Consequently the TFT + EWC comparisons are not robust:**

  | Comparison (S6, P1, vs TFT + EWC) | Mean difference (pp), committed → re-run | p_Holm, committed → re-run |
  |---|---|---|
  | final error, TFT fine-tuning | +0.206 → +0.178 | 0.005 → 0.060 |
  | forgetting, TFT fine-tuning | +0.187 → +0.218 | 0.050 → 0.008 |
  | forgetting, TFT + EWC F0 | −0.098 → −0.071 | 0.004 → 0.243 |
  | backward transfer, TFT + EWC F0 | −0.075 → −0.054 | 0.005 → 0.403 |
  | TFT + EWC vs ridge F1 | — | 0.300 → 0.041 |

  TFT + EWC itself moves from 0.677 to 0.704 pp. The directions agree in both runs (EWC lowers the TFT's mean error and mean forgetting), but which of the two is significant depends on the run.
- **Robust in both runs:** TFT + EWC is significantly worse than MLP fine-tuning and MLP + EWC, than linear extrapolation and gradient boosting, and better than persistence, ridge F0 and the schedule ridge; forgetting is substantial on S6 (about 0.17–0.46 pp) and negligible on S1.

**Implication.** Key findings 6 and 8 above (EWC significantly improves the TFT on S6; F0 forgets significantly less) are not reproducible as significance statements. A reportable version needs a run whose full environment is recorded (this one is), and a tuning stage with enough runs to separate neighbouring grid values. Any such change to the protocol must be recorded in `docs/DECISIONS.md` before the run.

## Figures

All figures are in `results/figures/`; notebooks 11–13 regenerate them.

| Figure | Content |
|---|---|
| `11_ic_curves.png` | S1 incremental capacity of the C/20 discharge at every RPT, per cell |
| `11_feature_trajectories.png`, `11_feature_association.png`, `11_nyquist_363.png` | S1 diagnostic features over ageing, their association with SOH and ΔSOH, and EIS at 3.63 V |
| `12_p1_rmse.png`, `12_p1_horizon.png`, `12_tft_ewc_matrix.png`, `12_example_forecasts.png` | S1 P1 error by method and horizon, TFT + EWC error matrix, and example forecasts |
| `13_soh_trajectories.png`, `13_ic_curves.png` | S6 SOH trajectories by test case, and incremental capacity |
| `13_p1_rmse.png`, `13_tft_ewc_matrix.png` | S6 P1 error by method, and the TFT + EWC error matrix (forgetting) |

## Open items (team)

- Dataset card for the HF mirror (S2), a compliance issue.
- Permission to use the Relyion data (S5).
- Affiliations, author order and the repository licence.

## Tables

<!-- BEGIN GENERATED TABLES -->

_Generated by `scripts/make_results_tables.py` from `results/forecasting.json` (commit `d710cad`), `results/forecasting_wheeler.json` (commit `53ab76f`). Do not edit by hand._

### Table `hyper`: Hyperparameters selected on validation data only

CSV: `results/tables/hyper.csv`

| dataset | lr_mlp | lr_tft | replay_buffer_per_task | ewc_lambda_mlp | ewc_lambda_tft |
|---|---|---|---|---|---|
| S1 | 0.001 | 0.03 | 50 | 10 | 1000 |
| S6 | 0.003 | 0.01 | 100 | 10 | 100 |

### Table `p1`: P1 (continual stream): final average test RMSE of SOH(k+h) in pp, mean and s.d. over runs; skill = 1 - RMSE/RMSE_persistence; forgetting in pp

CSV: `results/tables/p1.csv`

| dataset | name | rmse_pp_mean | rmse_pp_sd | n_runs | skill_vs_persistence | forgetting_pp_mean | forgetting_ci_low | forgetting_ci_high |
|---|---|---|---|---|---|---|---|---|
| S1 | Persistence | 0.632 | -- | 1 | 0.000 | -- | -- | -- |
| S1 | Linear extrapolation | 0.726 | -- | 1 | -0.149 | -- | -- | -- |
| S1 | Ridge, usage only | 0.450 | -- | 1 | 0.288 | -- | -- | -- |
| S1 | Ridge, F0 | 0.678 | -- | 1 | -0.073 | -- | -- | -- |
| S1 | Ridge, F1 | 0.491 | -- | 1 | 0.223 | -- | -- | -- |
| S1 | Gradient boosting, F1 | 0.466 | -- | 1 | 0.264 | -- | -- | -- |
| S1 | MLP fine-tuning | 0.492 | 0.029 | 50 | 0.222 | 0.030 | 0.021 | 0.040 |
| S1 | MLP replay | 0.475 | 0.021 | 50 | 0.249 | 0.025 | 0.017 | 0.033 |
| S1 | MLP + EWC | 0.497 | 0.033 | 50 | 0.213 | 0.040 | 0.031 | 0.050 |
| S1 | MLP joint (upper bound) | 0.500 | 0.016 | 5 | 0.209 | -- | -- | -- |
| S1 | TFT fine-tuning | 0.467 | 0.014 | 50 | 0.261 | 0.026 | 0.019 | 0.034 |
| S1 | TFT + EWC | 0.462 | 0.019 | 50 | 0.268 | 0.016 | 0.011 | 0.021 |
| S1 | TFT + EWC, F0 | 0.472 | 0.020 | 50 | 0.254 | 0.019 | 0.015 | 0.024 |
| S1 | TFT joint (upper bound) | 0.455 | 0.017 | 5 | 0.280 | -- | -- | -- |
| S6 | Persistence | 1.644 | -- | 1 | 0.000 | -- | -- | -- |
| S6 | Linear extrapolation | 0.343 | -- | 1 | 0.792 | -- | -- | -- |
| S6 | Ridge, usage only | 1.274 | -- | 1 | 0.225 | -- | -- | -- |
| S6 | Ridge, F0 | 0.801 | -- | 1 | 0.513 | -- | -- | -- |
| S6 | Ridge, F1 | 0.605 | -- | 1 | 0.632 | -- | -- | -- |
| S6 | Gradient boosting, F1 | 0.372 | -- | 1 | 0.773 | -- | -- | -- |
| S6 | MLP fine-tuning | 0.531 | 0.151 | 50 | 0.677 | 0.207 | 0.165 | 0.251 |
| S6 | MLP replay | 0.597 | 0.178 | 50 | 0.637 | 0.212 | 0.161 | 0.267 |
| S6 | MLP + EWC | 0.540 | 0.176 | 50 | 0.671 | 0.212 | 0.163 | 0.263 |
| S6 | MLP joint (upper bound) | 0.414 | 0.037 | 5 | 0.748 | -- | -- | -- |
| S6 | TFT fine-tuning | 0.883 | 0.538 | 50 | 0.463 | 0.461 | 0.315 | 0.629 |
| S6 | TFT + EWC | 0.677 | 0.261 | 50 | 0.588 | 0.274 | 0.203 | 0.357 |
| S6 | TFT + EWC, F0 | 0.621 | 0.257 | 50 | 0.623 | 0.176 | 0.115 | 0.246 |
| S6 | TFT joint (upper bound) | 0.427 | 0.049 | 5 | 0.741 | -- | -- | -- |

### Table `horizon`: P1 test RMSE (pp) by horizon h (RPTs or checks ahead)

CSV: `results/tables/horizon.csv`

| dataset | name | rmse_pp_h1 | rmse_pp_h2 | rmse_pp_h3 |
|---|---|---|---|---|
| S1 | Persistence | 0.490 | 0.647 | 0.826 |
| S1 | Linear extrapolation | 0.646 | 0.667 | 0.903 |
| S1 | Ridge, usage only | 0.401 | 0.480 | 0.488 |
| S1 | Ridge, F0 | 0.657 | 0.611 | 0.751 |
| S1 | Ridge, F1 | 0.478 | 0.439 | 0.572 |
| S1 | Gradient boosting, F1 | 0.469 | 0.444 | 0.458 |
| S1 | MLP fine-tuning | 0.511 | 0.453 | 0.477 |
| S1 | MLP replay | 0.501 | 0.428 | 0.458 |
| S1 | MLP + EWC | 0.513 | 0.460 | 0.491 |
| S1 | MLP joint (upper bound) | 0.529 | 0.451 | 0.487 |
| S1 | TFT fine-tuning | 0.462 | 0.452 | 0.475 |
| S1 | TFT + EWC | 0.459 | 0.450 | 0.465 |
| S1 | TFT + EWC, F0 | 0.463 | 0.459 | 0.484 |
| S1 | TFT joint (upper bound) | 0.475 | 0.431 | 0.421 |
| S6 | Persistence | 0.803 | 1.584 | 2.383 |
| S6 | Linear extrapolation | 0.209 | 0.347 | 0.459 |
| S6 | Ridge, usage only | 0.716 | 1.243 | 1.746 |
| S6 | Ridge, F0 | 0.352 | 0.727 | 1.179 |
| S6 | Ridge, F1 | 0.664 | 0.406 | 0.658 |
| S6 | Gradient boosting, F1 | 0.270 | 0.345 | 0.496 |
| S6 | MLP fine-tuning | 0.477 | 0.456 | 0.601 |
| S6 | MLP replay | 0.495 | 0.525 | 0.714 |
| S6 | MLP + EWC | 0.455 | 0.468 | 0.636 |
| S6 | MLP joint (upper bound) | 0.273 | 0.348 | 0.583 |
| S6 | TFT fine-tuning | 0.588 | 0.885 | 1.115 |
| S6 | TFT + EWC | 0.532 | 0.617 | 0.835 |
| S6 | TFT + EWC, F0 | 0.459 | 0.562 | 0.797 |
| S6 | TFT joint (upper bound) | 0.281 | 0.439 | 0.541 |

### Table `tests`: P1 paired comparisons: TFT + EWC minus the other method (pp; negative favours TFT + EWC), 95 % bootstrap CI, Holm-adjusted Wilcoxon p, rank-biserial r

CSV: `results/tables/tests.csv`

| dataset | metric | compared_with | diff_pp | ci_low | ci_high | n | p_holm | rank_biserial |
|---|---|---|---|---|---|---|---|---|
| S1 | final_avg_error | ewc | -0.035 | -0.044 | -0.026 | 50 | <0.001 | -0.865 |
| S1 | final_avg_error | finetune | -0.029 | -0.038 | -0.021 | 50 | <0.001 | -0.845 |
| S1 | final_avg_error | replay | -0.012 | -0.021 | -0.004 | 50 | 0.083 | -0.357 |
| S1 | final_avg_error | tft_ewc_F0 | -0.009 | -0.017 | -0.001 | 50 | 0.083 | -0.346 |
| S1 | final_avg_error | tft_finetune | -0.005 | -0.011 | 0.001 | 50 | 0.278 | -0.178 |
| S1 | forgetting | ewc | -0.024 | -0.034 | -0.015 | 50 | <0.001 | -0.688 |
| S1 | forgetting | finetune | -0.014 | -0.023 | -0.005 | 50 | 0.052 | -0.401 |
| S1 | forgetting | replay | -0.009 | -0.018 | -0.000 | 50 | 0.247 | -0.283 |
| S1 | forgetting | tft_ewc_F0 | -0.003 | -0.009 | 0.004 | 50 | 0.559 | -0.096 |
| S1 | forgetting | tft_finetune | -0.010 | -0.020 | -0.000 | 50 | 0.247 | -0.255 |
| S1 | final_avg_error | gbm_F1 | -0.003 | -0.008 | 0.002 | 50 | 0.175 | -0.222 |
| S1 | final_avg_error | linear_extrapolation | -0.264 | -0.269 | -0.259 | 50 | <0.001 | -1.000 |
| S1 | final_avg_error | persistence | -0.170 | -0.175 | -0.164 | 50 | <0.001 | -1.000 |
| S1 | final_avg_error | ridge_F0 | -0.216 | -0.221 | -0.210 | 50 | <0.001 | -1.000 |
| S1 | final_avg_error | ridge_F1 | -0.029 | -0.034 | -0.023 | 50 | <0.001 | -0.947 |
| S1 | final_avg_error | ridge_schedule | 0.012 | 0.007 | 0.017 | 50 | <0.001 | 0.658 |
| S6 | final_avg_error | ewc | 0.137 | 0.048 | 0.232 | 50 | 0.024 | 0.427 |
| S6 | final_avg_error | finetune | 0.146 | 0.061 | 0.238 | 50 | 0.006 | 0.503 |
| S6 | final_avg_error | replay | 0.081 | -0.008 | 0.176 | 50 | 0.274 | 0.180 |
| S6 | final_avg_error | tft_ewc_F0 | 0.057 | -0.024 | 0.133 | 50 | 0.061 | 0.351 |
| S6 | final_avg_error | tft_finetune | -0.206 | -0.338 | -0.091 | 50 | 0.005 | -0.520 |
| S6 | forgetting | ewc | 0.062 | -0.028 | 0.162 | 50 | 1.000 | 0.129 |
| S6 | forgetting | finetune | 0.067 | -0.020 | 0.162 | 50 | 1.000 | 0.150 |
| S6 | forgetting | replay | 0.062 | -0.028 | 0.162 | 50 | 1.000 | 0.100 |
| S6 | forgetting | tft_ewc_F0 | 0.098 | 0.023 | 0.171 | 50 | 0.004 | 0.536 |
| S6 | forgetting | tft_finetune | -0.187 | -0.325 | -0.067 | 50 | 0.050 | -0.402 |
| S6 | final_avg_error | gbm_F1 | 0.305 | 0.238 | 0.382 | 50 | <0.001 | 0.991 |
| S6 | final_avg_error | linear_extrapolation | 0.335 | 0.268 | 0.412 | 50 | <0.001 | 0.997 |
| S6 | final_avg_error | persistence | -0.967 | -1.034 | -0.890 | 50 | <0.001 | -1.000 |
| S6 | final_avg_error | ridge_F0 | -0.124 | -0.190 | -0.046 | 50 | <0.001 | -0.584 |
| S6 | final_avg_error | ridge_F1 | 0.072 | 0.005 | 0.149 | 50 | 0.300 | 0.170 |
| S6 | final_avg_error | ridge_schedule | -0.597 | -0.664 | -0.520 | 50 | <0.001 | -0.991 |

### Table `p2`: P2 (leave one group out: unseen cell in S1, unseen usage condition in S6): test RMSE (pp), range over held-out groups, S6 reuse-class accuracy / macro-F1

CSV: `results/tables/p2.csv`

| dataset | name | rmse_pp_mean | rmse_pp_min | rmse_pp_max | hardest_group | class_accuracy | class_macro_f1 |
|---|---|---|---|---|---|---|---|
| S1 | Persistence | 0.545 | 0.436 | 0.688 | v5 | -- | -- |
| S1 | Linear extrapolation | 0.511 | 0.284 | 0.725 | v5 | -- | -- |
| S1 | Ridge, usage only | 0.301 | 0.172 | 0.409 | w8 | -- | -- |
| S1 | Ridge, F0 | 0.418 | 0.158 | 0.779 | v5 | -- | -- |
| S1 | Ridge, F1 | 0.320 | 0.161 | 0.477 | v5 | -- | -- |
| S1 | Gradient boosting, F1 | 0.219 | 0.091 | 0.299 | w8 | -- | -- |
| S1 | MLP | 0.285 | 0.149 | 0.389 | w8 | -- | -- |
| S1 | TFT | 0.292 | 0.199 | 0.377 | v5 | -- | -- |
| S6 | Persistence | 2.817 | 2.025 | 4.327 | case7 | 0.850 | 0.635 |
| S6 | Linear extrapolation | 1.372 | 0.856 | 1.669 | case2 | 0.936 | 0.750 |
| S6 | Ridge, usage only | 1.177 | 0.870 | 2.288 | case7 | 0.952 | 0.827 |
| S6 | Ridge, F0 | 0.859 | 0.598 | 1.836 | case7 | 0.962 | 0.862 |
| S6 | Ridge, F1 | 1.137 | 0.588 | 2.188 | case7 | 0.961 | 0.853 |
| S6 | Gradient boosting, F1 | 0.710 | 0.395 | 1.839 | case7 | 0.966 | 0.882 |
| S6 | MLP | 0.821 | 0.476 | 2.225 | case7 | 0.962 | 0.868 |
| S6 | TFT | 0.868 | 0.551 | 2.390 | case7 | 0.962 | 0.877 |

<!-- END GENERATED TABLES -->

## D21 protocol P3: first-life pretraining, then the S6 stream (3 Oct 2026)

Results file: `results/forecasting_d21_p3.json`, written by `scripts/run_p3.py` at commit `57e37aa` (clean; Python 3.11.15, numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, torch 2.14.1). Protocol: D21 and its disclosures. Pool: S7–S9, 369 cells, 36,711 samples (train 21,174 / val 6,439 / test 6,942); stream: the seven S6 test cases, 2,409 samples, 10 orders; main seeds 0–4 and replicate seeds 5–9 (50 runs each per method).

**Tuning (validation only, 10 orders, one-SE rule, D11 edge rule).** MLP: learning rate 3e-4 (best value at the original lower edge; after extending the grid to 1e-4 and 3e-5 it is interior), EWC λ = 1000. TFT: learning rate 3e-3, EWC λ = 0.01 (the best value on the original grid was its lower edge 1; after extension the best is 0.1 and the one-SE rule selects 0.01). **For the pretrained TFT, validation selects almost no EWC.**

**Final average test RMSE on S6 (pp), mean (s.d.; median) over 50 runs:**

| Method | Main seeds | Replicate seeds |
|---|---|---|
| TFT from scratch, fine-tuning | 0.401 (0.169; 0.353) | 0.415 (0.241; 0.347) |
| MLP from scratch, fine-tuning | 0.496 (0.270; 0.377) | 0.475 (0.248; 0.385) |
| TFT pretrained + EWC | 0.509 (0.347; 0.402) | 0.591 (0.428; 0.443) |
| TFT pretrained, fine-tuning | 0.525 (0.346; 0.409) | 0.606 (0.441; 0.451) |
| MLP pretrained + EWC | 0.735 (0.436; 0.579) | 0.631 (0.293; 0.513) |
| MLP pretrained, fine-tuning | 0.844 (0.686; 0.530) | 0.777 (0.579; 0.500) |

Baselines (S6 training rows, D20 definitions, shared inputs): linear extrapolation 0.343, gradient boosting 0.451, schedule ridge 1.079, ridge 1.184, persistence 1.644.

**Pre-registered comparisons (each method vs pretrained TFT + EWC; D21 replication rule: p_Holm < 0.05 in both seed sets with the same sign and |difference| ≥ 0.05 pp in both = replicated; |difference| < 0.05 pp in both = practically equivalent; otherwise not replicated).** Differences are method minus TFT + EWC, main / replicate.

| Method | Final error | Forgetting | Backward transfer |
|---|---|---|---|
| TFT pretrained, fine-tuning | +0.016 / +0.015: **practically equivalent** | +0.016 / +0.016: **practically equivalent** | +0.017 / +0.016: **practically equivalent** |
| MLP pretrained, fine-tuning | +0.334 / +0.185 (p_Holm < 0.001 / < 0.001): **replicated** | +0.320 / +0.178 (< 0.001 / 0.002): **replicated** | +0.303 / +0.182 (< 0.001 / 0.041): **replicated** |
| MLP pretrained + EWC | +0.225 / +0.040: not replicated | +0.157 / −0.004: not replicated | +0.141 / +0.018: not replicated |
| TFT from scratch | −0.108 / −0.176 (0.24 / < 0.001): not replicated | −0.083 / −0.135: not replicated | −0.076 / −0.147: not replicated |
| MLP from scratch | −0.013 / −0.116: not replicated | +0.002 / −0.097: not replicated | −0.005 / −0.128: not replicated |

**Retention of first-life knowledge.** Pool test RMSE before → after the S6 stream: TFT pretrained + EWC 3.280 → 3.148 (main) and 3.246 → 3.120 (replicate); TFT pretrained fine-tuning 3.280 → 3.153 and 3.246 → 3.134; MLP pretrained + EWC 3.494 → 3.395 and 3.537 → 3.443; MLP pretrained fine-tuning 3.494 → 3.377 and 3.537 → 3.434. The stream does not degrade the pool error. For context, on the pool test split persistence gives 3.878 pp and linear extrapolation 3.001 pp (S7 7.008 / 5.069, S8 2.128 / 2.325, S9 1.099 / 1.151): **the pretrained models are less accurate on the pool than linear extrapolation.**

**What P3 shows.**
1. **EWC does not help the pretrained TFT** (validation selects λ = 0.01, and test differences are below 0.05 pp in both seed sets).
2. **Pretrained TFT + EWC beats pretrained MLP fine-tuning, replicated** in final error, forgetting and backward transfer.
3. **Pretraining on the first-life pool does not improve the S6 forecasts.** Both backbones are, on average, better from scratch; for the TFT the difference is significant in the replicate set only (not replicated). An exploratory contrast (outside the pre-registered family) shows pretraining worsens the MLP in both seed sets. A likely reason is that the pretrained models do not beat linear extrapolation even on the pool itself: with these shared inputs, pretraining does not learn a model that transfers.
4. **No forgetting of the pool:** the S6 stream leaves pool error unchanged or slightly lower, with or without EWC.
5. **The best neural forecaster, TFT from scratch (0.40–0.42 pp), is below gradient boosting (0.451 pp) on average, but linear extrapolation (0.343 pp) remains the most accurate method on S6.**
6. **Not attributable yet:** TFT from scratch reaches 0.401 pp here against 0.883 pp for TFT fine-tuning under D20. The two runs differ in inputs (calendar time, charge voltage and CV phase, chemistry and condition codes), in tuning (10 orders, one-SE and edge rules) and in preprocessing scope, so the gain cannot be attributed to any one change without an ablation.

### D21 addendum A1: what changed the TFT between D20 and D21 (post hoc, replicated on a second seed set)

Results files: `results/forecasting_d21_ablation_a1.json` (seeds 0–4, commit `0b739a2`, clean) and `results/forecasting_d21_ablation_a1_replicate.json` (seeds 5–9, specified before running; the file records commit `72427b8`, which was made while the run was in progress and changed only `paper/` and `scripts/make_framework_figure.py`; `src/` and `scripts/run_ablation_a1.py` are identical to `a02dbe0`, where the run started). TFT from scratch with fine-tuning on the S6 stream, 10 orders × 5 seeds per set. Final average test RMSE (pp), mean (median), main / replicate:

| Inputs | lr 1e-2 | lr 3e-3 |
|---|---|---|
| D20 F1 (with 17 diagnostics) | 0.891 (0.726) / 0.809 (0.669) | 0.767 (0.644) / 0.797 (0.701) |
| D20 F0 (no diagnostics) | 0.540 (0.438) / 0.636 (0.453) | 0.534 (0.422) / 0.589 (0.504) |
| D21 shared inputs | 0.424 (0.392) / 0.444 (0.379) | 0.401 (0.353) / 0.415 (0.347) |
| D21 shared, no calendar-time inputs | 0.731 (0.614) / 0.673 (0.585) | 0.619 (0.582) / 0.604 (0.482) |

Contrasts (mean paired difference in pp, main / replicate, with p_Holm), judged by the D21 replication rule:

| Contrast | lr 1e-2 | lr 3e-3 |
|---|---|---|
| D20 F0 − D20 F1 (drop diagnostics) | −0.351 / −0.173 (< 0.001 / 0.002): **replicated** | −0.234 / −0.208 (< 0.001 / 0.001): **replicated** |
| D21 shared − D21 no calendar (add calendar time) | −0.307 / −0.229 (< 0.001 / < 0.001): **replicated** | −0.218 / −0.189 (< 0.001 / < 0.001): **replicated** |
| D21 shared − D20 F1 | −0.467 / −0.365 (< 0.001 / < 0.001): **replicated** | −0.366 / −0.382 (< 0.001 / < 0.001): **replicated** |
| D21 shared − D20 F0 | −0.116 / −0.192 (0.076 / 0.006): not replicated | −0.132 / −0.174 (0.005 / < 0.001): **replicated** |

Learning-rate contrasts (3e-3 − 1e-2): D21 shared −0.023 / −0.029 and D20 F0 −0.006 / −0.047, both **practically equivalent** (|d| < 0.05 pp in both sets); D20 F1 −0.124 / −0.012 and D21 no calendar −0.112 / −0.070, not significant in either set (inconclusive).

- **Sanity check passed:** D20 F1 at 1e-2 gives 0.891 pp against 0.883 pp in D20.
- In the replicate, the bootstrap CI of the mean for D20 F0 − F1 at 1e-2 just includes zero (−0.342 to +0.005) while the Wilcoxon test is significant; the effect is skewed by a few runs.

**Reading.** The D20 → D21 gain of the TFT comes from dropping the 17 diagnostic features and adding calendar time, and both effects replicate on seeds 5–9 at both learning rates. The learning rate does not explain it. The analysis is still post hoc (decided after the P3 results) and is reported as explanatory.
