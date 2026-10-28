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

   Our recomputation from the raw S1 files reproduces the handover capacities exactly for all 112 RPT capacity rows (111 RPTs; g1's RPT 1 spans two files), and our S6 parsing reproduces the authors' own C/3 capacity on all 843 checks.
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

The D25–D27 figures are in `paper/figures/`, drawn from the committed results files by `scripts/make_d25_d27_figures.py` (shown in section 7 of `notebooks/colab_review2.ipynb`):

| Figure | Content |
|---|---|
| `d25_shrinkage.png` | D25, S6 stream: shrinkage factor of the TFT correction per horizon after the last task (main seeds, median and interquartile range over 50 runs), E1 and E2 (in the paper) |
| `d26_per_condition.png` | D26: test RMSE per held-out S6 condition for linear extrapolation, gradient boosting, the shrunk residual TFT + EWC and the jointly trained residual TFT (main seeds, mean) |
| `d27_coverage.png` | D27: coverage of 90 % intervals per held-out condition, by forecaster and calibration rows; the TFT from D26 (main seeds, mean) (in the paper, replacing the D27 table) |

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

## D23: learned residual on linear extrapolation, and horizons 1–6 (5 Oct 2026)

Results files: `results/forecasting_d23_e1.json` (horizons 1–3) and `results/forecasting_d23_e2.json` (horizons 1–6), collected at commit `5ae2f84` (clean). The units were computed by processes started at `c7fa340`; the only change between the two commits is the D23 run-log entry in `docs/DECISIONS.md`. Protocol fixed in D23 before any run (commit `4cc2df7`). S6 stream from scratch, D21 shared inputs, D21 tuning (10 tuning orders, one-SE rule, grid-edge extension), 10 orders × seeds 0–4 (main) and 5–9 (replicate). Reference: residual TFT + EWC. Differences are reference minus other (pp; negative favours the reference).

**Final average test RMSE (pp), mean, main / replicate:**

| Method | E1 (h = 1–3) | E2 (h = 1–6) |
|---|---|---|
| **TFT + EWC, residual** | **0.339 / 0.314** | **0.561 / 0.534** |
| TFT fine-tuning, residual | 0.347 / 0.337 | 0.673 / 0.612 |
| TFT + EWC, direct | 0.504 / 0.487 | 0.802 / 0.835 |
| TFT fine-tuning, direct | 0.424 / 0.444 | 0.658 / 0.694 |
| MLP + EWC, residual | 0.374 / 0.366 | 0.659 / 0.694 |
| MLP fine-tuning, residual | 0.405 / 0.403 | 0.656 / 0.695 |
| MLP + EWC, direct | 0.519 / 0.527 | 1.004 / 0.996 |
| MLP fine-tuning, direct | 0.413 / 0.351 | 0.781 / 0.775 |
| Linear extrapolation | 0.343 | 0.485 |
| Gradient boosting (shared inputs) | 0.451 | 0.898 |
| Persistence | 1.644 | 2.667 |

**Registered tests (D21 replication rule):**

| Comparison (residual TFT + EWC minus …) | E1 | E2 |
|---|---|---|
| **(a) linear extrapolation, final error** (primary) | −0.004 / −0.029 (p_Holm 0.46 / 0.046): **practically equivalent** | +0.076 / +0.049 (0.026 / 0.16): not replicated |
| **(b) direct TFT + EWC, final error** (primary) | −0.165 / −0.174 (< 0.001 / < 0.001): **replicated** | −0.241 / −0.301 (< 0.001 / < 0.001): **replicated** |
| residual TFT fine-tuning, final error | −0.008 / −0.023: practically equivalent | −0.112 / −0.078 (0.096 / 0.17): not replicated |
| residual TFT fine-tuning, forgetting | −0.004 / −0.020: practically equivalent | −0.189 / −0.134 (< 0.001 / 0.002): **replicated** |
| residual TFT fine-tuning, backward transfer | −0.005 / −0.020: practically equivalent | −0.145 / −0.108 (0.026 / 0.043): **replicated** |
| gradient boosting, final error | −0.112 / −0.138: **replicated** | −0.337 / −0.365: **replicated** |
| direct MLP + EWC, final error | −0.180 / −0.213: **replicated** | −0.443 / −0.462: **replicated** |
| residual MLP fine-tuning, final error | −0.066 / −0.089: **replicated** | −0.095 / −0.162: not replicated |
| residual MLP + EWC, final error | −0.035 / −0.052: not replicated | −0.098 / −0.160: not replicated |

**E2, per horizon (residual TFT + EWC minus linear extrapolation, pooled test RMSE after the last step, family of six):** h1 +0.189 / +0.144 and h2 +0.103 / +0.076 (both replicated, extrapolation better); h3 +0.068 / +0.043 and h4 +0.090 / +0.041 (not replicated); h5 +0.047 / −0.000 and h6 +0.021 / −0.027 (practically equivalent). Extrapolation's advantage shrinks with the horizon, but the residual model does not overtake it by h = 6.

**Tuning.** Residual TFT: lr 3e-3 in both experiments; EWC λ 1 in E1 (the λ grid was extended down to 0.01; best 1), 1000 in E2. Direct TFT: lr 1e-2 / λ 10 (E1), lr 3e-3 / λ 10⁴ (E2). Residual MLP in E2: lr 3e-5 after a low-edge extension (best 1e-4), i.e. it barely moves from its initial near-zero correction.

**Reading.**
1. **The registered positive result was not obtained.** The residual TFT + EWC does not beat linear extrapolation: it is practically equivalent at horizons 1–3 and not better at horizons 1–6.
2. **Learning the residual is what makes the TFT competitive.** It improves TFT + EWC by 0.17 pp (E1) and 0.24–0.30 pp (E2), replicated, and makes it the most accurate learned model of the study, ahead of gradient boosting (replicated in both experiments). It is the first learned model level with linear extrapolation on S6.
3. **EWC reduces forgetting, replicated, for the first time in this study.** With horizons 1–6, EWC lowers the residual TFT's forgetting from 0.28 / 0.22 to 0.09 pp and its backward transfer from 0.18 / 0.14 to 0.03 pp. Its effect on final error is not replicated. With horizons 1–3, where forgetting is already small for the residual TFT (about 0.1 pp), EWC and fine-tuning are practically equivalent.
4. The residual is hard to learn from the shared inputs: per horizon, extrapolation stays better at h = 1–2 and the gap closes, without reversing, by h = 5–6.

## D22: S6 between-check ageing cycles as inputs, extracted with Apache Spark (6 Oct 2026)

**Data and extraction.** All 827 ageing files (131.9 GB raw, 901,974,259 rows (`results/s6_ageing_row_count.json`); 13 download manifests merged, no SHA-256 conflict). Pandas extraction: 82,529 cycles, 80,287 valid (97 %), 827 blocks, 7 blocks below 50 valid cycles (inputs missing). No block ends after a valid check. Full Spark run (PySpark 4.2.0, local[6]): 827 map tasks, 82,529 cycles, 1,175 s; matches the pandas implementation on all 827 blocks under |Spark − pandas| ≤ 1e-9·|pandas| + 1e-12 (largest absolute difference 2.0e-13); the originally stated relative-only criterion failed on one near-zero input (deviation recorded in D22). Files: `results/d22_ageing_report.json`, `results/d22_spark_validation.json`, `results/d22_spark_validation_strict_relative.json`.

**Leakage audit.** Per-cycle discharged charge correlates r = 0.97–0.999 within cell with the previous check's capacity in every case (protocol scaling; excluded). Allowed responses carry health information: median within-cell Spearman ρ with the origin capacity −0.86 (relaxation after charge, mean) and 0.91 (CV-hold time, mean). `Internal_Resistance` is never recorded (inputs missing throughout).

**Forecasting** (`results/forecasting_d22_e1_f2.json`, commit `d69b495`, clean). D23 E1 setting, inputs F2 = D21 shared + 21 ageing inputs (Spark output). The committed D23 runs of residual TFT + EWC (D21 inputs) reproduce exactly on two recomputed units before they are used as reference (b). Final average test RMSE (pp), mean, main / replicate:

| Method (F2 inputs unless stated) | Final error |
|---|---|
| **TFT + EWC, residual** | **0.480 / 0.501** |
| TFT + EWC, residual, **D21 inputs** (D23) | 0.339 / 0.314 |
| TFT fine-tuning, residual | 0.756 / 0.645 |
| TFT + EWC, direct | 0.757 / 0.709 |
| MLP fine-tuning, residual | 0.398 / 0.354 |
| MLP + EWC, residual | 0.691 / 0.656 |
| Linear extrapolation | 0.343 |
| Gradient boosting (F2) | 0.495 |

**Registered tests (D21 replication rule), residual TFT + EWC with F2 minus …:**
- **(a) linear extrapolation: +0.138 / +0.159 (p_Holm < 0.001 / < 0.001), replicated — worse. The registered positive result was not obtained.**
- **(b) the same model with the D21 inputs: +0.141 / +0.188 (< 0.001 / < 0.001), replicated — the ageing inputs make it worse.**
- Better than the direct TFT + EWC (−0.276 / −0.208) and direct TFT fine-tuning, and than all direct MLPs and the ridge baselines (replicated); worse than residual MLP fine-tuning (+0.082 / +0.147, replicated); practically equivalent to gradient boosting (−0.015 / +0.006).

**Tuning.** Residual TFT: lr 0.1 (the grid was extended upward; 0.1 stayed best), λ 100. Residual MLP: lr 0.03, λ 100.

**Reading.** Adding the between-check ageing responses does not help the proposed model: residual TFT + EWC is 0.14–0.19 pp worse with them (replicated), and so are the other TFT and MLP variants and gradient boosting (0.495 against 0.451) compared with the D23 runs on the D21 inputs, except residual MLP fine-tuning, which is slightly better with them (0.398 / 0.354 against 0.405 / 0.403; not a registered comparison). No model beats linear extrapolation. The responses do track health within a cell, but mostly its level, which the SOH history already gives; with 21 extra inputs and 135–225 training samples per task, they add variance (TFT + EWC: mean 0.48 against median 0.40). This is an interpretation, not a tested claim. The Spark pipeline itself is validated and reusable.

## D24: RC-TFT (shift-adaptive EWC) and literature baselines (13–14 Oct 2026)

Results files: `results/forecasting_d24_e1.json` (horizons 1–3) and `results/forecasting_d24_e2.json` (horizons 1–6), run and collected at commit `947b33a` (clean tree, `git_dirty` false), Python 3.12.10, torch 2.13.0+cpu; all 13 parts ran, none failed (run log in `docs/DECISIONS.md`, D24). Protocol fixed in D24 before any run (`be2f2f5`). D23 setting: S6 stream from scratch, D21 shared inputs, 10 orders × seeds 0–4 (main) and 5–9 (replicate), D23 tuning rules. The committed D23 runs (residual and direct TFT and MLP) enter after two recomputed units reproduced (E1 0.464125 and 0.375621 pp, E2 0.480461 and 0.880787 pp, equal to about 1e-14). Reference: RC-TFT. Differences are RC-TFT minus other (pp; negative favours RC-TFT). Tables: `results/tables/d24_*.csv`, `paper/tables/d24_*.tex`; figures `paper/figures/d24_*.png`; summary `scripts/summarise_d24.py`.

**Final average test RMSE (pp), mean, main / replicate; forgetting (pp), main / replicate:**

| Method | E1 error | E1 forgetting | E2 error | E2 forgetting |
|---|---|---|---|---|
| **RC-TFT (residual TFT + SA-EWC)** | 0.343 / 0.321 | 0.100 / 0.072 | 0.548 / 0.564 | 0.066 / 0.084 |
| TFT + EWC, residual (fixed λ, D23) | 0.339 / 0.314 | 0.100 / 0.066 | 0.561 / 0.534 | 0.093 / 0.086 |
| TFT fine-tuning, residual | 0.347 / 0.337 | 0.103 / 0.086 | 0.673 / 0.612 | 0.282 / 0.220 |
| BiLSTM + EWC, residual | 0.328 / 0.370 | 0.091 / 0.136 | 0.669 / 0.765 | 0.229 / 0.318 |
| BiLSTM + EWC, direct | 0.323 / 0.323 | 0.053 / 0.056 | not run | |
| Transformer + EWC, residual | 0.360 / 0.366 | 0.064 / 0.080 | 0.609 / 0.705 | 0.129 / 0.198 |
| TCN + EWC, residual | 0.466 / 0.408 | 0.193 / 0.111 | 0.812 / 0.756 | 0.345 / 0.268 |
| TFT + EWC, direct | 0.504 / 0.487 | 0.228 / 0.217 | 0.802 / 0.835 | 0.135 / 0.166 |
| Linear extrapolation | 0.343 | | 0.485 | |
| SVR, residual (single fit on all tasks) | 0.429 | | 0.616 | |
| Random forest, direct (single fit) | 0.487 | | 0.946 | |
| Gradient boosting (single fit) | 0.510 | | 0.831 | |
| GPR, direct (single fit, ≤ 1,000 rows) | 1.374 | | 2.073 | |
| Persistence | 1.644 | | 2.667 | |

Gradient boosting differs from D23 (0.451 / 0.898 pp) because of the software environment, not the code: the night run used Python 3.12.10, numpy 1.26.4, pandas 2.2.3 and scikit-learn 1.5.2 instead of the pinned versions (Python 3.11.15, numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1). In the pinned environment the same `baselines` part gives 0.451 / 0.898 pp (`results/forecasting_d24_baselines_e1.json`, `..._e2.json`, commit `ecd882b`), identical to D23; persistence, linear extrapolation, ridge, GPR and SVR are identical in both environments and random forest differs by at most 0.01 pp. The paper reports the pinned values; the D24 tables and tests above use the run-environment values. The neural results are not affected in this way: the D23 reference units reproduced to about 1e-14 in the run environment. Parameters: TFT 12,708, BiLSTM 11,181, TCN 12,205, Transformer 11,905. Mean training time per stream: RC-TFT 54 s (E1) and 48 s (E2), fixed-λ TFT + EWC 39 s and 56 s, the three other sequence models 8–30 s.

**Registered primary tests (D21 replication rule):**

| RC-TFT minus … | E1 | E2 |
|---|---|---|
| TFT + EWC, residual (fixed λ), final error | +0.004 / +0.007: **practically equivalent** | −0.013 / +0.030: **practically equivalent** |
| TFT + EWC, residual (fixed λ), forgetting | +0.000 / +0.007: **practically equivalent** | −0.027 / −0.002 (p_Holm 0.080 / 0.92): **practically equivalent** |
| linear extrapolation, final error | +0.000 / −0.022: **practically equivalent** | +0.063 / +0.079 (0.011 / 0.060): not replicated |

**Secondary (final error).** E1: practically equivalent to BiLSTM + EWC (direct and residual), BiLSTM fine-tuning (residual), Transformer + EWC (residual), MLP + EWC (residual) and TFT fine-tuning (residual); better, replicated, than every direct TCN and Transformer arm, BiLSTM fine-tuning (direct), direct TFT + EWC, MLP + EWC (direct), MLP fine-tuning (residual), TCN fine-tuning (residual) and every tabular baseline (GPR, random forest, SVR, gradient boosting, ridge) and persistence. E2: better, replicated, than both BiLSTM residual arms, both TCN residual arms, all MLP arms, direct TFT + EWC and every tabular baseline except SVR residual (not replicated); against the Transformer residual arms and residual TFT fine-tuning, not replicated. Per horizon against linear extrapolation (E2): h1 +0.209 / +0.179 and h2 +0.115 / +0.113 (replicated, extrapolation better); h3–h4 not replicated; h5–h6 practically equivalent.

**SA-EWC weights.** The shift weights d_jt have median 0.42 (E1) and 0.40 (E2), range 0.05–1.14 (n = 2100 task pairs per experiment), so the penalty did vary with the measured shift. Tuning selected λ₀ = 1 in E1 (as the fixed λ in D23) and 10⁴ in E2 (D23 fixed λ: 1000).

**Reading.**
1. **The registered positive result for SA-EWC was not obtained.** RC-TFT and the fixed-λ residual TFT + EWC are practically equivalent on final error and forgetting in both experiments. Under the rule fixed in D24, RC-TFT is reported with this outcome and **the fixed-λ residual TFT + EWC stays the best-supported variant**, being the simpler of the two.
2. **No model beats linear extrapolation.** RC-TFT is level with it at horizons 1–3 and not better at 1–6 (extrapolation better at h = 1–2, replicated), as for the D23 model.
3. **Against the literature backbones**, the residual TFT is not uniquely good at short horizons: BiLSTM + EWC (direct 0.323, residual 0.328 pp) and Transformer + EWC (residual) are practically equivalent to RC-TFT in E1. Over horizons 1–6 the residual TFT is the most accurate learned model, better than BiLSTM and TCN (replicated); the Transformer comparison is not replicated. Direct BiLSTM was not run in E2 (budget, D24).
4. **Learned continual models beat the classical regressors.** GPR, random forest, SVR, gradient boosting and ridge, each fitted once on all tasks (not continual learners), are all worse than RC-TFT, replicated, except SVR residual in E2.
5. **The GPR baseline is a weak fit, not a ceiling for GPR.** GPR (a single maximum-likelihood fit on at most 1,000 training rows, D24 amendment) is worse than persistence in its residual arm (2.584 / 2.315 pp); its fitted kernel puts a length scale of 0.02 on one input, close to the lower bound of its range, so it nearly interpolates the training rows. The GPR numbers describe this fit and should not be read as the best accuracy GPR can reach on S6.
6. **The residual form does not help every backbone** (descriptive; residual and direct arms of the new backbones were not tested against each other). In E1 it lowers the Transformer's error (EWC 0.497 → 0.360 pp) and BiLSTM fine-tuning's (0.436 → 0.346 pp), leaves BiLSTM + EWC unchanged (0.323 direct, 0.328 residual) and does not help the TCN (EWC 0.461 direct, 0.466 residual; fine-tuning 0.450, 0.503).

**Spark scaling** (`results/d22_spark_scaling.json`, 120 blocks): every thread count matches pandas; times 1,467 / 297 / 173 / 150 s at 1 / 2 / 4 / 6 threads. The 1 → 2 thread speed-up (4.9×) is above linear, so the 1-thread run likely includes a one-off cost (JVM start-up or a cold file cache). Not explained; not to be reported as parallel efficiency until it is.

## D25: horizon-wise shrinkage, conformal intervals and residual pretraining (15–16 Oct 2026)

Results files: `results/forecasting_d25_e1.json` (horizons 1–3) and `results/forecasting_d25_e2.json` (horizons 1–6), run and collected at commit `a59debb` (clean tree, `git_dirty` false) in the pinned environment (Python 3.11.15, numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, torch 2.14.1+cpu); all 10 parts ran, none failed (run log in `docs/DECISIONS.md`, D25). Protocol fixed in D25 before any run (`a59debb`). D23 setting: S6 stream from scratch, D21 shared inputs, 10 orders × seeds 0–4 (main) and 5–9 (replicate). **D23 reproduction:** the raw scratch model of all 100 units per experiment equals the committed D23 unit (max_abs_diff 0.0), so "raw scratch" below *is* the D23 residual TFT + EWC. "Shrunk" = linear extrapolation + α_h × the learned correction, α_h fitted per horizon after each task on that task's validation rows. Differences are a − b (pp; negative favours a). Summary: `scripts/summarise_d25.py`.

**Final average test RMSE (pp), forgetting (pp), interval score and coverage at 90 %; mean, main / replicate:**

| Method | E1 error | E1 forgetting | E1 IS90 | E1 cov90 | E2 error | E2 forgetting | E2 IS90 | E2 cov90 |
|---|---|---|---|---|---|---|---|---|
| Residual TFT + EWC, raw (= D23) | 0.339 / 0.314 | 0.100 / 0.066 | – | – | 0.561 / 0.534 | 0.093 / 0.086 | – | – |
| **Residual TFT + EWC, shrunk** | **0.318 / 0.308** | 0.084 / 0.063 | 1.932 / 1.985 | 0.714 / 0.716 | **0.498 / 0.493** | 0.085 / 0.084 | 3.057 / 3.021 | 0.808 / 0.799 |
| Pretrained residual TFT + EWC, raw | 0.557 / 0.540 | 0.094 / 0.032 | – | – | 0.541 / 0.593 | 0.077 / 0.093 | – | – |
| Pretrained residual TFT + EWC, shrunk | 0.422 / 0.415 | 0.104 / 0.093 | 2.553 / 2.358 | 0.773 / 0.805 | 0.500 / 0.525 | 0.069 / 0.080 | 3.029 / 3.095 | 0.810 / 0.815 |
| Linear extrapolation (conformal intervals) | 0.343 | | 1.910 | 0.924 | 0.485 | | 3.048 | 0.948 |
| Persistence | 1.644 | | | | 2.667 | | | |

Mean 90 % interval width (pp), main / replicate: shrunk scratch model 0.945 / 0.963 (E1) and 2.124 / 2.010 (E2); conformal extrapolation 1.748 (E1) and 2.931 (E2). At 80 %, coverage is 0.565 / 0.592 (E1) and 0.707 / 0.698 (E2) for the shrunk scratch model against 0.793 (E1) and 0.850 (E2) for extrapolation.

**Registered primary tests (D21 replication rule):**

| Comparison | E1 | E2 |
|---|---|---|
| (a) shrunk scratch − linear extrapolation, final error | −0.025 / −0.035 (p_Holm 0.003 / 0.001): **practically equivalent** | +0.013 / +0.008 (1.00 / 0.99): **practically equivalent** |
| (b) shrunk pretrained − linear extrapolation, final error | +0.079 / +0.072 (< 0.001 / < 0.001): **replicated, extrapolation better** | +0.015 / +0.040 (1.00 / 0.033): **practically equivalent** |
| (c) raw pretrained − raw scratch (D23), final error | +0.218 / +0.227 (< 0.001 / < 0.001): **replicated, pretraining worse** | −0.020 / +0.059 (1.00 / 0.062): not replicated |
| (d) shrunk scratch − conformal extrapolation, interval score 90 % | +0.022 / +0.075 (0.86 / 0.55): not replicated | +0.010 / −0.027 (0.79 / 0.39): **practically equivalent** |

**Secondary.** Shrunk − raw, final error: scratch E1 −0.021 / −0.006 (practically equivalent), E2 −0.063 / −0.041 (p_Holm < 0.001 in both, but below 0.05 pp in the replicate: not replicated); pretrained E1 −0.135 / −0.125 (replicated, shrunk better), E2 −0.042 / −0.068 (not replicated). Forgetting: every shrunk − raw and pretrained − scratch comparison is practically equivalent or not replicated. Shrunk pretrained − shrunk scratch: E1 +0.104 / +0.107 (replicated, pretrained worse), E2 practically equivalent. Interval score at 80 %, shrunk scratch − conformal extrapolation: E1 not replicated; **E2 −0.179 / −0.218 (p_Holm 0.033 / 0.023): replicated, model better.** Shrunk pretrained − conformal extrapolation: worse at 90 % and 80 % in E1 (replicated), not replicated or practically equivalent in E2.

**E2, per horizon (shrunk − linear extrapolation, pooled test RMSE after the last step).** Scratch: practically equivalent at every horizon h = 1–6 (largest difference +0.041 pp, h4 main). For comparison, the raw D23 model was worse at h1–h2 (replicated, D23). Pretrained: practically equivalent at h1, h2 and h6, **worse at h3 (+0.055 / +0.052) and h4 (+0.080 / +0.067), replicated**, not replicated at h5. Linear extrapolation's RMSE by horizon: 0.217, 0.354, 0.462, 0.572, 0.744, 0.847 pp.

**Shrinkage factors after the last task** (median over the 100 runs per model): scratch E1 h1–h3 0.93, 1.00, 1.00; scratch E2 0.28, 0.66, 0.86, 1.00, 1.00, 1.00; pretrained E1 0.27, 0.93, 1.00; pretrained E2 0.26, 0.57, 0.82, 1.00, 1.00, 1.00.

**Tuning and pool.** Scratch arm: D23 values (lr 3e-3; λ 1 in E1, 1000 in E2). Pretrained arm: E1 lr 3e-3, λ 10⁵ (λ grid extended upward to 10⁷; best not at the final edge); E2 lr 1e-2, λ 1000 (no extension). Pool: 369 cells; 36,711 samples in E1 (train / validation / test 21,174 / 6,439 / 6,942) and 70,116 in E2 (39,178 / 10,232 / 11,144). Mean training time per S6 stream: scratch 44 / 41 s (E1) and 54 / 52 s (E2); pretrained 20 / 18 s and 31 / 32 s (fine-tuning only; pool pretraining time not recorded in the results files).

**Reading.**
1. **Q1, no registered positive result.** With shrinkage, the residual TFT + EWC is **as accurate as** linear extrapolation in both experiments (practically equivalent, E1 and E2), not better. Shrinkage removes the D23 model's loss to extrapolation over horizons 1–6 (E2: 0.561 / 0.534 → 0.498 / 0.493 pp; and at h1–h2 the replicated loss of D23 becomes practical equivalence), mainly by scaling the correction down at short horizons (median α 0.28 at h1, 0.66 at h2). Where the correction is kept in full (h4–h6), it does not beat extrapolation either.
2. **Q2, negative.** Residual pretraining on the first-life pool makes the model worse at horizons 1–3 (+0.22 pp, replicated) and does not help at 1–6 (not replicated). Shrinkage recovers part of the loss (E1: 0.557 → 0.422 pp) by turning the h1 correction down (α 0.27), but the pretrained model stays behind both the scratch model and extrapolation in E1. The pool's linear extrapolation is per day and S6's per cycle (fixed in D25 because the pool has no cycle count), so this result does not separate "first-life data does not transfer" from "the pool's residual target differs from S6's".
3. **Q3, no registered positive result.** The model's conformal intervals are not better than extrapolation's at 90 % (not replicated in E1, practically equivalent in E2). They are narrower (about 54 % of extrapolation's width in E1, 70 % in E2) and **under-cover**: 0.71–0.81 at nominal 90 % (extrapolation 0.92–0.95). The equal interval scores are a trade of width against misses, not better-calibrated intervals. At 80 % in E2 the model's interval score is better (replicated, secondary family).
4. **Forgetting** stays at 0.03–0.10 pp for every arm; neither shrinkage nor pretraining changes it (practically equivalent or not replicated).
5. **Status of the proposed model.** Across D23–D25, no learned model or variant beats linear extrapolation on the S6 continual stream (raw, shrunk, pretrained, SA-EWC, four backbones). The supported claim is equivalence: the shrunk residual TFT + EWC matches extrapolation at every horizon while keeping forgetting at about 0.1 pp. Interpretations 1–3 beyond the registered verdicts are post hoc.

## D26: unseen usage conditions (P2) for the residual TFT + EWC, by origin history (17–18 Oct 2026)

Results files: `results/forecasting_d26_e1.json` (horizons 1–3) and `results/forecasting_d26_e2.json` (horizons 1–6), run and collected at commit `9866d7a` (clean tree, `git_dirty` false) in the pinned environment (Python 3.11.15, numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.17.1, torch 2.14.1+cpu); all 28 parts ran, none failed, all seven folds are in the tests (run log in `docs/DECISIONS.md`, D26). Protocol fixed in D26 before any run (`ac5bd0a`; amendment and code `7a3d29c`). Setting: leave one S6 test case out (seven folds). The residual TFT + EWC is streamed over the other six cases in each of the 10 D23 orders, seeds 0–4 (main) and 5–9 (replicate), with the D23 learning rate and λ (no new tuning). Shrinkage and 90 % / 80 % conformal intervals are fitted once after the stream on the pooled validation rows of the six training cases. A run's P2 error is the mean over the seven folds of the held-out RMSE. H+ = origins where linear extrapolation has a fade rate; H− = the others, where it equals persistence. Differences are a − b (pp; negative favours a). Summary: `scripts/summarise_d26.py`.

**Disclosure (D26).** The D23 learning rate and λ were tuned on streams that used the validation rows of all seven cases, so each held-out case's validation rows took part in choosing those two values; in P2 they are test samples. The comparators (gradient boosting, ridge) use fixed D18 settings and were not tuned.

**Strata (n / H+ / H−), E1:** case1 279/261/18, case2 282/264/18, case3 231/219/12, case4 387/369/18, case5 363/345/18, case6 576/558/18, case7 291/273/18. E2: 444–1,125 samples per case, 24–36 of them H−.

**P2 error (pp), mean over runs, main / replicate; interval score (IS) and pooled coverage at 90 %:**

| Method | E1 all | E1 H+ | E1 H− | E1 IS90 | E1 cov90 | E2 all | E2 H+ | E2 H− | E2 IS90 | E2 cov90 |
|---|---|---|---|---|---|---|---|---|---|---|
| Residual TFT + EWC, raw | 0.885 / 0.900 | 0.793 / 0.802 | 1.717 / 1.768 | – | – | 1.595 / 1.623 | 1.464 / 1.485 | 2.805 / 2.845 | – | – |
| **Residual TFT + EWC, shrunk** | 0.924 / 0.945 | 0.738 / 0.743 | 2.325 / 2.450 | 6.13 / 6.28 | 0.661 / 0.655 | 1.620 / 1.653 | 1.418 / 1.441 | 3.322 / 3.407 | 9.96 / 10.35 | 0.682 / 0.671 |
| Residual TFT, joint (six cases at once) | 0.568 / 0.577 | 0.538 / 0.550 | 0.891 / 0.870 | – | – | 1.214 / 1.184 | 1.216 / 1.189 | 1.105 / 1.025 | – | – |
| Gradient boosting (single fit) | 0.535 | 0.508 | 0.789 | | | 1.051 | 1.049 | 0.961 | | |
| Ridge (single fit) | 0.771 | 0.691 | 1.539 | | | 1.299 | 1.258 | 1.794 | | |
| Linear extrapolation | 1.372 | 0.860 | 4.609 | 8.48 | 0.743 | 2.300 | 1.620 | 7.022 | 13.53 | 0.763 |
| Persistence | 2.817 | 2.663 | 4.609 | | | 4.898 | 4.732 | 7.022 | | |

Mean 90 % width (pp): shrunk model 0.99 (E1) and 2.20 / 2.13 (E2); conformal extrapolation 1.51 and 3.07. At 80 %, pooled coverage is 0.535 / 0.528 (E1) and 0.537 / 0.532 (E2) for the shrunk model, against 0.609 and 0.639 for extrapolation. S.d. of the P2 error over the 50 runs of a set: continual model 0.11–0.20 pp, joint model 0.02–0.05 pp.

**Per fold (all origins), pp; TFT columns are main-set means (the replicate set differs by at most 0.06 pp per fold in E1 and 0.12 pp in E2):**

| Fold | Shrunk TFT + EWC, E1 | Joint TFT, E1 | Gradient boosting, E1 | Linear extrapolation, E1 | Shrunk TFT + EWC, E2 | Joint TFT, E2 | Gradient boosting, E2 |
|---|---|---|---|---|---|---|---|
| case1 | 0.97 | 0.41 | 0.38 | 1.54 | 1.44 | 0.87 | 0.92 |
| case2 | 0.95 | 0.43 | 0.35 | 1.67 | 1.55 | 0.87 | 0.62 |
| case3 | 1.01 | 0.68 | 0.52 | 1.41 | 1.90 | 1.48 | 0.99 |
| case4 | 0.78 | 0.29 | 0.29 | 1.31 | 1.36 | 0.72 | 0.50 |
| case5 | 0.70 | 0.40 | 0.32 | 1.25 | 1.20 | 0.85 | 0.51 |
| case6 | 0.76 | 0.61 | 0.40 | 0.86 | 1.48 | 1.25 | 0.69 |
| case7 | 1.37 | 1.18 | 1.49 | 1.58 | 2.55 | 2.35 | 3.13 |

90 % coverage per fold (E1, main), shrunk model: 0.66, 0.76, 0.68, 0.78, 0.78, 0.59 and **0.38** (case7). Conformal extrapolation: 0.84, 0.74, 0.72, 0.78, 0.77, 0.84 and **0.40**.

**Registered primary tests (D21 replication rule; p_Holm < 1e-10 in every cell):**

| Comparison | E1 | E2 |
|---|---|---|
| (a) shrunk − linear extrapolation, H+ | −0.122 / −0.116: **replicated, model better** | −0.201 / −0.179: **replicated, model better** |
| (b) shrunk − linear extrapolation, all origins | −0.448 / −0.427: replicated, model better | −0.680 / −0.647: replicated, model better |
| (c) shrunk − gradient boosting, all origins | +0.389 / +0.410: **replicated, gradient boosting better** | +0.570 / +0.603: **replicated, gradient boosting better** |
| (d) shrunk − conformal extrapolation, IS90 | −2.35 / −2.20: replicated, model better | −3.57 / −3.18: replicated, model better |

**Registered coverage check:** mean pooled 90 % coverage 0.661 / 0.655 (E1) and 0.682 / 0.671 (E2), below 0.85, so coverage is **not adequate.**

**Secondary tests (replicated unless stated):**
- H−: shrunk is better than extrapolation (−2.29 / −2.16 E1, −3.70 / −3.62 E2) and worse than gradient boosting (+1.54 / +1.66, +2.36 / +2.45).
- Shrunk − raw: +0.040 / +0.045 (E1) and +0.025 / +0.030 (E2), **practically equivalent**.
- **Raw continual − joint residual TFT: +0.317 / +0.323 (E1) and +0.381 / +0.440 (E2), so the continual model is worse.**
- IS80, shrunk − conformal extrapolation: −1.49 / −1.40 (E1) and −2.02 / −1.86 (E2), model better.

**Reuse classes (descriptive, D20 cut-points).** Agreement with the measured class, mean over runs (macro-F1 in brackets):

| Method | E1 | E2 |
|---|---|---|
| Shrunk | 0.95 (0.81) | 0.94 (0.75) |
| Joint TFT | 0.97 (0.90) | 0.96 (0.86–0.88) |
| Gradient boosting | 0.975 (0.91) | 0.968 (0.91) |
| Linear extrapolation | 0.94 (0.75) | 0.92 (0.66) |
| Persistence | 0.85 (0.64) | 0.78 (0.50) |

**Shrinkage factors** (median over runs and folds, main / replicate): E1 h1–h3 0.59 / 0.59, 0.76 / 0.72, 0.73 / 0.70; E2 h1 0.24 / 0.27, h2 0.55 / 0.63, h3 0.75 / 0.78, h4 0.89 / 0.86, h5–h6 about 1.00. Mean training time per run (seven folds): 240–350 s.

**Reading.**
1. **Q1, positive (registered).** On an unseen usage condition, the shrunk residual TFT + EWC is more accurate than linear extrapolation at origins where extrapolation has a fade rate, by 0.12 pp (E1) and 0.18–0.20 pp (E2), replicated. This is the first registered comparison in which the proposed model beats extrapolation. It holds under the tuning disclosure above. The tests are over training randomness, with the seven conditions fixed; with seven conditions from one laboratory, this is not a claim about new conditions in general.
2. **Q2, negative.** Gradient boosting, a single fit on the same shared inputs, is more accurate than the continual model, replicated, by 0.39–0.41 pp (E1) and 0.57–0.60 pp (E2) on all origins, and also on H−. It is better on six of the seven folds; only on case7 (100 % depth of discharge) is the TFT better.
3. **Continual learning has a large cost here.** The same residual TFT trained jointly on the six cases is 0.32–0.44 pp better than the continual stream (replicated), close to gradient boosting in E1 (0.568 against 0.535 pp, not tested), and four to eight times more stable across runs. In P2 the continual model never sees the new condition, so it pays the stream's order effects (forgetting of earlier cases) without any benefit from adaptation.
4. **Shrinkage does not help in P2.** Shrunk and raw are practically equivalent. Shrinkage helps on H+ and hurts on H−, where it pulls the forecast towards extrapolation, which there equals persistence.
5. **Q3, no registered positive result.** The model's interval score is better than conformal extrapolation's (replicated), but coverage is not adequate: 0.66–0.68 at nominal 90 %, and 0.38 on case7. Extrapolation's intervals, which involve no learned model, also under-cover (0.74–0.76). Calibration on the pooled validation rows of six conditions does not transfer to a seventh. Those rows are mid-life origins (60–80 % of each cell's checks) of conditions the model was trained on, while the test covers every origin of a new condition. D26 does not separate which of these two differences causes the under-coverage.
6. **Status of the proposed model.** Across D23–D26:
   - On the continual stream (P1), the shrunk residual TFT + EWC is as accurate as extrapolation.
   - On unseen conditions (P2), it beats extrapolation but neither gradient boosting nor its own joint-trained form, and its intervals under-cover.

   Interpretations 2–5 beyond the registered verdicts are post hoc.

## D27: leave-one-condition-out conformal intervals for unseen usage conditions (19 Oct 2026)

Results files: `results/forecasting_d27_e1.json` (horizons 1–3) and `results/forecasting_d27_e2.json` (horizons 1–6), run at commit `494b36f` (clean tree, `git_dirty` false, no override) in the pinned environment (Python 3.11.15, numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.17.1). Protocol fixed in D27 before any code (`cb29730`). **Reproduction checks passed:** the D26 gradient-boosting per-fold RMSE and the D26 extrapolation interval metrics are reproduced exactly (max abs diff 0). Setting as D26: leave one S6 test case out, seven folds.

The calibration rules are:
- **GBM-LOCO:** gradient boosting (the D26 comparator), with per-horizon split-conformal half-widths from the six training cases, each predicted by a gradient boosting fitted on the other five.
- **LIN-all:** linear extrapolation, calibrated on every origin of the six training cases.
- **LIN-val:** extrapolation calibrated on their validation rows only (the D26 rule).
- **Mondrian:** the same rules with half-widths per (horizon, origin stratum).

The forecasters are deterministic, so there are no seed sets; E1 and E2 are not independent (E2 contains E1's horizons). No half-width was undersized.

**90 % intervals, pooled coverage (H+ / H−), mean interval score (IS90, pp), mean width (pp), share of samples whose interval lies in one reuse class ("decided") and class agreement among them:**

| Rule | E1 cov90 (H+ / H−) | E1 IS90 | E1 width | E1 decided (agreement) | E2 cov90 (H+ / H−) | E2 IS90 | E2 width | E2 decided (agreement) |
|---|---|---|---|---|---|---|---|---|
| **GBM-LOCO** | **0.902** (0.910 / 0.750) | **3.22** | 1.57 | 0.92 (0.997) | **0.896** (0.900 / 0.833) | **6.67** | 2.94 | 0.87 (0.994) |
| GBM-LOCO, Mondrian | 0.905 (0.900 / 0.992) | 3.14 | 1.63 | 0.91 (0.997) | 0.895 (0.893 / 0.929) | 6.63 | 3.03 | 0.86 (0.995) |
| LIN-all | 0.896 (0.942 / 0.017) | 7.08 | 3.58 | 0.86 (0.970) | 0.893 (0.941 / 0.013) | 11.78 | 6.64 | 0.77 (0.969) |
| LIN-all, Mondrian | 0.900 (0.899 / 0.917) | 4.84 | 3.19 | 0.86 (0.989) | 0.899 (0.898 / 0.912) | 8.45 | 5.21 | 0.79 (0.992) |
| LIN-val (D26 rule) | 0.743 (0.782 / 0.000) | 8.48 | 1.51 | 0.94 (0.955) | 0.763 (0.804 / 0.000) | 13.53 | 3.07 | 0.88 (0.948) |
| *D26 shrunk residual TFT + EWC (validation rows; main / replicate)* | *0.661 / 0.655* | *6.13 / 6.28* | *0.99* | – | *0.682 / 0.671* | *9.96 / 10.35* | *2.20 / 2.13* | – |

At 80 %, pooled coverage is 0.802 (E1) and 0.801 (E2) for GBM-LOCO, 0.805 and 0.804 for LIN-all, and 0.609 and 0.639 for LIN-val.

**90 % coverage per fold, GBM-LOCO:**
- E1: 0.99, 0.95, 0.94, 0.99, 0.97, 0.93, **0.48** (case7).
- E2: 0.96, 0.98, 0.94, 0.99, 0.99, 0.94, **0.38** (case7).

LIN-all reaches 0.74 (E1) and 0.71 (E2) on case7, and 0.87–0.96 on the other folds. The IS90 of GBM-LOCO is 1.7–2.2 pp (E1) and 3.1–3.9 pp (E2) on the six other folds, against 11.6 and 26.6 pp on case7.

**Registered results:**

| Question | E1 | E2 | Verdict |
|---|---|---|---|
| Q1: GBM-LOCO pooled 90 % coverage ≥ 0.85 | 0.902 | 0.896 | **positive** |
| Q2: GBM-LOCO − LIN-all, IS90 (Wilcoxon over seven folds) | −3.87 pp, 6/7 folds, p = 0.047 | −5.11 pp, 6/7 folds, p = 0.30 | **not established** (E2 not significant; case7 reverses: +3.6 and +11.0 pp) |
| Q3 (descriptive): LIN-all against LIN-val, pooled 90 % coverage | 0.896 against 0.743 | 0.893 against 0.763 | the D26 under-coverage of extrapolation is attributed to calibrating on validation-row origins only (reading fixed in D27) |

**Reading.**
1. **Q1, positive.** Gradient-boosting intervals calibrated by leaving one condition out reach the nominal level when pooled over the seven unseen conditions: 0.90 at 90 % and 0.80 at 80 % in both experiments. This is the first interval rule in the study that holds on unseen conditions.
2. **The guarantee is pooled, not per condition.** On case7 (100 % depth of discharge, the only condition at that depth) the intervals cover 0.48 (E1) and 0.38 (E2), and its interval score is five to nine times that of any other fold. Conformal calibration over conditions can only be as good as the spread of the training conditions. A condition outside that spread is not covered, and the pooled number hides this.
3. **Q2, not established.** GBM-LOCO has a better interval score than calibrated extrapolation on six of seven folds in both experiments, by 3–10 pp. The registered test is significant in E1 only, because case7 reverses the sign and has the largest magnitude.
4. **Q3: which origins are used matters more than the model.** Calibrating extrapolation on every origin of the six training conditions instead of their mid-life validation rows raises its coverage from 0.74–0.76 to 0.89–0.90, with no fitted model involved. Under this reading, the D26 under-coverage of the TFT's intervals comes at least partly from the calibration rows (validation rows only) and not only from the model. The TFT was not run with LOCO calibration (D27 "Not done"), so this is not shown for the TFT itself.
5. **Strata (secondary).**
   - Without Mondrian half-widths, extrapolation covers almost none of the H− origins (0.01–0.02), where it is persistence. Per-stratum half-widths fix this (0.91–0.92) and lower its interval score by about 30 %.
   - Gradient boosting needs this much less: H− coverage is 0.75 (E1) and 0.83 (E2) without Mondrian half-widths, and 0.99 and 0.93 with them.
6. **Reuse decisions (descriptive).** With GBM-LOCO, 87–92 % of held-out samples have a 90 % interval inside one reuse class, and among those the class agrees with the measured one for 99.4–99.7 % of samples. For the remaining samples the interval flags that the class is uncertain. Calibrated extrapolation decides fewer samples (77–86 %) with lower agreement (0.97).
7. **Recommended practice from D26–D27** (post hoc, for the paper's discussion):
   - For a new usage condition, prefer a forecaster fitted on all available conditions.
   - Calibrate its intervals on conditions held out from fitting, using every origin, with half-widths per origin stratum.
   - Report coverage per condition, not only pooled.

## D28: continual calibration and refit on an unseen usage condition (21 Oct 2026)

Results files: `results/forecasting_d28_e1.json` (horizons 1–3) and `results/forecasting_d28_e2.json` (horizons 1–6), run at commit `161c01f` (clean tree, `git_dirty` false, no override) in the pinned environment (Python 3.11.15, scikit-learn 1.9.1). Protocol fixed in D28 before any code (`54bd012`). **Reproduction check passed:** the STATIC arm reproduces the D27 GBM-LOCO per-fold coverage, interval score and RMSE exactly (max abs diff 0). Setting as D27: leave one S6 test case out, seven folds; the held-out case arrives over time, and only targets already measured are used.

The arms are:
- **STATIC:** D27 GBM-LOCO.
- **TRACK:** quantile tracking of the half-width from the errors observed on the new condition.
- **REFIT:** gradient boosting refitted with the new condition's measured targets, static half-widths.
- **REFIT+TRACK (proposed):** both.

Deterministic forecasters: E1 and E2 replace the seed sets and are not independent. Step size c chosen on the training cases only: 0.1–0.2 on six folds, 0.02 (E1) and 0.01 (E2) on case7.

**90 % intervals, pooled coverage (H+ / H−), mean interval score (IS90, pp), mean width (pp), share decided (agreement), and mean RMSE over folds (pp):**

| Arm | E1 cov90 (H+ / H−) | E1 IS90 | E1 width | E1 decided (agr.) | E1 RMSE | E2 cov90 (H+ / H−) | E2 IS90 | E2 width | E2 decided (agr.) | E2 RMSE |
|---|---|---|---|---|---|---|---|---|---|---|
| STATIC (D27) | 0.902 (0.910 / 0.750) | 3.22 | 1.57 | 0.92 (0.997) | 0.535 | 0.896 (0.900 / 0.833) | 6.67 | 2.94 | 0.87 (0.994) | 1.051 |
| TRACK | 0.893 (0.901 / 0.750) | 2.93 | 1.42 | 0.92 (0.997) | 0.535 | 0.878 (0.881 / 0.833) | 6.27 | 2.63 | 0.88 (0.993) | 1.051 |
| REFIT | 0.935 (0.946 / 0.717) | 2.51 | 1.57 | 0.92 (0.996) | 0.458 | 0.928 (0.933 / 0.842) | 5.27 | 2.94 | 0.87 (0.992) | 0.898 |
| **REFIT+TRACK** | **0.914** (0.924 / 0.717) | **2.29** | **1.35** | 0.93 (0.996) | 0.458 | **0.902** (0.906 / 0.842) | **4.74** | **2.46** | 0.88 (0.992) | 0.898 |

At 80 %, pooled coverage of REFIT+TRACK is 0.809 (E1) and 0.797 (E2).

**Case7 (100 % depth of discharge), 90 %:** coverage STATIC 0.48 / 0.38 (E1 / E2), TRACK 0.58 / 0.44, REFIT 0.67 / 0.53, REFIT+TRACK 0.72 / 0.58; IS90 11.6 / 26.6 pp (STATIC) against 6.4 / 16.3 pp (REFIT+TRACK); RMSE 1.49 / 3.13 pp against 1.09 / 2.49 pp.

**Registered results:**

| Question | E1 | E2 | Verdict |
|---|---|---|---|
| Q1 (primary): REFIT+TRACK case7 coverage ≥ 0.80 | 0.72 | 0.58 | **negative** (improved from 0.48 / 0.38, threshold not reached) |
| Q2: pooled ≥ 0.85 and every condition ≥ 0.80 | pooled 0.914, min 0.72 (case7) | pooled 0.902, min 0.58 (case7) | **negative** (pooled met; case7 below 0.80; the other six folds 0.93–0.96) |
| Q3: REFIT − STATIC, RMSE (Wilcoxon over seven folds) | −0.077 pp, 4/7 folds, p = 0.22 | −0.153 pp, 6/7 folds, p = 0.078 | **not established** |
| Q4: REFIT+TRACK − STATIC, IS90 | −0.93 pp, 6/7 folds, p = 0.047 | −1.93 pp, 7/7 folds, p = 0.016 | **positive** |

**Reading.**
1. **Q4, positive.** Updating the forecaster and the interval as the new condition's checks arrive lowers the 90 % interval score by 29 % in both experiments (3.22 → 2.29 pp and 6.67 → 4.74 pp), with intervals 14–17 % narrower and pooled coverage kept at 0.90–0.91. This is the first positive registered result for a continual method in the study: what is updated continually is the refit with stored data and the calibration, not network weights under a no-storage constraint.
2. **Q1 and Q2, negative.** Case7 coverage rises from 0.48 to 0.72 (E1) and 0.38 to 0.58 (E2), the largest change of any fold, but stays below the 0.80 threshold; the selected step on case7 was small (0.01–0.02), because the training cases needed little adaptation. A condition outside the training range is improved, not fixed.
3. **Attribution (descriptive).** Refitting does most of the work: REFIT alone lowers IS90 to 2.51 / 5.27 pp and raises case7 coverage to 0.67 / 0.53. Tracking alone narrows the intervals but loses some pooled coverage (0.89 / 0.88). Combined, they give the lowest interval score and the narrowest intervals.
4. **Q3, not established.** Refitting lowers the mean RMSE (0.535 → 0.458 pp and 1.051 → 0.898 pp), mostly on case7 (−0.40 and −0.64 pp) and case6, but not consistently enough across folds for the registered test.
5. **Reuse decisions (descriptive).** 92–93 % (E1) and 87–88 % (E2) of held-out samples have a 90 % interval inside one reuse class, with 99.2–99.7 % agreement, as for STATIC, with narrower intervals.

**D28 figures** (`scripts/make_d28_figures.py`; the per-condition figure from the committed results files, the case7 timeline recomputed with `scripts/run_d28.py` and checked against them):

| Figure | Content |
|---|---|
| `d28_per_condition.png` | 90 % coverage and 90 % interval score (log scale) per held-out condition for STATIC, TRACK, REFIT and REFIT+TRACK, E1 and E2 |
| `d28_case7_timeline.png` | Case7 held out (E1, horizon 1): absolute error of the static and refitted forecasts and the 90 % half-widths of STATIC and REFIT+TRACK over the check dates |

## D29: physics-structured Bayesian forecaster with a learned residual on unseen conditions (21 Oct 2026)

Results files: `results/forecasting_d29_e1.json` (horizons 1–3) and `results/forecasting_d29_e2.json` (horizons 1–6), run at commit `f2fdd73` (clean checkout of that commit, `git_dirty` false, no override; the main working copy had notebook outputs from a local session, so the run used a separate worktree with the same data and environment). Protocol fixed in D29 before any code (`64f85fe`). **Reproduction check passed:** the recomputed D27 gradient boosting reproduces the D27 per-fold RMSE exactly (max abs diff 0). Setting as D27/D28. The fitted exponent z is 0.80–0.85 in every fold.

**90 % intervals, pooled coverage (H+ / H−), mean IS90 (pp), mean width (pp) and mean RMSE over folds (pp); comparators from the committed D27 and D28 files:**

| Arm | E1 cov90 (H+ / H−) | E1 IS90 | E1 width | E1 RMSE | E2 cov90 (H+ / H−) | E2 IS90 | E2 width | E2 RMSE |
|---|---|---|---|---|---|---|---|---|
| PHYS-BAYES (uncalibrated) | 0.992 (0.991 / 1.000) | 8.02 | 7.69 | 0.872 | 0.973 (0.972 / 1.000) | 9.27 | 7.74 | 1.391 |
| PHYS-LOCO | 0.892 (0.907 / 0.608) | 4.22 | 2.31 | 0.872 | 0.894 (0.900 / 0.779) | 6.82 | 3.60 | 1.391 |
| **HYBRID-LOCO** | 0.898 (0.904 / 0.775) | 3.24 | 1.83 | 0.581 | 0.905 (0.911 / 0.796) | 5.77 | 3.05 | 1.016 |
| PHYS-CYC-LOCO | 0.893 (0.905 / 0.667) | 4.34 | 2.45 | 0.902 | 0.889 (0.894 / 0.792) | 7.27 | 3.89 | 1.478 |
| *D27 GBM-LOCO* | *0.902* | *3.22* | *1.57* | *0.535* | *0.896* | *6.67* | *2.94* | *1.051* |
| *D28 REFIT+TRACK* | *0.914* | *2.29* | *1.35* | *0.458* | *0.902* | *4.74* | *2.46* | *0.898* |

**Case7, 90 %:** HYBRID-LOCO coverage 0.58 (E1) and 0.54 (E2); PHYS-BAYES 0.93 and 0.80 (at a mean width of about 7.7 pp over all folds). Case7 RMSE: HYBRID 1.38 / 2.63 pp, PHYS (throughput clock) 1.79 / 3.11, PHYS-CYC 1.85 / 3.25, D27 gradient boosting 1.49 / 3.13, D28 refit 1.08 / 2.49.

**Registered results:**

| Question | E1 | E2 | Verdict |
|---|---|---|---|
| Q1 (primary): HYBRID-LOCO case7 coverage ≥ 0.80 | 0.58 | 0.54 | **negative** |
| Q2: HYBRID-LOCO − D27 GBM, RMSE | +0.046 pp, 2/7 folds better, p = 0.30 | −0.035 pp, 3/7, p = 1.00 | **not established** |
| Q3: HYBRID-LOCO − D28 REFIT+TRACK, IS90 | +0.95 pp, 0/7 better, p = 0.016 | +1.03 pp, 1/7, p = 0.031 | **negative** (D28 better) |

**Reading.**
1. **The proposed physics-structured model does not improve on gradient boosting, and D28 is better than it** (Q2 not established, Q3 negative in both experiments). A single power law of throughput explains SOH far worse than gradient boosting (RMSE 0.87 against 0.54 pp in E1); the learned residual recovers most of the gap (0.58 pp) but not all.
2. **Case7 is not fixed (Q1 negative).** The throughput clock lowers case7 error only slightly relative to a cycle clock (1.79 against 1.85 pp, 3.11 against 3.25 pp, Q4): counting the extra charge of a 100 % cycle is not enough to describe the faster fade at 100 % DoD. The hybrid improves case7 error over D27 (1.38 against 1.49, 2.63 against 3.13 pp) but less than the D28 refit (1.08, 2.49).
3. **Uncalibrated Bayesian intervals cover case7 but are uninformative (Q5).** PHYS-BAYES is the only rule in D26–D29 that covers case7 at 0.80 or more (0.93, 0.80), but its intervals are wide everywhere (pooled coverage 0.97–0.99, width about 7.7 pp, IS90 8–9 pp, worse than every calibrated rule). The model's posterior uncertainty does not single out the out-of-range condition.
4. **Conclusion for the study:** of D23–D29, the best unseen-condition method remains gradient boosting with leave-one-condition-out intervals, refitted and tracked as the new condition's checks arrive (D28). A physical DoD effect would need prior information from data that vary DoD (not done; D29 "Not done").

## D30: full regression and classification metric set for P1/P2 (29 Oct 2026)

Results files: `results/forecasting_d30_s1.json` and `results/forecasting_d30_s6.json`, run at commit `cb43efb` (clean checkout, `git_dirty` false, no override; Windows 11, Python 3.11.15, torch 2.14.1+cpu, pinned requirements). **Post hoc and supplementary** (D30): the same P2 forecasts as Table P2 and the P1 baselines, scored with more metrics; no registered test or claim changes. Tables `results/tables/d30_regression.csv`, `d30_classification.csv`, `d30_per_group.csv`; paper tables `paper/tables/d30_regression.tex`, `d30_classification.tex`; figures `paper/figures/d30_confusion.png`, `d30_error_by_group.png` (`scripts/make_d30_outputs.py`). P2 predictions outside Git: `data/predictions/d30_<dataset>_p2.csv.gz`.

**Reproduction check (with the 29 Oct deviation, D30 run log).** Every deterministic baseline reproduces the committed P1 and P2 RMSE to 1e-6 pp on both datasets. The neural methods do not reproduce bit-for-bit across platforms: S1 TFT on w10 −0.009 pp, S6 MLP up to +0.027 pp (case5), S6 TFT case7 +0.0002 pp; all within the 0.05 pp neural tolerance. P2 means over groups: S1 TFT 0.2922 → 0.2907, S6 MLP 0.8205 → 0.8257 pp; no method changes rank. The MLP and TFT rows below therefore describe the re-run, not exactly the Table P2 run. Re-running S1 from an empty cache on this machine gives identical values, so training is deterministic per platform.

**P2, regression, mean over held-out groups (pp):**

| Data | Method | RMSE | MAE | MedAE | Bias | R²_Δ |
|---|---|---|---|---|---|---|
| S1 | Persistence | 0.545 | 0.470 | 0.473 | +0.430 | −1.85 |
| S1 | Linear extrapolation | 0.511 | 0.365 | 0.226 | −0.014 | −1.31 |
| S1 | Ridge, usage only | 0.301 | 0.206 | 0.133 | −0.038 | 0.18 |
| S1 | **Gradient boosting, F1** | **0.219** | **0.148** | **0.102** | −0.021 | **0.54** |
| S1 | MLP | 0.285 | 0.211 | 0.162 | +0.009 | 0.26 |
| S1 | TFT | 0.291 | 0.210 | 0.149 | +0.027 | 0.22 |
| S6 | Persistence | 2.817 | 2.451 | 2.225 | +2.451 | −3.16 |
| S6 | Linear extrapolation | 1.372 | 0.738 | 0.333 | +0.050 | −0.05 |
| S6 | Ridge, F0 | 0.859 | 0.613 | 0.439 | +0.028 | 0.61 |
| S6 | **Gradient boosting, F1** | **0.710** | **0.488** | **0.321** | +0.097 | **0.74** |
| S6 | MLP | 0.826 | 0.608 | 0.444 | +0.220 | 0.66 |
| S6 | TFT | 0.868 | 0.621 | 0.424 | +0.235 | 0.62 |

**P2, S6, reuse class (C0–C3) and end of first life (SOH ≤ 80), all held-out samples pooled, mean over seeds:**

| Method | Acc. | Bal. acc. | Macro F1 | MCC | κ_w | EoL F1 | EoL MCC | EoL AUC |
|---|---|---|---|---|---|---|---|---|
| Persistence | 0.858 | 0.782 | 0.661 | 0.723 | 0.81 | 0.962 | 0.894 | 0.998 |
| Linear extrapolation | 0.939 | 0.840 | 0.777 | 0.876 | 0.91 | 0.989 | 0.965 | 0.999 |
| Ridge, F0 | 0.963 | 0.879 | 0.881 | 0.922 | 0.94 | 0.990 | 0.968 | 0.999 |
| **Gradient boosting, F1** | **0.966** | 0.894 | **0.893** | **0.928** | 0.95 | 0.989 | 0.966 | 0.999 |
| MLP | 0.962 | 0.863 | 0.871 | 0.921 | 0.94 | 0.989 | 0.966 | 0.999 |
| TFT | 0.963 | 0.877 | 0.882 | 0.923 | 0.94 | 0.989 | 0.965 | 0.999 |

Measured class counts, S6: C0 52, C1 182, C2 516, C3 1659. S1: all 279 samples are C0, so S1 class metrics are degenerate (accuracy 0.98–1.00 for every method; specificity, MCC and AUC undefined) and are not interpreted.

**Reading.**
1. **The extra regression metrics agree with RMSE.** Gradient boosting is best on S1 and S6 in RMSE, MAE, MedAE and R²_Δ; MLP and TFT come next; persistence and linear extrapolation are last. R²_Δ shows how much a forecaster adds to persistence: linear extrapolation adds nothing on the unseen battery (S1, −1.31) or condition (S6, −0.05).
2. **The neural models overestimate SOH on S6** (bias +0.22 and +0.24 pp against +0.10 for gradient boosting), almost all of it on case7 (TFT bias about +1.6 pp; `d30_error_by_group.png`), as in D26–D29.
3. **Reuse classes, S6:** gradient boosting, ridge F0, MLP and TFT are within 0.004 in accuracy and 0.007 in MCC; all clearly beat linear extrapolation (MCC 0.876) and persistence (0.723). The errors are almost all between neighbouring classes (`d30_confusion.png`); the hardest class is C1 (182 samples).
4. **End of first life does not separate the methods:** AUC 0.998–0.999 and F1 0.96–0.99 for every method, including persistence. Most S6 samples are far from 80 % SOH (69 % of them already in C3), so this decision is easy here; it is not evidence for any forecaster.
5. **P1 baselines** (per-task values in the results file) reproduce the committed P1 RMSE exactly; their class metrics are single-class per task and are not interpreted.
