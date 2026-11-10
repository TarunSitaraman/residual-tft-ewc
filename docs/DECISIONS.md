# Design decisions

Methodological decisions for the data pipeline and evaluation, with the reasoning behind each. Each entry states where the decision is implemented so results in the paper can be traced to code.

Status values: **Adopted** (in use), **Planned** (agreed, not yet implemented).

The data handover is final (28 Sep 2026): the data owner has provided every available file, and no further source information will arrive. Decisions that previously awaited the data owner are therefore taken by the team; where the source cannot settle a question, the paper reports a sensitivity analysis instead.

---

## D1. Model input features: 18, not 22

**Status:** Adopted (team decision; handover final)

`DATA_DICTIONARY.csv` documents **27** columns (the README previously said 28). Excluding identifiers and RPT/health columns leaves **18** model inputs:

| Group | Columns |
|---|---|
| Cycle count | `Cycle_Index` |
| Voltage | `Voltage_Mean`, `Voltage_Min`, `Voltage_Max` |
| Current | `Current_Mean`, `Current_Min`, `Current_Max` |
| Temperature | `Temperature_Mean`, `Temperature_Min`, `Temperature_Max` |
| Capacity | `Charge_Capacity_Max`, `Discharge_Capacity_Max` |
| Energy | `Charge_Energy_Max`, `Discharge_Energy_Max` |
| Power | `Power_Mean` |
| Ratios | `Capacity_Ratio_Mean`, `Energy_Efficiency_Mean` |
| Data volume | `Data_Points` |

Excluded:

- **Identifiers / time:** `Battery_ID`, `Cycling_Test`, `Cycle_Date` (used for grouping, ordering and splitting).
- **RPT / health columns:** `RPT`, `RPT_Number`, `Capacity_Ah`, `Reference_Capacity_Ah`, `SOH_Percent`, `Days_Since_RPT`. These are the target or derived from the RPT join, so using them as inputs would leak the target.

The earlier "22 features" wording in the handover notes is outdated or used a different counting convention; no source for 22 exists in the delivered files. The code never hard-codes the count: notebook 03 derives it from the dictionary and the observed files.

**Implemented in:** `src/data_pipeline.py` (`HEALTH_COLUMNS`, `ID_COLUMNS`, `identify_cycling_features`, `model_feature_columns`), `notebooks/03_feature_engineering.ipynb` §3.

## D2. Point-level data is profiled only

**Status:** Adopted

`cleaned_battery_data.csv` is point-level and is not described by the data dictionary. The modelling datasets are already aggregated per cycle, so the point-level file is only profiled (in chunks) for size, coverage and missingness.

It must not be used directly for modelling until a separate point-level feature-extraction pipeline is defined, documented and validated against `final_battery_cycling_features.csv`.

**Implemented in:** `src/data_pipeline.py` (`LARGE_FILES`, `profile_csv_chunked`), `notebooks/01_data_ingestion.ipynb` §4.

## D3. Same-day RPTs recorded as a date only

**Status:** Adopted — `"start"` as primary convention; sensitivity analysis under `"end"` planned

Each cycle is aligned to the most recent RPT **on or before** its `Cycle_Date` (backward as-of join per battery). When `Test_Date` has no time of day, pandas reads it as 00:00. What a date-only RPT means physically is not documented; it could be:

| Setting (`date_only_rpt`) | Meaning | Effect on a cycle later the same day |
|---|---|---|
| `"start"` (default) | RPT applies from the start of its day (timestamps as stored) | Uses that RPT |
| `"end"` | RPT applies from the end of its day | Uses the previous RPT |
| `"unknown"` | Time of the RPT is unknown | Flagged as ambiguous; not judged either way in validation |

RPTs that carry a time of day are never shifted. Only cycles on the calendar day of a date-only RPT are affected.

The default `"start"` matches the handover rule applied to timestamps as stored, and it is the convention the delivered aligned datasets were built with. Notebook 03 §6.1 counts the affected cycles and confirms which conventions the stored datasets are consistent with.

The handover is final and the RPT time of day is not recorded anywhere, so the physically correct convention cannot be confirmed. The paper therefore:

1. uses `"start"` for all primary results;
2. reports the share of cycles affected (cycles on the day of a date-only RPT);
3. re-labels those cycles under `"end"` with `align_to_rpt` and re-reports the headline metrics, so reviewers can see the conclusion does not depend on this convention.

Item 3 is **planned** (not yet implemented in notebook 04).

**Implemented in:** `src/data_pipeline.py` (`effective_rpt_time`, `same_day_as_date_only_rpt`, `align_to_rpt`, `validate_rpt_alignment`, `validate_training_subset`), `notebooks/03_feature_engineering.ipynb` §1 (`DATE_ONLY_RPT`) and §6.1.

## D4. SOH labels repeat between RPTs; persistence is a reference only

**Status:** Adopted

SOH is measured only at RPTs, so every cycle between two RPTs carries the same label. Two consequences:

1. **Persistence is never selected as the best model.** Predicting each battery's last training SOH is a useful reference, but it can score zero validation error when the validation window sits inside the last training RPT interval. That score says nothing about later life. Model selection considers learned models only.
2. **RPT overlap is reported.** For each battery, notebook 04 reports the share of test rows whose RPT interval was already seen in training (`test_rows_in_train_rpt_%`) and the number of distinct RPTs in the test period (`test_distinct_rpts`). Scores on overlapping rows partly measure interpolation within a known health state.

A stricter follow-up is to split on RPT boundaries or to evaluate on RPT-level targets.

**Implemented in:** `notebooks/04_baseline_model.ipynb` §3 and §6.

## D5. Evaluation: within-battery baseline, sequential groups for continual learning

**Status:** Baseline adopted; continual-learning protocol planned

The baseline splits each battery chronologically (70 % train / 15 % validation / 15 % test, no shuffling). Every battery appears in training, so it measures **later-life prediction**, not generalisation to an unseen battery group.

The continual-learning stage will introduce battery groups sequentially and, after each new group, report:

- **Adaptation:** performance on the newly introduced group.
- **Retention:** performance on every previously seen group.
- **Forgetting:** the drop on each earlier group relative to its score at earlier checkpoints.

All metrics are reported per battery group (RMSE, MAE, MAPE, R²).

**Implemented in:** `src/data_pipeline.py` (`temporal_split`, `check_split_ordering`, `metrics_by_group`), `notebooks/04_baseline_model.ipynb`. Continual-learning protocol: to be implemented in `src/continual_learning.py`.

## D6. Data location and execution

**Status:** Adopted

Datasets are never stored in Git. Notebooks read from `DATA_ROOT` (default `data/`, relative to the repository root) and stop with an explicit error listing any missing file. No substitute data is generated.

In the cloud runtime all seven CSVs, including `DATA_DICTIONARY.csv`, are mounted under one directory, for example:

```bash
export DATA_ROOT=/mnt/project-data
jupyter nbconvert --to notebook --execute notebooks/01_data_ingestion.ipynb --output-dir /tmp/executed
```

Executed notebooks go to an output directory so data-derived outputs are not committed. Only small `results/*.json` files (metrics, configuration, preprocessing table) may be versioned.

**Implemented in:** `src/data_pipeline.py` (`resolve_data_root`, `check_data_files`), `data/README.md`, `.gitignore`.

## D7. Revised handover file `ml_ready_dataset.csv` is audited, not adopted

**Status:** Adopted

The final handover package replaces `ml_training_dataset.csv` with `ml_ready_dataset.csv`: the same cycles plus `Test_Date` and `Temperature_{Mean,Min,Max}_Missing` flags, with missing temperatures pre-imputed using battery-level medians (overall median as fallback). The other five shared datasets are byte-identical to the previous package (`docs/REVISED_HANDOVER_COMPARISON.md`).

`ml_training_dataset.csv` remains the canonical supervised input, because:

- a battery-level median computed over a battery's whole history uses its later-life (validation/test) rows, which leaks future information into a chronological evaluation;
- the baseline fits imputation on the training split only, which the pre-imputed file cannot reproduce.

The revised file is useful in two ways: its missingness flags are exactly the explicit masks the proposed model needs, and it can be checked against the canonical file. Notebook 03 §9 audits it when present: identical rows and shared values, flags matching the canonical missing temperatures, and whether imputed values equal full-history battery medians. If a future experiment uses it, the imputed temperatures must be reset to missing (using the flags) and re-imputed from training rows only.

**Implemented in:** `src/data_pipeline.py` (`REVISED_TRAINING_FILE`, `audit_ml_ready`), `notebooks/03_feature_engineering.ipynb` §9.

## D8. Metrics: RMSE and MAE primary; R² only on pooled results

**Status:** Adopted

SOH is constant between RPTs, so one battery's validation or test period often spans only one or two SOH values. Per-battery R² is then undefined (constant target) or strongly negative even when the absolute error is small, which would misrepresent model quality. Therefore:

- **Primary metrics:** RMSE and MAE in SOH percentage points, per battery group and pooled.
- **MAPE:** reported for comparability with the literature; it is nearly proportional to MAE here because SOH stays close to 100 %.
- **R²:** reported and interpreted only for pooled results; per-group R² is shown for completeness but not used for conclusions or targets.

Literature error ranges (e.g. RMSE 0.6–1.5 %) come from other datasets and SOH definitions and are context, not targets (see README, Section 9).

**Implemented in:** `src/data_pipeline.py` (`regression_metrics`, `metrics_by_group`), `notebooks/04_baseline_model.ipynb` §6.

## D9. Every reported number is traceable to a commit

**Status:** Adopted

Result files record the configuration, random seed, feature list, data fingerprint (rows per group, date range, split sizes), library versions and the Git commit, together with whether the working tree had uncommitted changes (`git_dirty`). Numbers used in the paper must come from a run with `git_dirty: false`. Datasets are fingerprinted, not stored.

**Implemented in:** `src/data_pipeline.py` (`environment_info`), `notebooks/04_baseline_model.ipynb` §7 (`results/baseline_metrics.json`).

## D10. Reuse categories: four relative-fade classes

**Status:** Adopted (team decision, 28 Sep 2026; fixed before any real-data run)

Reuse categories are defined on fade since the first observed RPT, `F = 100 − SOH_Percent`, with neutral names:

| Class | Rule | Fade since first RPT |
|---|---|---|
| C0 stable | SOH > 95 (includes apparent recovery above 100) | < 5 pp |
| C1 moderate fade | 90 < SOH ≤ 95 | 5–10 pp |
| C2 substantial fade | 80 < SOH ≤ 90 | 10–20 pp |
| C3 severe fade | SOH ≤ 80 | ≥ 20 pp |

The earlier 80/60/40 % bands with application labels ("like-new", "moderate", "limited", "recycle") are withdrawn: they assume SOH relative to nominal capacity, whereas SOH here is relative to the first second-life measurement (see `docs/REUSE_CATEGORIES.md` §1).

- Classes describe capacity change, not suitability for an application. Application labels are allowed only in sensitivity analysis S-b, and only if the data provenance (Stanford second-life dataset) is confirmed.
- Cut-points are conventions, not tuned on the data. The 20 pp boundary is an analogy to the common "80 % of the starting value" convention, not an equivalence with first-life end of life.
- The same rule is applied to measured and predicted SOH; the primary E10 result thresholds the regression output rather than training a separate classifier.
- Pre-registered sensitivity analyses: cut-points shifted by ±2.5 pp (S-a1), a 3-class rule at 10/20 pp (S-a2), absolute-SOH classes if provenance is confirmed (S-b), fade rate per 100 days (S-c), and the `"end"` date-only convention (S-D3). None is used for selection.
- The evaluation protocol (RPT-interval and cycle units, accuracy, macro-F1 with the number of classes present, linear-weighted kappa, confusion matrix, boundary share, majority and persistence baselines, per-step category matrix) is specified in `docs/REUSE_CATEGORIES.md` §4.

**Implemented in:** `src/evaluation.py` (`REUSE_CUTPOINTS`, `REUSE_CLASS_NAMES`, `reuse_class`); E10 evaluation in `src/reuse.py` (interval and cycle units, class metrics, baselines, per-step category matrix, cut-point variants), `src/experiments.py` (`reuse_*` metrics for every continual-learning run), `notebooks/08_reuse_assessment.ipynb` (Stage A models, baselines, sensitivity analyses) and notebooks 06–07 (category results after each step).

**Sample size.** Each group's test period is its last 15 % of cycles, which typically spans only one or two RPT intervals, so the interval-level headline pools few samples (12 on the synthetic test data). Interval-level category results must be reported with their sample counts and interpreted as indicative; cycle-level results have more samples but are strongly correlated within an interval.

## D11. Hyperparameter tuning uses validation data only

**Status:** Adopted (fixed before any real-data run)

Continual-learning results are sensitive to each method's own hyperparameter, so these are selected by a fixed protocol rather than set by hand or chosen on test data:

Tuning has two stages:

1. **Backbone learning rate**, selected separately for the MLP and the TFT from the same grid {3·10⁻⁴, 10⁻³, 3·10⁻³, 10⁻²} (equal budget), using plain fine-tuning, by **validation plasticity** (error on each newly learned task). The two backbones need different learning rates (on synthetic checks the TFT trained much better at 10⁻² than at the MLP's 10⁻³), so one shared rate would handicap one of them. Plasticity is the criterion because the learning rate governs optimisation; using final error would reward under-training, which minimises forgetting. The selected rate applies to every method and to joint training on that backbone.
2. **Each method's own hyperparameter**, with the stage-1 learning rate: replay buffer size per task {50, 100, 200, 400}, EWC λ {1, 10, 10², 10³, 10⁴, 10⁵} for both MLP + EWC and TFT + EWC, LwF α {0.1, 0.5, 1, 2, 5}, by final average validation error.

- **Fixed, not tuned:** epochs, patience, batch size, network width and TFT window length. Grids are fixed in `src/experiments.py` (`DEFAULT_GRIDS`, `DEFAULT_BACKBONE_GRID`) before any real-data run.
- **Data:** streams are evaluated on the tasks' validation periods only; test periods are never read during tuning (unit-tested).
- **Orders:** tuning uses group orders guaranteed disjoint from the evaluation orders, so the selection is not fitted to the reported streams.
- **Criteria:** stage 1 as above; stage 2 final average validation error over all tasks after the last step, which reflects adaptation and retention together. Lowest wins; ties go to the earlier grid value.
- **Grid edges:** if a selected value lies at the edge of its grid, the grid is extended and tuning repeated before results are reported, so the optimum is not an artefact of the range.
- **Reporting:** the full tuning grid (every value, validation error, plasticity, forgetting) is saved with the results, so sensitivity to the hyperparameter is visible.

The proposed TFT + EWC model (E9) uses the same protocol (TFT learning rate in stage 1, λ in stage 2).

**Implemented in:** `src/experiments.py` (`DEFAULT_GRIDS`, `DEFAULT_BACKBONE_GRID`, `TuningConfig`, `tuning_orders`, `tune_hyperparameters`, `method_settings`, `run_stream(eval_split="val")`), notebooks 06 and 07 §3.

## D12. Proposed model: TFT + EWC, implemented for sequence-to-one estimation

**Status:** Adopted

The proposed model is a Temporal Fusion Transformer (Lim et al., *International Journal of Forecasting* 37(4), 2021) with Elastic Weight Consolidation. It is implemented directly in PyTorch (`src/models/tft_continual.py`) rather than through `pytorch-forecasting`, because:

- **The task is estimation, not forecasting.** Library TFTs take the target's own history as an input. Here past SOH labels must never be inputs: they are RPT step values (D4) and would leak the target. The model estimates the SOH of a cycle from a causal window of that battery's recent cycles' features.
- **One continual-learning code path.** The TFT plugs into the same strategy classes as the MLP (`TFTFineTune`, `TFTEWC`, `TFTJoint`), so the EWC Fisher, early stopping, frozen first-task preprocessing and target scaling are identical for both backbones.
- **Control and reproducibility.** Deterministic training, exact parameter counts, and access to interpretation outputs.

Architecture and inputs:

- TFT components: gated residual networks, per-variable embeddings and variable selection, LSTM encoder with gated skip connections, static-enrichment GRN, interpretable multi-head attention (shared values, head-averaged), position-wise GRN.
- Each variable is embedded from its standardised value and its missing indicator (D7), so imputed values are always marked.
- Attention is causal and ignores padded steps. The window builder only uses the same battery's current and earlier cycles, and label/RPT columns are removed from the window history (unit-tested).
- **No static covariates:** battery identity is not an input, so every method is task-agnostic.
- **Capacity-matched to the MLP:** `d_model = 8`, 2 heads gives about 12.0k parameters against the MLP's 13.1k, so differences reflect the architecture rather than model size.
- **Window:** 8 cycles, fixed a priori; windows of 4 and 16 are reported as a sensitivity analysis.
- The learning rate is tuned per backbone and λ per method under D11. In Stage A (notebooks 05 and 08) the MLP and TFT select their learning rate on validation from the same grid, as Ridge selects its alpha.

Evaluation follows `docs/RESEARCH_PLAN.md` Stage D: TFT + EWC against MLP + EWC (architecture), TFT fine-tuning (EWC), MLP replay and LwF (other families), and joint training on both backbones (upper bounds), on identical orders and seeds. Outcomes are reported as measured.

`pytorch-forecasting` and `pytorch-lightning` were removed from `requirements.txt` because nothing uses them.

**Implemented in:** `src/models/tft_continual.py` (`TemporalFusionTransformer`, `CycleWindows`), `src/continual_learning.py` (`TFTMixin`, `TFTFineTune`, `TFTEWC`, `TFTJoint`), `src/experiments.py`, `notebooks/07_tft_ewc.ipynb`, `tests/test_tft.py`.

## D13. Missing-temperature robustness is tested by removal, never by adding data

**Status:** Adopted

Temperature is missing for some cycles in the handover data (reported in notebook 02, never fabricated). E11 asks whether this matters and whether the handling matters:

- **Natural missingness:** test errors on cycles whose temperature is missing vs. complete cycles, per model and group.
- **Handling variants**, each fitted with the same validation-only selection (D11):
  - `impute_indicator`: training median plus missing indicator (the default everywhere else);
  - `impute_only`: training median with the indicators hidden (zeroed, so input width and model size stay identical);
  - `carry_forward`: the battery's most recent *earlier* temperature (causal; leading gaps keep default handling);
  - `drop_temperature`: temperature features removed.
  
  Differences to the default are paired over battery groups with Holm correction per model.
- **Stress test:** additional temperature values are *removed* from the test periods at increasing rates, either at random or as one contiguous outage per battery, with fixed seeds; all three temperature columns are removed together, as in the observed data. Models are not refitted. For the TFT, removal is applied to the cycle history its windows are read from, so it reaches every step of the window (unit-tested).

Values are only ever removed; carry-forward only copies a battery's own earlier observation and is applied identically to all splits, so no future or test information reaches training. Removal patterns are assumptions (random, block) and may differ from the real mechanism; notebook 02 reports the observed run lengths for comparison.

**Implemented in:** `src/robustness.py` (`HANDLING_VARIANTS`, `prepare_variant`, `carry_forward_temperature`, `mask_temperature`, `rows_in`, `stratified_errors`), `src/models/baseline_dnn.py` (`TabularPreprocessor(indicators=...)`), `src/continual_learning.py` (`missing_indicators`, `TFTMixin.set_history`), `src/experiments.py` (`fit_model`), `notebooks/09_missing_temperature.ipynb`, `tests/test_robustness.py`.

## D14. Protocol-scaled quantities are excluded; health indicators come from the monthly standardised segment

**Status:** Adopted (after the first real-data run)

The first real-data run showed:

- **The handover's cycle-level rows are segments of monthly cycling blocks.** Their capacity and energy columns are running totals within a block, and their 18 inputs barely track SOH within a battery: median |ρ| ≈ 0.09, against −0.99 for elapsed time. The notebook 04 baseline is worse than persistence.
- **The protocol scaled the discharge current to the last measured capacity.** Every monthly block starts with a standardised segment (CC charge at 1.46 A, CV charge, rest, 30-minute discharge, rest). Its discharge current equals 0.9966 × the capacity at the previous RPT for every battery (coefficient of variation 0.14 %), i.e. 1C of the last measured capacity. On the C/20 capacity basis (D17) the factor is 0.9998.

Rules:

- The segment's discharge current, discharged Ah and discharge energy (`health_indicators.PROTOCOL_SCALED`) encode the label and are never used as inputs. The handover's `Current_Min` and discharge-capacity columns partly carry the same scaling and must be treated the same way in any new model.
- Health indicators are extracted once per block from that segment and the duty cycle (`health_indicators.INDICATORS`, plus partial-charge ΔQ in fixed voltage windows). They are aligned to the RPTs bracketing the block: the previous RPT is known at block start; the closing RPT is the target.
- Blocks with a non-standard protocol (4 blocks charged at 1.05 A) are flagged and excluded from indicator models, never silently dropped.

On this dataset the indicators track SOH in the physically expected direction but do not beat elapsed time or persistence on held-out batteries (`docs/DATA_REPORT.md` §4.3). Estimating the current SOH is therefore not a meaningful benchmark here, and the task definition is reopened (report §5).

**Implemented in:** `src/health_indicators.py`, `notebooks/10_health_indicators.ipynb`, `tests/test_health_indicators.py`; schema handling in `src/data_pipeline.py` (`normalise_rpt_capacity`, `Date_Time`) and notebooks 01–03.

---

## D15. The team owns the data pipeline from the raw archive

**Status:** Adopted (29 Sep 2026)

- **Decision.** The handover is complete, and no further processing will come from outside the team. From now on, all processed data are produced by code in this repository from the public raw archive (S1 in `docs/EXTERNAL_DATA.md`).
- **Existing CSVs.** The handover CSVs are kept as frozen, verified inputs:
  - all 112 RPT capacity rows (111 RPTs; g1's RPT 1 spans two files) are reproduced exactly from the raw files;
  - every RPT file and cycling block matches the archive.
- **New inputs.** Anything the CSVs lack is extracted by committed scripts with checksums recorded: EIS spectra, raw RPT curves, and re-derived cycling features if needed.
- **Why.** Every number in the paper must trace to code in this repository and to a public, citable source. Undocumented processing outside the repository cannot be reviewed.
- **Implemented in:**
  - `scripts/fetch_osf_diagnostics.py` (raw RPT and EIS files);
  - `scripts/verify_external_data.py` (handover checked against the archive);
  - extraction code for new inputs, to follow.

---

## D16. Task reframed as SOH forecasting; health features come from the raw RPT and EIS files

**Status:** Adopted (29 Sep 2026). This decision was taken **after seeing the data** and must be reported as such in the paper.

**Decision: the task.** The primary task becomes **forecasting** the SOH at later RPTs, instead of estimating the current SOH from cycle-level features.
- Why: on this data, current-SOH estimation is beaten by persistence (`docs/DATA_REPORT.md` §4).
- Forecast inputs: the SOH history, the diagnostic features below, and usage (duty cycle, temperature).
- Baselines to beat: persistence and linear extrapolation.
- The horizon and the evaluation protocol are fixed before any model is tuned, in a follow-up entry.

**Decision: the features.** One row per (battery, RPT), every value measured at that RPT (`src/diagnostic_features.py`; build with `scripts/build_diagnostic_features.py`; analysis in notebook 11).
- Capacity test (C/20 discharge with C/2 pulses at 4.00, 3.63 and 3.26 V):
  - charge in each window between the pulse voltages;
  - total C/20 charge (`q_c20_ah`);
  - pulse resistance after 0.1, 1 and 8 s;
  - positions and heights of the incremental-capacity (dQ/dV) peaks in five fixed windows.
- EIS (RPT 5 on, same three voltages):
  - R0 (Z′ where Z″ crosses zero);
  - arc resistance;
  - arc peak;
  - Z′ at 1 kHz, 10 Hz, 1 Hz, 0.1 Hz and 0.01 Hz.
- No protocol current is used as a feature, because the currents are set from the previous capacity (see D14).

**Fixed selection rules.** These are applied by code and every application is logged (`data/diagnostic_selection_log.csv`):
- **Capacity test:** use the complete test (four C/20 segments), then the longest file. This resolves g1 RPT 1's partial second export.
- **EIS:** use non-empty, potentiostatic spectra only. Prefer the file on the same channel as the cell's capacity test in that RPT, then the one closest in time. This resolves RPT 9's files on another cell's channel and RPT 17's two EIS sessions.
- **Galvanostatic EIS** (`_gal`, w8 RPT 17 and 19) is excluded.
- **RPT 7 EIS** comes from the aggregate `_ALL` files. Their format was checked against RPT 5, 6 and 8 (identical to 5 µΩ, the files' rounding). Every sweep uses the same 43-point frequency grid. The resulting R0 equals the authors' own `R_HF` in `eis_7.mat` for all 18 values.
- **Off-protocol pulses** have their resistance set to missing: RPT 1 at 4.00 V (0.525 A), and v4 and w10 in RPT 17–19 (1.05 A instead of C/2).
- **IC curve:** built from the C/20 segments only. After each pulse, the samples until the voltage returns below its pre-pulse level (5–9 mAh) are dropped, so the pulses do not create peaks at the pulse voltages.

**Data-quality findings recorded, not repaired.**
- **RPT 2 was run at about 20 °C instead of 25 °C.** The article's Table 3 notes an incorrect chamber setpoint. Capacity and resistance at RPT 2 are not comparable with the other RPTs.
- **RPT 1 has no temperature logging.**
- **The handover's SOH labels depend on the pulse protocol.** Resolved by D17.

**Implemented in:**
- `src/raw_diagnostics.py`, `src/diagnostic_features.py`;
- `scripts/fetch_osf_diagnostics.py`, `scripts/build_diagnostic_features.py`;
- `notebooks/11_diagnostic_features.ipynb`;
- `tests/test_diagnostic_features.py`.

---

## D17. SOH labels use the C/20 capacity

**Status:** Adopted (29 Sep 2026), after seeing the data. Report it as such.

**Decision.** For every RPT, `Capacity_Ah` is the charge discharged in the C/20 steps of the capacity test (`q_c20_ah`), excluding the C/2 pulses. This is the definition of Moy et al. 2024, Eq. (1). The SOH formula is unchanged:
- `SOH_Percent = Capacity_Ah / Reference_Capacity_Ah * 100`;
- the reference is the battery's first RPT on the same basis.

**Why.** The handover's `Capacity_Ah` is the total charge discharged during the RPT, including the pulses, and the pulse contribution changed with the protocol:
- **RPT 1:** the 4.00 V pulse ran at 0.525 A for g1, v4, v5 and w10. Their reference therefore contains about 4 mAh less pulse charge, and every later label of those four cells is 0.08 pp too high.
- **v4 and w10 in RPT 17–19:** the 1.05 A pulses contribute about 7 mAh instead of 15, which moves those labels by another 0.17 pp.

These offsets are comparable to one RPT's fade (0.1–0.5 pp). On the C/20 basis they vanish. The largest difference between the two label sets is 0.089 pp.

**Scope.**
- **Uses the new labels:** notebooks 10 and 11, and all forecasting work (D16). They read `data/rpt_health_c20.csv`, written by `scripts/build_diagnostic_features.py`, which checks the SOH formula.
- **Superseded:** notebooks 01–09 (the current-SOH estimation task) keep the handover labels in `ml_training_dataset.csv`. Their results are not updated and are not reported as main results.

**Still recorded, not repaired:** RPT 2 was run at about 20 °C (D16). Its C/20 capacity is therefore not comparable with the 25 °C RPTs, and the forecasting protocol must state how RPT 2 is treated.

**Implemented in:** `src/diagnostic_features.py` (`add_c20_soh`, `health_table`), `scripts/build_diagnostic_features.py`, `tests/test_diagnostic_features.py`.


---

## D18. Forecasting protocol (fixed before any forecasting model is trained)

**Status:** Adopted (29 Sep 2026). It was committed before any forecasting model was fitted or evaluated. Any later change must be recorded here as a deviation, with its date and reason.

### Samples
- **Unit.** One sample per (battery, origin RPT *k*, horizon *h*), for *h* ∈ {1, 2, 3} RPTs ahead. RPTs are counted after the exclusions below.
- **Target.** The SOH change ΔSOH = SOH(*k*+*h*) − SOH(*k*) in pp, on the C/20 basis (D17). An error on ΔSOH equals the error on SOH(*k*+*h*), because SOH(*k*) is known at the origin. All errors are reported in SOH pp.
- **Horizons pooled.** One model serves all three horizons, with *h* as an input. Results are reported per horizon and pooled.
- **Excluded RPTs.** RPT 2 (run at about 20 °C, D16) is excluded as origin, as target and from the SOH history. It stays in the data with a flag.

### Inputs (known at the forecast origin)
- **SOH history:** SOH(*k*); days since the first RPT; the fade rate (least-squares slope of SOH against days) over the last three valid RPTs up to *k*, missing when fewer than three exist.
- **Diagnostics at *k*:** the 47 features of D16. EIS features are missing before RPT 5; missing values are imputed from training data with missing-value indicators.
- **Static:** second-life duty cycle (residential or commercial), and from Moy et al. 2024, Table 2:
  - first-life charge C-rate;
  - first-life cycle count;
  - capacity at end of first life.
- **Known-future schedule** for the interval from RPT *k* to RPT *k*+*h*:
  - days until the target RPT;
  - days of cycling at the 35 °C setpoint;
  - days of cycling at the 20 °C setpoint;
  - days without cycling.

  Setpoints are the nearest of 20/35 °C to the median cell temperature of each cycling block. `Cycling_1` has no temperature log and is 35 °C per the article's Table 3. These are planned test conditions, so the forecasts are conditional on the planned usage; the paper must say so.
- **Never used:** protocol currents (D14, D16), throughput (it is scaled from later capacities), and any label of an RPT after *k*.

### Two feature sets (pre-registered comparison)
- **F0:** SOH history + static + schedule.
- **F1:** F0 + diagnostics.

F1 − F0 answers whether the diagnostics help forecasting.

### Splits (per battery, purged; nothing shuffled)
- Each battery's valid RPTs are ordered in time. Cut A is at 60 % of them and cut B at 80 % (rounded down, at least one RPT apart).
- **Train:** targets ≤ A.
- **Validation:** origins ≥ A and targets ≤ B.
- **Test:** origins ≥ B.
- No training label lies after a validation origin, and no validation label lies after a test origin. The inputs at a test origin may use every RPT up to that origin, as a real forecaster would.

### Protocols
- **P1, continual stream (headline).** Batteries arrive one at a time as tasks.
  - 10 orders × 5 seeds (orders from `sample_orders`, seed 0).
  - After each task, every task's test set is evaluated. This gives the accuracy matrix, final average error, plasticity, BWT, forgetting and FWT, as in the existing framework.
- **P2, unseen battery.** Leave one battery out: train on all samples of the other five, test on all samples of the held-out battery. This is the "new battery arrives with its own history" case.

### Methods
- **Baselines**, fitted per protocol on the same training data:
  - persistence (ΔSOH = 0);
  - linear extrapolation (fade rate × days to target; 0 when the rate is missing);
  - schedule-only ridge (fade explained by days at 35 °C, 20 °C and rest);
  - ridge on F0 and F1;
  - gradient boosting on F1.
- **Continual:** MLP backbone with fine-tuning, replay and EWC; TFT backbone with fine-tuning and **TFT + EWC** (proposed); joint training for both backbones as upper bounds.
- **TFT window:** the four previous origins of the same battery and horizon, feature vectors only (never labels).
- **Tuning:** the existing two-stage, validation-only tuning (D11), same grids, on tuning orders disjoint from the evaluation orders. Stage 1 sets the learning rate per backbone; stage 2 sets the method hyperparameters.

### Metrics and tests
- **Primary:** RMSE of SOH(*k*+*h*) in pp; MAE secondary. Reported per horizon, per battery and pooled, always alongside persistence and linear extrapolation.
- **Skill** relative to persistence: 1 − RMSE_model / RMSE_persistence.
- **Paired comparisons** (P1: matched order × seed; P2: matched battery): Wilcoxon signed-rank with Holm correction over the fixed family (each method vs. TFT + EWC, and F1 vs. F0), rank-biserial effect sizes, and bootstrap 95 % CIs.

### Limitations stated in advance
- **Sample size:** about 6 test samples per battery per protocol step (P1), so confidence intervals will be wide.
- **Scope:** results are for one cell type and one laboratory.
- **Forecast type:** forecasts are conditional on the planned schedule.

### D18 deviations and disclosures
- **29 Sep 2026, code smoke tests.** While the runners were being written, each method, baseline and protocol was run once on one order with seed 0 and untuned defaults (learning rate 1e-3), and the test-split errors were printed. This checked that the code ran end to end and how long a run takes. No protocol element, grid, feature set or setting was changed afterwards. The reported results come only from the full tuned run in notebook 12.
- **Replay grid.** The D11 replay grid (50–400 samples per task) is larger than a forecasting task's training set (21–27 samples). Every grid value therefore stores the whole task, and replay becomes full rehearsal. This follows from reusing the D11 grids unchanged and is reported, not re-tuned.
- **P2 neural training.** "Train on all samples of the other five" is implemented as fitting on their train and test splits and early-stopping on their validation splits. Baselines, which have no early stopping, fit on all of them.

---

## D19. Second dataset: Wheeler et al. 2025 (LFP, first and second life)

**Status:** Adopted (29 Sep 2026). The dataset is registered as S6 in `docs/EXTERNAL_DATA.md`.

**Why a second dataset.** S1 alone has six cells of one type, 2.5–4.9 pp of fade and every sample in reuse class C0. So it can test neither heterogeneity across chemistries and usage nor the reuse classes.

**Selection criteria**, fixed before screening:
1. It is a public dataset with a licence that allows reuse and redistribution with attribution.
2. It has a series of capacity checks (RPT-like), so the D18 forecasting protocol applies.
3. It involves second-life use or ageing past first-life end of life.
4. It adds heterogeneity S1 lacks: chemistry, usage conditions, depth of fade.
5. Its metadata can be verified against primary records.

**Screened** (details and evidence in `docs/EXTERNAL_DATA.md`):

| Dataset | Outcome |
|---|---|
| Wheeler et al. 2025, 20 A123 LFP cells, 7 conditions, CC BY 4.0 | **Selected.** Meets all five criteria. |
| Stanford–Relyion (Cui et al. 2024), 8 retired Nissan Leaf LMO cells | **Deferred.** Best match on criterion 3, but no licence is declared, so it fails criterion 1. Permission will be requested. |
| Beatty et al. 2025 (Loughborough) | **Rejected** on criterion 2: capacity only at the start and end. |
| Mowri et al. 2025 | **Rejected** on criterion 2: only 7 monthly second-life cycles. Its data licence is also unverified. |
| NASA, Oxford, CALCE, Severson 2019 | **Not screened:** first life only (criterion 3). |

**Consequences for the paper.**
- **Reporting:** results on S6 are reported separately per dataset. There is no pooling across datasets and no claim of transfer between them unless tested.
- **What "second life" means in S6:** ageing below 80 % SOH under the same EV-like protocol at 50 °C, not a repurposed duty. State this explicitly.
- **Protocol:** the D18 protocol is applied to S6 with the characterisations every 100 cycles as RPTs, and the test cases (or cells) as tasks. Deviations forced by S6's structure (no calendar schedule, cycle-based spacing) are recorded before any model is run on S6.
- **29 Sep 2026, learning-rate grid extended (D11 edge rule).** In the first full run (commit `8848b08`), stage 1 selected the TFT learning rate at the upper edge of the D11 grid (1e-2). D11 requires the grid to be extended and tuning repeated when this happens.
  - The backbone grid was extended to {3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1} for **both** backbones (equal budget), and notebook 12 was re-run. The re-run is the reported result.
  - First-run values, for the record: P1 final average RMSE was 0.464 pp for TFT + EWC, 0.465 for TFT fine-tuning, 0.450 for the schedule-only ridge and 0.632 for persistence. These are superseded.
  - The replay buffer was also selected at a grid edge (50). Every grid value exceeds the task size (see above), so extending that grid cannot change the result, and it was left as is.

---

## D20. Forecasting protocol on the second dataset (S6), fixed before any model is run on it

**Status:** Adopted (29 Sep 2026). It was committed before any forecasting model was fitted on S6. It is D18, with the changes that S6's structure forces. Deviations are logged at the end of this entry.

**Checks as RPTs.**
- Every 50 °C characterisation (every 100 cycles) is an RPT.
- The initial and final characterisations are excluded: they ran at 25 °C, a different temperature (same reasoning as RPT 2 in D16).
- **Labels (D17 analogue):**
  - capacity = the C/25 discharge capacity of the check;
  - reference = the cell's first valid 50 °C check;
  - `SOH_Percent = Capacity_Ah / Reference_Capacity_Ah * 100`.
- **Step validity** (`src/wheeler.py`). A discharge step counts only if it starts from full charge (≥ 3.40 V) and reaches the 2.5 V cut-off; a charge step only if it reaches 3.65 V. A check without a valid C/25 discharge has no label and is excluded as origin and target. This includes the documented acquisition fault at cycle 100 in cells 1a, 2a, 2c and 3a.

**Samples.**
- Origin check *k*, target check *k*+*h*, *h* ∈ {1, 2, 3} valid checks ahead (100–300 cycles).
- Target: ΔSOH (pp).
- Errors are reported in SOH pp, as in D18.

**Inputs.**
- **F0:** SOH at the origin; cycles since the first check; fade rate (least-squares slope over the last three valid checks, pp per cycle); horizon; the five static test-case conditions from Wheeler et al. 2025, Table 2 (CC-CV or CC charge, charge voltage, charge C-rate, WLTP or 2C discharge, DoD); cycles to the target check (planned).
- **F1:** F0 plus diagnostics at the origin check:
  - C/25 discharge charge in three voltage windows;
  - IC peak positions and heights in five windows fixed from the fresh-cell curves;
  - C/25 charge–discharge hysteresis at half capacity;
  - a C/3 vs C/25 overpotential resistance proxy;
  - C/3/C/25 capacity ratio, C/3 discharge capacity, C/25 charge capacity and C/25 mid-discharge voltage.
- No current values are used.

**Tasks and splits.**
- A task is a **test case** (seven usage conditions; the cells of a case are pooled).
- Splits are per cell with the D18 purged 60/80 rule. A task's split is the union of its cells' splits.

**Protocols.**
- **P1:** a continual stream over the seven test cases, 10 orders × 5 seeds.
- **P2:** leave one test case out, i.e. an unseen usage condition (seven folds).

**Methods, tuning, metrics and tests.**
- As in D18. The backbone learning-rate grid is the extended D18 grid {3e-4, …, 1e-1} from the start.
- Linear extrapolation is fade rate × cycles to target. The "schedule" ridge uses the known usage: static conditions plus cycles to target.
- **Added secondary metric:** agreement of the forecast reuse class (D10 cut-points: C0 > 95, C1 90–95, C2 80–90, C3 ≤ 80 % SOH) with the measured class at the target, as accuracy and macro-F1. S6 spans 100 % down to about 30 % SOH, so all four classes occur.

**Limitations stated in advance.**
- Ageing is accelerated at 50 °C.
- "Second life" in S6 means ageing below 80 % SOH under the same EV-like protocol, not a repurposed duty.
- There are two or three cells per task.
- Results are reported per dataset; nothing is pooled with S1.

### D20 deviations and disclosures
- **29 Sep 2026, row order of the S6 exports (implementation fix, before any model run).** The exported characterisation rows are not in time order: rows of different steps are interleaved. So the first and last rows of a step were not its start and end, and the step-validity rule rejected 188 checks.
  - `read_check` now sorts by `Test_Time`. After the fix, exactly four checks lack a label: cells 1a, 2a, 2c and 3a at cycle 100, which are the acquisition faults documented by the authors.
  - The authors' capacity (maximum C/3 charge counter) does not depend on row order and matched ours on all 843 comparable checks both before and after the fix.
- **29 Sep 2026, never-observed features dropped (implementation fix, before any S6 result was seen).** The IC window near 3.22 V never contains an interior peak in the S6 data, so `ic_pos_p2` and `ic_height_p2` are always missing. Gradient boosting fails on an all-missing column, and the first full run of notebook 13 stopped at the baselines, after the continual-learning runs but before any result was printed or saved.
  - F1 now excludes features with no observed value in the samples. The rule uses no label information.
  - The notebook was re-run in full.
- **Smoke profile.** Notebook 13 has an `EXPERIMENT_PROFILE=smoke` setting (one order, one seed, two epochs) that checks the code end to end. Its output is not reportable, was not kept, and changed no setting.

---

## D21. Extended data and protocol for TFT-scale training

**Status:** Drafted 3 Oct 2026; reviewed and revised the same day (one-standard-error tie rule, 0.05 pp equivalence threshold, minimum series length, results files, pretraining mode); **adopted 3 Oct 2026 on the instruction of T. Sitaraman, before any model was fitted on S7–S9. The other team members have not yet reviewed it; their review is recorded here when it happens.** Any change after adoption is logged as a deviation. This entry is written **after** the S1 and S6 results, the reproducibility check (`docs/RESULTS.md`) and the data audit were seen, and must be reported as such. No model has been fitted on S7–S9.

### Why
1. **Data size.** S1 (279 samples) and S6 (2,409) are too small for the TFT to be the right model: it is beaten by an MLP, linear extrapolation and gradient boosting, and varies by 0.25–0.54 pp across task orders.
2. **Reproducibility.** Validation tuning with 3 runs per grid point selected the TFT's EWC λ by a 0.008 pp near-tie, and the S1 and S6 significance results changed with the numerical environment.
3. **A missing input.** On S6 training data, fade per 100 cycles rises with the calendar days those cycles took (median within-cell ρ = 0.64, positive in 20/20 cells), but no S6 input carries calendar time. S1 already has it.

### Data and roles
- **First-life pretraining pool:** S7 (228 cells), S8 (55), S9 (130); `docs/EXTERNAL_DATA.md`. No second-life claim rests on them.
- **Second-life continual stream:** S6 test cases (as D20) and S1 cells (as D18). Datasets are reported separately, as before.

### Labels (one SOH series per cell, relative to its first valid measurement)
- **S7:** capacity = |`delta_q_dischg_Ah`| of the check-up discharge runs (`cyc_condition` = 2, `CHECKUP_RT`: 25 °C, 1.0 A to 2.5 V), read from `cell_eocv2`. The stored value is negative although the data dictionary says it is non-negative; the absolute value is used and the discrepancy disclosed. Excluded: the manual check-up after the thermal-management failure (any check-up dated after 27 May 2024; air-cooled, not comparable, as RPT 2 in S1) and the 12 side-experiment records `P000_0_S20_*`. The 12 cells that took part in the side experiment (Readme v2, Table 1) are kept and flagged. `cap_aged_est_Ah` (an extrapolated estimate) is **not** the label.
- **S8, S9:** per-cycle data at the cycling protocol, not slow-rate checks. A **block** of *K* consecutive valid cycles is one pseudo-check; capacity = mean discharge capacity of the block. A cycle is valid only if its discharge reaches the protocol cut-off. *K* is fixed before any label is inspected beyond counting cycles: proposed *K* = 10. These capacities are measured at the cycling rate and temperature, which is stated in the paper.
- **S1, S6:** unchanged (D17, D20).
- **Minimum series:** a cell is used only if it has at least 5 valid checks (three for the fade rate, plus at least one target).

### Inputs
- **Shared across all datasets (pool and stream):** SOH at origin; fade rate (least-squares slope over the last three valid checks); horizon *h*; time since the first check and time to the target, **in calendar days and in equivalent full cycles** where the dataset records them (missing-value indicators otherwise); test temperature; charge and discharge C-rates; SoC window or DoD; chemistry (one-hot). For S6 this adds calendar days to target, which is the change motivated by finding 3; it is applied identically to every method.
- **Dataset-specific diagnostics** (S1 F1, S6 F1) are used only within their own dataset, as before.
- **Never inputs:** protocol currents set from a previous capacity (D14), throughput scaled from capacity, any label after the origin. Each new dataset's columns are screened with `scripts/audit_data.py` rules before use, and the screen is committed with the data tables.

### Splits and protocols
- Per-cell purged 60/80 splits within every dataset, as D18.
- **P1, P2 on S1 and S6:** unchanged, re-run with the inputs above.
- **P3 (new, headline for the TFT):** pretrain jointly (all pool cells together, not as a stream) on the pool's training splits (early stopping on the pool's validation splits), then learn the S6 test cases as a continual stream (10 orders × 5 seeds, as D20). Methods: TFT from scratch; TFT pretrained + fine-tuning; **TFT pretrained + EWC** (anchor = pretrained weights, Fisher information from the pool); the same three for the MLP; the D18 baselines. Metrics: the D18 continual-learning metrics on S6 **plus retention on the pool's test splits** after the stream (forgetting of first-life knowledge).

### Tuning (changed from D11)
- **10 tuning runs per grid point** instead of 3, on tuning orders disjoint from the evaluation orders.
- **Tie rule, fixed now (one-standard-error rule):** among the grid values whose mean validation error is within one standard error (over tuning runs) of the best, choose the most conservative: the smallest learning rate, the smallest EWC λ, the smallest replay buffer. The selection margin is reported with every selected value.

### Statistics (changed from D18)
- As D18 (paired Wilcoxon, Holm correction, rank-biserial effect sizes, bootstrap CIs), **plus medians** of every paired difference.
- **Replication rule:** a difference is called significant only if p_Holm < 0.05 in the main run **and** in a replicate run with a different, pre-specified seed set (seeds 5–9). Differences below 0.05 pp are reported as practically equivalent whatever their p-value. Rationale: 0.05 pp is about half of the label shift caused by a definitional choice alone (pulse charge in or out, up to 0.089 pp, D17); a model difference smaller than that is not meaningful. (On the results seen so far, a 0.02 pp threshold would classify every comparison the same way.)
- Every results file records the full environment, including torch.
- **Results files:** runs under D21 write new files (`results/forecasting_d21_*.json`). The existing `results/forecasting.json` and `results/forecasting_wheeler.json` stay unchanged as the pre-D21 record.

### S6 ageing cycles (route 1): download authorised, use deferred
The between-check ageing files of S6 (827 files, 23.2 GB compressed, 132 GB uncompressed; measured from the zip indexes on 3 Oct 2026) would give per-cycle covariates. They are **not** part of this entry: the download takes about 5 days on the current connection, and in the 70 % DoD cases the ageing discharge is set from the last C/3 charge capacity (Wheeler et al. 2025), which needs its own leakage rules. **3 Oct 2026: the download was authorised by T. Sitaraman** and started (stored gzip-compressed and checksummed like the characterisation files). Their use in any model needs a separate decision entry (D22) with those leakage rules, adopted before use.

### Limitations stated in advance
- The pool is first-life data under laboratory conditions; transfer to second life is tested only on S6 (aged at 50 °C) and S1.
- S7 check-ups are at a single temperature (25 °C); S8 and S9 labels are cycling-rate capacities, not slow-rate checks.
- Pseudo-checks (S8, S9) and check-ups (S7) differ in interval; time-to-target inputs carry this.

### D21 deviations and disclosures
- **3 Oct 2026, S7 cold check-ups excluded (implementation, before any model was fitted on S7).** Check-ups are run at room temperature (25 °C). Three check-ups (CU 11 of three cells aged at 0 °C) started at about 1 °C and read 18 pp below the cell's previous check-up (mean change into a warm check-up: −1.9 pp); each is its cell's last check-up. Check-ups starting below 20 °C are excluded (`pool.S7_MIN_CHECKUP_TEMP_C`), as RPT 2 of S1 and the 25 °C checks of S6 are excluded for temperature. The rule uses the measured start temperature only. Thirty-four check-ups started at 20–22 °C and are kept; their temperature is kept in the table (`check_temp_c`).
- **3 Oct 2026, S7 time inputs (implementation).** D21 lists equivalent full cycles among the shared time inputs "where the dataset records them". For S7 they would come from charge throughput, which in cycling cells shrinks with capacity and so partly encodes the label (as in D14), and whose future value is not known at the origin. S7 therefore uses calendar days only.
- **S7 table (3 Oct 2026, `scripts/build_pool_tables.py`):** 228 cells, 3,912 valid check-ups; 195 cells in 67 condition sets have at least 5 (3,807 check-ups), giving 10,251 forecasting samples. Reference capacities 2.899–2.944 Ah; SOH down to 32.1 %.
- **3 Oct 2026, S8 and S9 labels refined from the authors' protocols (before any model was fitted on S8 or S9; only the protocol descriptions and the per-cycle structure were read).** D21's block rule (mean discharge capacity of 10 consecutive valid cycles) assumed every cycle is a full discharge at one rate. The protocols show that this holds only for part of the data:
  - **S8 (XJTU, "Data Introduction-English.pdf"):** batches 1–2 discharge every cycle fully at 1C; batch 3 rotates the discharge rate through 0.5/1/2/3/5C; batches 4–6 discharge partially and add dedicated capacity-measurement cycles (labelled "[test capacity]" in each cycle's description: 1C in batches 4–5, 0.5C in batch 6). The first cycle of every cell is an initial 0.2C capacity test and is excluded (different rate). Labels: batches 1–3 use only the full 1C discharges to 2.5 V (in batch 3, every fifth cycle), in blocks of 10; batches 4–6 use the dedicated measurement cycles, **one measurement per check** (they are already checks). All S8 cells were stopped at 80 % SOH, and the room temperature was not controlled.
  - **S9 (Tongji):** every cycle is a full constant-current discharge to the cut-off (2.65 V NCA, 2.5 V NCM and blend) at one current per cell; blocks of 10 cycles as in D21.
  - **Validity:** a measurement cycle counts only if its discharge reaches the cut-off (within 0.02 V).
  - **Time inputs:** days to target, and cycles to target (the planned protocol fixes the number of cycles, so it is known at the origin). No throughput.
- **3 Oct 2026, S9 time-counter restarts (implementation fix, before any model was fitted on S9).** In 5 of the 130 S9 files the time counter restarts once while cycle numbers continue (`time_resets` in the table). The first build took each cycle's time from the counter, so checks after a restart sorted before earlier ones and SOH reached 112.8 % from interleaved series. Time is now made continuous by carrying the elapsed time across the restart; the length of any pause at the restart is not recorded, so `days_to_target` across it is a lower bound for those 5 cells. The build now asserts that time increases with cycle number in every cell (the first build's check ran after sorting by time and could not fail).
- **S8 early capacity rise.** In 41 of 55 S8 cells capacity rises over the first ~60 cycles (e.g. 1.83 to 1.92 Ah); the authors' per-cycle discharge capacity agrees with the integrated discharge current to within 1 mAh. SOH above 100 % is therefore real in S8 and is kept. (An apparent rise of up to 2.1 % in S9 came from the reference cycles described below, not from the cells.)
- **Pool tables (3 Oct 2026, `scripts/build_pool_tables.py`):** S7 195 cells / 67 conditions / 10,251 samples (SOH 32.1–100 %); S8 55 cells / 6 conditions / 9,699 samples (78.8–105.0 %); S9 119 of 130 cells / 10 conditions / 16,761 samples (66.6–100.0 %). In total 369 cells, 83 conditions and 36,711 forecasting samples, against 2,409 in S6.
- **3 Oct 2026, S9 short segments excluded (implementation fix, found by the pool audit, before any model was fitted on S9).** In some NCA files every 26th "cycle" is a short segment that discharges about 22 mAh (against about 3,100 mAh for a full cycle) but still touches the cut-off voltage, so it passed the full-discharge rule and pulled its 10-cycle block about 10 % low; the audit's SOH-jump check flagged 57 cells. A full discharge must now also deliver at least half of the cell's median per-cycle capacity (`pool.FULL_DISCHARGE_FRACTION`). The rule removes only these segments: real fade in S9 never goes below 66 % SOH. S9 changes from 5,913 to 5,875 checks and from 16,929 to 16,818 samples.
- **3 Oct 2026, S9 reference cycles excluded (implementation fix, found by the pool audit, before any model was fitted on S9).** After the previous fix, the audit still showed single-check SOH spikes of up to +4.2 pp in the blend cells. Their cause: periodic reference cycles (every 50th cycle in the blend conditions) that run for about 15 h instead of about 4 h and either discharge at a different current (2.5 A instead of 5 A in `CY25-05_2` and `CY25-05_4`) or contain two discharge steps whose charge accumulates beyond the cell's capacity (`CY25-05_1`, at the standard 2.5 A). Regular cycles must now discharge at the cell's standard current (within 2 %, `pool.CURRENT_TOL`) and contain exactly one discharge step. Both rules use the current record only. S9 then has 5,856 checks and 16,761 samples; no SOH increase between consecutive checks exceeds 2 pp (largest 1.79 pp, 3 above 1 pp), and the maximum SOH is 100.0 %.
- **3 Oct 2026, P3 implementation choices (before any P3 result was seen; a 2-epoch smoke run checked the code only and was deleted).**
  - **Shared inputs** (`src/p3.py`, `SHARED_FEATURES`), each with its source in the module docstring. Two inputs were added to D21's list so that S6's test cases stay distinguishable as in D20: the charge cut-off voltage (`v_charge_max`) and whether the charge has a CV phase (`cv_charge`). Without them, cases 1–3 had identical inputs. A condition a dataset does not define is missing, not guessed (e.g. S8 temperature: uncontrolled room temperature; S7 cyclic DoD: only a voltage window is given).
  - **S6 under the shared inputs** reproduces the D20 samples exactly (2,409 samples, identical targets and cycle counts).
  - **Pretraining is done once per (backbone, learning rate, seed)** and copied into every order and method; the pretrained fine-tuning and EWC variants start from the same weights, so they differ only by EWC. EWC's first anchor and Fisher information come from the pool (500 sampled training rows, as D11).
  - **Tuning:** stage 1 selects the learning rate per backbone with pretrained fine-tuning (criterion: plasticity, as D11); stage 2 the EWC strength with pretrained EWC (final average error); 10 tuning orders disjoint from the 10 evaluation orders; one-standard-error rule as above. The from-scratch variants use the same learning rate.
  - **Baselines** (persistence, linear extrapolation per cycle as in D20, ridge, gradient boosting on the shared inputs, schedule ridge) are fitted on all S6 training rows, as in P1; inputs never observed in those rows are dropped (as D20).
  - **Provenance:** the run starts from a commit that contains all P3 code (an untracked file would not mark the tree as dirty).
- **3 Oct 2026, D11 edge rule added to the P3 tuner (validation data only; no test result had been computed).** The first P3 run selected the MLP learning rate at the lower edge of the grid (best value 3e-4, the smallest). D11 requires the grid to be extended and tuning repeated when the selection is at a grid edge; the P3 runner only recorded edge selections. The run was stopped before any evaluation unit, and the tuner now extends the grid by two values beyond the edge on the grid's own ladder (learning rates 1-3 ladder, EWC strength powers of ten) and repeats the selection, at most twice; each extension is recorded with the selection. Completed tuning units were reused from the checkpoints (identical code in `src/`).

### D21 addendum A1: ablation of the D20 → D21 change for the TFT (post hoc; written 3 Oct 2026 after the P3 results, before this ablation was run)

**Why.** TFT from scratch reaches 0.401 pp in P3 (D21) against 0.883 pp for TFT fine-tuning in D20, but inputs and tuning changed together (RESULTS.md, P3 finding 6). This ablation attributes the difference. It is an explanatory analysis decided after seeing P3 and is reported as such; it changes no P3 result.

**Design (fixed now).** TFT from scratch with fine-tuning on the S6 stream, the 10 D20/P3 evaluation orders, seeds 0–4, 200 epochs, window 4 over (test case, cell × horizon), preprocessing fitted on the first task (as both D20 and P3 scratch runs). One code path (`p3.run_stream`); only two factors vary:
- **Inputs (4 levels):** D20 F1 (27 inputs, with diagnostics); D20 F0 (10 inputs: SOH at origin, cycles since first check, fade rate per cycle, horizon, cycles to target and the 5 test-case conditions; F1 without the 17 diagnostics); D21 shared inputs; D21 shared inputs without the explicit calendar-time inputs (`days_since_first`, `days_to_target`).
- **Learning rate (2 levels):** 1e-2 (selected for the TFT under D20) and 3e-3 (selected under D21).
- **Sanity check:** D20 F1 at 1e-2 should reproduce D20's TFT fine-tuning result (0.883 pp) up to the run-to-run variation documented in the reproducibility check.
- **Reported:** mean, s.d. and median final average error per cell of the 4 × 2 design; paired Wilcoxon tests (matched order × seed) for the input contrasts at each learning rate and the learning-rate contrast at each input level, Holm-corrected over that family. Script `scripts/run_ablation_a1.py`; results `results/forecasting_d21_ablation_a1.json`.

---

## D22. S6 between-check ageing cycles as inputs — adopted 5 Oct 2026 (see amendments)

**Status:** Draft (3 Oct 2026) for team review. Nothing in this entry may be run until it is adopted. The ageing files are being downloaded and verified (D21, route 1); no model has used them. Written after the P3 results and after reading one ageing file's structure (cell 1a, cycles 101–200), which must be stated.

**What the files hold.** One CSV per cell and block of 100 ageing cycles (827 files, 132 GB uncompressed), Arbin export: `Test_Time`, `Step_Index`, `Cycle_Index`, current, voltage, charge and discharge capacity and energy, `Internal_Resistance` (only on some equipment, Wheeler et al. 2025), `dV/dt` (the authors advise against it). In cell 1a (case 1, 70 % DoD), every cycle discharges the same 0.841 Ah: the protocol sets it to 70 % of the last C/3 charge capacity (Wheeler et al. 2025).

### Leakage rules (fixed before any use)
- **Never inputs (protocol-scaled, D14 analogue):** discharged and charged charge and energy per cycle, the discharge current setpoints, and anything computed from them alone. In the 70 % DoD cases they encode the previous check's capacity. The audit must show this explicitly (within-cell correlation of per-cycle discharged charge with the previous check's capacity) before the rule is relied on.
- **Allowed, measured responses:** end-of-discharge voltage, end-of-charge voltage after the CV phase, CV-phase duration and its final current, rest-voltage relaxation after charge and after discharge, and `Internal_Resistance` where recorded (missing otherwise, with an indicator). Ratios of protocol-scaled quantities (e.g. energy efficiency) are allowed only if the audit shows no within-cell correlation above 0.95 with the previous check's capacity.
- **Causality:** a sample with origin check *k* may use only ageing cycles completed before check *k*; nothing from cycles between *k* and the target.
- **Case 7 (100 % DoD):** every ageing cycle is a full discharge to 2.5 V, so its discharged charge is a capacity measurement at the ageing rate. It may serve as a dense label in a **separate, secondary analysis** only; the headline label stays the C/25 check capacity (D20).

### Features and models
- **F2 = D21 shared inputs + interval aggregates** of the allowed responses over the 100 cycles before the origin: mean, last value, and least-squares slope against cycle number. Used by every method, so comparisons stay matched.
- **Per-cycle TFT (optional, secondary):** a TFT whose window is the last *N* ageing cycles' allowed responses instead of the last 4 checks. It is a different model and is reported separately from P1–P3.

### Validation before use
1. Every file verified against the zip index (CRC-32) and its SHA-256 recorded (as now).
2. Completeness per cell: missing blocks, interrupted cycles, cycles with abnormal step structure (rules fixed before labels are read, as for S9).
3. Leakage audit of every F2 feature (rules above) in `scripts/audit_data.py`.
4. P1 and P3 re-run with F2 against the D21 inputs, under the D21 tuning and replication rules.

### Open questions for the team
- Whether to run the optional per-cycle TFT at all (compute and time).
- Whether the case-7 dense-label analysis is worth including.
- **4 Oct 2026, A1 replicate (decided before running it).** A1 is repeated on seeds 5–9 with everything else unchanged. Its contrasts are then judged by the D21 replication rule (significant after Holm correction in both seed sets, same sign, at least 0.05 pp in both); only replicated contrasts are stated as findings. Results: `results/forecasting_d21_ablation_a1_replicate.json`. **Outcome (4 Oct 2026):** dropping the diagnostics and adding calendar time replicate at both learning rates; D21 shared − D20 F0 replicates at 3e-3 only; the learning rate is practically equivalent for D21 shared and D20 F0 and inconclusive otherwise. Details in `docs/RESULTS.md`.

### Adoption and amendments (5 Oct 2026, written before any ageing feature was computed)

**Status change.** Adopted by Tarun on 5 Oct 2026 ("continue D22"); team review pending, as for D21. Since the draft, D23 has shown that learning the residual on linear extrapolation is the strongest form of the TFT, so the evaluation below uses the D23 setting. Before writing this, two ageing files were inspected for structure only (cell 1b, cycles 101–200; cell 1a, cycles 1101–1200): step order, currents and voltages of single cycles, and the check-cycle numbering. No feature was aggregated and nothing was related to any label.

**Structure found.** Each cycle is: charge (CC, then a CV hold inside the same step when the voltage limit is reached), 30-min rest, discharge (WLTP profile, or 2C in case 6), 30-min rest. `Cycle_Index` restarts in every file; the global cycle is the file's first cycle − 1 + `Cycle_Index`. `Internal_Resistance` is 0 (not recorded) in the cell-1 files. Late in life the charge can stop before the voltage limit (cell 1a), so the CV hold can be absent.

**Per-cycle responses (fixed here; the only ageing inputs).** Steps are classified by their median current: charge > +0.01 A, discharge < −0.01 A, rest otherwise.
1. `v_end_charge`: last voltage of the charge step.
2. `cv_seconds`: time in the charge step at or above `v_max − 0.002 V` (case `v_max`); missing for the CC-only cases 3 and 7.
3. `cv_end_current`: current at the end of the charge step if `cv_seconds > 0`, else missing (and always missing for cases 3 and 7).
4. `v_end_discharge`: last voltage of the discharge step.
5. `relax_after_charge`: voltage at the end of the rest after charge minus at its start.
6. `relax_after_discharge`: the same for the rest after discharge.
7. `internal_resistance`: last non-zero value in the cycle, else missing.
Charge or discharge durations, charge, energy and currents are excluded (protocol-scaled; for a fixed-current CC charge, the duration is the charge passed divided by the current).

**Causality and mapping.** The check at cycle *c* follows the ageing block of cycles *c* − 99 … *c*. An origin check at cycle *c* uses only that block; the builder asserts that every row of the block is timestamped before the check. The first check (cycle 0) has no block: its F2 values are missing and flagged.

**Completeness (fixed before any label is read).** A cycle is valid if it has a charge step, a rest, a discharge step and a rest, in that order. Invalid cycles are excluded from the aggregates and counted per cell. A block needs at least 50 valid cycles, otherwise its F2 values are missing. Missing or unreadable blocks are reported per cell, never filled.

**F2.** The D21 shared inputs plus, for each of the 7 responses, the mean, the last value and the least-squares slope against global cycle number over the block's valid cycles (21 inputs).

**Leakage audit (before any model).** (i) Within-cell correlation of per-cycle discharged charge with the previous check's capacity, to demonstrate the protocol scaling the rules exclude. (ii) For every F2 input, its within-cell Spearman correlation with the origin check's capacity, reported. Measured responses are not excluded for correlating with health, which is their purpose.

**Evaluation (replaces item 4 of "Validation before use").** The D23 E1 setting: S6 stream from scratch, horizons 1–3, residual and direct arms, TFT and MLP × fine-tuning and EWC, D21 tuning, seeds 0–4 (main) and 5–9 (replicate). Inputs F2 against the D21 shared inputs, run in the same runner. Primary tests: residual TFT + EWC with F2 against (a) linear extrapolation and (b) residual TFT + EWC with the D21 inputs. Same test families and replication rule as D23. Positive result: (a) replicated in favour of the F2 model.

**Before all 20 cells are downloaded and verified,** only feature extraction, the audit and smoke runs (not reportable) are done; the paper-profile run waits for the complete set. The case-7 dense-label analysis and the per-cycle TFT stay deferred.

**Deviation, 5 Oct 2026, 15:40 IST (before any model used the ageing inputs).** The first extraction (177 blocks of 7 cells) found only 31 % of cycles valid under the rule above (cell 1a: 2 %). Inspecting late-life blocks showed why: one ageing cycle (one `Cycle_Index`, 100 per file) then splits its charge into several steps (a short pause, sometimes a separate CV step) and its discharge into a WLTP part and a CC completion step, separated by short rests, and a low-current step sits between two 30-min rests after charge. The rule is therefore changed to: **a cycle is valid if all its charge steps precede all its discharge steps and a rest step directly follows the last of each.** The responses use the last charge and last discharge step, and the two relaxations are measured over the single rest step right after each (merging adjacent rests had made them span 30 or 60 min depending on the pattern). Decided from the step structure and the validity rates only; no ageing input had been related to any label. The extraction found one block ending after its check (cell 2a, cycle 100); that check is invalid (acquisition fault) and is never an origin, so no sample is affected. Leakage audit (i) on the first extraction: per-cycle discharged charge correlates r = 0.990–0.995 within cell with the previous check's capacity (cases 1, 2, 6), confirming the protocol scaling the rules exclude.

**Run log, 5 Oct 2026, 16:30 IST.** Re-extraction with the corrected rule (194 blocks, 7 cells): 97–98 % valid cycles in every cell; no block ends after a valid check; leakage audit (i) r = 0.993–0.995 (cases 1, 2, 6), 0.60 for cell 3a (7 blocks); audit (ii) in `results/d22_ageing_report.json`. `Internal_Resistance` is not recorded in any of these cells, so its three inputs are missing throughout (handled by the frozen preprocessor, as `rest_soc` on S6). A smoke run of `run_d23.py --inputs f2` (2 epochs, 1 order; not reportable, output deleted) checked the code end to end, including comparison (b) against the D23 reference runs. The paper-profile run waits for all 20 cells.

**Spark implementation, 5 Oct 2026, 21:45 IST.** The extraction is also implemented as an Apache Spark job (`scripts/spark_ageing_features.py`, PySpark 4.2.0, local mode on one machine; Java 21): one map task per 100-cycle block parses the raw file into per-cycle responses with the same tested function, and Spark SQL reduces the valid cycles of each block to the F2 inputs (mean, last value, slope as covariance/variance, the 50-valid-cycle rule). The job compares its output with the pandas builder and records the result in `results/d22_spark_validation.json`. A test on 6 blocks (4 also in the pandas output) gave identical values (relative difference < 1e-9, same missing values). The F2 inputs used for D22 will be those of the Spark job if the full-set validation passes; otherwise the pandas output, with the discrepancy reported.

**Run plan, 6 Oct 2026, 09:10 IST.** A detached pipeline (`data/raw/run_d22_pipeline.ps1`) waits for the download and then runs, stopping at the first failure: merge of the per-worker manifests with a completeness check (`scripts/merge_ageing_manifests.py`, all 827 files, one SHA-256 each); the pandas extraction and report; a gate requiring no causality violation at a valid check; the full Spark extraction, which exits with an error unless it matches pandas on every block; the D22 E1 parts and the collect step. `run_d23.py --inputs f2` reads the Spark output and refuses to run unless `results/d22_spark_validation.json` records identical values. Every output is reviewed before any result is reported.

**Run log, 6 Oct 2026, 17:00 IST.** Download complete: 827 files from 13 manifests, 131.9 GB raw, no SHA-256 conflict, none missing. (Two Cell 7a workers had run at once on the same files between 12:29 and 14:24 because a stop command missed one of them; Windows file locks made their writes fail rather than interleave, and a single clean worker then re-verified all 32 Cell 7a files against the zip index, size and CRC-32, and fetched the 2 missing ones.) Pandas extraction: 82,529 cycles, 80,287 valid, 827 blocks; one block ends after its check, at an invalid check (cell 2a, cycle 100), none at a valid check. Full Spark run: local[6], 827 map tasks, 82,529 cycles in 1,175 s. **Deviation in the validation gate.** The gate as written (relative difference < 1e-9) failed on one input, `age_v_end_charge_slope`: maximum absolute difference 1.3e-16 V per cycle, but relative 1.2e-5, because the true slope is about 1e-11 when the charge ends at the voltage limit (a value that is zero up to rounding). Every other input, the cycle counts and the missing-value pattern agreed. The criterion is changed to |Spark − pandas| ≤ 1e-9·|pandas| + 1e-12 with the same missing values; the absolute floor is far below the voltage resolution (~1e-4 V). Under it, all 827 blocks match (largest absolute difference over all inputs 2.0e-13). Both results are committed: `results/d22_spark_validation.json` (combined criterion) and `results/d22_spark_validation_strict_relative.json` (original criterion). Changed after seeing the failure; the failure is numerical rounding, not a difference in the computation.

**Outcome, 6 Oct 2026, 21:00 IST.** D22 E1 finished at 19:42 (`results/forecasting_d22_e1_f2.json`, commit `d69b495`); the D23 reference runs reproduced exactly on both recomputed units. Primary (a), residual TFT + EWC with F2 vs linear extrapolation: +0.138 / +0.159 pp, replicated, worse; **the registered positive result was not obtained.** Primary (b), F2 vs the D21 inputs: +0.141 / +0.188 pp, replicated, worse: the ageing inputs degrade the forecasts. Details in `docs/RESULTS.md`, D22 section.


---

## D23. Learned residual on linear extrapolation, and longer horizons (pre-registered)

**Status:** Adopted 5 Oct 2026 by Tarun, written and committed **before any D23 code was run**; team review of the protocol pending. Results go to `results/forecasting_d23_e1.json` and `results/forecasting_d23_e2.json`.

**Provenance (must be stated).** This entry was written after the P1, P3 and A1 results. In particular, the per-horizon S6 errors for horizons 1–3 were known: linear extrapolation 0.21/0.35/0.46 pp, gradient boosting 0.27/0.34/0.50 pp. The idea of learning a residual is therefore motivated post hoc. Horizons 4–6 on S6 have never been computed by any method.

**Question.** Linear extrapolation of the recent fade rate is the most accurate S6 forecaster (0.343 pp), and no learned model beats it. Can a continual learner that predicts only the *correction* to linear extrapolation beat it, and does any advantage grow with the horizon, where a straight line should fail first (accelerating fade)?

### Setup (fixed here)
- **Data and inputs:** S6 samples as in D21 (`p3.harmonised_samples`), the D21 shared inputs unchanged, the D18 purged per-cell splits, the seven S6 test cases as the stream. No pretraining (P3 showed none helps).
- **Linear-extrapolation forecast:** `lin = fade_rate_pp_per_cycle x cycles_to_target`, 0 where the fade rate is missing; this is exactly the D20/D21 linear-extrapolation baseline.
- **Residual arm:** the model target is `Delta - lin`; the forecast is `lin + prediction`. Because `lin` is known at the origin, every error (task RMSE, the error matrix, all CL metrics) is identical on the residual and the Delta scale, so the existing stream and metric code is reused unchanged.
- **Direct arm:** target Delta, as in D21 (the matched comparison).
- **Methods:** TFT and MLP backbones x fine-tuning and EWC, each in both arms (8 methods). Baselines: persistence, linear extrapolation, gradient boosting and ridge on the shared inputs, fitted as in D21.
- **Tuning (D21 rules, per backbone and arm):** learning rate on fine-tuning streams (criterion: plasticity), then EWC lambda on EWC streams (criterion: final average error); D21 grids; 10 tuning orders disjoint from the evaluation orders; one-standard-error rule; grid-edge extension on the grid's ladder.
- **Evaluation:** 10 orders x seeds 0–4 (main) and 5–9 (replicate).
- **E1:** horizons 1–3 (as D21).
- **E2:** horizons 1–6, everything else identical, tuned separately. Per-horizon errors are the RMSE over all tasks' test samples of that horizon after the last stream step.

### Tests (fixed here)
- **Primary, E1 and E2:** residual TFT + EWC against (a) linear extrapolation and (b) direct TFT + EWC, on final average error.
- **Secondary:** residual TFT + EWC against every other method and baseline on final average error, and against the other stream methods on forgetting and backward transfer; in E2, residual TFT + EWC against linear extrapolation at each horizon 1–6.
- Paired two-sided Wilcoxon on matched order x seed runs; against a deterministic baseline, the runs minus its single value. Holm correction within each seed set over all comparisons of one experiment (the per-horizon tests form their own family of six). The D21 replication rule decides: significant after Holm in both seed sets, same sign, |difference| >= 0.05 pp in both; |difference| < 0.05 pp in both is practically equivalent.
- **What counts as a positive result:** primary test (a) replicated in favour of residual TFT + EWC in E1 or E2. If the best residual model is the MLP rather than the TFT, that is reported as such and does not count as support for the TFT.
- Every outcome is reported, including a null or negative one; nothing here replaces the D21 results.

### Not done, with reasons
- **Leakage-inflation demonstration (considered, dropped):** the label leak of D14 concerns the superseded current-SOH task, where the protocol current is 0.9998 x the previous capacity; a leaky model there is trivially near-perfect and would not inform the forecasting results.
- **Knee-specific error analysis:** not pre-registered (defining the knee region needs the outcome); E2's horizon-wise errors are the pre-specified proxy.

### Run log
- **5 Oct 2026, 03:05 IST.** Code committed (`c7fa340`) after a smoke run (1 order, 2 epochs; not reportable, output deleted). The smoke run printed the deterministic E2 baselines at horizons 1-6 (linear extrapolation 0.22/0.35/0.46/0.57/0.74/0.85 pp), so these were seen before the paper run; they involve no tuning and change nothing above. Paper run started as 8 processes (E1, E2 x four parts); logs `data/raw/d23_*.log`.
- **5 Oct 2026, 11:45 IST. Outcome.** All parts finished (07:37); results collected at `5ae2f84`. Primary (a), residual TFT + EWC vs linear extrapolation: practically equivalent in E1 (−0.004 / −0.029 pp), not replicated in E2 (+0.076 / +0.049 pp); **the registered positive result was not obtained.** Primary (b), residual vs direct TFT + EWC: replicated in E1 and E2 (−0.17 and −0.24 to −0.30 pp). In E2, EWC lowers the residual TFT's forgetting and backward transfer (replicated). Details: `docs/RESULTS.md`, D23 section.

## D24. Proposed model RC-TFT (residual TFT with shift-adaptive EWC) and literature baselines (pre-registered)

**Status:** Adopted 12 Oct 2026 on the instruction of T. Sitaraman, written and committed **before any D24 code was run on real data** (code is tested on synthetic data in pytest temporary directories only). Results go to `results/forecasting_d24_e1.json` and `results/forecasting_d24_e2.json`. Plan and motivation: `docs/WORK_PLAN.md`.

**Provenance (must be stated).** Written after every earlier result, including D22 and D23. Two observations motivate it post hoc: forgetting appears only when groups differ (S1 cells 0.016–0.040 pp, S6 conditions 0.18–0.46 pp), and validation selected very different fixed EWC strengths across experiments (λ from 0.01 to 10⁵). The literature baselines answer a gap found in the audit of 12 Oct 2026: none of the model families of our own literature review (GPR, LSTM, TCN) had been run.

### The proposed model: RC-TFT
**RC-TFT** = *Residual Continual Temporal Fusion Transformer*: the D23 residual TFT (forecast = linear extrapolation + learned correction; label-free causal window of the four previous origins of the same cell and horizon; no task ID as input) trained continually with **shift-adaptive EWC (SA-EWC)**. TFT (Lim et al. 2021) and EWC (Kirkpatrick et al. 2017) are existing components; the contribution is their leakage-safe, extrapolation-anchored combination with a penalty that scales with the measured shift between battery groups. The D23 model (residual TFT + EWC with a fixed λ) is its ablation.

**SA-EWC (fixed here).**
- After task *j* ends, store, besides the EWC Fisher `F_j` and anchor `θ*_j`, the mean `μ_j` and standard deviation `σ_j` (population) of task *j*'s training inputs after the frozen first-task preprocessing (standardised values and missing indicators; dimension `D`). No rows are stored.
- When task *t* starts, for every finished task *j*: `d_{j,t} = sqrt((‖μ_t − μ_j‖² + ‖σ_t − σ_j‖²) / D)`, the 2-Wasserstein distance between diagonal Gaussians, per dimension. It uses only training inputs, never labels, and is fixed for the whole of task *t*.
- Penalty during task *t*: `(λ₀ / 2) Σ_j d_{j,t} Σ_p F_{j,p} (θ_p − θ*_{j,p})²`. A past task is protected more when the new task differs more from it. With every `d_{j,t} = 1` this is exactly the D23 EWC (unit-tested).
- λ₀ is tuned like λ (D21 grid and rules, criterion final average validation error). The backbone learning rate is the one D23 selected for the residual TFT (3e-3 in E1 and E2): it is selected on fine-tuning streams, which SA-EWC does not change.

### Literature baselines (fixed here)
Run in the D23 setting and runner: S6 stream from scratch, D21 shared inputs, D18 purged per-cell splits, the D23 evaluation orders and seeds (10 orders × seeds 0–4 main and 5–9 replicate) and the D23 tuning orders and rules (learning rate on fine-tuning streams, criterion plasticity; then EWC λ on EWC streams, criterion final average error; one-SE rule; grid-edge extension).

| Model | Literature link | Mode | Arms |
|---|---|---|---|
| BiLSTM (one layer, both directions over the window) | Meskine 2026 (ref 10) | continual: fine-tuning and EWC | residual (E1, E2); direct (E1) |
| TCN (two residual blocks of dilated causal 1-D convolutions, kernel 2, dilations 1 and 2) | Yao 2025 (ref 8), Meskine 2026 | continual: fine-tuning and EWC | residual (E1, E2); direct (E1) |
| Transformer encoder (one layer, two heads, learned positions, causal and padding mask) | closest ablation of the TFT (no variable selection or gating) | continual: fine-tuning and EWC | residual (E1, E2); direct (E1) |
| GPR (constant × Matérn ν = 2.5 with one length scale per input, plus white noise; hyperparameters by marginal likelihood, 3 restarts) | Takahashi 2023 (ref 4), Faraji-Niri 2023 (ref 3) | fitted once on all S6 training rows (as the D21 gradient boosting) | direct and residual |
| Random forest (500 trees, scikit-learn defaults otherwise) | classical baseline | as GPR | direct and residual |
| SVR (RBF kernel, standardised target, ε = 0.1; C from {0.1, 1, 10, 100} by RMSE on all S6 validation rows, then refitted on the training rows) | classical baseline | as GPR | direct and residual |

- The three sequence models use the TFT's input contract (window of 4, values plus missing indicators plus a padding flag; inputs only, never labels), its training loop, early stopping, frozen first-task preprocessing and target scaling, and EWC code. Their size is matched to the TFT within ±20 % of its parameter count; counts are recorded in the results file.
- Tabular baselines (GPR, random forest, SVR) use median imputation with missing indicators, standardisation and constant-column removal, fitted on training rows only. Like gradient boosting they see every task's training rows at once, so they are not continual learners; this is stated with every comparison.
- Direct arms in E2 are not run (compute budget, below).

### Tests (fixed here)
Reference: RC-TFT (`tft_residual_saewc`). The committed D23 runs (all eight methods, D21 inputs) enter the comparison after one main and one replicate unit of residual TFT + EWC are recomputed and reproduce to 1e-9 pp (as in D22); otherwise the run stops.
- **Primary (one Holm family per seed set):** RC-TFT against (a) residual TFT + EWC with a fixed λ (D23), on final average error and on forgetting, and (b) linear extrapolation, on final average error.
- **Secondary:** RC-TFT against every other method and baseline on final average error, and against the stream methods on forgetting and backward transfer (one family, as in D23); in E2, against linear extrapolation at each horizon 1–6 (a family of six).
- Paired two-sided Wilcoxon over matched (order, seed) runs; against a deterministic baseline, the runs minus its single value. The D21 replication rule decides (p_Holm < 0.05 in both seed sets, same sign, |difference| ≥ 0.05 pp in both = replicated; |difference| < 0.05 pp in both = practically equivalent; otherwise not replicated).
- **What counts as a positive result for SA-EWC:** (a) replicated in favour of RC-TFT on final error or on forgetting, and not replicated against it on the other. If it fails, RC-TFT is reported with that outcome and the fixed-λ model stays the best-supported variant; nothing is re-run to change it.
- Reported for every method: final average error (RMSE and per horizon), forgetting, backward transfer, plasticity, the error of every task after the last step (per-group table), training time and parameter count.

### Compute budget and run plan
One night on the team's ASUS PC (`scripts/run_d24_night.py`). D23 timings: about 23–37 s per TFT stream in E1 and 37–59 s in E2, single-threaded; D23 ran 8 processes in parallel. Parts in priority order (a part = tuning + 200 evaluation streams):
1. E1: RC-TFT; BiLSTM, TCN and Transformer residual.
2. E2: RC-TFT; BiLSTM, TCN and Transformer residual.
3. E1: BiLSTM, TCN and Transformer direct.
4. After the parts: the D23 reference reproduction, the tabular baselines and the results files.
5. Only if time remains, on an otherwise idle machine: the Spark scaling run (WS2.2 of the work plan: the same 120 blocks, spread over the cells, at 1, 2, 4 and 6 cores; outputs checked against pandas).
Parts that do not finish are reported as not run; nothing is shortened silently. Smoke runs (1 order, 2 epochs) only check the code path and are not reportable.

### Not done, with reasons
- **P2 (unseen condition) for the new models:** not in the one-night budget; planned as a follow-up.
- **Direct arms in E2 and MLP with SA-EWC:** budget; the MLP is not the proposed backbone.
- **Bagged GPR as in ref 4:** a single GPR with per-input length scales is used; bagging is not part of D24.

### Run log
- **12 Oct 2026. Code committed** (`be2f2f5`) after tests on synthetic data only (pytest temporary directories and the session scratch space; never real data): 168 tests pass, including an end-to-end smoke run of `scripts/run_d24.py` (two parts, collection with parts listed as not run) and of the night orchestrator's process pool. No D24 code has been run on S6. Sequence-model sizes: TFT 12,708 parameters; BiLSTM 11,181, TCN 12,205, Transformer 11,905 (within 20 %). Timing on synthetic data of S6 size at full epochs: one EWC stream takes about 25 s for the TFT and 7–9 s for the three baselines, so the plan is expected to take 4–5 h with 8 workers. Runbook: `docs/D24_NIGHT_RUN.md`.
- **12 Oct 2026. Amendment (before any D24 code has run on S6).** A timing check on synthetic data showed that one GPR fit with per-input length scales and 3 restarts takes 655 s on 1,120 training rows (4 CPU threads in the cloud session). Exact-GP cost grows with the cube of the rows, so each E2 fit (about twice the rows) would take of the order of 1.5 h, about 3 h for the two arms, at the end of the night. Two changes, fixed now: (1) GPR is fitted on a seeded random subsample of at most 1,000 training rows (`tabular_baselines.GPR_MAX_ROWS`; the rows used are recorded as `n_fit_rows`); the other tabular baselines still use every training row; (2) the deterministic and tabular baselines become a checkpointed part (`baselines`, one per experiment) that runs in the parallel phase after the eight residual parts, not in the collection step. Nothing else in D24 changes. Also added: `scripts/summarise_d24.py`, a Markdown summary read from the results files, for the run report.
- **13 Oct 2026. Baselines ahead of the night run (no protocol change).** `run_d24.py --baselines-only` computes the `baselines` part (deterministic and tabular baselines, single fits on all S6 training rows, no tuning orders or seeds) and writes `results/forecasting_d24_baselines_<exp>.json`, so the classical-ML comparison (GPR, random forest, SVR beside persistence, linear extrapolation, ridge and gradient boosting) is available before the model parts run. The part is checkpointed, so the night run reuses it unchanged; `make_d24_outputs.py` shows these baselines beside the committed D23 runs until the full D24 files exist.
- **13 Oct 2026. Night run not completed; baselines part run.** `run_d24_night.py --workers 8` from `main` (`ecd882b`, clean tree, 173 tests passed, none skipped; pre-flight passed). Smoke passed for all 13 parts (01:48). The paper run was stopped twice by the machine running out of memory (about 01:50, and on 12 Oct at about 23:42 on the earlier branch code), and a third start at 07:12 (deadline 14:00) was stopped by hand shortly after 08:13 to meet a 09:00 deadline, with all eight residual parts still tuning. No model part finished, so there is no RC-TFT, BiLSTM, TCN or Transformer result, no D23 reproduction check, no `results/forecasting_d24_<exp>.json` and no Spark scaling run; their checkpoints are kept and a re-run resumes. The `baselines` part was then run on its own (`run_d24.py --exp e1|e2 --baselines-only`, same commit, `git_dirty` false), giving `results/forecasting_d24_baselines_e1.json` and `..._e2.json`; `make_d24_outputs.py` shows them beside the committed D23 runs. The tabular baselines are single fits on all S6 training rows, not continual learners, so no paired tests against them are reported.
- **13–14 Oct 2026. Night run completed.** `run_d24_night.py --workers 8 --deadline 07:30 --spark-scaling` from `main` (`947b33a`, clean tree; an uncommitted notebook edit was stashed first; pre-flight passed), Python 3.12.10, torch 2.13.0+cpu. Started 21:05 IST; smoke passed for all 13 parts (21:16); paper run 21:16–23:34 (longest part E2 RC-TFT, 138 min); collection 23:34–23:44; Spark scaling 23:44–00:20; outputs 00:20. All 13 parts ran, none failed, none was stopped by the deadline (`parts_not_run` empty in both files, `git_dirty` false). D23 reproduction check passed in E1 and E2 (both units equal to the committed values to about 1e-14). Outputs: `results/forecasting_d24_e1.json`, `..._e2.json`, `results/d22_spark_scaling.json`, `results/d22_spark_validation_scale_c{1,2,4,6}.json`, tables and figures. Spark scaling: every thread count matched pandas; the speed-up from 1 to 2 threads (4.9x, efficiency 2.5) is above linear, so the 1-thread time probably includes a one-off cost (for example JVM start-up or a cold file cache) and is not a clean single-thread reference; this is not explained yet and the scaling figures should not be read as parallel efficiency until it is.
- **14 Oct 2026. Environment of the night run (recorded after the fact).** The night run used Python 3.12.10, numpy 1.26.4, pandas 2.2.3, scikit-learn 1.5.2 and torch 2.13.0+cpu, not the pinned versions of `requirements.txt` (Python 3.11.15, numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, torch 2.14.1). Effect, checked against the committed files: the D23 reference units reproduced to about 1e-14, so the neural results are unaffected in that respect; among the baselines, gradient boosting differs (E1 0.451 → 0.510 pp, E2 0.898 → 0.831 pp against the pinned-environment `forecasting_d24_baselines_*.json` and D23), random forest by at most 0.01 pp, and all others are identical. The D24 results files and tables keep the run-environment values; the paper reports the pinned values. Future runs should use the pinned environment (pre-flight now only checks that the libraries import, not their versions).

## D25. Calibrated residual TFT + EWC and residual pretraining on the first-life pool (pre-registered)

**Status:** Written 14 Oct 2026 on the instruction of T. Sitaraman and committed **before any D25 code was run on real data** (code tested on synthetic data in pytest temporary directories only); team review pending. Results go to `results/forecasting_d25_e1.json` and `results/forecasting_d25_e2.json`. Plan and motivation: `docs/WORK_PLAN.md`, section "D25"; runbook `docs/D25_NIGHT_RUN.md`.

**Provenance (must be stated).** Written after every earlier result, including D24. Three observations motivate it post hoc:
- In E2, linear extrapolation beats the residual TFT + EWC at horizons 1–2 (replicated, D23 and D24), so the learned correction adds error where the recent trend is already well determined.
- EWC already keeps the residual TFT's forgetting at about 0.07–0.10 pp, and SA-EWC changed nothing (D24), so a further penalty is unlikely to move the final error.
- Each S6 task has 135–225 training samples for a 12.7k-parameter model; more inputs made the model worse (D22). Pretraining on the pool (P3, D21) was tested only for the direct target, before the residual form existed.

The S6 per-horizon errors of every D23/D24 model and baseline are known; no D25 quantity (shrinkage factors, intervals, residual-scale pool results) has been computed on real data.

**Questions.**
- Q1: Does shrinking the learned correction per horizon, towards zero where it does not help on validation rows, make the residual model at least as accurate as linear extrapolation, or better?
- Q2: Does pretraining the residual model on the first-life pool (S7–S9) improve it on the S6 stream?
- Q3: Are the model's forecast intervals better than linear extrapolation's when both are calibrated in the same way?

### Setup (fixed here)
- **Data, inputs, splits, stream, orders, seeds:** as D23/D24: S6 samples (`p3.s6_samples`), the D21 shared inputs, the D18 purged per-cell splits, the seven S6 test cases as the stream, the D23 evaluation orders (10) with seeds 0–4 (main) and 5–9 (replicate), and the D23 tuning orders. E1 = horizons 1–3, E2 = horizons 1–6.
- **Model:** the D23 residual TFT + EWC (forecast = linear extrapolation + learned correction; window of 4 earlier origins of the same cell and horizon; no task ID).
- **Shrinkage (Q1; `src/calibration.py`).** After each task, for each horizon h, `alpha_h = clip(sum(p*y) / sum(p^2), 0, 1)` over that task's validation rows of horizon h (`p` = the model's correction, `y` = the residual target). The shrunk forecast is `lin + alpha_h * p`. A horizon without validation rows in the task keeps its previous factor; every factor starts at 1. Shrinkage has no hyperparameter and does not affect training, the weights or the EWC Fisher/anchor, so the raw and shrunk models come from the same streams.
- **Intervals (Q3).** After each task, per horizon and level (90 %, primary; 80 %, secondary), the split-conformal half-width is the `ceil((n+1)*level)`-th smallest absolute validation residual of the shrunk model (the largest one, flagged, if that index exceeds n). Linear extrapolation gets intervals in the same way from the same rows. Score: the interval score (Gneiting and Raftery 2007), in pp, mean over each task's test rows and then over tasks (as the final average error); coverage and mean width are reported with it.
- **Scratch arm.** The D23 learning rate and λ of the residual TFT (E1: 3e-3, λ 1; E2: 3e-3, λ 1000); no new tuning. The raw model of every unit must reproduce the committed D23 unit to 1e-9 pp; otherwise the collection stops and the run is reported as not reproducible.
- **Pretrained arm (Q2).** Pool S7–S9 samples (D21 inputs and splits, the experiment's horizons) on the residual scale. The pool's linear extrapolation is the D21 per-day fade rate (last three checks) times the days to the target, 0 where the rate is missing (the pool's calendar cells have no cycle count, so S6's per-cycle definition cannot be used; S6 keeps its D23 definition). Pretraining, frozen pool preprocessing and target scaling, and tuning follow D21 P3: learning rate on pretrained fine-tuning streams (criterion plasticity), then λ on pretrained EWC streams (criterion final average error), D21 grids, one-SE rule, grid-edge extension, validation rows only, raw (unshrunk) model. The calibrator is first fitted on the pool's validation rows, then after each S6 task. Evaluated method: pretrained residual TFT + EWC (raw and shrunk).

### Tests (fixed here)
Paired two-sided Wilcoxon over matched (order, seed) runs; against a deterministic baseline, the runs minus its single value; Holm correction within each family and seed set; the D21 replication rule decides (p_Holm < 0.05 in both seed sets, same sign, |difference| ≥ 0.05 pp in both = replicated; |difference| < 0.05 pp in both = practically equivalent; otherwise not replicated). Differences are a − b; negative favours a. Each experiment (E1, E2) is its own set of families.
- **Primary (one family):** (a) shrunk scratch model − linear extrapolation, final error; (b) shrunk pretrained model − linear extrapolation, final error; (c) raw pretrained model − raw scratch model (= D23), final error; (d) shrunk scratch model − conformal linear extrapolation, interval score at 90 %.
- **Secondary (one family):** shrunk − raw for the scratch and pretrained models (final error, forgetting); raw pretrained − raw scratch (forgetting); shrunk pretrained − shrunk scratch (final error); shrunk pretrained − conformal extrapolation (interval score 90 %); both shrunk models − conformal extrapolation (interval score 80 %).
- **Per horizon, E2 only (two families of six):** each shrunk model − linear extrapolation, RMSE at h = 1…6 after the last step.
- **What counts as a positive result:** for Q1, (a) or (b) replicated in favour of the model in E1 or E2; for Q2, (c) replicated in favour of pretraining; for Q3, (d) replicated in favour of the model. A test that is practically equivalent is reported as such ("as accurate as", never "better"). Every outcome is reported, including null and negative ones; nothing is re-run to change one.
- Reported for every model: final average error (and per task and per horizon), forgetting, backward transfer, plasticity, interval score, coverage and width (90 % and 80 %, per task), the shrinkage factors after every task, the pool sizes and pretraining times, and training time.
- **Naming.** RC-TFT keeps its D24 meaning. The shrunk model is called "calibrated residual TFT + EWC" until the results are known; any new name is chosen afterwards and the paper says so.

### Compute budget and run plan
One night on the ASUS (`scripts/run_d25_night.py`, runbook `docs/D25_NIGHT_RUN.md`), in the pinned environment (`requirements.txt`; the D24 run did not use it). Ten parts, at most six processes at a time (the D24 attempt with eight workers ran out of memory twice): `pre_tune` (E2, E1) first, the four `scratch_*` parts alongside, then `pre_main` and `pre_replicate` of each experiment once its `pre_tune` has finished. Estimates from recorded times (D21 TFT pretraining 2–6 min on the E1-size pool; D23 TFT + EWC streams 39 s (E1) and 56 s (E2)): scratch parts 35–50 min each, `pre_tune` 1–2 h, `pre_main`/`pre_replicate` 1–1.5 h each; about 3.5–4 h in all. No part starts after the deadline; parts that do not finish are reported as not run.

### Not done, with reasons
- **Quantile output head for the TFT** (considered): it changes the training loss, so the point forecast would no longer be the D23 model and Q1 and Q2 would be confounded with the loss change. Conformal intervals give Q3 without touching training.
- **Reuse-class decisions from the intervals:** in S6 P1 every test period is in class C3 (`docs/RESULTS.md`, key finding 10), so class decisions cannot be evaluated on this stream.
- **P2 (unseen condition), other backbones with shrinkage, new inputs, other CL penalties:** outside the one-night budget or already answered (D22, D24).

### Run log
- **14 Oct 2026. Code committed** after tests on synthetic data only (pytest temporary directories and the session scratch space; never real data): `src/calibration.py`, `scripts/run_d25.py`, `scripts/run_d25_night.py`, `scripts/summarise_d25.py`, `tests/test_d25.py`. The raw path of the calibrated stream equals `p3.run_stream` on a synthetic stream (tested), and a smoke run of all ten parts through the orchestrator, both collections and the summary passed. No D25 code has been run on S6 or the pool.
- **15–16 Oct 2026. Night run completed.** `run_d25_night.py --workers 6 --deadline 07:30` from `main` (`a59debb`, clean tree; uncommitted edits to `notebooks/colab_review2.ipynb` were stashed first, including a kernel-metadata re-save by the editor after the first stash; pre-flight passed), in the pinned environment: Python 3.11.15, numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, torch 2.14.1+cpu (the existing `.venv`, built on the same Python 3.11.15; `py -3.11` does not list that interpreter, so no new venv was created; `pip install -r requirements.txt` found every pin already installed; 180 tests passed, none failed). Started 00:02:47 IST; smoke passed for all 10 parts (00:32:04); paper run 00:32:04–03:33:35 (longest part E2 `pre_tune`, 108 min); collection 03:33:35–03:33:51. No restarts. All 10 parts ran, none failed, none was stopped by the deadline (`parts_not_run` empty in both files, `git_dirty` false). D23 reproduction check passed in E1 and E2 (100 raw scratch units each, max_abs_diff 0.0). The summary step (`summarise_d25.py`) wrote `data/cache/d25/d25_summary.md` and then exited with an error when printing it to the Windows console (`UnicodeEncodeError`, cp1252 cannot encode U+2212); the results files and the summary file are unaffected. Outputs: `results/forecasting_d25_e1.json`, `..._e2.json`.
- **15–16 Oct 2026. Rehearsal on a cloud machine (not reportable; recorded after the ASUS run above, which is the D25 run).** Commit `a59debb` (clean tree), Python 3.11.17, numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.17.1, torch 2.14.1+cpu (pinned versions; pre-flight passed without `--allow-unpinned`), 4 workers on 4 cores (Intel Xeon, 2.1 GHz). Data rebuilt from the original public archives with the repository's scripts: all 1,007 S6 characterisation files matched the committed SHA-256 manifest; the parse reproduced the authors' C/3 capacity exactly on all 843 checks; the pool downloads matched their registered MD5 values and gave exactly the D21 pool (36,711 samples, 369 cells, splits 21,174 / 6,439 / 6,942); S6 gave the D23 sample counts (2,409 / 4,638). `pytest`: 180 passed, none skipped. Smoke passed for all ten parts; all ten paper parts finished. A container restart stopped the run once (after e1 `pre_tune` and the four scratch parts); it resumed from the checkpoints with `--skip-smoke`. Part times: scratch 16–18 min (E1) and 27 min (E2); `pre_tune` 51 min (E1); `pre_main`/`pre_replicate` 35 min (E1) and 64–73 min (E2).
  **Collection stopped at the pre-registered reproduction check in both experiments**, so no D25 results file was written. No raw scratch unit equals the committed D23 unit to 1e-9 pp. E2: all 100 within 1e-4 pp (median 2.7e-7, largest 6.0e-5 pp). E1: 88 of 100 within 1e-4 pp (median 1.8e-6, largest 0.015 pp). The code, data, seeds and library versions are those of D23, so the differences are attributed to the CPU (floating-point results differing at the last bits and amplified by training and early stopping); D24's units reproduced to about 1e-14 on the ASUS. Under D25 this run is not reported. It served as a full rehearsal of the code path at paper scale. The tolerance is not changed. Disclosure: while checking reproduction during the run, the per-unit raw and shrunk final errors and interval scores of the E2 scratch units were printed and seen; no aggregate, test or verdict was computed or looked at.

## D26. Unseen usage conditions (P2) for the residual TFT + EWC, stratified by origin history (pre-registered)

**Status:** Written 16 Oct 2026 on the instruction of T. Sitaraman, **after** the D25 results; committed as a draft (`93cdd33`), then its open questions were settled on 17 Oct (below). The commit that records the settled questions is the binding pre-registration; it precedes any D26 code. One amendment and one clarification were made before any code ran on real data (run log, 17 Oct). Code and runbook: `scripts/run_d26.py`, `scripts/run_d26_night.py`, `scripts/summarise_d26.py`, `tests/test_d26.py`, `docs/D26_NIGHT_RUN.md`. Results will go to `results/forecasting_d26_e1.json` and `results/forecasting_d26_e2.json`.

**Provenance (must be stated).** Written after every earlier result, including D25 (`docs/RESULTS.md`, D25). The motivation is post hoc:
- On the S6 continual stream (P1), no learned model beats linear extrapolation (D23–D25); the shrunk residual TFT + EWC is practically equivalent to it at every horizon (D25).
- On unseen usage conditions (P2, D20, F1 inputs, horizons 1–3), learned models beat extrapolation by a wide margin (gradient boosting 0.710, TFT 0.868, linear extrapolation 1.372 pp). P2 has not been run for any residual model (D23–D25: not in budget).
- **Known confound in that comparison.** D20 P2 tests every sample of the held-out case, including origins early in a cell's life where the fade rate is missing and linear extrapolation falls back to persistence (ΔSOH = 0). Part of the learned models' P2 advantage may come from these origins. D26 therefore reports and tests the two kinds of origin separately.
- The D25 conformal intervals, calibrated per task on that task's validation rows, under-cover (0.71–0.82 at nominal 90 %). In P2 the calibration rows come from six conditions pooled, which tests whether the intervals hold on a condition never seen.

The P2 numbers of D20 are known. No P2 quantity for the residual, shrunk or continual models, and no per-stratum P2 result for any method, has been computed.

**Questions.**
- Q1: On an unseen usage condition, is the shrunk residual TFT + EWC more accurate than linear extrapolation **at origins where extrapolation has a fade rate**?
- Q2: Is it more accurate than gradient boosting (the best D20 P2 model)?
- Q3: Do its conformal intervals, calibrated on the other six conditions, reach the nominal coverage on the unseen one, and are they better than extrapolation's?

### Setup (fixed here)
- **Data, inputs, splits, horizons, label:** as D23–D25: S6 samples (`p3.s6_samples`), the D21 shared inputs, the D18 purged per-cell splits, E1 = horizons 1–3, E2 = horizons 1–6; SOH relative to the first valid check.
- **Folds:** leave one test case out (seven folds, D20). The held-out case is never used for training, early stopping, tuning, shrinkage or calibration; all its samples are test samples.
- **Origin strata.** H+ = origins at which the D23 linear-extrapolation baseline has a fade rate, as computed by the committed baseline code (no new definition); H− = the other origins, where it equals persistence. The code records each test sample's stratum. The H+/H− counts per fold are written to the results file (no model results are needed to compute them) and are reported with every result.
- **Proposed method (continual):** the D23 residual TFT + EWC streamed over the six training cases, then evaluated on the held-out case (zero-shot to a new condition). Orders: for evaluation order *o* (the 10 D23 orders), the fold's stream is order *o* with the held-out case removed; seeds 0–4 (main) and 5–9 (replicate). Each case's training rows are its train and test splits, and its validation split is used for early stopping (the D18 P2 rule). **Hyperparameters: the D23 values (lr 3e-3; λ 1 in E1, 1000 in E2), with no new tuning**, so nothing is selected with D26 data. **Disclosure (must be stated with every D26 result):** the D23 tuning streams used the validation rows of all seven cases, so each held-out case's validation rows took part in choosing these two values; in P2 those rows are test samples. Nested per-fold tuning was considered and not chosen (about seven times the tuning cost, two to three extra nights).
- **Shrinkage and intervals (amended 17 Oct, before any code; see run log):** the D25 calibrator code (`src/calibration.py`, unchanged), fitted **once, after the stream, on the validation rows of all six training cases pooled**, and applied to the held-out case. Linear extrapolation gets conformal intervals from the same rows in the same way.
- **Run-level metric:** for a matched (order, seed) run, the P2 error is the mean over the seven folds of the held-out test RMSE (pp), computed on all origins, on H+ and on H−. Per-fold values are reported as well.
- **Comparators** (each fitted per fold on the six training cases, same rows): persistence; linear extrapolation; gradient boosting and ridge on the D21 shared inputs (single fits, as in D23); the joint residual TFT (all six cases at once, D23 lr; upper bound for the continual model, seeds as above); the raw (unshrunk) residual TFT + EWC from the same runs.
- **Pinned environment** (`requirements.txt`); pre-flight refuses anything else unless `--allow-unpinned` is recorded in the run log.

### Tests (fixed here)
D25 rules: paired two-sided Wilcoxon over matched (order, seed) runs; against a single-fit baseline, the runs minus its single value; Holm correction within each family and seed set; the D21 replication rule (p_Holm < 0.05 in both seed sets, same sign, |difference| ≥ 0.05 pp in both = replicated; |difference| < 0.05 pp in both = practically equivalent; otherwise not replicated). Differences are a − b; negative favours a. E1 and E2 are separate sets of families.
- **Primary (one family):** (a) shrunk residual TFT + EWC − linear extrapolation, P2 error on **H+**; (b) the same on **all origins**; (c) shrunk residual TFT + EWC − gradient boosting, P2 error on all origins; (d) shrunk residual TFT + EWC − conformal linear extrapolation, interval score at 90 %, all origins.
- **Secondary (one family):** (a) and (c) on H−; shrunk − raw; raw continual − joint residual TFT (both unshrunk; clarified 17 Oct); interval score at 80 %.
- **Coverage (descriptive, no test):** mean 90 % and 80 % coverage over runs, per fold and pooled, for the model and for extrapolation. **Coverage is called adequate if the mean 90 % coverage pooled over folds is at least 0.85 in both seed sets**; this threshold is fixed here.
- **What counts as a positive result:** Q1: (a) replicated in favour of the model. A win on all origins (b) or on H− alone does **not** count for Q1, because on H− extrapolation is persistence. Q2: (c) replicated in favour. Q3: (d) replicated in favour **and** coverage adequate. Practical equivalence is reported as "as accurate as", never "better". Every outcome is reported, including null and negative ones; nothing is re-run to change one.
- **Power.** The tests are over training randomness (orders, seeds) with the seven conditions fixed. With seven folds, any claim about *new* conditions in general rests on seven conditions from one laboratory; the per-fold results, including the hardest fold, are reported beside every pooled number.

### Compute budget (estimate; checked on synthetic data 17 Oct, see run log)
Per experiment: 7 folds × 10 orders × 10 seeds = 700 continual streams of six tasks (D25 seven-task scratch streams took 41–54 s each), 70 joint fits, and the single-fit baselines. That is roughly 8–10 h of single-process CPU per experiment, about 1.5–2 h with six workers, so about 3.5–4.5 h for E1 and E2 together. One night on the ASUS, six workers, with a runbook `docs/D26_NIGHT_RUN.md` in the D25 format. Parts that do not finish by the deadline are reported as not run.

### Not done, with reasons
- **Residual pretraining on the pool:** negative or null in D25; not repeated until the pool and S6 share one definition of linear extrapolation.
- **Direct TFT + EWC in P2:** it would double the compute (beyond one night); residual against direct was answered in P1 (D23, replicated in E1 and E2).
- **The D24 backbones (BiLSTM, TCN, Transformer) and SA-EWC in P2:** budget; D24 answered the same-size backbone question in P1. Planned only if the residual TFT shows a P2 result worth comparing.
- **A pooled or cumulative conformal calibrator on the P1 stream:** it would be a fix evaluated on the same stream that showed the under-coverage (D25). Deferred to a decision with its own pre-registration, or to new data.
- **Reuse-class decisions:** in S6 P1 every test period is in class C3; P2 reuse-class agreement is reported descriptively (accuracy, macro-F1), as in D20, and not tested.

### Settled questions (17 Oct 2026, T. Sitaraman)
1. **Tuning:** the D23 values, no new tuning, with the disclosure above. Nested per-fold tuning is not run.
2. **Comparator inputs:** every D26 comparator (gradient boosting, ridge) uses the D21 shared inputs. The D20 P2 numbers (F1 inputs) stay in the record as history and are not mixed into D26 tables or tests.
3. **Coverage threshold:** mean 90 % coverage pooled over folds of at least 0.85 in both seed sets, as written above.

### Run log
- **16 Oct 2026. Drafted** after the D25 results and committed as a draft with the D25 section of `docs/RESULTS.md`; open questions not yet settled; no D26 code written or run.
- **17 Oct 2026. Open questions settled** (tuning: D23 values with disclosure; comparators on the D21 inputs; coverage threshold 0.85). This commit is the binding pre-registration. No D26 code written or run.
- **17 Oct 2026. Amendment and clarification, before any D26 code ran on real data** (T. Sitaraman). (1) The committed setup said the D25 calibrator is "updated after each of the six tasks as in D25" and the factors and half-widths "in force after the sixth task" are applied; that calibrator refits from the current task's validation rows only, so the held-out case would have been calibrated on the sixth task's rows alone, contradicting the provenance ("calibration rows come from six conditions pooled"). Resolved in favour of the pooled rule: one update, after the stream, on the validation rows of all six cases. (2) The secondary test "continual − joint residual TFT" compares the raw (unshrunk) continual model with the joint model.
- **17 Oct 2026. Code committed** after tests on synthetic data only (pytest temporary directories and the session scratch space; never real data): `scripts/run_d26.py`, `scripts/run_d26_night.py`, `scripts/summarise_d26.py`, `tests/test_d26.py`, runbook `docs/D26_NIGHT_RUN.md`; `scripts/run_d25_night.py` now takes its runner, data files and decision name from module settings so that D26 reuses its pre-flight and scheduler (D25 behaviour unchanged; D25 tests pass). Nothing under `src/` changed. Implementation choices fixed with the code: a part is one (seed set, fold); the joint model is fitted once per (fold, seed) and paired with every order of that seed; pooled coverage is computed over all held-out samples of all folds of a run; runs, comparators and tests use only the folds that finished in both seed sets (`folds_in_tests`), so every pair covers the same folds; `summarise_d26.py` writes UTF-8 to the console (the D25 summary step failed on the Windows code page). **Compute check (synthetic data of S6 size, 7 cases, 20 cells, 42 checks per cell; paper profile, single-threaded):** one continual unit 34 s (E1) and 55 s (E2), one joint fit 47 s and 94 s; about 20 h of CPU for both experiments, about 3.4 h with six workers plus the smoke stage, within the registered estimate. No D26 code has been run on S6.
- **17–18 Oct 2026. Night run completed.** `run_d26_night.py --workers 6 --deadline 07:30` from `main` (`9866d7a`, clean tree; pre-flight passed) in the pinned environment (the existing `.venv`): Python 3.11.15, numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.17.1, torch 2.14.1+cpu. Two earlier invocations at 23:01 IST with the system Python 3.12.10 (unpinned libraries) were refused by pre-flight before anything ran; `--allow-unpinned` was not used. Started 23:03:24 IST; smoke passed for all 28 parts (23:05:46); paper run 23:05:46–02:27:43 (parts: E2 42.5–53.0 min, E1 23.5–38.9 min); collection 02:27:43–02:28:00; summary written 02:28:01. No restarts. All 28 parts ran, none failed, none was stopped by the deadline (`parts_not_run` empty in both files, `git_dirty` false, `folds_in_tests` all seven cases). Outputs: `results/forecasting_d26_e1.json`, `..._e2.json`.

## D27. Forecast intervals for unseen usage conditions: leave-one-condition-out conformal calibration (P2) (pre-registered)

**Status:** Written 19 Oct 2026 on the instruction of T. Sitaraman, **after** the D26 results (`docs/RESULTS.md`, D26). This commit is the binding pre-registration; it precedes any D27 code. Code: `scripts/run_d27.py`, `tests/test_d27.py`. Results will go to `results/forecasting_d27_e1.json` and `results/forecasting_d27_e2.json`.

**Provenance (must be stated).** The motivation is post hoc, from D26:
- On unseen usage conditions (P2), gradient boosting (single fit on the D21 shared inputs) is the most accurate model (0.535 pp E1, 1.051 pp E2), better than the residual TFT + EWC (replicated, D26 Q2).
- Every D26 interval under-covers on the held-out condition: the shrunk residual TFT + EWC 0.66–0.68 and conformal linear extrapolation 0.74–0.76 at nominal 90 %, both 0.38–0.44 on case7. Their calibration rows were the **validation rows** (mid-life origins) of the six training conditions, and for the TFT those conditions had also been used for training. The test covers **every origin** of a condition used for neither. D26 does not separate the two differences.
- A forecast that feeds a reuse decision for a new battery population needs an interval that holds on that population. Split-conformal intervals are valid when calibration and test residuals are exchangeable; across conditions this requires the calibration residuals to come from conditions the forecaster was **not** fitted on.

Known before this pre-registration: the D26 point errors of every comparator, per fold and stratum, and the D26 interval metrics of the shrunk TFT and of extrapolation calibrated on validation rows. **Not computed for any rule below:** gradient-boosting intervals of any kind, and extrapolation intervals calibrated on all origins.

**Questions.**
- Q1: With leave-one-condition-out (LOCO) calibration, do gradient-boosting 90 % intervals reach adequate coverage on an unseen condition?
- Q2: Are those intervals better (90 % interval score) than extrapolation's intervals calibrated on all origins of the same six conditions?
- Q3 (mechanism, descriptive): does calibrating extrapolation on all origins of the six conditions, instead of their validation rows (the D26 rule), remove its D26 under-coverage? Extrapolation has no fitted parameters, so its D26 residuals were already out-of-sample: any change isolates the effect of which origins are used for calibration.

### Setup (fixed here)
- **Data, inputs, horizons, label, folds:** as D26 (S6 samples, D21 shared inputs, E1 = horizons 1–3, E2 = 1–6, SOH relative to the first valid check; leave one of the seven test cases out). All samples of the held-out case are test samples and are never used for fitting or calibration.
- **Forecasters (no tuning; fixed settings):** gradient boosting exactly as the D26 comparator (`FC.baseline_predictions`, `gbm_F1` on the observed shared inputs, D18 settings, seed 0; histogram gradient boosting without early stopping, so deterministic), fitted on all samples of the six training cases; and linear extrapolation (the D23 baseline, per cycle).
- **Calibration rules** (per horizon; the D25 `conformal_halfwidth` from `src/calibration.py`, unchanged, at 90 % and 80 %; interval = forecast ± q_h):
  - **GBM-LOCO (proposed):** for each of the six training cases j, gradient boosting is fitted on all samples of the other five training cases with the same settings and predicts every sample of j; the absolute residuals of the six cases are pooled per horizon. The interval is centred on the forecast of the six-case fit.
  - **LIN-all:** extrapolation, residuals of every sample of the six training cases, pooled per horizon.
  - **LIN-val (D26 rule):** extrapolation, residuals of the validation rows (D18 split) of the six training cases, pooled per horizon. This reproduces the D26 extrapolation intervals.
  - **Secondary, Mondrian by origin stratum:** GBM-LOCO and LIN-all with half-widths per (horizon, stratum), strata H+ and H− as D26.
- **Metrics:** per fold, coverage, mean width and mean interval score (Gneiting and Raftery) at 90 % and 80 %, on all origins, H+ and H−. Pooled coverage = covered held-out samples over all held-out samples of the seven folds (as D26); interval score and width = mean over folds of the per-fold mean (as D26).
- **Reuse decision support (descriptive, no test):** a held-out sample is *decided* if the lower and upper ends of its 90 % interval for the future SOH (origin SOH + forecast ± q) fall in the same D20 reuse class; reported are the share decided and the class agreement among decided samples, per rule.
- **Reproduction checks (stop conditions, as D25):** before any D27 quantity is reported, (i) the six-case gradient-boosting fit must reproduce the D26 per-fold RMSE of `gbm_shared` (all origins) and (ii) LIN-val must reproduce the D26 extrapolation interval score and pooled coverage at 90 % and 80 %, both to 1e-9. If either fails, no results file is written and the failure is reported.
- **Pinned environment** (`requirements.txt`) and a clean tree; the script refuses otherwise unless an override is recorded in the run log.

### Tests (fixed here)
The forecasters are deterministic, so there are no seed sets and the D21 replication rule over seeds does not apply; the unit of comparison is the held-out condition (seven folds). Requiring a result in both E1 and E2 replaces the replication rule; E1 and E2 are not independent (E2 contains E1's horizons), and this is stated with every result.
- **Q1, positive** if the pooled 90 % coverage of GBM-LOCO is at least 0.85 (the D26 threshold) in both E1 and E2. Per-fold coverage, its minimum and the case reaching it are reported with it.
- **Q2, positive** if the per-fold 90 % interval score of GBM-LOCO is lower than that of LIN-all with a two-sided exact Wilcoxon signed-rank p < 0.05 over the seven folds **and** a mean difference of at least 0.05 pp, in both E1 and E2. With seven folds the smallest attainable p is 0.016. Negative if the same holds in the other direction; otherwise not established.
- **Q3 (descriptive):** pooled and per-fold 90 % coverage of LIN-all against LIN-val. Reading fixed here: if LIN-all reaches 0.85 in both experiments and LIN-val does not, the D26 under-coverage of extrapolation is attributed to calibrating on validation-row origins only; if neither does, to the condition shift itself.
- **Secondary (descriptive):** the Mondrian rules; 80 % results; GBM-LOCO against the D26 shrunk TFT intervals (per fold; different runs, so no paired test); the reuse decision-support numbers.
- Every outcome is reported, including null and negative ones; nothing is re-run to change one.

### Compute
Per experiment, 7 × (1 + 6) = 49 gradient-boosting fits on at most about 4,000 rows, a few minutes on one CPU thread in all. Run on the ASUS with `python scripts/run_d27.py --exp e1` and `--exp e2`.

### Not done, with reasons
- **LOCO calibration for the TFT models.** The continual residual TFT + EWC would need six inner streams per (fold, order, seed), about six D26 nights; the joint residual TFT about one night. Neither is the most accurate P2 model (D26). Planned only if D27 shows that LOCO calibration fixes coverage.
- **A residual (correction-to-extrapolation) gradient boosting:** a new point model selected after seeing D26; not part of an interval question.
- **Tuning gradient boosting or weighting the calibration residuals by case:** fixed settings keep D27 free of choices made on D27 data; each case enters the pooled quantile with its sample count, as in D26.

### Run log
- **19 Oct 2026. Pre-registered** after the D26 results; no D27 code written or run.
- **19 Oct 2026. Code committed** (`494b36f`) after a test on synthetic data only (pytest temporary directory): `scripts/run_d27.py`, `tests/test_d27.py`. Nothing under `src/` changed. Implementation choices fixed with the code: the S6 frame, features and cases are built as in D26 (`p3.s6_samples`, `p3.SHARED_FEATURES`); each fit uses the row order of the D26 comparator (`FC.leave_one_out_task`, train and test splits then validation split); validation rows for LIN-val come from `FC.assign_split` of each training case, as in D26; the Q2 test is `scipy.stats.wilcoxon(..., method="exact")`, two-sided. Full test suite: 185 passed with a UTF-8 console; without it, the D25 summary test fails on the Windows code page (the known D25 summary console error; no results affected).
- **19 Oct 2026. Run** on the ASUS from `494b36f` (clean tree; pre-flight passed; no override) in the pinned environment (the existing `.venv`: Python 3.11.15, numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.17.1): `python scripts/run_d27.py --exp e1`, then `--exp e2`, each run once. Reproduction checks passed in both experiments (max abs diff 0: the D26 gradient-boosting per-fold RMSE and the D26 extrapolation interval metrics are reproduced exactly). No half-width was undersized. Outputs: `results/forecasting_d27_e1.json`, `..._e2.json`.

## D28. Continual calibration and refit on an unseen usage condition as its checks arrive (P2) (pre-registered)

**Status:** Written 21 Oct 2026 on the instruction of T. Sitaraman, **after** the D27 results (`docs/RESULTS.md`, D27). This commit is the binding pre-registration; it precedes any D28 code. Code: `scripts/run_d28.py`, `tests/test_d28.py`. Results will go to `results/forecasting_d28_e1.json` and `results/forecasting_d28_e2.json`.

**Provenance (must be stated).** The motivation is post hoc, from D27 and a literature search made after it:
- Gradient boosting with leave-one-condition-out (LOCO) calibration covers 0.90 pooled over the seven S6 conditions, but only 0.48 (E1) and 0.38 (E2) on case7 (100 % depth of discharge, outside the range of the other conditions); its interval score there is five to nine times that of any other fold. The pooled guarantee hides a population for which neither forecast nor interval can be trusted.
- In deployment the checks of a new population arrive over time. Online conformal methods keep coverage under distribution shift by updating the interval from the errors observed so far: adaptive conformal inference (Gibbs and Candès, NeurIPS 2021) and its quantile-tracking form (Angelopoulos, Candès and Tibshirani, "Conformal PID control for time series prediction", NeurIPS 2023). The search found no battery application of either.
- D23–D27 found that continual learning of network weights did not pay off when earlier data can be stored. D28 tests continual updating of what is cheap and cannot forget: the calibration of the interval and a refit of the forecaster with the new population's observed targets.

Known before this pre-registration: every D27 quantity (per fold and pooled) and the D26 point errors of every comparator. **Not computed for any arm below except STATIC:** any online interval, any refit with held-out-case samples.

**Questions.**
- Q1 (primary): With continual calibration and refit, do 90 % intervals on case7 reach adequate coverage?
- Q2: Do they hold on every condition, not only pooled?
- Q3: Does refitting with the new condition's observed targets lower the point error?
- Q4: Is the interval score better than that of the D27 static intervals?

### Setup (fixed here)
- **Data, inputs, horizons, label, folds, forecaster:** as D27 (S6 samples, D21 shared inputs, E1 = horizons 1–3, E2 = 1–6, SOH relative to the first valid check; leave one of the seven test cases out; gradient boosting exactly as the D26/D27 comparator, fixed settings, seed 0, deterministic). Every sample of the held-out case is a test sample and is scored, as in D27.
- **Arrival of the held-out case (no look-ahead).** A sample is *issued* at its origin check time t (`Cycle_Date`) and *matures* at its target check time m = t + `days_to_target`. When a sample is issued, the only held-out-case samples known are those with m ≤ t (their targets observed). Samples are processed in order of issue time; ties keep the frame order.
- **Arms:**
  - **STATIC (D27 GBM-LOCO):** the six-case fit and the LOCO half-widths q0_h (per horizon and level). Must reproduce D27.
  - **TRACK (continual calibration):** centre as STATIC. Per horizon h and level 1 − a, the half-width starts at q0_h; whenever a held-out-case sample of horizon h matures (in order of m, ties in issue order), q_h ← max(0, q_h + η_h (e − a)), where e = 1 if that sample's target lay outside the interval issued for it (quantile tracking, Angelopoulos et al. 2023). Step η_h = c · q0_h.
  - **REFIT (continual refit):** at issue time t, gradient boosting with the same settings fitted on the six training cases (D27 row order) followed by the held-out-case samples matured by t; refitted whenever that set changes; with none matured, the STATIC forecast. Half-widths static q0_h.
  - **REFIT+TRACK (proposed):** REFIT centre and TRACK half-widths (errors e measured on the REFIT intervals issued).
- **Step size c (chosen without the held-out case).** Grid c ∈ {0.01, 0.02, 0.05, 0.1, 0.2, 0.5}. For each fold, each of the six training cases j is treated as a pseudo-held-out case: centre = D27's inner LOCO forecast for j (fitted on the other five training cases), q0 = half-widths from the LOCO residuals of the other five training cases, and TRACK is run over j's samples. The c with the lowest mean 90 % interval score over the six pseudo-folds is used for that fold (ties: smaller c), at both levels and for both TRACK arms.
- **Metrics:** as D27: per fold coverage, width and interval score at 90 % and 80 %, on all origins, H+ and H−; pooled coverage over the seven folds; RMSE per fold; reuse decision support (share of 90 % intervals within one reuse class, class agreement among them).
- **Reproduction check (stop condition):** STATIC must reproduce the D27 GBM-LOCO per-fold coverage and interval score at 90 % and 80 %, and the D27 per-fold gradient-boosting RMSE, to 1e-9. If not, no results file is written.
- **Pinned environment** and a clean tree, as D27.

### Tests (fixed here)
Deterministic forecasters: requiring a result in both E1 and E2 replaces the seed replication rule (E1 and E2 are not independent; stated with every result). The unit of comparison is the held-out condition (seven folds); two-sided exact Wilcoxon signed-rank tests, smallest attainable p 0.016.
- **Q1, positive** if the 90 % coverage of REFIT+TRACK on case7 is at least 0.80 in both E1 and E2.
- **Q2, positive** if REFIT+TRACK reaches pooled 90 % coverage of at least 0.85 and a per-condition minimum of at least 0.80, in both E1 and E2.
- **Q3, positive** if the per-fold RMSE of REFIT is lower than that of STATIC with p < 0.05 and a mean difference of at least 0.05 pp, in both E1 and E2; negative if the same holds in the other direction; otherwise not established. Case7 RMSE is reported with it.
- **Q4, positive** if the per-fold 90 % interval score of REFIT+TRACK is lower than that of STATIC with p < 0.05 and a mean difference of at least 0.05 pp, in both E1 and E2; negative and not established as for Q3.
- **Secondary (descriptive):** TRACK and REFIT alone (attribution), 80 % results, H+/H− strata, the selected c per fold, reuse decision support.
- Every outcome is reported, including null and negative ones; nothing is re-run to change one.

### Compute
Per experiment and fold, one gradient-boosting refit per distinct set of matured samples (at most about 200 per case) plus the D27 fits; tens of minutes on one CPU thread in all. Run with `python scripts/run_d28.py --exp e1` and `--exp e2`.

### Not done, with reasons
- **The TFT models:** not the most accurate P2 model (D26), and an online refit of a continual TFT at every check would take nights; whether continual calibration and refit work is answered by the deterministic forecaster.
- **Adaptive conformal inference on the quantile level (Gibbs and Candès 2021):** with a few hundred calibration residuals per horizon, a level beyond the sample's range cannot widen the interval further; quantile tracking acts on the half-width directly.
- **Tuning gradient boosting:** fixed settings, as D27.

### Run log
- **21 Oct 2026. Pre-registered** after the D27 results; no D28 code written or run.
- **21 Oct 2026. Code committed** (`161c01f`) after a test on synthetic data only (pytest temporary directory): `scripts/run_d28.py`, `tests/test_d28.py`. Nothing under `src/` changed. Implementation choices fixed with the code: refits use the gradient boosting of `FC._gbm(0)` on the observed shared inputs (tested equal to the D27 comparator on the same rows); matured held-out samples are appended to the D27 fit rows in frame order; maturity is the origin check time plus `days_to_target`. Full test suite: 187 passed.
- **21 Oct 2026. Run** from `161c01f` (clean tree; pre-flight passed; no override) in the pinned environment: `python scripts/run_d28.py --exp e1`, then `--exp e2`, each run once. Reproduction check passed in both (max abs diff 0). Outputs: `results/forecasting_d28_e1.json`, `..._e2.json`. Results: `docs/RESULTS.md`, D28 (Q1 and Q2 negative, Q3 not established, Q4 positive).

## D29. Physics-structured Bayesian forecaster with a learned residual on unseen usage conditions (P2) (pre-registered)

**Status:** Written 21 Oct 2026 on the instruction of T. Sitaraman, **after** the D28 results (`docs/RESULTS.md`, D28). This commit is the binding pre-registration; it precedes any D29 code. Code: `scripts/run_d29.py`, `tests/test_d29.py`. Results will go to `results/forecasting_d29_e1.json` and `results/forecasting_d29_e2.json`.

**Provenance (must be stated).** Post hoc, from D23–D28 and an inspection of the S6 conditions made after D28:
- With case7 held out, every training condition runs at 70 % depth of discharge (DoD) and case7 at 100 %, so DoD is constant in training: its effect cannot be learned from S6, and a tree model cannot extrapolate in it. Case7 coverage stayed at 0.38–0.72 in D27–D28.
- EWC approximates sequential Bayesian updating (Laplace); on a model with few parameters the exact update is cheap and needs no stored data.
- Inspected before writing: the S6 sample frame (first check at cycle 0 with SOH 100 for every cell; per-condition settings). Not computed: any quantity of the model below.

**Questions.**
- Q1 (primary): Do the HYBRID-LOCO 90 % intervals cover case7 adequately?
- Q2: Is HYBRID-LOCO more accurate than the D27 gradient boosting?
- Q3: Does HYBRID-LOCO have a better 90 % interval score than D28 REFIT+TRACK?
- Q4 (descriptive): Does the throughput clock matter on case7?
- Q5 (descriptive): Do the uncalibrated Bayesian intervals cover?

### Setup (fixed here)
- **Data, horizons, label, folds, arrival:** as D27/D28 (S6 samples, E1 = horizons 1–3, E2 = 1–6, leave one of the seven test cases out; every held-out sample scored).
- **Clock:** equivalent full cycles E = cycle × DoD (cycle = the check's cycle count; target cycle = origin cycle + `cycles_to_target`). Capacity fade's effect on throughput per cycle is ignored (stated limitation).
- **Ageing law (mean):** for cell b, SOH(E) = 100 − a_b · E^z, a semi-empirical power law of throughput as used for graphite–LFP cells. The exponent z is shared: chosen on the training conditions by profile least squares over z ∈ {0.30, 0.35, …, 1.50} (each training cell's a fitted by least squares on all its valid checks; total squared error minimised).
- **Hierarchical prior:** a_b ~ N(μ, τ²) with μ and τ² the mean and variance of the training cells' fitted a; observation noise σ² = pooled residual variance of those fits (n − 1 degrees of freedom per cell).
- **Sequential Bayesian update:** at an origin of a held-out cell, the posterior of a_b is the conjugate Gaussian update of the prior with that cell's checks at or before the origin cycle (no other held-out data). Forecast Δ̂ = −â (x_t − x_o), x = E^z; predictive variance v = (x_t − x_o)² Var(a) + 2σ².
- **Arms:**
  - **PHYS-BAYES:** physics centre, interval ± Φ⁻¹((1 + level)/2) √v (no calibration).
  - **PHYS-LOCO:** physics centre, per-horizon split-conformal half-widths from leave-one-condition-out residuals of the six training cases (each predicted with z, μ, τ², σ² refitted on the other five), `conformal_halfwidth` unchanged.
  - **HYBRID-LOCO (proposed):** physics centre plus a gradient boosting (D27 settings, shared inputs, seed 0) fitted on the six training cases to the residual Δ − physics centre; LOCO half-widths as PHYS-LOCO, each inner case predicted by physics and residual model refitted on the other five.
  - **PHYS-CYC-LOCO (Q4):** as PHYS-LOCO with E = cycle (DoD ignored).
- **Comparators (read from committed files, per fold):** D27 GBM-LOCO and D28 REFIT+TRACK.
- **Reproduction check (stop condition):** the D27 gradient boosting, recomputed in the script, must reproduce the D27 per-fold RMSE to 1e-9.
- **Metrics:** as D27/D28 (coverage, width, interval score at 90 % and 80 %; RMSE; reuse decision support).

### Tests (fixed here)
Deterministic: a result must hold in both E1 and E2 (not independent; stated). Seven folds, two-sided exact Wilcoxon, smallest p 0.016; a direction needs p < 0.05 and |mean difference| ≥ 0.05 pp.
- **Q1, positive** if HYBRID-LOCO 90 % coverage on case7 ≥ 0.80 in both E1 and E2.
- **Q2, positive** if HYBRID-LOCO per-fold RMSE is lower than D27 gradient boosting in both; negative in the other direction; otherwise not established.
- **Q3, positive** if HYBRID-LOCO per-fold IS90 is lower than D28 REFIT+TRACK in both; negative and not established as Q2.
- **Q4 (descriptive):** case7 RMSE of PHYS-LOCO against PHYS-CYC-LOCO.
- **Q5 (descriptive):** pooled and per-fold coverage of PHYS-BAYES.
- Every outcome is reported; nothing is re-run to change one.

### Not done, with reasons
- **A DoD prior from the first-life pool (S7–S9):** mapping their protocols onto a DoD stress factor is a separate study; here the throughput clock is the only physical DoD effect.
- **Refit and tracking on top of HYBRID:** the physics part already updates at every check; combining with D28 is left until HYBRID is known to help.
- **Foundation models (Chronos-2, TimesFM):** not installed in the pinned environment; a separate decision.

### Run log
- **21 Oct 2026. Pre-registered** after the D28 results; no D29 code written or run.
- **21 Oct 2026. Code committed** (`f2fdd73`) after a test on synthetic data only: `scripts/run_d29.py`, `tests/test_d29.py`. Nothing under `src/` changed. Implementation choices fixed with the code: a cell's checks are the union of its samples' origins and targets (cycle, SOH); the gradient boosting of the residual is `run_d28.gbm_fit_predict` (the D27 settings).
- **21 Oct 2026. Run** from `f2fdd73` in a separate clean worktree of that commit (the main working copy held notebook outputs from a local session, which the pre-flight correctly refused), with the same data directory and pinned environment: `python scripts/run_d29.py --exp e1`, then `--exp e2`, each run once (the first attempt in the main working copy stopped at pre-flight and wrote nothing). Reproduction check passed in both (max abs diff 0). Outputs: `results/forecasting_d29_e1.json`, `..._e2.json`. Results: `docs/RESULTS.md`, D29 (Q1 negative, Q2 not established, Q3 negative).

## D30. Full regression and classification metric set for the P1/P2 forecasts (supplementary reporting)

**Status:** Written 28 Oct 2026 on the instruction of T. Sitaraman, **after** all results through D29. **Post hoc:** these metrics are added after the RMSE results were known. They are supplementary descriptions of the same predictions, not new hypotheses; no method choice, test or claim in D18–D29 changes because of them, and no D30 metric enters a registered test. Code: `src/metrics.py`, `tests/test_metrics.py`, `scripts/run_d30.py`, `scripts/make_d30_outputs.py`, `notebooks/16_d30_metrics.ipynb`.

### Metrics (fixed here, before any D30 number is computed)
On the SOH level (relative to the first observed check), error e = forecast SOH − measured SOH, in pp:
- **Regression:** MSE, RMSE, MAE, MedAE, bias (mean e; positive = SOH overestimated), MAPE (%), NRMSE = RMSE / (max − min of measured SOH over the same samples), R² on SOH, and R²_Δ on the change since the origin (R² of forecast Δ against measured Δ). RMSE and MAE equal the existing `forecast_errors` values (checked in the tests).
- **Reuse class (D10 cut-points, C0–C3):** accuracy, balanced accuracy, macro precision / recall / specificity / F1 over the classes present in the measured labels (a measured class never forecast has precision 0 in the average), MCC (multi-class), Cohen's κ unweighted and linear-weighted, per-class values, confusion matrix.
- **Binary end-of-first-life decision** (SOH ≤ 80, i.e. C3, positive): accuracy, balanced accuracy, precision, recall, specificity, F1, MCC, κ, and AUC-ROC with −(forecast SOH) as the score (a ranking measure; no probability model).
- **Not computed: log loss.** Every forecaster gives point forecasts; converting the D27–D28 interval half-widths into class probabilities would need a distributional assumption that was never tested. Stated as a limitation.
- Undefined values are NaN (never 0): R² for a constant target, MCC / κ / AUC with one class, precision of a class never forecast (per class), recall of an absent class.

### Scope and aggregation
- **Forecasts:** those of paper Tables P1 and P2 (notebooks 12, 13; S1 and S6), recomputed with the committed hyperparameter selections (no re-tuning). P2 (all baselines, MLP and TFT, seeds 0–4) and the P1 baselines by default; the P1 continual methods with `--with-p1`.
- **Stop condition:** the recomputed RMSE per (P2 method, held-out group) and per P1 method must equal the committed values (`results/forecasting.json`, `results/forecasting_wheeler.json`) to 1e-6 pp; otherwise nothing is written. *Amended 29 Oct 2026 (deviation, see run log):* 1e-6 pp for the deterministic baselines, 0.05 pp for the neural methods (MLP, TFT, P1 continual methods).
- **Regression table:** mean over held-out groups (as Table P2, whose RMSE column it reproduces; `make_d30_outputs.py` checks this), s.d. over groups in the CSV.
- **Classification table:** all held-out samples pooled, per seed, then mean over seeds. A held-out group mostly covers one or two classes, so per-group class metrics are degenerate; per-group values stay in the results file. Measured class counts are reported with the table. In S1 the measured classes are expected to be nearly all C0, so S1 class metrics are descriptive at most.
- **Predictions** are written outside Git to `<data>/predictions/d30_<dataset>_p2.csv.gz`, so further metrics need no retraining.

### Run log
- **28 Oct 2026. Code committed** after unit tests on synthetic data only (every metric checked against scikit-learn). Not run on the real data: the session that wrote the code had no access to the datasets.
- **28–29 Oct 2026. First real-data runs stopped at the reproduction check** (Windows 11, Python 3.11.15, torch 2.14.1+cpu, pinned requirements, clean checkout of `ec8db7f`). Every deterministic baseline reproduced the committed P1 and P2 RMSE to 1e-6 pp on both datasets. The neural methods did not: S1, TFT on w10 −0.0091 pp (all other S1 neural values within 1e-4); S6, MLP on 6 of 7 conditions, up to +0.027 pp (case5), TFT on case7 +0.0002 pp. Over held-out groups the P2 means move from 0.2922 to 0.2907 (S1 TFT) and 0.8205 to 0.8257 (S6 MLP); no method changes rank. The committed values come from a different machine (29 Sep run, commit `d710cad`); the code paths of the models are unchanged since, so the cause is taken to be non-bit-identical neural training across platforms (not confirmed: the committed files hold no per-seed P2 values).
- **29 Oct 2026. Deviation (decided by T. Sitaraman after seeing the differences above).** The stop condition keeps 1e-6 pp for the deterministic baselines and uses 0.05 pp per (method, group) for the neural methods, the existing D21 practical-equivalence threshold, not a value fitted to the observed differences. The results file stores every difference (`reproduction_check.abs_diffs`). Consequence for reporting: the D30 neural-method metrics describe a re-run whose RMSE differs from Table P2 by the amounts above; the paper states this next to the D30 tables.

## D31. Online adaptation of the joint residual TFT with EWC on an unseen usage condition as its checks arrive (P2) (pre-registered)

**Status:** Written 31 Oct 2026 on the instruction of T. Sitaraman, **after** every D23–D30 result. The commit that adds this entry is the binding pre-registration; it precedes any D31 code. Code: `scripts/run_d31.py`, `tests/test_d31.py` (to be written). Results will go to `results/forecasting_d31_e1.json` and `results/forecasting_d31_e2.json`.

**Provenance (must be stated).** Post hoc, from D26–D29 and a requirement from the supervising faculty that the proposed model be a deep-learning model:
- On unseen S6 conditions (P2) the most accurate method is gradient boosting refitted with the new condition's measured targets as they arrive (D28 REFIT: 0.458 pp E1, 0.898 pp E2; static gradient boosting 0.535 / 1.051 pp). D28 adapted only gradient boosting; its "Not done" list excluded the TFT on compute grounds.
- The joint residual TFT (fitted once on the six training conditions, D26) is close to static gradient boosting (0.57 / 0.58 pp E1, 1.21 / 1.18 pp E2, main / replicate), and better than the continual (EWC-streamed) residual TFT by 0.32–0.44 pp (replicated). No TFT has been adapted on the held-out condition.
- D29 showed that a physics-structured centre (power law of throughput) does not improve on gradient boosting; with case7 held out, depth of discharge is constant in training, so no model can learn its effect from S6. D31 does not try to.
- **Deployment setting that motivates EWC here.** A model trained on the six conditions is shipped to a new battery population whose checks arrive over time; the training conditions' data are not available at the deployment site (only the trained weights and the Fisher information computed before shipping). Fine-tuning on the new population alone may forget the training conditions; EWC anchored at the shipped weights limits that without the stored data. This is the setting D23–D26 lacked (there every earlier group could be stored).

Known before this pre-registration: every D26–D30 quantity, including the D26 joint residual TFT per fold and seed and the D28 per-fold RMSE of every arm. **Not computed for any arm below except JOINT-STATIC:** any TFT forecast fitted with held-out-case samples, any forgetting measurement for an adapted TFT.

**Questions.**
- Q1 (primary): Is the online-adapted residual TFT + EWC more accurate than gradient boosting refitted online (D28 REFIT)?
- Q2: Does online adaptation make the residual TFT more accurate than without it (JOINT-STATIC)?
- Q3: Does EWC, compared with plain fine-tuning, reduce forgetting of the training conditions, and at what cost on the new one?
- Q4 (descriptive): the same comparison against static gradient boosting (D27/D28 STATIC), per fold and on case7.

### Setup (fixed here)
- **Data, inputs, horizons, label, folds:** as D26–D28: S6 samples (`p3.s6_samples`), the D21 shared inputs, the residual target on linear extrapolation (`p3.residual_target`), E1 = horizons 1–3, E2 = 1–6, SOH relative to the first valid check; leave one of the seven test cases out. Every sample of the held-out case is a test sample and is scored, as in D27/D28.
- **Base model (JOINT-STATIC):** the D26 joint residual TFT, `p3.build_strategy("tft_joint", ...)`, fitted on the six training cases with the D26 training tasks (train = train and test splits, early stopping on the validation split), the D23 hyperparameters (lr 3e-3, 200 epochs, `run_p3.TFT`) and the D26 seeds: 0–4 (main) and 5–9 (replicate). Raw (unshrunk) forecasts; D26 found shrinkage and the raw correction practically equivalent on P2. The D26 cache may be reused; the reproduction check below decides.
- **Anchor (computed before the held-out case is seen):** θ* = the base model's weights; F = the diagonal empirical Fisher of the base model on up to 500 seeded rows of the pooled training rows of the six training cases, computed as by the existing EWC strategy (`fisher_samples` 500, seed rule unchanged). Only θ*, F and the input scaling are kept after this point; the training cases' rows are not used again for fitting.
- **Arrival of the held-out case (no look-ahead):** exactly D28 (a sample is issued at its origin check time and matures at origin time + `days_to_target`; when a sample is issued, only held-out samples with maturity at or before the issue time are known; ties keep frame order). The D28 arrival code is reused, not rewritten. Inputs of the held-out case's history window (inputs only, never labels) are available as in D26.
- **Update rule (both adapted arms):** whenever the set of matured held-out samples grows, the model is fine-tuned **from its current weights** for E_ft epochs with Adam at the D23 lr on **all** matured held-out samples (residual target, batch settings unchanged), with no early stopping (no held-out validation rows exist without look-ahead). A sample issued at time t is forecast by the model updated with every sample matured by t; with none matured, by the base model. Input scaling stays frozen at the base model's.
- **Arms:**
  - **JOINT-STATIC:** the base model, never updated. Must reproduce D26.
  - **FT-ONLINE:** update rule with loss = MSE on the matured held-out samples.
  - **EWC-ONLINE (proposed):** update rule with loss = MSE + (λ/2) Σ_i F_i (θ_i − θ*_i)².
- **Selection of E_ft and λ, without the held-out case:** for each fold, seed 0 only, each of the six training cases j is a pseudo-held-out case: a joint residual TFT is fitted on the other five (same settings), its anchor computed on their rows, and the arms are run over j's samples with the arrival rule. Grids: E_ft in {5, 20}; λ in {0.1, 1, 10} × λ_D23 (λ_D23 = 1 in E1, 1000 in E2). FT-ONLINE takes the E_ft, and EWC-ONLINE the (E_ft, λ), with the lowest mean pseudo-held-out RMSE over the six cases (ties: fewer epochs, then larger λ). The selected values are written to the results file per fold. **Disclosure (as D26):** lr and λ_D23 come from D23 tuning streams that used the validation rows of all seven cases.
- **Metrics:** per fold and run, held-out RMSE (pp) on all origins, H+ and H−, and per horizon; pooled over folds as D26 (mean over folds). **Forgetting:** RMSE on the validation rows of the six training cases (D18 split, pooled) of the base model and of each adapted arm after its last update, and their difference. Reuse-class agreement of the held-out forecasts (descriptive, D20 classes).
- **Comparators (read from committed files, per fold):** D28 REFIT and STATIC RMSE (`results/forecasting_d28_e*.json`, all origins). Gradient boosting is not refitted in D31.
- **Reproduction check (stop condition):** JOINT-STATIC per (fold, seed) RMSE on all origins must equal the D26 joint values: to 1e-9 pp if run on the D26 platform and pinned environment, otherwise within the 0.05 pp neural tolerance of the D30 deviation, with every difference stored. The D28 comparator values are copied, and their means over folds must equal the D28 table (0.458 / 0.898 REFIT, 0.535 / 1.051 STATIC). If either fails, no results file is written.
- **Pinned environment** (`requirements.txt`) and a clean tree; pre-flight refuses otherwise unless an override is recorded in the run log.

### Tests (fixed here)
Differences are a − b; negative favours a. E1 and E2 are not independent (stated with every result).
- **Unit of comparison for Q1, Q2 and Q4: the held-out condition.** For each seed set, the per-fold value of a TFT arm is its mean RMSE over that set's five seeds. Two-sided exact Wilcoxon signed-rank over the seven folds (smallest attainable p 0.016).
  - **Q1, positive** if EWC-ONLINE − D28 REFIT has p < 0.05 and a mean difference of at most −0.05 pp in E1 and E2 and in both seed sets (in practice at least six of seven folds better each time). **Negative** if the same holds in the other direction. **As accurate as** if the absolute mean difference is below 0.05 pp in all four. Otherwise **not established**.
  - **Q2:** EWC-ONLINE − JOINT-STATIC, same rule.
  - **Q4 (descriptive):** EWC-ONLINE − D28 STATIC, same statistics reported, no verdict.
- **Q3 (training randomness, D21 replication rule):** paired two-sided Wilcoxon over the 35 matched (fold, seed) runs per seed set, Holm within the family {EWC-ONLINE − FT-ONLINE on held-out RMSE, EWC-ONLINE − FT-ONLINE on forgetting RMSE}; replicated if p_Holm < 0.05 in both seed sets with the same sign and an absolute difference of at least 0.05 pp in both; practically equivalent if below 0.05 pp in both; otherwise not replicated. **Positive** if forgetting is lower (replicated) and held-out RMSE is not replicated worse; a replicated loss on held-out RMSE is reported with it as the cost.
- Per-fold values, case7 and the per-horizon RMSE are reported beside every pooled number. Practical equivalence is reported as "as accurate as", never "better". Every outcome is reported, including null and negative ones; nothing is re-run to change one.

### Compute (estimate; to be checked on synthetic data before any real-data run)
Per experiment: 70 base fits (D26 cache; 47 s E1, 94 s E2 each if refitted), 42 pseudo-held-out joint fits (seed 0) for selection, and per (fold, seed) one adapted pass per arm with one fine-tune per distinct maturity time of the held-out case (U updates of E_ft epochs on at most a few hundred rows), plus the selection grid (8 configurations × 6 pseudo-cases × 7 folds). U is counted on the real sample frame (counting maturity times needs no model) and written to the run log with the compute check. **Fallback fixed here:** if the check estimates more than two nights with six workers, updates happen at every k-th distinct maturity time, with k the smallest integer that fits two nights, fixed and logged before any real-data model run.

### Not done, with reasons
- **Intervals for the adapted TFT** (LOCO calibration, quantile tracking): first establish whether the point forecast competes; planned as a separate decision if Q1 is positive or as accurate as.
- **Replay of stored training rows during adaptation:** excluded by the deployment setting (training data not available at the new site); D28 REFIT, which does use them, is the comparator.
- **Starting from the continual (EWC-streamed) residual TFT:** worse than the joint model by 0.32–0.44 pp (D26); the question here is adaptation to the new condition.
- **Re-tuning lr, architecture or the D21 inputs:** fixed, so that nothing but E_ft and λ is chosen, and those without the held-out case.
- **S1:** D28 was not run on S1, so there is no refit comparator there; an S1 replication needs its own decision.
- **Gradient boosting refitted without the training rows** (the deployment-matched comparator): gradient boosting cannot be updated without its training data; that asymmetry favours the comparator and is stated with Q1 rather than tested.

### Run log
- **31 Oct 2026. Pre-registered** after the D23–D30 results; no D31 code written or run.
- **31 Oct 2026. Code and compute check** (before any real-data model run). `scripts/run_d31.py`, `scripts/run_d31_night.py`, `tests/test_d31.py` (4 tests, synthetic data only). Choices the registration leaves open, fixed here: each update uses a fresh Adam optimiser at the D23 lr and batch size 64, seeded `seed + 100000 + update index`; FT-ONLINE is the same anchored copy with λ = 0 (no penalty term); the Fisher uses the `EWC` first-task seed rule (model seed + 1) on the train and test splits of the six cases (the joint model's training rows); forgetting is scored on their validation splits; the D26 reproduction tolerance is 1e-9 on this PC (the D26 platform), 0.05 pp only with `--other-platform`, which is recorded. **Compute:** one adapted pass on synthetic data (72 samples, 12 updates, E_ft 20) takes 15.8 s, about 60 ms per mini-batch irrespective of its size, so cost follows the mini-batch count. Updates per held-out case (counted on the real frames, no model): E1 case1–7 = 91, 94, 76, 127, 118, 191, 94; E2 = 92, 94, 77, 128, 120, 192, 95. Worst case (both arms selecting E_ft 20): E1 45 h, E2 80 h, 125 core-hours, about 21 h with six workers, within two nights (taken as 2 × 12 h): the fallback is not used (k = 1). The case6 folds set the critical path, so the launcher runs E2 first and the heaviest folds first.
- **1 Nov 2026. Run and results** (commit `10cadea`, clean tree, pinned versions; `scripts/run_d31_night.py --workers 6`, 31 Oct 22:15 to 1 Nov 05:22 local, 7.1 h; logs in `data/cache/d31/logs/`, git-ignored). Stop conditions passed: JOINT-STATIC reproduces D26 joint per (fold, seed) with max abs diff 0.0 (tolerance 1e-9); D28 REFIT/STATIC means 0.458/0.535 (E1) and 0.898/1.051 (E2) match the registered values. **Selection:** in all 14 folds both arms chose E_ft = 5 (the smaller grid value); EWC-ONLINE chose the largest λ (10 × λ_D23) in all seven E1 folds and five of seven E2 folds (1 × λ_D23 in case2 and case4). The selection therefore sits at the grid edge towards adapting less; this is a property of the registered grid, recorded as a limitation, and the grid is not extended after the fact. **Mean held-out RMSE (pp), main set:** E1 JOINT-STATIC 0.568, FT-ONLINE 0.757, EWC-ONLINE 0.704; E2 1.214, 1.376, 1.130. Forgetting RMSE: E1 0.206, 0.626, 0.330; E2 0.441, 1.248, 0.519. **Verdicts:** Q1 (EWC-ONLINE vs D28 REFIT): **not established**; EWC-ONLINE is worse in every cell (E1 +0.245/+0.244, p = 0.016 in both sets, 0 of 7 folds better; E2 +0.232/+0.197, p = 0.156/0.078). Q2 (vs JOINT-STATIC): **not established**, and the two set-ups disagree: E1 worse (+0.136/+0.126, p = 0.031/0.047, 1 of 7 folds better), E2 better (−0.085/−0.089, p = 0.078/0.016, 6 and 7 of 7 folds better). Q3 (EWC-ONLINE vs FT-ONLINE): **positive in E1 and E2**, replicated with Holm correction on both held-out RMSE (E1 −0.054/−0.050, E2 −0.246/−0.242) and forgetting RMSE (E1 −0.296/−0.290, E2 −0.729/−0.657; all p_Holm < 1e-6, n = 35). Q4 (vs D28 STATIC, descriptive): E1 +0.168/+0.167, E2 +0.079/+0.044; EWC-ONLINE is better only on case7 in E1 and on two folds in E2. **Reading:** among online updates of the joint model, EWC is clearly better than plain fine-tuning on both the new condition and the old ones, but updating the joint model online does not beat refitting the small D28 model, and only helps over not updating at the longer E2 horizons. Results: `results/forecasting_d31_e1.json`, `results/forecasting_d31_e2.json`.

## D32. Gradient boosting under the continual-learning constraint on unseen usage conditions (P2) (pre-registered)

**Status:** Written 1 Nov 2026 on the instruction of T. Sitaraman, **after** every D23–D31 result. The commit that adds this entry is the binding pre-registration; it precedes any D32 code. Code: `scripts/run_d32.py`, `tests/test_d32.py` (to be written). Results will go to `results/forecasting_d32_e1.json` and `results/forecasting_d32_e2.json`.

**Provenance (must be stated).** Post hoc, and motivated by the wish to show where the proposed model stands against gradient boosting:
- On the S6 continual stream (P1) the residual TFT + EWC is more accurate than gradient boosting, replicated in E1 and E2 (D23: −0.112 / −0.138 and −0.337 / −0.365 pp; D24 likewise), although gradient boosting there sees every task's rows at once.
- On unseen conditions (P2), gradient boosting fitted on all samples of the six training cases is more accurate than the continual shrunk residual TFT + EWC, replicated (D26 Q2: +0.39 / +0.41 pp E1, +0.57 / +0.60 pp E2). That comparison is asymmetric: the TFT learned the six cases one after another without keeping earlier cases' data; gradient boosting had all of them at once. Every D20–D31 gradient-boosting comparator had this access (D31 "Not done": "gradient boosting cannot be updated without its training data; that asymmetry favours the comparator and is stated with Q1 rather than tested"). D32 tests it.
- **Setting:** the continual setting of the paper: conditions arrive one at a time and a condition's data are not kept after its task (storage, data ownership across sites). A learner then holds only what it carries forward: weights (TFT + EWC: weights and Fisher information) or, for gradient boosting, its ensemble of trees.

Known before this pre-registration: every D26 quantity, **including the per-run P2 errors of the shrunk and raw residual TFT + EWC that D32 reuses as its proposed arm**, and the D26 gradient boosting fitted on all six cases. **Not computed for any rule below:** any gradient-boosting fit on fewer than all six training cases, or fitted case by case.

**Questions.**
- Q1 (primary): On an unseen condition, is the continual shrunk residual TFT + EWC more accurate than gradient boosting trained under the same constraint by warm-starting (GBM-WARM)?
- Q2 (primary): Is it more accurate than gradient boosting refitted on the latest case only (GBM-LAST)?
- Q3 (descriptive): the cost of the constraint for gradient boosting: GBM-WARM and GBM-LAST against the D26 all-cases fit (GBM-ALL), per fold and on case7.

### Setup (fixed here)
- **Data, inputs, horizons, label, folds, orders:** exactly D26: S6 samples (`p3.s6_samples`), the D21 shared inputs observed in the fold's training cases (the D26 baseline rule), E1 = horizons 1–3, E2 = 1–6, SOH relative to the first valid check; leave one of the seven test cases out; every sample of the held-out case is a test sample. Streams: for evaluation order *o* (the 10 D23/D26 orders), the fold's stream is order *o* with the held-out case removed (six cases).
- **Proposed arm (not refitted):** the D26 shrunk residual TFT + EWC, per (set, order, seed) P2 error read from `results/forecasting_d26_e*.json` (run at `9866d7a`); the raw residual TFT + EWC likewise for the secondary family. Seeds 0–4 main, 5–9 replicate.
- **Gradient boosting:** `FC._gbm(0)` (histogram gradient boosting, D18 settings: 200 iterations, lr 0.05, 8 leaves, at least 5 samples per leaf, L2 1.0; no early stopping at these sizes), direct ΔSOH target, scored on the residual scale like D26 (`run_d26.scores`). A training case contributes **all** its samples (the D20/D26 P2 rule for training cases).
  - **GBM-WARM (continual):** in stream order, the first case is fitted with 200 iterations; for each later case `warm_start` is set, `max_iter` is raised by 200, and the model is fitted on that case's samples only. Earlier cases' rows are not used again. Bin edges come from the first case (scikit-learn warm-start behaviour, fixed here).
  - **GBM-LAST:** fitted (200 iterations) on all samples of the last case of the fold's stream only.
  - **GBM-ALL:** the D26 comparator, refitted with the same code on all samples of the six cases (reproduction check).
  Both constrained arms are deterministic given (fold, order); a run (order *o*, seed *s*) is paired with the order-*o* value.
- **Run-level metric:** as D26: P2 error = mean over the seven folds of the held-out RMSE (pp), all origins and H+.
- **Reproduction checks (stop conditions):** (i) GBM-ALL per fold must equal the D26 `gbm_shared` per-fold RMSE to 1e-9; (ii) the D26 shrunk and raw run values read from the file must reproduce the D26 summary means to 1e-12; (iii) the orders must equal D26's. If any fails, no results file is written.
- **Pinned environment** (`requirements.txt`) and a clean tree; the script refuses otherwise unless an override is recorded in the run log.

### Tests (fixed here)
D26 rules: paired two-sided Wilcoxon over the 50 matched (order, seed) runs per seed set; differences a − b (negative favours a); Holm within each family and seed set; D21 replication rule (`run_d25.verdict`). E1 and E2 are separate families and are not independent (stated with every result).
- **Primary family:** (a) shrunk − GBM-WARM, P2 error on all origins; (b) shrunk − GBM-LAST, P2 error on all origins.
- **Secondary family:** (a) and (b) on H+; raw − GBM-WARM and raw − GBM-LAST on all origins.
- **Positive result:** Q1 is positive in an experiment if (a) is "replicated, a better" there; Q2 likewise with (b). Practical equivalence is reported as "as accurate as", never "better". GBM-ALL stays in every table as the comparator with access to all stored data, and the D26 Q2 verdict (GBM-ALL better, replicated) is reported beside any D32 result.
- **Power and scope:** the tests are over training randomness and orders with the seven conditions fixed; per-fold values and case7 are reported beside every pooled number.

### Compute (estimate)
Per experiment: 7 folds × 10 orders × (6 GBM-WARM stages + 1 GBM-LAST) plus 7 GBM-ALL fits, about 500 gradient-boosting fits on at most about 2,000 rows: minutes on one core. No neural network is trained.

### Not done, with reasons
- **Refitting the TFT:** the D26 runs are reused (same folds, orders, seeds); refitting would only reproduce them.
- **Gradient boosting with replay of a stored subset:** any stored subset breaks the constraint the comparison is about; GBM-ALL is the full-storage reference.
- **Retention (error on earlier cases) for the GBM arms:** D26 stored no retention metric for the TFT in P2, so there is nothing to pair it with.
- **Tuning gradient boosting for the warm-start setting** (iterations per stage, learning rate): the D18 settings are kept, as for every gradient-boosting comparator since D18; 200 iterations per stage gives each case the budget of one D18 fit.
- **S1:** not part of D26's P2 design; not run.

### Run log
- **1 Nov 2026. Pre-registered** after the D23–D31 results; no D32 code written or run.
- **1 Nov 2026. Code, run and results.** Code `scripts/run_d32.py`, `tests/test_d32.py` (2 tests, synthetic data only) committed at `fdcf0d6`; both experiments run on that commit (clean tree, pinned environment), a few minutes each. Reproduction checks passed: GBM-ALL per fold equals D26 `gbm_shared` (max abs diff 0.0), the D26 run values reproduce the D26 summary (max abs diff 7e-16), the orders equal D26's. **Mean P2 error (pp), all origins:** E1 shrunk residual TFT + EWC 0.924 / 0.945 (main / replicate), GBM-WARM 2.652, GBM-LAST 0.865, GBM-ALL 0.535; E2 1.620 / 1.653, 3.494, 1.593, 1.051. **Verdicts:** Q1 (shrunk − GBM-WARM) **positive in E1 and E2** (−1.727 / −1.707 and −1.873 / −1.840 pp, p_Holm < 1e-14, 50 of 50 runs better in each set). Q2 (shrunk − GBM-LAST) **not positive**: E1 replicated, GBM-LAST better (+0.060 / +0.080, p_Holm 0.022 / 0.003); E2 not replicated (+0.027 / +0.060, p_Holm 0.081 / 0.081). Secondary: on H+ the shrunk model and GBM-LAST are practically equivalent in E1 (−0.027 / −0.021) and not replicated in E2 (−0.128 / −0.106); raw − GBM-LAST practically equivalent in E1 and E2. Q3 (descriptive): the constraint costs gradient boosting 0.33 pp (GBM-LAST) and 2.12 pp (GBM-WARM) in E1, 0.54 and 2.44 pp in E2; on case7 GBM-ALL 1.489 / 3.127, GBM-LAST 1.865 / 3.222, GBM-WARM 3.553 / 5.105 (E1 / E2). **Diagnosis of GBM-WARM (post hoc, must be stated with Q1):** with warm start, scikit-learn keeps the bin edges of the first case. Every usage-condition input (C-rates, DoD, charge voltage, CV charge, temperature, chemistry flags) is constant within a case, so it has a single bin after the first case and later trees cannot split on it: GBM-WARM cannot tell the conditions apart. The Q1 result is therefore largely a property of this warm-start implementation, not evidence that the TFT is better than continual gradient boosting in general; it is reported as registered with this caveat, and GBM-LAST is the informative constrained comparator. **Reading:** under the continual constraint the proposed model is about as accurate as gradient boosting refitted on the latest case (slightly worse in E1, level in E2), and both are far behind gradient boosting with all six cases stored. Results: `results/forecasting_d32_e1.json`, `results/forecasting_d32_e2.json`.

## Tooling note (not a protocol decision): results dashboard on Tableau Public with a JS page (2 Nov 2026)

No experiment, protocol or result changes; this note takes no D-number.

- **Decision.** The interactive results dashboard uses **Tableau Public**, embedded in a static JS page (`dashboard/`) with the Tableau Embedding API v3. Power BI was the alternative.
- **Reasons.** After export the data are small: about 850 S6 checks, 111 S1 RPTs, 38.5k saved forecasts and 42k run × task rows. They are public (CC BY 4.0), and the repository is public. Tableau Public is free to publish and runs on Windows and macOS. Its JS API sets filters and reads mark selections without paid capacity. Power BI's JS embedding needs Pro or Embedded capacity and Azure AD, and its free "publish to web" offers no API.
- **Costs accepted.** Published views are public and use extracts, with no live connection. Workbook authoring is a manual step in Tableau Desktop Public Edition (`dashboard/tableau/WORKBOOK_SPEC.md`).
- **Data rules carried over.** `scripts/export_dashboard_data.py` only reshapes committed results files and the label tables. SOH stays relative to the first valid check (D17), and reuse classes use `src.evaluation.reuse_class`. Missing temperatures stay empty and are flagged. S1 and S6 are never pooled, and every view shows the baselines next to the learned models. Tables built from the data folder go to `data/dashboard/` (git-ignored). Tables built from results files go to `results/dashboard/`, committed like `results/tables/`.

## D33. Exploratory search for a forecaster better than both linear extrapolation and gradient boosting (exploratory, not reported)

**Status:** Run 2 Nov 2026, after every D23–D32 result, at the request of T. Sitaraman to look for a method that beats both traditional comparators in both S6 settings. **Exploratory only:** nothing here is a reported result, enters the paper, or may be cited as evidence. It is logged so that the number of variants tried is on record (multiplicity) and so that none of them is later run as if it were a fresh, pre-registered idea. Scripts were kept outside the repository (not committed); the inputs are the D26 runner's S6 tables at commit `43086c3`.

**Data used (no test data).**
- **P1:** fitted on the train split of all seven S6 cases, scored on their validation split only.
- **P2:** nested leave-one-condition-out inside each of the seven D26 outer folds: for outer case *f* and each inner case *j* ≠ *f*, fit on the other five cases and score on *j*. The outer held-out case was never used. Mean over the 42 (*f*, *j*) pairs.

**Static candidates (RMSE in pp; P1 validation | P2 nested inner):**

| Candidate | E1 (h 1–3) | E2 (h 1–6) |
|---|---|---|
| LIN: linear extrapolation | 0.410 \| 1.372 | 0.808 \| 2.300 |
| GBM: gradient boosting, D26 inputs | 0.391 \| **0.573** | 0.733 \| **1.058** |
| GBM+: GBM with fade rate and LIN forecast as inputs | 0.417 \| 0.588 | 0.744 \| 1.063 |
| RGBM: LIN + GBM on the residual | 0.470 \| 0.648 | 0.854 \| 1.093 |
| RATE: GBM on fade per cycle to target, times span | 0.584 \| 0.696 | 0.961 \| 1.177 |
| RRATE: LIN + GBM on residual per cycle, times span | 0.559 \| 0.679 | 0.931 \| 1.128 |
| SWITCH: LIN where a fade rate exists, else GBM | 0.410 \| 0.869 | 0.808 \| 1.598 |
| AVG: mean of LIN and GBM where a fade rate exists, else GBM | **0.359** \| 0.618 | **0.720** \| 1.098 |

**Online candidates, P2 nested inner, E1 only** (D28 arrival rule; RMSE pp): STATIC GBM 0.573; LIN 1.372; online GBM refit (D28 REFIT) 0.479; refit with the new condition's rows weighted ×5: 0.471; online per-horizon inverse-MSE combination of LIN and the refit: 0.485 (×1) and 0.479 (×5); the combination beat the plain refit in 4 of 7 outer folds. E2 was not run.

**Reading.**
1. No candidate beats GBM in P2: every combination with LIN or rate target is worse on unseen conditions.
2. P1 validation is misleading here: GBM beats LIN on validation rows (0.391 vs 0.410) yet loses to it on the P1 test rows (D23/D25), because trees cannot extrapolate into the later life that the test split covers. AVG's P1 validation advantage therefore does not predict a test advantage and was not pursued.
3. Online, up-weighting and combining add at most 0.008 pp over the D28 refit, below the 0.05 pp equivalence threshold.
4. Conclusion: with seven usage conditions from one laboratory, P1 rewards trend smoothness and P2 rewards stored cross-condition data; no single method found here wins both. No variant is carried forward to a test-set run. A confirmatory test of a new method requires new data and its own pre-registration before any result on those data is computed.

## D34. Unseen usage conditions on the KIT first-life pool (S7): the continual residual TFT + EWC against gradient boosting and linear extrapolation over many conditions (P2) (pre-registered)

**Status:** Drafted 2 Nov 2026 on the instruction of T. Sitaraman, **after** every D23–D33 result. The commit that adds this entry is the binding pre-registration; it precedes any D34 code. Any change before code runs on real data is logged as an amendment in the run log; any change after that is a deviation. Code (to be written): `scripts/run_d34.py`, `tests/test_d34.py` (synthetic data only), runbook `docs/D34_NIGHT_RUN.md`. Results will go to `results/forecasting_d34_e1.json` and `results/forecasting_d34_e2.json`.

**Provenance (must be stated).** Post hoc and motivated by the wish to find a setting where the proposed model beats the traditional forecasters:
- On S6, the proposed model beats gradient boosting on the continual stream (D23/D24) and linear extrapolation on unseen conditions (D26), but loses to gradient boosting with all conditions stored on unseen conditions (D26) and is level with or slightly behind gradient boosting refitted on the latest condition (D32). D33 (exploratory) found no variant that beats both comparators on S6.
- Every S6 claim about new conditions rests on seven conditions from one laboratory, with the depth-of-discharge effect unidentifiable when case7 is held out (D28). S7 has 67 usage conditions (195 cells with at least five valid check-ups), including calendar, cyclic and WLTP-profile ageing at four temperatures, so it can test the claim over many conditions.
- **Hypothesis, fixed here:** under the continual constraint (conditions arrive one at a time and earlier conditions' data are not kept), the residual TFT + EWC carries forward what it learned about earlier conditions, while gradient boosting refitted on the latest condition sees one condition only. With 48 training conditions instead of six, the proposed model's advantage over that constrained comparator should be larger than on S6 (D32: about 0.06 pp the other way). Gradient boosting with every condition stored is expected to stay at least as accurate; it is tested, not assumed.

**What is already known about S7 (must be disclosed).**
- S7 has been used only as part of the first-life pretraining pool (D21 P3, D25). Pretrained models were scored on the pool's per-cell test splits, pooled with S8 and S9 (`docs/RESULTS.md`, P3 retention: TFT pool test RMSE about 3.1–3.3 pp).
- The S7 table is described in D21 (195 cells, 67 condition sets, 10,251 samples, SOH 32.1–100 %).
- For this entry, only the structure was counted (2 Nov 2026; no model fitted, no forecast or error computed): 16 calendar conditions (48 cells, 1,329 check-ups), 39 cyclic (111 cells, 1,719) and 12 WLTP-profile (36 cells, 759); 63 conditions have 3 cells, 2 have 2 and 2 have 1; ageing temperatures 0, 10, 25 and 40 °C; check-ups every 20–21 days (10th–90th percentile); 5–28 check-ups per cell (median 23).
- **No leave-one-condition-out result, and no linear-extrapolation, gradient-boosting or ridge result of any kind, has been computed on S7.**

**Questions.**
- Q1 (primary hypothesis): On an unseen S7 condition, is the shrunk residual TFT + EWC, learned as a stream of conditions, more accurate than gradient boosting refitted on the latest condition of the same stream (the D32 constrained comparator)?
- Q2: Is it more accurate than linear extrapolation at origins where extrapolation has a fade rate (H+)?
- Q3: Is it more accurate than gradient boosting fitted on all training conditions at once (data stored)?

### Setup (fixed here)
- **Data:** the committed S7 check table (`data/pool_s7_checks.csv`, built by `scripts/build_pool_tables.py`, D21 labels and exclusions), cells with at least five valid check-ups. Samples from `p3.harmonised_samples(..., "S7", horizons)` with the D21 shared inputs (`p3.SHARED_FEATURES`; S7 uses calendar days only, cycle inputs missing as in D21). Label: SOH relative to the cell's first valid check-up (D21). E1 = horizons 1–3 check-ups, E2 = horizons 1–6 (about 20–60 and 20–120 days ahead). Per-cell purged splits as D18.
- **Task = condition.** In the samples, `Battery_ID` is set to the S7 condition (as for S6 test cases), so each condition is one task of the stream and TFT windows stay within (condition, cell × horizon).
- **Condition sets, assigned once by seeded stratified sampling (seed 34) before any model is run:**
  - **Tuning conditions (7):** used only to select hyperparameters, never trained on in an evaluation stream and never tested. Stratified by ageing type (calendar, cyclic, WLTP) in proportion to their counts.
  - **Test conditions (60):** split into **five outer folds of 12 conditions**, stratified by ageing type and ageing temperature.
  - The assignment is written to the results file and to a committed table (`results/d34_condition_sets.csv`) before tuning starts.
- **Per outer fold:** the 12 held-out conditions are never used for training, early stopping, tuning, shrinkage or calibration; all their samples are test samples. The other 48 test conditions are the fold's training conditions. For each training condition, the train and test splits are its training rows and its validation split is for early stopping (D18 P2 rule).
- **Proposed method (continual):** the D23 residual TFT + EWC (`p3.build_strategy("tft_ewc", ...)`, residual on the D21 per-day linear extrapolation) streamed over the fold's 48 training conditions, then evaluated zero-shot on each held-out condition; shrunk with the D25 calibrator (`src/calibration.py`, unchanged) fitted **once after the stream on the validation rows of the 48 training conditions pooled** (the D26 rule). The raw (unshrunk) model from the same runs is reported as well.
- **Orders and seeds:** five stream orders per fold (`P.sample_orders(conditions, 5, 0)`: the fold's 48 conditions in check-table order, then four seeded permutations), seeds 0–2 (main set) and 5–7 (replicate set): 15 runs per fold and seed set. Fewer runs than D26 (10 × 5) because the tests below are over conditions (60), not over runs; the runs only average out training randomness.
- **Tuning (nested per outer fold, on the tuning conditions only):**
  - Streams over the fold's 48 training conditions in tuning orders disjoint from the evaluation orders (seeded, seed 1); criterion = mean RMSE over the seven tuning conditions (all their samples), zero-shot after the stream, raw model.
  - Stage 1: learning rate {1e-3, 3e-3, 1e-2} with fine-tuning (no EWC). Stage 2: EWC λ {0.01, 0.1, 1, 10, 100, 1000} at the stage-1 learning rate. Five tuning runs (orders × seeds) per grid point, **fewer than D21's ten** (budget; stated with the results).
  - D21 one-standard-error rule (most conservative value within one standard error of the best: smallest learning rate, smallest λ) and the D11 edge rule (extend the grid by two values on its ladder and repeat, at most twice). Selected values, margins and any extension are recorded per fold.
  - Why not the D23 values: the EWC implementation keeps one Fisher term per completed task (`src/continual_learning.py`, `EWC._after_task`), so the penalty after 48 tasks sums eight times as many terms as after six; a λ tuned on S6 streams does not carry over.
- **Comparators** (each per outer fold, on the same rows):
  - **GBM-LAST** (Q1): gradient boosting (`FC._gbm(0)`, D26 settings, D21 shared inputs; inputs never observed in its rows are dropped, as D20) refitted on the training and test splits of the **last condition of each evaluation order** only; one fit per (fold, order), paired with that order's runs.
  - **GBM-ALL** (Q3): the same model on all 48 training conditions' training rows at once; one fit per fold.
  - **Linear extrapolation** (Q2): the D21 pool definition (least-squares per-day fade rate over the last three valid check-ups times days to the target; persistence where no rate exists), the same quantity the residual model corrects.
  - Persistence and ridge (D26 settings) as descriptive references; the joint residual TFT (all 48 conditions at once, the fold's selected learning rate, seeds as above) as the upper bound of the continual model.
- **Origin strata:** H+ / H− as D26 (the committed baseline code decides whether extrapolation has a fade rate). Counts per fold and condition are written to the results file.
- **Pinned environment** (`requirements.txt`); pre-flight refuses anything else unless `--allow-unpinned` is recorded in the run log. The run starts from a clean commit containing all D34 code.

### Metric and tests (fixed here)
- **Unit of analysis: the held-out condition.** For condition *c*, seed set *S* and method *m*: RMSE (pp) over all test samples of *c* (all its cells and horizons), averaged over the 15 runs of *S* (over the five GBM-LAST fits for GBM-LAST; single fits for GBM-ALL, extrapolation, persistence and ridge). Computed on all origins, on H+ and on H−; conditions without H+ samples are left out of H+ tests only, and their number is reported.
- **Test:** paired two-sided Wilcoxon signed-rank over the 60 test conditions, difference a − b (negative favours a), Holm correction within each family and seed set. **Replication rule (D21):** replicated if p_Holm < 0.05 in both seed sets with the same sign and the mean difference over conditions is at least 0.05 pp in both; practically equivalent if |mean difference| < 0.05 pp in both; otherwise not replicated. Reported with each test: mean and median difference, number of conditions favouring each side, rank-biserial effect size, and the five fold means.
- **Primary family (per experiment):** (a) shrunk TFT + EWC − GBM-LAST, all origins (Q1); (b) shrunk TFT + EWC − linear extrapolation, H+ (Q2); (c) shrunk TFT + EWC − GBM-ALL, all origins (Q3).
- **Secondary family (per experiment):** (a) and (c) on H+ and on H−; (b) on all origins; shrunk − raw; raw continual − joint residual TFT; GBM-LAST − GBM-ALL (the cost of the constraint for gradient boosting).
- **Descriptive (no test):** results by ageing type (calendar, cyclic, WLTP) and by ageing temperature; per-horizon RMSE; the worst condition per method; 90 % and 80 % interval coverage of the shrunk model, pooled and per ageing type (D26 calibrator), with the D26 adequacy threshold (pooled 90 % coverage at least 0.85 in both seed sets) reported but not part of any question.
- **What counts as a positive result:** Q1: (a) replicated in favour of the model in E1 **and** E2. Q2: (b) replicated in favour in E1 and E2; a win on all origins or on H− alone does not count (on H− extrapolation is persistence). Q3: (c) replicated in favour in E1 and E2. A result in one experiment only is reported as such and not as a positive answer. Practical equivalence is reported as "as accurate as", never "better". Every outcome is reported, including null and negative ones; nothing is re-run to change one.
- **Scope of any claim.** S7 is first-life NMC/C-SiO data from one laboratory, with check-ups at 25 °C; depth of discharge is the SoC window of the ageing profile. A positive result supports generalisation to new usage conditions under the continual constraint; it is **not** a second-life result and is reported beside, never pooled with, S1 and S6.

### Compute budget (estimate; to be checked on synthetic data of S7 size before any real run)
Per experiment and outer fold: 15 tuning grid points (stage 1 three, stage 2 six, plus any edge extension) × 5 runs, and 5 orders × 6 seeds evaluation streams, each a stream of 48 short tasks; joint fits 6 per fold; single-fit baselines negligible. From the D26 timings (about 6 s per S6 task), roughly 4–5 min per E1 stream and twice that for E2, giving about 70–90 h of single-process CPU for both experiments, **about 12–15 h with six workers (two nights).** Fixed now: if the synthetic check estimates more than 18 h with six workers for both experiments, the number of tuning runs per grid point drops from five to three, and nothing else changes; if it is still above 18 h, E2 reuses E1's selected hyperparameters per fold (disclosed). The run is started by T. Sitaraman in a terminal outside the assistant session (as D31), with checkpoints so that a stopped run resumes.

### Not done, with reasons
- **Gradient boosting with a bounded replay buffer** (a constrained comparator with memory comparable to EWC's stored Fisher terms): not part of the constraint as defined in D32; would need its own memory-matching rule. Planned only if Q1 is positive.
- **The pool's S8 and S9 as further conditions:** different chemistries and label definitions (cycling-rate pseudo-checks, D21); mixing them would confound condition with dataset.
- **Pretraining on S8/S9 before the S7 stream:** negative in D25 (residual pretraining); not repeated here.
- **Independent replication** on a second many-condition dataset (Stroebl et al. 2024, *Scientific Data*, 279 Samsung INR21700-50E cells, 71 ageing conditions, figshare, CC BY 4.0): to be registered separately after D34, whatever its outcome; not downloaded yet.

### Settled points (2 Nov 2026, T. Sitaraman: defaults accepted)
1. Outer folds: five folds of 12 conditions (default) rather than leave-one-condition-out over 60 conditions (about 12 times the compute).
2. Seven tuning conditions withheld from testing (default) rather than nested tuning conditions drawn per fold from the training conditions (keeps all 67 testable, more complex).
3. Runs per condition: 5 orders × 3 seeds per seed set (default).

### Run log
- **2 Nov 2026. Drafted** after the D23–D33 results. S7 structure counted (above); no D34 code written or run; no model fitted on S7 outside D21/D25.
- **2 Nov 2026. Defaults accepted and pre-registered** (T. Sitaraman). This commit is the binding pre-registration. No D34 code written or run.
- **2 Nov 2026. Code committed** after tests on synthetic data only (pytest temporary directories and the session scratch space; never real data): `scripts/run_d34.py`, `scripts/run_d34_night.py`, `tests/test_d34.py` (5 tests), runbook `docs/D34_NIGHT_RUN.md`; nothing under `src/` changed. **Condition sets** written by `run_d34.py --write-sets` from the S7 check table (counts and condition metadata only; no model fitted) and committed as `results/d34_condition_sets.csv`: tuning conditions 2 calendar, 4 cyclic, 1 WLTP; test folds of 12 (folds 0–3: 3 calendar, 7 cyclic, 2 WLTP; fold 4: 2, 7, 3). The paper profile refuses to run unless the computed assignment equals the committed file. **Implementation choices fixed with the code:** the five tuning runs per grid point are five tuning orders (seed-1 permutations of the fold's 48 conditions, the table order dropped; asserted disjoint from the evaluation orders), each with seed 0; tuning is shared by both seed sets; stage 2 re-reads stage 1 from its checkpoints; GBM-ALL, GBM-LAST and ridge are fitted on the training rows as registered (train and test splits; the validation split is unused by single fits); persistence is ΔSOH = 0; collection uses only folds finished in both seed sets; the smoke profile uses two tuning runs (the one-standard-error rule needs a standard error). **Compute check (synthetic data of S7 size: 67 conditions, 3 cells, 18–28 check-ups; paper profile, single-threaded):** one EWC stream over 48 conditions 2.2 min (E1) and 3.0 min (E2); one joint fit 2.1 and 1.3 min. About 34 h of CPU for both experiments, about 6 h with six workers (about 9 h if every λ grid is extended twice), below the registered 18 h threshold, so the fallback rule does not apply. No D34 code has been run on S7.
- **3–4 Nov 2026. Run.** From a clean worktree of `main` at `6d155e0` (the repository folder had another session's uncommitted dashboard edits, so it failed the clean-tree pre-flight and was not used), `DATA_ROOT` pointing to the repository's `data/`, pinned `.venv` (Python 3.11.15). Invocations: 11:08 dry run; 13:10 `run_d34_night.py --workers 6 --deadline 07:30`, which stopped during the smoke stage (last event 13:55, no paper part started); 18:44 the same command again (pre-flight passed; smoke resumed from checkpoints and passed at 19:13). Paper parts 19:13 (3 Nov) to 08:15 (4 Nov), about 13 h, twice the synthetic estimate; all 40 parts finished, none failed, none stopped by the deadline (the last part started at 07:24). Collection: E1 written at 08:15; **E2 collection failed** with `KeyError: 'rmse_h5'`: a held-out condition with short series has no sample at horizon 5, so its score has no `rmse_h5` entry. **Fix (implementation, before any E2 result was computed):** in `condition_values`, a horizon a condition does not have is missing (NaN), as already done for conditions without H+ samples. It affects only the per-horizon descriptive values; `src/` is unchanged, so the checkpoints stay valid. Both experiments are re-collected from the checkpoints at the commit with this fix, and the E1 file written at 08:15 is replaced.
- **4 Nov 2026. Results** (`results/forecasting_d34_e1.json`, `..._e2.json`, collected at `474f74f`, clean tree; that commit was later rebased onto `main` as `a5ed900` with identical `src/`, runner scripts and `requirements.txt`, and is kept reachable as tag `d34-collection`; all five folds in the tests, no part not run; 60 test conditions; 10,251 E1 and 18,762 E2 samples; 88 % of held-out samples are H+). **Mean RMSE over the 60 conditions (pp), main / replicate:** E1 shrunk TFT + EWC 5.201 / 5.184, raw 4.787 / 4.707, joint residual TFT 4.123 / 4.328, GBM-LAST 6.501, GBM-ALL 3.845, linear extrapolation 5.313, ridge 4.313, persistence 6.605; E2 7.489 / 7.481, 6.824 / 6.711, 5.790 / 6.182, 8.398, 5.117, 7.896, 6.183, 9.108. **Primary verdicts (D21 rule over conditions):** Q1 shrunk − GBM-LAST: E1 −1.300 / −1.317 (p_Holm 0.0013 / 0.0011; 39 of 60 conditions better), **replicated, model better**; E2 −0.909 / −0.917 (p_Holm 0.10 / 0.10; 37 of 60), **not replicated**. Q2 shrunk − linear extrapolation on H+: E1 −0.010 / −0.000, **practically equivalent**; E2 −0.072 / −0.080 (p_Holm 0.21 / 0.22), **not replicated**. Q3 shrunk − GBM-ALL: E1 +1.355 / +1.338, E2 +2.372 / +2.365, **replicated, GBM-ALL better** (50 and 55 of 60 conditions). **Registered answers (positive only if replicated in E1 and E2): Q1 not positive (E1 only), Q2 not positive, Q3 negative.** Secondary (all replicated): against GBM-LAST the model is better on H+ (−2.17 / −2.16 E1, −2.26 / −2.27 E2) and worse on H− (+1.43 / +1.37, +2.07 / +2.04); it is better than linear extrapolation on all origins (−0.11 / −0.13, −0.41 / −0.42; on H− extrapolation is persistence); shrinkage makes it worse (+0.41 / +0.48, +0.67 / +0.77); the continual raw model is worse than the joint one (+0.66 / +0.38, +1.03 / +0.53); the constraint costs gradient boosting 2.66 pp (E1) and 3.28 pp (E2). Descriptive: by ageing type (E1 main, shrunk / GBM-LAST / GBM-ALL / extrapolation) calendar 0.51 / 5.00 / 0.39 / 0.52, cyclic 6.92 / 7.11 / 4.94 / 7.08, WLTP 5.70 / 6.49 / 4.75 / 5.78; pooled 90 % coverage 0.71 (E1) and 0.57 (E2), below 0.85 (not adequate). **Tuning (must be stated with every result):** the selected learning rates and EWC strengths are far below the S6 values and mostly below the registered grids, reached through the D11 edge extensions: E1 lr 1e-3 / 1e-4 / 1e-3 / 1e-4 / 1e-4 and λ 1e-4 / 1e-2 / 1e-2 / 1e-4 / 1e-5 (folds 0–4); E2 lr 1e-5 / 1e-4 / 1e-5 / 3e-3 / 3e-4 and λ 1e-6 / 1e-4 / 1e-4 / 1 / 1e-6; no final selection at a grid edge. With λ this small, EWC adds almost no penalty in most folds: the proposed model here is in effect fine-tuning with a small learning rate. Errors are 5–10 times those on S6 because S7 fades to 32 % SOH with knees, and every method's error is dominated by the cyclic and WLTP conditions.

## D35. Gradient boosting under the continual-learning constraint on the S6 stream (P1) (post hoc, not pre-registered)

**Status:** Written 3 Nov 2026 on the instruction of T. Sitaraman, **after** the result was seen. Code: `scripts/run_d35.py`, `tests/test_d35.py`. Results: `results/forecasting_d35_e1.json`, `results/forecasting_d35_e2.json`.

**Provenance (must be stated with every D35 number).**
- **Why.** D32 tested gradient boosting refitted on the latest case only (GBM-LAST) on unseen conditions (P2), but not on the continual stream (P1), so the headline table (paper Table III) had no P1 value for gradient boosting under the constraint. The question came up while choosing a main-text figure; it is asked because the answer could favour the proposed model.
- **Not pre-registered.** The comparison was first run on 3 Nov 2026 as a scratch script on the author's machine (pinned environment; the D23 all-tasks gradient boosting reproduced, 0.4513 / 0.8984 pp against the committed 0.451 / 0.898). This entry and `scripts/run_d35.py` were written after that output was seen. The committed results files come from re-running the committed script on a clean tree. Every D35 result is reported as **post hoc**.
- **Setting.** As D32: tasks arrive one at a time and a task's data are not kept after its step; gradient boosting carries forward only its ensemble, so under the constraint it is refitted on the latest task.

### Setup
- **Data, inputs, horizons, label, splits, orders:** exactly D23/D25: S6 samples (`p3.s6_samples`), the D21 shared inputs observed in the stream's training rows, E1 = horizons 1–3, E2 = 1–6, SOH relative to the first valid check, purged chronological splits, the 10 D23 evaluation orders.
- **GBM-LAST:** `FC._gbm(0)` (D18 settings) fitted on the training rows of the latest task only, after each step; every task's test split is scored after every step (accuracy matrix R; `ev.cl_metrics`: final average error, forgetting, plasticity, backward transfer). Two arms: **direct** target ΔSOH (as every gradient-boosting comparator since D18) and **residual** target ΔSOH − linear extrapolation (as the proposed model, D23). Both are deterministic given the order.
- **Proposed arm (not refitted):** the D25 shrunk residual TFT + EWC from scratch, per (set, order, seed) final average error read from `results/forecasting_d25_e*.json`.
- **Reproduction checks (stop conditions):** gradient boosting fitted on all tasks' training rows equals the committed D23 `gbm_shared` to 1e-9 pp; the orders equal D25's. Pinned environment and clean tree, as D32.

### Tests
D32 pairing: each (order, seed) run is paired with its order's GBM-LAST value; two-sided Wilcoxon over the 50 runs per seed set; differences shrunk − GBM-LAST (negative favours the proposed model); Holm over the two arms within each seed set; D21 replication rule (`run_d25.verdict`). E1 and E2 are separate and not independent.

### Not done, with reasons
- **GBM-WARM on the stream:** D32 showed that warm-started scikit-learn gradient boosting keeps the first case's bin edges and cannot use the usage inputs afterwards; it would be a weak comparator.
- **Tuning gradient boosting for one task at a time:** the D18 settings are kept, as for every gradient-boosting comparator.

### Run log
- **3 Nov 2026. Scratch run (author's machine, pinned environment), before this entry.** Final average error (pp), E1: GBM-LAST direct 0.912 (forgetting 0.447), residual 0.561 (0.171); E2: direct 1.599 (0.884), residual 0.956 (0.338); D25 shrunk residual TFT + EWC 0.318 / 0.308 (E1) and 0.498 / 0.493 (E2), main / replicate. The proposed model was more accurate for all 10 orders in both seed sets against both arms (order-mean pairing).
- **3–4 Nov 2026. Committed runs** (author's machine) of `scripts/run_d35.py` from `5fddf96`. Both with `--allow-unpinned`: Python 3.12.10 instead of the pinned 3.11; every library version matched `requirements.txt`. The reproduction checks passed exactly in both experiments (gradient boosting on all tasks 0.4512561254029118 / 0.8983726883094832 pp, difference 0.0 from D23; orders equal D25's). **E1** ran at `5fddf96`; the file records `git_dirty: true` because the author's uncommitted dashboard files (HTML/JS/CSS only) were restored while the run was in progress, after its pre-flight check had passed on a clean tree. **E2** ran at `b8c71f4` on a clean tree; `5fddf96..b8c71f4` changes dashboard files only (no Python, `src/`, `scripts/`, `tests/` or `results/` change), so both runs used identical D35 code. **Final average error (pp):** E1 GBM-LAST direct 0.912 (forgetting 0.447), residual 0.561 (0.171); E2 direct 1.599 (0.884), residual 0.956 (0.338). **Verdicts (D32 pairing over 50 runs per seed set, Holm over the two arms):** shrunk residual TFT + EWC − GBM-LAST residual −0.243 / −0.254 (E1) and −0.458 / −0.463 (E2), − GBM-LAST direct −0.594 / −0.605 and −1.101 / −1.106 pp, all **replicated, a better** (p_Holm ≤ 4e-12). **Reading (post hoc):** under the continual constraint, gradient boosting forgets on the stream and is less accurate than the proposed model there, while on unseen conditions it is slightly better (E1) or level (E2) (D32). Among the methods that keep no earlier data, only the proposed model is within 0.07 pp (E1) and 0.05 pp (E2) of the best in both settings. Gradient boosting with all conditions stored stays more accurate on unseen conditions (D26). Results: `results/forecasting_d35_e1.json`, `results/forecasting_d35_e2.json`.

## D36. Confirmatory test on a new many-condition dataset (S10, Stroebl et al. 2024): the D34 protocol with shrinkage only where extrapolation has a fade rate (P2) (pre-registered)

**Status:** Written 4 Nov 2026 on the instruction of T. Sitaraman, **after** the D34 results. The commit that adds this entry is the binding pre-registration; it precedes the S10 check table (no S10 capacity, SOH or forecast has been computed) and any D36 code. Changes before code runs on real data are logged as amendments in the run log; changes after that are deviations. Code (to be written): `scripts/build_stroebl_checks.py`, `scripts/run_d36.py`, `scripts/run_d36_night.py`, `tests/test_d36.py` (synthetic data only). Results: `results/forecasting_d36_e1.json`, `results/forecasting_d36_e2.json`. The download script `scripts/fetch_stroebl.py` was written and run before this entry (raw files only; see S10 below).

**Provenance (must be stated).** Post hoc, motivated by D34 and by the wish to find a setting where the proposed model beats the traditional forecasters:
- In D34 (S7, 60 unseen conditions) the shrunk residual TFT + EWC beat gradient boosting refitted on the latest condition (GBM-LAST) for horizons 1–3 (replicated) but not 1–6 (not replicated), was level with linear extrapolation where it has a fade rate, and lost to gradient boosting with all conditions stored (GBM-ALL).
- **Diagnosis (post hoc, from the D34 results):** the D25 shrinkage pulls the forecast towards linear extrapolation. At origins where extrapolation has no fade rate (H−, a cell's first two check-ups), extrapolation is persistence, and shrinkage raised the model's H− error from 4.43 to 6.66 pp (E1); on H+ it lowered it slightly (4.67 to 4.40). An approximate recomputation from the stored D34 values suggested that shrinking on H+ only would give −1.94 (E1) and −1.72 pp (E2) against GBM-LAST. That number was computed on seen data and is not evidence; D36 tests the rule on new data.
- **Rule under test, fixed here (SEL):** the shrunk forecast where linear extrapolation has a fade rate (H+), the raw residual model's forecast elsewhere (H−). It uses only what is known at the origin. The D25 calibrator itself is unchanged.
- No other change to the D34 protocol: same model, tuning, comparators, tests and replication rule, so that D36 is a confirmatory replication of D34 on new data with this one change.

### S10: data (to be added to `docs/EXTERNAL_DATA.md` with the code)
- **Article:** F. Stroebl, R. Petersohn, B. Schricker, F. Schaeufl, O. Bohlen and H. Palm, "A multi-stage lithium-ion battery aging dataset using various experimental design methodologies," *Scientific Data*, vol. 11, 2024, doi:10.1038/s41597-024-03859-z (PMC11412976, CC BY 4.0).
- **Dataset:** "Multi-Stage Lithium Ion Battery Aging Study," figshare, version 1, 6 Aug 2024, doi:10.6084/m9.figshare.25975315, **CC BY 4.0** (figshare record).
- **Content (article and `experiments_meta.csv`):** 278 Samsung INR21700-50E cells in two stages; 71 test points (conditions): stage 1 has 11 calendar and 25 cyclic, stage 2 has 11 calendar and 24 cyclic; 49 conditions with 3 cells and 22 with 6; each condition tested in one of three laboratories. Factors: ambient temperature (10, 23, 35, 45 °C), maximum SOC, depth of discharge (0–0.6) and charge and discharge C-rates (0.2–1.0 and 0.2–2.0). Check-ups (CU and extended check-ups, exCU) at 23 °C during ageing; entry (ET) and final (AT) characterisation at 10, 45 and 23 °C.
- **Download (before this entry, raw files only):** `scripts/fetch_stroebl.py` downloads each per-cell zip, checks its MD5 against the figshare record, and keeps only the 23 °C check-ups (`*_CU`, `*_exCU`, `*_ET_T23`, `*_AT_T23`) and every `*_meta.txt`, gzip-compressed in `data/raw/stroebl2024/` (git-ignored). The manifest `docs/external_data/metadata/stroebl2024_manifest.json` lists every zip's MD5, all its entries, and the SHA-256 of each kept file. To define the label, two zips and the article's protocol tables were read; no capacity was computed.
- **Label (fixed here):** capacity = charge (Ah, |∫ I dt|) of the capacity-measurement discharge, `step_type` 22 (section A: CC at 1C, CV at 2.5 V to C/20), in each 23 °C check-up (CU, exCU, ET_T23, AT_T23). If a file holds more than one `step_type` 22 block, the last complete one is used; a block is complete if its minimum voltage is at most 2.55 V and its final current magnitude is at most 0.3 A (about C/16). A check-up without a complete block is invalid. A check-up whose cell surface temperature at the start of the block lies outside 18–28 °C is invalid (the S7 cold-check rule, D21). SOH = capacity / the cell's first valid check-up capacity × 100. The authors define capacity as the mean of the charge and discharge capacities; the discharge capacity is used here for consistency with S1, S6 and S7 (D17, D21), and this is stated in the paper.
- **Time:** check-up date = the measurement start date in the file's `*_meta.txt` (one-day resolution); calendar days since the first check and to the target. No throughput or cycle inputs (they partly encode the label, as for S7, D21).
- **Conditions (inputs):** the D21 shared inputs mapped as for S7: `temp_c` = ageing ambient temperature; calendar cells: `calendar` 1, C-rates 0, `dod` 0, `rest_soc` = maximum SOC; cyclic cells: `calendar` 0, `charge_c_rate`, `discharge_c_rate`, `dod` from `experiments_meta.csv`. One input is added for S10 only, **`soc_max`** (maximum SOC, defined for both types), because the cyclic SOC window is a design factor of the study and no shared input carries it. `v_charge_max` and `cv_charge` are missing (not stated per condition); chemistry flags are all 0 (a constant within the dataset).
- **Cells and conditions used:** cells with at least five valid check-ups; a condition is used if at least one of its cells is. A condition is identified by (stage, test point), since test-point names repeat across stages with different settings.

### Setup (as D34 unless stated)
- **Condition sets (seed 36, assigned once from the check table before any model is run):** seven tuning conditions drawn per ageing type in proportion to its count (largest remainder); the other conditions are the test conditions, in **five outer folds** dealt round-robin after sorting by (ageing type, stage, ageing temperature, random key). Committed as `results/d36_condition_sets.csv`; the paper run refuses any other assignment. If fewer than 71 conditions survive the five-check rule, the same procedure applies to those that do.
- **Model, tuning, orders, seeds:** exactly D34: residual TFT + EWC streamed over the fold's training conditions, five orders, seeds 0–2 (main) and 5–7 (replicate); nested tuning on the tuning conditions (stage 1 learning rate {1e-3, 3e-3, 1e-2} with fine-tuning, stage 2 EWC λ {0.01 … 1000}), five tuning runs per grid point, one-standard-error rule and D11 edge rule (two extensions at most); the D25 calibrator fitted once after the stream on the pooled validation rows of the training conditions; the joint residual TFT per seed.
- **Comparators:** exactly D34: GBM-LAST (last condition of each evaluation order), GBM-ALL, linear extrapolation (D21 per-day rate over the last three check-ups × days to target), persistence, ridge.
- **Horizons:** E1 = 1–3 check-ups, E2 = 1–6.

### Questions and tests
- **Unit and rule:** as D34: per held-out condition and seed set, RMSE averaged over runs; paired two-sided Wilcoxon over test conditions; Holm within family and seed set; D21 replication rule (p_Holm < 0.05 in both seed sets, same sign, |mean difference| ≥ 0.05 pp in both); a question is answered positively only if replicated in favour in **both E1 and E2**.
- **Primary family:** Q1 SEL − GBM-LAST, all origins; Q2 SEL − linear extrapolation, H+ (on H+ SEL equals the D34 shrunk model, so Q2 replicates D34 Q2 on new data); Q3 SEL − GBM-ALL, all origins.
- **Secondary family:** Q4 SEL − shrunk (D34 rule), all origins (does the fix help on new data); SEL − raw; SEL − GBM-LAST on H+ and on H−; SEL − GBM-ALL on H+ and on H−; SEL − linear extrapolation, all origins; raw continual − joint; GBM-LAST − GBM-ALL.
- **Descriptive:** by ageing type, stage, ageing temperature and laboratory; per horizon; the worst condition per method; 90 % and 80 % coverage of the D25 intervals (centred on the shrunk forecast, as D34).
- **Scope:** first-life cells of one type (Samsung INR21700-50E) from one study run in three laboratories; laboratory is confounded with condition. A positive result supports generalisation to new usage conditions under the continual constraint, on a dataset not used to design the rule; it is not a second-life result.

### Compute budget
To be timed on synthetic data of S10 size before any real run (as D34). S10 has shorter series than S7 (calendar cells about 8 check-ups, cyclic cells 5–13), so fewer samples per condition; the D34 runtime (13 h on six workers) is the upper reference. The D34 fallback rule applies unchanged (18 h threshold).

### Not done, with reasons
- **Replay buffer** (the larger remaining cost in D34, the continual model being 0.66–1.03 pp behind the joint one): it changes the constraint and needs a memory-matched gradient-boosting comparator; a separate decision if wanted.
- **A curved extrapolation base** for long horizons: D29 (physics power law) was negative on S6; not repeated here.
- **Re-tuning the shrinkage itself** or choosing between SEL, shrunk and raw on S10: SEL is fixed here; the others are reported as secondary.

### Run log
- **4 Nov 2026. Pre-registered** (T. Sitaraman). S10 zips downloading with `scripts/fetch_stroebl.py` (raw files only); no check table, capacity or model on S10.
- **4 Nov 2026. Amendment (implementation, before any model was fitted on S10; T. Sitaraman).** Building the check table showed four things the entry did not fix: (1) the study has **279** cells, not 278; (2) the internal serial in `*_meta.txt` repeats across stages and in some zips names another test point, so a zip's stage is taken from its ET_T23 date (stage 1 began Sep-Oct 2021, stage 2 Dec 2022-Feb 2023, more than a year apart; the build checks that every (stage, cell) occurs once and exists in `experiments_meta.csv`); (3) three check-ups have no usable date (2 malformed, 1 missing meta file) and are invalid, never dated by guess; (4) three cells have metadata dates out of order (e.g. stage-1 TP_z25_01: check-ups 9-29 all carry the date of check-up 3), so in check-up order a check-up is kept only if its date is after the last kept one and before the final test (13 check-ups invalid). The registered label, temperature and five-check rules are unchanged. Result (`docs/EXTERNAL_DATA.md`, S10): 2,587 valid check-ups, 211 cells, all 71 conditions (2-3 cells each); 66 calendar cells have only ET and AT by design. Condition sets (`results/d36_condition_sets.csv`, seed 36): tuning 2 calendar and 5 cyclic conditions; test folds of 13 (fold 4: 12). Samples: 6,495 (E1) and 11,093 (E2); H+ 81 % and 78 %. No model has been fitted on S10.
- **4 Nov 2026. Code committed** after tests on synthetic data only: `scripts/build_stroebl_checks.py`, `scripts/run_d36.py` (a subclass of the D34 runner), `scripts/run_d36_night.py`, `tests/test_d36.py` (4 tests). `scripts/run_d34.py` was made reusable by a later decision (dataset hooks, method list and test families as class attributes; D34 defaults unchanged); re-collecting D34 from its checkpoints with the changed code reproduces both committed D34 results files exactly except the timestamp. **Compute check (synthetic data of S10 size: 71 conditions, 3 cells, 9-17 check-ups; paper profile, single-threaded):** one EWC stream over 51 conditions 1.7 min (E1) and 2.4 min (E2); one joint fit 3.5 and 1.2 min; about 0.8 times the D34 synthetic timings (D34 real run 13 h), below the 18 h threshold, so the fallback rule does not apply.
- **4 Nov 2026. Run** from `ea9f605` in a separate clean worktree of that commit with the main data directory (`DATA_ROOT`), pinned environment (Python 3.11.15): `python scripts/run_d36_night.py --workers 6`, 15:18 to 20:50 local. The first start was stopped by the host during the paper stage (low memory) after the smoke stage had passed; it was resumed from the checkpoints at 16:01 with the same command and commit. 40 smoke and 40 paper parts finished, none failed; logs in `data/cache/d36/logs/` (git-ignored).
- **4 Nov 2026. Results** (`results/forecasting_d36_e1.json`, `..._e2.json`, collected at `ea9f605`, clean tree; all five folds in the tests, no part not run; 64 test conditions; 6,495 E1 and 11,093 E2 samples). **Mean RMSE over the 64 conditions (pp), main / replicate:** E1 SEL 1.040 / 1.038, shrunk 1.076 / 1.061, raw 1.110 / 1.103, joint residual TFT 0.813 / 0.807, GBM-LAST 1.883, GBM-ALL 1.388, linear extrapolation 1.215, ridge 1.386, persistence 1.392; E2 1.542 / 1.552, 1.550 / 1.560, 1.592 / 1.598, 0.961 / 1.009, 2.382, 4.773, 1.900, 1.838, 2.181. **Primary verdicts (D21 rule over conditions):** Q1 SEL − GBM-LAST: E1 −0.842 / −0.845, E2 −0.840 / −0.830 (p_Holm 1e-11 in all four; 64 of 64 conditions better in each), **replicated, model better in E1 and E2**. Q2 SEL − linear extrapolation on H+: E1 −0.065 / −0.061 (p_Holm 3e-4 / 1e-3; 45 and 42 of 64), **replicated, model better**; E2 −0.190 / −0.147 (p_Holm 0.015 / 0.078; 39 and 36 of 64), **not replicated**. Q3 SEL − GBM-ALL: E1 −0.348 / −0.350 (p_Holm 8e-4 / 1e-3), labelled **replicated, model better** by the registered rule, which takes the direction from the mean difference; the rank test points the other way (median +0.174 / +0.154, GBM-ALL better on 52 and 51 of 64 conditions, rank-biserial +0.49), and the negative mean comes from a few calendar conditions where GBM-ALL fails (s1_k04-k06, s2_k05: SEL better by 7.6-9.6 pp). E2 −3.231 / −3.221 (p_Holm 0.99 / 0.99; 20 and 19 of 64), **not replicated**. **Registered answers (positive only if replicated in E1 and E2): Q1 positive, Q2 not positive (E1 only), Q3 not positive.** Q3 must always be reported with the mean-rank disagreement above; it is not evidence that the model beats GBM-ALL. Secondary: SEL − shrunk practically equivalent in E1 and E2; SEL − raw replicated better in E1 (−0.069 / −0.065), practically equivalent in E2 (−0.050 / −0.045); against GBM-LAST SEL is better on H+ (−0.94 / −0.93, −1.07 / −1.02) and on H− (−0.61 / −0.63, −0.31 / −0.36), all replicated; SEL is better than linear extrapolation on all origins (−0.17 / −0.18, −0.36 / −0.35, replicated); the continual raw model is worse than the joint one (+0.30 / +0.30, +0.63 / +0.59, replicated). Descriptive (main set, SEL / GBM-LAST / GBM-ALL / extrapolation): E1 calendar 0.75 / 1.73 / 2.22 / 0.85, cyclic 1.17 / 1.95 / 1.01 / 1.38; E2 calendar 1.39 / 2.15 / 12.53 / 1.48, cyclic 1.61 / 2.49 / 1.25 / 2.09. Pooled 90 % coverage of the shrunk model 0.74 (E1) and 0.43 (E2), below 0.85 (not adequate). **Tuning (must be stated with every result):** E1 lr 3e-4 / 1e-4 / 1e-3 / 1e-3 / 3e-3, λ 1e-6 / 1e-6 / 1e-4 / 1e-1 / 1e-2 (folds 0-4), best value at the final grid edge in folds 0 and 1; E2 lr 1e-3 / 1e-3 / 1e-3 / 1e-3 / 3e-3, λ 1e-6 / 1e-3 / 1e-4 / 1e-4 / 1e-2, no final edge. As on S7, λ is small in most folds, so EWC adds little penalty there. **Reading:** on a new dataset and on unseen conditions, the proposed model with selective shrinkage is more accurate than gradient boosting refitted on the latest condition on every condition in both experiments, and more accurate than linear extrapolation overall; gradient boosting with all conditions stored remains better on most conditions (cyclic ones in particular) but fails badly on several calendar conditions, and the joint residual TFT (which stores all data) is the most accurate method in both experiments.

## D37. Post hoc reading of the committed D25, D26, D34 and D36 results (exploratory, not reported)

**Status:** Done 9 Nov 2026 at the request of T. Sitaraman, who asked where the remaining error of the proposed model comes from and which changes could remove it. **Exploratory only:** nothing here is a reported result or may be cited as evidence. It is logged, as D33 was, so that it is on record that the test results of S6, S7 and S10 were read per condition before D38 was designed. No model was fitted and no data file was opened: every number below is recomputed from the committed results files (`results/forecasting_d25_e*.json`, `..._d26_e*.json`, `..._d34_e*.json`, `..._d36_e*.json`, main seed set unless stated). The scripts were not kept.

**What was read.**
1. **The continual constraint is the largest known cost.** Already registered as secondary tests: the raw continual model minus the joint residual TFT is +0.32 / +0.38–0.44 pp on S6 (D26, E1 / E2), +0.66 / +1.03 pp on S7 (D34) and +0.30 / +0.63 pp on S10 (D36).
2. **Which method is best per held-out condition (lowest RMSE of the nine or eight methods in the file).** S10 E1: joint residual TFT on 37 of 64 conditions, GBM-ALL on 21; E2: 51 and 7. S7 E1: GBM-ALL on 43 of 60, joint residual TFT on 4; E2: 45 and 12. Medians (pp): S10 E1 joint 0.354, GBM-ALL 0.353, SEL 0.552; S10 E2 0.511, 0.832, 1.134; S7 E1 joint 2.607, GBM-ALL 2.706, shrunk 4.106.
3. **EWC strength in the nested tuning (20 fold × experiment cases on S7 and S10).** The selected λ is at most 0.01 in 18 cases. The best λ of stage 2 is within one standard error of plain fine-tuning at the selected learning rate (stage 1), or worse than it, in 15 cases; in 5 cases a small or moderate λ is better by more than one standard error. λ = 1000 is worse than the best λ in all 20.
4. **Calibration rows.** Known from the registered results: shrinkage fitted on the pooled validation rows hurts at H− origins on S7 (D34), and the 90 % intervals cover 0.71 / 0.57 (S7, E1 / E2) and 0.74 / 0.43 (S10).
5. **Run-to-run spread on S6 (D25, D26), split by `order_id`.** Stream, shrunk model, E1: mean 0.318 pp, s.d. over seeds at a fixed order 0.095 pp, s.d. of the order means 0.066 pp; E2: 0.498, 0.066, 0.068. Unseen conditions, raw model, E1: 0.885, 0.071, 0.126; E2: 1.595, 0.143, 0.142. The joint model: 0.03–0.05 pp over seeds.
6. **One S10 condition dominates the means.** On `s2_z02` (cyclic, 45 °C) every forecaster built on a trend is at 17–19 pp in E1 (joint 17.6, SEL 18.3, linear extrapolation 18.6, GBM-LAST 18.4), GBM-ALL at 12.0 and persistence at 4.6. Without this one condition the E1 means are joint 0.547, SEL 0.766, linear extrapolation 0.939, GBM-ALL 1.219 pp. Whether its check-up series is a real capacity jump or a data problem has not been looked at (D36 found out-of-order dates in three S10 cells).

**Reading (not evidence).** The error left in the proposed model is mostly not in the network: the same network trained with all conditions stored is the most accurate method on most S10 conditions and second on S7 and S6. What separates the two is the stream. EWC as tuned adds little to it on long streams, part of the spread is seed noise that averaging removes, and the calibrator is fitted on rows that do not resemble an unseen condition.

**Consequence.** D38 tests one model that changes these three things. Because the S7 and S10 test conditions were read here, D38 on those datasets is a post hoc development test, however its protocol is fixed; a confirmatory test needs data not used so far.

## D38. Residual TFT with experience replay, a seed ensemble and predict-before-training calibration on unseen usage conditions (S10, then S7; P2) (protocol fixed in advance; post hoc on conditions already tested)

**Status:** Drafted 9 Nov 2026 on the instruction of T. Sitaraman (answers of 9 Nov: a small stored memory is allowed if it is the better design; the proposed model may change to a residual TFT with replay, EWC optional; the aim is a model that beats the comparators in most settings). The commit that adds this entry precedes all D38 code. **It becomes binding when T. Sitaraman accepts the defaults under "Open points", recorded in the run log before any D38 code runs on real data.** Changes before that are amendments; changes after the first real-data run are deviations. Code (to be written): `scripts/run_d38.py`, `scripts/run_d38_night.py`, `tests/test_d38.py` (synthetic data only), runbook `docs/D38_NIGHT_RUN.md`; nothing under `src/` changes, so the D34 and D36 checkpoints stay valid. Results: `results/forecasting_d38_s10_e1.json`, `..._s10_e2.json`, `..._s7_e1.json`, `..._s7_e2.json`.

**Provenance (must be stated with every D38 number).**
- Post hoc, and motivated by the wish for a model that beats the traditional forecasters. Designed after every D23–D36 result and after the per-condition reading of the S7 and S10 test results in D37.
- The test conditions of D38 are those of D34 (S7) and D36 (S10). They have been tested once with the earlier model and read per condition. D38 is therefore **not confirmatory**: it is a development test with a protocol fixed in advance. A confirmatory test on a dataset not used so far is planned as a separate decision (see "Not done").
- Every design value below is fixed here from first principles or from earlier decisions, not chosen on the test conditions. The only tuned value is the learning rate of the replay model, selected on the seven tuning conditions of each dataset, which have never been tested.

**Model under test (V2), fixed here.** The D23 residual TFT (same network, inputs, window, residual on the D21 per-day linear extrapolation), with three changes:
1. **Experience replay instead of EWC.** After a condition is learned, `B` of its training rows and `B/4` of its validation rows are kept (seeded random draw). While a later condition is learned, every mini-batch loss is the mean squared error on the current batch plus, with weight 1, the mean squared error on a batch of the same size drawn from the stored training rows; early stopping uses the current condition's validation rows together with the stored validation rows. No EWC penalty. **Primary `B` = 32 rows** (about one cell's rows per condition for horizons 1–3); `B` = 8 is the smaller-memory arm.
2. **Seed ensemble.** The forecast is the mean of the forecasts of the three models of a seed set (main: seeds 0–2; replicate: seeds 5–7), trained on the same stream in the same order. Orders are not ensembled: a real stream has one order.
3. **Predict-before-training calibration.** Before the model is trained on the condition at position *t* of the stream (*t* ≥ 2), it forecasts every row of that condition (its training and validation rows) with the weights it has then; the ensemble forecast and the target of each such row are kept (two numbers per row, no inputs). After the stream, on the rows of the **second half of the stream** (positions above half its length), per horizon and origin stratum (H+: linear extrapolation has a fade rate; H−: it does not):
   - the shrinkage factor is the D25 formula, α = clip(Σ p·y / Σ p², 0, 1), on the residual scale;
   - the 90 % and 80 % half-widths are the split-conformal quantiles (D25 rule) of |y − α·p|.
   A stratum with fewer than 30 such rows uses its horizon's rows of both strata; a horizon with fewer than 30 keeps α = 1 and is flagged. The forecast is linear extrapolation + α × the ensemble correction; the interval is the forecast ± the half-width.
- **Why these three (reasons, not evidence).** (1) With every row stored, replay is joint training, whose error is already measured (D34, D36), so the memory arm moves the model between two measured end points; EWC tuned to almost zero on these streams (D34, D36 run logs). (2) The squared error of a mean forecast equals the mean squared error of its members minus their spread, so an ensemble is never worse in mean squared error than its average member. (3) Split-conformal intervals cover at the nominal level on average over new conditions that are exchangeable with the calibration conditions, and a condition forecast before it is trained on is, for the model, a new condition; the folds of D34 and D36 were drawn at random from one pool. The earlier calibrator used mid-life validation rows of conditions already learned (D26 reading 5, D27 Q3).
- **What is stored per learned condition:** 40 rows (32 training, 8 validation) and two numbers per forecast row, against two vectors of 12,708 numbers per condition for the multi-penalty EWC of D23. Reported with the results; no claim rests on it.

**Setup (as D34 for S7 and D36 for S10 unless stated).**
- **Data, labels, inputs, splits, condition sets, folds, orders, seeds:** unchanged and read from the committed files (`results/d34_condition_sets.csv`, `results/d36_condition_sets.csv`): five outer folds, the fold's other test conditions as the stream, five evaluation orders, seeds 0–2 (main) and 5–7 (replicate), E1 = horizons 1–3 and E2 = 1–6. The check tables are used as committed; the audit of `s2_z02` (D37, item 6) does not change them within D38.
- **Arms per (fold, order, seed):**
  - **A0:** the D34 / D36 model unchanged (residual TFT + EWC, the learning rate and λ recorded per fold in the committed D34 / D36 results files; no re-tuning), with the forecasts kept. Gives the earlier proposed model (shrunk on S7; SEL on S10), the reproduction check, and the ensemble without replay.
  - **R32:** the replay model with `B` = 32.
  - **R8:** the replay model with `B` = 8, at R32's learning rate.
- **Tuning (replay learning rate only, per fold):** streams with `B` = 32 over the fold's training conditions in the five D34 tuning orders, seed 0, scored zero-shot on the seven tuning conditions (mean of per-condition RMSE, single raw model); grid {1e-3, 3e-3, 1e-2}; D21 one-standard-error rule and D11 edge rule (at most two extensions), as D34 stage 1.
- **Reproduction check (stop condition):** per held-out condition and seed set, the A0 arm's mean RMSE over its 15 runs (raw and shrunk; SEL on S10) is compared with the committed D34 / D36 value. A largest absolute difference above 0.05 pp (the D30 tolerance for neural runs across platforms) stops collection; the difference is recorded in any case. GBM-ALL, GBM-LAST, linear extrapolation, persistence and ridge are recomputed with the D34 code and compared the same way (tolerance 1e-6 pp).
- **Comparators:** GBM-ALL, GBM-LAST, linear extrapolation, persistence, ridge (D34 definitions); the joint residual TFT (per-condition values read from the committed D34 / D36 files, not refitted); and **GBM-ER** (new): gradient boosting (D18 settings) fitted on the training rows of the last condition of each evaluation order plus the rows a seed-0 replay model holds at that point (the stored training and validation rows of every earlier condition), for `B` = 32 and `B` = 8. It is the comparator with the same stored rows that D34 and D36 listed as not done.
- **Pinned environment, clean tree, checkpoints, smoke stage before the paper stage:** as D34.

**Questions and tests.**
- **Unit and rule:** as D34. Per held-out condition and seed set, the RMSE of a method averaged over the five orders (ensemble methods, GBM-ER, GBM-LAST) or the 15 runs (single models); paired two-sided Wilcoxon over the test conditions; Holm within family and seed set; D21 replication rule.
- **Direction rule (new, fixed because of the D36 Q3 case):** a comparison counts in favour of V2 only if the registered verdict is "replicated" with a negative mean difference **and**, in both seed sets, the median difference is negative and more conditions favour V2 than the comparator. A replicated mean with a median or count pointing the other way is reported as "mean and rank disagree" and is not a win.
- **Primary family (per dataset and experiment), V2 = R32 ensemble, calibrated:**
  - Q1: V2 − GBM-ALL, all origins (all conditions' data stored).
  - Q2: V2 − GBM-ER with `B` = 32, all origins (the same stored rows).
  - Q3: V2 − linear extrapolation, H+.
  - Q4: V2 − the earlier proposed model (S7: shrunk; S10: SEL; single-run mean), all origins.
- **Q5 (coverage, no test):** pooled 90 % coverage of V2's intervals at least 0.85 in both seed sets (the D26 threshold); 80 % coverage, interval score, width and coverage by ageing type are reported.
- **A question is answered positively for a dataset only if it holds in E1 and in E2.** S10 and S7 are reported separately and never pooled.
- **Secondary family:** V2 − joint residual TFT; V2 − GBM-LAST; V2 − ridge; V2 − GBM-ALL on H+ and on H−; attribution in steps: A0 ensemble − A0 single raw (ensemble alone), R32 single raw − A0 single raw (replay alone), R32 ensemble − R32 single raw, V2 − R32 ensemble uncalibrated (calibration alone); R8 ensemble calibrated − V2 (memory size); R8 ensemble calibrated − GBM-ER with `B` = 8.
- **Descriptive:** by ageing type, temperature, stage and laboratory (S10); per horizon; the worst condition per method; the number of conditions on which each method is best; every mean also without `s2_z02` (S10), named here in advance because D37 found it dominates the means; shrinkage factors and half-widths; coverage per evaluation order (order 0 is the table order, where the second half of the stream is not a random half).

**What is expected, stated in advance.** Q5 is the most likely to be met, because of what split-conformal calibration guarantees. Q2, Q3 and Q4 are likely on S10. Q1 is open on S10: V2 must recover most of the distance to the joint model, which is itself better than GBM-ALL on a majority of conditions there. On S7, GBM-ALL was best on 43–45 of 60 conditions and ahead of the joint model, so Q1 is not expected to be met there. Every outcome is reported; nothing is re-run to change one.

**Scope of any claim.** First-life cells; S10 one cell type in three laboratories, S7 one cell type in one laboratory; not a second-life result. V2 keeps rows of earlier conditions, so it is compared as a bounded-memory method: the "no earlier data kept" statements of D32, D35 and the paper's headline table apply to the EWC model, not to V2.

**Compute (to be checked on synthetic data of S10 size before any real run).** A replay stream costs about twice a plain stream (two batches per step). Per experiment and dataset: 75 tuning streams, and 150 streams for each of A0, R32 and R8, in units of the D36 timings (about 1.8 min per plain stream on average) roughly 25–30 CPU hours, so **about 9–11 h with six workers for S10** (both experiments) and about twice that for S7. The parts are run in the order tuning, R32, A0, R8, each arm as its own parts, so a stopped run still has the primary arm. Fixed now: if the synthetic check estimates more than 14 h with six workers for S10, R8 is not run on S10 and is reported as not run; nothing else changes. S7 starts only after S10 has been collected.

**Not done, with reasons.**
- **EWC together with replay:** would double the replay arms; EWC tuned to almost zero on these streams. A separate arm if wanted.
- **A robust base in place of linear extrapolation, or online weighting of forecasters:** D33 found online combination adds at most 0.008 pp on S6; a robust base would be designed on `s2_z02`, one condition. Not part of V2.
- **Tuning `B`, the replay weight, the ensemble size or the calibration half:** fixed here; `B` = 8 is reported as a second arm, not a choice.
- **The continual stream (P1) on S7 and S10:** never run; it is the setting where gradient boosting was weakest on S6. A separate decision after D38.
- **Confirmatory test on new data:** to be registered separately once a dataset passes the `docs/EXTERNAL_DATA.md` checks. Leads noted 9 Nov 2026, licences **not yet verified**: Wildfeuer et al. 2023 (TUM, mediaTUM record 1713382 with update doi:10.14459/2024mp1734374) and a UConn LFP first- and second-life dataset (digitalcommons.lib.uconn.edu/reil_datasets/3).

**Open points (defaults stand unless T. Sitaraman changes them before the first real-data run).**
1. Primary memory `B` = 32 training rows and 8 validation rows per condition; second arm `B` = 8 and 2.
2. Ensemble of the three seeds of a seed set (no extra training runs).
3. S10 first, S7 second; S7 only after S10 is collected.
4. The audit of `s2_z02` is read-only and does not change the S10 check table within D38.

### Run log
- **9 Nov 2026. Drafted** after D37. No D38 code written or run; no data file opened in this session (the session has no datasets).
- **9 Nov 2026. Code committed** after tests on synthetic data only (pytest temporary directories and the session scratch space; never real data): `scripts/run_d38.py`, `scripts/run_d38_night.py`, `tests/test_d38.py` (10 tests), runbook `docs/D38_NIGHT_RUN.md` with the prompts for the data machine; nothing under `src/` changed. **Checks on synthetic data:** the a0 arm equals the D36 continual unit exactly (raw, shrunk and SEL RMSE per held-out condition, largest absolute difference 0.0), so the reproduction check should be exact on the machine that ran D36; with a base results file present, the recomputed GBM-ALL, GBM-LAST, linear extrapolation, persistence and ridge agree with it to 0.0. **Implementation choices fixed with the code:**
  - The replay batch has the size of the current batch (fewer if fewer rows are stored) and is drawn anew at every step with a generator seeded per condition; which rows are stored depends only on (seed, position of the condition in the stream).
  - "Second half of the stream" is the 0-based positions from ⌊T/2⌋ on, T being the number of training conditions.
  - A flagged horizon (fewer than 30 calibration rows) takes its half-widths from the rows it has.
  - Per held-out condition, an ensemble method's value is the RMSE of the ensemble forecast averaged over the five orders; its coverage pools the samples of the five orders.
  - GBM-ER is fitted on the last condition's training rows plus the stored training and validation rows of the earlier conditions (seed 0).
  - The joint residual TFT is read from the committed D34 / D36 `condition_values` (not refitted).
  - The folds in the tests are those where r32 finished in both seed sets; another arm enters only if it finished for all of them; comparisons that need a missing arm are listed under `tests_not_run`.
  - The a0 arm also gives SEL on S7 and a calibrated a0 ensemble (`tft_residual_ewc_ens_cal`); both are descriptive only.
  - Collection stops if the reproduction check fails; `--record-mismatch` writes the results anyway and is a deviation to log here.
  - The stream code of `run_d38.py` enters the checkpoint key (as `src/` does); the collection code does not.
  - Smoke profile: 2 orders, 2 seeds per set, 2 tuning runs, 2 epochs.
- **9 Nov 2026. Compute check and amendment to "Compute" (before any real-data run).** On synthetic data of S10 size (71 conditions, 3 cells, 9–17 check-ups; paper profile; one stream over 51 conditions, cloud session): a0 0.98 min (E1) and 1.13 min (E2), r32 0.73 and 0.87 min, r8 0.60 and 0.87 min. The entry's assumption that a replay stream costs about twice a plain stream was wrong: it costs no more than the EWC stream, whose penalty sums one term per learned condition. D38 has 525 streams per experiment on S10, as D36 had (5.5 h real time with six workers), so about 5–7 h is expected for S10, below the 14 h threshold: **the fallback does not apply and r8 is run.** No D38 code has been run on real data.
- **11 Nov 2026. Defaults accepted; D38 binding.** T. Sitaraman accepts the four defaults under "Open points" unchanged (memory 32 + 8 rows per condition with a second arm of 8 + 2; ensemble of the three seeds of a seed set; S10 first, S7 only after S10 is collected; the `s2_z02` audit read-only). Checks on the data machine at `ac4509a`: `tests/test_d38.py`, `tests/test_d36.py` and `tests/test_d34.py` pass (19 tests) in the pinned `.venv` (Python 3.11.15); `scripts/run_d36.py --write-sets` leaves the tree clean; `scripts/run_d38_night.py --dataset s10 --dry-run` reports 70 parts and "pre-flight passed". No D38 code has run on real data yet: the dry run trains nothing.
