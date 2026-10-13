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

Literature error ranges (e.g. RMSE 0.6–1.5 %) come from other datasets and SOH definitions and are context, not targets (see `CLAUDE.md`).

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
