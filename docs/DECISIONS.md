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
  - all 112 RPT capacities are reproduced exactly from the raw files;
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

## D21. Extended data and protocol for TFT-scale training — DRAFT, not adopted

**Status:** **Draft for team review (3 Oct 2026). Nothing in this entry may be run until the team adopts it**; adoption is recorded here with its date, and any change after adoption is logged as a deviation. This entry is written **after** the S1 and S6 results, the reproducibility check (`docs/RESULTS.md`) and the data audit were seen, and must be reported as such. No model has been fitted on S7–S9.

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

### Inputs
- **Shared across all datasets (pool and stream):** SOH at origin; fade rate (least-squares slope over the last three valid checks); horizon *h*; time since the first check and time to the target, **in calendar days and in equivalent full cycles** where the dataset records them (missing-value indicators otherwise); test temperature; charge and discharge C-rates; SoC window or DoD; chemistry (one-hot). For S6 this adds calendar days to target, which is the change motivated by finding 3; it is applied identically to every method.
- **Dataset-specific diagnostics** (S1 F1, S6 F1) are used only within their own dataset, as before.
- **Never inputs:** protocol currents set from a previous capacity (D14), throughput scaled from capacity, any label after the origin. Each new dataset's columns are screened with `scripts/audit_data.py` rules before use, and the screen is committed with the data tables.

### Splits and protocols
- Per-cell purged 60/80 splits within every dataset, as D18.
- **P1, P2 on S1 and S6:** unchanged, re-run with the inputs above.
- **P3 (new, headline for the TFT):** pretrain on the pool's training splits (early stopping on the pool's validation splits), then learn the S6 test cases as a continual stream (10 orders × 5 seeds, as D20). Methods: TFT from scratch; TFT pretrained + fine-tuning; **TFT pretrained + EWC** (anchor = pretrained weights, Fisher information from the pool); the same three for the MLP; the D18 baselines. Metrics: the D18 continual-learning metrics on S6 **plus retention on the pool's test splits** after the stream (forgetting of first-life knowledge).

### Tuning (changed from D11)
- **10 tuning runs per grid point** instead of 3, on tuning orders disjoint from the evaluation orders.
- **Tie rule, fixed now:** if the best grid value is within one standard error (over tuning runs) of a neighbour, choose the smaller learning rate or the smaller EWC λ. The selection margin is reported with every selected value.

### Statistics (changed from D18)
- As D18 (paired Wilcoxon, Holm correction, rank-biserial effect sizes, bootstrap CIs), **plus medians** of every paired difference.
- **Replication rule:** a difference is called significant only if p_Holm < 0.05 in the main run **and** in a replicate run with a different, pre-specified seed set (seeds 5–9). Differences below 0.02 pp are reported as practically equivalent whatever their p-value.
- Every results file records the full environment, including torch.

### S6 ageing cycles (route 1), deferred
The between-check ageing files of S6 (827 files, 23.2 GB compressed, 132 GB uncompressed; measured from the zip indexes on 3 Oct 2026) would give per-cycle covariates. They are **not** part of this entry: the download takes about 5 days on the current connection, and in the 70 % DoD cases the ageing discharge is set from the last C/3 charge capacity (Wheeler et al. 2025), which needs its own leakage rules. A separate decision entry will cover them.

### Limitations stated in advance
- The pool is first-life data under laboratory conditions; transfer to second life is tested only on S6 (aged at 50 °C) and S1.
- S7 check-ups are at a single temperature (25 °C); S8 and S9 labels are cycling-rate capacities, not slow-rate checks.
- Pseudo-checks (S8, S9) and check-ups (S7) differ in interval; time-to-target inputs carry this.
