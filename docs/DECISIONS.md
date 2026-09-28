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

**Implemented in:** `src/evaluation.py` (`REUSE_CUTPOINTS`, `REUSE_CLASS_NAMES`, `reuse_class`). The E10 evaluation itself is planned (W7).

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
- The learning rate is tuned per backbone and λ per method under D11.

Evaluation follows `docs/RESEARCH_PLAN.md` Stage D: TFT + EWC against MLP + EWC (architecture), TFT fine-tuning (EWC), MLP replay and LwF (other families), and joint training on both backbones (upper bounds), on identical orders and seeds. Outcomes are reported as measured.

`pytorch-forecasting` and `pytorch-lightning` were removed from `requirements.txt` because nothing uses them.

**Implemented in:** `src/models/tft_continual.py` (`TemporalFusionTransformer`, `CycleWindows`), `src/continual_learning.py` (`TFTMixin`, `TFTFineTune`, `TFTEWC`, `TFTJoint`), `src/experiments.py`, `notebooks/07_tft_ewc.ipynb`, `tests/test_tft.py`.
