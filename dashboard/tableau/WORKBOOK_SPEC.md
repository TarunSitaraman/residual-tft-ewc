# Tableau Public workbook: specification

This is the build sheet for the Tableau Public workbook embedded by `dashboard/`. Author it in **Tableau Desktop Public Edition** (free, Windows or macOS), then publish it to Tableau Public. The JS page relies on the names in **§1 Contract**; everything else is a design guide.

Inputs come from `python scripts/export_dashboard_data.py`:

| Folder | Files | In Git |
|---|---|---|
| `results/dashboard/` | `dim_group`, `dim_method`, `fact_stream_run`, `fact_stream_task`, `fact_stream_baseline`, `fact_unseen_run`, `fact_unseen_fold`, `fact_tests`, `fact_verdicts`, `fact_coverage`, `fact_horizon`, `fact_d30_group`, `fact_headline` (CSV) | yes |
| `data/dashboard/` | `fact_soh`, `fact_predictions` (CSV, built from the data folder) | no |

**Starting point:** `python scripts/build_tableau_workbook.py` writes `data/dashboard/tableau/tableau_source.xlsx`, with one self-contained, pre-filtered sheet per view, and `residual_tft_ewc.twb`, a starter workbook that implements §1–§3 as generated XML. Open the `.twb` and correct it against this spec rather than building from scratch. It was generated without Tableau, so expect to fix sorts, mark sizes and colours.

Use **extracts** (Tableau Public has no live connections). Commit the `.twb` to `dashboard/tableau/residual_tft_ewc.twb` once it exists. Keep the `.twbx` and its extract out of Git, like the data.

## 1. Contract with the JS page

| Item | Name the page expects | Notes |
|---|---|---|
| Dashboard 1 | `Overview` | URL goes in `TABLEAU.views.overview` (`dashboard/config.js`) |
| Dashboard 2 | `SOH explorer` | `TABLEAU.views.soh` |
| Dashboard 3 | `Forecast accuracy` | `TABLEAU.views.accuracy` |
| Dashboard 4 | `Continual learning` | `TABLEAU.views.continual` |
| Dashboard 5 | `Reuse & uncertainty` | `TABLEAU.views.reuse` |
| Field | `Dataset` | rename `dataset`; values `S1`, `S6` |
| Field | `Setting` | rename `setting`; values `e1`, `e2` (alias them "E1 (h = 1-3)", "E2 (h = 1-6)") |
| Field | `Protocol` | rename `protocol`; values `P1`, `P2` |
| Field | `Group` | rename `group`; values `g1`…`w10`, `case1`…`case7` (and `all` for pooled rows) |
| Field | `Method Family` | `family` from `dim_method`, joined on `method` |

The page calls `applyFilterAsync(<field>, [value], "replace")` on every worksheet of the active dashboard, or `clearFilterAsync` when "All" is chosen. A worksheet without the field is skipped. A parameter with the same name is set too. When a mark is selected, the page reads its fields, and a `Group` field opens the details panel. **Every per-group view keeps `Group` in the view, on a shelf or on Detail.** The per-group views are the error heatmap, D30 RMSE, SOH trajectories, backward-transfer heatmap, interval coverage and parity plot. Method-level summaries (headline, horizon, adaptation vs retention, forgetting by arrival, confusion matrix) aggregate over groups and must not be split by `Group`.

Publish each dashboard with "Show sheets as tabs" off. Copy each dashboard's share URL (`https://public.tableau.com/views/<Workbook>/<Dashboard>`) into `dashboard/config.js`.

## 2. Data model

- Relate (Tableau relationships, not joins) every fact table to `dim_method` on `method`, and to `dim_group` on `dataset` + `group`.
- Order groups by `dim_group.group_order` (sort the `Group` field by this measure, ascending). Never sort groups by their error.
- `dataset` and `protocol` are already on every S6 fact table. The tables built from `results/tables` carry `dataset`.

Calculated fields:

| Name | Formula | Used in |
|---|---|---|
| `Skill vs persistence` | `1 - [rmse] / {FIXED [Dataset],[Setting],[Group]: MIN(IF [method]="persistence" THEN [rmse] END)}` | accuracy |
| `Is proposed` | `[is_proposed]` | colour emphasis |
| `Reuse class band` | `IF [soh_pct] > 95 THEN "C0" ELSEIF [soh_pct] > 90 THEN "C1" ELSEIF [soh_pct] > 80 THEN "C2" ELSE "C3" END` | SOH explorer reference bands (the same rule as `src/evaluation.py`; the exported `reuse_class` column is authoritative) |
| `Retention gap` | `[final_error] - [diagonal_error]` | continual (equals forgetting on the last step) |
| `Within one class` | `ABS([class_distance]) <= 1` | reuse |

## 3. Dashboards

Shared rules: one y-axis per chart (never dual-axis); baselines (persistence, linear extrapolation, gradient boosting) visible wherever a learned model is shown; the proposed model is drawn in the accent colour (`#2a78d6`) and everything else in neutral greys or in the fixed family order below; per-group views use `group_order`; tooltips show value, s.d. and run count.

Family colour order (fixed, never cycled; taken from `scripts/make_d24_outputs.py`): TFT `#2a78d6`, MLP `#eb6834`, BiLSTM `#1baf7a`, TCN `#eda100`, Transformer `#e87ba4`, baselines grey `#8a8981`, GBM `#4a3aa7`.

1. **Overview**
   - *Headline table* (`fact_headline`): rows method, columns `protocol` × `setting`, text `rmse_main` / `rmse_replicate`. The verdict rows below come from the rows with `diff_main`.
   - *Verdict strip* (`fact_verdicts`): for the selected setting, a count of verdicts by experiment.
2. **SOH explorer** (`fact_soh`)
   - Lines of `soh_pct` against `check_index` (S1: RPT number; S6: cycle), one line per `unit`, coloured by `group`, faceted by `Dataset` (S1 and S6 are never on one axis).
   - Reference bands at 95 / 90 / 80 % SOH labelled C0–C3. Axis title: "SOH, % of first valid check".
   - Mark invalid checks (`valid = false`) as hollow points with no line. Tooltip `temperature_source`; missing temperatures are shown as "missing, not filled".
3. **Forecast accuracy**
   - *Heatmap* (`fact_unseen_fold` mean `error` by method × `Group`; P1 equivalent from `fact_stream_task.final_error`). Sequential single hue; label each cell.
   - *Horizon lines* (`fact_horizon`): `rmse_pp` against `horizon`, one line per method.
   - *D30 metrics* (`fact_d30_group`): pick `metric` with a parameter; bars per `Group` for the S1 cells and S6 conditions.
4. **Continual learning** (`fact_stream_task`, `fact_stream_run`)
   - *Adaptation vs retention*: scatter of mean `diagonal_error` (adaptation, error on the newest task) against mean `final_error` (retention, error on that task after the whole stream), one point per method; diagonal reference line.
   - *Forgetting by arrival position*: mean `forgetting` against `arrival_pos`, lines per strategy (EWC, fine-tuning, SA-EWC).
   - *Backward transfer heatmap*: `Group` × `arrival_pos`, mean `backward_transfer` on a diverging scale with grey at zero.
   - Seed spread: box plots of `final_avg_error` by method, split by `seed_set` (main / replicate).
5. **Reuse & uncertainty**
   - *Confusion matrix* (`fact_predictions`): `class_true` × `class_pred`, count, filtered to one method.
   - *Parity plot*: `soh_pred` against `soh_target`, with the 1:1 line.
   - *Coverage* (`fact_coverage`): `coverage` per `Group` at `level` 0.9, by calibration rule, with a reference line at 0.9.
   - *Within one class*: share of `Within one class` by method.

## 4. Publishing checklist

- [ ] Title and caption on every dashboard: "SOH relative to each cell's first valid capacity check".
- [ ] Attribution text on every dashboard: Moy et al. 2024 (S1, CC BY 4.0); Wheeler et al. 2025 (S6, CC BY 4.0); list of changes in `docs/EXTERNAL_DATA.md`.
- [ ] Field names of §1 present; `Group` in every per-group view.
- [ ] Dashboard size "Automatic" (or a range of 800–1400 px wide) so the embed fills the page.
- [ ] URLs pasted into `dashboard/config.js`, then the page checked (dashboard/README.md, "Verify").
