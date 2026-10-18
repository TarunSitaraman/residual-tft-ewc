# Work plan: from the current state to a submittable paper

Written 12 Oct 2026 after an audit of the repository at `8b65384`. This plan says what is still missing for the three deliverables (big-data preprocessing results, the proposed deep-learning model, and a comparison with at least four existing ML/DL models with tables and figures), in what order to do it, and how to tell when each item is done. Rules from `CLAUDE.md` and `AGENTS.md` apply to every item: no fabricated or synthetic data in results, battery groups kept separate, chronological purged splits, protocols written into `docs/DECISIONS.md` before the runs they govern, every reported number traceable to a clean commit.

## Decisions and status (updated 14 Oct 2026)

| Question | Decision |
|---|---|
| Proposed model | **RC-TFT**: residual TFT with **shift-adaptive EWC** (the TFT + EWC hybrid), pre-registered as D24. The D23 fixed-λ model is its ablation. **Outcome (14 Oct):** SA-EWC's positive-result criterion was not met (practically equivalent to fixed λ on error and forgetting, E1 and E2), so under D24 the fixed-λ residual TFT + EWC stays the best-supported variant; RC-TFT is reported with its null result (`docs/RESULTS.md`, D24). |
| Name | *Residual Continual TFT*: R = forecasts the correction to linear extrapolation; C = continual training with shift-adaptive EWC; TFT = the (cited) backbone. |
| Compute | One night on the ASUS (`docs/D24_NIGHT_RUN.md`); parts in a fixed priority order, anything unfinished reported as not run. |
| Licence | Proprietary, all rights reserved, while the repository is private (`LICENSE`). |
| Paper | Kept long for now; cut later. Results come first. |
| Next step for the model | **D25** (pre-registered 14 Oct): shrink the learned correction per horizon, add conformal forecast intervals, and pretrain the residual model on the first-life pool; one night on the ASUS (`docs/D25_NIGHT_RUN.md`). Plan: section 6 below. |

| Item | Status |
|---|---|
| WS1 docs fixes | Done: README references synced with the bibliography, licence, stub removed, paper `\todo` resolved (`paper_single.pdf` still needs a LaTeX rebuild) |
| WS2.1 preprocessing table | Done from committed files (`scripts/make_preprocessing_table.py`); per-dataset pool counts are added when run on the data machine |
| WS2.2 Spark scaling | Run 13–14 Oct (`results/d22_spark_scaling.json`); outputs match pandas, but the 1-thread time looks inflated (super-linear 1 → 2 speed-up), to be explained before reporting |
| WS3 literature baselines | Done: run 13–14 Oct, all parts finished (`results/forecasting_d24_e1.json`, `..._e2.json`) |
| WS4 shift-adaptive EWC | Done: run 13–14 Oct; null result (practically equivalent to fixed λ) |
| WS5 tables and figures | Done from the D24 files (`results/tables/d24_*`, `paper/tables/d24_*`, `paper/figures/d24_*`) |
| WS6 P2 for new models | Not in tonight's budget |
| D25 calibrated residual TFT + EWC, residual pretraining | Run 15–16 Oct, all 10 parts finished (`results/forecasting_d25_e1.json`, `..._e2.json`); no registered positive result: the shrunk model is practically equivalent to linear extrapolation, pretraining is worse (E1) or not replicated (E2), intervals under-cover (`docs/RESULTS.md`, D25) |
| D26 P2 for the residual TFT + EWC, by origin history | Run 17–18 Oct, all 28 parts finished (`results/forecasting_d26_e1.json`, `..._e2.json`): beats linear extrapolation on H+ origins (Q1, replicated); gradient boosting better (Q2, replicated); joint training better than the continual stream (replicated); intervals under-cover, 0.66–0.68 at 90 % (Q3 not met) (`docs/RESULTS.md`, D26) |
| WS7 paper | D24 added to methods, results, discussion, conclusion and abstract (14 Oct); `paper_single.pdf` rebuilt with tectonic 0.15; **9 pages, one over the 8-page limit** (about half a page of references spills over): cuts still to be chosen |

## 0. Where we are (audit summary)

| Deliverable | Status | Evidence | Gap |
|---|---|---|---|
| Big-data preprocessing | **Done; presentation incomplete** | 827 raw S6 ageing files (131.9 GB, 901,974,259 rows) → 82,529 cycles (97 % valid) → 827 blocks × 21 inputs with PySpark 4.2; Spark matches pandas on all 827 blocks; leakage audit; SHA-256 manifests (`results/d22_*.json`, `results/s6_ageing_row_count.json`) | No single preprocessing-results table across all datasets; no scaling measurement (time vs cores); `results/figures/` is git-ignored although `docs/RESULTS.md` points to it |
| Proposed DL model: **TFT + EWC** | **Implemented and evaluated; novelty framing weak** | D12 (model), D18/D20 (P1, P2), D21 (P3), D23 (residual form). Residual TFT + EWC is the most accurate learned model on S6 (0.339 / 0.314 pp, h = 1–3) and EWC cuts its forgetting by about two thirds at h = 1–6 (replicated) | TFT (Lim 2021) and EWC (Kirkpatrick 2017) are existing methods, and the README says so. The paper must state precisely what is new about *our* TFT + EWC. The registered goal of beating linear extrapolation was not met |
| ≥ 4 existing models compared | **Met in number, weak against the literature** | Persistence, linear extrapolation, ridge, gradient boosting, MLP (fine-tuning, replay, EWC, joint), TFT direct | None of the families in our own literature review is implemented: GPR (refs 3, 4), LSTM/BiLSTM (ref 10), TCN (refs 8, 10). No consolidated comparison table, no per-group table in the paper, no literature-context table |

Other findings: the README reference list disagrees with the Crossref-checked `paper/references.bib` (refs 3, 6, 7, and claims for refs 3 and 9 that the paper says could not be verified); the paper still has one `\todo` (repository visibility and licence) and the repository has no `LICENSE`; `src/models/reuse_head.py` is a stub. Tests: 148 pass, 2 skip without `data/wheeler_checks.csv`.

**Constraint.** The datasets are not in Git. Every new run (WS3, WS4, WS5) needs the machine that holds `data/` (or a fresh fetch, about 27 GB for S6). Code, tests and smoke runs on synthetic fixtures can be done anywhere; paper-profile runs only on the data machine.

## 1. The proposed model: what we claim and what we add

The proposed model stays **TFT + EWC**. It is the model the benchmarks were designed around, and changing it now would invalidate the pre-registered comparisons. What has to change is how precisely the contribution is stated, plus one optional, pre-registered extension.

### 1.1 What is specific to our TFT + EWC (already implemented; to be stated in the paper)

1. **Label-free causal window.** Unlike library TFTs, the encoder never sees past SOH labels or protocol currents, only inputs known at the origin, with missing indicators per variable (D12, D14). This is what makes it leakage-safe on RPT-labelled data.
2. **Extrapolation-anchored residual output (D23).** The network learns only the correction to linear extrapolation of the recent fade rate: `Δ̂ = Δ̂_lin + g_θ(x)`. This turned the TFT from the worst into the best learned model (−0.17 pp at h = 1–3, −0.24 to −0.30 pp at h = 1–6, replicated).
3. **Multi-penalty EWC on the residual.** One diagonal Fisher per finished task, applied to the residual network. On the heterogeneous S6 stream it reduces forgetting from 0.28 / 0.22 to 0.09 pp at h = 1–6 (replicated).
4. **Task-agnostic.** No battery or group ID is an input, so the model needs no task label at test time.

Proposed name in the paper: **Residual Continual TFT (RC-TFT): an extrapolation-anchored Temporal Fusion Transformer with Elastic Weight Consolidation.** The text must say that TFT and EWC are existing components and that the contribution is their leakage-safe, extrapolation-anchored combination for continual SOH forecasting, together with its evaluation. Claims stay as measured: level with linear extrapolation at h = 1–3, not better.

### 1.2 Optional extension (team decision; pre-register as D24 before any run)

Forgetting appears only when groups differ (S1: 0.016–0.040 pp; S6: 0.18–0.46 pp), and validation picked very different EWC strengths across experiments (λ from 0.01 to 1000). Both point to one fixed λ being the wrong design. Candidate extension, still TFT + EWC:

- **Shift-adaptive EWC.** When task *t* arrives, compute a distribution-shift score `s_t` between its training inputs and the earlier tasks' inputs (for example the energy distance on the standardised shared inputs, using training rows only). Weight the EWC penalty as `λ_t = λ₀ · f(s_t)`, with `f` fixed in D24 (for example `s_t / median(s)`) and λ₀ tuned as now. Small shift means little protection and more plasticity; large shift means more protection.
- It uses only training inputs of the current and earlier tasks, so it stays leakage-free and task-agnostic at test time.
- Registered test: shift-adaptive residual TFT + EWC against fixed-λ residual TFT + EWC (D23 reference) on final error and forgetting, under the D21 replication rule. A null result is reported as such.

If the team does not adopt D24, RC-TFT as in 1.1 is the proposed model and WS4 is skipped.

## 2. Workstreams

Effort is a rough estimate for one person. "Data machine" means the PC with `data/`.

### WS1. Documentation and integrity fixes (no data needed)

| # | Task | Done when |
|---|---|---|
| 1.1 | Sync README §12 and §14 with `paper/references.bib`: ref 3 first author Faraji-Niri, ref 6 "Vignesh S" et al., ref 7 authors (Lee, Kim, Kim, Choi); remove or mark as unverified the findings quoted for refs 3, 6, 7 and 9 | README and bib agree entry by entry |
| 1.2 | Track the figures `docs/RESULTS.md` lists, or point the list at `paper/figures/` | Every figure path in the docs exists in Git |
| 1.3 | Team decision on licence and repository visibility; add `LICENSE`; remove the `\todo` | `grep -c todo paper/sections/*.tex` is 0 |
| 1.4 | Either implement `reuse_head.py` or delete the stub and keep the README statement that the head was not built | No stub file presented as a component |
| 1.5 | Rename the proposed model consistently (RC-TFT, §1.1) in README, RESULTS, paper | One name everywhere |

Effort: 0.5–1 day.

### WS2. Big-data preprocessing results (mostly existing data)

| # | Task | Output | Done when |
|---|---|---|---|
| 2.1 | **Preprocessing funnel table** for every dataset (S1, S6 checks, S6 ageing, S7, S8, S9): raw files, raw size, raw rows → parsed units (RPTs, checks, cycles) → valid after quality rules → forecasting samples (train / val / test), missingness per input group, runtime | `scripts/make_preprocessing_table.py` → `results/preprocessing_funnel.json`, `results/tables/preprocessing.csv`, `paper/tables/preprocessing.tex` | Every number read from a committed results file or recomputed by the script; nothing typed by hand |
| 2.2 | **Spark scaling run** on the data machine: full job at `local[1]`, `local[2]`, `local[4]`, `local[6]`, one run each (two if time allows), recording wall time, per-stage time and peak memory | `results/d22_spark_scaling.json`, figure `paper/figures/spark_scaling.png` (time and speed-up vs cores) | Outputs identical to the validated run at every core count (max abs diff recorded) |
| 2.3 | **Data-quality figure**: share of valid cycles per cell and block, and the leakage-audit correlations | `paper/figures/s6_ageing_quality.png` | Generated from `results/d22_ageing_report.json` |
| 2.4 | Paper: short "Preprocessing results" paragraph plus table 2.1 and the scaling figure; state single-machine local mode | Paper section | Fits the page budget (see WS7) |

Effort: 1–2 days, plus about 2–3 h of machine time for 2.2. Only 2.2 needs the data machine; 2.1 needs it if some counts are not yet in a results file.

### WS3. Literature baselines (code anywhere; runs on the data machine)

Add four model families from our literature review, run under exactly the D23 E1/E2 protocol (S6 stream, D21 shared inputs, purged splits, 10 tuning orders, one-SE rule, 10 orders × seeds 0–4 and 5–9).

| Model | Literature link | How it runs | Arms |
|---|---|---|---|
| **GPR** (Matérn + white noise, ARD) | Takahashi 2023 (bagged GPR), Faraji-Niri 2023 (EIS + ML) | Non-continual baseline fitted on all S6 training rows, like gradient boosting; kernel hyperparameters by marginal likelihood, nothing tuned on test | direct, residual |
| **LSTM** (1–2 layers; bidirectional variant noted only if used) | Meskine 2026 (BiLSTM) | Continual: fine-tuning and EWC, same window builder as the TFT (W = 4, inputs only) | direct, residual |
| **TCN** (dilated causal 1-D conv, residual blocks) | Yao 2025 (PNN + TCN), Meskine 2026 | Continual: fine-tuning and EWC, same windows | direct, residual |
| **Vanilla Transformer encoder** | closest ablation of the TFT (no variable selection, no gating) | Continual: fine-tuning and EWC, same windows | direct, residual |

Design rules:
- Parameter counts matched to the TFT within ±20 % (as D12 matched MLP and TFT) and reported.
- Same preprocessing (first-task-fitted, frozen), early stopping, Adam, batch 64, epoch cap, deterministic seeds.
- Grids fixed in D24 before any run; grid-edge extension as in D21.
- Optional, if time: random forest and SVR as extra classical baselines (cheap, non-continual).

Tasks:

| # | Task | Done when |
|---|---|---|
| 3.1 | `src/models/sequence_baselines.py`: `LSTMRegressor`, `TCNRegressor`, `TransformerRegressor` with the TFT's input contract | Unit tests: shapes, causality (output unchanged when future steps change), padding mask, determinism |
| 3.2 | Strategy classes in `src/continual_learning.py` (`LSTMFineTune`, `LSTMEWC`, …) by reusing `TFTMixin`'s window path | Tests mirror `tests/test_tft.py` |
| 3.3 | `GPRBaseline` alongside the gradient-boosting baseline | Test: fits, predicts, no test rows in the fit |
| 3.4 | Extend `scripts/run_d23.py` (or a new `scripts/run_d24.py`) with the new parts; smoke profile on test fixtures | Smoke run passes; output deleted (not reportable) |
| 3.5 | Paper-profile runs on the data machine, E1 and E2 | Results JSON written from a clean commit |

Effort: 3–4 days of code and tests. Machine time: D23 took about 4.5 h for 8 parts in parallel; 6 new continual parts × 2 arms × 2 experiments is roughly 1–1.5 days on the same PC.

### WS4. Shift-adaptive EWC (only if D24 adopts §1.2)

| # | Task | Done when |
|---|---|---|
| 4.1 | Write D24 in `docs/DECISIONS.md`: shift score, `f`, grids, tests, positive-result criterion, provenance (post hoc motivation stated) | Committed before any D24 code runs |
| 4.2 | `ShiftAdaptiveEWC` strategy (works for both MLP and TFT backbones) | Tests: λ_t uses training inputs only; reduces to fixed EWC when `f ≡ 1` |
| 4.3 | Paper runs, E1 and E2, alongside WS3 | Results JSON from a clean commit |

Effort: 1–2 days of code; runs share the WS3 machine time.

### WS5. Consolidated comparison: tables and figures

All generated by script from committed results files (extend `scripts/make_results_tables.py` / `make_d21_figures.py`; never edited by hand).

| # | Output | Content |
|---|---|---|
| 5.1 | **Main comparison table** (`tables/comparison.tex`) | Every method on S6 E1 and E2: RMSE and MAE (mean ± s.d., main / replicate), skill vs persistence, forgetting, backward transfer, parameters, training time; best value in bold; deterministic baselines marked |
| 5.2 | **Statistical table** (`tables/tests_d24.tex`) | RC-TFT vs each method: mean difference, 95 % bootstrap CI, p_Holm, rank-biserial r, replication verdict |
| 5.3 | **Per-group table** (`tables/per_group.tex`) | Final RMSE per S6 test case (case 1–7) for each method, from `final_error_per_task` (already stored in the D23 run records) and per-task baseline errors; required by `CLAUDE.md` |
| 5.4 | **Literature-context table** | Method, dataset, chemistry, SOH definition, reported error, our nearest equivalent. Header note: not directly comparable (different datasets and SOH definitions) |
| 5.5 | Figures | (a) error by method with 95 % CI (bar or dot plot); (b) RMSE vs horizon 1–6 for all methods; (c) forgetting vs final error scatter (adaptation–retention trade-off); (d) error matrix (task × stage) for RC-TFT and its fine-tuning counterpart; (e) critical-difference diagram (Friedman / Nemenyi) over methods; (f) example forecasts for one cell per condition |

Figure rules: print-legible (≥ 8 pt at column width), axis labels with units (pp, cycles, checks), colour-blind-safe palette, same method colours across all figures, PNG at 300 dpi plus PDF.

Effort: 2–3 days once WS3 results exist. 5.3 can be built now from the D23 files.

### WS6. Robustness and reuse (existing code; smaller items)

| # | Task | Notes |
|---|---|---|
| 6.1 | P2 (leave one condition out) for RC-TFT and the new baselines | Learned models beat extrapolation on unseen conditions (gradient boosting 0.71 vs 1.37 pp), which is the strongest case for a learned model; worth reporting for the proposed model |
| 6.2 | Reuse-class results for the new models in P2 | Accuracy, macro-F1, weighted kappa, confusion matrix |
| 6.3 | Uncertainty (optional) | Seed-ensemble spread or GPR variance; only if reported honestly |

Effort: 1 day of code reuse; runs alongside WS3.

### WS7. Paper revision

| # | Task |
|---|---|
| 7.1 | Abstract and contributions: name RC-TFT, state the novel combination precisely (§1.1), add the comparison with GPR, LSTM, TCN and Transformer, add the preprocessing and scaling results |
| 7.2 | Methods: new baselines (one paragraph and a parameter table), D24 if adopted |
| 7.3 | Results: replace the residual table with the consolidated table 5.1; add per-group table, comparison figure and statistical table; keep the reproducibility section |
| 7.4 | Limitations: unchanged honesty items (relative SOH, small groups, single-machine Spark, not better than extrapolation unless D24/WS3 show otherwise) |
| 7.5 | Page budget: the conference version is 8 pages. If the new tables do not fit, move P1/P2 detail on S1 and the A1 ablation to a supplement or the long version (tag `paper-long-draft`) |
| 7.6 | `python scripts/flatten_paper.py`, rebuild `paper_single.pdf`, `--check` passes, zero `\todo` |

Effort: 2–3 days.

## 3. Order and dependencies

```
WS1 (docs fixes) ─────────────────────────────────────────────┐
WS2.1, 2.3, 5.3 (from existing results) ──────────────────────┤
D24 written and committed ─► WS3 code + tests ─► WS3/WS4/WS6 runs (data machine) ─► WS5 ─► WS7
                            WS4 code + tests ─┘                 WS2.2 scaling run ─┘
```

Suggested sequence:

1. **Week 1:** WS1; WS2.1 and 2.3; per-group table 5.3 from the D23 files; team decision on D24 (§1.2) and on the licence; write and commit D24.
2. **Week 2:** WS3 and WS4 code and tests; smoke runs; then paper runs on the data machine (about 1–1.5 days) together with the Spark scaling run.
3. **Week 3:** WS5 tables and figures; WS6; WS7 paper revision, flatten, final checks.

## 4. Decisions needed from the team

1. Adopt the shift-adaptive EWC extension (§1.2, D24), or keep RC-TFT as in §1.1 only?
2. Model name in the paper (RC-TFT proposed).
3. Which machine runs the paper profile, and is it free for about two days?
4. Licence and repository visibility (`\todo` in the paper).
5. Whether the new tables replace S1 material in the 8-page version or the paper moves to a longer format.

## 5. Definition of done

- Every number in the paper comes from a results JSON written on a clean commit, with environment recorded.
- D24 committed before its runs; deviations logged.
- At least four literature model families (GPR, LSTM, TCN, Transformer) plus the existing baselines compared to RC-TFT under one protocol, with the main, statistical and per-group tables and the figures listed in WS5.
- Preprocessing funnel table and Spark scaling figure in the paper.
- `pytest` passes; `flatten_paper.py --check` passes; zero `\todo`; README, RESULTS and paper agree.

## 6. D25: improving the residual TFT + EWC (14 Oct 2026)

Pre-registration: `docs/DECISIONS.md`, D25. Runbook: `docs/D25_NIGHT_RUN.md`. Code: `src/calibration.py`, `scripts/run_d25.py`, `scripts/run_d25_night.py`, `scripts/summarise_d25.py`, `tests/test_d25.py`.

### 6.1 Where the model stands after D24

| Observation (committed results) | What it implies |
|---|---|
| No learned model beats linear extrapolation: 0.343 pp (h = 1–3), 0.485 pp (h = 1–6) | Extrapolation is the bar; the learned part is a small correction on top of it |
| In E2, extrapolation is better at h = 1–2 (replicated, D23 and D24); the gap closes by h = 5–6 | The correction adds error exactly where the trend is already well determined |
| EWC keeps the residual TFT's forgetting at about 0.07–0.10 pp; shift-adaptive EWC changed nothing (D24) | Forgetting is no longer the bottleneck; more penalty work cannot move the final error much |
| 135–225 training samples per S6 task for 12.7k parameters; the 21 Spark ageing inputs made the model worse (D22); a same-size BiLSTM ties it at h = 1–3 (D24) | Data, not architecture, limits the model |
| P3 pretraining (D21) was tested only with the direct target, where even on the pool the model was worse than extrapolation | Pretraining the residual model is untested |
| On unseen conditions (P2), gradient boosting beats extrapolation (0.71 vs 1.37 pp on S6) | The strongest case for a learned model is transfer to new conditions (WS6.1) |

### 6.2 What D25 changes, and why

1. **Horizon-wise shrinkage of the correction (Q1).** After each task, one factor per horizon in [0, 1], fitted in closed form on that task's validation rows, scales the learned correction. Where the correction does not help, the factor goes to 0 and the forecast becomes extrapolation, so the model cannot lose there; where it helps, it is kept. No hyperparameter, no change to training, so the raw (D23) and shrunk models come from the same runs and the D23 numbers are reproduced unit by unit as a check.
2. **Conformal forecast intervals (Q3).** Per horizon, from the same validation rows; extrapolation gets intervals in exactly the same way, so the comparison is fair. Scored with the interval score (width plus a penalty for misses), with coverage and width reported. This is the output a second-life decision needs ("above 80 % SOH in three checks, with 90 % confidence?"), and the one place a learned model can differ from extrapolation in more than the point forecast.
3. **Pretraining the residual model on the 369-cell pool (Q2).** Tests whether the data shortage can be eased by first-life data now that the model only learns a correction.

### 6.3 What we expect, honestly

- Shrinkage most likely makes the model **as accurate as** extrapolation at every horizon (practically equivalent) rather than better: the D23 per-horizon gaps (E2, main seeds) are positive at every horizon. A replicated win needs the correction to help at some horizons, which pretraining may provide.
- Pretraining is the change with the largest possible gain and the largest uncertainty: direct-target pretraining did not help in D21.
- Intervals are the most likely place for a clear result in either direction; if the model's intervals are not better, that is reported too.

### 6.4 What we stop doing

- No further continual-learning penalties or EWC variants on S6 (D24 answered this).
- No more inputs on S6 (D22 answered this).
- No more backbone comparisons at the same size (D24 answered this).

Every extra variant tested on the same data weakens the statistics; reviewers notice this.

### 6.5 Steps

| # | Task | Where | Status |
|---|---|---|---|
| 6.5.1 | Pre-register D25; code and synthetic-data tests | cloud | Done 14 Oct |
| 6.5.2 | Night run (pinned environment, 6 workers) | ASUS | Done 15–16 Oct |
| 6.5.3 | Run log; results committed | ASUS | Done 16 Oct |
| 6.5.4 | D25 section of `docs/RESULTS.md`; tables and figures (error by horizon with shrinkage, interval score and coverage per task, shrinkage factors) | cloud | Section done 16 Oct; tables and figures pending |
| 6.5.5 | Paper: one methods paragraph (shrinkage, conformal intervals, residual pretraining), results, limitations | cloud | After 6.5.4 |
| 6.5.6 | Next (not in D25): P2 for the D25 model and the D24 baselines (WS6.1), the setting where learned models already beat extrapolation | both | Done as D26 (residual TFT + EWC only; the D24 backbones were not run, D26 "Not done"); run 17–18 Oct, written up 19 Oct |

