// Fixed text of the project pages. Condensed from README.md, paper/sections/ and dashboard/tableau/WORKBOOK_SPEC.md;
// keep it in step with those files. Numbers that change with new results are read from summary.json, not written here.

export const PROJECT = {
  title: "Residual TFT-EWC",
  subtitle: "Continual learning for state-of-health forecasting and reuse screening of heterogeneous second-life EV batteries",
  question: "Retired electric-vehicle batteries still hold useful capacity and can serve a second life, for example in stationary storage. Before reuse we must know how healthy each one is and how fast it will keep ageing. Batteries differ in history and duty, and new groups keep arriving. Can one model keep learning from each new battery group without forgetting the earlier ones, and are its forecasts good enough to support reuse screening?",
  status: "IEEE conference paper, in preparation. Every reported number traces to a results file written by code in the repository on a clean commit; every experiment from D18 on was pre-registered before it ran, and null and negative results are reported alongside positive ones.",
  authors: ["Tarun Sitaraman", "Sohan Sandeep Krishnan", "Darshan D"],
  links: {
    paper: "https://github.com/TarunSitaraman/residual-tft-ewc/raw/main/paper/paper_single.pdf",
    supplement: "https://github.com/TarunSitaraman/residual-tft-ewc/raw/main/paper/supplement.pdf",
    repo: "https://github.com/TarunSitaraman/residual-tft-ewc",
  },
};

export const FINDINGS = [
  ["Competitive in both settings", "Residual TFT-EWC is the only model competitive both on the continual stream and on unseen usage conditions. Linear extrapolation degrades on unseen conditions; gradient boosting trails on the stream."],
  ["Level with the strongest stream baseline", "On the LFP dataset (S6) it matches linear extrapolation of the recent fade rate (0.32 against 0.34 pp RMSE, horizons 1-3; practically equivalent), which no model beats, and is 0.13-0.41 pp more accurate than gradient boosting (replicated)."],
  ["EWC cuts forgetting by about two thirds", "At horizons 1-6, forgetting of the residual TFT falls from 0.28 / 0.22 pp to 0.09 pp, replicated across two seed sets."],
  ["Ahead of extrapolation on unseen conditions", "It is 0.43-0.68 pp more accurate than extrapolation (replicated) and assigns the measured fade class to 95.5 % of samples. Gradient boosting (0.54 pp) and the jointly trained TFT (0.57 pp) are more accurate because they store every earlier condition's data; held to the same no-storage constraint, gradient boosting is within 0.08 pp."],
  ["Intervals need held-out calibration", "Intervals hold on a new condition only when calibrated on conditions held out from fitting. Refitting and tracking as the new condition's checks arrive cut the 90 % interval score by 29 %. The out-of-range condition (100 % depth of discharge) is improved, not fixed."],
  ["Diagnostics describe the present", "EIS, pulse resistance and incremental-capacity peaks describe the current SOH well but add little for the next change."],
  ["Reproducibility is reported", "Some earlier significance results did not survive a re-run on a different platform; those comparisons are reported as not robust."],
];

export const AT_A_GLANCE = [
  ["Data", "Public battery-ageing datasets, downloaded from their original archives and verified file by file (SHA-256 / CRC-32 manifests)"],
  ["Task", "Forecast the state of health (SOH) of a cell at its next 1-6 capacity checks"],
  ["Models", "Baselines (persistence, linear extrapolation, ridge, gradient boosting), an MLP, and the proposed Temporal Fusion Transformer (TFT) with Elastic Weight Consolidation (EWC)"],
  ["Learning setting", "Continual learning: groups arrive one at a time; accuracy, forgetting and transfer are measured"],
  ["Big-data component", "An Apache Spark job that reduces about 132 GB (902 million rows) of raw ageing data to block-level features"],
  ["Reuse screening", "Forecast SOH mapped to four capacity-based fade classes (C0-C3), a screening aid rather than an application verdict"],
  ["Uncertainty", "Split-conformal forecast intervals, calibrated leave-one-condition-out and updated online as a new condition's checks arrive"],
];

export const CONTRIBUTIONS = [
  "A leakage-controlled benchmark for forecasting the SOH of second-life cells from five public datasets (395 cells after quality rules), with two label artefacts removed: test currents set from the previous capacity, and pulse charge counted in the capacity.",
  "An evaluation of a TFT with EWC against fine-tuning, replay, joint training, simple forecasters and the model families used in the literature (BiLSTM, TCN, Transformer, GPR, random forest, SVR) under three protocols, reporting adaptation, forgetting and retention.",
  "Residual TFT-EWC, a continual learner that corrects linear extrapolation rather than replacing it.",
  "Forecast intervals for unseen usage conditions, calibrated on conditions held out from fitting.",
  "A reproducibility analysis: several significance results on small battery datasets depend on the numerical environment, so the main results must pass a replication rule.",
];

export const GLOSSARY = [
  ["Cell / battery group", "One physical cell (S1) or one usage condition (S6). Groups are the unit never mixed across train and test."],
  ["RPT / check", "A slow, standardised capacity test done every few weeks. Its discharge capacity is the trusted health measurement. S6 calls it a check."],
  ["SOH", "Capacity / capacity at the cell's first valid check × 100. Relative to the first measurement, not to the nominal rating."],
  ["pp", "Percentage points of SOH, the unit of every error. An RMSE of 0.4 pp means forecasts are off by about 0.4 SOH points."],
  ["Forecasting task", "From origin check k, predict the SOH change to check k + h, for h = 1-3 (E1) or 1-6 (E2)."],
  ["Persistence", "Predict no change. The baseline every forecaster must beat."],
  ["Linear extrapolation", "Continue the recent fade rate."],
  ["Continual learning", "Train on tasks in sequence and never revisit old data (except by explicit replay)."],
  ["Catastrophic forgetting", "After learning a new task the model gets worse on an old one."],
  ["EWC", "Elastic Weight Consolidation: a penalty that keeps the weights that mattered for earlier tasks from moving, weighted by each finished task's Fisher information."],
  ["TFT", "Temporal Fusion Transformer: gated residual networks, variable selection, an LSTM encoder and interpretable attention. Input: the last four checks of one cell."],
  ["Residual target", "The TFT predicts only a correction to linear extrapolation, shrunk per horizon."],
  ["Adaptation / retention / backward transfer", "Error on the newest task; error on older tasks after later learning; how learning a later task changes old-task error."],
  ["Reuse classes", "From fade F = 100 - SOH: C0 < 5 pp, C1 5-10, C2 10-20, C3 ≥ 20 pp. They describe capacity change, not suitability for an application."],
];

export const SOURCES = [
  ["S1", "Stanford second-life dataset (Moy et al., 2024), OSF 8jnr5, CC BY 4.0", "6 LG M50T NMC cells on grid duty cycles; 111 RPTs", "Dataset 1 of the forecasting study; second-life stream of 6 cells"],
  ["S6", "Wheeler et al., 2025, Recherche Data Gouv 10.57745/OLBXKT, CC BY 4.0", "20 A123 LFP cells, 7 usage conditions, aged at 50 °C; 847 checks (843 labelled); 2,409 forecasting samples; about 132 GB raw", "Dataset 2: the main continual stream, and the Spark workload"],
  ["S7-S9", "Luh and Blank 2024 (KIT), Wang et al. 2024 (XJTU), Zhu et al. 2022 (Tongji)", "228 + 55 + 130 first-life cells; 369 cells and 36,711 samples after filtering", "First-life pretraining pool (protocol P3)"],
];

export const DATA_RULES = [
  ["SOH is relative", "Measured against the first RPT, which in S1 is already an aged state. Not comparable with literature values on nominal capacity."],
  ["Labels repeat between checks", "SOH only changes at a check, so estimating current SOH rewards copying the last label. The task was reframed as forecasting (D16)."],
  ["Temporal alignment", "A cycle uses only the most recent RPT on or before its date."],
  ["No leakage via test currents", "Protocol currents are set from the previous capacity, so they are never inputs (D14)."],
  ["Label correction", "The handover capacity included pulse charge; labels were rebuilt from the raw C/20 discharge (D17). The S6 parser reproduces the authors' C/3 capacity on all 843 checks."],
  ["Groups stay separate", "Splits are chronological within each cell, purged and never shuffled. Results are reported per group."],
  ["Missing temperature is not invented", "Missingness is flagged, imputation is fit on training data only, and robustness is tested by removing values."],
  ["No synthetic data in results", "Synthetic data may only test code paths and stay outside the repository."],
];

export const SPARK = [
  ["Workload", "827 CSV files (one per cell and 100-cycle block), about 132 GB raw, 901,974,259 data rows"],
  ["Map", "One Spark task per block extracts per-cycle measured responses (end-of-charge and end-of-discharge voltage, CV hold time and final current, relaxation)"],
  ["Cache", "The per-cycle table (82,529 cycles) is cached in memory"],
  ["Reduce", "Spark SQL groups valid cycles by (cell, block): mean, last value and slope against cycle number; 21 block-level inputs"],
  ["Validate", "All 827 blocks match the pandas implementation (largest absolute difference about 2e-13)"],
  ["Runtime", "PySpark 4.2.0 in local mode on one Windows PC; no cluster. 19.6 min on the first run"],
  ["Outcome", "The features did not improve the forecast (residual TFT + EWC 0.14-0.19 pp worse, replicated); reported as a negative result"],
];

export const PIPELINE = [
  ["Acquire", "Resumable fetchers download every public archive and verify each file (CRC-32, SHA-256, MD5)."],
  ["Build labels and features", "S1 RPT and EIS files to C/20 labels and diagnostics; S6 check files to C/25 and C/3 capacity; S7-S9 to a common pool; S6 ageing blocks through Spark."],
  ["Forecasting samples", "Origin check k to SOH change at k + h; inputs known at k only; purged chronological splits (train ≤ 60 %, validation 60-80 %, test ≥ 80 % of each cell's timeline)."],
  ["Models", "Baselines, MLP and TFT backbones; fine-tuning, replay, EWC and joint training."],
  ["Protocols", "P1 continual stream, P2 leave one group out, P3 pretrain then stream."],
  ["Evaluation", "RMSE, forgetting, backward transfer; Wilcoxon tests with Holm correction, bootstrap CIs, reuse classes."],
  ["Outputs", "results/*.json, then tables, figures, docs/RESULTS.md, the paper and this site."],
];

export const BASELINES = [
  ["Persistence", "SOH change = 0"],
  ["Linear extrapolation", "Continue the recent fade rate"],
  ["Ridge", "Linear model on the inputs (usage only, F0, F1 variants)"],
  ["Gradient boosting", "Histogram boosting on the shared inputs"],
];

export const STRATEGIES = [
  ["Fine-tuning", "Train on the new task only (the forgetting baseline)"],
  ["Replay", "Keep a small buffer of old samples per task"],
  ["EWC", "Multi-penalty EWC with a diagonal Fisher per finished task"],
  ["Joint", "Train on all tasks at once (the upper bound, not a continual method)"],
];

export const PROTOCOLS = [
  ["P1", "Continual stream", "Tasks arrive one at a time (S1: cells; S6: usage conditions); 10 task orders × 5 seeds, plus replicate seeds 5-9 from D21"],
  ["P2", "Leave one group out", "Train on the other groups, test on the held-out group (unseen usage condition)"],
  ["P3", "Pretrain then stream", "Pretrain on the first-life pool S7-S9, then learn the S6 stream"],
  ["Joint", "All tasks at once", "Upper bound"],
];

export const STATS = [
  ["Metrics", "RMSE of SOH(k + h) in pp (MAE secondary); skill = 1 - RMSE / RMSE of persistence; forgetting; backward transfer; adaptation; reuse-class accuracy and macro-F1."],
  ["Tests", "Paired Wilcoxon tests with Holm correction, rank-biserial effect sizes, bootstrap 95 % CIs."],
  ["Replication rule (D21)", "A result counts as replicated only if p_Holm < 0.05 in both seed sets, with the same sign and |difference| ≥ 0.05 pp; |difference| < 0.05 pp in both is practically equivalent."],
  ["Deterministic comparisons (D27-D29)", "The unit is the held-out condition: two-sided exact Wilcoxon over the seven S6 conditions, and a result must hold in both horizon sets."],
  ["Pre-registration", "Protocols D18, D20-D29 and D31 were written into docs/DECISIONS.md before the runs they govern; deviations and post-hoc analyses are labelled."],
];

export const STREAM_TABLE = {
  caption: "Final average test RMSE (pp) on S6, experiment D23 E1 (horizons 1-3). Learned methods: mean over 50 runs, main / replicate seeds. Baselines: single deterministic fits.",
  head: ["Method", "RMSE (pp)"],
  rows: [["TFT + EWC, residual", "0.339 / 0.314"], ["Linear extrapolation", "0.343"], ["TFT fine-tuning, residual", "0.347 / 0.337"],
    ["MLP + EWC, residual", "0.374 / 0.366"], ["MLP fine-tuning, direct", "0.413 / 0.351"], ["TFT fine-tuning, direct", "0.424 / 0.444"],
    ["Gradient boosting", "0.451"], ["TFT + EWC, direct", "0.504 / 0.487"], ["Persistence", "1.644"]],
};

export const UNSEEN_TABLE = {
  caption: "Unseen S6 usage condition (P2, seven held-out conditions, D26-D32): test RMSE (pp), mean over folds. TFT rows: mean over 50 runs, main / replicate seeds; other rows: single deterministic fits.",
  head: ["Method", "Horizons 1-3", "Horizons 1-6"],
  rows: [["Residual TFT + EWC, shrunk (continual, D26)", "0.92 / 0.94", "1.62 / 1.65"], ["Residual TFT, joint (D26)", "0.57 / 0.58", "1.21 / 1.18"],
    ["Joint residual TFT adapted online with EWC (D31)", "0.70 / 0.70", "1.13 / 1.10"], ["Gradient boosting, latest condition only (D32)", "0.865", "1.593"],
    ["Gradient boosting (D26/D27)", "0.535", "1.051"], ["Gradient boosting, refitted online (D28)", "0.458", "0.898"],
    ["Physics law + gradient-boosting residual (D29)", "0.581", "1.016"], ["Ridge", "0.77", "1.30"], ["Linear extrapolation", "1.37", "2.30"], ["Persistence", "2.82", "4.90"]],
};

export const LEARNED = [
  "Diagnostics describe the present, not the next change: IC peak, pulse and EIS resistances track current SOH (|ρ| 0.83-0.97 within cells) but are weakly associated with the next change (median |ρ| ≤ 0.3 for most features).",
  "The best methods beat persistence on both datasets, but not every method does. On S1 the best is a ridge on planned usage alone (0.450 pp); on S6, linear extrapolation.",
  "Forgetting depends on task heterogeneity: negligible across the S1 cells (0.016-0.040 pp), substantial across S6 usage conditions (0.18-0.46 pp).",
  "Calendar time is a key input: on S6, fade per 100 cycles rises with the days those cycles took (within-cell ρ = 0.64, positive in 20 of 20 cells).",
  "Pretraining on first-life data (P3) did not help S6 forecasts.",
  "Residual learning makes the TFT competitive (0.17 pp better than direct at horizons 1-3, replicated). The registered goal of beating extrapolation on the stream was not reached; on unseen conditions it does beat it.",
  "EWC lowers forgetting of the residual TFT at horizons 1-6 (0.28 / 0.22 pp to 0.09 pp, replicated); its effect on final error is not replicated.",
  "Reuse classes are informative only in P2 on S6: the proposed model assigns the measured class to 95.5 % of held-out samples, against 93.9 % for extrapolation and 85.8 % for persistence; ridge (96.5 %) and gradient boosting (97.7 %) do better. Every error is between neighbouring classes.",
  "Intervals: residuals from seen conditions understate the error on a new one (coverage 0.66-0.68 at nominal 90 %); residuals from held-out conditions reproduce it (0.90).",
  "Reproducibility: a re-run in a different environment kept all baselines identical, but several TFT significance statements flipped across p = 0.05. Those comparisons are reported as not robust.",
];

export const LIMITATIONS = [
  ["The task changed after seeing data", "The study moved from current-SOH estimation to forecasting once persistence beat the first baselines (D16). The forecasting protocol was fixed before any forecasting model was trained."],
  ["Small datasets", "S1 has 6 cells and S6 has 2-3 cells per condition, so leave-one-group-out tests have little power."],
  ["Relative SOH", "Not directly comparable with literature numbers on different datasets or SOH definitions."],
  ["S6 \"second life\"", "S6 is aged at 50 °C; second life means continued ageing below 80 % SOH under the same protocol, not a repurposed duty cycle."],
  ["Conditional forecasts", "Forecasts assume the planned usage is known."],
  ["Spark is single-machine", "It demonstrates the map-reduce design and validates against pandas; it is not a cluster benchmark."],
  ["Some significance statements are not robust", "They depend on the numerical environment and are reported as such."],
  ["Stored data wins on unseen conditions", "Gradient boosting and joint training, which store every earlier condition, are more accurate there."],
  ["Out-of-range conditions", "With 100 % depth of discharge held out, every training condition runs at 70 %, so neither forecasts nor intervals can be trusted there."],
  ["Post-hoc motivation", "Experiments from D23 on were each motivated by earlier results; each was pre-registered before it ran."],
  ["Novelty", "The contribution is an integrated, battery-focused evaluation. TFT, EWC and reuse classification are not individually novel."],
];

// Exported tables: what each holds. Row counts and columns are read from manifest.json and the files themselves.
export const TABLES = [
  ["dim_group", "results", "One row per battery group: S1 cells and S6 usage conditions, with chemistry and the charge and discharge settings."],
  ["dim_method", "results", "One row per method: label, family (TFT, MLP, GBM, baseline, …), kind and whether it is the proposed model."],
  ["fact_headline", "results", "The paper's headline comparison (Table III): test RMSE of each method per protocol and setting, plus the proposed-minus-baseline differences and their verdicts."],
  ["fact_verdicts", "results", "Registered pairwise comparisons with Holm-corrected p-values in both seed sets and the replication verdict."],
  ["fact_tests", "results", "Every paired test: mean difference, bootstrap CI, p-value, Holm p-value and rank-biserial effect size."],
  ["fact_stream_run", "results", "Continual stream (P1), one row per run (method × seed × task order): final error, forgetting, transfer, per-horizon RMSE, interval coverage."],
  ["fact_stream_task", "results", "Continual stream, one row per run and task: arrival position, error right after learning it, final error, forgetting and backward transfer."],
  ["fact_stream_baseline", "results", "Deterministic baselines on the stream, per group and pooled."],
  ["fact_unseen_run", "results", "Unseen usage condition (P2), one row per run: pooled error, per-horizon RMSE, class accuracy, interval coverage."],
  ["fact_unseen_fold", "results", "Unseen usage condition, one row per run and held-out condition."],
  ["fact_coverage", "results", "Prediction-interval coverage, interval score and width per calibration rule and held-out condition."],
  ["fact_horizon", "results", "Pooled RMSE at each forecast horizon (study D30)."],
  ["fact_d30_group", "results", "Study D30 metrics per method and group (RMSE, MAE, class metrics)."],
  ["fact_soh", "local", "SOH trajectory of every cell: one row per capacity check, with capacity, reference capacity, temperature source and reuse class. Built from the local data folder."],
  ["fact_predictions", "local", "Every saved D30 forecast: origin and target check, horizon, measured and forecast SOH, error and reuse classes. Built from the local data folder."],
];

export const TABLEAU_CONTRACT = {
  dashboards: [["Overview", "overview"], ["SOH explorer", "soh"], ["Forecast accuracy", "accuracy"], ["Continual learning", "continual"], ["Reuse & uncertainty", "reuse"]],
  fields: [["Dataset", "S1, S6"], ["Setting", "e1, e2 (aliased E1 (h = 1-3), E2 (h = 1-6))"], ["Protocol", "P1, P2"], ["Group", "g1 … w10, case1 … case7, all"], ["Method Family", "from dim_method, joined on method"]],
};
