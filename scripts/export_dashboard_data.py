"""Export tidy (long-format) tables for the results dashboard (dashboard/, Tableau Public workbook).

Every number is read from a committed results file or from a table built by this repository; nothing is recomputed
except reshaping, the reuse class of a SOH value (``src.evaluation.reuse_class`` rule, cut-points 5/10/20 pp) and
the arrival position of a task in a continual-learning order. Two output folders:

- ``results/dashboard/`` (committed, like ``results/tables/``): tables built only from ``results/*.json`` and
  ``results/tables/*.csv``, plus ``summary.json`` (headline KPIs read natively by the JS page) and ``manifest.json``.
- ``<data>/dashboard/`` (git-ignored, like ``data/``): tables built from the data folder (SOH trajectories from
  ``rpt_health_c20.csv`` and ``wheeler_checks.csv``; saved D30 forecasts). Skipped, with a message, when the inputs
  are absent.

SOH is relative to the first valid capacity check of each cell (D17), never to nominal capacity. Missing
temperatures are left empty and flagged (``temperature_source``), never filled. S1 and S6 are never pooled.

Usage (from the repository root)::

    python scripts/export_dashboard_data.py                 # data folder: $DATA_ROOT or data/
    python scripts/export_dashboard_data.py --data-root /mnt/project-data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import evaluation as ev  # noqa: E402

REUSE_CUTPOINTS, REUSE_CLASS_NAMES = ev.REUSE_CUTPOINTS, ev.REUSE_CLASS_NAMES

#: Groups in their canonical order. S1: one cell per group; S6: one usage condition (test case) per group.
S1_GROUPS: tuple[str, ...] = ("g1", "v4", "v5", "w8", "w9", "w10")
#: S6 test cases (src/wheeler.py TEST_CASES): charge mode, max voltage (V), charge C-rate, discharge, depth of discharge.
S6_CASES: tuple[tuple, ...] = (
    ("case1", "CC-CV", 3.65, 1 / 3, "WLTP", 0.7), ("case2", "CC-CV", 3.55, 1 / 3, "WLTP", 0.7),
    ("case3", "CC", 3.65, 1 / 3, "WLTP", 0.7), ("case4", "CC-CV", 3.65, 2.0, "WLTP", 0.7),
    ("case5", "CC-CV", 3.55, 2.0, "WLTP", 0.7), ("case6", "CC-CV", 3.65, 2.0, "CC 2C", 0.7),
    ("case7", "CC", 3.65, 1 / 3, "WLTP", 1.0),
)
SETTINGS = {"e1": "E1 (h = 1-3)", "e2": "E2 (h = 1-6)"}
#: Continual-stream (P1) and unseen-condition (P2) experiments read from results/forecasting_<exp>_<setting>.json.
STREAM_EXPS = ("d23", "d24", "d25")
UNSEEN_EXPS = ("d26",)

BASELINE_NAMES = {
    "persistence": "Persistence", "linear_extrapolation": "Linear extrapolation", "ridge_schedule": "Ridge, usage only",
    "ridge_shared": "Ridge", "ridge_F0": "Ridge, F0", "ridge_F1": "Ridge, F1", "gbm_shared": "Gradient boosting",
    "gbm_F1": "Gradient boosting, F1", "gbm_all": "Gradient boosting, all conditions stored",
    "gbm_last": "Gradient boosting, latest condition only", "gbm_warm": "Gradient boosting, warm-started",
    "gpr_direct": "GPR, direct", "gpr_residual": "GPR, residual", "rf_direct": "Random forest, direct",
    "rf_residual": "Random forest, residual", "svr_direct": "SVR, direct", "svr_residual": "SVR, residual",
    "linear_extrapolation_conformal": "Linear extrapolation (conformal)", "d28_refit": "Gradient boosting, refit (D28)",
}
SPECIAL_NAMES = {
    "tft_residual_saewc": "Residual TFT + SA-EWC", "tft_residual_ewc_shrunk": "Residual TFT-EWC (proposed)",
    "tft_residual_ewc_selective": "Residual TFT-EWC, shrunk at H+ only (D36 proposed)",
    "tft_pre_residual_ewc": "Residual TFT + EWC, pretrained", "tft_pre_residual_ewc_shrunk": "Residual TFT-EWC, pretrained",
    "tft_residual_joint": "Residual TFT, joint", "joint_static": "Residual TFT, joint (static)",
    "ft_online": "Residual TFT, online fine-tuning", "ewc_online": "Residual TFT + EWC, online",
    # D18/D20 names (results/forecasting.json, forecasting_wheeler.json; labels of scripts/make_results_tables.py)
    "finetune": "MLP fine-tuning", "replay": "MLP replay", "ewc": "MLP + EWC", "joint": "MLP joint",
    "tft_finetune": "TFT fine-tuning", "tft_ewc": "TFT + EWC", "tft_ewc_F0": "TFT + EWC, F0", "tft_joint": "TFT joint",
    "mlp": "MLP", "tft": "TFT",
    # D27 interval calibration rules (docs/RESULTS.md, D27)
    "gbm_loco": "GBM-LOCO intervals", "gbm_loco_mondrian": "GBM-LOCO intervals, Mondrian",
    "lin_all": "LIN-all intervals", "lin_all_mondrian": "LIN-all intervals, Mondrian",
    "lin_val": "LIN-val intervals (D26 rule)",
    # D28 interval updates on the new condition (docs/RESULTS.md, D28)
    "d28_static": "Static (D27 GBM-LOCO)", "d28_track": "Tracking", "d28_refit_static": "Refit",
    "d28_refit_track": "Refit + tracking",
}
D28_ARMS = {"static": "d28_static", "track": "d28_track", "refit": "d28_refit_static", "refit_track": "d28_refit_track"}
D27_RULES = ("gbm_loco", "gbm_loco_mondrian", "lin_all", "lin_all_mondrian", "lin_val", *D28_ARMS.values())
LEGACY_MLP = ("finetune", "replay", "ewc", "joint", "mlp")
FAMILY_NAME = {"tft": "TFT", "mlp": "MLP", "lstm": "BiLSTM", "tcn": "TCN", "transformer": "Transformer"}
STRATEGY = {"ewc": "+ EWC", "finetune": "fine-tuning", "saewc": "+ SA-EWC", "joint": "joint"}
PROPOSED = "tft_residual_ewc_shrunk"


# ---------- small helpers ----------
def reuse_class(soh) -> np.ndarray:
    """Class index 0-3 by src.evaluation.reuse_class, which refuses NaN; unlabelled rows stay NaN (no class)."""
    soh = np.asarray(soh, dtype=float)
    out = np.full(soh.shape, np.nan)
    ok = ~np.isnan(soh)
    out[ok] = ev.reuse_class(soh[ok])
    return out


def class_name(idx) -> list:
    return [REUSE_CLASS_NAMES[int(i)] if pd.notna(i) else None for i in idx]


def num(x):
    """Float or NaN (None, missing and non-finite values become NaN)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def method_label(m: str) -> str:
    if m in BASELINE_NAMES:
        return BASELINE_NAMES[m]
    if m in SPECIAL_NAMES:
        return SPECIAL_NAMES[m]
    parts = m.split("_")
    if len(parts) == 3 and parts[0] in FAMILY_NAME and parts[2] in STRATEGY:
        return f"{FAMILY_NAME[parts[0]]} {STRATEGY[parts[2]]}, {parts[1]}"
    return m.replace("_", " ")


def method_family(m: str) -> str:
    if m in D27_RULES:
        return "GBM" if m.startswith(("gbm", "d28")) else "Baseline"
    if m in BASELINE_NAMES:
        return "GBM" if m.startswith("gbm") or m == "d28_refit" else "Baseline"
    if m in ("joint_static", "ft_online", "ewc_online") or m.startswith("tft"):
        return "TFT"
    if m in LEGACY_MLP:
        return "MLP"
    return FAMILY_NAME.get(m.split("_")[0], "Other")


def method_kind(m: str) -> str:
    if m in D27_RULES:
        return "interval calibration"
    if m in ("persistence", "linear_extrapolation", "linear_extrapolation_conformal"):
        return "deterministic"
    if m in BASELINE_NAMES:
        return "tabular baseline"
    if m.endswith("joint") or m == "joint_static" or m == "joint":
        return "joint (upper bound)"
    if m.endswith("_online"):
        return "online adaptation"
    return "continual"


def load_json(results: Path, name: str) -> dict | None:
    p = results / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ---------- dimensions ----------
def dim_group() -> pd.DataFrame:
    rows = [{"dataset": "S1", "group": g, "group_order": i, "chemistry": "NMC (LG M50T)", "unit": "cell",
             "description": f"Second-life cell {g}, grid duty cycle", "charge_mode": None, "v_max": math.nan,
             "charge_c_rate": math.nan, "discharge": None, "dod": math.nan} for i, g in enumerate(S1_GROUPS)]
    rows += [{"dataset": "S6", "group": c, "group_order": i, "chemistry": "LFP (A123)", "unit": "usage condition",
              "description": f"{cm} to {v:.2f} V at {cr:.2g}C, {d} discharge, {dod:.0%} DoD, 50 degC",
              "charge_mode": cm, "v_max": v, "charge_c_rate": cr, "discharge": d, "dod": dod}
             for i, (c, cm, v, cr, d, dod) in enumerate(S6_CASES)]
    return pd.DataFrame(rows)


def dim_method(methods: Iterable[str]) -> pd.DataFrame:
    ms = sorted(set(methods))
    return pd.DataFrame({"method": ms, "label": [method_label(m) for m in ms], "family": [method_family(m) for m in ms],
                         "kind": [method_kind(m) for m in ms], "is_proposed": [m == PROPOSED for m in ms]})


# ---------- P1: continual stream ----------
STREAM_RUN_COLS = ("final_avg_error", "plasticity", "forgetting", "backward_transfer", "forward_transfer",
                   "rmse_h1", "rmse_h2", "rmse_h3", "cov90", "width90", "is90", "train_time_s")
TASK_KEYS = {"final_error": "final_error_per_task", "diagonal_error": "diagonal_error_per_task",
             "forgetting": "forgetting_per_task", "backward_transfer": "backward_transfer_per_task",
             "forward_transfer": "forward_transfer_per_task"}


def stream_tables(d: dict, experiment: str, setting: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(run-level, run x task, baselines) rows of one continual-stream results file."""
    orders = d.get("orders") or []
    runs, tasks = [], []
    for r in d.get("runs", []):
        base = {"experiment": experiment.upper(), "setting": setting, "seed_set": r.get("set"),
                "part": r.get("part"), "method": r["method"], "seed": r.get("seed"), "order_id": r.get("order_id")}
        runs.append({**base, **{c: num(r.get(c)) for c in STREAM_RUN_COLS}})
        oid = r.get("order_id")
        order = orders[int(oid)] if oid is not None and int(oid) < len(orders) else None
        per = {k: (r.get(src) or {}) for k, src in TASK_KEYS.items()}
        for task in sorted(set().union(*[set(v) for v in per.values()])):
            tasks.append({**base, "group": task, "arrival_pos": (order.index(task) + 1) if order and task in order else None,
                          **{k: num(v.get(task)) for k, v in per.items()}})
    base_rows = []
    for b in d.get("baselines", []):
        per = b.get("final_error_per_task") or b.get("final_per_task") or {}
        common = {"experiment": experiment.upper(), "setting": setting, "method": b["method"]}
        base_rows.append({**common, "group": "all", "final_error": num(b.get("final_avg_error")),
                          **{f"rmse_h{h}": num(b.get(f"rmse_h{h}")) for h in (1, 2, 3)}})
        base_rows += [{**common, "group": g, "final_error": num(v)} for g, v in per.items()]
    return pd.DataFrame(runs), pd.DataFrame(tasks), pd.DataFrame(base_rows)


# ---------- P2: unseen usage condition ----------
UNSEEN_RUN_COLS = ("p2_error", "p2_error_hplus", "p2_error_hminus", "rmse_h1", "rmse_h2", "rmse_h3",
                   "class_accuracy", "class_macro_f1", "cov90", "width90", "is90", "cov80", "width80", "is80")


def unseen_tables(d: dict, experiment: str, setting: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(run-level, run x held-out condition) rows of a D26-style results file (runs and baselines)."""
    runs, folds = [], []
    items = [(r, False) for r in d.get("runs", [])] + [(b, True) for b in d.get("baselines", [])]
    for r, is_base in items:
        base = {"experiment": experiment.upper(), "setting": setting, "seed_set": None if is_base else r.get("set"),
                "method": r["method"], "seed": r.get("seed"), "order_id": r.get("order_id"), "is_baseline": is_base}
        runs.append({**base, **{c: num(r.get(c)) for c in UNSEEN_RUN_COLS}})
        hplus, cov = r.get("per_fold_hplus") or {}, r.get("cov90_per_fold") or {}
        folds += [{**base, "group": g, "error": num(v), "error_hplus": num(hplus.get(g)), "cov90": num(cov.get(g))}
                  for g, v in (r.get("per_fold") or {}).items()]
    return pd.DataFrame(runs), pd.DataFrame(folds)


def d32_tables(d: dict, setting: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gradient boosting under the continual constraint (warm-started, latest only) and the stored-all refit."""
    runs, folds = [], []
    for r in d.get("gbm_runs", []):
        base = {"experiment": "D32", "setting": setting, "seed_set": None, "method": r["method"], "seed": r.get("seed"),
                "order_id": r.get("order_id"), "is_baseline": True}
        runs.append({**base, **{c: num(r.get(c)) for c in UNSEEN_RUN_COLS}})
        folds += [{**base, "group": g, "error": num(v), "error_hplus": math.nan, "cov90": math.nan}
                  for g, v in (r.get("per_fold") or {}).items()]
    for key, s in (d.get("summary") or {}).items():
        if key.startswith(("main_", "replicate_")):
            continue
        runs.append({"experiment": "D32", "setting": setting, "seed_set": None, "method": key, "seed": None,
                     "order_id": None, "is_baseline": True, **{c: num(s.get(c)) for c in UNSEEN_RUN_COLS}})
    return pd.DataFrame(runs), pd.DataFrame(folds)


def d31_tables(d: dict, setting: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Online adaptation (D31): summaries per seed set and method, and their per-condition errors."""
    runs, folds = [], []
    for key, s in (d.get("summary") or {}).items():
        seed_set, method = key.split("_", 1)
        base = {"experiment": "D31", "setting": setting, "seed_set": seed_set, "method": method, "seed": None,
                "order_id": None, "is_baseline": False}
        runs.append({**base, "p2_error": num(s.get("rmse")), "p2_error_hplus": num(s.get("rmse_hplus")),
                     "p2_error_hminus": num(s.get("rmse_hminus")), "rmse_h1": num(s.get("rmse_h1")),
                     "rmse_h2": num(s.get("rmse_h2")), "rmse_h3": num(s.get("rmse_h3")),
                     "class_accuracy": num(s.get("class_accuracy")), "forgetting_rmse": num(s.get("forgetting_rmse"))})
        folds += [{**base, "group": g, "error": num(v), "error_hplus": math.nan, "cov90": math.nan}
                  for g, v in (s.get("per_fold") or {}).items()]
    return pd.DataFrame(runs), pd.DataFrame(folds)


# ---------- tests and verdicts ----------
def tests_table(d: dict, experiment: str, setting: str) -> pd.DataFrame:
    """Per-seed-set paired tests (Wilcoxon, Holm) in one layout: reference a, comparator b, diff = a - b."""
    rows = []
    src = d.get("tests_vs_reference") or d.get("tests") or []
    if isinstance(src, dict):   # D31: {seed_set: {question: {...}}}
        for seed_set, qs in src.items():
            for q, v in (qs or {}).items():
                if isinstance(v, dict) and "mean_diff" in v:
                    rows.append({"seed_set": seed_set, "family": "registered", "a": q, "b": None, "metric": "rmse",
                                 "mean_diff": num(v.get("mean_diff")), "p_value": num(v.get("p_value"))})
    else:
        for v in src:
            rows.append({"seed_set": v.get("set"), "family": v.get("family"), "a": v.get("a", v.get("reference")),
                         "b": v.get("b", v.get("other")), "metric": v.get("metric"), "n": num(v.get("n")),
                         "mean_diff": num(v.get("mean_diff")), "ci_low": num(v.get("ci_low")),
                         "ci_high": num(v.get("ci_high")), "p_value": num(v.get("p_value")),
                         "p_holm": num(v.get("p_holm")), "rank_biserial": num(v.get("rank_biserial"))})
    out = pd.DataFrame(rows)
    if len(out):
        out.insert(0, "setting", setting)
        out.insert(0, "experiment", experiment.upper())
    return out


def verdicts_table(d: dict, experiment: str, setting: str) -> pd.DataFrame:
    """Replication-rule verdicts (main and replicate seed sets together)."""
    rows = [{"experiment": experiment.upper(), "setting": setting, "family": v.get("family"), "a": v.get("a"),
             "b": v.get("b"), "metric": v.get("metric"), "diff_main": num(v.get("diff_main")),
             "diff_replicate": num(v.get("diff_replicate")), "p_holm_main": num(v.get("p_holm_main")),
             "p_holm_replicate": num(v.get("p_holm_replicate")), "verdict": v.get("verdict")}
            for v in d.get("verdicts", [])]
    return pd.DataFrame(rows)


MANY = (("d34", "S7"), ("d36", "S10"))   # many-condition tests (docs/RESULTS.md, D34 and D36)
MANY_COLS = ("rmse", "rmse_hplus", "rmse_hminus", "cov90", "is90")


def many_condition_table(d: dict, experiment: str, dataset: str, setting: str) -> pd.DataFrame:
    """D34/D36: one row per held-out condition, method and seed set (mean over that condition's runs)."""
    rows = [{"dataset": dataset, "protocol": "P2", "experiment": experiment.upper(), "setting": setting,
             "seed_set": r.get("set"), "method": r.get("method"), "condition": r.get("condition"),
             "fold": r.get("fold"), "age_type": r.get("age_type"), "age_temp": num(r.get("age_temp")),
             "stage": r.get("stage"), "lab": r.get("lab"), **{c: num(r.get(c)) for c in MANY_COLS}}
            for r in d.get("condition_values", [])]
    return pd.DataFrame(rows)


# ---------- D38: replay ensemble on S10 and S7 (docs/DECISIONS.md, D38) ----------
D38_DATASETS = ("s10", "s7")
D38_V2 = "tft_residual_er32_ens_cal"
D38_SEED_SETS = ("main", "replicate")
#: The registered primary questions Q1-Q4 (comparator, metric); Q4's comparator is the file's ``previous_model``
#: (S10: SEL; S7: shrunk). Q5 is the pooled 90 % coverage of V2 (no test).
D38_QUESTIONS = (("Q1", "gbm_all", "rmse"), ("Q2", "gbm_er32", "rmse"), ("Q3", "linear_extrapolation", "rmse_hplus"),
                 ("Q4", None, "rmse"))
D38_NAMES = {
    "tft_residual_er32_ens_cal": "V2: replay 32, ensemble, calibrated",
    "tft_residual_er32_ens": "Replay 32, ensemble", "tft_residual_er32": "Replay 32, single run",
    "tft_residual_er8_ens_cal": "Replay 8, ensemble, calibrated", "tft_residual_er8_ens": "Replay 8, ensemble",
    "tft_residual_er8": "Replay 8, single run",
    "tft_residual_ewc": "A0: residual TFT + EWC, single run", "tft_residual_ewc_ens": "A0 ensemble",
    "tft_residual_ewc_ens_cal": "A0 ensemble, calibrated",
    "tft_residual_ewc_selective": "SEL: residual TFT-EWC, shrunk at H+ only (D36)",
    "tft_residual_ewc_shrunk": "Residual TFT-EWC, shrunk (D34)",
    "tft_residual_joint": "Residual TFT, joint (all data stored)",
    "gbm_all": "GBM-ALL (all data stored)", "gbm_er32": "GBM-ER32 (32 stored rows per condition)",
    "gbm_er8": "GBM-ER8 (8 stored rows per condition)", "gbm_last": "GBM-LAST (latest condition only)",
    "linear_extrapolation": "Linear extrapolation", "persistence": "Persistence", "ridge_shared": "Ridge",
}
D38_SHORT = {"gbm_all": "GBM-ALL", "gbm_er32": "GBM-ER32", "linear_extrapolation": "linear extrapolation (H+)",
             "tft_residual_ewc_selective": "SEL (D36 model)", "tft_residual_ewc_shrunk": "shrunk TFT-EWC (D34 model)"}
D38_VERDICT_COLS = ("dataset", "experiment", "seed_set", "question", "comparison", "mean_diff_pp", "median_diff_pp",
                    "n_favour_v2", "n_conditions", "p_holm", "verdict", "answer", "coverage_90")
D38_METHOD_COLS = ("dataset", "experiment", "seed_set", "method", "mean_rmse_pp", "median_rmse_pp", "coverage_90",
                   "coverage_80")


def d38_tables(results: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(verdicts by question, RMSE by method) of results/forecasting_d38_<dataset>_<e>.json.

    Every number is read from the file. ``answer`` applies the registered rule of D38 to stored fields: a question is
    positive for a dataset only if it holds in E1 and in E2, where Q1-Q4 hold when the verdict is ``in_favour`` (the
    D38 direction rule) and Q5 holds when ``coverage_check.adequate``; it is empty unless both files are present.
    Q5 has no test, so its verdict, differences, counts and p-value are empty."""
    verdicts, methods, holds = [], [], {}
    for ds in D38_DATASETS:
        for e in SETTINGS:
            d = load_json(results, f"forecasting_d38_{ds}_{e}")
            if d is None:
                continue
            v2 = d.get("v2", D38_V2)
            cov = (d.get("descriptive") or {}).get("coverage") or {}
            for q, b, metric in D38_QUESTIONS:
                b = b or d.get("previous_model")
                ver = next((x for x in d.get("verdicts", []) if x.get("family") == "primary" and x.get("a") == v2
                            and x.get("b") == b and x.get("metric") == metric), None)
                holds.setdefault((ds, q), []).append(bool(ver and ver.get("in_favour")))
                for s in D38_SEED_SETS:
                    t = next((x for x in d.get("tests", []) if x.get("set") == s and x.get("family") == "primary"
                              and x.get("a") == v2 and x.get("b") == b and x.get("metric") == metric), {})
                    verdicts.append({"dataset": ds, "experiment": e.upper(), "seed_set": s, "question": q,
                                     "comparison": f"V2 vs {D38_SHORT.get(b, b)}", "mean_diff_pp": num(t.get("mean_diff")),
                                     "median_diff_pp": num(t.get("median_diff")),
                                     "n_favour_v2": t.get("n_conditions_a_better"), "n_conditions": t.get("n"),
                                     "p_holm": num(t.get("p_holm")), "verdict": (ver or {}).get("verdict"),
                                     "coverage_90": math.nan})
            check = (d.get("descriptive") or {}).get("coverage_check") or {}
            holds.setdefault((ds, "Q5"), []).append(check.get("method") == v2 and bool(check.get("adequate")))
            for s in D38_SEED_SETS:
                verdicts.append({"dataset": ds, "experiment": e.upper(), "seed_set": s, "question": "Q5",
                                 "comparison": "V2 90% interval coverage (at least 0.85)",
                                 "coverage_90": num(((cov.get(v2) or {}).get(s) or {}).get("90", {}).get("pooled"))})
            desc = d.get("descriptive") or {}
            med = {(r["set"], r["method"]): r.get("rmse") for r in desc.get("median_over_conditions", [])}
            for r in desc.get("mean_over_conditions", []):
                c = (cov.get(r["method"]) or {}).get(r["set"]) or {}
                methods.append({"dataset": ds, "experiment": e.upper(), "seed_set": r["set"],
                                "method": D38_NAMES.get(r["method"], r["method"]), "mean_rmse_pp": num(r.get("rmse")),
                                "median_rmse_pp": num(med.get((r["set"], r["method"]))),
                                "coverage_90": num((c.get("90") or {}).get("pooled")),
                                "coverage_80": num((c.get("80") or {}).get("pooled"))})
    vt = pd.DataFrame(verdicts, columns=list(D38_VERDICT_COLS))
    if len(vt):
        answer = {k: ("positive" if all(h) else "negative") if len(h) == len(SETTINGS) else None
                  for k, h in holds.items()}
        vt["answer"] = [answer[(ds, q)] for ds, q in zip(vt["dataset"], vt["question"])]
        for c in ("n_favour_v2", "n_conditions"):
            vt[c] = pd.to_numeric(vt[c], errors="coerce").astype("Int64")
    return vt, pd.DataFrame(methods, columns=list(D38_METHOD_COLS))


# ---------- other committed tables ----------
def coverage_table(d: dict, setting: str, experiment: str = "D27", names: dict | None = None) -> pd.DataFrame:
    """Split-conformal intervals (D27 rules, D28 arms): pooled and per-condition coverage, interval score and width."""
    rows = []
    for method, levels in (d.get("summary") or {}).items():
        for level, s in (levels or {}).items():
            if not isinstance(s, dict):
                continue
            common = {"experiment": experiment, "setting": setting, "method": (names or {}).get(method, method),
                      "level": num(level) / 100}
            rows.append({**common, "group": "all", "coverage": num(s.get("coverage_pooled")),
                         "interval_score": num(s.get("interval_score")), "width": num(s.get("width")),
                         "decided_share": num(s.get("decided_share"))})
            cov, isc = s.get("coverage_per_fold") or {}, s.get("interval_score_per_fold") or {}
            rows += [{**common, "group": g, "coverage": num(v), "interval_score": num(isc.get(g)), "width": math.nan,
                      "decided_share": math.nan} for g, v in cov.items()]
    return pd.DataFrame(rows)


def horizon_table(tables: Path) -> pd.DataFrame:
    p = tables / "horizon.csv"
    if not p.exists():
        return pd.DataFrame()
    t = pd.read_csv(p).melt(id_vars=["dataset", "method", "name"], var_name="h", value_name="rmse_pp")
    t["horizon"] = t.pop("h").str.extract(r"h(\d+)$", expand=False).astype(int)
    return t.rename(columns={"name": "label"})


def d30_group_table(tables: Path) -> pd.DataFrame:
    """D30 per-group regression and classification metrics (unseen condition / held-out cell), long format."""
    p = tables / "d30_per_group.csv"
    if not p.exists():
        return pd.DataFrame()
    t = pd.read_csv(p).rename(columns={"Battery_ID": "group"})
    long = t.melt(id_vars=["dataset", "method", "group"], var_name="metric", value_name="value")
    long["metric_kind"] = long["metric"].str.split("_").str[0].map({"reg": "regression", "cls": "reuse class",
                                                                  "eol": "end of life"})
    return long.dropna(subset=["value"])


def headline_from_json(results: Path) -> dict:
    """The numbers and registered verdicts of make_headline_table.collect() read straight from the JSON files, for
    environments without torch. The post hoc P1 test against gradient boosting needs the D25 runner and is left out."""
    out = {}
    for e in SETTINGS:
        d23, d25, d26, d32 = (load_json(results, f"forecasting_{n}_{e}") for n in ("d23", "d25", "d26", "d32"))
        if any(x is None for x in (d23, d25, d26, d32)):
            continue
        s25 = {(r["set"], r["method"]): r["final_avg_error_mean"] for r in d25["summary"]}
        s26 = {(r["set"], r["method"]): r["p2_error_mean"] for r in d26["summary"]}
        b23 = {b["method"]: b["final_avg_error"] for b in d23["baselines"]}
        b25 = {b["method"]: b["final_avg_error"] for b in d25["baselines"]}
        b26 = {b["method"]: b["p2_error"] for b in d26["baselines"]}

        def v(rows, a, b, metric):
            x = next((r for r in rows if r["a"] == a and r["b"] == b and r["metric"] == metric), None)
            return None if x is None else {"diff": (x["diff_main"], x["diff_replicate"]), "verdict": x["verdict"]}

        out[e] = {
            "p1": {"proposed": (s25[("main", PROPOSED)], s25[("replicate", PROPOSED)]),
                   "lin": b25["linear_extrapolation"], "gbm": b23["gbm_shared"], "gbm_last": None,
                   "persistence": b25["persistence"],
                   "v_lin": v(d25["verdicts"], PROPOSED, "linear_extrapolation", "final_avg_error"),
                   "v_gbm": None, "v_gbm_last": None},
            "p2": {"proposed": (s26[("main", PROPOSED)], s26[("replicate", PROPOSED)]),
                   "lin": b26["linear_extrapolation"], "gbm": b26["gbm_shared"],
                   "gbm_last": d32["summary"]["gbm_last"]["p2_error"], "persistence": b26["persistence"],
                   "v_lin": v(d26["verdicts"], PROPOSED, "linear_extrapolation", "p2_error"),
                   "v_gbm": v(d26["verdicts"], PROPOSED, "gbm_shared", "p2_error"),
                   "v_gbm_last": v(d32["verdicts"], PROPOSED, "gbm_last", "p2_error")},
        }
    return out


def headline_table(results: Path = ROOT / "results") -> pd.DataFrame:
    """Rows of the paper's headline table (scripts/make_headline_table.py); without torch, the JSON-only version."""
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        import make_headline_table as H  # noqa: E402  (imports the D25 runner, which needs torch)
        data = H.collect() if results.resolve() == (ROOT / "results").resolve() else headline_from_json(results)
    except ImportError as exc:
        print(f"  headline: {exc}; using the JSON-only version (no post hoc P1 test against gradient boosting)")
        data = headline_from_json(results)
    proto = {"p1": "Continual stream (P1)", "p2": "Unseen usage condition (P2)"}
    rows = []
    for e, parts in data.items():
        for p, v in parts.items():
            for k, lab in (("proposed", PROPOSED), ("lin", "linear_extrapolation"), ("gbm", "gbm_shared"),
                           ("gbm_last", "gbm_last"), ("persistence", "persistence")):
                x = v[k]
                if x is None:
                    continue
                main, rep = (x if isinstance(x, tuple) else (x, x))
                rows.append({"protocol": proto[p], "setting": SETTINGS[e], "method": lab, "label": method_label(lab),
                             "rmse_main": num(main), "rmse_replicate": num(rep)})
            for k, lab in (("v_lin", "linear_extrapolation"), ("v_gbm", "gbm_shared"), ("v_gbm_last", "gbm_last")):
                t = v[k]
                if t is None:
                    continue
                rows.append({"protocol": proto[p], "setting": SETTINGS[e], "method": f"vs {lab}",
                             "label": f"Proposed vs {method_label(lab)}", "diff_main": num(t["diff"][0]),
                             "diff_replicate": num(t["diff"][1]), "verdict": t["verdict"],
                             "post_hoc": bool(t.get("post_hoc", False)) or (p == "p1" and k == "v_gbm_last")})   # D35 is post hoc
    return pd.DataFrame(rows)


# ---------- data-derived tables ----------
def soh_table(data: Path) -> pd.DataFrame:
    """SOH trajectories: S1 C/20 labels (D17) and S6 C/25 labels, relative to each cell's first valid check."""
    frames = []
    p = data / "rpt_health_c20.csv"
    if p.exists():
        t = pd.read_csv(p)
        frames.append(pd.DataFrame({
            "dataset": "S1", "group": t["Battery_ID"].astype(str), "unit": t["Battery_ID"].astype(str),
            "check_index": t["RPT_Number"].astype(int), "check_label": t["RPT"].astype(str),
            "check_time": pd.to_datetime(t["Test_Date"], errors="coerce"), "cycle": math.nan,
            "capacity_ah": t["Capacity_Ah"], "reference_capacity_ah": t["Reference_Capacity_Ah"],
            "soh_pct": t["SOH_Percent"], "temperature_c": t.get("rpt_temp_mean", pd.Series(math.nan, index=t.index)),
            "valid": t["SOH_Percent"].notna()}))
    p = data / "wheeler_checks.csv"
    if p.exists():
        t = pd.read_csv(p)
        frames.append(pd.DataFrame({
            "dataset": "S6", "group": t["Battery_ID"].astype(str), "unit": t["cell"].astype(str),
            "check_index": t["cycle"].astype(int), "check_label": "cycle " + t["cycle"].astype(str),
            "check_time": pd.to_datetime(t.get("check_time"), errors="coerce"), "cycle": t["cycle"].astype(float),
            "capacity_ah": t["Capacity_Ah"], "reference_capacity_ah": t["Reference_Capacity_Ah"],
            "soh_pct": t["SOH_Percent"], "temperature_c": math.nan,
            "valid": t["valid"].astype(str).str.lower().isin(("true", "1"))}))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # Temperature: S1 RPT mean where recorded; S6 checks were run in a 50 degC chamber with no per-check reading.
    out["temperature_source"] = np.where(out["temperature_c"].notna(), "measured (RPT mean)",
                                         np.where(out["dataset"] == "S6", "not recorded (50 degC chamber)",
                                                  "missing, not filled"))
    out["fade_pp"] = 100.0 - out["soh_pct"]
    out["reuse_class"] = class_name(reuse_class(out["soh_pct"]))
    out = out.sort_values(["dataset", "unit", "check_index"], kind="mergesort").reset_index(drop=True)
    return out


def predictions_table(data: Path) -> pd.DataFrame:
    """Saved D30 forecasts (scripts/run_d30.py), one row per method, seed and test sample, with reuse classes."""
    frames = []
    for ds in ("s1", "s6"):
        p = data / "predictions" / f"d30_{ds}_p2.csv.gz"
        if p.exists():
            frames.append(pd.read_csv(p).assign(dataset=ds.upper()))
    if not frames:
        return pd.DataFrame()
    t = pd.concat(frames, ignore_index=True).rename(columns={"Battery_ID": "group"})
    t["error_pp"] = t["soh_pred"] - t["soh_target"]
    t["class_true"] = class_name(reuse_class(t["soh_target"]))
    t["class_pred"] = class_name(reuse_class(t["soh_pred"]))
    t["class_distance"] = (reuse_class(t["soh_pred"]) - reuse_class(t["soh_target"]))
    return t


# ---------- summary for the JS page ----------
PER_GROUP_BASELINES = ("linear_extrapolation", "gbm_shared", "persistence")


def per_group(stream_task: pd.DataFrame, stream_base: pd.DataFrame, unseen_fold: pd.DataFrame) -> pd.DataFrame:
    """Per-condition test RMSE (pp) of the proposed model (mean, s.d. over the main seed set's runs) and the
    baselines, in both S6 protocols: P1 final error per task (D25; gradient boosting from D23) and P2 error per
    held-out condition (D26)."""
    rows = []
    for e in SETTINGS:
        if len(stream_task):
            t = stream_task[(stream_task["experiment"] == "D25") & (stream_task["setting"] == e)
                            & (stream_task["method"] == PROPOSED) & (stream_task["seed_set"] == "main")]
            for g, x in t.groupby("group")["final_error"]:
                rows.append({"protocol": "P1", "setting": e, "method": PROPOSED, "group": g, "rmse_mean": x.mean(),
                             "rmse_sd": x.std(), "n_runs": int(x.notna().sum())})
        if len(stream_base):
            b = stream_base[(stream_base["setting"] == e) & (stream_base["group"] != "all")
                            & stream_base["method"].isin(PER_GROUP_BASELINES)
                            & (((stream_base["experiment"] == "D25") & (stream_base["method"] != "gbm_shared"))
                               | ((stream_base["experiment"] == "D23") & (stream_base["method"] == "gbm_shared")))]
            rows += [{"protocol": "P1", "setting": e, "method": r.method, "group": r.group, "rmse_mean": r.final_error,
                      "rmse_sd": math.nan, "n_runs": 1} for r in b.itertuples()]
        if len(unseen_fold):
            u = unseen_fold[(unseen_fold["experiment"] == "D26") & (unseen_fold["setting"] == e)]
            prop = u[(u["method"] == PROPOSED) & (u["seed_set"] == "main")]
            for g, x in prop.groupby("group")["error"]:
                rows.append({"protocol": "P2", "setting": e, "method": PROPOSED, "group": g, "rmse_mean": x.mean(),
                             "rmse_sd": x.std(), "n_runs": int(x.notna().sum())})
            base = u[u["is_baseline"].astype(bool) & u["method"].isin(PER_GROUP_BASELINES)]
            rows += [{"protocol": "P2", "setting": e, "method": r.method, "group": r.group, "rmse_mean": r.error,
                      "rmse_sd": math.nan, "n_runs": 1} for r in base.itertuples()]
    out = pd.DataFrame(rows)
    if len(out):
        out["label"] = out["method"].map(method_label)
    return out


def summary(headline: pd.DataFrame, stream_base: pd.DataFrame, unseen_runs: pd.DataFrame,
            groups: pd.DataFrame, counts: dict, groups_rmse: pd.DataFrame | None = None) -> dict:
    def pick(df, **kw):
        m = pd.Series(True, index=df.index)
        for k, v in kw.items():
            m &= df[k] == v
        return df[m]

    kpis = []
    if len(stream_base):
        lin = pick(stream_base, experiment="D25", setting="e1", method="linear_extrapolation", group="all")
        if len(lin):
            kpis.append({"id": "p1_lin", "label": "Continual stream: linear extrapolation",
                         "value": float(lin["final_error"].iloc[0]), "unit": "pp RMSE",
                         "note": "S6, h = 1-3; the proposed model is practically equivalent, no learned model beats it",
                         "source": "results/forecasting_d25_e1.json"})
    if len(unseen_runs):
        g = pick(unseen_runs, experiment="D26", setting="e1", method="gbm_shared")
        if len(g):
            kpis.append({"id": "p2_gbm", "label": "Unseen condition: gradient boosting", "value": float(g["p2_error"].iloc[0]),
                         "unit": "pp RMSE", "note": "S6, h = 1-3; the most accurate method on unseen usage conditions",
                         "source": "results/forecasting_d26_e1.json"})
        j = pick(unseen_runs, experiment="D31", setting="e1", method="joint_static", seed_set="main")
        if len(j):
            kpis.append({"id": "p2_joint", "label": "Residual TFT trained jointly", "value": float(j["p2_error"].iloc[0]),
                         "unit": "pp RMSE", "note": "Unseen condition (S6, h = 1-3); close to gradient boosting",
                         "source": "results/forecasting_d31_e1.json"})
    if len(headline):
        h = headline[(headline["method"] == PROPOSED) & headline["protocol"].str.startswith("Unseen")
                     & (headline["setting"] == SETTINGS["e1"])]
        if len(h):
            kpis.append({"id": "p2_proposed", "label": "Residual TFT-EWC (proposed)",
                         "value": float(h["rmse_main"].iloc[0]), "unit": "pp RMSE",
                         "note": "Unseen condition (S6, h = 1-3), mean of 50 runs, main seed set",
                         "source": "results/forecasting_d26_e1.json (paper Table III)"})
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git_commit(),
        "kpis": kpis,
        "counts": counts,
        "groups": json.loads(groups.to_json(orient="records")),
        "headline": json.loads(headline.to_json(orient="records")) if len(headline) else [],
        "per_group": json.loads(groups_rmse.to_json(orient="records")) if groups_rmse is not None and len(groups_rmse) else [],
        "caveats": [
            "SOH is relative to each cell's first valid capacity check, not to nominal capacity.",
            "S1 (NMC) and S6 (LFP) are analysed separately; no cross-chemistry transfer is claimed.",
            "Reuse classes describe capacity fade since the first check, not suitability for an application.",
            "Errors are reported per battery group and against simple baselines (persistence, linear extrapolation).",
            "Some earlier significance results did not replicate on a different platform; see docs/RESULTS.md.",
        ],
    }


# ---------- I/O ----------
def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def sha256(p: Path) -> str:
    # Hash with LF line endings so a Windows (core.autocrlf) checkout gives the same hash as the Git blob.
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write(frames: dict[str, pd.DataFrame], out: Path) -> dict[str, int]:
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, df in frames.items():
        if df is None or df.empty:
            print(f"  {name}: no rows, not written")
            continue
        df.to_csv(out / f"{name}.csv", index=False, float_format="%.10g")
        counts[name] = int(len(df))
        print(f"  {name}: {len(df):,} rows -> {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return counts


def build_results(results: Path) -> dict[str, pd.DataFrame]:
    tables = results / "tables"
    s_runs, s_tasks, s_base, u_runs, u_folds, tests, verdicts, cov = ([] for _ in range(8))
    for exp in STREAM_EXPS + UNSEEN_EXPS + ("d31", "d32"):
        for e in SETTINGS:
            d = load_json(results, f"forecasting_{exp}_{e}")
            if d is None:
                continue
            if exp in STREAM_EXPS:
                r, t, b = stream_tables(d, exp, e)
                s_runs.append(r), s_tasks.append(t), s_base.append(b)
            elif exp in UNSEEN_EXPS:
                r, f = unseen_tables(d, exp, e)
                u_runs.append(r), u_folds.append(f)
            elif exp == "d31":
                r, f = d31_tables(d, e)
                u_runs.append(r), u_folds.append(f)
            else:
                r, f = d32_tables(d, e)
                u_runs.append(r), u_folds.append(f)
            tests.append(tests_table(d, exp, e))
            verdicts.append(verdicts_table(d, exp, e))
    many = []
    for (exp, dataset), e in ((m, e) for m in MANY for e in SETTINGS):
        d = load_json(results, f"forecasting_{exp}_{e}")
        if d is not None:
            many.append(many_condition_table(d, exp, dataset, e))
            verdicts.append(verdicts_table(d, exp, e).assign(dataset=dataset))
    for e in SETTINGS:
        d = load_json(results, f"forecasting_d35_{e}")
        if d is not None:
            verdicts.append(verdicts_table(d, "d35", e))
    for e in SETTINGS:
        d = load_json(results, f"forecasting_d27_{e}")
        if d is not None:
            cov.append(coverage_table(d, e))
        d = load_json(results, f"forecasting_d28_{e}")
        if d is not None:
            cov.append(coverage_table(d, e, "D28", D28_ARMS))

    cat = lambda xs: pd.concat([x for x in xs if len(x)], ignore_index=True) if any(len(x) for x in xs) else pd.DataFrame()  # noqa: E731
    frames = {"fact_stream_run": cat(s_runs), "fact_stream_task": cat(s_tasks), "fact_stream_baseline": cat(s_base),
              "fact_unseen_run": cat(u_runs), "fact_unseen_fold": cat(u_folds), "fact_tests": cat(tests),
              "fact_verdicts": cat(verdicts), "fact_many_condition": cat(many), "fact_coverage": cat(cov), "fact_horizon": horizon_table(tables),
              "fact_d30_group": d30_group_table(tables), "fact_headline": headline_table(results)}
    methods = set()
    for df in frames.values():
        if "method" in df:
            methods |= set(df["method"].dropna().astype(str))
    # Every S6 fact table carries the dataset and protocol, so the dashboard's Dataset and Protocol filters apply.
    for name, proto in (("fact_stream_run", "P1"), ("fact_stream_task", "P1"), ("fact_stream_baseline", "P1"),
                        ("fact_unseen_run", "P2"), ("fact_unseen_fold", "P2"), ("fact_coverage", "P2")):
        df = frames[name]
        if len(df):
            df.insert(0, "protocol", proto)
            df.insert(0, "dataset", "S6")
            for c in ("seed", "order_id"):
                if c in df:
                    df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    if len(frames["fact_d30_group"]):
        frames["fact_d30_group"].insert(1, "protocol", "P2")
    frames = {"dim_group": dim_group(), "dim_method": dim_method(m for m in methods if not m.startswith("vs ")),
              **frames}
    # D38 tables carry readable method names (not keys of dim_method), so they are added after dim_method is built.
    frames["fact_d38_verdicts"], frames["fact_d38_method"] = d38_tables(results)
    return frames


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", str(ROOT / "data")))
    ap.add_argument("--out-results", default=None, help="default: <results>/dashboard")
    ap.add_argument("--out-data", default=None, help="default: <data-root>/dashboard")
    args = ap.parse_args(argv)
    results, data = Path(args.results), Path(args.data_root)
    if not data.is_absolute():
        data = ROOT / data
    out_r = Path(args.out_results) if args.out_results else results / "dashboard"
    out_d = Path(args.out_data) if args.out_data else data / "dashboard"

    print("Results-derived tables:")
    frames = build_results(results)
    counts = write(frames, out_r)

    print("Data-derived tables:")
    dframes = {"fact_soh": soh_table(data), "fact_predictions": predictions_table(data)}
    if all(df.empty for df in dframes.values()):
        print(f"  no inputs in {data} (rpt_health_c20.csv, wheeler_checks.csv, predictions/); skipped")
        dcounts = {}
    else:
        dcounts = write(dframes, out_d)

    summ = summary(frames["fact_headline"], frames["fact_stream_baseline"], frames["fact_unseen_run"],
                   frames["dim_group"], {**counts, **dcounts},
                   per_group(frames["fact_stream_task"], frames["fact_stream_baseline"], frames["fact_unseen_fold"]))
    (out_r / "summary.json").write_text(json.dumps(summ, indent=2, allow_nan=False, default=str), encoding="utf-8")
    sources = sorted(p for p in results.glob("forecasting_d[23]*_e*.json")) + sorted((results / "tables").glob("*.csv"))
    manifest = {"generated_utc": summ["generated_utc"], "commit": summ["commit"], "row_counts": {**counts, **dcounts},
                "data_root": str(data.relative_to(ROOT)) if data.is_relative_to(ROOT) else "external (DATA_ROOT)",
                "sources": {p.relative_to(results.parent).as_posix() if p.is_relative_to(results.parent)
                                                    else str(p): sha256(p) for p in sources}}
    (out_r / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote summary.json and manifest.json to {out_r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
