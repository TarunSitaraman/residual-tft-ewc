"""Run decision D30: the full regression and classification metric set for the headline P1/P2 forecasts.

Re-runs the forecasts behind paper Tables P1 and P2 (notebooks 12 and 13: S1 and S6, protocol D18/D20) with the
hyperparameters selected there (read from the committed results files, no re-tuning), and scores the same predictions
with every metric of ``src/metrics.py``:

- **P2** (unseen battery in S1, unseen usage condition in S6): all baselines, MLP and TFT, every seed.
- **P1 baselines** (deterministic, fitted once on all training splits).
- ``--with-p1`` adds the P1 continual methods (10 orders x 5 seeds per method; slow).

**Stop condition.** The per-group RMSE of every P2 method (mean over seeds) and the P1 RMSE of every method run must
reproduce the committed values (``results/forecasting.json``, ``results/forecasting_wheeler.json``) to ``TOL``;
otherwise nothing is written. The new metrics are therefore computed on the predictions that the paper reports.

Outputs: ``results/forecasting_d30_<dataset>.json`` (per group and seed, summaries) and, outside Git, the P2
predictions ``<data>/predictions/d30_<dataset>_p2.csv.gz`` so further metrics need no retraining.

Usage (from the repository root)::

    python scripts/run_d30.py --dataset s1
    python scripts/run_d30.py --dataset s6
    python scripts/run_d30.py --dataset s6 --with-p1
    python scripts/run_d30.py --dataset s6 --profile smoke --allow-dirty   # code check only (not reportable)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import data_pipeline as dp  # noqa: E402
from src import evaluation as ev  # noqa: E402
from src import experiments as X  # noqa: E402
from src import forecasting as FC  # noqa: E402
from src import metrics as M  # noqa: E402

COMMITTED = {"s1": ROOT / "results" / "forecasting.json", "s6": ROOT / "results" / "forecasting_wheeler.json"}
TOL = 1e-6                          # pp; RMSE reproduction tolerance (results files store 10 significant digits)
SUMMARY_KEYS = ([f"reg_{k}" for k in M.REGRESSION_KEYS] + [f"cls_{k}" for k in M.CLASS_KEYS]
                + [f"eol_{k}" for k in M.BINARY_KEYS])


# ---------------------------------------------------------------------------
# Data and configuration (as notebooks 12 and 13)
# ---------------------------------------------------------------------------

def load_samples(dataset: str, data_root: Path) -> tuple[pd.DataFrame, dict, dict | None]:
    """Forecasting samples, feature sets and baseline column overrides, built exactly as in notebook 12 / 13."""
    if dataset == "s1":
        dp.check_data_files(data_root, required=["diagnostic_features.csv", "cleaned_battery_data.csv"])
        table = dp.read_dataset(data_root / "diagnostic_features.csv")
        temp_col = next(c for c in dp.read_header(data_root / "cleaned_battery_data.csv") if c.startswith("Aux_Temperature"))
        points = dp.read_dataset(data_root / "cleaned_battery_data.csv",
                                 usecols=["Date_Time", "Source_Folder", "Cycling_Test", temp_col])
        samples = FC.build_samples(table, FC.cycling_blocks(points))
        return samples, FC.feature_sets(samples), None
    from src import wheeler as W

    dp.check_data_files(data_root, required=["wheeler_checks.csv"])
    checks = dp.read_dataset(data_root / "wheeler_checks.csv")
    checks["check_time"] = pd.to_datetime(checks["check_time"])
    samples = W.build_samples(checks)
    return samples, W.feature_sets(samples), W.BASELINE_COLUMNS


def tuned_config(sets: dict, committed: dict, profile: str):
    """The notebook's ``tuning["tuned_cl_config"]`` rebuilt from the committed selections (no re-tuning)."""
    if profile == "smoke":
        cfg = FC.cl_config(sets["F1"], n_orders=1, seeds=(0,))
        cfg.common_kwargs["epochs"] = 2
        cfg.backbone_kwargs = {"tft": {**FC.TFT_SETTINGS, "lr": 1e-3}, "mlp": {"lr": 1e-3}}
        return cfg
    t = committed["tuning"]
    cfg = FC.cl_config(sets["F1"])
    cfg.backbone_kwargs = {b: {**cfg.backbone_kwargs.get(b, {}), **v} for b, v in t["backbone_selected"].items()}
    cfg.method_kwargs = {m: dict(p) for m, p in t["selected"].items()}
    return cfg


# ---------------------------------------------------------------------------
# Reproduction check (stop condition)
# ---------------------------------------------------------------------------

def reproduction(p2: pd.DataFrame, p1: pd.DataFrame | None, committed: dict) -> dict:
    """Max abs difference between recomputed and committed RMSE: P2 per (method, group), P1 per method."""
    diffs = {}
    per_group = p2.groupby(["method", dp.GROUP_COL])["RMSE"].mean().unstack()
    for row in committed["p2_rmse"]:
        m = row["method"]
        for g in per_group.columns:
            if g in row and m in per_group.index:
                diffs[f"p2/{m}/{g}"] = abs(float(per_group.loc[m, g]) - float(row[g]))
            else:
                diffs[f"p2/{m}/{g}"] = float("inf")              # missing on one side
    if p1 is not None:
        committed_p1 = {r["method"]: r["mean"] for r in committed["p1_rmse"]}
        keys = [c for c in ("order_id", "seed") if c in p1]
        per_run = p1.groupby(["method", *keys], dropna=False)["RMSE"].mean()
        for m, v in per_run.groupby("method").mean().items():
            diffs[f"p1/{m}"] = abs(float(v) - float(committed_p1[m])) if m in committed_p1 else float("inf")
    worst = max(diffs, key=diffs.get) if diffs else None
    return {"tolerance": TOL, "n_compared": len(diffs), "max_abs_diff": float(diffs[worst]) if worst else float("nan"),
            "worst": worst, "passed": bool(diffs) and all(v <= TOL for v in diffs.values())}


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def per_group_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Every D30 metric per (method, group): mean over seeds (or over orders x seeds in P1)."""
    keys = [k for k in SUMMARY_KEYS if k in rows]
    return rows.groupby(["method", dp.GROUP_COL])[keys].mean().reset_index()


def over_groups(per_group: pd.DataFrame) -> pd.DataFrame:
    """Mean and s.d. over groups of each metric (NaN groups skipped; ``<metric>_n`` counts the defined ones)."""
    out = []
    for m, g in per_group.groupby("method", sort=False):
        row = {"method": m}
        for k in [c for c in SUMMARY_KEYS if c in g]:
            v = g[k].to_numpy(float)
            row[k], row[f"{k}_sd"], row[f"{k}_n"] = (float(np.nanmean(v)) if np.isfinite(v).any() else float("nan"),
                                                     float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else float("nan"),
                                                     int(np.isfinite(v).sum()))
        out.append(row)
    return pd.DataFrame(out)


def pooled(preds: pd.DataFrame) -> pd.DataFrame:
    """Every D30 metric on all held-out samples pooled, per seed, then the mean over seeds (and s.d. over seeds)."""
    rows = []
    for m, g in preds.groupby("method", sort=False):
        per_seed = [M.flat(s["soh_target"], s["soh_pred"], s["soh_origin"]) for _, s in g.groupby("seed")]
        f = pd.DataFrame(per_seed)
        row = {"method": m, "n_seeds": len(per_seed), "n_samples": int(len(g) / len(per_seed))}
        for k in [c for c in SUMMARY_KEYS if c in f]:
            row[k] = float(f[k].mean())
            row[f"{k}_sd"] = float(f[k].std(ddof=1)) if len(f) > 1 else float("nan")
        # Confusion matrix of seed 0 (baselines have only seed 0), for the figure.
        first = per_seed[0]
        row["confusion"] = [[first[f"cls_cm_{i}_{j}"] for j in range(len(ev.REUSE_CUTPOINTS) + 1)]
                            for i in range(len(ev.REUSE_CUTPOINTS) + 1)]
        rows.append(row)
    return pd.DataFrame(rows)


def class_support(samples: pd.DataFrame) -> dict:
    """Measured reuse-class counts of all samples' targets, per group: shows where classification is single-class."""
    c = ev.reuse_class(samples["soh_target"].to_numpy(float))
    t = pd.crosstab(samples[dp.GROUP_COL], c).reindex(columns=range(len(ev.REUSE_CUTPOINTS) + 1), fill_value=0)
    return {str(g): [int(v) for v in r] for g, r in t.iterrows()}


# ---------------------------------------------------------------------------

def main() -> int:
    import run_d27

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", choices=tuple(COMMITTED), required=True)
    ap.add_argument("--profile", choices=("paper", "smoke"), default="paper")
    ap.add_argument("--with-p1", action="store_true", help="also re-run the P1 continual methods (slow)")
    ap.add_argument("--out", type=Path, help="default results/forecasting_d30_<dataset>.json")
    ap.add_argument("--allow-dirty", action="store_true", help="code check only; recorded in the results file")
    ap.add_argument("--allow-unpinned", action="store_true", help="code check only; recorded in the results file")
    a = ap.parse_args()
    problems = run_d27.preflight(a.allow_dirty, a.allow_unpinned)
    if problems:
        print("PRE-FLIGHT:\n" + "\n".join(problems), flush=True)
        return 2

    import torch
    torch.set_num_threads(1)
    data_root = dp.resolve_data_root(os.environ.get("DATA_ROOT", "data"), ROOT)
    committed = json.loads(COMMITTED[a.dataset].read_text())
    samples, sets, columns = load_samples(a.dataset, data_root)
    if a.profile == "paper" and len(samples) != committed["n_samples"]:
        raise SystemExit(f"{len(samples)} samples, committed run had {committed['n_samples']}: data differ")
    cfg = tuned_config(sets, committed, a.profile)
    cp = FC.Checkpoint(data_root / "cache" / f"d30_{a.dataset}", [FC.Checkpoint.frame_hash(samples), a.profile])
    print(f"D30 {a.dataset}: {len(samples)} samples, {dp.groups_in(samples)}", flush=True)

    p2, preds = FC.run_p2(samples, cfg, sets, columns=columns, checkpoint=cp, keep_predictions=True)
    base_p1 = FC.run_baselines_p1(samples, sets, columns=columns)
    p1_cl = None
    if a.with_p1:
        p1_cl = FC.run_p1(samples, cfg, checkpoint=cp)["final_per_task"]
    p1 = pd.concat([base_p1.assign(seed=0), *([p1_cl] if p1_cl is not None else [])], ignore_index=True)

    rep = reproduction(p2, p1, committed) if a.profile == "paper" else {"skipped": "smoke profile"}
    print(f"reproduction check: {rep}", flush=True)
    if a.profile == "paper" and not rep["passed"]:
        print("STOP: committed RMSE not reproduced; no results file written (D30).", flush=True)
        return 3

    pred_dir = data_root / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"d30_{a.dataset}_p2{'' if a.profile == 'paper' else '_smoke'}.csv.gz"
    preds.to_csv(pred_path, index=False)

    p2_groups, p1_groups = per_group_table(p2), per_group_table(p1)
    out = a.out or ROOT / "results" / (f"forecasting_d30_{a.dataset}" + ("" if a.profile == "paper" else "_smoke") + ".json")
    X.write_results(out, {
        "decision": "D30", "dataset": a.dataset, "profile": a.profile, "source_results": COMMITTED[a.dataset].name,
        "n_samples": int(len(samples)), "groups": dp.groups_in(samples),
        "overrides": {"allow_dirty": a.allow_dirty, "allow_unpinned": a.allow_unpinned},
        "eol_threshold_soh": M.EOL_SOH, "reuse_cutpoints": list(ev.REUSE_CUTPOINTS),
        "class_support": class_support(samples), "reproduction_check": rep,
        "predictions_file": str(pred_path.relative_to(data_root)), "predictions_rows": int(len(preds)),
        "p2_pooled": pooled(preds), "p2_over_groups": over_groups(p2_groups), "p2_per_group": p2_groups,
        "p2_per_seed": p2.drop(columns=[c for c in p2 if c.startswith("cls_cm_")]),
        "p1_over_groups": over_groups(p1_groups), "p1_per_group": p1_groups,
        "p1_includes_continual_methods": bool(a.with_p1),
    }, ROOT)
    show = pooled(preds).set_index("method")[["reg_RMSE", "reg_MAE", "reg_MedAE", "reg_Bias", "reg_NRMSE", "reg_R2_change",
                                              "cls_accuracy", "cls_balanced_accuracy", "cls_mcc", "eol_auc_roc"]]
    print(show.round(3).to_string(), flush=True)
    print(f"written {out.relative_to(ROOT)} and {pred_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
