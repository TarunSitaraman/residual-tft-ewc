"""Run decision D29: physics-structured Bayesian forecaster with a learned residual on unseen usage conditions (P2).

Setting as D27/D28 (S6 samples, E1 = horizons 1-3, E2 = 1-6, seven folds, every held-out sample scored).

- **Ageing law:** SOH(E) = 100 - a_b * E**z per cell, E = cycle * DoD (equivalent full cycles); z shared, chosen on
  the training cases by profile least squares; a_b ~ N(mu, tau2) across cells; noise sigma2 from the training fits.
- **Update:** at an origin, the conjugate Gaussian posterior of a_b from that cell's checks up to the origin cycle.
- **Arms:** PHYS-BAYES (uncalibrated Gaussian intervals), PHYS-LOCO and HYBRID-LOCO (physics + gradient-boosting
  residual; leave-one-condition-out conformal half-widths), PHYS-CYC-LOCO (clock = cycles, DoD ignored).

The D27 gradient boosting is recomputed and must reproduce the D27 per-fold RMSE (stop condition). Comparators D27
GBM-LOCO and D28 REFIT+TRACK are read from the committed results files.

Usage (from the repository root)::

    python scripts/run_d29.py --exp e1
    python scripts/run_d29.py --exp e2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_d26  # noqa: E402
import run_d27  # noqa: E402
import run_d28  # noqa: E402
from src import experiments as X  # noqa: E402
from src import forecasting as FC  # noqa: E402
from src.calibration import LEVELS  # noqa: E402

ARMS = ("phys_bayes", "phys_loco", "hybrid_loco", "phys_cyc_loco")
Z_GRID = tuple(np.round(np.arange(0.30, 1.5001, 0.05), 2))
Q1_MIN, TOL, PRIMARY_CASE = 0.80, 1e-9, "case7"


def checks(frame: pd.DataFrame) -> pd.DataFrame:
    """Every valid check of every cell (cycle, SOH, DoD), from the origins and targets of its samples."""
    a = frame[["cell", "Battery_ID", "origin_cycle", "soh_origin", "dod"]].set_axis(
        ["cell", "case", "cycle", "soh", "dod"], axis=1)
    b = frame.assign(cycle=frame["origin_cycle"] + frame["cycles_to_target"])[
        ["cell", "Battery_ID", "cycle", "soh_target", "dod"]].set_axis(["cell", "case", "cycle", "soh", "dod"], axis=1)
    out = pd.concat([a, b], ignore_index=True).drop_duplicates(["cell", "cycle"]).sort_values(["cell", "cycle"])
    return out.reset_index(drop=True)


def clock(cycle: np.ndarray, dod: np.ndarray, kind: str) -> np.ndarray:
    return np.asarray(cycle, float) * (np.asarray(dod, float) if kind == "efc" else 1.0)


def fit_population(chk: pd.DataFrame, kind: str) -> dict:
    """Shared exponent z (profile least squares on the grid), prior mean/variance of a, and noise variance."""
    e = clock(chk["cycle"], chk["dod"], kind)
    loss = 100.0 - chk["soh"].to_numpy(float)
    cells = chk["cell"].to_numpy()
    best = None
    for z in Z_GRID:
        x = e ** z
        sse, a = 0.0, {}
        for c in np.unique(cells):
            m = cells == c
            a[c] = float(np.sum(x[m] * loss[m]) / np.sum(x[m] ** 2))
            sse += float(np.sum((loss[m] - a[c] * x[m]) ** 2))
        if best is None or sse < best[1] - 1e-12:
            best = (float(z), sse, a)
    z, sse, a = best
    dof = sum(int((cells == c).sum()) - 1 for c in a)
    vals = np.array(list(a.values()))
    return {"z": z, "mu": float(vals.mean()), "tau2": float(vals.var(ddof=1)), "sigma2": float(sse / dof),
            "kind": kind, "n_cells": len(a)}


def phys_predict(frame: pd.DataFrame, chk: pd.DataFrame, pop: dict) -> tuple[np.ndarray, np.ndarray]:
    """Forecast Delta and predictive variance from each cell's posterior given its checks up to the origin."""
    z, mu, tau2, s2 = pop["z"], pop["mu"], pop["tau2"], pop["sigma2"]
    by_cell = {c: g for c, g in chk.groupby("cell")}
    mean, var = np.empty(len(frame)), np.empty(len(frame))
    dod = frame["dod"].to_numpy(float)
    xo = clock(frame["origin_cycle"], dod, pop["kind"]) ** z
    xt = clock(frame["origin_cycle"] + frame["cycles_to_target"], dod, pop["kind"]) ** z
    for i, (cell, oc) in enumerate(zip(frame["cell"], frame["origin_cycle"])):
        g = by_cell[cell]
        g = g[g["cycle"] <= oc]
        x = clock(g["cycle"], g["dod"], pop["kind"]) ** z
        prec = 1.0 / tau2 + np.sum(x ** 2) / s2
        a_hat = (mu / tau2 + np.sum(x * (100.0 - g["soh"].to_numpy(float))) / s2) / prec
        mean[i] = -a_hat * (xt[i] - xo[i])
        var[i] = (xt[i] - xo[i]) ** 2 / prec + 2.0 * s2
    return mean, var


def hybrid_predict(fit: pd.DataFrame, test: pd.DataFrame, chk: pd.DataFrame, features: list[str],
                   kind: str = "efc") -> tuple[np.ndarray, np.ndarray, dict]:
    """Physics centre and physics + gradient-boosting residual, both fitted on ``fit`` only."""
    pop = fit_population(chk[chk["case"].isin(fit["Battery_ID"].unique())], kind)
    p_fit, _ = phys_predict(fit, chk, pop)
    p_test, v_test = phys_predict(test, chk, pop)
    resid = fit.assign(**{FC.TARGET: fit[FC.TARGET].to_numpy(float) - p_fit})
    return p_test, p_test + run_d28.gbm_fit_predict(resid, test, features), {**pop, "var": v_test}


class D29(run_d27.D27):
    def __init__(self, exp: str) -> None:
        super().__init__(exp)
        self.chk = checks(self.direct)

    def fold(self, case: str) -> dict:
        d, feats, chk = self.direct, self.features, self.chk
        test = d[d["Battery_ID"] == case].reset_index(drop=True)
        y = test[FC.TARGET].to_numpy(float)
        fit_rows = run_d27.p2_fit_rows(d, case)
        p_gbm = run_d27.gbm_predict(fit_rows, test, feats)                   # D27 reproduction
        phys, hybrid, pop = hybrid_predict(fit_rows, test, chk, feats)
        phys_cyc, _, pop_cyc = hybrid_predict(fit_rows, test, chk, feats, kind="cyc")
        others = d[d["Battery_ID"] != case].reset_index(drop=True)
        cal = {"phys": [], "hybrid": [], "cyc": []}
        for j in sorted(others["Battery_ID"].unique()):
            rows = others[others["Battery_ID"] == j].reset_index(drop=True)
            inner_fit = run_d27.p2_fit_rows(others, j)
            yj = rows[FC.TARGET].to_numpy(float)
            p_j, h_j, _ = hybrid_predict(inner_fit, rows, chk, feats)
            c_j, _, _ = hybrid_predict(inner_fit, rows, chk, feats, kind="cyc")
            cal["phys"].append((rows, np.abs(yj - p_j)))
            cal["hybrid"].append((rows, np.abs(yj - h_j)))
            cal["cyc"].append((rows, np.abs(yj - c_j)))
        pooled = {k: (pd.concat([f for f, _ in v], ignore_index=True), np.concatenate([r for _, r in v]))
                  for k, v in cal.items()}
        centres = {"phys_bayes": phys, "phys_loco": phys, "hybrid_loco": hybrid, "phys_cyc_loco": phys_cyc}
        source = {"phys_loco": "phys", "hybrid_loco": "hybrid", "phys_cyc_loco": "cyc"}
        out = {"case": case, "n": int(len(test)), "rmse_gbm_d27": run_d26.rmse(p_gbm - y),
               "rmse": {arm: run_d26.rmse(centres[arm] - y) for arm in ARMS},
               "population": {k: v for k, v in pop.items() if k != "var"},
               "population_cyc": {k: v for k, v in pop_cyc.items() if k != "var"}, "rules": {}}
        for arm in ARMS:
            out["rules"][arm] = {}
            for lv in LEVELS:
                if arm == "phys_bayes":
                    qv = stats.norm.ppf((1 + lv) / 2) * np.sqrt(pop["var"])
                else:
                    frame, resid = pooled[source[arm]]
                    q = run_d27.halfwidths(resid, run_d27.keys_of(frame, False), lv)[0]
                    qv = run_d28.static_q(test, q)
                out["rules"][arm][f"{round(lv * 100)}"] = run_d28.block(test, centres[arm], qv, lv)
        return out


def reproduction(folds: list[dict], d27: dict) -> dict:
    ref = {f["case"]: f["rmse_gbm"] for f in d27["per_fold"]}
    diffs = {f["case"]: abs(f["rmse_gbm_d27"] - ref[f["case"]]) for f in folds}
    worst = max(diffs.values())
    return {"abs_diff_per_fold": diffs, "max_abs_diff": worst, "passed": bool(worst <= TOL)}


def questions(folds: list[dict], summary: dict, d27: dict, d28: dict) -> dict:
    cases = [f["case"] for f in folds]
    r27 = {f["case"]: f for f in d27["per_fold"]}
    r28 = {f["case"]: f for f in d28["per_fold"]}
    hy = summary["hybrid_loco"]["90"]
    c7 = next(f for f in folds if f["case"] == PRIMARY_CASE)
    return {
        "q1_case7_coverage": hy["coverage_per_fold"].get(PRIMARY_CASE), "q1_threshold": Q1_MIN,
        "q2_rmse_hybrid_vs_d27_gbm": run_d28.paired_test([f["rmse"]["hybrid_loco"] for f in folds],
                                                         [r27[c]["rmse_gbm"] for c in cases], cases),
        "q3_is90_hybrid_vs_d28_refit_track": run_d28.paired_test(
            [f["rules"]["hybrid_loco"]["90"]["all"]["interval_score"] for f in folds],
            [r28[c]["rules"]["refit_track"]["90"]["all"]["interval_score"] for c in cases], cases),
        "q4_case7_rmse": {"phys_efc": c7["rmse"]["phys_loco"], "phys_cycles": c7["rmse"]["phys_cyc_loco"],
                          "d27_gbm": r27[PRIMARY_CASE]["rmse_gbm"], "d28_refit": r28[PRIMARY_CASE]["rmse"]["refit"]},
        "q5_phys_bayes_coverage": {"pooled": summary["phys_bayes"]["90"]["coverage_pooled"],
                                   "per_fold": summary["phys_bayes"]["90"]["coverage_per_fold"]},
        "comparators_is90": {"d27_gbm_loco": d27["summary"]["gbm_loco"]["90"]["interval_score"],
                             "d28_refit_track": d28["summary"]["refit_track"]["90"]["interval_score"]}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp", choices=tuple(run_d26.HORIZONS), required=True)
    ap.add_argument("--d27", type=Path, help="default results/forecasting_d27_<exp>.json")
    ap.add_argument("--d28", type=Path, help="default results/forecasting_d28_<exp>.json")
    ap.add_argument("--out", type=Path, help="default results/forecasting_d29_<exp>.json")
    ap.add_argument("--allow-dirty", action="store_true", help="code check only; recorded in the results file")
    ap.add_argument("--allow-unpinned", action="store_true", help="code check only; recorded in the results file")
    a = ap.parse_args()
    problems = run_d27.preflight(a.allow_dirty, a.allow_unpinned)
    if problems:
        print("PRE-FLIGHT:\n" + "\n".join(problems), flush=True)
        return 2
    d27 = json.loads((a.d27 or ROOT / "results" / f"forecasting_d27_{a.exp}.json").read_text())
    d28 = json.loads((a.d28 or ROOT / "results" / f"forecasting_d28_{a.exp}.json").read_text())
    run = D29(a.exp)
    if sorted(d27["folds"]) != run.cases or sorted(d28["folds"]) != run.cases:
        raise SystemExit("D27/D28 folds differ from the S6 cases")
    print(f"D29 {a.exp}: {len(run.direct)} S6 samples, {len(run.chk)} checks, {len(run.cases)} folds", flush=True)
    folds = []
    for c in run.cases:
        f = run.fold(c)
        folds.append(f)
        print(f"  {c}: z {f['population']['z']}, RMSE gbm {f['rmse_gbm_d27']:.3f} phys {f['rmse']['phys_loco']:.3f} "
              f"hybrid {f['rmse']['hybrid_loco']:.3f} cyc {f['rmse']['phys_cyc_loco']:.3f} | cov90 hybrid "
              f"{f['rules']['hybrid_loco']['90']['all']['coverage']:.2f} bayes "
              f"{f['rules']['phys_bayes']['90']['all']['coverage']:.2f}", flush=True)
    rep = reproduction(folds, d27)
    print(f"reproduction check: max abs diff {rep['max_abs_diff']:.3g}, passed {rep['passed']}", flush=True)
    if not rep["passed"]:
        print("STOP: D27 not reproduced; no results file written (D29).", json.dumps(rep, indent=1), flush=True)
        return 3
    summary = {arm: {lv: run_d27.pooled_metrics(folds, arm, lv) for lv in ("90", "80")} for arm in ARMS}
    q = questions(folds, summary, d27, d28)
    out = a.out or ROOT / "results" / f"forecasting_d29_{a.exp}.json"
    X.write_results(out, {
        "decision": "D29", "experiment": a.exp, "horizons": list(run_d26.HORIZONS[a.exp]), "n_samples": len(run.direct),
        "features": run.features, "folds": run.cases, "z_grid": list(Z_GRID),
        "overrides": {"allow_dirty": a.allow_dirty, "allow_unpinned": a.allow_unpinned},
        "reproduction_check": rep, "questions": q, "summary": summary, "per_fold": folds}, ROOT)
    for arm in ARMS:
        s = summary[arm]["90"]
        print(f"{arm:14s} cov90 {s['coverage_pooled']:.3f} (min {s['coverage_min']:.2f} {s['coverage_min_case']}) "
              f"IS90 {s['interval_score']:.3f} width {s['width']:.3f} RMSE "
              f"{np.mean([f['rmse'][arm] for f in folds]):.3f}", flush=True)
    print(f"comparators IS90: D27 {q['comparators_is90']['d27_gbm_loco']:.3f}, D28 refit+track "
          f"{q['comparators_is90']['d28_refit_track']:.3f}", flush=True)
    print(f"Q1 case7 cov90 {q['q1_case7_coverage']:.3f}; Q2 RMSE {q['q2_rmse_hybrid_vs_d27_gbm']['mean_diff']:+.3f} pp "
          f"p {q['q2_rmse_hybrid_vs_d27_gbm']['p_value']:.3f} {q['q2_rmse_hybrid_vs_d27_gbm']['verdict']}; Q3 IS90 "
          f"{q['q3_is90_hybrid_vs_d28_refit_track']['mean_diff']:+.3f} pp p "
          f"{q['q3_is90_hybrid_vs_d28_refit_track']['p_value']:.3f} {q['q3_is90_hybrid_vs_d28_refit_track']['verdict']}; "
          f"Q4 case7 RMSE {q['q4_case7_rmse']}", flush=True)
    print(f"written {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
