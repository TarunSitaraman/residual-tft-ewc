"""Decision D32: gradient boosting under the continual-learning constraint on unseen usage conditions (P2).

The D26 continual shrunk residual TFT + EWC (per-run P2 errors read from results/forecasting_d26_<exp>.json, not
refitted) against gradient boosting that, like the TFT, sees the six training cases one at a time without keeping
earlier cases' rows: GBM-WARM (warm-started, 200 more iterations per case) and GBM-LAST (latest case only). GBM-ALL
(all six cases at once, the D26 comparator) is refitted as the reproduction check and kept as the full-storage
reference. Tests and verdicts as registered in docs/DECISIONS.md, D32.

    python scripts/run_d32.py --exp e1
    python scripts/run_d32.py --exp e2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_d25  # noqa: E402
import run_d26  # noqa: E402
import run_d27  # noqa: E402
from src import evaluation as ev  # noqa: E402
from src import experiments as X  # noqa: E402
from src import forecasting as FC  # noqa: E402

SHR, RAW = run_d26.SHR, run_d26.RAW
WARM, LAST, ALL = "gbm_warm", "gbm_last", "gbm_all"
STAGE_ITER = 200                                    # D18 max_iter; added per case for GBM-WARM
METRICS = {"p2_error": "rmse", "p2_error_hplus": "rmse_hplus"}
PRIMARY = [(SHR, WARM, "p2_error"), (SHR, LAST, "p2_error")]
SECONDARY = [(SHR, WARM, "p2_error_hplus"), (SHR, LAST, "p2_error_hplus"), (RAW, WARM, "p2_error"),
             (RAW, LAST, "p2_error")]
TOL_GBM, TOL_RUNS = 1e-9, 1e-12


def fit_frame(r: run_d26.Runner, case: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """The D26 baseline rows for one fold: all samples of the other six cases, the held-out samples, observed inputs."""
    task = FC.leave_one_out_task(r.direct, case)
    fit = pd.concat([task.train, task.val], ignore_index=True)
    return fit, task.test, [f for f in r.features if fit[f].notna().any()]


def score(r: run_d26.Runner, case: str, model, test: pd.DataFrame, cols: list[str]) -> dict:
    frame, direct, hplus = r.held(case)
    return run_d26.scores(frame, direct, model.predict(test[cols]) - frame["lin_forecast"].to_numpy(float), hplus)


def gbm_warm(fit: pd.DataFrame, order6, cols: list[str]):
    """One case at a time; earlier cases' rows are never used again (bins come from the first case)."""
    m = FC._gbm(0)
    for i, c in enumerate(order6):
        rows = fit[fit["Battery_ID"] == c]
        if i:
            m.set_params(warm_start=True, max_iter=STAGE_ITER * (i + 1))
        m.fit(rows[cols], rows[FC.TARGET].to_numpy(float))
    return m


def gbm_on(rows: pd.DataFrame, cols: list[str]):
    return FC._gbm(0).fit(rows[cols], rows[FC.TARGET].to_numpy(float))


def gbm_folds(r: run_d26.Runner) -> dict:
    """{case: {"all": scores, "warm": {order_id: scores}, "last": {order_id: scores}}}."""
    out = {}
    for case in r.cases:
        fit, test, cols = fit_frame(r, case)
        f = {"all": score(r, case, gbm_on(fit, cols), test, cols), "warm": {}, "last": {}}
        for oid, order in enumerate(r.orders):
            order6 = r.fold_order(order, case)
            f["warm"][oid] = score(r, case, gbm_warm(fit, order6, cols), test, cols)
            f["last"][oid] = score(r, case, gbm_on(fit[fit["Battery_ID"] == order6[-1]], cols), test, cols)
        out[case] = f
        print(f"{case}: all {f['all']['rmse']:.3f}, warm {np.mean([s['rmse'] for s in f['warm'].values()]):.3f}, "
              f"last {np.mean([s['rmse'] for s in f['last'].values()]):.3f}", flush=True)
    return out


def gbm_rows(r: run_d26.Runner, folds: dict) -> pd.DataFrame:
    """Run-level GBM values (mean over the seven folds), one per (arm, order); GBM-ALL has one value."""
    rows = []
    for arm, key in ((WARM, "warm"), (LAST, "last")):
        for oid in range(len(r.orders)):
            rows.append({"method": arm, "order_id": oid,
                         **{m: float(np.mean([folds[c][key][oid][k] for c in r.cases])) for m, k in METRICS.items()},
                         "per_fold": {c: folds[c][key][oid]["rmse"] for c in r.cases}})
    rows.append({"method": ALL, "order_id": None,
                 **{m: float(np.mean([folds[c]["all"][k] for c in r.cases])) for m, k in METRICS.items()},
                 "per_fold": {c: folds[c]["all"]["rmse"] for c in r.cases}})
    return pd.DataFrame(rows)


def reproduction(r: run_d26.Runner, d26: dict, folds: dict, runs: pd.DataFrame) -> dict:
    gbm = next(b for b in d26["baselines"] if b["method"] == run_d26.GBM)["per_fold"]
    d_gbm = max(abs(folds[c]["all"]["rmse"] - gbm[c]) for c in r.cases)
    summ = {(s["set"], s["method"]): s["p2_error_mean"] for s in d26["summary"]}
    d_runs = max(abs(g["p2_error"].mean() - summ[k]) for k, g in runs.groupby(["set", "method"]))
    orders_ok = d26["orders"] == [list(o) for o in r.orders]
    return {"gbm_all_max_abs_diff": d_gbm, "d26_runs_max_abs_diff": float(d_runs), "orders_equal": orders_ok,
            "tolerances": {"gbm": TOL_GBM, "runs": TOL_RUNS}, "d26_commit": d26["environment"].get("git_commit"),
            "passed": bool(d_gbm <= TOL_GBM and d_runs <= TOL_RUNS and orders_ok)}


def d26_runs(d26: dict) -> pd.DataFrame:
    runs = pd.DataFrame(d26["runs"])
    return runs[runs["method"].isin([SHR, RAW])][["set", "order_id", "seed", "method", *METRICS]]


def paired(runs: pd.DataFrame, gbm: pd.DataFrame, a: str, b: str, metric: str, set_name: str) -> dict:
    """TFT run (order o, seed s) minus the GBM arm's order-o value."""
    ra = runs[(runs["set"] == set_name) & (runs["method"] == a)].sort_values(["order_id", "seed"])
    vb = gbm[gbm["method"] == b].set_index("order_id")[metric].reindex(ra["order_id"]).to_numpy(float)
    return {"set": set_name, "a": a, "b": b, "metric": metric,
            **ev.paired_comparison(ra[metric].to_numpy(float), vb)}


def tests(runs: pd.DataFrame, gbm: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    rows = []
    for fam, comps in {"primary": PRIMARY, "secondary": SECONDARY}.items():
        for set_name in run_d26.SETS:
            got = [paired(runs, gbm, *c, set_name) for c in comps]
            holm = ev.holm_correction({(x["a"], x["b"], x["metric"]): x["p_value"] for x in got})
            rows += [{**x, "family": fam, "p_holm": holm[(x["a"], x["b"], x["metric"])]} for x in got]
    t = pd.DataFrame(rows)
    verdicts = []
    for (fam, a, b, m), g in t.groupby(["family", "a", "b", "metric"], sort=False):
        mn, rp = g[g["set"] == "main"].iloc[0], g[g["set"] == "replicate"].iloc[0]
        verdicts.append({"family": fam, "a": a, "b": b, "metric": m, "diff_main": float(mn["mean_diff"]),
                         "diff_replicate": float(rp["mean_diff"]), "p_holm_main": float(mn["p_holm"]),
                         "p_holm_replicate": float(rp["p_holm"]), "verdict": run_d25.verdict(mn, rp)})
    return rows, verdicts


def summary(runs: pd.DataFrame, gbm: pd.DataFrame) -> dict:
    out = {f"{s}_{m}": {k: float(g[k].mean()) for k in METRICS}
           for (s, m), g in runs.groupby(["set", "method"])}
    for arm, g in gbm.groupby("method"):
        out[arm] = {k: float(g[k].mean()) for k in METRICS}
        out[arm]["per_fold"] = {c: float(np.mean([pf[c] for pf in g["per_fold"]])) for c in g["per_fold"].iloc[0]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp", choices=tuple(run_d26.HORIZONS), required=True)
    ap.add_argument("--profile", default="paper", help="paper (real runs) or smoke (synthetic tests)")
    ap.add_argument("--d26", type=Path, help="D26 results file (default results/forecasting_d26_<exp>.json)")
    ap.add_argument("--out", type=Path, help="output (default results/forecasting_d32_<exp>.json)")
    ap.add_argument("--allow-dirty", action="store_true", help="code check only; recorded in the results file")
    ap.add_argument("--allow-unpinned", action="store_true", help="code check only; recorded in the results file")
    a = ap.parse_args()
    problems = run_d27.preflight(a.allow_dirty, a.allow_unpinned)
    if problems:
        print("PRE-FLIGHT:\n" + "\n".join(problems), flush=True)
        return 2
    d26 = json.loads((a.d26 or ROOT / "results" / f"forecasting_d26_{a.exp}.json").read_text())
    r = run_d26.Runner(a.exp, a.profile)
    print(f"D32 {a.exp}: {len(r.direct)} S6 samples, {len(r.cases)} folds, {len(r.orders)} orders", flush=True)
    folds = gbm_folds(r)
    runs = d26_runs(d26)
    rep = reproduction(r, d26, folds, runs)
    print(f"reproduction: {json.dumps(rep)}", flush=True)
    if not rep["passed"]:
        print("STOP: reproduction check failed; no results file written (D32).", flush=True)
        return 3
    gbm = gbm_rows(r, folds)
    rows, verdicts = tests(runs, gbm)
    out = a.out or ROOT / "results" / f"forecasting_d32_{a.exp}.json"
    X.write_results(out, {
        "decision": "D32", "experiment": a.exp, "profile": a.profile, "horizons": list(run_d26.HORIZONS[a.exp]),
        "n_samples": len(r.direct), "features": r.features, "folds": r.cases, "orders": [list(o) for o in r.orders],
        "stage_iterations": STAGE_ITER, "overrides": {"allow_dirty": a.allow_dirty, "allow_unpinned": a.allow_unpinned},
        "reproduction_check": rep, "summary": summary(runs, gbm), "gbm_runs": gbm.to_dict("records"),
        "gbm_per_fold": {c: {"all": f["all"], "warm": {str(k): v for k, v in f["warm"].items()},
                             "last": {str(k): v for k, v in f["last"].items()}} for c, f in folds.items()},
        "tests": rows, "verdicts": verdicts}, ROOT)
    for v in verdicts:
        print(f"{v['family']:9s} {v['a']} - {v['b']} [{v['metric']}]: {v['diff_main']:+.3f} / {v['diff_replicate']:+.3f}"
              f" (p_Holm {v['p_holm_main']:.2g} / {v['p_holm_replicate']:.2g}): {v['verdict']}", flush=True)
    print(f"written {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
