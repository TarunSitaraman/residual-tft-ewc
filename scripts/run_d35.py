"""Decision D35: gradient boosting under the continual-learning constraint on the S6 stream (P1) (post hoc).

Fills the P1 cell that D32 left empty: gradient boosting refitted on the latest task only (GBM-LAST; earlier tasks'
rows are not kept), evaluated like every stream method: after each step on every task's test split, giving the
accuracy matrix R, the final average error, forgetting and plasticity (``src.evaluation.cl_metrics``). Two arms:
direct target (as every gradient-boosting comparator since D18) and residual target (as the proposed model, D23).

Same samples, splits, D21 shared inputs, ten evaluation orders and gradient-boosting settings as D23/D25. The
proposed arm is not refitted: the D25 shrunk residual TFT + EWC runs are read from results/forecasting_d25_<exp>.json.
Tests: the D32 pairing (each (order, seed) run paired with the order's GBM-LAST value), Wilcoxon over the 50 runs
per seed set, Holm over the two arms within each seed set, D21 replication rule (``run_d25.verdict``).

Reproduction checks (stop conditions): GBM fitted on all tasks' training rows equals the committed D23
``gbm_shared`` stream baseline to 1e-9 pp; the orders equal D25's.

    python scripts/run_d35.py --exp e1
    python scripts/run_d35.py --exp e2
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

SHR = "tft_residual_ewc_shrunk"
ARMS = {"gbm_last_direct": "direct", "gbm_last_residual": "residual"}
TOL = 1e-9
GBM_COLS = {"rate": "fade_rate_pp_per_cycle", "span": "cycles_to_target",
            "schedule": ["charge_c_rate", "discharge_c_rate", "dod", "v_charge_max", "cv_charge", "cycles_to_target",
                         "days_to_target"]}


def frame(r: run_d26.Runner, arm: str) -> pd.DataFrame:
    """Direct target (Delta) or residual target (Delta - lin_forecast); errors are identical on both scales."""
    return r.direct if arm == "direct" else r.s6


def observed(r: run_d26.Runner, tasks) -> list[str]:
    train = pd.concat([t.train for t in tasks], ignore_index=True)
    return [f for f in r.features if train[f].notna().any()]


def gbm_last_stream(r: run_d26.Runner, order, arm: str) -> dict:
    """GBM refitted on the latest task's training rows only; R[i, j] = test RMSE on task j after step i."""
    tasks = FC.forecast_stream(frame(r, arm), order)
    cols = observed(r, tasks)
    errors = {}
    for i, task in enumerate(tasks):
        m = FC._gbm(0).fit(task.train[cols], task.train[FC.TARGET])
        for t in tasks:
            errors[(i, t.name)] = ev.task_error(t.test[FC.TARGET], m.predict(t.test[cols]), "RMSE")
    R = ev.accuracy_matrix(errors, [t.name for t in tasks])
    m = ev.cl_metrics(R)
    return {"final_avg_error": m["final_avg_error"], "forgetting": m["forgetting"], "plasticity": m["plasticity"],
            "backward_transfer": m["backward_transfer"], "R": R.to_numpy().tolist()}


def gbm_all_stream(r: run_d26.Runner) -> float:
    """The D23 stream baseline: GBM fitted on every task's training rows (reproduction check)."""
    tasks = FC.forecast_stream(r.direct, r.orders[0])
    train = pd.concat([t.train for t in tasks], ignore_index=True)
    tests = {t.name: t.test for t in tasks}
    cols = observed(r, tasks)
    preds = FC.baseline_predictions(train, tests, {"F0": cols, "F1": cols}, seed=0, columns=GBM_COLS)["gbm_F1"]
    return float(np.mean([ev.task_error(tests[k][FC.TARGET], v, "RMSE") for k, v in preds.items()]))


def tests(runs: pd.DataFrame, gbm: dict[str, list[float]]) -> tuple[list[dict], list[dict]]:
    """Shrunk run minus the order's GBM-LAST value, per seed set; Holm over the two arms; D21 rule."""
    rows = []
    for s in ("main", "replicate"):
        sub = runs[runs["set"] == s].sort_values(["order_id", "seed"])
        res = {}
        for name, vals in gbm.items():
            b = np.array([vals[o] for o in sub["order_id"]], float)
            res[name] = ev.paired_comparison(sub["final_avg_error"].to_numpy(float), b)
        holm = ev.holm_correction({k: v["p_value"] for k, v in res.items()})
        for name, c in res.items():
            rows.append({"set": s, "a": SHR, "b": name, "metric": "final_avg_error", **c, "p_holm": holm[name]})
    verdicts = []
    for name in gbm:
        m, rp = (next(x for x in rows if x["set"] == s and x["b"] == name) for s in ("main", "replicate"))
        verdicts.append({"family": "primary", "a": SHR, "b": name, "metric": "final_avg_error",
                         "diff_main": m["mean_diff"], "diff_replicate": rp["mean_diff"],
                         "p_holm_main": m["p_holm"], "p_holm_replicate": rp["p_holm"],
                         "verdict": run_d25.verdict(m, rp)})
    return rows, verdicts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp", choices=tuple(run_d26.HORIZONS), required=True)
    ap.add_argument("--profile", default="paper", help="paper (real runs) or smoke (synthetic tests)")
    ap.add_argument("--out", type=Path, help="output (default results/forecasting_d35_<exp>.json)")
    ap.add_argument("--allow-dirty", action="store_true", help="code check only; recorded in the results file")
    ap.add_argument("--allow-unpinned", action="store_true", help="code check only; recorded in the results file")
    a = ap.parse_args()
    problems = run_d27.preflight(a.allow_dirty, a.allow_unpinned)
    if problems:
        print("PRE-FLIGHT:\n" + "\n".join(problems), flush=True)
        return 2
    r = run_d26.Runner(a.exp, a.profile)
    d23 = json.loads((ROOT / "results" / f"forecasting_d23_{a.exp}.json").read_text())
    d25 = json.loads((ROOT / "results" / f"forecasting_d25_{a.exp}.json").read_text())
    print(f"D35 {a.exp}: {len(r.direct)} S6 samples, {len(r.cases)} tasks, {len(r.orders)} orders", flush=True)

    committed = next(b["final_avg_error"] for b in d23["baselines"] if b["method"] == "gbm_shared")
    reproduced = gbm_all_stream(r)
    rep = {"gbm_all_reproduced": reproduced, "gbm_all_committed_d23": committed,
           "abs_diff": abs(reproduced - committed), "orders_equal_d25": [list(o) for o in r.orders] == d25["orders"]}
    rep["passed"] = rep["abs_diff"] < TOL and rep["orders_equal_d25"]
    print(f"reproduction: {json.dumps(rep)}", flush=True)
    if not rep["passed"] and a.profile == "paper":
        print("STOP: reproduction check failed; no results file written (D35).", flush=True)
        return 3

    per_order = {}
    for name, arm in ARMS.items():
        per_order[name] = []
        for k, o in enumerate(r.orders):
            per_order[name].append(gbm_last_stream(r, o, arm))
            print(f"{name}: order {k + 1}/{len(r.orders)}", flush=True)
    runs = pd.DataFrame([x for x in d25["runs"] if x["method"] == SHR])
    rows, verdicts = tests(runs, {n: [x["final_avg_error"] for x in v] for n, v in per_order.items()})
    summary = {n: {k: float(np.mean([x[k] for x in v])) for k in ("final_avg_error", "forgetting", "plasticity",
                                                                  "backward_transfer")}
               for n, v in per_order.items()}
    summary["proposed"] = {s: float(runs[runs["set"] == s]["final_avg_error"].mean()) for s in ("main", "replicate")}
    out = a.out or ROOT / "results" / f"forecasting_d35_{a.exp}.json"
    X.write_results(out, {
        "decision": "D35", "experiment": a.exp, "profile": a.profile, "horizons": list(run_d26.HORIZONS[a.exp]),
        "n_samples": len(r.direct), "features": r.features, "orders": [list(o) for o in r.orders],
        "overrides": {"allow_dirty": a.allow_dirty, "allow_unpinned": a.allow_unpinned},
        "reproduction_check": rep, "summary": summary,
        "gbm_per_order": {n: v for n, v in per_order.items()}, "tests": rows, "verdicts": verdicts}, ROOT)
    for v in verdicts:
        print(f"{v['a']} - {v['b']}: {v['diff_main']:+.3f} / {v['diff_replicate']:+.3f}"
              f" (p_Holm {v['p_holm_main']:.2g} / {v['p_holm_replicate']:.2g}): {v['verdict']}", flush=True)
    print(f"written {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
