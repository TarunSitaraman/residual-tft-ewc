"""Run decision D27: leave-one-condition-out (LOCO) conformal intervals for unseen usage conditions (P2).

Setting (D27, as D26): S6 samples with the D21 shared inputs, E1 = horizons 1-3, E2 = 1-6, seven folds, one per
held-out test case; every sample of the held-out case is a test sample and is never used for fitting or calibration.

- **GBM-LOCO:** gradient boosting exactly as the D26 comparator, fitted on all samples of the six training cases.
  Its per-horizon split-conformal half-widths come from the six training cases, each predicted by a gradient boosting
  fitted on the other five (same settings).
- **LIN-all:** linear extrapolation, half-widths from every sample of the six training cases.
- **LIN-val:** linear extrapolation, half-widths from the validation rows of the six training cases (the D26 rule).
- **Mondrian (secondary):** GBM-LOCO and LIN-all with half-widths per (horizon, origin stratum H+/H-).

Before anything is written, the six-case gradient boosting must reproduce the D26 per-fold RMSE and LIN-val the D26
extrapolation interval metrics (stop conditions, D27). Half-widths: ``src/calibration.py`` (D25, unchanged).

Usage (from the repository root)::

    python scripts/run_d27.py --exp e1
    python scripts/run_d27.py --exp e2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_d25  # noqa: E402
import run_d25_night  # noqa: E402
import run_d26  # noqa: E402
from src import data_pipeline as dp  # noqa: E402
from src import evaluation as ev  # noqa: E402
from src import experiments as X  # noqa: E402
from src import forecasting as FC  # noqa: E402
from src import p3  # noqa: E402
from src.calibration import LEVELS, conformal_halfwidth, interval_metrics  # noqa: E402

COLS = {"rate": "fade_rate_pp_per_cycle", "span": "cycles_to_target",
        "schedule": ["charge_c_rate", "discharge_c_rate", "dod", "v_charge_max", "cv_charge", "cycles_to_target",
                     "days_to_target"]}                    # the D26 comparator's columns
RULES = ("gbm_loco", "lin_all", "lin_val", "gbm_loco_mondrian", "lin_all_mondrian")
ADEQUATE = 0.85                                            # pooled 90 % coverage, both experiments (D26 threshold)
MIN_DIFF = 0.05                                            # pp, Q2
TOL = 1e-9                                                 # reproduction checks


def gbm_predict(fit: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    """The D26 comparator's gradient boosting (``gbm_F1`` on the observed shared inputs, seed 0)."""
    observed = [f for f in features if fit[f].notna().any()]
    return FC.baseline_predictions(fit, {"t": test}, {"F0": observed, "F1": observed}, seed=0, columns=COLS)["gbm_F1"]["t"]


def p2_fit_rows(direct: pd.DataFrame, held: str) -> pd.DataFrame:
    """All samples of the cases other than ``held``, in the row order of the D26 comparator fit."""
    task = FC.leave_one_out_task(direct, held)
    return pd.concat([task.train, task.val], ignore_index=True)


def halfwidths(resid: np.ndarray, keys: np.ndarray, level: float) -> tuple[dict, list]:
    """Split-conformal half-width per key (horizon, or (horizon, stratum)) and the keys flagged undersized."""
    out, small = {}, []
    for k in sorted(set(map(tuple, keys)) if keys.ndim > 1 else set(keys.tolist())):
        m = np.all(keys == k, axis=1) if keys.ndim > 1 else keys == k
        out[k], under = conformal_halfwidth(resid[m], level)
        if under:
            small.append(str(k))
    return out, small


def keys_of(frame: pd.DataFrame, mondrian: bool) -> np.ndarray:
    h = frame["horizon"].astype(int).to_numpy()
    return np.column_stack([h, frame["fade_rate_pp_per_cycle"].notna().to_numpy(int)]) if mondrian else h


def interval_block(frame: pd.DataFrame, centre: np.ndarray, q: dict, mondrian: bool, level: float,
                   hplus: np.ndarray) -> dict:
    """Interval metrics on all origins, H+ and H-, counts for pooling, and the reuse decision support."""
    y = frame[FC.TARGET].to_numpy(float)
    k = keys_of(frame, mondrian)
    qv = np.array([q[tuple(r)] if mondrian else q[r] for r in (map(tuple, k) if mondrian else k)], float)
    out = {"all": interval_metrics(y, centre, qv, level), "n": int(len(y)),
           "covered": int(np.sum(np.abs(y - centre) <= qv))}
    for name, m in (("hplus", hplus), ("hminus", ~hplus)):
        out[name] = {**interval_metrics(y[m], centre[m], qv[m], level), "n": int(m.sum()),
                     "covered": int(np.sum(np.abs(y[m] - centre[m]) <= qv[m]))} if m.any() else {}
    soh0 = frame["soh_origin"].to_numpy(float)
    lo, hi = ev.reuse_class(soh0 + centre - qv), ev.reuse_class(soh0 + centre + qv)
    true_c, pred_c = ev.reuse_class(frame["soh_target"].to_numpy(float)), ev.reuse_class(soh0 + centre)
    decided = lo == hi
    out["decided_share"] = float(decided.mean())
    out["decided_agreement"] = float(np.mean(true_c[decided] == pred_c[decided])) if decided.any() else float("nan")
    return out


class D27:
    def __init__(self, exp: str) -> None:
        self.exp = exp                                           # D26's frame, features and cases
        wc = dp.read_dataset(run_d25.data_dir() / "wheeler_checks.csv")
        self.direct = p3.s6_samples(wc, run_d26.HORIZONS[exp])
        self.features = list(p3.SHARED_FEATURES)
        self.cases = sorted(self.direct["Battery_ID"].unique())

    def fold(self, case: str) -> dict:
        d, feats = self.direct, self.features
        test = d[d["Battery_ID"] == case].reset_index(drop=True)
        hplus = test["fade_rate_pp_per_cycle"].notna().to_numpy()
        y = test[FC.TARGET].to_numpy(float)
        p_gbm = gbm_predict(p2_fit_rows(d, case), test, feats)
        lin = test["lin_forecast"].to_numpy(float)
        others = d[d["Battery_ID"] != case].reset_index(drop=True)
        # Calibration residuals (Delta scale), one block per training case.
        cal = {"gbm": [], "lin_all": [], "lin_val": []}
        for j in sorted(others["Battery_ID"].unique()):
            rows = others[others["Battery_ID"] == j].reset_index(drop=True)
            r_gbm = rows[FC.TARGET].to_numpy(float) - gbm_predict(p2_fit_rows(others, j), rows, feats)
            r_lin = rows[FC.TARGET].to_numpy(float) - rows["lin_forecast"].to_numpy(float)
            val = (FC.assign_split(rows)["split"] == "val").to_numpy()
            cal["gbm"].append((rows, r_gbm))
            cal["lin_all"].append((rows, r_lin))
            cal["lin_val"].append((rows[val].reset_index(drop=True), r_lin[val]))
        pooled = {k: (pd.concat([f for f, _ in v], ignore_index=True), np.concatenate([r for _, r in v]))
                  for k, v in cal.items()}
        spec = {"gbm_loco": ("gbm", p_gbm, False), "lin_all": ("lin_all", lin, False), "lin_val": ("lin_val", lin, False),
                "gbm_loco_mondrian": ("gbm", p_gbm, True), "lin_all_mondrian": ("lin_all", lin, True)}
        out = {"case": case, "n": int(len(test)), "rmse_gbm": run_d26.rmse(p_gbm - y), "rmse_lin": run_d26.rmse(lin - y),
               "calibration_rows": {k: int(len(v[0])) for k, v in pooled.items()}, "rules": {}}
        for rule, (src, centre, mondrian) in spec.items():
            frame, resid = pooled[src]
            out["rules"][rule] = {}
            for lv in LEVELS:
                q, small = halfwidths(np.abs(resid), keys_of(frame, mondrian), lv)
                block = interval_block(test, centre, q, mondrian, lv, hplus)
                block["halfwidths"] = {str(k): float(v) for k, v in q.items()}
                block["undersized"] = small
                out["rules"][rule][f"{round(lv * 100)}"] = block
        return out


def pooled_metrics(folds: list[dict], rule: str, lv: str) -> dict:
    b = [f["rules"][rule][lv] for f in folds]
    res = {"coverage_pooled": sum(x["covered"] for x in b) / sum(x["n"] for x in b),
           "interval_score": float(np.mean([x["all"]["interval_score"] for x in b])),
           "width": float(np.mean([x["all"]["width"] for x in b])),
           "coverage_per_fold": {f["case"]: x["all"]["coverage"] for f, x in zip(folds, b)},
           "interval_score_per_fold": {f["case"]: x["all"]["interval_score"] for f, x in zip(folds, b)},
           "decided_share": float(np.mean([x["decided_share"] for x in b])),
           "decided_agreement": float(np.nanmean([x["decided_agreement"] for x in b]))}
    for s in ("hplus", "hminus"):
        n = sum(x[s].get("n", 0) for x in b)
        res[f"coverage_pooled_{s}"] = sum(x[s].get("covered", 0) for x in b) / n if n else float("nan")
    cov = res["coverage_per_fold"]
    res["coverage_min"], res["coverage_min_case"] = min(cov.values()), min(cov, key=cov.get)
    return res


def reproduction(folds: list[dict], d26: dict) -> dict:
    """D27 stop conditions: D26 gradient-boosting per-fold RMSE and D26 extrapolation interval metrics (LIN-val)."""
    gbm = {f["case"]: abs(f["rmse_gbm"] - d26["baselines_by_fold"][f["case"]]["gbm_shared"]["rmse"]) for f in folds}
    linc = next(r for r in d26["runs"] if r["method"] == run_d26.LINC)
    lin = {}
    for lv in ("90", "80"):
        m = pooled_metrics(folds, "lin_val", lv)
        lin[f"is{lv}"] = abs(m["interval_score"] - linc[f"is{lv}"])
        lin[f"cov{lv}"] = abs(m["coverage_pooled"] - linc[f"cov{lv}"])
    worst = max([*gbm.values(), *lin.values()])
    return {"gbm_rmse_abs_diff": gbm, "lin_val_abs_diff": lin, "max_abs_diff": worst, "passed": bool(worst <= TOL)}


def q2_test(folds: list[dict]) -> dict:
    a = np.array([f["rules"]["gbm_loco"]["90"]["all"]["interval_score"] for f in folds])
    b = np.array([f["rules"]["lin_all"]["90"]["all"]["interval_score"] for f in folds])
    d = a - b
    p = float(stats.wilcoxon(d, alternative="two-sided", method="exact").pvalue)
    return {"mean_diff": float(d.mean()), "per_fold": {f["case"]: float(x) for f, x in zip(folds, d)},
            "n_gbm_better": int((d < 0).sum()), "p_value": p,
            "direction": ("gbm_loco better" if d.mean() <= -MIN_DIFF else "lin_all better" if d.mean() >= MIN_DIFF
                          else "difference below 0.05 pp") if p < 0.05 else "not significant"}


def preflight(allow_dirty: bool, allow_unpinned: bool) -> list[str]:
    problems = []
    st = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True,
                        text=True)
    if st.stdout.strip() and not allow_dirty:
        problems.append("tracked files differ from the commit; commit or stash first:\n" + st.stdout)
    mism = run_d25_night.version_mismatches()
    if mism and not allow_unpinned:
        problems.append("library versions differ from requirements.txt: " + "; ".join(mism))
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp", choices=tuple(run_d26.HORIZONS), required=True)
    ap.add_argument("--d26", type=Path, help="D26 results file (default results/forecasting_d26_<exp>.json)")
    ap.add_argument("--out", type=Path, help="output (default results/forecasting_d27_<exp>.json)")
    ap.add_argument("--allow-dirty", action="store_true", help="code check only; recorded in the results file")
    ap.add_argument("--allow-unpinned", action="store_true", help="code check only; recorded in the results file")
    a = ap.parse_args()
    problems = preflight(a.allow_dirty, a.allow_unpinned)
    if problems:
        print("PRE-FLIGHT:\n" + "\n".join(problems), flush=True)
        return 2
    d26 = json.loads((a.d26 or ROOT / "results" / f"forecasting_d26_{a.exp}.json").read_text())
    run = D27(a.exp)
    if sorted(d26["folds_in_tests"]) != run.cases:
        raise SystemExit(f"D26 folds {d26['folds_in_tests']} differ from the S6 cases {run.cases}")
    print(f"D27 {a.exp}: {len(run.direct)} S6 samples, {len(run.cases)} folds", flush=True)
    folds = [run.fold(c) for c in run.cases]
    rep = reproduction(folds, d26)
    print(f"reproduction check: max abs diff {rep['max_abs_diff']:.3g}, passed {rep['passed']}", flush=True)
    if not rep["passed"]:
        print("STOP: D26 not reproduced; no results file written (D27).", json.dumps(rep, indent=1), flush=True)
        return 3
    summary = {rule: {lv: pooled_metrics(folds, rule, lv) for lv in ("90", "80")} for rule in RULES}
    q2 = q2_test(folds)
    out = a.out or ROOT / "results" / f"forecasting_d27_{a.exp}.json"
    X.write_results(out, {
        "decision": "D27", "experiment": a.exp, "horizons": list(run_d26.HORIZONS[a.exp]), "n_samples": len(run.direct),
        "features": run.features, "folds": run.cases, "overrides": {"allow_dirty": a.allow_dirty,
                                                                      "allow_unpinned": a.allow_unpinned},
        "reproduction_check": rep, "q1_coverage_gbm_loco_90": summary["gbm_loco"]["90"]["coverage_pooled"],
        "q1_threshold": ADEQUATE, "q2_test": q2, "summary": summary, "per_fold": folds}, ROOT)
    for rule in RULES:
        s = summary[rule]["90"]
        print(f"{rule:20s} cov90 {s['coverage_pooled']:.3f} (min {s['coverage_min']:.2f} {s['coverage_min_case']}) "
              f"IS90 {s['interval_score']:.3f} width {s['width']:.3f} decided {s['decided_share']:.2f}", flush=True)
    print(f"Q2 GBM-LOCO - LIN-all IS90: {q2['mean_diff']:+.3f} pp, p {q2['p_value']:.3f}, "
          f"{q2['n_gbm_better']}/7 folds: {q2['direction']}", flush=True)
    print(f"written {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
