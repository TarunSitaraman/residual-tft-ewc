"""Run decision D28: continual calibration (quantile tracking) and refit on an unseen usage condition (P2).

Setting (D28, as D27): S6 samples with the D21 shared inputs, E1 = horizons 1-3, E2 = 1-6, seven folds, one per
held-out test case; every sample of the held-out case is scored. The held-out case arrives over time: a sample is
issued at its origin check time and matures at its target check time; when a sample is issued, only held-out-case
samples that have matured by then are known.

- **STATIC:** D27 GBM-LOCO (six-case gradient boosting, leave-one-condition-out half-widths q0_h).
- **TRACK:** STATIC centre; per horizon the half-width starts at q0_h and moves by c * q0_h * (e - a) whenever a
  sample of that horizon matures (e = 1 if its target lay outside the interval issued for it).
- **REFIT:** gradient boosting refitted on the six training cases plus the matured held-out samples; static q0_h.
- **REFIT+TRACK (proposed):** both.

The step c is chosen per fold on the six training cases only (pseudo-held-out, D28). STATIC must reproduce the D27
results file before anything is written (stop condition).

Usage (from the repository root)::

    python scripts/run_d28.py --exp e1
    python scripts/run_d28.py --exp e2
"""

from __future__ import annotations

import argparse
import heapq
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
from src import evaluation as ev  # noqa: E402
from src import experiments as X  # noqa: E402
from src import forecasting as FC  # noqa: E402
from src.calibration import LEVELS, interval_metrics  # noqa: E402

ARMS = ("static", "track", "refit", "refit_track")
C_GRID = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
Q1_MIN, Q2_POOLED, Q2_MIN = 0.80, 0.85, 0.80                # 90 % coverage thresholds
MIN_DIFF = 0.05                                             # pp, Q3 and Q4
TOL = 1e-9
PRIMARY_CASE = "case7"


def times(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Issue time (origin check) and maturity time (target check) of every sample, as datetime64."""
    t = pd.to_datetime(frame["Cycle_Date"]).to_numpy()
    m = (pd.to_datetime(frame["Cycle_Date"]) + pd.to_timedelta(frame["days_to_target"].astype(float), unit="D")).to_numpy()
    return t, m


def track(frame: pd.DataFrame, centre: np.ndarray, q0: dict, level: float, c: float) -> np.ndarray:
    """Half-width issued to every sample under quantile tracking with delayed feedback (D28 TRACK rule)."""
    t, m = times(frame)
    y, h = frame[FC.TARGET].to_numpy(float), frame["horizon"].astype(int).to_numpy()
    a, q = 1.0 - level, {k: float(v) for k, v in q0.items()}
    issued, heap = np.empty(len(frame)), []
    for rank, i in enumerate(np.argsort(t, kind="stable")):
        while heap and heap[0][0] <= t[i]:                   # everything matured by this issue time
            _, _, k = heapq.heappop(heap)
            e = float(abs(y[k] - centre[k]) > issued[k])
            q[h[k]] = max(0.0, q[h[k]] + c * q0[h[k]] * (e - a))
        issued[i] = q[h[i]]
        heapq.heappush(heap, (m[i], rank, i))
    return issued


def gbm_fit_predict(fit: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    """The D27 gradient boosting (``gbm_F1`` of ``FC.baseline_predictions`` on the observed inputs), alone."""
    observed = [f for f in features if fit[f].notna().any()]
    return FC._gbm(0).fit(fit[observed], fit[FC.TARGET].to_numpy(float)).predict(test[observed])


def refit_predictions(fit_rows: pd.DataFrame, test: pd.DataFrame, features: list[str],
                      p_static: np.ndarray) -> tuple[np.ndarray, int]:
    """REFIT centre: at each issue time, refit on ``fit_rows`` plus the held-out samples matured by then."""
    t, m = times(test)
    pred, fits = p_static.copy(), 0
    for ti in np.unique(t):
        matured = np.flatnonzero(m <= ti)
        if matured.size == 0:
            continue
        issued = np.flatnonzero(t == ti)
        fit = pd.concat([fit_rows, test.iloc[matured]], ignore_index=True)
        pred[issued] = gbm_fit_predict(fit, test.iloc[issued], features)
        fits += 1
    return pred, fits


def block(frame: pd.DataFrame, centre: np.ndarray, qv: np.ndarray, level: float) -> dict:
    """D27 interval block (all, H+, H-, pooling counts, reuse decision support) for per-sample half-widths."""
    y = frame[FC.TARGET].to_numpy(float)
    hplus = frame["fade_rate_pp_per_cycle"].notna().to_numpy()
    cov = np.abs(y - centre) <= qv
    out = {"all": interval_metrics(y, centre, qv, level), "n": int(len(y)), "covered": int(cov.sum())}
    for name, msk in (("hplus", hplus), ("hminus", ~hplus)):
        out[name] = {**interval_metrics(y[msk], centre[msk], qv[msk], level), "n": int(msk.sum()),
                     "covered": int(cov[msk].sum())} if msk.any() else {}
    soh0 = frame["soh_origin"].to_numpy(float)
    lo, hi = ev.reuse_class(soh0 + centre - qv), ev.reuse_class(soh0 + centre + qv)
    true_c, pred_c = ev.reuse_class(frame["soh_target"].to_numpy(float)), ev.reuse_class(soh0 + centre)
    decided = lo == hi
    out["decided_share"] = float(decided.mean())
    out["decided_agreement"] = float(np.mean(true_c[decided] == pred_c[decided])) if decided.any() else float("nan")
    return out


def static_q(frame: pd.DataFrame, q: dict) -> np.ndarray:
    return frame["horizon"].astype(int).map(q).to_numpy(float)


def select_c(inner: dict[str, tuple[pd.DataFrame, np.ndarray]]) -> tuple[float, dict]:
    """Step size from the six training cases only: each is pseudo-held-out with the other five's q0 (D28)."""
    scores = {}
    for c in C_GRID:
        per = []
        for j, (rows, pred) in inner.items():
            rest = [(r, p) for k, (r, p) in inner.items() if k != j]
            frame = pd.concat([r for r, _ in rest], ignore_index=True)
            resid = np.abs(frame[FC.TARGET].to_numpy(float) - np.concatenate([p for _, p in rest]))
            q0 = run_d27.halfwidths(resid, run_d27.keys_of(frame, False), 0.9)[0]
            qv = track(rows, pred, q0, 0.9, c)
            per.append(interval_metrics(rows[FC.TARGET].to_numpy(float), pred, qv, 0.9)["interval_score"])
        scores[c] = float(np.mean(per))
    best = min(C_GRID, key=lambda c: (scores[c], c))
    return best, {str(c): s for c, s in scores.items()}


class D28(run_d27.D27):
    def fold(self, case: str) -> dict:
        d, feats = self.direct, self.features
        test = d[d["Battery_ID"] == case].reset_index(drop=True)
        y = test[FC.TARGET].to_numpy(float)
        fit_rows = run_d27.p2_fit_rows(d, case)
        p_static = run_d27.gbm_predict(fit_rows, test, feats)
        others = d[d["Battery_ID"] != case].reset_index(drop=True)
        inner = {}
        for j in sorted(others["Battery_ID"].unique()):
            rows = others[others["Battery_ID"] == j].reset_index(drop=True)
            inner[j] = (rows, run_d27.gbm_predict(run_d27.p2_fit_rows(others, j), rows, feats))
        frame = pd.concat([r for r, _ in inner.values()], ignore_index=True)
        resid = np.abs(frame[FC.TARGET].to_numpy(float) - np.concatenate([p for _, p in inner.values()]))
        q0 = {lv: run_d27.halfwidths(resid, run_d27.keys_of(frame, False), lv)[0] for lv in LEVELS}
        c, c_scores = select_c(inner)
        p_refit, n_fits = refit_predictions(fit_rows, test, feats, p_static)
        centres = {"static": p_static, "track": p_static, "refit": p_refit, "refit_track": p_refit}
        out = {"case": case, "n": int(len(test)), "c": c, "c_scores": c_scores, "n_refits": n_fits,
               "rmse": {arm: run_d26.rmse(centres[arm] - y) for arm in ("static", "refit")},
               "q0": {f"{round(lv * 100)}": {str(k): float(v) for k, v in q0[lv].items()} for lv in LEVELS},
               "rules": {}}
        for arm in ARMS:
            out["rules"][arm] = {}
            for lv in LEVELS:
                qv = (track(test, centres[arm], q0[lv], lv, c) if arm.endswith("track")
                      else static_q(test, q0[lv]))
                out["rules"][arm][f"{round(lv * 100)}"] = block(test, centres[arm], qv, lv)
        return out


def reproduction(folds: list[dict], d27: dict) -> dict:
    """Stop condition: STATIC reproduces D27 GBM-LOCO per-fold coverage, interval score and RMSE."""
    ref = {f["case"]: f for f in d27["per_fold"]}
    diffs = {}
    for f in folds:
        r = ref[f["case"]]
        diffs[f["case"]] = max([abs(f["rmse"]["static"] - r["rmse_gbm"])] + [
            abs(f["rules"]["static"][lv]["all"][k] - r["rules"]["gbm_loco"][lv]["all"][k])
            for lv in ("90", "80") for k in ("coverage", "interval_score")])
    worst = max(diffs.values())
    return {"abs_diff_per_fold": diffs, "max_abs_diff": worst, "passed": bool(worst <= TOL)}


def paired_test(a: np.ndarray, b: np.ndarray, cases: list[str]) -> dict:
    """Two-sided exact Wilcoxon of a - b over folds; direction needs p < 0.05 and |mean| >= 0.05 pp."""
    d = np.asarray(a, float) - np.asarray(b, float)
    p = 1.0 if np.all(d == 0) else float(stats.wilcoxon(d, alternative="two-sided", method="exact").pvalue)
    verdict = ("positive" if d.mean() <= -MIN_DIFF else "negative" if d.mean() >= MIN_DIFF
               else "not established") if p < 0.05 else "not established"
    return {"mean_diff": float(d.mean()), "per_fold": dict(zip(cases, map(float, d))), "n_better": int((d < 0).sum()),
            "p_value": p, "verdict": verdict}


def questions(folds: list[dict], summary: dict) -> dict:
    cases = [f["case"] for f in folds]
    s = summary["refit_track"]["90"]
    return {
        "q1_case7_coverage": s["coverage_per_fold"].get(PRIMARY_CASE), "q1_threshold": Q1_MIN,
        "q2_pooled": s["coverage_pooled"], "q2_min": s["coverage_min"], "q2_min_case": s["coverage_min_case"],
        "q3_rmse_refit_vs_static": paired_test([f["rmse"]["refit"] for f in folds],
                                               [f["rmse"]["static"] for f in folds], cases),
        "q4_is90_refit_track_vs_static": paired_test(
            [f["rules"]["refit_track"]["90"]["all"]["interval_score"] for f in folds],
            [f["rules"]["static"]["90"]["all"]["interval_score"] for f in folds], cases)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp", choices=tuple(run_d26.HORIZONS), required=True)
    ap.add_argument("--d27", type=Path, help="D27 results file (default results/forecasting_d27_<exp>.json)")
    ap.add_argument("--out", type=Path, help="output (default results/forecasting_d28_<exp>.json)")
    ap.add_argument("--allow-dirty", action="store_true", help="code check only; recorded in the results file")
    ap.add_argument("--allow-unpinned", action="store_true", help="code check only; recorded in the results file")
    a = ap.parse_args()
    problems = run_d27.preflight(a.allow_dirty, a.allow_unpinned)
    if problems:
        print("PRE-FLIGHT:\n" + "\n".join(problems), flush=True)
        return 2
    d27 = json.loads((a.d27 or ROOT / "results" / f"forecasting_d27_{a.exp}.json").read_text())
    run = D28(a.exp)
    if sorted(d27["folds"]) != run.cases:
        raise SystemExit(f"D27 folds {d27['folds']} differ from the S6 cases {run.cases}")
    print(f"D28 {a.exp}: {len(run.direct)} S6 samples, {len(run.cases)} folds", flush=True)
    folds = []
    for c in run.cases:
        folds.append(run.fold(c))
        f = folds[-1]
        print(f"  {c}: c {f['c']}, {f['n_refits']} refits, RMSE static {f['rmse']['static']:.3f} refit "
              f"{f['rmse']['refit']:.3f}, cov90 static {f['rules']['static']['90']['all']['coverage']:.2f} "
              f"refit+track {f['rules']['refit_track']['90']['all']['coverage']:.2f}", flush=True)
    rep = reproduction(folds, d27)
    print(f"reproduction check: max abs diff {rep['max_abs_diff']:.3g}, passed {rep['passed']}", flush=True)
    if not rep["passed"]:
        print("STOP: D27 not reproduced; no results file written (D28).", json.dumps(rep, indent=1), flush=True)
        return 3
    summary = {arm: {lv: run_d27.pooled_metrics(folds, arm, lv) for lv in ("90", "80")} for arm in ARMS}
    q = questions(folds, summary)
    out = a.out or ROOT / "results" / f"forecasting_d28_{a.exp}.json"
    X.write_results(out, {
        "decision": "D28", "experiment": a.exp, "horizons": list(run_d26.HORIZONS[a.exp]), "n_samples": len(run.direct),
        "features": run.features, "folds": run.cases, "c_grid": list(C_GRID),
        "overrides": {"allow_dirty": a.allow_dirty, "allow_unpinned": a.allow_unpinned},
        "reproduction_check": rep, "questions": q, "summary": summary, "per_fold": folds}, ROOT)
    for arm in ARMS:
        s = summary[arm]["90"]
        print(f"{arm:12s} cov90 {s['coverage_pooled']:.3f} (min {s['coverage_min']:.2f} {s['coverage_min_case']}) "
              f"IS90 {s['interval_score']:.3f} width {s['width']:.3f} decided {s['decided_share']:.2f} "
              f"agree {s['decided_agreement']:.3f}", flush=True)
    print(f"Q1 case7 cov90 {q['q1_case7_coverage']:.3f}; Q2 pooled {q['q2_pooled']:.3f} min {q['q2_min']:.3f} "
          f"({q['q2_min_case']}); Q3 RMSE {q['q3_rmse_refit_vs_static']['mean_diff']:+.3f} pp p "
          f"{q['q3_rmse_refit_vs_static']['p_value']:.3f} {q['q3_rmse_refit_vs_static']['verdict']}; Q4 IS90 "
          f"{q['q4_is90_refit_track_vs_static']['mean_diff']:+.3f} pp p "
          f"{q['q4_is90_refit_track_vs_static']['p_value']:.3f} {q['q4_is90_refit_track_vs_static']['verdict']}",
          flush=True)
    print(f"written {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
