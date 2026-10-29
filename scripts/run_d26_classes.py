"""Reuse-class confusion and metrics (D30 set) of the proposed model on unseen S6 conditions (D26 E1 set-up).

D26 recorded only class accuracy and macro-F1. This script re-runs, unchanged, the D26 E1 units of the shrunk
residual TFT + EWC (main seed set: 10 orders x 5 seeds, seven held-out conditions) and the four single-fit
comparators, keeps each fold's reuse-class confusion counts, and checks every fold's RMSE against
results/forecasting_d26_e1.json (tolerance 1e-6; it stops on a mismatch). Classes are pooled over the seven
held-out conditions within a run and averaged over runs, as in D30. Units are cached in data/cache/d26_classes,
so an interrupted run resumes.

    python scripts/run_d26_classes.py      # ~2 h on CPU; writes results/forecasting_d26_classes_e1.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_d26  # noqa: E402
from src import evaluation as ev  # noqa: E402
from src import forecasting as FC  # noqa: E402
from src import metrics as M  # noqa: E402
from src import p3  # noqa: E402
from src.calibration import HorizonCalibrator  # noqa: E402
from src.data_pipeline import environment_info  # noqa: E402

EXP, TOL = "e1", 1e-6
K = len(ev.REUSE_CUTPOINTS) + 1
MODEL = "tft_residual_ewc_shrunk"


def confusion(direct: pd.DataFrame, residual_pred: np.ndarray, frame: pd.DataFrame) -> list[list[int]]:
    soh_o = direct["soh_origin"].to_numpy(float)
    pred = soh_o + frame["lin_forecast"].to_numpy(float) + np.asarray(residual_pred, float)
    cm = M.confusion(ev.reuse_class(direct["soh_target"].to_numpy(float)), ev.reuse_class(pred), K)
    return cm.tolist()


def model_unit(r: run_d26.Runner, case: str, order, seed: int, lr: float, lam: float) -> dict:
    order6 = r.fold_order(order, case)

    def fit():
        frame, direct, _ = r.held(case)
        s = p3.build_strategy("tft_ewc", r.features, seed, r.s6, r.settings(lr, lam))
        tasks = r.training_tasks(order6)
        for t in tasks:
            s.fit_task(t)
        pooled = pd.concat([t.val for t in tasks], ignore_index=True)
        cal = HorizonCalibrator(run_d26.HORIZONS[EXP]).update(s, pooled, "pooled six cases")
        p = cal.shrink(s.predict(frame), frame)
        return {"rmse": run_d26.rmse(p - frame[FC.TARGET].to_numpy(float)), "cm": confusion(direct, p, frame)}
    ckpt = FC.Checkpoint(r.data / "cache" / "d26_classes", [FC.Checkpoint.frame_hash(r.direct), EXP, r.profile])
    return ckpt(f"cont_{case}_{'-'.join(order6)}_s{seed}", fit, {"lr": lr, "lam": lam})


def baseline_cms(r: run_d26.Runner) -> dict[str, dict[str, dict]]:
    """{method: {case: {rmse, cm}}}, the single fits of ``Runner.baselines_by_fold`` with their confusion counts."""
    cols = {"rate": "fade_rate_pp_per_cycle", "span": "cycles_to_target",
            "schedule": ["charge_c_rate", "discharge_c_rate", "dod", "v_charge_max", "cv_charge", "cycles_to_target",
                         "days_to_target"]}
    labels = {"persistence": "persistence", "linear_extrapolation": run_d26.LIN, "ridge_F0": "ridge_shared",
              "gbm_F1": run_d26.GBM}
    out: dict[str, dict[str, dict]] = {v: {} for v in labels.values()}
    for case in r.cases:
        task = FC.leave_one_out_task(r.direct, case)
        fit = pd.concat([task.train, task.val], ignore_index=True)
        observed = [f for f in r.features if fit[f].notna().any()]
        preds = FC.baseline_predictions(fit, {"t": task.test}, {"F0": observed, "F1": observed}, seed=0, columns=cols)
        frame, direct, _ = r.held(case)
        lin = frame["lin_forecast"].to_numpy(float)
        for k, p in preds.items():
            if k in labels:
                res = p["t"] - lin
                out[labels[k]][case] = {"rmse": run_d26.rmse(res - frame[FC.TARGET].to_numpy(float)),
                                        "cm": confusion(direct, res, frame)}
    return out


def pooled(cms: list[list[list[int]]]) -> dict:
    m = M.classification_from_confusion(np.sum(np.array(cms), axis=0))
    return {k: m[k] for k in M.CLASS_KEYS} | {"confusion": m["confusion"].tolist()}


def main() -> int:
    committed = json.loads((ROOT / "results" / f"forecasting_d26_{EXP}.json").read_text())
    r = run_d26.Runner(EXP, "paper")
    t = r.d23_tuning()
    mismatches = []

    base = baseline_cms(r)
    for method, folds in base.items():
        for case, v in folds.items():
            want = committed["baselines_by_fold"][case][method]["rmse"]
            if abs(v["rmse"] - want) > TOL:
                mismatches.append((method, case, v["rmse"], want))

    runs = []
    ref = {(x["order_id"], x["seed"]): x for x in committed["runs"] if x["method"] == MODEL and x["set"] == "main"}
    for oid, order in enumerate(r.orders):
        for seed in r.cfg["main_seeds"]:
            units = {case: model_unit(r, case, order, seed, t["lr"], t["ewc_lambda"]) for case in r.cases}
            for case, u in units.items():
                want = ref[(oid, seed)]["per_fold"][case]
                if abs(u["rmse"] - want) > TOL:
                    mismatches.append((MODEL, f"{case} o{oid} s{seed}", u["rmse"], want))
            runs.append({"order_id": oid, "seed": seed, **pooled([u["cm"] for u in units.values()])})
            print(f"order {oid} seed {seed}: accuracy {runs[-1]['accuracy']:.4f}", flush=True)
    if mismatches:
        print(f"{len(mismatches)} folds do not reproduce D26 (first: {mismatches[0]}); not writing results")
        return 1

    summary = {MODEL: {k: {"mean": float(np.mean([x[k] for x in runs])), "std": float(np.std([x[k] for x in runs], ddof=1))}
                       for k in M.CLASS_KEYS}}
    summary[MODEL]["confusion_mean"] = np.mean([x["confusion"] for x in runs], axis=0).tolist()
    for method, folds in base.items():
        p = pooled([v["cm"] for v in folds.values()])
        summary[method] = {k: {"mean": p[k], "std": 0.0} for k in M.CLASS_KEYS} | {"confusion_mean": p["confusion"]}
    out = {"generated_utc": datetime.now(timezone.utc).isoformat(), "environment": environment_info(),
           "decision": "D26 E1 reuse classes (D30 metric set)", "source": f"results/forecasting_d26_{EXP}.json",
           "reproduction": {"tolerance": TOL, "folds_checked": len(runs) * len(r.cases) + 4 * len(r.cases)},
           "n_runs": len(runs), "summary": summary, "runs": runs}
    path = ROOT / "results" / f"forecasting_d26_classes_{EXP}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
