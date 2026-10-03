"""Run protocol P3 (decision D21): pretrain on the first-life pool, then the S6 stream.

Stages (all validation-only until the final evaluation):
1. backbone learning rate per backbone (pretrained fine-tuning, criterion: plasticity);
2. EWC strength per backbone (pretrained EWC, criterion: final average error);
   both on 10 tuning orders disjoint from the evaluation orders, with the one-standard-error rule;
3. evaluation: 10 orders x seeds 0-4 (main) and 5-9 (replicate), for TFT and MLP:
   from scratch + fine-tuning, pretrained + fine-tuning, pretrained + EWC; plus the D20 baselines.

Pretraining depends only on (backbone, learning rate, seed), so each pretrained model is trained once,
checkpointed, and copied into every order and method. Every unit of work is checkpointed under
data/cache/p3, so an interrupted run resumes.

Usage (from the repository root)::

    python scripts/run_p3.py                    # full run -> results/forecasting_d21_p3.json
    python scripts/run_p3.py --profile smoke    # code check only (not reportable)
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_pipeline as dp  # noqa: E402
from src import evaluation as ev  # noqa: E402
from src import experiments as X  # noqa: E402
from src import forecasting as FC  # noqa: E402
from src import p3  # noqa: E402
from src import protocols as P  # noqa: E402
from src import wheeler as W  # noqa: E402

LR_GRID = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
LAMBDA_GRID = [1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0]
TFT = {"window": 4, "window_group": ("Battery_ID", "Cycling_Test")}
VARIANTS = (("mlp", "scratch", "finetune"), ("mlp", "pre", "finetune"), ("mlp", "pre", "ewc"),
            ("tft", "scratch", "tft_finetune"), ("tft", "pre", "tft_finetune"), ("tft", "pre", "tft_ewc"))
PROFILES = {"paper": {"n_orders": 10, "n_tuning_orders": 10, "main_seeds": (0, 1, 2, 3, 4), "replicate_seeds": (5, 6, 7, 8, 9),
                      "epochs": 200},
            "smoke": {"n_orders": 1, "n_tuning_orders": 2, "main_seeds": (0,), "replicate_seeds": (5,), "epochs": 2}}


def variant_name(backbone: str, init: str, method: str) -> str:
    return f"{backbone}_{init}_{'ewc' if method.endswith('ewc') else 'finetune'}"


class Runner:
    def __init__(self, profile: str) -> None:
        self.cfg = PROFILES[profile]
        self.profile = profile
        frames = [p3.harmonised_samples(dp.read_dataset(ROOT / "data" / f"pool_{d.lower()}_checks.csv")
                                        .assign(check_time=lambda t: pd.to_datetime(t["check_time"])), d)
                  for d in ("S7", "S8", "S9")]
        self.pool_s = pd.concat(frames, ignore_index=True)
        wc = dp.read_dataset(ROOT / "data" / "wheeler_checks.csv")
        wc["check_time"] = pd.to_datetime(wc["check_time"])
        self.s6 = p3.harmonised_samples(p3.s6_checks(wc), "S6")
        # D20 per-cycle fade rate, for the linear-extrapolation baseline as defined there.
        d20 = W.build_samples(wc)[["cell", "origin_cycle", "horizon", "fade_rate_pp_per_cycle"]]
        cyc = self.s6["Cycle_Date"].map(dict(zip(wc["check_time"], wc["cycle"])))
        self.s6 = self.s6.assign(origin_cycle=cyc.astype(int)).merge(d20, on=["cell", "origin_cycle", "horizon"],
                                                                     how="left", validate="one_to_one")
        self.pool_task = p3.pool_task(self.pool_s)
        self.history = pd.concat([self.pool_s, self.s6], ignore_index=True)
        self.features = list(p3.SHARED_FEATURES)
        cases = sorted(self.s6["Battery_ID"].unique())
        both = P.sample_orders(cases, self.cfg["n_orders"] + self.cfg["n_tuning_orders"], 0)
        self.orders, self.tuning_orders = both[:self.cfg["n_orders"]], both[self.cfg["n_orders"]:]
        self.ckpt = FC.Checkpoint(ROOT / "data" / "cache" / "p3",
                                  [FC.Checkpoint.frame_hash(self.pool_s), FC.Checkpoint.frame_hash(self.s6), profile])

    # -- building blocks ----------------------------------------------------
    def settings(self, backbone: str, lr: float, extra: dict | None = None) -> dict:
        return {"lr": lr, "epochs": self.cfg["epochs"], **(TFT if backbone == "tft" else {}), **(extra or {})}

    def pretrained(self, backbone: str, lr: float, seed: int):
        """Pretrained EWC strategy (Fisher and anchor from the pool); checkpointed."""
        method = "tft_ewc" if backbone == "tft" else "ewc"

        def fit():
            s = p3.build_strategy(method, self.features, seed, self.history, self.settings(backbone, lr))
            t = time.perf_counter()
            p3.pretrain(s, self.pool_task)
            s.pretrain_time_s_ = time.perf_counter() - t
            s.windows = None if backbone == "tft" else getattr(s, "windows", None)   # history is rebuilt on load
            return s
        s = self.ckpt(f"pretrain_{backbone}_s{seed}", fit, {"lr": lr})
        if backbone == "tft":
            s.set_history(self.history)
        return s

    def strategy(self, backbone: str, init: str, method: str, lr: float, seed: int, lam: float | None):
        settings = self.settings(backbone, lr, {"ewc_lambda": lam} if method.endswith("ewc") else None)
        if init == "scratch":
            return p3.build_strategy(method, self.features, seed, self.history, settings)
        pre = self.pretrained(backbone, lr, seed)
        s = p3.build_strategy(method, self.features, seed, self.history, settings)
        s.preprocessor, s.target_mean_, s.target_scale_ = pre.preprocessor, pre.target_mean_, pre.target_scale_
        s.model = copy.deepcopy(pre.model)
        s.tasks_seen_ = list(pre.tasks_seen_)
        if method.endswith("ewc"):
            s.fishers_, s.optima_ = copy.deepcopy(pre.fishers_), copy.deepcopy(pre.optima_)
        return s

    def stream_run(self, variant: tuple, order, seed: int, lr: float, lam: float | None, split: str) -> dict:
        backbone, init, method = variant
        name = f"{split}_{variant_name(*variant)}_{'-'.join(order)}_s{seed}"

        def fit():
            s = self.strategy(backbone, init, method, lr, seed, lam)
            out = p3.run_stream(s, FC.forecast_stream(self.s6, order), self.pool_task, eval_split=split)
            per_task = {t: float(out["R"].iloc[-1][t]) for t in out["R"].columns}
            return {"metrics": out["metrics"], "pool_error_before": out["pool_error_before"],
                    "pool_error_after": out["pool_error_after"], "final_per_task": per_task,
                    "train_time_s": float(np.sum(out["train_time_s"])), "R": out["R"].to_dict()}
        return self.ckpt(name, fit, {"lr": lr, "lam": lam})

    # -- tuning (D21: 10 runs per grid point, one-standard-error rule) -------
    @staticmethod
    def one_se(table: pd.DataFrame, param: str, criterion: str) -> tuple[float, dict]:
        g = table.groupby(param)[criterion].agg(["mean", "std", "count"])
        g["se"] = g["std"] / np.sqrt(g["count"])
        best = g["mean"].idxmin()
        eligible = g[g["mean"] <= g.loc[best, "mean"] + g.loc[best, "se"]]
        chosen = float(min(eligible.index))                      # most conservative: smallest value
        return chosen, {"best_value": float(best), "best_mean": float(g.loc[best, "mean"]),
                        "best_se": float(g.loc[best, "se"]), "chosen": chosen,
                        "chosen_mean": float(g.loc[chosen, "mean"]), "table": g.reset_index().to_dict("records")}

    def tune(self) -> dict:
        out = {}
        for backbone in ("mlp", "tft"):
            ft = (backbone, "pre", "tft_finetune" if backbone == "tft" else "finetune")
            rows = [{"lr": lr, "order": i, **self.stream_run(ft, o, 0, lr, None, "val")["metrics"]}
                    for lr in LR_GRID for i, o in enumerate(self.tuning_orders)]
            lr, lr_info = self.one_se(pd.DataFrame(rows), "lr", "plasticity")
            ewc = (backbone, "pre", "tft_ewc" if backbone == "tft" else "ewc")
            rows = [{"lam": lam, "order": i, **self.stream_run(ewc, o, 0, lr, lam, "val")["metrics"]}
                    for lam in LAMBDA_GRID for i, o in enumerate(self.tuning_orders)]
            lam, lam_info = self.one_se(pd.DataFrame(rows), "lam", "final_avg_error")
            out[backbone] = {"lr": lr, "ewc_lambda": lam, "lr_selection": lr_info, "lambda_selection": lam_info,
                             "lr_at_grid_edge": lr in (min(LR_GRID), max(LR_GRID)),
                             "lambda_at_grid_edge": lam in (min(LAMBDA_GRID), max(LAMBDA_GRID))}
            print(f"tuned {backbone}: lr {lr} (best {lr_info['best_value']}), lambda {lam} (best {lam_info['best_value']})",
                  flush=True)
        return out

    # -- evaluation ----------------------------------------------------------
    def evaluate(self, tuned: dict) -> pd.DataFrame:
        rows = []
        for seeds_name in ("main_seeds", "replicate_seeds"):
            for seed in self.cfg[seeds_name]:
                for variant in VARIANTS:
                    t = tuned[variant[0]]
                    for oid, order in enumerate(self.orders):
                        r = self.stream_run(variant, order, seed, t["lr"], t["ewc_lambda"], "test")
                        rows.append({"set": seeds_name.split("_")[0], "method": variant_name(*variant), "seed": seed,
                                     "order_id": oid, **r["metrics"], "pool_error_before": r["pool_error_before"],
                                     "pool_error_after": r["pool_error_after"],
                                     "rmse_final_mean_over_tasks": float(np.mean(list(r["final_per_task"].values()))),
                                     "train_time_s": r["train_time_s"]})
                print(f"evaluated {seeds_name} seed {seed}", flush=True)
        return pd.DataFrame(rows)

    def baselines(self) -> pd.DataFrame:
        """D20 baselines on the S6 test splits, fitted on all S6 training rows (as P1)."""
        tasks = FC.forecast_stream(self.s6, self.orders[0])
        train = pd.concat([t.train for t in tasks], ignore_index=True)
        tests = {t.name: t.test for t in tasks}
        # As in D20: inputs never observed in the training rows (e.g. rest_soc on S6) are dropped; no label is used.
        observed = [f for f in self.features if train[f].notna().any()]
        sets = {"F0": observed, "F1": observed}
        cols = {"rate": "fade_rate_pp_per_cycle", "span": "cycles_to_target",
                "schedule": ["charge_c_rate", "discharge_c_rate", "dod", "v_charge_max", "cv_charge", "cycles_to_target",
                             "days_to_target"]}
        preds = FC.baseline_predictions(train, tests, sets, seed=0, columns=cols)
        rows = []
        for name, by_task in preds.items():
            if name == "ridge_F1":
                continue                                        # F0 = F1 here (one shared input set)
            label = {"ridge_F0": "ridge_shared", "gbm_F1": "gbm_shared"}.get(name, name)
            errs = {k: ev.task_error(tests[k][FC.TARGET], v, "RMSE") for k, v in by_task.items()}
            rows.append({"method": label, "rmse_final_mean_over_tasks": float(np.mean(list(errs.values()))), **errs})
        return pd.DataFrame(rows)


def compare(runs: pd.DataFrame, reference: str, metrics=("final_avg_error", "forgetting", "backward_transfer")) -> pd.DataFrame:
    """Paired Wilcoxon (matched order x seed) of every method against ``reference``, Holm-corrected per set."""
    out = []
    for set_name, g in runs.groupby("set"):
        key = ["order_id", "seed"]
        ref = g[g["method"] == reference].set_index(key)
        pv, rows = {}, []
        for method in sorted(set(g["method"]) - {reference}):
            other = g[g["method"] == method].set_index(key)
            for m in metrics:
                a, b = other[m].reindex(ref.index).to_numpy(float), ref[m].to_numpy(float)
                res = ev.paired_comparison(a, b)
                d = a - b
                rows.append({"set": set_name, "metric": m, "method": method, "baseline": reference,
                             "median_diff": float(np.median(d)), **res})
                pv[(method, m)] = res["p_value"]
        holm = ev.holm_correction(pv)
        out += [{**r, "p_holm": holm[(r["method"], r["metric"])]} for r in rows]
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", choices=tuple(PROFILES), default="paper")
    args = ap.parse_args()
    r = Runner(args.profile)
    print(f"pool: {len(r.pool_s)} samples ({r.pool_s['cell'].nunique()} cells); S6: {len(r.s6)} samples; "
          f"orders {len(r.orders)}, tuning orders {len(r.tuning_orders)}", flush=True)
    tuned = r.tune()
    runs = r.evaluate(tuned)
    base = r.baselines()
    summary = (runs.groupby(["set", "method"])[["final_avg_error", "forgetting", "backward_transfer", "plasticity",
                                               "pool_error_before", "pool_error_after", "train_time_s"]]
               .agg(["mean", "std", "median"]))
    summary.columns = ["_".join(c) for c in summary.columns]
    tests = compare(runs, "tft_pre_ewc")
    pre = {f"{b}_s{s}": float(getattr(r.pretrained(b, tuned[b]["lr"], s), "pretrain_time_s_", np.nan))
           for b in ("mlp", "tft") for s in (*r.cfg["main_seeds"], *r.cfg["replicate_seeds"])}
    name = "forecasting_d21_p3.json" if args.profile == "paper" else "forecasting_d21_p3_smoke.json"
    X.write_results(ROOT / "results" / name, {
        "profile": args.profile, "decision": "D21", "protocol": "P3",
        "n_samples": {"pool": len(r.pool_s), "pool_cells": int(r.pool_s["cell"].nunique()), "s6": len(r.s6)},
        "pool_split_sizes": {k: len(getattr(r.pool_task, k)) for k in ("train", "val", "test")},
        "features": r.features, "orders": [list(o) for o in r.orders], "tuning_orders": [list(o) for o in r.tuning_orders],
        "tuning": tuned, "runs": runs.to_dict("records"), "summary": summary.reset_index().to_dict("records"),
        "baselines": base.to_dict("records"), "tests_vs_tft_pre_ewc": tests.to_dict("records"),
        "pretrain_time_s": pre}, ROOT)
    print(f"written results/{name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
