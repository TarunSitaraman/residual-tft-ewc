"""Run decision D23: learned residual on linear extrapolation (E1, horizons 1-3; E2, horizons 1-6) on the S6 stream.

Arms: ``direct`` (target Delta, as D21) and ``residual`` (target Delta - lin_forecast; forecast = lin + prediction;
every error is identical on both scales). Methods: TFT and MLP x fine-tuning and EWC, from scratch. Tuning, the
evaluation orders and seeds, and the replication rule are those of D21 (see ``run_p3.py``).

Each unit is checkpointed under data/cache/d23, so parts can run in parallel processes and a final run without
``--part`` only collects them and writes the results file.

Usage (from the repository root)::

    python scripts/run_d23.py --exp e1 --part tft_residual   # one part (tune + evaluate)
    python scripts/run_d23.py --exp e1                       # all parts, then results/forecasting_d23_e1.json
    python scripts/run_d23.py --exp e1 --profile smoke       # code check only (not reportable)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_p3  # noqa: E402
from src import data_pipeline as dp  # noqa: E402
from src import evaluation as ev  # noqa: E402
from src import experiments as X  # noqa: E402
from src import forecasting as FC  # noqa: E402
from src import p3  # noqa: E402
from src import protocols as P  # noqa: E402

HORIZONS = {"e1": (1, 2, 3), "e2": (1, 2, 3, 4, 5, 6)}
PARTS = ("mlp_direct", "mlp_residual", "tft_direct", "tft_residual")
REFERENCE = "tft_residual_ewc"
CL_METRICS = ("final_avg_error", "forgetting", "backward_transfer")


class Runner(run_p3.Runner):
    """D23 runner; reuses the D21 selection code (``one_se``, ``select``) of the P3 runner."""

    def __init__(self, exp: str, profile: str) -> None:
        self.cfg, self.profile, self.exp = run_p3.PROFILES[profile], profile, exp
        wc = dp.read_dataset(ROOT / "data" / "wheeler_checks.csv")
        self.frames = {"direct": p3.s6_samples(wc, HORIZONS[exp])}
        self.frames["residual"] = p3.residual_target(self.frames["direct"])
        self.features = list(p3.SHARED_FEATURES)
        cases = sorted(self.frames["direct"]["Battery_ID"].unique())
        both = P.sample_orders(cases, self.cfg["n_orders"] + self.cfg["n_tuning_orders"], 0)
        self.orders, self.tuning_orders = both[:self.cfg["n_orders"]], both[self.cfg["n_orders"]:]
        self.ckpt = FC.Checkpoint(ROOT / "data" / "cache" / "d23",
                                  [FC.Checkpoint.frame_hash(self.frames["direct"]), exp, profile])

    def stream_run(self, part: str, method: str, order, seed: int, lr: float, lam: float | None, split: str) -> dict:
        backbone, arm = part.split("_")
        name = f"{split}_{part}_{method}_{'-'.join(order)}_s{seed}"

        def fit():
            frame = self.frames[arm]
            kind = ("tft_" if backbone == "tft" else "") + method
            settings = self.settings(backbone, lr, {"ewc_lambda": lam} if method == "ewc" else None)
            s = p3.build_strategy(kind, self.features, seed, frame, settings)
            stream = FC.forecast_stream(frame, order)
            out = p3.run_stream(s, stream, None, eval_split=split)
            final = pd.concat([getattr(t, split) for t in stream], ignore_index=True)
            pred = np.concatenate([s.predict(getattr(t, split)) for t in stream])
            return {"metrics": out["metrics"], "horizon_rmse": horizon_rmse(final, pred),
                    "train_time_s": float(np.sum(out["train_time_s"]))}
        return self.ckpt(name, fit, {"lr": lr, "lam": lam})

    def tune(self, part: str) -> dict:
        lr, lr_info = self.select(lambda v, o: self.stream_run(part, "finetune", o, 0, v, None, "val")["metrics"],
                                  run_p3.LR_GRID, "lr", "plasticity")
        lam, lam_info = self.select(lambda v, o: self.stream_run(part, "ewc", o, 0, lr, v, "val")["metrics"],
                                    run_p3.LAMBDA_GRID, "lam", "final_avg_error")
        print(f"tuned {part}: lr {lr} (best {lr_info['best_value']}), lambda {lam} (best {lam_info['best_value']})",
              flush=True)
        return {"lr": lr, "ewc_lambda": lam, "lr_selection": lr_info, "lambda_selection": lam_info}

    def evaluate(self, part: str, tuned: dict) -> list[dict]:
        rows = []
        for set_name in ("main", "replicate"):
            for seed in self.cfg[f"{set_name}_seeds"]:
                for method in ("finetune", "ewc"):
                    for oid, order in enumerate(self.orders):
                        r = self.stream_run(part, method, order, seed, tuned["lr"], tuned["ewc_lambda"], "test")
                        rows.append({"set": set_name, "method": f"{part}_{method}", "seed": seed, "order_id": oid,
                                     **r["metrics"], **{f"rmse_h{h}": v for h, v in r["horizon_rmse"].items()},
                                     "train_time_s": r["train_time_s"]})
            print(f"evaluated {part} {set_name}", flush=True)
        return rows

    def baselines(self) -> list[dict]:
        """D21 baselines on the S6 test splits (fitted on all S6 training rows), task-mean and per-horizon RMSE."""
        frame = self.frames["direct"]
        tasks = FC.forecast_stream(frame, self.orders[0])
        train = pd.concat([t.train for t in tasks], ignore_index=True)
        tests = {t.name: t.test for t in tasks}
        observed = [f for f in self.features if train[f].notna().any()]
        cols = {"rate": "fade_rate_pp_per_cycle", "span": "cycles_to_target",
                "schedule": ["charge_c_rate", "discharge_c_rate", "dod", "v_charge_max", "cv_charge", "cycles_to_target",
                             "days_to_target"]}
        preds = FC.baseline_predictions(train, tests, {"F0": observed, "F1": observed}, seed=0, columns=cols)
        final = pd.concat(tests.values(), ignore_index=True)
        rows = []
        for name, by_task in preds.items():
            if name == "ridge_F1":
                continue                                        # F0 = F1 here (one shared input set)
            label = {"ridge_F0": "ridge_shared", "gbm_F1": "gbm_shared"}.get(name, name)
            errs = [ev.task_error(tests[k][FC.TARGET], v, "RMSE") for k, v in by_task.items()]
            hr = horizon_rmse(final, np.concatenate([by_task[k] for k in tests]))
            rows.append({"method": label, "final_avg_error": float(np.mean(errs)),
                         **{f"rmse_h{h}": v for h, v in hr.items()}})
        return rows


def horizon_rmse(frame: pd.DataFrame, pred: np.ndarray) -> dict[int, float]:
    err = np.asarray(pred, float) - frame[FC.TARGET].to_numpy(float)
    h = frame["horizon"].to_numpy()
    return {int(k): float(np.sqrt(np.mean(err[h == k] ** 2))) for k in sorted(np.unique(h))}


def tests(runs: pd.DataFrame, base: pd.DataFrame, families: dict[str, tuple[list[str], bool, list[str]]]) -> list[dict]:
    """Paired Wilcoxon of ``REFERENCE`` against other methods and baselines, Holm-corrected per seed set and family.

    ``families`` maps a name to (metrics, include the other stream methods?, baselines to include). A baseline is a
    single value, repeated against the runs, and only for the metrics it has. Differences are reference minus
    other: negative favours ``REFERENCE``."""
    out = []
    key = ["order_id", "seed"]
    for set_name, g in runs.groupby("set"):
        ref = g[g["method"] == REFERENCE].set_index(key)
        for family, (metrics, with_methods, baselines) in families.items():
            others = [(m, g[g["method"] == m].set_index(key)) for m in sorted(set(g["method"]) - {REFERENCE})] if with_methods else []
            others += [(b["method"], b) for b in base.to_dict("records") if b["method"] in baselines]
            rows, pv = [], {}
            for name, other in others:
                for m in metrics:
                    if isinstance(other, dict):
                        if m not in other:
                            continue
                        a = np.full(len(ref), float(other[m]))
                    else:
                        a = other[m].reindex(ref.index).to_numpy(float)
                    b = ref[m].to_numpy(float)
                    res = ev.paired_comparison(b, a)
                    rows.append({"set": set_name, "family": family, "metric": m, "reference": REFERENCE, "other": name,
                                 "median_diff": float(np.median(b - a)), **res})
                    pv[(name, m)] = res["p_value"]
            holm = ev.holm_correction(pv)
            out += [{**r, "p_holm": holm[(r["other"], r["metric"])]} for r in rows]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp", choices=tuple(HORIZONS), required=True)
    ap.add_argument("--part", choices=PARTS)
    ap.add_argument("--profile", choices=tuple(run_p3.PROFILES), default="paper")
    args = ap.parse_args()
    r = Runner(args.exp, args.profile)
    print(f"{args.exp}: {len(r.frames['direct'])} S6 samples, horizons {HORIZONS[args.exp]}", flush=True)
    parts = (args.part,) if args.part else PARTS
    tuned, rows = {}, []
    for part in parts:
        tuned[part] = r.tune(part)
        rows += r.evaluate(part, tuned[part])
    if args.part:
        return 0                                                # results are collected by a run without --part
    runs, base = pd.DataFrame(rows), pd.DataFrame(r.baselines())
    hcols = [f"rmse_h{h}" for h in HORIZONS[args.exp]]
    summary = runs.groupby(["set", "method"])[[*CL_METRICS, "plasticity", *hcols, "train_time_s"]].agg(["mean", "std", "median"])
    summary.columns = ["_".join(c) for c in summary.columns]
    # D23: final error vs every method and baseline; forgetting and BWT vs the stream methods (one family);
    # in E2, the per-horizon errors vs linear extrapolation (a family of six).
    families = {"final": (list(CL_METRICS), True, list(base["method"]))}
    if args.exp == "e2":
        families["horizon"] = (hcols, False, ["linear_extrapolation"])
    name = f"forecasting_d23_{args.exp}" + (".json" if args.profile == "paper" else "_smoke.json")
    X.write_results(ROOT / "results" / name, {
        "profile": args.profile, "decision": "D23", "experiment": args.exp, "horizons": list(HORIZONS[args.exp]),
        "n_samples": len(r.frames["direct"]), "features": r.features, "orders": [list(o) for o in r.orders],
        "tuning_orders": [list(o) for o in r.tuning_orders], "tuning": tuned, "runs": runs.to_dict("records"),
        "summary": summary.reset_index().to_dict("records"), "baselines": base.to_dict("records"),
        "tests_vs_reference": tests(runs, base, families)}, ROOT)
    print(summary[["final_avg_error_mean", "final_avg_error_median"]].round(3).to_string(), flush=True)
    print(base.round(3).to_string(), flush=True)
    print(f"written results/{name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
