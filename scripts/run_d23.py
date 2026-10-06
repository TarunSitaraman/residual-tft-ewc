"""Run decision D23 (and D22 with ``--inputs f2``): learned residual on linear extrapolation on the S6 stream.

D23: E1 (horizons 1-3) and E2 (horizons 1-6) with the D21 shared inputs. D22: the same setting with F2 = the
D21 inputs plus the ageing-block aggregates from the Spark job (``data/s6_ageing_blocks_spark.csv``, scripts/spark_ageing_features.py, validated against scripts/build_s6_ageing_features.py);
residual TFT + EWC with F2 is compared with the committed D23 runs of residual TFT + EWC on the D21 inputs, after
checking that units of those runs reproduce exactly.

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
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_p3  # noqa: E402
from src import ageing as A  # noqa: E402
from src import data_pipeline as dp  # noqa: E402
from src import evaluation as ev  # noqa: E402
from src import experiments as X  # noqa: E402
from src import forecasting as FC  # noqa: E402
from src import p3  # noqa: E402
from src import protocols as P  # noqa: E402

HORIZONS = {"e1": (1, 2, 3), "e2": (1, 2, 3, 4, 5, 6)}
PARTS = ("mlp_direct", "mlp_residual", "tft_direct", "tft_residual")
REFERENCE = "tft_residual_ewc"
REFERENCE_D21 = "tft_residual_ewc_d21_inputs"
CL_METRICS = ("final_avg_error", "forgetting", "backward_transfer")


class Runner(run_p3.Runner):
    """D23 runner; reuses the D21 selection code (``one_se``, ``select``) of the P3 runner."""

    def __init__(self, exp: str, profile: str, inputs: str = "d21") -> None:
        self.cfg, self.profile, self.exp, self.inputs = run_p3.PROFILES[profile], profile, exp, inputs
        wc = dp.read_dataset(ROOT / "data" / "wheeler_checks.csv")
        direct = p3.s6_samples(wc, HORIZONS[exp])
        self.features = list(p3.SHARED_FEATURES)
        if inputs == "f2":
            # D22: the origin check at cycle c uses the ageing block ending at c (causality checked by the builder).
            # D22: the inputs come from the Spark job, used only if its full-set validation against pandas passed.
            check = json.loads((ROOT / "results" / "d22_spark_validation.json").read_text())
            if not check["identical_within_1e-9_relative"]:
                raise SystemExit("Spark ageing inputs differ from the pandas implementation; see d22_spark_validation.json")
            blocks = pd.read_csv(ROOT / "data" / "s6_ageing_blocks_spark.csv")[["cell", "block_end", *A.feature_names()]]
            direct = direct.merge(blocks.rename(columns={"block_end": "origin_cycle"}), on=["cell", "origin_cycle"],
                                  how="left", validate="many_to_one")
            self.features += A.feature_names()
        self.frames = {"direct": direct, "residual": p3.residual_target(direct)}
        cases = sorted(self.frames["direct"]["Battery_ID"].unique())
        both = P.sample_orders(cases, self.cfg["n_orders"] + self.cfg["n_tuning_orders"], 0)
        self.orders, self.tuning_orders = both[:self.cfg["n_orders"]], both[self.cfg["n_orders"]:]
        self.ckpt = FC.Checkpoint(ROOT / "data" / "cache" / "d23",
                                  [FC.Checkpoint.frame_hash(self.frames["direct"]), exp, profile, inputs])

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


def add_d21_reference(runs: pd.DataFrame, exp: str, profile: str) -> tuple[pd.DataFrame, dict]:
    """D22 comparison (b): add the committed D23 runs of residual TFT + EWC (D21 inputs) as ``REFERENCE_D21``,
    after re-computing one main and one replicate unit and checking they reproduce exactly."""
    d23 = json.loads((ROOT / "results" / f"forecasting_d23_{exp}.json").read_text())
    ref = pd.DataFrame(d23["runs"])
    ref = ref[ref["method"] == REFERENCE].assign(method=REFERENCE_D21)
    keep = [c for c in runs.columns if c in ref.columns]
    if profile != "paper":                                   # smoke: orders differ from D23's; no reproduction check
        n = runs[["set", "seed", "order_id"]].drop_duplicates().merge(ref, on=["set", "seed", "order_id"])
        return pd.concat([runs, n[keep]], ignore_index=True), {"checked": False}
    tuned = d23["tuning"]["tft_residual"]
    r = Runner(exp, profile, "d21")
    checks = []
    for set_name, seed, oid in (("main", 0, 0), ("replicate", 5, 3)):
        got = r.stream_run("tft_residual", "ewc", r.orders[oid], seed, tuned["lr"], tuned["ewc_lambda"], "test")["metrics"]
        want = ref[(ref["set"] == set_name) & (ref["seed"] == seed) & (ref["order_id"] == oid)].iloc[0]
        checks.append({"set": set_name, "seed": seed, "order_id": oid, "recomputed": got["final_avg_error"],
                       "committed": float(want["final_avg_error"])})
    ok = all(abs(c["recomputed"] - c["committed"]) < 1e-9 for c in checks)
    if not ok:
        raise SystemExit(f"committed D23 runs do not reproduce: {checks}")
    return pd.concat([runs, ref[keep]], ignore_index=True), {"checked": True, "units": checks}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp", choices=tuple(HORIZONS), required=True)
    ap.add_argument("--part", choices=PARTS)
    ap.add_argument("--profile", choices=tuple(run_p3.PROFILES), default="paper")
    ap.add_argument("--inputs", choices=("d21", "f2"), default="d21", help="f2: D22 ageing inputs added")
    args = ap.parse_args()
    r = Runner(args.exp, args.profile, args.inputs)
    print(f"{args.exp}: {len(r.frames['direct'])} S6 samples, horizons {HORIZONS[args.exp]}", flush=True)
    parts = (args.part,) if args.part else PARTS
    tuned, rows = {}, []
    for part in parts:
        tuned[part] = r.tune(part)
        rows += r.evaluate(part, tuned[part])
    if args.part:
        return 0                                                # results are collected by a run without --part
    runs, base = pd.DataFrame(rows), pd.DataFrame(r.baselines())
    repro = None
    if args.inputs == "f2":
        runs, repro = add_d21_reference(runs, args.exp, args.profile)
    hcols = [f"rmse_h{h}" for h in HORIZONS[args.exp]]
    summary = runs.groupby(["set", "method"])[[*CL_METRICS, "plasticity", *hcols, "train_time_s"]].agg(["mean", "std", "median"])
    summary.columns = ["_".join(c) for c in summary.columns]
    # D23: final error vs every method and baseline; forgetting and BWT vs the stream methods (one family);
    # in E2, the per-horizon errors vs linear extrapolation (a family of six).
    families = {"final": (list(CL_METRICS), True, list(base["method"]))}
    if args.exp == "e2":
        families["horizon"] = (hcols, False, ["linear_extrapolation"])
    stem = f"forecasting_d23_{args.exp}" if args.inputs == "d21" else f"forecasting_d22_{args.exp}_f2"
    name = stem + (".json" if args.profile == "paper" else "_smoke.json")
    X.write_results(ROOT / "results" / name, {
        "profile": args.profile, "decision": "D23" if args.inputs == "d21" else "D22", "inputs": args.inputs,
        "d21_reference_reproduction": repro, "experiment": args.exp, "horizons": list(HORIZONS[args.exp]),
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
