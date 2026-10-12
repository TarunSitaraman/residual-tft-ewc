"""Run decision D24: RC-TFT (residual TFT with shift-adaptive EWC) and the literature baselines on the S6 stream.

Setting, evaluation orders and seeds, tuning orders and rules, and the replication rule are those of D23
(``run_d23.py``, ``run_p3.py``). Parts (a part = tuning + 200 evaluation streams, each unit checkpointed under
``<data>/cache/d24``, so parts run in parallel processes and resume after an interruption):

- ``tft_residual_saewc``: RC-TFT. Learning rate = the one D23 selected for the residual TFT; lambda_0 tuned.
- ``{lstm,tcn,transformer}_{residual,direct}``: the sequence baselines, fine-tuning and EWC, tuned as in D23.

A run without ``--part`` collects: it checks that the committed D23 residual TFT + EWC reproduces on one main and one
replicate unit, fits the deterministic baselines (the D23 ones plus GPR, random forest and SVR, direct and residual),
adds the committed D23 runs, runs the registered tests and writes ``results/forecasting_d24_<exp>.json``. With
``--collect-only`` it uses only the parts that finished (a part writes ``done_<part>.json`` when complete) and lists
the others as not run.

Usage (from the repository root)::

    python scripts/run_d24.py --exp e1 --part tft_residual_saewc   # one part
    python scripts/run_d24.py --exp e1 --collect-only              # results/forecasting_d24_e1.json
    python scripts/run_d24.py --exp e1 --part lstm_residual --profile smoke   # code check only (not reportable)
    python scripts/run_d24_night.py                                # the whole one-night plan (D24 run plan)
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

import run_d23  # noqa: E402
import run_p3  # noqa: E402
from src import data_pipeline as dp  # noqa: E402
from src import evaluation as ev  # noqa: E402
from src import experiments as X  # noqa: E402
from src import forecasting as FC  # noqa: E402
from src import p3  # noqa: E402
from src import protocols as P  # noqa: E402
from src import tabular_baselines as TB  # noqa: E402
from src.models.sequence_baselines import SEQUENCE_BACKBONES, n_parameters  # noqa: E402
from src.models.tft_continual import TemporalFusionTransformer  # noqa: E402

HORIZONS = run_d23.HORIZONS
REFERENCE = "tft_residual_saewc"                 # RC-TFT
FIXED = "tft_residual_ewc"                       # D23: residual TFT + EWC with a fixed lambda (the ablation)
SEQ = tuple(SEQUENCE_BACKBONES)                  # lstm, tcn, transformer
BASELINES = "baselines"                          # deterministic and tabular baselines, one checkpointed part
PARTS = {"e1": (REFERENCE, *[f"{b}_residual" for b in SEQ], *[f"{b}_direct" for b in SEQ], BASELINES),
         "e2": (REFERENCE, *[f"{b}_residual" for b in SEQ], BASELINES)}
GPR_ROWS = {"paper": 1000, "smoke": 200}         # D24 amendment: GPR on at most this many training rows (seeded)
CL_METRICS = run_d23.CL_METRICS


def data_dir() -> Path:
    """``DATA_ROOT`` (as the notebooks) or ``data/``; synthetic test data must live outside the repository."""
    d = Path(os.environ.get("DATA_ROOT", ROOT / "data"))
    return d if d.is_absolute() else ROOT / d


class Runner(run_d23.Runner):
    def __init__(self, exp: str, profile: str) -> None:
        self.cfg, self.profile, self.exp, self.inputs = run_p3.PROFILES[profile], profile, exp, "d21"
        self.data = data_dir()
        wc = dp.read_dataset(self.data / "wheeler_checks.csv")
        direct = p3.s6_samples(wc, HORIZONS[exp])
        self.features = list(p3.SHARED_FEATURES)
        self.frames = {"direct": direct, "residual": p3.residual_target(direct)}
        cases = sorted(direct["Battery_ID"].unique())
        both = P.sample_orders(cases, self.cfg["n_orders"] + self.cfg["n_tuning_orders"], 0)   # = D23's orders
        self.orders, self.tuning_orders = both[:self.cfg["n_orders"]], both[self.cfg["n_orders"]:]
        self.ckpt = FC.Checkpoint(self.data / "cache" / "d24", [FC.Checkpoint.frame_hash(direct), exp, profile])

    # -- building blocks ----------------------------------------------------
    def settings(self, backbone: str, lr: float, extra: dict | None = None) -> dict:
        seq = run_p3.TFT if backbone in ("tft", *SEQ) else {}
        return {"lr": lr, "epochs": self.cfg["epochs"], **seq, **(extra or {})}

    def stream_run(self, part: str, method: str, order, seed: int, lr: float, lam: float | None, split: str) -> dict:
        backbone, arm = part.split("_")[:2]
        name = f"{split}_{backbone}_{arm}_{method}_{'-'.join(order)}_s{seed}"

        def fit():
            frame = self.frames[arm]
            kind = method if backbone == "mlp" else f"{backbone}_{method}"
            extra = {"ewc_lambda": lam} if method in ("ewc", "saewc") else None
            s = p3.build_strategy(kind, self.features, seed, frame, self.settings(backbone, lr, extra))
            stream = FC.forecast_stream(frame, order)
            out = p3.run_stream(s, stream, None, eval_split=split)
            final = pd.concat([getattr(t, split) for t in stream], ignore_index=True)
            pred = np.concatenate([s.predict(getattr(t, split)) for t in stream])
            res = {"metrics": out["metrics"], "horizon_rmse": run_d23.horizon_rmse(final, pred),
                   "train_time_s": float(np.sum(out["train_time_s"]))}
            if method == "saewc":
                res["shift_weights"] = s.shift_weights_
            return res
        return self.ckpt(name, fit, {"lr": lr, "lam": lam})

    def d23_tuning(self, part: str) -> dict:
        return json.loads((ROOT / "results" / f"forecasting_d23_{self.exp}.json").read_text())["tuning"][part]

    def tune(self, part: str) -> dict:
        if part != REFERENCE:
            return super().tune(part)
        lr = self.d23_tuning("tft_residual")["lr"]          # D24: the backbone learning rate D23 selected
        lam, info = self.select(lambda v, o: self.stream_run(part, "saewc", o, 0, lr, v, "val")["metrics"],
                                run_p3.LAMBDA_GRID, "lam", "final_avg_error")
        print(f"tuned {part}: lr {lr} (from D23), lambda_0 {lam} (best {info['best_value']})", flush=True)
        return {"lr": lr, "ewc_lambda": lam, "lr_source": "D23 tuning of tft_residual", "lambda_selection": info}

    def evaluate(self, part: str, tuned: dict) -> list[dict]:
        methods = ("saewc",) if part == REFERENCE else ("finetune", "ewc")
        label = (lambda m: REFERENCE) if part == REFERENCE else (lambda m: f"{part}_{m}")
        rows = []
        for set_name in ("main", "replicate"):
            for seed in self.cfg[f"{set_name}_seeds"]:
                for method in methods:
                    for oid, order in enumerate(self.orders):
                        r = self.stream_run(part, method, order, seed, tuned["lr"], tuned["ewc_lambda"], "test")
                        rows.append({"set": set_name, "method": label(method), "seed": seed, "order_id": oid,
                                     **r["metrics"], **{f"rmse_h{h}": v for h, v in r["horizon_rmse"].items()},
                                     "train_time_s": r["train_time_s"],
                                     **({"shift_weights": r["shift_weights"]} if "shift_weights" in r else {})})
            print(f"evaluated {part} {set_name}", flush=True)
        return rows

    def done_path(self, part: str) -> Path:
        return self.ckpt.dir / f"done_{part}.json"

    def run_part(self, part: str) -> tuple[dict, list[dict]]:
        if part == BASELINES:
            self.cached_baselines()
            self.done_path(part).parent.mkdir(parents=True, exist_ok=True)
            self.done_path(part).write_text("{}")
            return {}, []
        tuned = self.tune(part)
        rows = self.evaluate(part, tuned)
        self.done_path(part).parent.mkdir(parents=True, exist_ok=True)
        self.done_path(part).write_text(json.dumps(tuned, default=str))
        return tuned, rows

    # -- baselines ------------------------------------------------------------
    def cached_baselines(self) -> list[dict]:
        """``baselines`` once per experiment and profile, checkpointed (the ``baselines`` part computes it in the
        parallel phase of the night run; collection then only reads it)."""
        return self.ckpt("baselines", self.baselines, {"gpr_max_rows": GPR_ROWS[self.profile]})

    def baselines(self) -> list[dict]:
        """D23 baselines plus the D24 tabular ones (direct and residual), each with per-task test RMSE."""
        frame = self.frames["direct"]
        tasks = FC.forecast_stream(frame, self.orders[0])
        train = pd.concat([t.train for t in tasks], ignore_index=True)
        val = pd.concat([t.val for t in tasks], ignore_index=True)
        tests = {t.name: t.test for t in tasks}
        observed = [f for f in self.features if train[f].notna().any()]
        cols = {"rate": "fade_rate_pp_per_cycle", "span": "cycles_to_target",
                "schedule": ["charge_c_rate", "discharge_c_rate", "dod", "v_charge_max", "cv_charge", "cycles_to_target",
                             "days_to_target"]}
        preds = FC.baseline_predictions(train, tests, {"F0": observed, "F1": observed}, seed=0, columns=cols)
        preds = {{"ridge_F0": "ridge_shared", "gbm_F1": "gbm_shared"}.get(k, k): v for k, v in preds.items()
                 if k != "ridge_F1"}                                  # F0 = F1 here (one shared input set)
        configs = {}
        for model in TB.TABULAR_MODELS:
            for arm in ("direct", "residual"):
                out = TB.fit_predict(model, train, val, tests, self.features, arm=arm, seed=0,
                                     gpr_max_rows=GPR_ROWS[self.profile])
                preds[f"{model}_{arm}"], configs[f"{model}_{arm}"] = out["pred"], out["config"]
        final = pd.concat(tests.values(), ignore_index=True)
        rows = []
        for name, by_task in preds.items():
            per_task = {k: ev.task_error(tests[k][FC.TARGET], v, "RMSE") for k, v in by_task.items()}
            hr = run_d23.horizon_rmse(final, np.concatenate([by_task[k] for k in tests]))
            rows.append({"method": name, "final_avg_error": float(np.mean(list(per_task.values()))),
                         "final_error_per_task": per_task, **{f"rmse_h{h}": v for h, v in hr.items()},
                         **({"config": configs[name]} if name in configs else {})})
        return rows

    def parameter_counts(self) -> dict[str, int]:
        n = len(self.features)
        counts = {"tft": n_parameters(TemporalFusionTransformer(n, 8, 2, 0.1))}
        counts.update({b: n_parameters(cls(n)) for b, cls in SEQUENCE_BACKBONES.items()})
        return counts


def d23_runs(r: Runner) -> tuple[pd.DataFrame, dict]:
    """The committed D23 runs (every method), after one main and one replicate unit of residual TFT + EWC are
    recomputed here and reproduce to 1e-9 pp (paper profile only; smoke orders differ from D23's)."""
    d23 = json.loads((ROOT / "results" / f"forecasting_d23_{r.exp}.json").read_text())
    runs = pd.DataFrame(d23["runs"])
    if r.profile != "paper":
        return runs.iloc[0:0], {"checked": False, "reason": "smoke profile"}
    if [list(o) for o in r.orders] != d23["orders"]:
        raise SystemExit("evaluation orders differ from D23's")
    tuned = d23["tuning"]["tft_residual"]
    checks = []
    for set_name, seed, oid in (("main", 0, 0), ("replicate", 5, 3)):
        got = r.stream_run("tft_residual", "ewc", r.orders[oid], seed, tuned["lr"], tuned["ewc_lambda"], "test")["metrics"]
        want = runs[(runs["method"] == FIXED) & (runs["set"] == set_name) & (runs["seed"] == seed)
                    & (runs["order_id"] == oid)].iloc[0]
        checks.append({"set": set_name, "seed": seed, "order_id": oid, "recomputed": got["final_avg_error"],
                       "committed": float(want["final_avg_error"])})
    if not all(abs(c["recomputed"] - c["committed"]) < 1e-9 for c in checks):
        raise SystemExit(f"committed D23 runs do not reproduce: {checks}")
    return runs, {"checked": True, "units": checks, "d23_commit": d23["environment"].get("git_commit")}


def collect(r: Runner, parts: list[str], collect_only: bool) -> int:
    tuned, rows, not_run = {}, [], []
    for part in PARTS[r.exp]:
        if part not in parts or part == BASELINES:
            continue
        if collect_only and not r.done_path(part).exists():
            not_run.append(part)
            continue
        tuned[part], part_rows = r.run_part(part)     # cached units only when the part is done
        rows += part_rows
    if REFERENCE not in tuned:
        raise SystemExit(f"{REFERENCE} has not finished; nothing to compare")
    runs = pd.DataFrame(rows)
    ref_runs, repro = d23_runs(r)
    keep = [c for c in ref_runs.columns if c in runs.columns or c.startswith("rmse_h")]
    runs = pd.concat([runs, ref_runs[keep]], ignore_index=True)
    base = pd.DataFrame(r.cached_baselines())
    hcols = [f"rmse_h{h}" for h in HORIZONS[r.exp]]
    summary = runs.groupby(["set", "method"])[[*CL_METRICS, "plasticity", *hcols, "train_time_s"]].agg(["mean", "std", "median"])
    summary.columns = ["_".join(c) for c in summary.columns]
    families = {"primary": (["final_avg_error", "forgetting"], [FIXED], ["linear_extrapolation"]),
                "final": (list(CL_METRICS), True, list(base["method"]))}
    if r.exp == "e2":
        families["horizon"] = (hcols, False, ["linear_extrapolation"])
    name = f"forecasting_d24_{r.exp}" + (".json" if r.profile == "paper" else "_smoke.json")
    out = ROOT / "results" / name if r.profile == "paper" else r.data / "cache" / "d24" / name
    X.write_results(out, {
        "profile": r.profile, "decision": "D24", "experiment": r.exp, "horizons": list(HORIZONS[r.exp]),
        "reference": REFERENCE, "parts_run": sorted(tuned), "parts_not_run": not_run,
        "d23_reference_reproduction": repro, "parameter_counts": r.parameter_counts(),
        "n_samples": len(r.frames["direct"]), "features": r.features, "orders": [list(o) for o in r.orders],
        "tuning_orders": [list(o) for o in r.tuning_orders], "tuning": tuned, "runs": runs.to_dict("records"),
        "summary": summary.reset_index().to_dict("records"), "baselines": base.to_dict("records"),
        "tests_vs_reference": run_d23.tests(runs, base[["method", "final_avg_error", *hcols]], families,
                                            reference=REFERENCE)}, ROOT)
    print(summary[["final_avg_error_mean", "forgetting_mean"]].round(3).to_string(), flush=True)
    print(base[["method", "final_avg_error"]].round(3).to_string(), flush=True)
    print(f"written {out}" + (f"; not run: {not_run}" if not_run else ""), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp", choices=tuple(HORIZONS), required=True)
    ap.add_argument("--part", help="one part: " + ", ".join(sorted(set(PARTS["e1"]) | set(PARTS["e2"]))))
    ap.add_argument("--profile", choices=tuple(run_p3.PROFILES), default="paper")
    ap.add_argument("--collect-only", action="store_true", help="use finished parts only; list the others as not run")
    args = ap.parse_args()
    r = Runner(args.exp, args.profile)
    print(f"D24 {args.exp} ({args.profile}): {len(r.frames['direct'])} S6 samples, horizons {HORIZONS[args.exp]}",
          flush=True)
    if args.part:
        if args.part not in PARTS[args.exp]:
            raise SystemExit(f"part {args.part!r} is not in the D24 plan for {args.exp}: {PARTS[args.exp]}")
        r.run_part(args.part)
        print(f"part {args.part} done", flush=True)
        return 0
    return collect(r, list(PARTS[args.exp]), args.collect_only)


if __name__ == "__main__":
    sys.exit(main())
