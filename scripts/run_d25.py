"""Run decision D25: calibrated residual TFT + EWC (horizon-wise shrinkage, conformal intervals) and residual
pretraining on the first-life pool, on the S6 stream.

Setting, evaluation and tuning orders, seeds and the replication rule are those of D23/D24. Parts (each unit
checkpointed under ``<data>/cache/d25``, so parts run in parallel processes and resume after an interruption):

- ``scratch_main``, ``scratch_replicate``: the D23 residual TFT + EWC (D23's learning rate and lambda, no new
  tuning) on seeds 0-4 / 5-9, with the D25 calibrator. Each stream gives the raw model (must equal the committed D23
  unit to 1e-9 pp; checked for every unit at collection), the shrunk model and the intervals of the shrunk model and
  of linear extrapolation.
- ``pre_tune``: residual TFT pretrained on the S7-S9 pool (residual target, pool linear extrapolation =
  fade rate per day x days to the target), tuned as in D21 P3 (learning rate on pretrained fine-tuning streams,
  criterion plasticity; then lambda on pretrained EWC streams, criterion final average error; one-SE rule;
  grid-edge extension), validation rows only.
- ``pre_main``, ``pre_replicate``: the pretrained residual TFT + EWC with the D25 calibrator (first fitted on the
  pool's validation rows), seeds 0-4 / 5-9. Need ``pre_tune``.

A run without ``--part`` collects the finished parts (``--collect-only`` lists unfinished ones as not run), runs the
registered tests and writes ``results/forecasting_d25_<exp>.json``.

Usage (from the repository root)::

    python scripts/run_d25.py --exp e1 --part scratch_main
    python scripts/run_d25.py --exp e1 --collect-only
    python scripts/run_d25.py --exp e1 --part pre_tune --profile smoke     # code check only (not reportable)
    python scripts/run_d25_night.py                                        # the whole one-night plan
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
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
from src.calibration import HorizonCalibrator, run_stream_calibrated  # noqa: E402

HORIZONS = {"e1": (1, 2, 3), "e2": (1, 2, 3, 4, 5, 6)}
PARTS = ("scratch_main", "scratch_replicate", "pre_tune", "pre_main", "pre_replicate")
DEPENDS = {"pre_main": "pre_tune", "pre_replicate": "pre_tune"}
POOL = ("S7", "S8", "S9")

SCR_RAW = "tft_residual_ewc"                      # = the committed D23 model (reproduced unit by unit)
SCR_SHR = "tft_residual_ewc_shrunk"
PRE_RAW = "tft_pre_residual_ewc"
PRE_SHR = "tft_pre_residual_ewc_shrunk"
LIN = "linear_extrapolation"
LINC = "linear_extrapolation_conformal"          # extrapolation with intervals from the same calibration rows
CL_METRICS = ("final_avg_error", "forgetting", "backward_transfer")
EQUIV, ALPHA = 0.05, 0.05                         # D21 replication rule

# Registered tests (D25): (family, a, b, metric); differences are a - b (negative favours a).
PRIMARY = [(SCR_SHR, LIN, "final_avg_error"), (PRE_SHR, LIN, "final_avg_error"),
           (PRE_RAW, SCR_RAW, "final_avg_error"), (SCR_SHR, LINC, "is90")]
SECONDARY = [(SCR_SHR, SCR_RAW, "final_avg_error"), (SCR_SHR, SCR_RAW, "forgetting"),
             (PRE_SHR, PRE_RAW, "final_avg_error"), (PRE_SHR, PRE_RAW, "forgetting"),
             (PRE_RAW, SCR_RAW, "forgetting"), (PRE_SHR, SCR_SHR, "final_avg_error"),
             (PRE_SHR, LINC, "is90"), (SCR_SHR, LINC, "is80"), (PRE_SHR, LINC, "is80")]


def data_dir() -> Path:
    d = Path(os.environ.get("DATA_ROOT", ROOT / "data"))
    return d if d.is_absolute() else ROOT / d


def pool_residual_frame(data: Path, horizons) -> pd.DataFrame:
    """Pool samples (S7-S9, D21 shared inputs) on the residual scale. The pool has no per-cycle fade rate for its
    calendar cells, so its linear extrapolation is the D21 per-day fade rate (last three checks) times the days to
    the target, 0 where the rate is missing (as on S6)."""
    frames = [p3.harmonised_samples(dp.read_dataset(data / f"pool_{d.lower()}_checks.csv")
                                    .assign(check_time=lambda t: pd.to_datetime(t["check_time"])), d, horizons)
              for d in POOL]
    s = pd.concat(frames, ignore_index=True)
    s["lin_forecast"] = np.nan_to_num(s["fade_rate_pp_per_day"].to_numpy(float) * s["days_to_target"].to_numpy(float))
    return p3.residual_target(s)


def per_task(R: pd.DataFrame) -> dict:
    return {t: float(R.iloc[-1][t]) for t in R.columns}


class Runner(run_p3.Runner):
    """D25 runner; reuses the D21 selection code (``one_se``, ``select``) of the P3 runner."""

    def __init__(self, exp: str, profile: str) -> None:
        self.cfg, self.profile, self.exp = run_p3.PROFILES[profile], profile, exp
        self.data = data_dir()
        wc = dp.read_dataset(self.data / "wheeler_checks.csv")
        self.direct = p3.s6_samples(wc, HORIZONS[exp])                    # exactly D23's frame
        self.s6 = p3.residual_target(self.direct)
        self.features = list(p3.SHARED_FEATURES)
        cases = sorted(self.direct["Battery_ID"].unique())
        both = P.sample_orders(cases, self.cfg["n_orders"] + self.cfg["n_tuning_orders"], 0)   # = D23's orders
        self.orders, self.tuning_orders = both[:self.cfg["n_orders"]], both[self.cfg["n_orders"]:]
        self.ckpt = FC.Checkpoint(self.data / "cache" / "d25", [FC.Checkpoint.frame_hash(self.direct), exp, profile])
        self._pool = None

    # -- data ---------------------------------------------------------------------
    def pool(self) -> tuple[pd.DataFrame, P.Task, pd.DataFrame]:
        """(pool frame, pool task, window history = pool + S6), built on first use (the scratch parts never need it)."""
        if self._pool is None:
            frame = pool_residual_frame(self.data, HORIZONS[self.exp])
            self._pool = (frame, p3.pool_task(frame), pd.concat([frame, self.s6], ignore_index=True))
        return self._pool

    def settings(self, lr: float, lam: float | None = None) -> dict:
        return {"lr": lr, "epochs": self.cfg["epochs"], **run_p3.TFT, **({"ewc_lambda": lam} if lam is not None else {})}

    def d23_tuning(self) -> dict:
        return json.loads((ROOT / "results" / f"forecasting_d23_{self.exp}.json").read_text())["tuning"]["tft_residual"]

    # -- units -------------------------------------------------------------------
    @staticmethod
    def _slim(out: dict) -> dict:
        return {"raw": {"metrics": out["raw"]["metrics"], "horizon_rmse": out["raw"]["horizon_rmse"],
                        "final_per_task": per_task(out["raw"]["R"])},
                "shrunk": {"metrics": out["shrunk"]["metrics"], "horizon_rmse": out["shrunk"]["horizon_rmse"],
                           "final_per_task": per_task(out["shrunk"]["R"])},
                "intervals": out["intervals"], "calibration": out["calibration"],
                "train_time_s": float(np.sum(out["train_time_s"]))}

    def scratch_unit(self, order, seed: int, lr: float, lam: float) -> dict:
        def fit():
            s = p3.build_strategy("tft_ewc", self.features, seed, self.s6, self.settings(lr, lam))
            return self._slim(run_stream_calibrated(s, FC.forecast_stream(self.s6, order),
                                                    HorizonCalibrator(HORIZONS[self.exp]), "test"))
        return self.ckpt(f"test_scratch_{'-'.join(order)}_s{seed}", fit, {"lr": lr, "lam": lam})

    def pretrained(self, lr: float, seed: int):
        """Residual TFT + EWC pretrained on the pool (Fisher and anchor from the pool); checkpointed per (lr, seed)."""
        frame, task, history = self.pool()

        def fit():
            s = p3.build_strategy("tft_ewc", self.features, seed, history, self.settings(lr))
            t = time.perf_counter()
            p3.pretrain(s, task)
            s.pretrain_time_s_ = time.perf_counter() - t
            s.windows = None                                        # rebuilt from the history on load
            return s
        s = self.ckpt(f"pretrain_s{seed}", fit, {"lr": lr})
        s.set_history(history)
        return s

    def pre_strategy(self, method: str, lr: float, seed: int, lam: float | None):
        """As ``run_p3.Runner.strategy`` for a pretrained start: the pretrained network, frozen pool preprocessing and
        target scaling, and (EWC) the pool Fisher and anchor, under this method's settings."""
        pre = self.pretrained(lr, seed)
        s = p3.build_strategy(f"tft_{method}", self.features, seed, self.pool()[2],
                              self.settings(lr, lam if method == "ewc" else None))
        s.preprocessor, s.target_mean_, s.target_scale_ = pre.preprocessor, pre.target_mean_, pre.target_scale_
        s.model = copy.deepcopy(pre.model)
        s.tasks_seen_ = list(pre.tasks_seen_)
        if method == "ewc":
            s.fishers_, s.optima_ = copy.deepcopy(pre.fishers_), copy.deepcopy(pre.optima_)
        return s

    def pre_val_unit(self, method: str, order, lr: float, lam: float | None) -> dict:
        def fit():
            s = self.pre_strategy(method, lr, 0, lam)
            return {"metrics": p3.run_stream(s, FC.forecast_stream(self.s6, order), None, eval_split="val")["metrics"]}
        return self.ckpt(f"val_pre_{method}_{'-'.join(order)}_s0", fit, {"lr": lr, "lam": lam})

    def pre_unit(self, order, seed: int, lr: float, lam: float) -> dict:
        def fit():
            s = self.pre_strategy("ewc", lr, seed, lam)
            cal = HorizonCalibrator(HORIZONS[self.exp]).update(s, self.pool()[1].val, "pool")
            out = self._slim(run_stream_calibrated(s, FC.forecast_stream(self.s6, order), cal, "test"))
            out["pretrain_time_s"] = float(getattr(self.pretrained(lr, seed), "pretrain_time_s_", np.nan))
            return out
        return self.ckpt(f"test_pre_{'-'.join(order)}_s{seed}", fit, {"lr": lr, "lam": lam})

    # -- parts --------------------------------------------------------------------
    def done_path(self, part: str) -> Path:
        return self.ckpt.dir / f"done_{part}.json"

    def tune_pre(self) -> dict:
        lr, lr_info = self.select(lambda v, o: self.pre_val_unit("finetune", o, v, None)["metrics"],
                                  run_p3.LR_GRID, "lr", "plasticity")
        lam, lam_info = self.select(lambda v, o: self.pre_val_unit("ewc", o, lr, v)["metrics"],
                                    run_p3.LAMBDA_GRID, "lam", "final_avg_error")
        print(f"tuned pretrained residual TFT: lr {lr} (best {lr_info['best_value']}), lambda {lam} "
              f"(best {lam_info['best_value']})", flush=True)
        frame, task, _ = self.pool()
        return {"lr": lr, "ewc_lambda": lam, "lr_selection": lr_info, "lambda_selection": lam_info,
                "pool_samples": len(frame), "pool_cells": int(frame["cell"].nunique()),
                "pool_split_sizes": {k: len(getattr(task, k)) for k in ("train", "val", "test")}}

    def tuned(self, part: str) -> dict:
        if part.startswith("scratch"):
            t = self.d23_tuning()
            return {"lr": t["lr"], "ewc_lambda": t["ewc_lambda"], "source": "D23 tuning of tft_residual"}
        path = self.done_path("pre_tune")
        if not path.exists():
            raise SystemExit(f"{part} needs pre_tune to have finished ({path} missing)")
        return json.loads(path.read_text())

    def evaluate(self, part: str) -> list[dict]:
        kind, set_name = part.split("_")
        t = self.tuned(part)
        rows = []
        for seed in self.cfg[f"{set_name}_seeds"]:
            for oid, order in enumerate(self.orders):
                if kind == "scratch":
                    u = self.scratch_unit(order, seed, t["lr"], t["ewc_lambda"])
                    names = {"raw": SCR_RAW, "shrunk": SCR_SHR}
                else:
                    u = self.pre_unit(order, seed, t["lr"], t["ewc_lambda"])
                    names = {"raw": PRE_RAW, "shrunk": PRE_SHR}
                key = {"set": set_name, "seed": seed, "order_id": oid, "part": part}
                for arm, method in names.items():
                    iv = {}
                    if arm == "shrunk":
                        for tag, v in u["intervals"].items():
                            if tag.startswith("model_"):
                                lv = tag.split("_")[1]
                                iv.update({f"is{lv}": v["interval_score"], f"cov{lv}": v["coverage"],
                                           f"width{lv}": v["width"]})
                    rows.append({**key, "method": method, **u[arm]["metrics"],
                                 **{f"rmse_h{h}": e for h, e in u[arm]["horizon_rmse"].items()},
                                 "final_per_task": u[arm]["final_per_task"], "train_time_s": u["train_time_s"],
                                 **iv, **({"alpha_final": u["calibration"][-1].get("alpha")} if arm == "shrunk" else {})})
                if kind == "scratch":                          # extrapolation intervals from the same rows (one per run)
                    iv = {}
                    for tag, v in u["intervals"].items():
                        if tag.startswith("lin_"):
                            lv = tag.split("_")[1]
                            iv.update({f"is{lv}": v["interval_score"], f"cov{lv}": v["coverage"], f"width{lv}": v["width"]})
                    rows.append({**key, "method": LINC, **iv})
        print(f"evaluated {part}", flush=True)
        return rows

    def run_part(self, part: str) -> list[dict]:
        if part not in PARTS:
            raise SystemExit(f"unknown part {part!r}; choose from {PARTS}")
        self.done_path(part).parent.mkdir(parents=True, exist_ok=True)
        if part == "pre_tune":
            if not self.done_path(part).exists():
                self.done_path(part).write_text(json.dumps(self.tune_pre(), default=str))
            return []
        rows = self.evaluate(part)
        self.done_path(part).write_text("{}")
        return rows

    # -- baselines ----------------------------------------------------------------
    def baselines(self) -> list[dict]:
        """Linear extrapolation and persistence (deterministic) on the S6 test splits: task-mean and per-horizon RMSE."""
        tasks = FC.forecast_stream(self.s6, self.orders[0])
        final = pd.concat([t.test for t in tasks], ignore_index=True)
        out = []
        for name, pred in ((LIN, lambda f: np.zeros(len(f))), ("persistence", lambda f: -f["lin_forecast"].to_numpy(float))):
            errs = {t.name: ev.task_error(t.test[FC.TARGET], pred(t.test), "RMSE") for t in tasks}
            err = np.concatenate([pred(t.test) for t in tasks]) - final[FC.TARGET].to_numpy(float)
            h = final["horizon"].to_numpy()
            out.append({"method": name, "final_avg_error": float(np.mean(list(errs.values()))), "final_per_task": errs,
                        **{f"rmse_h{k}": float(np.sqrt(np.mean(err[h == k] ** 2))) for k in sorted(np.unique(h))}})
        return out


# -- tests ------------------------------------------------------------------------------------------------

def paired(runs: pd.DataFrame, base: pd.DataFrame, a: str, b: str, metric: str, set_name: str) -> dict | None:
    key = ["order_id", "seed"]
    g = runs[runs["set"] == set_name]
    ra = g[g["method"] == a].set_index(key)
    if ra.empty or metric not in ra or ra[metric].isna().all():
        return None
    if b in set(base["method"]):
        if metric not in base.columns:
            return None
        vb = np.full(len(ra), float(base.loc[base["method"] == b, metric].iloc[0]))
    else:
        rb = g[g["method"] == b].set_index(key)
        if rb.empty or metric not in rb:
            return None
        vb = rb[metric].reindex(ra.index).to_numpy(float)
    va = ra[metric].to_numpy(float)
    return {"set": set_name, "a": a, "b": b, "metric": metric, **ev.paired_comparison(va, vb)}


def run_tests(runs: pd.DataFrame, base: pd.DataFrame, exp: str) -> tuple[list[dict], list[dict]]:
    families = {"primary": PRIMARY, "secondary": SECONDARY}
    if exp == "e2":
        hs = [f"rmse_h{h}" for h in HORIZONS[exp]]
        families["horizon_scratch"] = [(SCR_SHR, LIN, m) for m in hs]
        families["horizon_pre"] = [(PRE_SHR, LIN, m) for m in hs]
    rows = []
    for fam, comps in families.items():
        for set_name in ("main", "replicate"):
            got = [r for c in comps if (r := paired(runs, base, *c, set_name)) is not None]
            holm = ev.holm_correction({(r["a"], r["b"], r["metric"]): r["p_value"] for r in got})
            rows += [{**r, "family": fam, "p_holm": holm[(r["a"], r["b"], r["metric"])]} for r in got]
    verdicts = []
    t = pd.DataFrame(rows)
    if not t.empty:
        for (fam, a, b, m), g in t.groupby(["family", "a", "b", "metric"], sort=False):
            if set(g["set"]) != {"main", "replicate"}:
                continue
            mn, rp = g[g["set"] == "main"].iloc[0], g[g["set"] == "replicate"].iloc[0]
            verdicts.append({"family": fam, "a": a, "b": b, "metric": m, "diff_main": mn["mean_diff"],
                             "diff_replicate": rp["mean_diff"], "p_holm_main": mn["p_holm"],
                             "p_holm_replicate": rp["p_holm"], "verdict": verdict(mn, rp)})
    return rows, verdicts


def verdict(main, rep) -> str:
    """D21 replication rule: significant after Holm in both seed sets, same sign, |d| >= 0.05 in both = replicated;
    |d| < 0.05 in both = practically equivalent; otherwise not replicated."""
    dm, dr = float(main["mean_diff"]), float(rep["mean_diff"])
    if np.isnan(dm) or np.isnan(dr):
        return "not testable"
    if abs(dm) < EQUIV and abs(dr) < EQUIV:
        return "practically equivalent"
    pm, pr = float(main["p_holm"]), float(rep["p_holm"])
    if pm < ALPHA and pr < ALPHA and np.sign(dm) == np.sign(dr) and min(abs(dm), abs(dr)) >= EQUIV:
        return "replicated, " + ("a better" if dm < 0 else "a worse")
    return "not replicated"


def reproduction_check(r: Runner, runs: pd.DataFrame) -> dict:
    """Every raw scratch unit must equal the committed D23 residual TFT + EWC unit to 1e-9 pp (paper profile)."""
    if r.profile != "paper":
        return {"checked": False, "reason": "smoke profile"}
    d23 = json.loads((ROOT / "results" / f"forecasting_d23_{r.exp}.json").read_text())
    if [list(o) for o in r.orders] != d23["orders"]:
        raise SystemExit("evaluation orders differ from D23's")
    ref = pd.DataFrame(d23["runs"])
    ref = ref[ref["method"] == SCR_RAW].set_index(["set", "seed", "order_id"])["final_avg_error"]
    got = runs[runs["method"] == SCR_RAW].set_index(["set", "seed", "order_id"])["final_avg_error"]
    if got.empty:
        return {"checked": False, "reason": "no scratch part finished"}
    diff = (got - ref.reindex(got.index)).abs()
    out = {"checked": True, "units": int(len(got)), "max_abs_diff": float(diff.max()),
           "d23_commit": d23["environment"].get("git_commit")}
    if diff.isna().any() or diff.max() > 1e-9:
        raise SystemExit(f"raw scratch units do not reproduce the committed D23 runs: {out}")
    return out


def collect(r: Runner, collect_only: bool) -> int:
    rows, done, not_run = [], [], []
    for part in PARTS:
        if collect_only and not r.done_path(part).exists():
            not_run.append(part)
            continue
        rows += r.run_part(part)
        done.append(part)
    runs = pd.DataFrame(rows)
    if runs.empty:
        raise SystemExit("no evaluation part has finished; nothing to collect")
    repro = reproduction_check(r, runs)
    base = pd.DataFrame(r.baselines())
    hcols = [f"rmse_h{h}" for h in HORIZONS[r.exp]]
    cols = [c for c in (*CL_METRICS, "plasticity", *hcols, "is90", "cov90", "width90", "is80", "cov80", "width80",
                        "train_time_s") if c in runs.columns]
    summary = runs.groupby(["set", "method"])[cols].agg(["mean", "std", "median"])
    summary.columns = ["_".join(c) for c in summary.columns]
    tests, verdicts = run_tests(runs, base, r.exp)
    tuned = {}
    if any(p.startswith("scratch") for p in done):
        tuned["scratch"] = r.tuned("scratch_main")
    if "pre_tune" in done:
        tuned["pre"] = r.tuned("pre_main")
    name = f"forecasting_d25_{r.exp}" + (".json" if r.profile == "paper" else "_smoke.json")
    out = ROOT / "results" / name if r.profile == "paper" else r.data / "cache" / "d25" / name
    X.write_results(out, {
        "profile": r.profile, "decision": "D25", "experiment": r.exp, "horizons": list(HORIZONS[r.exp]),
        "parts_run": done, "parts_not_run": not_run, "d23_reference_reproduction": repro,
        "n_samples": len(r.direct), "features": r.features, "orders": [list(o) for o in r.orders],
        "tuning_orders": [list(o) for o in r.tuning_orders], "tuning": tuned, "runs": runs.to_dict("records"),
        "summary": summary.reset_index().to_dict("records"), "baselines": base.to_dict("records"),
        "tests": tests, "verdicts": verdicts}, ROOT)
    show = [c for c in ("final_avg_error_mean", "forgetting_mean", "is90_mean", "cov90_mean") if c in summary.columns]
    print(summary[show].round(3).to_string(), flush=True)
    print(base[["method", "final_avg_error"]].round(3).to_string(), flush=True)
    for v in verdicts:
        if v["family"] == "primary":
            print(f"{v['a']} - {v['b']} ({v['metric']}): {v['diff_main']:+.3f} / {v['diff_replicate']:+.3f}: "
                  f"{v['verdict']}", flush=True)
    print(f"written {out}" + (f"; not run: {not_run}" if not_run else ""), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp", choices=tuple(HORIZONS), required=True)
    ap.add_argument("--part", choices=PARTS)
    ap.add_argument("--profile", choices=tuple(run_p3.PROFILES), default="paper")
    ap.add_argument("--collect-only", action="store_true", help="use finished parts only; list the others as not run")
    args = ap.parse_args()
    r = Runner(args.exp, args.profile)
    print(f"D25 {args.exp} ({args.profile}): {len(r.direct)} S6 samples, horizons {HORIZONS[args.exp]}", flush=True)
    if args.part:
        r.run_part(args.part)
        print(f"part {args.part} done", flush=True)
        return 0
    return collect(r, args.collect_only)


if __name__ == "__main__":
    sys.exit(main())
