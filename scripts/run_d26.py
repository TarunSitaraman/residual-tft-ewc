"""Run decision D26: unseen usage conditions (P2) for the residual TFT + EWC, stratified by origin history.

Setting (D26): S6 samples with the D21 shared inputs, D18 purged per-cell splits, E1 = horizons 1-3, E2 = 1-6.
Seven folds, one per held-out test case; every sample of the held-out case is a test sample and is never used for
training, early stopping, shrinkage or calibration.

- **Continual model:** the D23 residual TFT + EWC (D23 learning rate and lambda, no new tuning) streamed over the six
  training cases in the order of D23 evaluation order *o* with the held-out case removed; each case's training rows
  are its train and test splits, its validation split early-stops (D18 P2 rule). After the stream, the D25 calibrator
  (``src/calibration.py``, unchanged) is fitted **once on the validation rows of all six cases pooled** (D26
  amendment, 17 Oct) and applied to the held-out case: shrunk forecast and conformal intervals at 90 % and 80 %;
  linear extrapolation gets intervals from the same rows.
- **Joint residual TFT:** all six cases at once (same rows, D23 learning rate), per seed; compared raw with the raw
  continual model.
- **Single-fit comparators** (per fold, all samples of the six cases, as D20 P2): persistence, linear extrapolation,
  ridge and gradient boosting on the D21 shared inputs.
- **Strata:** H+ = origins where linear extrapolation has a fade rate (``fade_rate_pp_per_cycle`` present, the D23
  baseline's own rate), H- = the others, where it equals persistence.

A run's P2 error is the mean over the seven folds of the held-out RMSE (pp), on all origins, H+ and H-. Parts are
``<set>_f<k>`` (seed set, fold k of the sorted cases); each unit is checkpointed under ``<data>/cache/d26``. A run
without ``--part`` collects finished parts (``--collect-only`` lists the others as not run), runs the registered
tests and writes ``results/forecasting_d26_<exp>.json``.

Usage (from the repository root)::

    python scripts/run_d26.py --exp e1 --part main_f0
    python scripts/run_d26.py --exp e1 --collect-only
    python scripts/run_d26.py --exp e1 --part main_f0 --profile smoke     # code check only (not reportable)
    python scripts/run_d26_night.py                                       # the whole one-night plan
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
import run_p3  # noqa: E402
from src import data_pipeline as dp  # noqa: E402
from src import evaluation as ev  # noqa: E402
from src import experiments as X  # noqa: E402
from src import forecasting as FC  # noqa: E402
from src import p3  # noqa: E402
from src import protocols as P  # noqa: E402
from src.calibration import HorizonCalibrator, interval_metrics  # noqa: E402

HORIZONS = run_d25.HORIZONS
N_FOLDS = 7                                         # the seven S6 test cases
SETS = ("main", "replicate")
PARTS = tuple(f"{s}_f{k}" for s in SETS for k in range(N_FOLDS))

SHR = "tft_residual_ewc_shrunk"
RAW = "tft_residual_ewc"
JOINT = "tft_residual_joint"
LIN = "linear_extrapolation"
LINC = "linear_extrapolation_conformal"
GBM = "gbm_shared"
STRATA = {"": "all", "_hplus": "hplus", "_hminus": "hminus"}

# Registered tests (D26): (a, b, metric); differences are a - b (negative favours a).
PRIMARY = [(SHR, LIN, "p2_error_hplus"), (SHR, LIN, "p2_error"), (SHR, GBM, "p2_error"), (SHR, LINC, "is90")]
SECONDARY = [(SHR, LIN, "p2_error_hminus"), (SHR, GBM, "p2_error_hminus"), (SHR, RAW, "p2_error"),
             (RAW, JOINT, "p2_error"), (SHR, LINC, "is80")]
COVERAGE_ADEQUATE = 0.85                            # mean pooled 90 % coverage, both seed sets (D26)


def rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err ** 2))) if len(err) else float("nan")


def scores(frame: pd.DataFrame, direct: pd.DataFrame, pred: np.ndarray, hplus: np.ndarray) -> dict:
    """Held-out errors of a residual-scale prediction: RMSE on all origins, H+ and H-, per horizon, and the D20
    reuse-class agreement (forecast SOH = origin SOH + lin + prediction)."""
    err = np.asarray(pred, float) - frame[FC.TARGET].to_numpy(float)
    h = frame["horizon"].to_numpy()
    fe = FC.forecast_errors(direct, frame["lin_forecast"].to_numpy(float) + pred)
    return {"rmse": rmse(err), "rmse_hplus": rmse(err[hplus]), "rmse_hminus": rmse(err[~hplus]),
            **{f"rmse_h{k}": rmse(err[h == k]) for k in sorted(np.unique(h))},
            **{k: fe[k] for k in ("class_accuracy", "class_macro_f1") if k in fe}}


def intervals(frame: pd.DataFrame, cal: HorizonCalibrator, centre_model: np.ndarray, hplus: np.ndarray) -> dict:
    """Interval metrics of the shrunk model and of linear extrapolation, all origins and H+, with the counts needed to
    pool coverage over folds."""
    y = frame[FC.TARGET].to_numpy(float)
    out = {}
    for lv in cal.levels:
        for which, centre in (("model", centre_model), ("lin", np.zeros(len(frame)))):
            q = cal.halfwidths(frame, which, lv)
            m = interval_metrics(y, centre, q, lv)
            mp = interval_metrics(y[hplus], centre[hplus], q[hplus], lv) if hplus.any() else {}
            out[f"{which}_{round(lv * 100)}"] = {**m, "n": int(len(y)), "covered": int(np.sum(np.abs(y - centre) <= q)),
                                                "hplus": mp}
    return out


class Runner:
    def __init__(self, exp: str, profile: str) -> None:
        self.cfg, self.profile, self.exp = run_p3.PROFILES[profile], profile, exp
        self.data = run_d25.data_dir()
        wc = dp.read_dataset(self.data / "wheeler_checks.csv")
        self.direct = p3.s6_samples(wc, HORIZONS[exp])                     # D23's frame
        self.s6 = p3.residual_target(self.direct)
        self.features = list(p3.SHARED_FEATURES)
        self.cases = sorted(self.direct["Battery_ID"].unique())
        both = P.sample_orders(self.cases, self.cfg["n_orders"] + self.cfg["n_tuning_orders"], 0)   # = D23's orders
        self.orders = both[:self.cfg["n_orders"]]
        self.ckpt = FC.Checkpoint(self.data / "cache" / "d26", [FC.Checkpoint.frame_hash(self.direct), exp, profile])

    # -- data ---------------------------------------------------------------------
    def held(self, case: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """(residual frame, direct frame, H+ mask) of every sample of the held-out case."""
        m = (self.direct["Battery_ID"] == case).to_numpy()
        d = self.direct[m].reset_index(drop=True)
        return self.s6[m].reset_index(drop=True), d, d["fade_rate_pp_per_cycle"].notna().to_numpy()

    def training_tasks(self, order) -> list[P.Task]:
        """One task per training case: train = its train and test splits, val = its validation split (D18 P2 rule)."""
        tasks = []
        for case in order:
            sp = FC.assign_split(self.s6[self.s6["Battery_ID"] == case])
            part = lambda keep: sp[sp["split"].isin(keep)].drop(columns="split").reset_index(drop=True)
            tasks.append(P.Task(name=case, train=part(["train", "test"]), val=part(["val"]), test=part([])))
        return tasks

    def fold_order(self, order, case: str) -> tuple[str, ...]:
        return tuple(c for c in order if c != case)

    def settings(self, lr: float, lam: float | None = None) -> dict:
        return {"lr": lr, "epochs": self.cfg["epochs"], **run_p3.TFT, **({"ewc_lambda": lam} if lam is not None else {})}

    def d23_tuning(self) -> dict:
        t = json.loads((ROOT / "results" / f"forecasting_d23_{self.exp}.json").read_text())["tuning"]["tft_residual"]
        return {"lr": t["lr"], "ewc_lambda": t["ewc_lambda"], "source": "D23 tuning of tft_residual"}

    # -- units ---------------------------------------------------------------------
    def continual_unit(self, case: str, order, seed: int, lr: float, lam: float) -> dict:
        order6 = self.fold_order(order, case)

        def fit():
            frame, direct, hplus = self.held(case)
            tasks = self.training_tasks(order6)
            s = p3.build_strategy("tft_ewc", self.features, seed, self.s6, self.settings(lr, lam))
            by_step = []
            for t in tasks:
                s.fit_task(t)
                by_step.append(rmse(s.predict(frame) - frame[FC.TARGET].to_numpy(float)))
            pooled = pd.concat([t.val for t in tasks], ignore_index=True)
            cal = HorizonCalibrator(HORIZONS[self.exp]).update(s, pooled, "pooled six cases")
            p_raw = s.predict(frame)
            p_shr = cal.shrink(p_raw, frame)
            return {"raw": scores(frame, direct, p_raw, hplus), "shrunk": scores(frame, direct, p_shr, hplus),
                    "intervals": intervals(frame, cal, p_shr, hplus), "alpha": dict(cal.alpha_),
                    "calibration_rows": int(len(pooled)), "undersized": cal.history_[-1].get("undersized", {}),
                    "heldout_rmse_by_step": by_step, "train_time_s": float(np.sum(s.training_time_))}
        return self.ckpt(f"cont_{case}_{'-'.join(order6)}_s{seed}", fit, {"lr": lr, "lam": lam})

    def joint_unit(self, case: str, seed: int, lr: float) -> dict:
        def fit():
            frame, direct, hplus = self.held(case)
            s = p3.build_strategy("tft_joint", self.features, seed, self.s6, self.settings(lr))
            s.fit_all(self.training_tasks(self.fold_order(self.cases, case)))
            return {"raw": scores(frame, direct, s.predict(frame), hplus), "train_time_s": float(np.sum(s.training_time_))}
        return self.ckpt(f"joint_{case}_s{seed}", fit, {"lr": lr})

    # -- parts ---------------------------------------------------------------------
    def done_path(self, part: str) -> Path:
        return self.ckpt.dir / f"done_{part}.json"

    def fold_case(self, part: str) -> tuple[str, str]:
        set_name, fold = part.rsplit("_f", 1)
        k = int(fold)
        if k >= len(self.cases):
            raise SystemExit(f"{part}: fold {k} but only {len(self.cases)} test cases")
        return set_name, self.cases[k]

    def run_part(self, part: str) -> dict:
        """All units of one (seed set, fold): continual streams over every order and seed, joint fits per seed."""
        if part not in PARTS:
            raise SystemExit(f"unknown part {part!r}; choose from {PARTS}")
        set_name, case = self.fold_case(part)
        t = self.d23_tuning()
        out = {"case": case, "continual": {}, "joint": {}}
        for seed in self.cfg[f"{set_name}_seeds"]:
            out["joint"][seed] = self.joint_unit(case, seed, t["lr"])
            for oid, order in enumerate(self.orders):
                out["continual"][(oid, seed)] = self.continual_unit(case, order, seed, t["lr"], t["ewc_lambda"])
        self.done_path(part).parent.mkdir(parents=True, exist_ok=True)
        self.done_path(part).write_text("{}")
        print(f"part {part} ({case}) done", flush=True)
        return out

    # -- single-fit comparators ------------------------------------------------------
    def baselines_by_fold(self) -> dict[str, dict[str, dict]]:
        """{case: {method: scores}}: persistence, linear extrapolation, ridge and gradient boosting (D21 shared
        inputs) fitted on all samples of the other six cases (D20 P2 rule), scored like the models."""
        cols = {"rate": "fade_rate_pp_per_cycle", "span": "cycles_to_target",
                "schedule": ["charge_c_rate", "discharge_c_rate", "dod", "v_charge_max", "cv_charge", "cycles_to_target",
                             "days_to_target"]}
        labels = {"persistence": "persistence", "linear_extrapolation": LIN, "ridge_F0": "ridge_shared", "gbm_F1": GBM}
        out = {}
        for case in self.cases:
            task = FC.leave_one_out_task(self.direct, case)
            fit = pd.concat([task.train, task.val], ignore_index=True)
            observed = [f for f in self.features if fit[f].notna().any()]
            preds = FC.baseline_predictions(fit, {"t": task.test}, {"F0": observed, "F1": observed}, seed=0, columns=cols)
            frame, direct, hplus = self.held(case)
            lin = frame["lin_forecast"].to_numpy(float)
            out[case] = {labels[k]: scores(frame, direct, p["t"] - lin, hplus) for k, p in preds.items() if k in labels}
        return out


# -- collection -----------------------------------------------------------------------

def run_rows(r: Runner, parts: dict[str, dict], folds: list[str]) -> pd.DataFrame:
    """One row per (set, order, seed, method) over ``folds``: P2 error = mean over the folds of the held-out RMSE,
    by stratum; pooled interval coverage over all held-out samples; per-fold values kept."""
    acc: dict = {}
    for part, out in parts.items():
        set_name, case = r.fold_case(part)
        if case not in folds:
            continue
        for (oid, seed), u in out["continual"].items():
            for method, sc in ((RAW, u["raw"]), (SHR, u["shrunk"]), (JOINT, out["joint"][seed]["raw"])):
                acc.setdefault((set_name, oid, seed, method), {})[case] = {"scores": sc}
            acc[(set_name, oid, seed, SHR)][case].update(iv={k: v for k, v in u["intervals"].items() if k.startswith("model")},
                                                         alpha=u["alpha"], train_time_s=u["train_time_s"])
            acc.setdefault((set_name, oid, seed, LINC), {})[case] = {
                "iv": {k: v for k, v in u["intervals"].items() if k.startswith("lin")}}
    rows = []
    for (set_name, oid, seed, method), folds in acc.items():
        row = {"set": set_name, "order_id": oid, "seed": seed, "method": method, "n_folds": len(folds),
               "folds": sorted(folds)}
        if method != LINC:
            for suffix, key in STRATA.items():
                row[f"p2_error{suffix}"] = float(np.nanmean([f["scores"][f"rmse{suffix}"] for f in folds.values()]))
            row["per_fold"] = {c: f["scores"]["rmse"] for c, f in folds.items()}
            row["per_fold_hplus"] = {c: f["scores"]["rmse_hplus"] for c, f in folds.items()}
            for k in HORIZONS[r.exp]:
                row[f"rmse_h{k}"] = float(np.mean([f["scores"][f"rmse_h{k}"] for f in folds.values()]))
            for k in ("class_accuracy", "class_macro_f1"):
                if all(k in f["scores"] for f in folds.values()):
                    row[k] = float(np.mean([f["scores"][k] for f in folds.values()]))
        if method in (SHR, LINC):
            which = "model" if method == SHR else "lin"
            for lv in ("90", "80"):
                iv = [f["iv"][f"{which}_{lv}"] for f in folds.values()]
                row[f"is{lv}"] = float(np.mean([v["interval_score"] for v in iv]))
                row[f"width{lv}"] = float(np.mean([v["width"] for v in iv]))
                row[f"cov{lv}"] = float(sum(v["covered"] for v in iv) / sum(v["n"] for v in iv))   # pooled
                row[f"cov{lv}_per_fold"] = {c: f["iv"][f"{which}_{lv}"]["coverage"] for c, f in folds.items()}
        if method == SHR:
            row["alpha_per_fold"] = {c: f["alpha"] for c, f in folds.items()}
            row["train_time_s"] = float(np.sum([f["train_time_s"] for f in folds.values()]))
        rows.append(row)
    return pd.DataFrame(rows)


def baseline_rows(r: Runner, by_fold: dict, cases: list[str]) -> pd.DataFrame:
    """Single-fit comparators over the same folds as the runs: one value per method."""
    rows = []
    for method in by_fold[cases[0]]:
        sc = [by_fold[c][method] for c in cases]
        row = {"method": method, **{f"p2_error{s}": float(np.nanmean([x[f"rmse{s}"] for x in sc])) for s in STRATA},
               "per_fold": {c: by_fold[c][method]["rmse"] for c in cases},
               "per_fold_hplus": {c: by_fold[c][method]["rmse_hplus"] for c in cases},
               **{f"rmse_h{k}": float(np.mean([x[f"rmse_h{k}"] for x in sc])) for k in HORIZONS[r.exp]}}
        for k in ("class_accuracy", "class_macro_f1"):
            if all(k in x for x in sc):
                row[k] = float(np.mean([x[k] for x in sc]))
        rows.append(row)
    return pd.DataFrame(rows)


def run_tests(runs: pd.DataFrame, base: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    rows = []
    for fam, comps in {"primary": PRIMARY, "secondary": SECONDARY}.items():
        for set_name in SETS:
            got = [x for c in comps if (x := run_d25.paired(runs, base, *c, set_name)) is not None]
            holm = ev.holm_correction({(x["a"], x["b"], x["metric"]): x["p_value"] for x in got})
            rows += [{**x, "family": fam, "p_holm": holm[(x["a"], x["b"], x["metric"])]} for x in got]
    verdicts = []
    t = pd.DataFrame(rows)
    if not t.empty:
        for (fam, a, b, m), g in t.groupby(["family", "a", "b", "metric"], sort=False):
            if set(g["set"]) != set(SETS):
                continue
            mn, rp = g[g["set"] == "main"].iloc[0], g[g["set"] == "replicate"].iloc[0]
            verdicts.append({"family": fam, "a": a, "b": b, "metric": m, "diff_main": mn["mean_diff"],
                             "diff_replicate": rp["mean_diff"], "p_holm_main": mn["p_holm"],
                             "p_holm_replicate": rp["p_holm"], "verdict": run_d25.verdict(mn, rp)})
    return rows, verdicts


def coverage_check(runs: pd.DataFrame) -> dict:
    """D26: coverage is adequate if the mean pooled 90 % coverage of the shrunk model is >= 0.85 in both seed sets."""
    s = runs[runs["method"] == SHR]
    by_set = {k: float(g["cov90"].mean()) for k, g in s.groupby("set")} if "cov90" in s else {}
    return {"threshold": COVERAGE_ADEQUATE, "mean_cov90": by_set,
            "adequate": bool(set(by_set) == set(SETS) and all(v >= COVERAGE_ADEQUATE for v in by_set.values()))}


def strata_counts(r: Runner) -> dict:
    out = {}
    for c in r.cases:
        _, _, hp = r.held(c)
        out[c] = {"n": int(len(hp)), "hplus": int(hp.sum()), "hminus": int((~hp).sum())}
    return out


def collect(r: Runner, collect_only: bool) -> int:
    parts, done, not_run = {}, [], []
    for part in PARTS:
        if int(part.rsplit("_f", 1)[1]) >= len(r.cases):
            continue
        if collect_only and not r.done_path(part).exists():
            not_run.append(part)
            continue
        parts[part] = r.run_part(part)
        done.append(part)
    # Runs and comparators use only the folds finished in both seed sets, so every pair covers the same folds.
    finished = {s: {r.fold_case(p)[1] for p in parts if p.startswith(s + "_")} for s in SETS}
    folds = sorted(set.intersection(*finished.values()))
    if not folds:
        raise SystemExit("no fold has finished in both seed sets; nothing to collect")
    runs = run_rows(r, parts, folds)
    by_fold = r.baselines_by_fold()
    base = baseline_rows(r, by_fold, folds)
    tests, verdicts = run_tests(runs, base)
    cov = coverage_check(runs)
    hcols = [f"rmse_h{h}" for h in HORIZONS[r.exp]]
    cols = [c for c in ("p2_error", "p2_error_hplus", "p2_error_hminus", *hcols, "is90", "cov90", "width90", "is80",
                        "cov80", "width80", "class_accuracy", "class_macro_f1", "train_time_s") if c in runs.columns]
    summary = runs.groupby(["set", "method"])[cols].agg(["mean", "std", "median"])
    summary.columns = ["_".join(c) for c in summary.columns]
    name = f"forecasting_d26_{r.exp}" + (".json" if r.profile == "paper" else "_smoke.json")
    out = ROOT / "results" / name if r.profile == "paper" else r.data / "cache" / "d26" / name
    X.write_results(out, {
        "profile": r.profile, "decision": "D26", "experiment": r.exp, "horizons": list(HORIZONS[r.exp]),
        "parts_run": done, "parts_not_run": not_run, "folds_in_tests": folds, "strata_counts": strata_counts(r),
        "n_samples": len(r.direct), "features": r.features, "orders": [list(o) for o in r.orders],
        "tuning": r.d23_tuning(), "calibration": "D25 calibrator, one update on the pooled validation rows of the six "
                                                 "training cases (D26 amendment 17 Oct)",
        "coverage_check": cov, "runs": runs.to_dict("records"), "summary": summary.reset_index().to_dict("records"),
        "baselines": base.to_dict("records"), "baselines_by_fold": by_fold, "tests": tests, "verdicts": verdicts}, ROOT)
    show = [c for c in ("p2_error_mean", "p2_error_hplus_mean", "is90_mean", "cov90_mean") if c in summary.columns]
    print(summary[show].round(3).to_string(), flush=True)
    print(base[["method", "p2_error", "p2_error_hplus"]].round(3).to_string(), flush=True)
    for v in verdicts:
        if v["family"] == "primary":
            print(f"{v['a']} - {v['b']} ({v['metric']}): {v['diff_main']:+.3f} / {v['diff_replicate']:+.3f}: "
                  f"{v['verdict']}", flush=True)
    print(f"coverage check: {cov}", flush=True)
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
    print(f"D26 {args.exp} ({args.profile}): {len(r.direct)} S6 samples, {len(r.cases)} cases, "
          f"horizons {HORIZONS[args.exp]}", flush=True)
    if args.part:
        r.run_part(args.part)
        return 0
    return collect(r, args.collect_only)


if __name__ == "__main__":
    sys.exit(main())
