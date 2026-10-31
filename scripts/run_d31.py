"""Run decision D31: online adaptation of the joint residual TFT with EWC on an unseen usage condition (P2).

Setting (D31, as D26-D28): S6 samples with the D21 shared inputs, residual target on linear extrapolation, E1 =
horizons 1-3, E2 = 1-6, seven folds, one per held-out test case; every sample of the held-out case is scored.

- **JOINT-STATIC:** the D26 joint residual TFT (six training cases, D23 lr, seeds 0-4 main, 5-9 replicate), never
  updated. Must reproduce D26 (stop condition).
- **Anchor:** theta* = its weights; F = the diagonal empirical Fisher on up to 500 seeded rows of the pooled training
  rows of the six cases (the ``EWC`` rule). Nothing else of the training cases is used for fitting afterwards.
- **Arrival:** exactly D28 (``run_d28.times``): a sample is issued at its origin check and matures at its target check.
  Whenever the matured held-out set grows, the model is fine-tuned from its current weights for E_ft epochs (Adam, D23
  lr, no early stopping) on all matured held-out samples; each sample is forecast by the model updated with every
  sample matured by its issue time.
- **FT-ONLINE:** loss = MSE. **EWC-ONLINE (proposed):** loss = MSE + (lambda / 2) sum_i F_i (theta_i - theta*_i)^2.
- **Selection** (per fold, seed 0, without the held-out case): each training case is pseudo-held-out from a joint TFT
  fitted on the other five; E_ft in {5, 20}, lambda in {0.1, 1, 10} x lambda_D23.

Parts are folds (``f<k>``, the k-th sorted case): selection, then both seed sets. Units are checkpointed under
``<data>/cache/d31``. A run without ``--part`` collects every part, checks D26 and D28, runs the registered tests and
writes ``results/forecasting_d31_<exp>.json``.

Usage (from the repository root)::

    python scripts/run_d31.py --exp e1 --count-updates           # updates per held-out case (no model)
    python scripts/run_d31.py --exp e1 --part f0
    python scripts/run_d31.py --exp e1                           # collect, check, test, write
    python scripts/run_d31.py --verdicts                         # Q1-Q3 over E1 and E2 (both files)
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
import torch
from scipy import stats
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_d25  # noqa: E402
import run_d26  # noqa: E402
import run_d27  # noqa: E402
import run_d28  # noqa: E402
from src import evaluation as ev  # noqa: E402
from src import experiments as X  # noqa: E402
from src import forecasting as FC  # noqa: E402
from src import p3  # noqa: E402
from src import protocols as P  # noqa: E402
from src.models.baseline_dnn import set_determinism  # noqa: E402

HORIZONS = run_d26.HORIZONS
SETS = run_d26.SETS
ARMS = ("joint_static", "ft_online", "ewc_online")
E_FT_GRID = (5, 20)
LAM_FACTORS = (0.1, 1.0, 10.0)
LAMBDA_D23 = {"e1": 1.0, "e2": 1000.0}
FISHER_SAMPLES = 500
MIN_DIFF = 0.05                     # pp: Q1, Q2, Q4 direction and equivalence; Q3 replication rule
TOL, NEURAL_TOL = 1e-9, 0.05        # D26 reproduction: same platform / another platform (D30 deviation)
D28_REGISTERED = {"e1": {"refit": 0.458, "static": 0.535}, "e2": {"refit": 0.898, "static": 1.051}}


# -- model pieces ----------------------------------------------------------------------

def anchor(base, pooled_train: pd.DataFrame, settings: dict, seed: int, history: pd.DataFrame):
    """An EWC strategy holding the base model's weights, scaling and windows, with one (F, theta*) term computed
    on ``pooled_train`` by the ``EWC`` rule (``fisher_samples`` seeded rows, first-task seed)."""
    s = p3.build_strategy("tft_ewc", base.features, seed, history,
                          {**settings, "ewc_lambda": 0.0, "fisher_samples": FISHER_SAMPLES})
    s.preprocessor, s.target_mean_, s.target_scale_ = base.preprocessor, base.target_mean_, base.target_scale_
    s.model = copy.deepcopy(base.model)
    s._after_task(P.Task(name="six training cases", train=pooled_train, val=pooled_train.iloc[:0],
                         test=pooled_train.iloc[:0]))
    s.model.eval()
    return s


def arm_copy(anchored, lam: float):
    """Independent copy of the anchored model for one arm; ``lam`` 0 is plain fine-tuning (no penalty)."""
    s = copy.copy(anchored)
    s.model = copy.deepcopy(anchored.model)
    s.ewc_lambda = float(lam)
    s._flat_cache = (None, -1)
    return s


def finetune(s, rows: pd.DataFrame, epochs: int, seed: int) -> None:
    """E_ft epochs of Adam from the current weights on ``rows``; no early stopping (D31 update rule)."""
    set_determinism(seed)
    gen = torch.Generator().manual_seed(seed)
    x, y = s._xy(rows)
    opt = torch.optim.Adam(s.model.parameters(), lr=s.lr, weight_decay=s.weight_decay)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        s.model.train()
        perm = torch.randperm(len(x), generator=gen)
        for i in range(0, len(x), s.batch_size):
            idx = perm[i:i + s.batch_size]
            loss = loss_fn(s.model(x[idx]), y[idx])
            if s.ewc_lambda > 0:
                loss = loss + s.penalty()
            opt.zero_grad()
            loss.backward()
            opt.step()
    s.model.eval()


def online(s, frame: pd.DataFrame, epochs: int, seed: int) -> tuple[np.ndarray, int]:
    """Forecasts under the D28 arrival rule; the model is fine-tuned whenever the matured set grows."""
    t, m = run_d28.times(frame)
    pred, known, updates = np.empty(len(frame)), 0, 0
    for ti in np.unique(t):
        matured = np.flatnonzero(m <= ti)                      # frame order (ties keep it)
        if matured.size > known:
            finetune(s, frame.iloc[matured], epochs, seed + 100_000 + updates)
            known, updates = matured.size, updates + 1
        issued = np.flatnonzero(t == ti)
        pred[issued] = s.predict(frame.iloc[issued])
    return pred, updates


def n_updates(frame: pd.DataFrame) -> int:
    t, m = run_d28.times(frame)
    counts = [int((m <= ti).sum()) for ti in np.unique(t)]
    return int(sum(b > a for a, b in zip([0] + counts, counts)))


def configs(exp: str) -> list[tuple[str, int, float]]:
    """(arm, E_ft, lambda): FT-ONLINE over E_ft; EWC-ONLINE over E_ft x lambda."""
    lams = [f * LAMBDA_D23[exp] for f in LAM_FACTORS]
    return [("ft_online", e, 0.0) for e in E_FT_GRID] + [("ewc_online", e, lam) for e in E_FT_GRID for lam in lams]


def select(scores: dict[tuple, float], arm: str) -> tuple[int, float]:
    """Lowest mean pseudo-held-out RMSE; ties: fewer epochs, then larger lambda."""
    cand = [(v, e, -lam) for (a, e, lam), v in scores.items() if a == arm]
    v, e, neg = min(cand)
    return e, -neg


# -- runner ----------------------------------------------------------------------------

class D31(run_d26.Runner):
    def __init__(self, exp: str, profile: str) -> None:
        super().__init__(exp, profile)
        self.ckpt = FC.Checkpoint(self.data / "cache" / "d31", [FC.Checkpoint.frame_hash(self.direct), exp, profile])
        self.lr = self.d23_tuning()["lr"]

    def joint(self, cases, seed: int):
        """The D26 joint residual TFT on ``cases`` (same rows, order and settings as ``Runner.joint_unit``)."""
        tasks = self.training_tasks(tuple(cases))
        s = p3.build_strategy("tft_joint", self.features, seed, self.s6, self.settings(self.lr))
        s.fit_all(tasks)
        return s, tasks

    def rows(self, case: str) -> pd.DataFrame:
        return self.s6[self.s6["Battery_ID"] == case].reset_index(drop=True)

    def selection(self, case: str) -> dict:
        """Per fold, seed 0: every training case pseudo-held-out from a joint TFT on the other five."""
        six = self.fold_order(self.cases, case)
        grid = configs(self.exp)
        per_case = {}
        for j in six:
            def fit(j=j):
                five = tuple(c for c in six if c != j)
                base, tasks = self.joint(five, 0)
                anchored = anchor(base, pd.concat([t.train for t in tasks], ignore_index=True),
                                  self.settings(self.lr), 0, self.s6)
                frame = self.rows(j)
                y = frame[FC.TARGET].to_numpy(float)
                return {f"{a}|{e}|{lam}": run_d26.rmse(online(arm_copy(anchored, lam), frame, e, 0)[0] - y)
                        for a, e, lam in grid}
            per_case[j] = self.ckpt(f"sel_{case}_{j}", fit, {"lr": self.lr, "grid": grid})
        mean = {k: float(np.mean([v[k] for v in per_case.values()])) for k in per_case[six[0]]}
        scores = {(a, int(e), float(lam)): v for k, v in mean.items() for a, e, lam in [k.split("|")]}
        ft_e, _ = select(scores, "ft_online")
        ewc_e, ewc_lam = select(scores, "ewc_online")
        return {"ft_online": {"e_ft": ft_e}, "ewc_online": {"e_ft": ewc_e, "lambda": ewc_lam},
                "mean_pseudo_rmse": mean, "per_pseudo_case": per_case}

    def unit(self, case: str, seed: int, chosen: dict) -> dict:
        """JOINT-STATIC, FT-ONLINE and EWC-ONLINE for one (fold, seed), with forgetting on the six cases."""
        def fit():
            base, tasks = self.joint(self.fold_order(self.cases, case), seed)
            frame, direct, hplus = self.held(case)
            y = frame[FC.TARGET].to_numpy(float)
            val = pd.concat([t.val for t in tasks], ignore_index=True)
            yv = val[FC.TARGET].to_numpy(float)
            anchored = anchor(base, pd.concat([t.train for t in tasks], ignore_index=True), self.settings(self.lr),
                              seed, self.s6)
            p_static = base.predict(frame)
            out = {"joint_static": {"scores": run_d26.scores(frame, direct, p_static, hplus),
                                    "forgetting_rmse": run_d26.rmse(base.predict(val) - yv), "n_updates": 0}}
            for arm, lam in (("ft_online", 0.0), ("ewc_online", chosen["ewc_online"].get("lambda", 0.0))):
                start = time.perf_counter()
                s = arm_copy(anchored, lam)
                pred, u = online(s, frame, chosen[arm]["e_ft"], seed)
                out[arm] = {"scores": run_d26.scores(frame, direct, pred, hplus),
                            "forgetting_rmse": run_d26.rmse(s.predict(val) - yv), "n_updates": u,
                            "time_s": time.perf_counter() - start}
            out["validation_rows"] = int(len(val))
            return out
        return self.ckpt(f"unit_{case}_s{seed}", fit, {"lr": self.lr, "chosen": chosen})

    def part_path(self, case: str) -> Path:
        return self.ckpt.dir / f"part_{case}.json"

    def run_part(self, part: str) -> dict:
        k = int(part.lstrip("f"))
        if k >= len(self.cases):
            raise SystemExit(f"{part}: only {len(self.cases)} test cases")
        case = self.cases[k]
        sel = self.selection(case)
        chosen = {a: sel[a] for a in ("ft_online", "ewc_online")}
        print(f"{case}: selected {chosen}", flush=True)
        units = {}
        for set_name in SETS:
            for seed in self.cfg[f"{set_name}_seeds"]:
                u = units[seed] = self.unit(case, seed, chosen)
                print(f"  {case} s{seed}: " + ", ".join(f"{a} {u[a]['scores']['rmse']:.3f}" for a in ARMS)
                      + f" | forgetting " + ", ".join(f"{u[a]['forgetting_rmse']:.3f}" for a in ARMS), flush=True)
        out = {"case": case, "selection": sel, "units": {str(s): u for s, u in units.items()},
               "n_updates": n_updates(self.held(case)[0])}
        self.part_path(case).parent.mkdir(parents=True, exist_ok=True)
        self.part_path(case).write_text(json.dumps(out, default=float))
        return out


# -- checks and tests --------------------------------------------------------------------

def d26_joint(d26: dict) -> dict[str, dict[int, float]]:
    """{case: {seed: held-out RMSE}} of the D26 joint residual TFT (one row per seed, order 0)."""
    out: dict[str, dict[int, float]] = {}
    for r in d26["runs"]:
        if r["method"] == run_d26.JOINT and r["order_id"] == 0:
            for c, v in r["per_fold"].items():
                out.setdefault(c, {})[int(r["seed"])] = float(v)
    return out


def reproduction(parts: dict[str, dict], d26: dict, other_platform: bool) -> dict:
    ref = d26_joint(d26)
    diffs = {f"{c}_s{s}": abs(u["joint_static"]["scores"]["rmse"] - ref[c][int(s)])
             for c, p in parts.items() for s, u in p["units"].items()}
    worst = max(diffs.values())
    tol = NEURAL_TOL if other_platform else TOL
    return {"abs_diff": diffs, "max_abs_diff": worst, "tolerance": tol, "other_platform": other_platform,
            "passed": bool(worst <= tol)}


def comparators(d28: dict, cases: list[str], exp: str, registered: bool) -> dict:
    by = {f["case"]: f["rmse"] for f in d28["per_fold"]}
    out = {arm: {c: float(by[c][arm]) for c in cases} for arm in ("refit", "static")}
    means = {arm: float(np.mean(list(v.values()))) for arm, v in out.items()}
    ok = all(round(means[a], 3) == D28_REGISTERED[exp][a] for a in means) if registered else True
    return {"per_fold": out, "means": means, "registered": D28_REGISTERED[exp] if registered else None, "passed": ok}


def fold_test(a: list[float], b: list[float], cases: list[str]) -> dict:
    """Two-sided exact Wilcoxon of a - b over the folds (unit: the held-out condition)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    p = 1.0 if np.all(d == 0) else float(stats.wilcoxon(d, alternative="two-sided", method="exact").pvalue)
    return {"mean_diff": float(d.mean()), "p_value": p, "n_better": int((d < 0).sum()),
            "per_fold": dict(zip(cases, map(float, d)))}


def per_fold_mean(parts: dict, cfg: dict, set_name: str, arm: str, key: str = "rmse") -> list[float]:
    seeds = [str(s) for s in cfg[f"{set_name}_seeds"]]
    return [float(np.mean([parts[c]["units"][s][arm]["scores"][key] for s in seeds])) for c in sorted(parts)]


def tests(parts: dict, cfg: dict, comp: dict) -> dict:
    cases = sorted(parts)
    out = {}
    for set_name in SETS:
        ewc = per_fold_mean(parts, cfg, set_name, "ewc_online")
        out[set_name] = {
            "q1_ewc_online_vs_d28_refit": fold_test(ewc, [comp["per_fold"]["refit"][c] for c in cases], cases),
            "q2_ewc_online_vs_joint_static": fold_test(ewc, per_fold_mean(parts, cfg, set_name, "joint_static"), cases),
            "q4_ewc_online_vs_d28_static": fold_test(ewc, [comp["per_fold"]["static"][c] for c in cases], cases)}
        q3 = {}
        for metric, get in (("heldout_rmse", lambda u, a: u[a]["scores"]["rmse"]),
                            ("forgetting_rmse", lambda u, a: u[a]["forgetting_rmse"])):
            pairs = [(get(parts[c]["units"][str(s)], "ewc_online"), get(parts[c]["units"][str(s)], "ft_online"))
                     for c in cases for s in cfg[f"{set_name}_seeds"]]
            d = np.array([a - b for a, b in pairs])
            p = 1.0 if np.all(d == 0) else float(stats.wilcoxon(d, alternative="two-sided").pvalue)
            q3[metric] = {"mean_diff": float(d.mean()), "p_value": p, "n": int(len(d))}
        holm = ev.holm_correction({k: v["p_value"] for k, v in q3.items()})
        for k in q3:
            q3[k]["p_holm"] = float(holm[k])
        out[set_name]["q3_ewc_online_vs_ft_online"] = q3
    return out


def fold_verdict(cells: list[dict]) -> str:
    """Q1/Q2 rule over E1 x E2 x {main, replicate}: all four significant with mean <= -0.05 (positive) or >= 0.05
    (negative); all four |mean| < 0.05: as accurate as; otherwise not established."""
    if all(c["p_value"] < 0.05 and c["mean_diff"] <= -MIN_DIFF for c in cells):
        return "positive"
    if all(c["p_value"] < 0.05 and c["mean_diff"] >= MIN_DIFF for c in cells):
        return "negative"
    if all(abs(c["mean_diff"]) < MIN_DIFF for c in cells):
        return "as accurate as"
    return "not established"


def verdicts(results: dict[str, dict]) -> dict:
    out = {}
    for q in ("q1_ewc_online_vs_d28_refit", "q2_ewc_online_vs_joint_static"):
        out[q] = fold_verdict([results[e]["tests"][s][q] for e in results for s in SETS])
    out["q4_ewc_online_vs_d28_static"] = "descriptive"
    for e, r in results.items():
        q3 = {m: run_d25.verdict(r["tests"]["main"]["q3_ewc_online_vs_ft_online"][m],
                                 r["tests"]["replicate"]["q3_ewc_online_vs_ft_online"][m])
              for m in ("heldout_rmse", "forgetting_rmse")}
        positive = (q3["forgetting_rmse"] == "replicated, a better" and q3["heldout_rmse"] != "replicated, a worse")
        out[f"q3_{e}"] = {**q3, "positive": positive}
    return out


def summary(parts: dict, cfg: dict, exp: str) -> dict:
    keys = ["rmse", "rmse_hplus", "rmse_hminus", *[f"rmse_h{h}" for h in HORIZONS[exp]], "class_accuracy"]
    out = {}
    for set_name in SETS:
        for arm in ARMS:
            row = {k: float(np.mean(per_fold_mean(parts, cfg, set_name, arm, k))) for k in keys}
            seeds = [str(s) for s in cfg[f"{set_name}_seeds"]]
            row["forgetting_rmse"] = float(np.mean([parts[c]["units"][s][arm]["forgetting_rmse"]
                                                    for c in parts for s in seeds]))
            row["per_fold"] = dict(zip(sorted(parts), per_fold_mean(parts, cfg, set_name, arm)))
            out[f"{set_name}_{arm}"] = row
    return out


def collect(r: D31, a) -> int:
    parts = {}
    for k in range(len(r.cases)):
        path = r.part_path(r.cases[k])
        parts[r.cases[k]] = json.loads(path.read_text()) if path.exists() else r.run_part(f"f{k}")
    d26 = json.loads((a.d26 or ROOT / "results" / f"forecasting_d26_{a.exp}.json").read_text())
    d28 = json.loads((a.d28 or ROOT / "results" / f"forecasting_d28_{a.exp}.json").read_text())
    rep = reproduction(parts, d26, a.other_platform)
    comp = comparators(d28, r.cases, a.exp, registered=r.profile == "paper" and a.d28 is None)
    print(f"reproduction check: max abs diff {rep['max_abs_diff']:.3g} (tol {rep['tolerance']}), passed "
          f"{rep['passed']}; D28 comparators {comp['means']}, passed {comp['passed']}", flush=True)
    if not (rep["passed"] and comp["passed"]):
        print("STOP: D26 or D28 not reproduced; no results file written (D31).", flush=True)
        return 3
    t = tests(parts, r.cfg, comp)
    name = f"forecasting_d31_{a.exp}" + (".json" if r.profile == "paper" else "_smoke.json")
    out = a.out or (ROOT / "results" / name if r.profile == "paper" else r.data / "cache" / "d31" / name)
    X.write_results(out, {
        "decision": "D31", "experiment": a.exp, "profile": r.profile, "horizons": list(HORIZONS[a.exp]),
        "n_samples": len(r.direct), "features": r.features, "folds": r.cases, "lr": r.lr,
        "lambda_d23": LAMBDA_D23[a.exp], "grid": configs(a.exp), "fisher_samples": FISHER_SAMPLES,
        "overrides": {"allow_dirty": a.allow_dirty, "allow_unpinned": a.allow_unpinned},
        "reproduction_check": rep, "comparators": comp, "summary": summary(parts, r.cfg, a.exp), "tests": t,
        "per_fold": parts}, ROOT)
    for k, v in summary(parts, r.cfg, a.exp).items():
        print(f"{k:24s} RMSE {v['rmse']:.3f} H+ {v['rmse_hplus']:.3f} forgetting {v['forgetting_rmse']:.3f}",
              flush=True)
    for s in SETS:
        for q, v in t[s].items():
            if q.startswith("q3"):
                print(f"{s} {q}: " + "; ".join(f"{m} {x['mean_diff']:+.3f} p_holm {x['p_holm']:.3f}"
                                               for m, x in v.items()), flush=True)
            else:
                print(f"{s} {q}: {v['mean_diff']:+.3f} pp, p {v['p_value']:.3f}, better on {v['n_better']}/7",
                      flush=True)
    print(f"written {out}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp", choices=tuple(HORIZONS))
    ap.add_argument("--part", help="f<k>: selection and both seed sets for the k-th sorted test case")
    ap.add_argument("--profile", choices=("paper", "smoke"), default="paper")
    ap.add_argument("--count-updates", action="store_true", help="updates per held-out case (no model)")
    ap.add_argument("--verdicts", action="store_true", help="Q1-Q3 verdicts from both results files")
    ap.add_argument("--d26", type=Path)
    ap.add_argument("--d28", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--other-platform", action="store_true",
                    help="D26 reproduction within the 0.05 pp neural tolerance instead of 1e-9 (recorded)")
    ap.add_argument("--allow-dirty", action="store_true", help="code check only; recorded in the results file")
    ap.add_argument("--allow-unpinned", action="store_true", help="code check only; recorded in the results file")
    a = ap.parse_args()
    if a.verdicts:
        res = {e: json.loads((ROOT / "results" / f"forecasting_d31_{e}.json").read_text()) for e in HORIZONS}
        print(json.dumps(verdicts(res), indent=1))
        return 0
    if not a.exp:
        ap.error("--exp is required")
    if a.profile == "paper" and not a.count_updates:
        problems = run_d27.preflight(a.allow_dirty, a.allow_unpinned)
        if problems:
            print("PRE-FLIGHT:\n" + "\n".join(problems), flush=True)
            return 2
    r = D31(a.exp, a.profile)
    print(f"D31 {a.exp} ({a.profile}): {len(r.direct)} S6 samples, {len(r.cases)} folds, lr {r.lr}", flush=True)
    if a.count_updates:
        for c in r.cases:
            f = r.held(c)[0]
            print(f"  {c}: {len(f)} samples, {n_updates(f)} updates", flush=True)
        return 0
    if a.part:
        r.run_part(a.part)
        return 0
    return collect(r, a)


if __name__ == "__main__":
    sys.exit(main())
