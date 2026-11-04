"""Run decision D34: unseen usage conditions on the KIT first-life pool (S7), tests over conditions.

Setting (docs/DECISIONS.md, D34): S7 samples with the D21 shared inputs, one task per ageing condition, D18 purged
per-cell splits, E1 = horizons 1-3, E2 = 1-6 check-ups. The 67 conditions are split once (seed 34) into 7 tuning
conditions (never trained on in an evaluation stream, never tested) and 60 test conditions in five outer folds of 12.

Per outer fold (the other 48 test conditions are the training conditions; train + test splits train, the validation
split early-stops, D18 P2 rule):
- **Tuning** (parts ``tune1_f<k>``, ``tune2_f<k>``): streams over the 48 conditions in five tuning orders, scored
  zero-shot on the seven tuning conditions (mean of per-condition RMSE, raw model). Stage 1 learning rate with
  fine-tuning, stage 2 EWC lambda at that rate; D21 one-standard-error rule and D11 edge rule (``run_p3``).
- **Evaluation** (parts ``<set>_f<k>``): the residual TFT + EWC streamed over the 48 conditions (5 orders x 3 seeds per
  seed set), shrunk with the D25 calibrator fitted once on the pooled validation rows of the 48 conditions; the joint
  residual TFT per seed.
- **Single fits** (at collection): GBM-ALL (all 48 conditions' training rows), GBM-LAST (the last condition of each
  evaluation order), linear extrapolation (D21 per-day rate x days to target), persistence, ridge.

Tests are over the held-out conditions: per condition and seed set, the RMSE averaged over runs; paired Wilcoxon over
conditions, Holm within family and seed set, D21 replication rule (``run_d25.verdict``).

Usage (from the repository root)::

    python scripts/run_d34.py --write-sets                     # results/d34_condition_sets.csv (commit before runs)
    python scripts/run_d34.py --exp e1 --part tune1_f0
    python scripts/run_d34.py --exp e1 --collect-only
    python scripts/run_d34_night.py                            # the whole plan
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

import run_d25  # noqa: E402
import run_d26  # noqa: E402
import run_p3  # noqa: E402
from src import data_pipeline as dp  # noqa: E402
from src import evaluation as ev  # noqa: E402
from src import experiments as X  # noqa: E402
from src import forecasting as FC  # noqa: E402
from src import p3  # noqa: E402
from src import protocols as P  # noqa: E402
from src.calibration import HorizonCalibrator, interval_metrics  # noqa: E402

HORIZONS = run_d25.HORIZONS
SET_SEED, N_TUNING, N_FOLDS = 34, 7, 5
SETS = ("main", "replicate")
TYPES = {1: "calendar", 2: "cyclic", 3: "wltp"}
PROFILES = {
    "paper": {"n_orders": 5, "main_seeds": (0, 1, 2), "replicate_seeds": (5, 6, 7), "tuning_runs": 5, "epochs": 200,
              "lr_grid": (1e-3, 3e-3, 1e-2), "lam_grid": (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)},
    "smoke": {"n_orders": 1, "main_seeds": (0,), "replicate_seeds": (5,), "tuning_runs": 2, "epochs": 2,   # 2: one-SE needs an s.e.
              "lr_grid": (3e-3, 1e-2), "lam_grid": (1.0, 10.0)},
}
PARTS = tuple([f"tune1_f{k}" for k in range(N_FOLDS)] + [f"tune2_f{k}" for k in range(N_FOLDS)]
              + [f"{s}_f{k}" for s in SETS for k in range(N_FOLDS)])
DEPENDS = {**{f"tune2_f{k}": f"tune1_f{k}" for k in range(N_FOLDS)},
           **{f"{s}_f{k}": f"tune2_f{k}" for s in SETS for k in range(N_FOLDS)}}

SHR, RAW, JOINT = "tft_residual_ewc_shrunk", "tft_residual_ewc", "tft_residual_joint"
LAST, ALL, LIN, PERS, RIDGE = "gbm_last", "gbm_all", "linear_extrapolation", "persistence", "ridge_shared"
PRIMARY = [(SHR, LAST, "rmse"), (SHR, LIN, "rmse_hplus"), (SHR, ALL, "rmse")]
SECONDARY = [(SHR, LAST, "rmse_hplus"), (SHR, LAST, "rmse_hminus"), (SHR, ALL, "rmse_hplus"),
             (SHR, ALL, "rmse_hminus"), (SHR, LIN, "rmse"), (SHR, RAW, "rmse"), (RAW, JOINT, "rmse"),
             (LAST, ALL, "rmse")]
COVERAGE_ADEQUATE = 0.85


# -- condition sets (fixed once, D34) ---------------------------------------------------------------------------------

def condition_table(checks: pd.DataFrame) -> pd.DataFrame:
    """One row per condition with at least one cell of >= 5 valid check-ups, in check-table order."""
    c = checks[checks["n_valid_checks"] >= 5]
    return (c.groupby("condition", sort=False)
             .agg(age_type=("age_type", "first"), age_temp=("age_temp", "first"), cells=("cell", "nunique"))
             .reset_index())


def assign_sets(conds: pd.DataFrame, n_tuning: int = N_TUNING, n_folds: int = N_FOLDS,
                seed: int = SET_SEED, keys: tuple[str, ...] = ("age_type", "age_temp")) -> pd.DataFrame:
    """Tuning conditions: n_tuning drawn per ageing type in proportion to its count (largest remainder). Test
    conditions: sorted by (type, temperature, random key) and dealt round-robin into n_folds (stratified)."""
    rng = np.random.default_rng(seed)
    out = conds.assign(u=rng.random(len(conds)), set="test", fold=-1)
    counts = out["age_type"].value_counts().sort_index()
    quota = counts * n_tuning / len(out)
    k = np.floor(quota).astype(int)
    for t in (quota - k).sort_values(ascending=False, kind="stable").index[: n_tuning - int(k.sum())]:
        k[t] += 1
    for t, n in k.items():
        idx = out[out["age_type"] == t].sort_values("u").index[:n]
        out.loc[idx, "set"] = "tuning"
    test = out[out["set"] == "test"].sort_values([*keys, "u"])
    out.loc[test.index, "fold"] = np.arange(len(test)) % n_folds
    return out.drop(columns="u")


# -- runner -----------------------------------------------------------------------------------------------------------

class Runner:
    """D34 on S7. A later decision on another dataset (D36) subclasses it and overrides the class attributes and the
    data hooks (``read_checks``, ``samples``, ``make_sets``, ``unit_scores``); D34's own behaviour is unchanged."""
    NAME, DECISION, CHECKS = "d34", "D34", "pool_s7_checks.csv"
    CONT = {RAW: "raw", SHR: "shrunk"}                  # continual methods: name -> key in a continual unit
    PRIMARY, SECONDARY = PRIMARY, SECONDARY
    DESC_KEYS = ("age_type", "age_temp")

    def __init__(self, exp: str, profile: str) -> None:
        self.cfg, self.exp, self.profile = PROFILES[profile], exp, profile
        self.data = run_d25.data_dir()
        checks = self.read_checks()
        s, self.features = self.samples(checks, HORIZONS[exp])
        s["Battery_ID"] = s["condition"]                                       # task = condition
        s["lin_forecast"] = np.nan_to_num(s["fade_rate_pp_per_day"].to_numpy(float) * s["days_to_target"].to_numpy(float))
        self.direct, self.res = s.reset_index(drop=True), p3.residual_target(s).reset_index(drop=True)
        self.sets = self.make_sets(checks)
        self.tuning = list(self.sets.loc[self.sets["set"] == "tuning", "condition"])
        self.ckpt = FC.Checkpoint(self.data / "cache" / self.NAME, [FC.Checkpoint.frame_hash(self.direct), exp, profile])

    # -- dataset hooks --
    @classmethod
    def read_checks(cls) -> pd.DataFrame:
        return dp.read_dataset(run_d25.data_dir() / cls.CHECKS).assign(check_time=lambda t: pd.to_datetime(t["check_time"]))

    @staticmethod
    def samples(checks: pd.DataFrame, horizons) -> tuple[pd.DataFrame, list[str]]:
        return p3.harmonised_samples(checks, "S7", horizons), list(p3.SHARED_FEATURES)

    @staticmethod
    def make_sets(checks: pd.DataFrame) -> pd.DataFrame:
        return assign_sets(condition_table(checks))

    @staticmethod
    def unit_scores(frame, direct, raw, shr, hp) -> dict:
        return {"raw": run_d26.scores(frame, direct, raw, hp), "shrunk": run_d26.scores(frame, direct, shr, hp)}

    # -- data --
    def fold_conditions(self, k: int) -> tuple[list[str], list[str]]:
        """(held-out conditions, training conditions in check-table order) of outer fold k."""
        t = self.sets[self.sets["set"] == "test"]
        return list(t.loc[t["fold"] == k, "condition"]), list(t.loc[t["fold"] != k, "condition"])

    def orders(self, k: int) -> list[tuple[str, ...]]:
        return P.sample_orders(self.fold_conditions(k)[1], self.cfg["n_orders"], 0)

    def tuning_orders(self, k: int) -> list[tuple[str, ...]]:
        """Seed-1 permutations (the first, table order, is dropped), disjoint from the evaluation orders."""
        got = P.sample_orders(self.fold_conditions(k)[1], self.cfg["tuning_runs"] + 1, 1)[1:]
        assert not set(got) & set(self.orders(k)), "tuning and evaluation orders overlap"
        return got

    def held(self, cond: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        m = (self.direct["Battery_ID"] == cond).to_numpy()
        d = self.direct[m].reset_index(drop=True)
        return self.res[m].reset_index(drop=True), d, d["fade_rate_pp_per_day"].notna().to_numpy()

    def task(self, cond: str) -> P.Task:
        sp = FC.assign_split(self.res[self.res["Battery_ID"] == cond])
        part = lambda keep: sp[sp["split"].isin(keep)].drop(columns="split").reset_index(drop=True)
        return P.Task(name=cond, train=part(["train", "test"]), val=part(["val"]), test=part([]))

    def settings(self, lr: float, lam: float | None = None) -> dict:
        return {"lr": lr, "epochs": self.cfg["epochs"], **run_p3.TFT, **({"ewc_lambda": lam} if lam is not None else {})}

    def stream(self, method: str, order, seed: int, lr: float, lam: float | None = None):
        s = p3.build_strategy(method, self.features, seed, self.res, self.settings(lr, lam))
        tasks = [self.task(c) for c in order]
        for t in tasks:
            s.fit_task(t)
        return s, tasks

    # -- tuning --
    def tuning_unit(self, k: int, oid: int, method: str, lr: float, lam: float | None) -> dict:
        order = self.tuning_orders(k)[oid]

        def fit():
            s, _ = self.stream(method, order, 0, lr, lam)
            per = {c: run_d26.rmse(s.predict(self.held(c)[0]) - self.held(c)[0][FC.TARGET].to_numpy(float))
                   for c in self.tuning}
            return {"criterion": float(np.mean(list(per.values()))), "per_condition": per}
        return self.ckpt(f"tune_f{k}_o{oid}_{method}", fit, {"lr": lr, "lam": lam})

    def select(self, k: int, method: str, grid, param: str, fixed: dict) -> tuple[float, dict]:
        """D21 one-SE selection with the D11 edge rule (at most two extensions), as ``run_p3.Runner.select``."""
        grid, ext = sorted(grid), []
        for _ in range(3):
            rows = [{param: v, "order": o, **self.tuning_unit(k, o, method, **{**fixed, param: v})}
                    for v in grid for o in range(self.cfg["tuning_runs"])]
            chosen, info = run_p3.Runner.one_se(pd.DataFrame(rows).drop(columns="per_condition"), param, "criterion")
            best = info["best_value"]
            if best not in (grid[0], grid[-1]) or len(ext) == 2:
                break
            new = run_p3.ladder(grid[0], -1, param) if best == grid[0] else run_p3.ladder(grid[-1], +1, param)
            ext.append({"edge": "low" if best == grid[0] else "high", "added": new})
            grid = sorted(grid + new)
        info.update(grid=grid, extensions=ext, best_at_final_grid_edge=info["best_value"] in (grid[0], grid[-1]))
        return chosen, info

    def tuned(self, k: int, stage: int) -> dict:
        lr, info1 = self.select(k, "tft_finetune", self.cfg["lr_grid"], "lr", {"lam": None})
        if stage == 1:
            return {"lr": lr, "stage1": info1}
        lam, info2 = self.select(k, "tft_ewc", self.cfg["lam_grid"], "lam", {"lr": lr})
        return {"lr": lr, "ewc_lambda": lam, "stage1": info1, "stage2": info2}

    # -- evaluation --
    def continual_unit(self, k: int, oid: int, seed: int, lr: float, lam: float) -> dict:
        held, _ = self.fold_conditions(k)

        def fit():
            s, tasks = self.stream("tft_ewc", self.orders(k)[oid], seed, lr, lam)
            pooled = pd.concat([t.val for t in tasks], ignore_index=True)
            cal = HorizonCalibrator(HORIZONS[self.exp]).update(s, pooled, "pooled 48 training conditions")
            out = {}
            for c in held:
                frame, direct, hp = self.held(c)
                raw = s.predict(frame)
                shr = cal.shrink(raw, frame)
                y = frame[FC.TARGET].to_numpy(float)
                iv = {}
                for lv in cal.levels:
                    q = cal.halfwidths(frame, "model", lv)
                    m = interval_metrics(y, shr, q, lv)
                    iv[str(round(lv * 100))] = {"n": int(len(y)), "covered": int(np.sum(np.abs(y - shr) <= q)),
                                                "interval_score": m["interval_score"], "width": m["width"]}
                out[c] = {**self.unit_scores(frame, direct, raw, shr, hp), "intervals": iv}
            return {"conditions": out, "alpha": dict(cal.alpha_), "train_time_s": float(np.sum(s.training_time_))}
        return self.ckpt(f"cont_f{k}_o{oid}_s{seed}", fit, {"lr": lr, "lam": lam})

    def joint_unit(self, k: int, seed: int, lr: float) -> dict:
        held, train = self.fold_conditions(k)

        def fit():
            s = p3.build_strategy("tft_joint", self.features, seed, self.res, self.settings(lr))
            s.fit_all([self.task(c) for c in train])
            return {c: run_d26.scores(*self.held(c)[:2], s.predict(self.held(c)[0]), self.held(c)[2]) for c in held}
        return self.ckpt(f"joint_f{k}_s{seed}", fit, {"lr": lr})

    # -- parts --
    def done_path(self, part: str) -> Path:
        return self.ckpt.dir / f"done_{part}.json"

    def run_part(self, part: str) -> dict:
        if part not in PARTS:
            raise SystemExit(f"unknown part {part!r}; choose from {PARTS}")
        name, k = part.rsplit("_f", 1)
        k = int(k)
        if name in ("tune1", "tune2"):
            out = self.tuned(k, int(name[-1]))
        else:
            t = self.tuned(k, 2)
            out = {"tuning": t, "joint": {}, "continual": {}}
            for seed in self.cfg[f"{name}_seeds"]:
                out["joint"][seed] = self.joint_unit(k, seed, t["lr"])
                for oid in range(self.cfg["n_orders"]):
                    out["continual"][(oid, seed)] = self.continual_unit(k, oid, seed, t["lr"], t["ewc_lambda"])
        self.done_path(part).parent.mkdir(parents=True, exist_ok=True)
        self.done_path(part).write_text("{}")
        print(f"part {part} done", flush=True)
        return out

    # -- single fits --
    def single_fits(self, k: int) -> dict:
        """{method: {condition: scores}}; GBM_LAST is {order_id: {condition: scores}}."""
        held, train = self.fold_conditions(k)
        fit = pd.concat([self.task(c).train for c in train], ignore_index=True)       # residual scale
        obs = lambda f: [c for c in self.features if f[c].notna().any()]

        def on(model_rows, factory):
            cols = obs(model_rows)
            m = factory(cols).fit(model_rows[cols], model_rows[FC.TARGET].to_numpy(float))
            return lambda frame: m.predict(frame[cols])
        preds = {ALL: on(fit, lambda c: FC._gbm(0)), RIDGE: on(fit, FC._ridge),
                 LIN: lambda f: np.zeros(len(f)), PERS: lambda f: -f["lin_forecast"].to_numpy(float)}
        out = {m: {c: run_d26.scores(*self.held(c)[:2], p(self.held(c)[0]), self.held(c)[2]) for c in held}
               for m, p in preds.items()}
        out[LAST] = {}
        for oid, order in enumerate(self.orders(k)):
            p = on(self.task(order[-1]).train, lambda c: FC._gbm(0))
            out[LAST][oid] = {c: run_d26.scores(*self.held(c)[:2], p(self.held(c)[0]), self.held(c)[2]) for c in held}
        return out


# -- collection -------------------------------------------------------------------------------------------------------

METRICS = ("rmse", "rmse_hplus", "rmse_hminus")


def condition_values(r: Runner, parts: dict, singles: dict, folds: list[int]) -> pd.DataFrame:
    """One row per (set, method, condition): metric averaged over that set's runs (or GBM-LAST's orders)."""
    rows = []
    hcols = [f"rmse_h{h}" for h in HORIZONS[r.exp]]
    keep = METRICS + tuple(hcols)
    for s in SETS:
        for k in folds:
            out = parts[f"{s}_f{k}"]
            held, _ = r.fold_conditions(k)
            units = list(out["continual"].values())
            for c in held:
                for method, scs in (*((m, [u["conditions"][c][key] for u in units]) for m, key in r.CONT.items()),
                                    (JOINT, [j[c] for j in out["joint"].values()]),
                                    (LAST, [o[c] for o in singles[k][LAST].values()]),
                                    *((m, [singles[k][m][c]]) for m in (ALL, LIN, PERS, RIDGE))):
                    row = {"set": s, "method": method, "condition": c, "fold": k, "n_runs": len(scs)}
                    # A condition may have no sample at a long horizon (short series): that metric is missing.
                    vals = {m: [x.get(m, np.nan) for x in scs] for m in keep}
                    row.update({m: float(np.nanmean(x)) if np.isfinite(x).any() else np.nan for m, x in vals.items()})
                    if method == SHR:
                        for lv in ("90", "80"):
                            iv = [u["conditions"][c]["intervals"][lv] for u in units]
                            row[f"cov{lv}"] = sum(v["covered"] for v in iv) / sum(v["n"] for v in iv)
                            row[f"covered{lv}"], row[f"n{lv}"] = sum(v["covered"] for v in iv), sum(v["n"] for v in iv)
                            row[f"is{lv}"] = float(np.mean([v["interval_score"] for v in iv]))
                    rows.append(row)
    v = pd.DataFrame(rows)
    meta = r.sets.set_index("condition")
    for key in r.DESC_KEYS:
        v[key] = v["condition"].map(meta[key])
    v["age_type"] = v["age_type"].map(TYPES)
    return v


def run_tests(v: pd.DataFrame, primary=PRIMARY, secondary=SECONDARY) -> tuple[list[dict], list[dict]]:
    rows = []
    for fam, comps in {"primary": primary, "secondary": secondary}.items():
        for s in SETS:
            g = v[v["set"] == s]
            got = []
            for a, b, m in comps:
                va = g[g["method"] == a].set_index("condition")[m]
                vb = g[g["method"] == b].set_index("condition")[m].reindex(va.index)
                pc = ev.paired_comparison(va.to_numpy(float), vb.to_numpy(float))
                d = (va - vb).dropna()
                got.append({"set": s, "a": a, "b": b, "metric": m, **pc,
                            "n_conditions_a_better": int((d < 0).sum()), "n_conditions_b_better": int((d > 0).sum())})
            holm = ev.holm_correction({(x["a"], x["b"], x["metric"]): x["p_value"] for x in got})
            rows += [{**x, "family": fam, "p_holm": holm[(x["a"], x["b"], x["metric"])]} for x in got]
    t = pd.DataFrame(rows)
    verdicts = []
    for (fam, a, b, m), g in t.groupby(["family", "a", "b", "metric"], sort=False):
        mn, rp = g[g["set"] == "main"].iloc[0], g[g["set"] == "replicate"].iloc[0]
        verdicts.append({"family": fam, "a": a, "b": b, "metric": m, "diff_main": float(mn["mean_diff"]),
                         "diff_replicate": float(rp["mean_diff"]), "p_holm_main": float(mn["p_holm"]),
                         "p_holm_replicate": float(rp["p_holm"]), "n_main": int(mn["n"]),
                         "verdict": run_d25.verdict(mn, rp)})
    return rows, verdicts


def descriptive(v: pd.DataFrame, hcols: list[str], keys=("age_type", "age_temp")) -> dict:
    by = lambda keys: (v.groupby(["set", "method", *keys])[["rmse", "rmse_hplus"]].mean().reset_index()
                       .to_dict("records"))
    shr = v[v["method"] == SHR]
    cov = {}
    for s, g in shr.groupby("set"):
        cov[s] = {lv: {"pooled": float(g[f"covered{lv}"].sum() / g[f"n{lv}"].sum()),
                       **{t: float(h[f"covered{lv}"].sum() / h[f"n{lv}"].sum()) for t, h in g.groupby("age_type")}}
                  for lv in ("90", "80")}
    worst = (v.loc[v.groupby(["set", "method"])["rmse"].idxmax(), ["set", "method", "condition", "rmse"]]
             .to_dict("records"))
    return {"mean_over_conditions": v.groupby(["set", "method"])[["rmse", "rmse_hplus", "rmse_hminus", *hcols]]
            .mean().reset_index().to_dict("records"),
            **{f"by_{k}": by([k]) for k in keys}, "worst_condition": worst,
            "coverage": cov, "coverage_check": {"threshold": COVERAGE_ADEQUATE, "adequate": bool(
                set(cov) == set(SETS) and all(c["90"]["pooled"] >= COVERAGE_ADEQUATE for c in cov.values()))}}


def collect(r: Runner, collect_only: bool) -> int:
    parts, not_run = {}, []
    for part in PARTS:
        if collect_only and not r.done_path(part).exists():
            not_run.append(part)
            continue
        parts[part] = r.run_part(part)
    folds = [k for k in range(N_FOLDS) if all(f"{s}_f{k}" in parts for s in SETS)]
    if not folds:
        raise SystemExit("no fold has finished in both seed sets; nothing to collect")
    singles = {k: r.single_fits(k) for k in folds}
    v = condition_values(r, parts, singles, folds)
    tests, verdicts = run_tests(v, r.PRIMARY, r.SECONDARY)
    hcols = [f"rmse_h{h}" for h in HORIZONS[r.exp]]
    strata = {}
    for c in r.sets["condition"]:
        hp = r.held(c)[2]
        strata[c] = {"n": int(len(hp)), "hplus": int(hp.sum())}
    name = f"forecasting_{r.NAME}_{r.exp}" + (".json" if r.profile == "paper" else "_smoke.json")
    out = ROOT / "results" / name if r.profile == "paper" else r.data / "cache" / r.NAME / name
    X.write_results(out, {
        "decision": r.DECISION, "profile": r.profile, "experiment": r.exp, "horizons": list(HORIZONS[r.exp]),
        "parts_not_run": not_run, "folds_in_tests": folds, "n_samples": len(r.direct), "features": r.features,
        "condition_sets": r.sets.to_dict("records"), "strata_counts": strata,
        "orders": {k: [list(o) for o in r.orders(k)] for k in folds},
        "tuning": {k: parts[f"main_f{k}"]["tuning"] for k in folds}, "settings": r.cfg,
        "condition_values": v.to_dict("records"), "descriptive": descriptive(v, hcols, r.DESC_KEYS),
        "tests": tests, "verdicts": verdicts}, ROOT)
    print(v.groupby(["set", "method"])[["rmse", "rmse_hplus"]].mean().round(3).to_string(), flush=True)
    for x in verdicts:
        print(f"{x['family']:9s} {x['a']} - {x['b']} [{x['metric']}]: {x['diff_main']:+.3f} / {x['diff_replicate']:+.3f}"
              f" (p_Holm {x['p_holm_main']:.2g} / {x['p_holm_replicate']:.2g}): {x['verdict']}", flush=True)
    print(f"written {out}" + (f"; not run: {not_run}" if not_run else ""), flush=True)
    return 0


def write_sets(cls=Runner) -> int:
    sets = cls.make_sets(cls.read_checks())
    out = ROOT / "results" / f"{cls.NAME}_condition_sets.csv"
    sets.to_csv(out, index=False)
    print(sets.groupby(["set", "fold", "age_type"]).size().to_string(), f"\nwritten {out}", flush=True)
    return 0


def check_sets(r: Runner) -> None:
    """The paper run must use exactly the committed assignment."""
    f = ROOT / "results" / f"{r.NAME}_condition_sets.csv"
    if r.profile == "paper" and not (f.exists() and pd.read_csv(f)[["condition", "set", "fold"]]
                                     .equals(r.sets[["condition", "set", "fold"]].reset_index(drop=True))):
        raise SystemExit(f"{f} missing or different from the computed assignment; run --write-sets and commit it")


def main(cls=Runner) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp", choices=tuple(HORIZONS))
    ap.add_argument("--part", choices=PARTS)
    ap.add_argument("--profile", choices=tuple(PROFILES), default="paper")
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--write-sets", action="store_true")
    a = ap.parse_args()
    if a.write_sets:
        return write_sets(cls)
    if not a.exp:
        ap.error("--exp is required")
    r = cls(a.exp, a.profile)
    check_sets(r)
    print(f"{cls.DECISION} {a.exp} ({a.profile}): {len(r.direct)} samples, {len(r.sets)} conditions", flush=True)
    if a.part:
        r.run_part(a.part)
        return 0
    return collect(r, a.collect_only)


if __name__ == "__main__":
    sys.exit(main())
