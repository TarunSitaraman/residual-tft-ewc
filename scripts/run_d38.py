"""Run decision D38: the residual TFT with experience replay, a seed ensemble and predict-before-training
calibration (V2) on unseen usage conditions, on S10 (the D36 setting) and S7 (the D34 setting).

Everything not stated here is ``run_d34.py`` / ``run_d36.py`` unchanged: data, inputs, splits, condition sets, outer
folds, evaluation and tuning orders, seeds, the single-fit comparators and the tests over held-out conditions (D21
replication rule). Nothing under ``src/`` changes (the D34 and D36 checkpoints stay valid); the replay strategy and
the calibrator of D38 live in this file.

Arms per (outer fold, evaluation order, seed), each kept with its forecasts so that seed ensembles can be formed:

- ``a0``: the D34 / D36 model unchanged (residual TFT + EWC at the learning rate and lambda of the committed D34 / D36
  results file). Gives the earlier proposed model, the reproduction check and the ensemble without replay.
- ``r32``: residual TFT with experience replay, 32 training and 8 validation rows kept per learned condition.
- ``r8``: the same with 8 and 2 rows, at r32's learning rate.

Parts: ``tune_f<k>`` (learning rate of the replay model on the tuning conditions), then ``<arm>_<set>_f<k>``.
Collection forms, per evaluation order, the mean forecast of the three seeds of a seed set, calibrates it on the
forecasts the ensemble made for each condition of the second half of the stream *before* it was trained on that
condition (shrinkage factor and conformal half-widths per horizon and origin stratum), and scores the held-out
conditions. V2 = ``tft_residual_er32_ens_cal``.

    python scripts/run_d38.py --exp s10_e1 --part tune_f0
    python scripts/run_d38.py --exp s10_e1 --collect-only
    python scripts/run_d38_night.py --dataset s10           # the whole plan for one dataset
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_d25  # noqa: E402
import run_d26  # noqa: E402
import run_d34 as B34  # noqa: E402
import run_d36 as B36  # noqa: E402
from src import continual_learning as CL  # noqa: E402
from src import experiments as X  # noqa: E402
from src import forecasting as FC  # noqa: E402
from src import p3  # noqa: E402
from src.calibration import LEVELS, HorizonCalibrator, conformal_halfwidth, interval_metrics  # noqa: E402

HORIZONS = run_d25.HORIZONS
N_FOLDS, SETS = B34.N_FOLDS, B34.SETS
PROFILES = {
    "paper": {"n_orders": 5, "main_seeds": (0, 1, 2), "replicate_seeds": (5, 6, 7), "tuning_runs": 5, "epochs": 200,
              "lr_grid": (1e-3, 3e-3, 1e-2)},
    "smoke": {"n_orders": 2, "main_seeds": (0, 1), "replicate_seeds": (5, 6), "tuning_runs": 2, "epochs": 2,
              "lr_grid": (3e-3, 1e-2)},
}
BUFFERS = {"r32": 32, "r8": 8}                       # training rows kept per learned condition; validation rows: B // 4
ARMS = ("r32", "a0", "r8")                           # run order: the primary arm first
PARTS = tuple([f"tune_f{k}" for k in range(N_FOLDS)]
              + [f"{a}_{s}_f{k}" for a in ARMS for s in SETS for k in range(N_FOLDS)])
DEPENDS = {f"{a}_{s}_f{k}": f"tune_f{k}" for a in BUFFERS for s in SETS for k in range(N_FOLDS)}
MIN_CAL = 30                                         # fewer calibration rows in a stratum: use the whole horizon
TOL_NEURAL, TOL_EXACT = 0.05, 1e-6                   # reproduction tolerances (pp)
COVERAGE_ADEQUATE = B34.COVERAGE_ADEQUATE

RAW, SHR, SEL, JOINT = B34.RAW, B34.SHR, B36.SEL, B34.JOINT
ALL, LAST, LIN, PERS, RIDGE = B34.ALL, B34.LAST, B34.LIN, B34.PERS, B34.RIDGE
A0_ENS, A0_CAL = "tft_residual_ewc_ens", "tft_residual_ewc_ens_cal"
ER = {a: f"tft_residual_er{b}" for a, b in BUFFERS.items()}
ER_ENS = {a: f"{m}_ens" for a, m in ER.items()}
ER_CAL = {a: f"{m}_ens_cal" for a, m in ER.items()}
GBM_ER = {a: f"gbm_er{b}" for a, b in BUFFERS.items()}
V2 = ER_CAL["r32"]
METRICS = B34.METRICS


# -- experience replay (D38) ------------------------------------------------------------------------------------------

def draw(frame: pd.DataFrame, k: int, seed: int, position: int, salt: int) -> pd.DataFrame:
    """``k`` rows of ``frame`` (all of them if it has fewer), a seeded draw that depends only on (seed, position of the
    condition in the stream, salt), kept in row order. Used by the replay model and by the GBM-ER comparator."""
    k = min(int(k), len(frame))
    if k == 0:
        return frame.iloc[:0]
    rng = np.random.default_rng([int(seed), int(position), int(salt)])
    return frame.iloc[np.sort(rng.choice(len(frame), size=k, replace=False))].copy()


def stored_rows(tasks, buffer: int, seed: int) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """The training and validation rows a replay model with ``seed`` holds after learning ``tasks`` in this order."""
    tr = [draw(t.train, buffer, seed, i, 0) for i, t in enumerate(tasks)]
    va = [draw(t.val, buffer // 4, seed, i, 1) for i, t in enumerate(tasks)]
    return [f for f in tr if len(f)], [f for f in va if len(f)]


class ExperienceReplay(CL.Strategy):
    """Experience replay with mixed batches and a validation memory (D38).

    After a task, ``buffer_per_task`` of its training rows and ``buffer_per_task // 4`` of its validation rows are
    kept (``draw``). While a later task is learned, each mini-batch loss is the MSE on the current batch plus
    ``replay_weight`` times the MSE on an equally large batch drawn from the kept training rows, and early stopping
    uses the current task's validation rows together with the kept validation rows. Unlike ``Replay`` (which
    concatenates the buffer to the training rows), the cost of a step does not grow with the number of tasks.
    """

    name = "er"

    def __init__(self, features, buffer_per_task: int = 32, replay_weight: float = 1.0, **kwargs) -> None:
        super().__init__(features, **kwargs)
        self.buffer_per_task, self.replay_weight = int(buffer_per_task), float(replay_weight)
        self.buffer_train_: list[pd.DataFrame] = []
        self.buffer_val_: list[pd.DataFrame] = []
        self._bx = self._by = self._bgen = None

    def config(self) -> dict:
        return {**super().config(), "buffer_per_task": self.buffer_per_task,
                "val_buffer_per_task": self.buffer_per_task // 4, "replay_weight": self.replay_weight}

    def _before_task(self, task) -> None:
        super()._before_task(task)
        self._bx = None
        if self.buffer_train_ and self.replay_weight > 0:
            self._bx, self._by = self._xy(pd.concat(self.buffer_train_, ignore_index=True))
            self._bgen = torch.Generator().manual_seed(self.seed + 1000 * len(self.tasks_seen_) + 7)

    def _penalty(self, xb: torch.Tensor) -> torch.Tensor:
        if self._bx is None:
            return torch.zeros(())
        idx = torch.randperm(len(self._bx), generator=self._bgen)[: min(len(xb), len(self._bx))]
        return self.replay_weight * torch.mean((self.model(self._bx[idx]) - self._by[idx]) ** 2)

    def _train(self, train: pd.DataFrame, val: pd.DataFrame, task_index: int) -> None:
        if self.buffer_val_:
            val = pd.concat([f for f in (val, *self.buffer_val_) if len(f)], ignore_index=True)
        super()._train(train, val, task_index)

    def _after_task(self, task) -> None:
        super()._after_task(task)
        pos = len(self.tasks_seen_)                                   # this task's position (appended after the hook)
        for frame, k, salt, store in ((task.train, self.buffer_per_task, 0, self.buffer_train_),
                                      (task.val, self.buffer_per_task // 4, 1, self.buffer_val_)):
            rows = draw(frame, k, self.seed, pos, salt)
            if len(rows):
                store.append(rows)


class TFTExperienceReplay(CL.TFTMixin, ExperienceReplay):
    """The residual TFT with experience replay (the D38 model; the residual target is set by the caller)."""

    name = "tft_er"


CL.STRATEGIES.setdefault("tft_er", TFTExperienceReplay)


# -- predict-before-training calibration (D38) ------------------------------------------------------------------------

def fit_calibration(p, y, h, hp, horizons, levels=LEVELS, min_rows: int = MIN_CAL) -> dict:
    """Per (horizon, stratum): the D25 shrinkage factor ``clip(sum(p*y) / sum(p**2), 0, 1)`` and the split-conformal
    half-widths of ``|y - alpha * p|``, all on the residual scale. A stratum (H+ True / H- False) with fewer than
    ``min_rows`` rows uses its horizon's rows of both strata; a horizon with fewer keeps alpha = 1 and is flagged."""
    p, y, h, hp = np.asarray(p, float), np.asarray(y, float), np.asarray(h), np.asarray(hp, bool)
    cal = {}
    for k in horizons:
        mk = h == k
        for s in (True, False):
            m, source = mk & (hp == s), "stratum"
            if m.sum() < min_rows:
                m, source = mk, "horizon"
            alpha = 1.0
            if m.sum() < min_rows:
                source = "flagged"
            else:
                den = float(np.sum(p[m] ** 2))
                if den > 0:
                    alpha = float(np.clip(np.sum(p[m] * y[m]) / den, 0.0, 1.0))
            cal[(int(k), s)] = {"alpha": alpha, "n": int(m.sum()), "source": source,
                                "q": {lv: conformal_halfwidth(y[m] - alpha * p[m], lv)[0] for lv in levels}}
    return cal


def apply_calibration(cal: dict, p, h, hp, levels=LEVELS) -> tuple[np.ndarray, dict]:
    """(shrunk prediction, {level: half-width per row}) for rows with horizon ``h`` and stratum ``hp``."""
    p, h, hp = np.asarray(p, float), np.asarray(h), np.asarray(hp, bool)
    keys = list(zip(h.astype(int).tolist(), hp.tolist()))
    alpha = np.array([cal[k]["alpha"] for k in keys], float)
    return alpha * p, {lv: np.array([cal[k]["q"][lv] for k in keys], float) for lv in levels}


# -- runner -----------------------------------------------------------------------------------------------------------

class V2Mixin:
    """D38 on top of a D34-type runner (``BASE`` names the decision whose setting, sets and results are reused)."""
    DECISION, BASE, PREVIOUS = "D38", "", SHR

    def __init__(self, exp: str, profile: str) -> None:
        super().__init__(exp, profile)
        self.cfg = PROFILES[profile]
        self._ref, self._preq = None, {}
        # src/ is unchanged, so the code that decides a stream's result and lives in this file enters the checkpoint
        # key as well; the collection code does not, so a fix there keeps the finished units.
        code = "".join(inspect.getsource(o) for o in (draw, ExperienceReplay, TFTExperienceReplay, V2Mixin.stream_unit))
        self.ckpt = FC.Checkpoint(self.data / "cache" / self.NAME,
                                  [FC.Checkpoint.frame_hash(self.direct), exp, profile, code, sorted(BUFFERS.items())])

    # -- the committed results of the base decision (learning rate and lambda of a0, the joint model's values) --
    def reference_path(self) -> Path:
        if self.profile == "paper":
            return ROOT / "results" / f"forecasting_{self.BASE}_{self.exp}.json"
        return self.data / "cache" / self.BASE / f"forecasting_{self.BASE}_{self.exp}_smoke.json"

    def reference(self) -> dict | None:
        if self._ref is None:
            f = self.reference_path()
            if f.exists():
                self._ref = json.loads(f.read_text())
            elif self.profile == "paper":
                raise SystemExit(f"{f} missing: D38 reuses the committed {self.BASE.upper()} tuning and joint values")
        return self._ref

    def a0_settings(self, k: int) -> tuple[float, float]:
        ref = self.reference()
        if ref is None or str(k) not in ref.get("tuning", {}):       # smoke profile without a base smoke file
            return float(self.cfg["lr_grid"][0]), 1.0
        t = ref["tuning"][str(k)]
        return float(t["lr"]), float(t["ewc_lambda"])

    def tuned_replay(self, k: int) -> dict:
        lr, info = self.select(k, "tft_er", self.cfg["lr_grid"], "lr", {"lam": None})    # buffer 32 (class default)
        return {"lr": lr, "stage1": info}

    # -- streams --
    def preq_frame(self, k: int, oid: int) -> pd.DataFrame:
        """Rows forecast before training: every row (training and validation) of the conditions at positions 1.. of
        evaluation order ``oid``, with the 0-based position ``pos`` and the stratum ``hplus``."""
        if (k, oid) not in self._preq:
            frames = []
            for i, c in enumerate(self.orders(k)[oid]):
                if i:
                    t = self.task(c)
                    frames.append(pd.concat([t.train, t.val], ignore_index=True).assign(pos=i))
            out = pd.concat(frames, ignore_index=True)
            self._preq[(k, oid)] = out.assign(hplus=out["fade_rate_pp_per_day"].notna().to_numpy())
        return self._preq[(k, oid)]

    def stream_unit(self, k: int, oid: int, seed: int, arm: str, lr: float, lam: float | None) -> dict:
        """One stream of ``arm``: forecasts of every later condition before it is trained on (``preq``), raw
        forecasts of the held-out conditions (``pred``) and, for a0, the D25-shrunk ones (``shr``, the D34 rule)."""
        held, _ = self.fold_conditions(k)

        def fit():
            extra = {"buffer_per_task": BUFFERS[arm]} if arm in BUFFERS else {"ewc_lambda": lam}
            s = p3.build_strategy("tft_er" if arm in BUFFERS else "tft_ewc", self.features, seed, self.res,
                                  {**self.settings(lr), **extra})
            tasks, preq = [self.task(c) for c in self.orders(k)[oid]], []
            for i, t in enumerate(tasks):
                if i:
                    preq.append(s.predict(pd.concat([t.train, t.val], ignore_index=True)))
                s.fit_task(t)
            out = {"preq": np.concatenate(preq), "pred": {c: s.predict(self.held(c)[0]) for c in held},
                   "train_time_s": float(np.sum(s.training_time_))}
            if arm == "a0":
                cal = HorizonCalibrator(HORIZONS[self.exp]).update(
                    s, pd.concat([t.val for t in tasks], ignore_index=True), "pooled training conditions")
                out["shr"] = {c: cal.shrink(out["pred"][c], self.held(c)[0]) for c in held}
            return out
        return self.ckpt(f"{arm}_f{k}_o{oid}_s{seed}", fit, {"lr": lr, "lam": lam})

    # -- parts --
    def run_part(self, part: str) -> dict:
        if part not in PARTS:
            raise SystemExit(f"unknown part {part!r}; choose from {PARTS}")
        name, k = part.rsplit("_f", 1)
        k = int(k)
        if name == "tune":
            out = self.tuned_replay(k)
        else:
            arm, s = name.split("_")
            lr, lam = self.a0_settings(k) if arm == "a0" else (self.tuned_replay(k)["lr"], None)
            out = {"lr": lr, "lam": lam,
                   "units": {(oid, seed): self.stream_unit(k, oid, seed, arm, lr, lam)
                             for seed in self.cfg[f"{s}_seeds"] for oid in range(self.cfg["n_orders"])}}
        self.done_path(part).parent.mkdir(parents=True, exist_ok=True)
        self.done_path(part).write_text("{}")
        print(f"part {part} done", flush=True)
        return out

    # -- GBM with the same stored rows --
    def gbm_er(self, k: int, arm: str) -> dict:
        """{order_id: {condition: scores}}: gradient boosting (D18 settings) on the training rows of the last
        condition of the order plus the rows a seed-0 replay model holds of every earlier condition."""
        held, _ = self.fold_conditions(k)
        out = {}
        for oid, order in enumerate(self.orders(k)):
            tasks = [self.task(c) for c in order]
            tr, va = stored_rows(tasks[:-1], BUFFERS[arm], 0)
            rows = pd.concat([tasks[-1].train, *tr, *va], ignore_index=True)
            cols = [c for c in self.features if rows[c].notna().any()]
            m = FC._gbm(0).fit(rows[cols], rows[FC.TARGET].to_numpy(float))
            out[oid] = {c: run_d26.scores(*self.held(c)[:2], m.predict(self.held(c)[0][cols]), self.held(c)[2])
                        for c in held}
        return out


class S10Runner(V2Mixin, B36.Runner):
    NAME, BASE, PREVIOUS = "d38_s10", "d36", SEL


class S7Runner(V2Mixin, B34.Runner):
    NAME, BASE, PREVIOUS = "d38_s7", "d34", SHR


RUNNERS = {"s10": S10Runner, "s7": S7Runner}
EXPERIMENTS = tuple(f"{d}_{e}" for d in RUNNERS for e in HORIZONS)


# -- collection -------------------------------------------------------------------------------------------------------

def _mean_scores(scs: list[dict], keep) -> dict:
    vals = {m: np.array([x.get(m, np.nan) for x in scs], float) for m in keep}
    return {m: float(np.nanmean(x)) if np.isfinite(x).any() else np.nan for m, x in vals.items()}


def arm_values(r, k: int, s: str, arm: str, units: dict, keep) -> tuple[list[dict], list[dict], list[dict]]:
    """Rows (one per method and held-out condition), coverage rows per order and calibration records per order for
    one arm, fold and seed set."""
    held, train = r.fold_conditions(k)
    seeds, n_orders = r.cfg[f"{s}_seeds"], r.cfg["n_orders"]
    single = {"raw": RAW if arm == "a0" else ER[arm]}
    ens, cal_name = (A0_ENS, A0_CAL) if arm == "a0" else (ER_ENS[arm], ER_CAL[arm])
    got = {c: {} for c in held}                                       # condition -> method -> list of scores
    cov, cals = [], []
    add = lambda c, m, sc: got[c].setdefault(m, []).append(sc)
    for oid in range(n_orders):
        pf = r.preq_frame(k, oid)
        late = (pf["pos"] >= len(train) // 2).to_numpy()              # second half of the stream
        p = np.mean([units[(oid, sd)]["preq"] for sd in seeds], axis=0)
        cal = fit_calibration(p[late], pf.loc[late, FC.TARGET].to_numpy(float), pf.loc[late, "horizon"].to_numpy(),
                              pf.loc[late, "hplus"].to_numpy(), HORIZONS[r.exp])
        cals.append({"set": s, "fold": k, "order": oid, "arm": arm, "n_rows": int(late.sum()),
                     **{f"h{h}_{'hplus' if hp else 'hminus'}": {**v, "q": {str(round(lv * 100)): q for lv, q in v["q"].items()}}
                        for (h, hp), v in cal.items()}})
        for c in held:
            frame, direct, hp = r.held(c)
            y, h = frame[FC.TARGET].to_numpy(float), frame["horizon"].to_numpy()
            for sd in seeds:
                u = units[(oid, sd)]
                add(c, single["raw"], run_d26.scores(frame, direct, u["pred"][c], hp))
                if arm == "a0":
                    add(c, SHR, run_d26.scores(frame, direct, u["shr"][c], hp))
                    add(c, SEL, run_d26.scores(frame, direct, np.where(hp, u["shr"][c], u["pred"][c]), hp))
            mean = np.mean([units[(oid, sd)]["pred"][c] for sd in seeds], axis=0)
            add(c, ens, run_d26.scores(frame, direct, mean, hp))
            shrunk, q = apply_calibration(cal, mean, h, hp)
            add(c, cal_name, run_d26.scores(frame, direct, shrunk, hp))
            for lv in LEVELS:
                m = interval_metrics(y, shrunk, q[lv], lv)
                cov.append({"set": s, "method": cal_name, "condition": c, "fold": k, "order": oid,
                            "level": str(round(lv * 100)), "n": int(len(y)),
                            "covered": int(np.sum(np.abs(y - shrunk) <= q[lv])),
                            "interval_score": m["interval_score"], "width": m["width"]})
    rows = [{"set": s, "method": m, "condition": c, "fold": k, "n_runs": len(scs), **_mean_scores(scs, keep)}
            for c in held for m, scs in got[c].items()]
    return rows, cov, cals


def condition_values(r, parts: dict, singles: dict, gbm: dict, folds: list[int], arms: list[str]) -> tuple:
    """(values, coverage rows, calibration records). ``values`` has one row per (set, method, condition)."""
    keep = METRICS + tuple(f"rmse_h{h}" for h in HORIZONS[r.exp])
    rows, cov, cals = [], [], []
    for s in SETS:
        for k in folds:
            held, _ = r.fold_conditions(k)
            for arm in arms:
                a, b, c = arm_values(r, k, s, arm, parts[f"{arm}_{s}_f{k}"]["units"], keep)
                rows, cov, cals = rows + a, cov + b, cals + c
            for c in held:
                per_order = {LAST: singles[k][LAST], **{GBM_ER[a]: gbm[(k, a)] for a in arms if a in BUFFERS}}
                for m, by_order in per_order.items():
                    scs = [o[c] for o in by_order.values()]
                    rows.append({"set": s, "method": m, "condition": c, "fold": k, "n_runs": len(scs),
                                 **_mean_scores(scs, keep)})
                for m in (ALL, LIN, PERS, RIDGE):
                    rows.append({"set": s, "method": m, "condition": c, "fold": k, "n_runs": 1,
                                 **_mean_scores([singles[k][m][c]], keep)})
    v = pd.DataFrame(rows)
    ref = r.reference()
    if ref is not None:                                               # the joint model is read, not refitted
        j = pd.DataFrame(ref["condition_values"])
        j = j[(j["method"] == JOINT) & j["condition"].isin(v["condition"])]
        v = pd.concat([v, j[[c for c in v.columns if c in j.columns]]], ignore_index=True)
    c = pd.DataFrame(cov)
    for lv in ("90", "80"):
        g = c[c["level"] == lv].groupby(["set", "method", "condition"])
        add = pd.DataFrame({f"covered{lv}": g["covered"].sum(), f"n{lv}": g["n"].sum(),
                            f"is{lv}": g["interval_score"].mean(), f"width{lv}": g["width"].mean()}).reset_index()
        add[f"cov{lv}"] = add[f"covered{lv}"] / add[f"n{lv}"]
        v = v.merge(add, on=["set", "method", "condition"], how="left")
    meta = r.sets.set_index("condition")
    for key in r.DESC_KEYS:
        v[key] = v["condition"].map(meta[key])
    v["age_type"] = v["age_type"].map(B34.TYPES)
    return v, c, cals


def test_families(r, methods: set) -> tuple[list, list, list]:
    """(primary, secondary, comparisons left out because an arm or the reference is missing), D38."""
    primary = [(V2, ALL, "rmse"), (V2, GBM_ER["r32"], "rmse"), (V2, LIN, "rmse_hplus"), (V2, r.PREVIOUS, "rmse")]
    secondary = [(V2, JOINT, "rmse"), (V2, LAST, "rmse"), (V2, RIDGE, "rmse"), (V2, ALL, "rmse_hplus"),
                 (V2, ALL, "rmse_hminus"), (A0_ENS, RAW, "rmse"), (ER["r32"], RAW, "rmse"),
                 (ER_ENS["r32"], ER["r32"], "rmse"), (V2, ER_ENS["r32"], "rmse"), (ER_CAL["r8"], V2, "rmse"),
                 (ER_CAL["r8"], GBM_ER["r8"], "rmse")]
    ok = lambda t: t[0] in methods and t[1] in methods
    left = [list(t) for t in primary + secondary if not ok(t)]
    return [t for t in primary if ok(t)], [t for t in secondary if ok(t)], left


def direction(tests: list[dict], verdicts: list[dict]) -> list[dict]:
    """D38 direction rule: a comparison is in favour of ``a`` only if it is replicated with a negative mean and, in
    both seed sets, the median difference is negative and more conditions favour ``a``."""
    t = pd.DataFrame(tests)
    out = []
    for x in verdicts:
        g = t[(t["family"] == x["family"]) & (t["a"] == x["a"]) & (t["b"] == x["b"]) & (t["metric"] == x["metric"])]
        rank = bool(len(g) == 2 and (g["median_diff"] < 0).all()
                    and (g["n_conditions_a_better"] > g["n_conditions_b_better"]).all())
        rep = x["verdict"] == "replicated, a better"
        out.append({**x, "median_main": float(g[g["set"] == "main"]["median_diff"].iloc[0]),
                    "median_replicate": float(g[g["set"] == "replicate"]["median_diff"].iloc[0]),
                    "a_better_main": int(g[g["set"] == "main"]["n_conditions_a_better"].iloc[0]),
                    "a_better_replicate": int(g[g["set"] == "replicate"]["n_conditions_a_better"].iloc[0]),
                    "in_favour": rep and rank, "mean_and_rank_disagree": rep and not rank})
    return out


def reproduction(r, v: pd.DataFrame) -> dict:
    """Largest absolute difference (pp) per method between the recomputed per-condition RMSE and the committed value
    of the base decision. In the smoke profile nothing is enforced (the base smoke run uses other seeds)."""
    ref = r.reference()
    if ref is None:
        return {"checked": False, "reason": "no reference file"}
    j = pd.DataFrame(ref["condition_values"]).set_index(["set", "method", "condition"])["rmse"]
    out, bad = {}, []
    for m in (RAW, SHR, SEL, ALL, LAST, LIN, PERS, RIDGE):
        g = v[v["method"] == m].set_index(["set", "method", "condition"])["rmse"]
        common = g.index.intersection(j.index)
        if not len(common):
            continue
        out[m] = float((g.loc[common] - j.loc[common]).abs().max())
        if out[m] > (TOL_NEURAL if m in (RAW, SHR, SEL) else TOL_EXACT):
            bad.append(m)
    return {"checked": r.profile == "paper", "max_abs_diff_pp": out, "outside_tolerance": bad,
            "tolerance_pp": {"neural": TOL_NEURAL, "deterministic": TOL_EXACT}}


def descriptive(r, v: pd.DataFrame, cov: pd.DataFrame, cals: list[dict]) -> dict:
    hcols = [f"rmse_h{h}" for h in HORIZONS[r.exp]]
    w = v[v["set"] == "main"].pivot(index="condition", columns="method", values="rmse")
    out = {"mean_over_conditions": v.groupby(["set", "method"])[["rmse", "rmse_hplus", "rmse_hminus", *hcols]]
           .mean().reset_index().to_dict("records"),
           "median_over_conditions": v.groupby(["set", "method"])["rmse"].median().reset_index().to_dict("records"),
           **{f"by_{k}": v.groupby(["set", "method", k])[["rmse", "rmse_hplus"]].mean().reset_index().to_dict("records")
              for k in r.DESC_KEYS},
           "worst_condition": v.loc[v.groupby(["set", "method"])["rmse"].idxmax(),
                                    ["set", "method", "condition", "rmse"]].to_dict("records"),
           "n_conditions_best_main": w.eq(w.min(axis=1), axis=0).sum().astype(int).to_dict()}
    if "s2_z02" in set(v["condition"]):                               # named in D38 before any D38 result
        out["mean_without_s2_z02"] = (v[v["condition"] != "s2_z02"].groupby(["set", "method"])["rmse"].mean()
                                      .reset_index().to_dict("records"))
    c = cov.merge(v[["condition", "age_type"]].drop_duplicates(), on="condition")
    pooled = lambda g: float(g["covered"].sum() / g["n"].sum())
    coverage = {}
    for (m, s, lv), g in c.groupby(["method", "set", "level"]):
        coverage.setdefault(m, {}).setdefault(s, {})[lv] = {
            "pooled": pooled(g), "interval_score": float(g["interval_score"].mean()), "width": float(g["width"].mean()),
            "by_age_type": {t: pooled(h) for t, h in g.groupby("age_type")},
            "by_order": {str(o): pooled(h) for o, h in g.groupby("order")}}
    out["coverage"] = coverage
    v2 = coverage.get(V2, {})
    out["coverage_check"] = {"method": V2, "threshold": COVERAGE_ADEQUATE, "adequate": bool(
        set(v2) == set(SETS) and all(x["90"]["pooled"] >= COVERAGE_ADEQUATE for x in v2.values()))}
    a = pd.DataFrame([{"arm": x["arm"], "set": x["set"], "cell": k, "alpha": val["alpha"], "source": val["source"]}
                      for x in cals for k, val in x.items() if isinstance(val, dict) and "alpha" in val])
    out["alpha_median"] = a.groupby(["arm", "set", "cell"])["alpha"].median().reset_index().to_dict("records")
    out["calibration_cells_not_from_own_stratum"] = int((a["source"] != "stratum").sum())
    return out


def collect(r, collect_only: bool, record_mismatch: bool = False) -> int:
    parts, not_run = {}, []
    for part in PARTS:
        if collect_only and not r.done_path(part).exists():
            not_run.append(part)
            continue
        parts[part] = r.run_part(part)
    done = lambda arm, k: all(f"{arm}_{s}_f{k}" in parts for s in SETS)
    folds = [k for k in range(N_FOLDS) if done("r32", k)]             # the primary arm decides the folds
    if not folds:
        raise SystemExit("no fold has finished arm r32 in both seed sets; nothing to collect")
    arms = [a for a in ARMS if all(done(a, k) for k in folds)]
    singles = {k: r.single_fits(k) for k in folds}
    gbm = {(k, a): r.gbm_er(k, a) for k in folds for a in arms if a in BUFFERS}
    v, cov, cals = condition_values(r, parts, singles, gbm, folds, arms)
    primary, secondary, left = test_families(r, set(v["method"]))
    tests, verdicts = B34.run_tests(v, primary, secondary)
    verdicts = direction(tests, verdicts)
    rep = reproduction(r, v)
    if rep.get("checked") and rep["outside_tolerance"] and not record_mismatch:
        raise SystemExit(f"reproduction check failed for {rep['outside_tolerance']}: {rep['max_abs_diff_pp']}; "
                         "nothing written. Find the cause, or record the deviation in the D38 run log and collect "
                         "with --record-mismatch")
    ds = r.NAME.split("_")[1]
    name = f"forecasting_d38_{ds}_{r.exp}" + (".json" if r.profile == "paper" else "_smoke.json")
    out = ROOT / "results" / name if r.profile == "paper" else r.data / "cache" / r.NAME / name
    X.write_results(out, {
        "decision": r.DECISION, "dataset": ds, "base_decision": r.BASE.upper(), "profile": r.profile,
        "experiment": r.exp, "horizons": list(HORIZONS[r.exp]), "parts_not_run": not_run, "folds_in_tests": folds,
        "arms_in_tests": arms, "tests_not_run": left, "n_samples": len(r.direct), "features": r.features,
        "v2": V2, "previous_model": r.PREVIOUS, "buffers": {a: {"train": b, "val": b // 4} for a, b in BUFFERS.items()},
        "condition_sets": r.sets.to_dict("records"),
        "orders": {k: [list(o) for o in r.orders(k)] for k in folds},
        "tuning": {k: {"replay": parts[f"tune_f{k}"] if f"tune_f{k}" in parts else None,
                       "a0": dict(zip(("lr", "ewc_lambda"), r.a0_settings(k)))} for k in folds},
        "settings": r.cfg, "reproduction": rep, "condition_values": v.to_dict("records"),
        "calibration": cals, "descriptive": descriptive(r, v, cov, cals), "tests": tests, "verdicts": verdicts}, ROOT)
    print(v.groupby(["set", "method"])[["rmse", "rmse_hplus"]].mean().round(3).to_string(), flush=True)
    for x in verdicts:
        print(f"{x['family']:9s} {x['a']} - {x['b']} [{x['metric']}]: {x['diff_main']:+.3f} / {x['diff_replicate']:+.3f}"
              f" (p_Holm {x['p_holm_main']:.2g} / {x['p_holm_replicate']:.2g}; median {x['median_main']:+.3f} / "
              f"{x['median_replicate']:+.3f}): {x['verdict']}"
              + ("; in favour" if x["in_favour"] else "; MEAN AND RANK DISAGREE" if x["mean_and_rank_disagree"] else ""),
              flush=True)
    print(f"reproduction: {rep}", flush=True)
    print(f"written {out}" + (f"; not run: {not_run}" if not_run else ""), flush=True)
    return 0


def check_sets(r) -> None:
    """The paper run must use exactly the committed assignment of the base decision."""
    f = ROOT / "results" / f"{r.BASE}_condition_sets.csv"
    if r.profile == "paper" and not (f.exists() and pd.read_csv(f)[["condition", "set", "fold"]]
                                     .equals(r.sets[["condition", "set", "fold"]].reset_index(drop=True))):
        raise SystemExit(f"{f} missing or different from the computed assignment")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp", choices=EXPERIMENTS, required=True, help="<dataset>_<experiment>, e.g. s10_e1")
    ap.add_argument("--part", choices=PARTS)
    ap.add_argument("--profile", choices=tuple(PROFILES), default="paper")
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--record-mismatch", action="store_true",
                    help="write the results although the reproduction check failed (a deviation to log in D38)")
    a = ap.parse_args()
    ds, exp = a.exp.split("_")
    r = RUNNERS[ds](exp, a.profile)
    check_sets(r)
    print(f"D38 {a.exp} ({a.profile}): {len(r.direct)} samples, {len(r.sets)} conditions", flush=True)
    if a.part:
        r.run_part(a.part)
        return 0
    return collect(r, a.collect_only, a.record_mismatch)


if __name__ == "__main__":
    sys.exit(main())
