"""Protocol P3 (decision D21): pretrain on the first-life pool, then learn the S6 test cases as a stream.

Shared inputs
-------------
Every dataset is mapped onto one input set (D21, "Inputs"): SOH history, calendar and cycle time,
planned time to the target, and the operating conditions below. A condition that a dataset does not
define is missing (NaN), never guessed; the frozen preprocessor imputes it and adds a missing-value
indicator. Sources of the hand-entered values:

- S7: the data dictionary (``Data Structure v05.3.xlsx``, sheets CFG and StateValues): C-rates are
  current / nominal capacity; age_type 1 calendar, 2 cyclic, 3 WLTP profile; profiles 2 = 10–100 % SoC,
  3 and 4 = 10–90 % SoC. Cyclic cells have a voltage window but no stated DoD (missing).
- S8: ``Data Introduction-English.pdf`` (XJTU): per-batch charge and discharge rates, CC-CV charge to
  4.2 V; "room
  temperature", not controlled (missing); batches 4–6 discharge partially (DoD missing).
- S9: Zhu et al. 2022, Methods and Table 1: cells named ``CYX-Y/Z`` = X degC, Y C charge, Z C discharge;
  every cycle a full discharge (DoD 1), CC-CV charge to 4.2 V.
- S6: Wheeler et al. 2025, Table 2 (``wheeler.TEST_CASES``); WLTP discharge is C/3 on average; 50 degC.

Pretraining and the stream
--------------------------
``pretrain`` fits a strategy on the pool as one task with the strategies' own code: the frozen
preprocessor and target scaling are fitted on the pool's training rows, training early-stops on the
pool's validation rows, and ``_after_task`` runs (for EWC: Fisher information and anchor = the
pretrained weights). The S6 test cases then follow with ``fit_task`` exactly as in P1, so no S6
statistics enter the preprocessing.
"""

from __future__ import annotations

import copy
import re
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src import evaluation as ev
from src import forecasting as FC
from src import pool
from src import protocols as P
from src import wheeler as W
from src.continual_learning import make_strategy

TARGET = FC.TARGET
CHEMISTRIES: tuple[str, ...] = ("NMC", "NCA", "NMC_NCA", "LFP")
SHARED_FEATURES: tuple[str, ...] = (
    "soh_origin", "fade_rate_pp_per_day", "horizon", "days_since_first", "days_to_target",
    "cycles_since_first", "cycles_to_target",
    "temp_c", "charge_c_rate", "discharge_c_rate", "dod", "v_charge_max", "cv_charge", "calendar", "rest_soc",
    *[f"chem_{c}" for c in CHEMISTRIES],
)

#: S8 batches (XJTU description): (charge C-rate, discharge C-rate, DoD). Batch 3 rotates the discharge
#: through 0.5/1/2/3/5C (mean 2.3C); batch 4 the same to 3.0 V (partial); batch 5 random-walk
#: discharge (rate and DoD not fixed); batch 6 the GEO profile at 0.667C with variable duration.
S8_CONDITIONS: dict[str, tuple[float, float, float]] = {
    "Batch-1": (2.0, 1.0, 1.0), "Batch-2": (3.0, 1.0, 1.0), "Batch-3": (2.0, 2.3, 1.0),
    "Batch-4": (2.0, 2.3, np.nan), "Batch-5": (3.0, np.nan, np.nan), "Batch-6": (2.0, 0.667, np.nan),
}
S7_PROFILE_DOD: dict[int, float] = {2: 0.9, 3: 0.8, 4: 0.8}
S9_CHEM: dict[str, str] = {"D1": "NCA", "D2": "NMC", "D3": "NMC_NCA"}


def _rate(code: str) -> float:
    """S9 rate codes in file names: '025' -> 0.25, '05' -> 0.5, '1' -> 1, '2' -> 2."""
    return float("0." + code[1:]) if code.startswith("0") else float(code)


def conditions_s7(t: pd.DataFrame) -> pd.DataFrame:
    cal, cyc = t["age_type"] == 1, t["age_type"] == 2
    return pd.DataFrame({
        "temp_c": t["age_temp"].astype(float),
        "charge_c_rate": np.where(cyc, t["age_chg_rate"], np.where(cal, 0.0, np.nan)),
        "discharge_c_rate": np.where(cyc, t["age_dischg_rate"], np.where(cal, 0.0, np.nan)),
        "dod": np.where(cal, 0.0, t["age_profile"].map(S7_PROFILE_DOD).astype(float)),
        "v_charge_max": np.where(cyc, t["v_max_target_V"], np.nan),
        "cv_charge": np.nan,
        "calendar": cal.astype(float),
        "rest_soc": np.where(cal, t["age_soc"] / 100.0, np.nan),
        "chemistry": "NMC",
    }, index=t.index)


def conditions_s8(t: pd.DataFrame) -> pd.DataFrame:
    batch = t["condition"].str.replace("S8-", "", regex=False)
    vals = np.array([S8_CONDITIONS[b] for b in batch], dtype=float).reshape(-1, 3)
    return pd.DataFrame({"temp_c": np.nan, "charge_c_rate": vals[:, 0], "discharge_c_rate": vals[:, 1],
                         "dod": vals[:, 2], "v_charge_max": 4.2, "cv_charge": 1.0, "calendar": 0.0, "rest_soc": np.nan,
                         "chemistry": "NMC"}, index=t.index)


def conditions_s9(t: pd.DataFrame) -> pd.DataFrame:
    m = t["condition"].str.extract(r"S9-(D\d)_.*-CY(\d+)-(\d+)_(\d+)$")
    if m.isna().any().any():
        raise ValueError("unparsed S9 condition names")
    return pd.DataFrame({"temp_c": m[1].astype(float), "charge_c_rate": m[2].map(_rate),
                         "discharge_c_rate": m[3].map(_rate), "dod": 1.0, "v_charge_max": 4.2, "cv_charge": 1.0,
                         "calendar": 0.0, "rest_soc": np.nan,
                         "chemistry": m[0].map(S9_CHEM)}, index=t.index)


def conditions_s6(t: pd.DataFrame) -> pd.DataFrame:
    tc = W.TEST_CASES.set_index("Battery_ID")
    case = t["condition"]
    return pd.DataFrame({"temp_c": 50.0, "charge_c_rate": case.map(tc["charge_c_rate"]).astype(float),
                         "discharge_c_rate": np.where(case.map(tc["cc2c_discharge"]) == 1, 2.0, 1 / 3),
                         "dod": case.map(tc["dod"]).astype(float), "v_charge_max": case.map(tc["v_max"]).astype(float),
                         "cv_charge": case.map(tc["cccv_charge"]).astype(float), "calendar": 0.0, "rest_soc": np.nan,
                         "chemistry": "LFP"}, index=t.index)


CONDITIONS = {"S7": conditions_s7, "S8": conditions_s8, "S9": conditions_s9, "S6": conditions_s6}


def harmonised_samples(checks: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """D21 samples of one dataset with the shared inputs and the window keys the TFT needs.

    ``Battery_ID`` is the task (the dataset for the pool, the test case for S6), ``Cycling_Test``
    identifies (cell, horizon) and ``Cycle_Index`` the origin position, so a TFT window holds earlier
    origins of the same cell and horizon only.
    """
    cond = CONDITIONS[dataset](checks)
    # The documented mapping above always wins over any same-named column of the check table.
    t = checks.drop(columns=[c for c in cond.columns if c in checks.columns]).join(cond)
    static = ["temp_c", "charge_c_rate", "discharge_c_rate", "dod", "v_charge_max", "cv_charge", "calendar", "rest_soc",
              "chemistry"]
    s = pool.build_samples(t, static=static)
    for c in CHEMISTRIES:
        s[f"chem_{c}"] = (s["chemistry"] == c).astype(float)
    for c in ("cycles_since_first", "cycles_to_target"):
        if c not in s:
            s[c] = np.nan
    s["Battery_ID"] = s["condition"] if dataset == "S6" else dataset
    s["Cycling_Test"] = s["cell"].astype(str) + "_h" + s["horizon"].astype(str)
    s["Cycle_Index"] = s["origin_pos"].astype(int)
    return s


def s6_checks(wheeler_checks: pd.DataFrame) -> pd.DataFrame:
    """S6 check table in the pool layout (valid 50 degC checks only; label as D20)."""
    v = wheeler_checks[wheeler_checks["valid"]].copy()
    v = v.assign(dataset="S6", condition=v["Battery_ID"], check_time=pd.to_datetime(v["check_time"]))
    return pool.add_soh(v[["dataset", "cell", "condition", "cycle", "check_time", "Capacity_Ah"]])


def pool_task(samples: pd.DataFrame) -> P.Task:
    """The whole pool as one task with per-cell purged splits (D18 rule, unchanged)."""
    sp = FC.assign_split(samples)
    parts = {k: sp[sp["split"] == k].drop(columns="split").reset_index(drop=True) for k in ("train", "val", "test")}
    return P.Task(name="pool", **parts)


def pretrain(strategy, task: P.Task):
    """Fit ``strategy`` on ``task`` with its own code path (see module docstring); returns it."""
    strategy._initialise(task.train)
    strategy._before_task(task)
    strategy._train(strategy._training_frame(task), task.val, task_index=0)
    strategy._after_task(task)
    strategy.tasks_seen_.append(task.name)
    return strategy


def run_stream(strategy, stream: Sequence[P.Task], pool: P.Task | None, eval_split: str = "test",
               metric: str = "RMSE") -> dict:
    """Continue ``strategy`` (pretrained or not) through the S6 stream; evaluate every task after every step.

    Reference errors ``b_j`` are those of the strategy before the stream (the pretrained model, or an
    untrained one), so forward transfer measures what the stream itself adds. With a pool, the pool's
    ``eval_split`` error is recorded before and after the stream (retention of first-life knowledge).
    """
    if strategy.model is None:
        ref = copy.deepcopy(strategy).initialise(stream[0])
    else:
        ref = strategy
    b = {t.name: ev.task_error(getattr(t, eval_split)[TARGET], ref.predict(getattr(t, eval_split)), metric)
         for t in stream}
    pool_err = lambda s: (ev.task_error(getattr(pool, eval_split)[TARGET], s.predict(getattr(pool, eval_split)), metric)
                          if pool is not None and s.model is not None else np.nan)
    pool_before = pool_err(strategy)
    errors = {}
    for i, task in enumerate(stream):
        strategy.fit_task(task)
        for t in stream:
            frame = getattr(t, eval_split)
            errors[(i, t.name)] = ev.task_error(frame[TARGET], strategy.predict(frame), metric)
    R = ev.accuracy_matrix(errors, [t.name for t in stream])
    return {"R": R, "metrics": ev.cl_metrics(R, reference_errors=b), "reference_errors": b,
            "pool_error_before": pool_before, "pool_error_after": pool_err(strategy),
            "train_time_s": list(strategy.training_time_), "config": strategy.config()}


def build_strategy(method: str, features: Sequence[str], seed: int, history: pd.DataFrame | None,
                   settings: Mapping) -> object:
    """``method`` is the D18 strategy name (``finetune``, ``ewc``, ``tft_finetune``, ``tft_ewc``)."""
    kwargs = dict(settings)
    if method.startswith("tft"):
        kwargs.update(history=history)
    return make_strategy(method, features, seed=seed, target=TARGET, **kwargs)
