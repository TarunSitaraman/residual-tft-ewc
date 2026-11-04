"""Run decision D36: the D34 protocol on S10 (Stroebl et al. 2024) with shrinkage only where extrapolation has a
fade rate (SEL).

Everything not stated here is ``run_d34.py`` unchanged (parts, tuning, comparators, tests over held-out conditions,
D21 replication rule). D36 changes: the data (``data/pool_s10_checks.csv``, ``scripts/build_stroebl_checks.py``), the
S10 condition mapping with the extra input ``soc_max``, the condition sets (seed 36, stratified by ageing type, stage
and temperature), and the proposed forecast SEL = shrunk where linear extrapolation has a fade rate (H+), raw
elsewhere. Primary tests: SEL - GBM-LAST, SEL - linear extrapolation on H+, SEL - GBM-ALL.

    python scripts/run_d36.py --write-sets
    python scripts/run_d36.py --exp e1 --part tune1_f0
    python scripts/run_d36.py --exp e1 --collect-only
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_d26  # noqa: E402
import run_d34 as B  # noqa: E402
from src import p3, pool  # noqa: E402

SEL = "tft_residual_ewc_selective"
SET_SEED = 36
EXTRA = ["soc_max"]                                   # S10 only: the maximum SOC (storage SOC or cycling window top)


def condition_table(checks: pd.DataFrame) -> pd.DataFrame:
    c = checks[checks["n_valid_checks"] >= 5]
    return (c.groupby("condition", sort=False)
             .agg(age_type=("age_type", "first"), stage=("stage", "first"), age_temp=("age_temp", "first"),
                  lab=("lab", "first"), cells=("cell", "nunique"))
             .reset_index())


def conditions_s10(t: pd.DataFrame) -> pd.DataFrame:
    """D36 mapping onto the D21 shared inputs; undefined inputs are missing, never guessed."""
    cal = t["age_type"] == 1
    return pd.DataFrame({
        "temp_c": t["age_temp"].astype(float),
        "charge_c_rate": np.where(cal, 0.0, t["age_chg_rate"]).astype(float),
        "discharge_c_rate": np.where(cal, 0.0, t["age_dischg_rate"]).astype(float),
        "dod": np.where(cal, 0.0, t["dod"]).astype(float),
        "v_charge_max": np.nan, "cv_charge": np.nan,
        "calendar": cal.astype(float),
        "rest_soc": np.where(cal, t["soc_max"], np.nan).astype(float),
        "soc_max": t["soc_max"].astype(float),
        "chemistry": "none",                          # one cell type: no chemistry flag is set
    }, index=t.index)


class Runner(B.Runner):
    NAME, DECISION, CHECKS = "d36", "D36", "pool_s10_checks.csv"
    CONT = {B.RAW: "raw", B.SHR: "shrunk", SEL: "sel"}
    PRIMARY = [(SEL, B.LAST, "rmse"), (SEL, B.LIN, "rmse_hplus"), (SEL, B.ALL, "rmse")]
    SECONDARY = [(SEL, B.SHR, "rmse"), (SEL, B.RAW, "rmse"), (SEL, B.LAST, "rmse_hplus"), (SEL, B.LAST, "rmse_hminus"),
                 (SEL, B.ALL, "rmse_hplus"), (SEL, B.ALL, "rmse_hminus"), (SEL, B.LIN, "rmse"), (B.RAW, B.JOINT, "rmse"),
                 (B.LAST, B.ALL, "rmse")]
    DESC_KEYS = ("age_type", "age_temp", "stage", "lab")

    @staticmethod
    def samples(checks: pd.DataFrame, horizons) -> tuple[pd.DataFrame, list[str]]:
        """As ``p3.harmonised_samples`` (D21 samples and window keys), with the S10 condition mapping."""
        cond = conditions_s10(checks)
        t = checks.drop(columns=[c for c in cond.columns if c in checks.columns]).join(cond)
        s = pool.build_samples(t, static=list(cond.columns), horizons=horizons)
        for c in p3.CHEMISTRIES:
            s[f"chem_{c}"] = 0.0
        for c in ("cycles_since_first", "cycles_to_target"):
            s[c] = np.nan
        s["Cycling_Test"] = s["cell"].astype(str) + "_h" + s["horizon"].astype(str)
        s["Cycle_Index"] = s["origin_pos"].astype(int)
        return s, list(p3.SHARED_FEATURES) + EXTRA

    @staticmethod
    def make_sets(checks: pd.DataFrame) -> pd.DataFrame:
        return B.assign_sets(condition_table(checks), seed=SET_SEED, keys=("age_type", "stage", "age_temp"))

    @staticmethod
    def unit_scores(frame, direct, raw, shr, hp) -> dict:
        sel = np.where(hp, shr, raw)                  # D36: shrink only where extrapolation has a fade rate
        return {**B.Runner.unit_scores(frame, direct, raw, shr, hp), "sel": run_d26.scores(frame, direct, sel, hp)}


if __name__ == "__main__":
    sys.exit(B.main(Runner))
