"""S6 between-check ageing cycles (decision D22): per-cycle measured responses and per-block aggregates.

Each ageing file holds one cell's block of 100 cycles (``..._agingAAAAtoBBBB.csv``); the check at cycle *c*
follows the block that ends at *c*. Step types come from each step's median current. A cycle is valid if all its
charge steps precede all its discharge steps and a rest step follows the last of each (D22 amendments, as
corrected on 5 Oct 2026 after the step structure of late-life cycles was seen). Only the seven
measured responses below are inputs. ``q_discharged`` and ``t_end`` are kept for the leakage audit and the
causality check only.
"""

from __future__ import annotations

import gzip
import lzma
import re
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["DateTime", "Step_Index", "Cycle_Index", "Current", "Voltage", "Step_Time", "Discharge_Capacity",
           "Internal_Resistance"]
RESPONSES: tuple[str, ...] = ("v_end_charge", "cv_seconds", "cv_end_current", "v_end_discharge",
                              "relax_after_charge", "relax_after_discharge", "internal_resistance")
AGGREGATES: tuple[str, ...] = ("mean", "last", "slope")
CURRENT_EPS = 0.01          # A: median current above +eps is charge, below -eps discharge, else rest
CV_TOL_V = 0.002            # V below the case's charge limit that counts as the CV hold
MIN_VALID_CYCLES = 50       # per block, else the block's aggregates are missing
EXCEL_EPOCH = pd.Timestamp("1899-12-30")
BLOCK_NAME = re.compile(r"cell(\w+?)_aging(\d{4})to(\d{4})\.csv", re.IGNORECASE)


def feature_names() -> list[str]:
    return [f"age_{r}_{a}" for r in RESPONSES for a in AGGREGATES]


def block_range(path: str | Path) -> tuple[str, int, int]:
    """(cell, first cycle, last cycle) from a file name such as ``WWH_18650_cell1a_aging0101to0200.csv.xz``."""
    m = BLOCK_NAME.search(Path(path).name)
    if not m:
        raise ValueError(f"not an ageing block file: {path}")
    return f"Cell{m.group(1)}", int(m.group(2)), int(m.group(3))


def read_block(path: str | Path) -> pd.DataFrame:
    s = str(path)
    opener = lzma.open if s.endswith(".xz") else gzip.open if s.endswith(".gz") else open   # early files are .gz
    with opener(path, "rb") as fh:
        return pd.read_csv(fh, sep=";", usecols=COLUMNS, dtype={c: "float64" for c in COLUMNS})


def _segments(raw: pd.DataFrame, v_limit: float) -> pd.DataFrame:
    """One row per contiguous step (Step_Index run) of each cycle, in time order."""
    r = raw.reset_index(drop=True)
    new = (r["Step_Index"] != r["Step_Index"].shift()) | (r["Cycle_Index"] != r["Cycle_Index"].shift())
    r = r.assign(seg=new.cumsum(), at_limit=r["Voltage"] >= v_limit - CV_TOL_V)
    g = r.groupby("seg", sort=True)
    seg = g.agg(cycle=("Cycle_Index", "first"), i_med=("Current", "median"), v_first=("Voltage", "first"),
                v_last=("Voltage", "last"), i_last=("Current", "last"), t_end=("DateTime", "max"),
                step_t_end=("Step_Time", "max"))
    first_at = r[r["at_limit"]].groupby("seg")["Step_Time"].min()
    seg["cv_seconds"] = (seg["step_t_end"] - first_at.reindex(seg.index)).fillna(0.0)
    seg["kind"] = np.where(seg["i_med"] > CURRENT_EPS, "C", np.where(seg["i_med"] < -CURRENT_EPS, "D", "R"))
    return seg


def cycle_responses(raw: pd.DataFrame, first_cycle: int, v_limit: float, cccv: bool) -> pd.DataFrame:
    """Per-cycle validity and responses of one block (``first_cycle``: the block's first global cycle)."""
    seg = _segments(raw, v_limit)
    ir = raw[raw["Internal_Resistance"] != 0].groupby("Cycle_Index")["Internal_Resistance"].last()
    qd = raw.groupby("Cycle_Index")["Discharge_Capacity"].max()
    rows = []
    for cyc, s in seg.groupby("cycle", sort=True):
        s = s.reset_index(drop=True)
        kinds = list(s["kind"])
        c_idx = [i for i, k in enumerate(kinds) if k == "C"]
        d_idx = [i for i, k in enumerate(kinds) if k == "D"]
        # Valid: all charge segments before all discharge segments, and a rest step right after the last of each
        # (late-life cycles split the charge and the discharge into several steps; D22 amendment of 5 Oct 2026).
        valid = (bool(c_idx) and bool(d_idx) and c_idx[-1] < d_idx[0]
                 and c_idx[-1] + 1 < len(kinds) and kinds[c_idx[-1] + 1] == "R"
                 and d_idx[-1] + 1 < len(kinds) and kinds[d_idx[-1] + 1] == "R")
        out = {"global_cycle": int(first_cycle - 1 + cyc), "valid": valid, "t_end": float(s["t_end"].max()),
               "q_discharged": float(qd.get(cyc, np.nan))}
        if valid:
            ch, r1 = s.iloc[c_idx[-1]], s.iloc[c_idx[-1] + 1]
            d, r2 = s.iloc[d_idx[-1]], s.iloc[d_idx[-1] + 1]
            cv = float(ch["cv_seconds"]) if cccv else np.nan
            out.update({"v_end_charge": float(ch["v_last"]), "cv_seconds": cv,
                        "cv_end_current": float(ch["i_last"]) if cccv and cv > 0 else np.nan,
                        "v_end_discharge": float(d["v_last"]),
                        "relax_after_charge": float(r1["v_last"] - r1["v_first"]),
                        "relax_after_discharge": float(r2["v_last"] - r2["v_first"]),
                        "internal_resistance": float(ir.get(cyc, np.nan))})
        rows.append(out)
    return pd.DataFrame(rows)


def block_features(cycles: pd.DataFrame) -> dict[str, float]:
    """Mean, last value and least-squares slope against global cycle of each response over the valid cycles."""
    v = cycles[cycles["valid"]].sort_values("global_cycle")
    out: dict[str, float] = {"age_valid_cycles": float(len(v)), "age_invalid_cycles": float((~cycles["valid"]).sum())}
    enough = len(v) >= MIN_VALID_CYCLES
    for r in RESPONSES:
        x = v[["global_cycle", r]].dropna() if enough and r in v else pd.DataFrame()
        out[f"age_{r}_mean"] = float(x[r].mean()) if len(x) else np.nan
        out[f"age_{r}_last"] = float(x[r].iloc[-1]) if len(x) else np.nan
        out[f"age_{r}_slope"] = float(np.polyfit(x["global_cycle"], x[r], 1)[0]) if len(x) >= 2 else np.nan
    return out


def excel_time(serial: float | np.ndarray) -> pd.Timestamp | pd.DatetimeIndex:
    """Arbin ``DateTime`` (Excel serial days) as a timestamp."""
    return EXCEL_EPOCH + pd.to_timedelta(serial, unit="D")
