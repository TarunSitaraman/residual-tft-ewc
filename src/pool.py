"""First-life pretraining pool (decision D21): check tables and forecasting samples.

S7 (Luh and Blank 2024, KIT) is implemented here; S8 and S9 follow the same output layout.

One row per (cell, valid check) with the capacity label, the check time and the cell's
static ageing conditions; ``build_samples`` turns a check table into D21 forecasting
samples (origin check k, target k + h, target ΔSOH in pp).

Leakage rules (D14, D21):
- the label is the *measured* check-up discharge capacity, never the extrapolated
  ``cap_aged_est_Ah``;
- charge throughput is never an input: in cycling cells it shrinks with capacity, so it
  partly encodes the label, and future throughput is not known at the origin;
- time to target is in calendar days, taken from the check-up schedule.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

HORIZONS: tuple[int, ...] = (1, 2, 3)
MIN_VALID_CHECKS = 5                       # D21: three for the fade rate plus at least one target
TARGET = "dsoh"

#: S7: check-ups after the thermal-management failure (27 May 2024) are excluded (Readme v2).
S7_LAST_VALID = pd.Timestamp("2024-05-27")
S7_CHECKUP = 2                             # cyc_condition CHECKUP_RT (Data Structure v05.3, StateValues)
S7_CU_DISCHARGE_A = -1.0                   # check-up capacity discharge current
#: Check-ups run at room temperature (25 degC). Three check-ups (CU 11 of three cells aged at 0 degC)
#: started at about 1 degC and read 18 pp low; check-ups starting below this are excluded (D21 disclosures).
S7_MIN_CHECKUP_TEMP_C = 20.0
S7_FILE = re.compile(r"cell_eocv2_P(\d{3})_(\d)_S(\d+)_C(\d+)\.csv$")
#: Static ageing conditions, read from the regular-operation rows (not the check-up rows).
S7_STATIC: tuple[str, ...] = ("age_type", "age_temp", "age_soc", "age_chg_rate", "age_dischg_rate", "age_profile",
                              "v_max_target_V", "v_min_target_V")


def s7_cell_checks(eoc: pd.DataFrame, cell: str, condition: str) -> pd.DataFrame:
    """Valid check-ups of one S7 cell, in time order, with its static conditions.

    A check-up counts if it is a check-up discharge run (``cyc_condition`` = CHECKUP_RT,
    ``cyc_charged`` = 0) at the check-up current, dated before the thermal failure, with a
    finite discharged charge, starting at room temperature (>= 20 degC). The stored ``delta_q_dischg_Ah`` is negative although the data
    dictionary says it is non-negative; its absolute value is the capacity (D21).
    """
    cu = eoc[(eoc["cyc_condition"] == S7_CHECKUP) & (eoc["cyc_charged"] == 0)].copy()
    cu["check_time"] = pd.to_datetime(cu["timestamp_s"], unit="s")
    cu = cu[(cu["check_time"] < S7_LAST_VALID) & np.isclose(cu["i_dischg_min_A"], S7_CU_DISCHARGE_A)
            & (cu["t_start_degC"] >= S7_MIN_CHECKUP_TEMP_C) & cu["delta_q_dischg_Ah"].notna()]
    op = eoc[eoc["cyc_condition"] == 0]
    static = (op if len(op) else eoc)[list(S7_STATIC)].iloc[0] if len(eoc) else pd.Series(index=S7_STATIC, dtype=float)
    out = pd.DataFrame({"dataset": "S7", "cell": cell, "condition": condition,
                        "check_index": cu["num_cycles_checkup"].to_numpy(int),
                        "check_time": cu["check_time"].to_numpy(),
                        "Capacity_Ah": cu["delta_q_dischg_Ah"].abs().to_numpy(float),
                        "check_temp_c": cu["t_start_degC"].to_numpy(float)})
    for col in S7_STATIC:
        out[col] = static[col]
    return out.sort_values("check_time").reset_index(drop=True)


def s7_check_table(eoc_zip: Path | str) -> pd.DataFrame:
    """Check table of all S7 cells (side-experiment records ``P000`` excluded), with SOH."""
    rows = []
    with zipfile.ZipFile(eoc_zip) as zf:
        for name in sorted(zf.namelist()):
            m = S7_FILE.search(name)
            if not m or m.group(1) == "000":
                continue
            eoc = pd.read_csv(zf.open(name), sep=";")
            rows.append(s7_cell_checks(eoc, cell=f"P{m.group(1)}-{m.group(2)}", condition=f"P{m.group(1)}"))
    return add_soh(pd.concat(rows, ignore_index=True))


def add_soh(table: pd.DataFrame) -> pd.DataFrame:
    """``SOH_Percent = Capacity_Ah / Reference_Capacity_Ah * 100``; reference = the cell's first valid check."""
    t = table.sort_values(["cell", "check_time"]).copy()
    t["Reference_Capacity_Ah"] = t.groupby("cell")["Capacity_Ah"].transform("first")
    t["SOH_Percent"] = t["Capacity_Ah"] / t["Reference_Capacity_Ah"] * 100
    t["n_valid_checks"] = t.groupby("cell")["cell"].transform("size")
    return t.reset_index(drop=True)


def build_samples(table: pd.DataFrame, static: Sequence[str], horizons: Sequence[int] = HORIZONS,
                  min_checks: int = MIN_VALID_CHECKS) -> pd.DataFrame:
    """D21 samples: origin check k, target k + h; inputs known at k; target ΔSOH (pp)."""
    rows = []
    for cell, g in table[table["n_valid_checks"] >= min_checks].groupby("cell", sort=True):
        g = g.sort_values("check_time").reset_index(drop=True)
        days = ((g["check_time"] - g["check_time"].iloc[0]).dt.total_seconds() / 86400).to_numpy()
        soh = g["SOH_Percent"].to_numpy(float)
        for i in range(len(g)):
            rate = float(np.polyfit(days[i - 2:i + 1], soh[i - 2:i + 1], 1)[0]) if i >= 2 else np.nan
            base = {"dataset": g["dataset"].iloc[0], "cell": cell, "condition": g["condition"].iloc[0],
                    "origin_pos": i, "n_valid_rpts": len(g), "Cycle_Date": g["check_time"].iloc[i],
                    "soh_origin": soh[i], "days_since_first": days[i], "fade_rate_pp_per_day": rate,
                    **{c: g[c].iloc[0] for c in static}}
            for h in horizons:
                j = i + h
                if j < len(g):
                    rows.append({**base, "horizon": h, "target_pos": j, "days_to_target": days[j] - days[i],
                                 "soh_target": soh[j], TARGET: soh[j] - soh[i]})
    return pd.DataFrame(rows)
