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
        cyc = g["cycle"].to_numpy(float) if "cycle" in g else None     # planned cycle count (S8, S9)
        for i in range(len(g)):
            rate = float(np.polyfit(days[i - 2:i + 1], soh[i - 2:i + 1], 1)[0]) if i >= 2 else np.nan
            base = {"dataset": g["dataset"].iloc[0], "cell": cell, "condition": g["condition"].iloc[0],
                    "origin_pos": i, "n_valid_rpts": len(g), "Cycle_Date": g["check_time"].iloc[i],
                    "soh_origin": soh[i], "days_since_first": days[i], "fade_rate_pp_per_day": rate,
                    **{c: g[c].iloc[0] for c in static},
                    **({"cycles_since_first": cyc[i] - cyc[0]} if cyc is not None else {})}
            for h in horizons:
                j = i + h
                if j < len(g):
                    rows.append({**base, "horizon": h, "target_pos": j, "days_to_target": days[j] - days[i],
                                 **({"cycles_to_target": cyc[j] - cyc[i]} if cyc is not None else {}),
                                 "soh_target": soh[j], TARGET: soh[j] - soh[i]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# S8 (XJTU) and S9 (Tongji): checks from per-cycle data (D21 disclosures, 3 Oct 2026)
# ---------------------------------------------------------------------------

BLOCK_CYCLES = 10                       # regular full-discharge cycles averaged into one check
CUTOFF_TOL_V = 0.02
CURRENT_TOL = 0.02                       # relative tolerance on a cycle's median discharge current (S9)
FULL_DISCHARGE_FRACTION = 0.5           # of the cell's median per-cycle capacity (S9 validity)


def block_checks(meas: pd.DataFrame, block: int) -> pd.DataFrame:
    """Group consecutive valid measurement cycles into checks (mean capacity; time and cycle of the last one).

    ``meas`` has ``cycle``, ``capacity_ah`` and ``elapsed_s`` (time since the cell's start), in cycle order.
    An incomplete final block is dropped, so every check averages exactly ``block`` measurements.
    """
    m = meas.reset_index(drop=True)
    n = len(m) // block * block
    m = m.iloc[:n].assign(block=np.arange(n) // block)
    g = m.groupby("block")
    return pd.DataFrame({"cycle": g["cycle"].last(), "Capacity_Ah": g["capacity_ah"].mean(),
                         "elapsed_s": g["elapsed_s"].last()}).reset_index(drop=True)


S8_BATCH_RULES: dict[str, tuple[str, int]] = {
    # batch: (which cycles are capacity measurements, cycles per check)
    "Batch-1": ("1C full discharge", BLOCK_CYCLES), "Batch-2": ("1C full discharge", BLOCK_CYCLES),
    "Batch-3": ("1C full discharge", BLOCK_CYCLES),
    "Batch-4": ("test capacity", 1), "Batch-5": ("test capacity", 1), "Batch-6": ("test capacity", 1),
}


def s8_cell_measurements(mat, batch: str) -> pd.DataFrame:
    """Valid capacity-measurement cycles of one S8 cell (authors' per-cycle discharge capacity).

    The first cycle (initial 0.2C test) is never used. Batches 1–3: cycles described exactly as a
    1C discharge (full, to 2.5 V); batches 4–6: cycles labelled "[test capacity]".
    """
    rule, _ = S8_BATCH_RULES[batch]
    data, dcap = np.atleast_1d(mat["data"]), np.asarray(mat["summary"].discharge_capacity_Ah, float)
    t0 = pd.to_datetime(np.atleast_1d(data[0].system_time)[0], format="%Y-%m-%d,%H:%M:%S")
    rows = []
    for i, c in enumerate(data):
        desc = str(c.description)
        if i == 0 or "0.2C discharge" in desc:
            continue
        is_meas = ("[test capacity]" in desc) if rule == "test capacity" else desc.endswith("and 1C discharge")
        if not is_meas or np.min(c.voltage_V) > 2.5 + CUTOFF_TOL_V:
            continue
        t_end = pd.to_datetime(np.atleast_1d(c.system_time)[-1], format="%Y-%m-%d,%H:%M:%S")
        rows.append({"cycle": i, "capacity_ah": float(dcap[i]), "elapsed_s": (t_end - t0).total_seconds()})
    return pd.DataFrame(rows, columns=["cycle", "capacity_ah", "elapsed_s"])


def s8_check_table(zip_path: Path | str) -> pd.DataFrame:
    import io

    import scipy.io as sio

    out = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(n for n in zf.namelist() if n.endswith(".mat") and "/Batch-" in n):
            batch, fname = name.split("/")[-2], name.split("/")[-1][:-4]
            mat = sio.loadmat(io.BytesIO(zf.read(name)), squeeze_me=True, struct_as_record=False)
            checks = block_checks(s8_cell_measurements(mat, batch), S8_BATCH_RULES[batch][1])
            out.append(checks.assign(dataset="S8", cell=f"S8-{batch}-{fname}", condition=f"S8-{batch}"))
    return _finish(pd.concat(out, ignore_index=True))


S9_FILE = re.compile(r"CY(\d+)-(\d+)_(\d+)-#(\d+)\.csv$")
S9_CHEMISTRY: dict[str, str] = {"Dataset_1_NCA_battery": "NCA", "Dataset_2_NCM_battery": "NCM",
                                "Dataset_3_NCM_NCA_battery": "NCM+NCA"}      # Zhu et al. 2022, abstract


def s9_cell_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cycle discharge capacity of one S9 cell; a cycle counts only if it discharges to the cell's cut-off.

    In three files the time counter restarts once while cycle numbers continue; time is made continuous
    by carrying the elapsed time across the restart (the length of any pause there is not recorded).
    ``attrs["time_resets"]`` gives the number of restarts.
    """
    t = df["time/s"].to_numpy(float)
    jumps = np.flatnonzero(np.diff(t) < 0)
    offset = np.zeros_like(t)
    for j in jumps:
        offset[j + 1:] += t[j] - t[j + 1]
    df = df.assign(**{"time/s": t + offset})
    g = df.groupby("cycle number")
    per = pd.DataFrame({"capacity_ah": g["Q discharge/mA.h"].max() / 1000, "v_min": g["Ecell/V"].min(),
                        "elapsed_s": g["time/s"].max()})
    if "<I>/mA" in df:
        # Regular cycles discharge at one current per cell. Periodic reference cycles (e.g. every 50th cycle of
        # the blend cells: mostly 2.5 A for ~15 h instead of 5 A for ~4 h) are not comparable and are dropped.
        i_dis = df.loc[df["<I>/mA"] < -10].groupby("cycle number")["<I>/mA"].median()
        standard = i_dis.round(-1).mode().iloc[0]
        per = per[(i_dis.reindex(per.index) - standard).abs() <= CURRENT_TOL * abs(standard)]
        # A regular cycle has exactly one discharge step; reference cycles at the standard current but with
        # two discharge steps (every 50th cycle of blend condition CY25-05_1) accumulate more charge than the cell holds.
        dis = (df["<I>/mA"] < -10).astype(int)
        starts = (dis.diff() == 1) | ((df["cycle number"].diff() != 0) & (dis == 1))
        steps = starts.groupby(df["cycle number"]).sum()
        per = per[steps.reindex(per.index) == 1]
    cutoff = per["v_min"].round(2).mode().iloc[0]          # 2.65 V (NCA) or 2.5 V (NCM, blend)
    # Some "cycles" are short special segments (e.g. every 26th cycle in some NCA files discharges ~22 mAh of
    # ~3,100) that still touch the cut-off; a full discharge must deliver at least half of the cell's median.
    floor = FULL_DISCHARGE_FRACTION * per["capacity_ah"].median()
    per = per[(per["v_min"] <= cutoff + CUTOFF_TOL_V) & (per["capacity_ah"] >= floor)]
    out = per.reset_index(names="cycle")[["cycle", "capacity_ah", "elapsed_s"]]
    out.attrs["time_resets"] = len(jumps)
    return out


def s9_check_table(zip_paths: Sequence[Path | str], workers: int = 1) -> pd.DataFrame:
    """Check table of all S9 cells; with ``workers`` > 1 the archives are read in parallel."""
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as ex:
            parts = list(ex.map(_s9_zip_checks, [str(z) for z in zip_paths]))
    else:
        parts = [_s9_zip_checks(str(z)) for z in zip_paths]
    return _finish(pd.concat(parts, ignore_index=True))


def _s9_zip_checks(zp: str) -> pd.DataFrame:
    """Checks of every cell in one S9 archive."""
    out = []
    with zipfile.ZipFile(zp) as zf:
        for name in sorted(n for n in zf.namelist() if n.endswith(".csv")):
            m = S9_FILE.search(name)
            if not m:
                continue
            df = pd.read_csv(zf.open(name), usecols=["time/s", "Ecell/V", "Q discharge/mA.h", "cycle number", "<I>/mA"])
            meas = s9_cell_measurements(df)
            checks = block_checks(meas, BLOCK_CYCLES).assign(time_resets=meas.attrs["time_resets"])
            cond = f"S9-{name.split('/')[0].replace('Dataset_', 'D')}-CY{m[1]}-{m[2]}_{m[3]}"
            out.append(checks.assign(dataset="S9", cell=f"{cond}-#{m[4]}", condition=cond,
                                     temp_c=float(m[1]), chemistry=S9_CHEMISTRY[name.split("/")[0]]))
    return pd.concat(out, ignore_index=True)


def _finish(t: pd.DataFrame) -> pd.DataFrame:
    """Common layout: check_time as elapsed time (no absolute date is needed), SOH, check counts."""
    t = t.assign(check_time=pd.Timestamp("2000-01-01") + pd.to_timedelta(t["elapsed_s"], unit="s"))
    return add_soh(t)
