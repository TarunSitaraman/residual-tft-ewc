"""Second dataset (S6, D19): Wheeler et al. 2025, 20 A123 18650 Graphite/LFP cells.

Characterisation checks every 100 cycles (fetched by
``scripts/fetch_wheeler_characterisation.py``) play the role of RPTs. Each
check (Wheeler et al. 2025, Methods): C/3 discharge (step 11), rest, C/3 charge
(13), rest, C/25 discharge (16), rest, C/25 charge (18), rest. Ageing and
intermediate checks at 50 degC; the initial and final characterisations at
25 degC are not used (different temperature; decision D20).

Labels follow D17: capacity is the slow (C/25) discharge capacity, the SOH
reference is the cell's first 50 degC check. Features are measured at the check
itself; nothing uses a later check. Samples are laid out for the forecasting
runners (``src/forecasting.py``) with the **test case** as the task
(``Battery_ID``) and the cell kept in ``cell``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from src import data_pipeline as dp
from src import forecasting as FC

RAW_SUBDIR = Path("raw") / "wheeler2025"
NOMINAL_AH = 1.1

#: Wheeler et al. 2025, Table 2: charge type (1 = CC-CV, 0 = CC only), end-of-charge
#: voltage (V), charge C-rate, discharge (0 = WLTP at C/3 on average, 1 = CC 2C), DoD.
TEST_CASES: pd.DataFrame = pd.DataFrame(
    [("case1", 1, 3.65, 1 / 3, 0, 0.7), ("case2", 1, 3.55, 1 / 3, 0, 0.7), ("case3", 0, 3.65, 1 / 3, 0, 0.7),
     ("case4", 1, 3.65, 2.0, 0, 0.7), ("case5", 1, 3.55, 2.0, 0, 0.7), ("case6", 1, 3.65, 2.0, 1, 0.7),
     ("case7", 0, 3.65, 1 / 3, 0, 1.0)],
    columns=[dp.GROUP_COL, "cccv_charge", "v_max", "charge_c_rate", "cc2c_discharge", "dod"],
)
STATIC_FEATURES: tuple[str, ...] = ("cccv_charge", "v_max", "charge_c_rate", "cc2c_discharge", "dod")

#: IC peak windows (V) of the C/25 discharge, from the fresh-cell (cycle 0, 50 degC)
#: curves only: graphite staging peaks near 3.195, 3.22, 3.245 V; main LFP/graphite
#: peak near 3.30 V; the 3.335 V peak that vanishes with lithium-inventory loss.
IC_WINDOWS: dict[str, tuple[float, float]] = {
    "p1": (3.175, 3.21), "p2": (3.21, 3.235), "p3": (3.235, 3.275), "p4": (3.28, 3.315), "p5": (3.315, 3.36),
}
IC_GRID = np.round(np.arange(2.90, 3.50, 0.001), 3)
IC_SMOOTH_MV = 5.0
#: Charge windows (V) of the C/25 discharge: above, on and below the main plateau.
Q_WINDOWS: tuple[tuple[float, float], ...] = ((3.55, 3.315), (3.315, 3.27), (3.27, 2.5))   # (from, to) in discharge
#: A step is complete only if it reaches its protocol cut-off (article, Table 3):
#: discharges end at 2.5 V, charges at 3.65 V (3.55 V is only the ageing limit of cases 2 and 5).
DISCHARGE_CUTOFF_V, CHARGE_CUTOFF_V, CUTOFF_TOL_V = 2.5, 3.65, 0.01
#: A discharge counts only if it starts from a full, rested cell. Fresh checks start
#: at 3.51-3.53 V; some C/3 discharges start mid-way (e.g. 3.24 V) because the check
#: began where an ageing cycle ended.
FULL_START_V = 3.40


def cell_case(cell: str) -> str:
    """``"Cell4b"`` or ``"cell4b"`` -> ``"case4"``."""
    return "case" + re.search(r"(\d)", cell).group(1)


def read_check(path: Path | str) -> pd.DataFrame:
    """One characterisation file (semicolon-separated Arbin export, optionally gzip-compressed), in time order.

    The exported rows are not in time order (rows of different steps are
    interleaved), so they are sorted by ``Test_Time``; otherwise the first and
    last rows of a step are not its start and end.
    """
    ts = pd.read_csv(path, sep=";", usecols=["Test_Time", "DateTime", "Step_Index", "Step_Time", "Current", "Voltage",
                                              "Charge_Capacity", "Discharge_Capacity"])
    return ts.sort_values("Test_Time", kind="mergesort").reset_index(drop=True)


def _step(ts: pd.DataFrame, k: int, discharge: bool) -> pd.DataFrame:
    """Rows of step ``k`` if the step is complete (a discharge from full charge to 2.5 V, a charge to 3.65 V)."""
    s = ts[ts["Step_Index"] == k]
    if len(s) <= 10:
        return s.iloc[0:0]
    start, end = s["Voltage"].iloc[0], s["Voltage"].iloc[-1]
    ok = (end <= DISCHARGE_CUTOFF_V + CUTOFF_TOL_V and start >= FULL_START_V) if discharge \
        else end >= CHARGE_CUTOFF_V - CUTOFF_TOL_V
    return s if ok else s.iloc[0:0]


def _q_at(v: np.ndarray, q: np.ndarray, level: float) -> float:
    """Discharged charge when the (monotonised) voltage first reaches ``level``."""
    vm = np.minimum.accumulate(v)
    if level >= vm[0]:
        return 0.0
    if level <= vm[-1]:
        return float(q[-1])
    return float(np.interp(level, vm[::-1], q[::-1]))


def ic_curve(dis: pd.DataFrame, grid: np.ndarray = IC_GRID, smooth_mv: float = IC_SMOOTH_MV) -> np.ndarray:
    """-dQ/dV (Ah/V) of a C/25 discharge on ``grid``."""
    v = np.minimum.accumulate(dis["Voltage"].to_numpy(float))
    q = dis["Discharge_Capacity"].to_numpy(float)
    qg = np.interp(grid, v[::-1], q[::-1], left=np.nan, right=np.nan)
    out = np.full(len(grid), np.nan)
    ok = np.isfinite(qg)
    if ok.sum() > 10:
        out[ok] = -np.gradient(gaussian_filter1d(qg[ok], smooth_mv / 1000 / np.median(np.diff(grid))), grid[ok])
    return out


def check_features(ts: pd.DataFrame) -> dict:
    """Capacities and diagnostic features of one characterisation (NaN where a step is missing).

    Capacity counters reset at every charge and discharge step (article, Table 4).
    """
    d3, c3, d25, c25 = (_step(ts, k, dis) for k, dis in ((11, True), (13, False), (16, True), (18, False)))
    r: dict = {"check_time": pd.Timestamp("1899-12-30") + pd.to_timedelta(ts["DateTime"].replace(0, np.nan).min(), unit="D")}
    r["q_c25_dis"] = float(d25["Discharge_Capacity"].max()) if len(d25) else np.nan
    r["q_c25_ch"] = float(c25["Charge_Capacity"].max()) if len(c25) else np.nan
    r["q_c3_dis"] = float(d3["Discharge_Capacity"].max()) if len(d3) else np.nan
    # Authors' capacity definition (max charge counter of step 13, no validity rule), kept only
    # to check our parsing against their extractedData.mat; not a feature.
    s13 = ts[ts["Step_Index"] == 13]
    r["authors_c3_charge_ah"] = float(s13["Charge_Capacity"].max()) if len(s13) else np.nan
    r["complete"] = bool(len(d3) and len(c3) and len(d25) and len(c25))
    if len(d25):
        v, q = d25["Voltage"].to_numpy(float), d25["Discharge_Capacity"].to_numpy(float)
        for hi, lo in Q_WINDOWS:
            r[f"q_{hi:.3f}_{lo:.3f}"] = _q_at(v, q, lo) - _q_at(v, q, hi)
        ic = ic_curve(d25)
        for name, (lo, hi) in IC_WINDOWS.items():
            m = (IC_GRID >= lo) & (IC_GRID <= hi) & np.isfinite(ic)
            j = np.flatnonzero(m)[np.argmax(ic[m])] if m.any() else None
            edge = j is None or IC_GRID[j] <= lo + 0.002 or IC_GRID[j] >= hi - 0.002
            r[f"ic_pos_{name}"] = np.nan if edge else float(IC_GRID[j])
            r[f"ic_height_{name}"] = np.nan if edge else float(ic[j])
        r["v_c25_dis_mid"] = float(np.interp(0.5 * q[-1], q, v))
    if len(d25) and len(c25):
        vc, qc = c25["Voltage"].to_numpy(float), c25["Charge_Capacity"].to_numpy(float)
        # Charge-discharge hysteresis at half of the discharged capacity (polarisation proxy).
        r["hysteresis_c25_mid"] = float(np.interp(0.5 * r["q_c25_dis"], qc, vc)) - r["v_c25_dis_mid"]
    if len(d3) and len(d25):
        # Overpotential between C/3 and C/25 discharge at the same discharged charge (resistance proxy, ohm).
        q_mid = 0.5 * min(r["q_c3_dis"], r["q_c25_dis"])
        v3 = float(np.interp(q_mid, d3["Discharge_Capacity"], d3["Voltage"]))
        v25 = float(np.interp(q_mid, d25["Discharge_Capacity"], d25["Voltage"]))
        i3, i25 = abs(d3["Current"].median()), abs(d25["Current"].median())
        r["r_c3_c25"] = (v25 - v3) / (i3 - i25)
        r["q_ratio_c3_c25"] = r["q_c3_dis"] / r["q_c25_dis"]
    return r


def list_checks(raw_root: Path) -> pd.DataFrame:
    """Every 50 degC characterisation file with its cell, test case and cycle number."""
    rows = []
    for p in sorted(Path(raw_root).glob("Cell*/agingdata_50degC/*charac*.csv*")):
        m = re.search(r"cell(\d[a-c])_charac(\d+)", p.name, re.I)
        rows.append({"cell": f"cell{m.group(1).lower()}", dp.GROUP_COL: cell_case(m.group(1)), "cycle": int(m.group(2)),
                     "path": p})
    return pd.DataFrame(rows).sort_values(["cell", "cycle"]).reset_index(drop=True)


def _one(path: str) -> dict:
    return check_features(read_check(path))


def build_table(raw_root: Path, workers: int = 4) -> pd.DataFrame:
    """One row per (cell, 50 degC check) with capacities, features and C/25 SOH (D17 analogue).

    ``SOH_Percent = q_c25_dis / q_c25_dis(first check) * 100``; checks with a
    missing C/25 discharge (a documented acquisition fault at cycle 100 in
    cells 1a, 2a, 2c and 3a) have no label and are flagged ``valid = False``.
    """
    from concurrent.futures import ProcessPoolExecutor

    checks = list_checks(raw_root)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        feats = list(ex.map(_one, checks["path"].astype(str), chunksize=4))
    t = pd.concat([checks.drop(columns="path"), pd.DataFrame(feats)], axis=1)
    t["valid"] = t["q_c25_dis"].notna() & (t["q_c25_dis"] > 0.2 * NOMINAL_AH)
    ref = t[t["valid"]].sort_values("cycle").groupby("cell")["q_c25_dis"].first()
    t["Capacity_Ah"] = t["q_c25_dis"].where(t["valid"])
    t["Reference_Capacity_Ah"] = t["cell"].map(ref)
    t["SOH_Percent"] = t["Capacity_Ah"] / t["Reference_Capacity_Ah"] * 100
    return t


def feature_columns(table: pd.DataFrame) -> list[str]:
    """Diagnostic features at a check (no identifiers, labels or the label's own capacity)."""
    prefixes = ("q_3", "q_2", "ic_pos_", "ic_height_", "v_c25_", "hysteresis_", "r_c3_", "q_ratio_", "q_c3_dis", "q_c25_ch")
    return [c for c in table.columns if c.startswith(prefixes)]


def build_samples(table: pd.DataFrame, horizons: Sequence[int] = FC.HORIZONS) -> pd.DataFrame:
    """Forecasting samples (D20): origin check k, target check k+h among the cell's valid checks."""
    diag = feature_columns(table)
    t = table[table["valid"]].sort_values(["cell", "cycle"]).reset_index(drop=True)
    rows = []
    for cell, g in t.groupby("cell", sort=True):
        g = g.reset_index(drop=True)
        cyc, soh = g["cycle"].to_numpy(float), g["SOH_Percent"].to_numpy(float)
        for i in range(len(g)):
            o = g.iloc[i]
            rate = np.nan
            if i >= 2:
                rate = float(np.polyfit(cyc[i - 2:i + 1], soh[i - 2:i + 1], 1)[0])
            base = {dp.GROUP_COL: o[dp.GROUP_COL], "cell": cell, "Cycle_Index": int(o["cycle"]), "Cycle_Date": o["check_time"],
                    "origin_cycle": int(o["cycle"]), "origin_pos": i, "n_valid_rpts": len(g), "soh_origin": soh[i],
                    "cycles_since_first": cyc[i] - cyc[0], "fade_rate_pp_per_cycle": rate, **{c: o[c] for c in diag}}
            for h in horizons:
                j = i + h
                if j >= len(g):
                    continue
                rows.append({**base, "Cycling_Test": f"{cell}_h{h}", "horizon": h, "target_cycle": int(g["cycle"].iloc[j]),
                             "target_pos": j, "cycles_to_target": cyc[j] - cyc[i], "soh_target": soh[j],
                             FC.TARGET: soh[j] - soh[i]})
    out = pd.DataFrame(rows).merge(TEST_CASES, on=dp.GROUP_COL, how="left", validate="many_to_one")
    return out.sort_values([dp.GROUP_COL, "cell", "Cycle_Date", "horizon"], kind="mergesort").reset_index(drop=True)


SCHEDULE_FEATURES: tuple[str, ...] = (*STATIC_FEATURES, "cycles_to_target")
BASELINE_COLUMNS: dict = {"rate": "fade_rate_pp_per_cycle", "span": "cycles_to_target", "schedule": SCHEDULE_FEATURES}


def feature_sets(samples: pd.DataFrame) -> dict[str, list[str]]:
    """F0 (history, static test conditions, cycles to target) and F1 (F0 + diagnostics), as in D18."""
    f0 = ["soh_origin", "cycles_since_first", "fade_rate_pp_per_cycle", "horizon", *SCHEDULE_FEATURES]
    return {"F0": f0, "F1": f0 + [c for c in feature_columns(samples) if c in samples.columns]}
