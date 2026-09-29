"""Health indicators from the monthly standardised segment of the point-level data (decision D14).

Every monthly cycling block in ``cleaned_battery_data.csv`` begins with the same
sequence in its first cycle: rest (step 1), constant-current charge (2),
constant-voltage charge (3), rest (4), 1C discharge (5), rest (6), then the
grid-storage duty cycle (7). Repeated under the same conditions each month,
this segment yields candidate health indicators (HIs), one row per block,
aligned to the RPTs that bracket the block.

Protocol leakage: the step-5 discharge current is set to 1C of the capacity
measured at the previous RPT (checked by ``protocol_scaling``), so it and the
quantities it determines encode the label. They are recorded for the check but
never used as indicators (``PROTOCOL_SCALED``).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src import data_pipeline as dp

#: Voltage windows (V) for the partial-charge indicator: charge accumulated
#: between two voltages during the constant-current charge (robust to where
#: the charge starts).
DEFAULT_WINDOWS: tuple[tuple[float, float], ...] = ((3.7, 3.9), (3.8, 4.0), (3.9, 4.1), (3.8, 4.1), (4.0, 4.15))
STANDARD_CC_CURRENT = 1.46  # A, the protocol's constant-current charge
CC_TOLERANCE = 0.05

#: Recorded but never used as indicators: set from the previous RPT capacity
#: (discharge current and what it fixes) or constant by protocol design.
PROTOCOL_SCALED: tuple[str, ...] = ("dis_current", "dis_charge_ah", "dis_energy_wh", "cc_current", "dis_duration_s")

INDICATORS: tuple[str, ...] = (
    "cc_start_v", "cc_charge_ah", "cc_duration_s", "cv_charge_ah", "cv_duration_s",
    "rest_charge_dv", "dis_start_v", "dis_end_v", "dis_mean_v", "ir_proxy_ohm",
    "rest_discharge_dv", "dis_temp_rise", "duty_v_p05", "duty_v_p95", "duty_temp_mean",
)


def _temperature_column(points: pd.DataFrame) -> str | None:
    cols = [c for c in points.columns if c.startswith("Aux_Temperature")]
    return cols[0] if cols else None


def window_charge(voltage: np.ndarray, charge: np.ndarray, low: float, high: float) -> float:
    """Charge (Ah) accumulated between ``low`` and ``high`` volts during a CC charge.

    Voltage is made non-decreasing before interpolation (it rises monotonically
    under CC charge apart from measurement noise). NaN if the charge does not
    span the whole window.
    """
    v = np.maximum.accumulate(np.asarray(voltage, float))
    q = np.asarray(charge, float)
    if len(v) < 2 or v[0] > low or v[-1] < high:
        return np.nan
    return float(np.interp(high, v, q) - np.interp(low, v, q))


def extract_block_indicators(points: pd.DataFrame, windows: Sequence[tuple[float, float]] = DEFAULT_WINDOWS) -> pd.DataFrame:
    """One row of indicators per (battery, cycling test) block of the point-level data.

    Steps are identified by ``Step_Index`` within the block's first cycle. Missing
    steps give NaN indicators and are listed in ``missing_steps``; blocks whose
    CC charge does not use the standard current are flagged
    (``nonstandard_charge``), not dropped.
    """
    temp = _temperature_column(points)
    pts = points.assign(**{dp.GROUP_COL: points["Source_Folder"].astype(str).str.lower()})
    pts = pts.sort_values([dp.GROUP_COL, "Date_Time", "Test_Time(s)"], kind="mergesort")
    rows = []
    for (battery, test), g in pts.groupby([dp.GROUP_COL, "Cycling_Test"], sort=True):
        first = g[g["Cycle_Index"] == g["Cycle_Index"].min()]
        steps = {k: s for k, s in first.groupby("Step_Index")}
        r: dict = {dp.GROUP_COL: battery, "Cycling_Test": test, "block_start": g["Date_Time"].min(),
                   "block_end": g["Date_Time"].max(), "missing_steps": [k for k in range(1, 8) if k not in steps and not (k == 7 and (g["Step_Index"] == 7).any())]}
        if 2 in steps:
            s = steps[2]
            r.update(cc_current=s["Current(A)"].median(), cc_start_v=s["Voltage(V)"].iloc[0],
                     cc_charge_ah=s["Charge_Capacity(Ah)"].max() - s["Charge_Capacity(Ah)"].min(),
                     cc_duration_s=s["Step_Time(s)"].max())
            standard = abs(r["cc_current"] - STANDARD_CC_CURRENT) <= CC_TOLERANCE
            for low, high in windows:
                r[f"dq_{low}_{high}"] = window_charge(s["Voltage(V)"], s["Charge_Capacity(Ah)"], low, high) if standard else np.nan
        if 3 in steps:
            s = steps[3]
            r.update(cv_charge_ah=s["Charge_Capacity(Ah)"].max() - s["Charge_Capacity(Ah)"].min(), cv_duration_s=s["Step_Time(s)"].max())
        if 4 in steps:
            s = steps[4]
            r.update(rest_charge_dv=s["Voltage(V)"].iloc[0] - s["Voltage(V)"].iloc[-1])
        if 5 in steps:
            s = steps[5]
            r.update(dis_current=s["Current(A)"].median(),
                     dis_charge_ah=s["Discharge_Capacity(Ah)"].max() - s["Discharge_Capacity(Ah)"].min(),
                     dis_energy_wh=s["Discharge_Energy(Wh)"].max() - s["Discharge_Energy(Wh)"].min(),
                     dis_duration_s=s["Step_Time(s)"].max(), dis_start_v=s["Voltage(V)"].iloc[0],
                     dis_end_v=s["Voltage(V)"].iloc[-1], dis_mean_v=s["Voltage(V)"].mean())
            if temp and s[temp].notna().any():
                r["dis_temp_rise"] = s[temp].max() - s[temp].dropna().iloc[0]
            if 4 in steps and abs(s["Current(A)"].iloc[0]) > 0.5:
                # Resistance proxy from the first logged discharge sample (60 s after the step starts),
                # so it mixes ohmic and early polarisation contributions.
                r["ir_proxy_ohm"] = (steps[4]["Voltage(V)"].iloc[-1] - s["Voltage(V)"].iloc[0]) / abs(s["Current(A)"].iloc[0])
        if 6 in steps:
            s = steps[6]
            r.update(rest_discharge_dv=s["Voltage(V)"].iloc[-1] - s["Voltage(V)"].iloc[0])
        duty = g[g["Step_Index"] == 7]
        if len(duty):
            r.update(duty_v_p05=duty["Voltage(V)"].quantile(0.05), duty_v_p95=duty["Voltage(V)"].quantile(0.95),
                     duty_temp_mean=duty[temp].mean() if temp else np.nan)
        r["nonstandard_charge"] = bool(abs(r.get("cc_current", np.nan) - STANDARD_CC_CURRENT) > CC_TOLERANCE) if "cc_current" in r else True
        rows.append(r)
    return pd.DataFrame(rows)


def attach_rpt_labels(blocks: pd.DataFrame, health: pd.DataFrame) -> pd.DataFrame:
    """Add the RPTs bracketing each block: the last RPT before it and the first RPT on or after its end.

    ``soh_next`` (the RPT closing the block) is the estimation target; ``soh_prev``
    is what is already known when the block starts. Dates are compared at day
    resolution, matching the date-only RPT timestamps.
    """
    h = health.assign(**{dp.GROUP_COL: health[dp.GROUP_COL].astype(str)}).sort_values([dp.GROUP_COL, "Test_Date"])
    out = blocks.copy()
    for col in ("soh_prev", "soh_next", "capacity_prev", "rpt_prev", "rpt_next"):
        out[col] = np.nan if not col.startswith("rpt") else None
    for i, r in out.iterrows():
        hb = h[h[dp.GROUP_COL] == r[dp.GROUP_COL]]
        prev = hb[hb["Test_Date"] < r["block_start"].normalize()]
        nxt = hb[hb["Test_Date"] >= r["block_end"].normalize()]
        if len(prev):
            out.at[i, "soh_prev"], out.at[i, "capacity_prev"], out.at[i, "rpt_prev"] = prev["SOH_Percent"].iloc[-1], prev["Capacity_Ah"].iloc[-1], prev["RPT"].iloc[-1]
        if len(nxt):
            out.at[i, "soh_next"], out.at[i, "rpt_next"] = nxt["SOH_Percent"].iloc[0], nxt["RPT"].iloc[0]
    first = h.groupby(dp.GROUP_COL)["Test_Date"].min()
    out["days_since_first_rpt"] = (out["block_start"].dt.normalize() - out[dp.GROUP_COL].map(first)).dt.days
    return out


def protocol_scaling(blocks: pd.DataFrame) -> pd.DataFrame:
    """Ratio of the step-5 discharge current to the previous RPT capacity, per battery.

    A ratio near 1 with a tiny spread means the discharge current was set to 1C
    of the last measured capacity, i.e. it encodes the label.
    """
    b = blocks[~blocks["nonstandard_charge"] & blocks["capacity_prev"].notna()].copy()
    b["current_to_prev_capacity"] = -b["dis_current"] / b["capacity_prev"]
    stats = b.groupby(dp.GROUP_COL)["current_to_prev_capacity"].agg(["mean", "std", "count"])
    stats.loc["ALL"] = [b["current_to_prev_capacity"].mean(), b["current_to_prev_capacity"].std(), len(b)]
    stats["cv_percent"] = stats["std"] / stats["mean"] * 100
    return stats


def leave_one_battery_out(data: pd.DataFrame, feature_sets: Mapping[str, Sequence[str]], target: str = "soh_next") -> pd.DataFrame:
    """RMSE on each held-out battery for ridge models on several feature sets.

    Standardisation and median imputation are fitted on the training batteries
    only; the ridge penalty is chosen by internal cross-validation on them.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = {}
    batteries = sorted(data[dp.GROUP_COL].unique())
    for name, cols in feature_sets.items():
        errs = {}
        for b in batteries:
            tr, te = data[data[dp.GROUP_COL] != b], data[data[dp.GROUP_COL] == b]
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
            model.fit(tr[list(cols)], tr[target])
            errs[b] = float(np.sqrt(np.mean((model.predict(te[list(cols)]) - te[target]) ** 2)))
        out[name] = errs
    res = pd.DataFrame(out)
    res.loc["mean"] = res.mean()
    return res
