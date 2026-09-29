"""Health features from the raw RPT capacity tests and EIS spectra (decision D16).

One row of features per (battery, RPT), all measured *at* that RPT, for use as
inputs to SOH forecasting (predicting the SOH at later RPTs).

Capacity test with pulses (Moy et al., 2024, Table 5): C/20 discharge from
4.2 V to 2.5 V, interrupted at 4.00, 3.63 and 3.26 V by an 8 s C/2 discharge
pulse, a 10 s rest and an 8 s C/2 charge pulse.

* ``q_c20_ah``: charge discharged in the C/20 segments only (the article's
  capacity definition; the handover's ``Capacity_Ah`` also counts the pulses).
* ``q_<hi>_<lo>``: charge in each C/20 segment between the pulse voltages.
* ``r_pulse_<v>_<t>``: discharge-pulse resistance at 4.00/3.63/3.26 V after
  0.1, 1 and 8 s, ``(V_ref - V(t)) / (I_ref - I_pulse)`` with the C/20 current
  and voltage just before the pulse as reference.
* ``ic_pos_<k>``, ``ic_height_<k>``: position (V) and height (Ah/V) of the
  largest incremental-capacity peak in fixed voltage windows of the C/20
  discharge (quasi-open-circuit).

EIS (from RPT 5; 10 kHz to 0.01 Hz, at 3.26/3.63/4.00 V):

* ``eis_r0_<v>``: Z' where Z'' crosses zero on the high-frequency side
  (ohmic resistance);
* ``eis_rarc_<v>``: Z' at the arc/diffusion valley minus R0 (SEI and
  charge-transfer arc);
* ``eis_arc_peak_<v>``: maximum -Z'' of the arc; ``eis_zre_<f>_<v>``: Z' at
  fixed frequencies.

Leakage guard: protocol currents are set from the previous capacity, so no
current value is returned as a feature (``PROTOCOL_CURRENTS`` lists what is
recorded for checks only).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema

PULSE_VOLTAGES: tuple[float, ...] = (4.00, 3.63, 3.26)
PULSE_TIMES_S: tuple[float, ...] = (0.1, 1.0, 8.0)
C20_MIN_DURATION_S = 300.0
PULSE_MAX_DURATION_S = 30.0
STANDARD_PULSE_C_RATE = 0.5          # C/2
PULSE_RATE_TOLERANCE = 0.25          # pulses below 75 % of C/2 are flagged
IC_GRID = np.round(np.arange(2.70, 4.152, 0.002), 3)
IC_SMOOTH_MV = 10.0
#: Voltage windows (V) for incremental-capacity peaks, chosen from the peak
#: pattern of the fresh-cell C/20 discharge (docs/DECISIONS.md D16).
IC_WINDOWS: dict[str, tuple[float, float]] = {
    "p1": (3.40, 3.53), "p2": (3.55, 3.72), "p3": (3.74, 3.86), "p4": (3.86, 3.98), "p5": (4.00, 4.12),
}
EIS_FREQS_HZ: tuple[float, ...] = (1000.0, 10.0, 1.0, 0.1, 0.01)
PROTOCOL_CURRENTS: tuple[str, ...] = ("c20_current_a", "pulse_current_4.00", "pulse_current_3.63", "pulse_current_3.26")


def _segments(ts: pd.DataFrame) -> list[pd.DataFrame]:
    """Contiguous runs of the same (Cycle_Index, Step_Index)."""
    key = ts["Step_Index"].astype(str) + "_" + ts["Cycle_Index"].astype(str)
    run = (key != key.shift()).cumsum()
    return [g for _, g in ts.groupby(run, sort=True)]


def _kind(seg: pd.DataFrame) -> str:
    i = seg["Current(A)"].median()
    dur = seg["Step_Time(s)"].max()
    if abs(i) < 1e-3:
        return "rest"
    if i < 0 and dur >= C20_MIN_DURATION_S and abs(i) < 1.0:
        return "c20"
    if i < 0 and dur <= PULSE_MAX_DURATION_S:
        return "pulse_dis"
    if i > 0 and dur <= PULSE_MAX_DURATION_S:
        return "pulse_chg"
    return "charge" if i > 0 else "other_dis"


def _at(seg: pd.DataFrame, t: float) -> float:
    """Voltage at the sample closest to step time ``t`` (s)."""
    st = seg["Step_Time(s)"].to_numpy()
    return float(seg["Voltage(V)"].to_numpy()[np.argmin(np.abs(st - t))])


def ic_curve(c20: Sequence[pd.DataFrame], grid: np.ndarray = IC_GRID, smooth_mv: float = IC_SMOOTH_MV) -> np.ndarray:
    """-dQ/dV (Ah/V) of the C/20 discharge on ``grid``.

    Segments are joined into one discharge. After each pulse the voltage first
    relaxes back to the level at which the previous segment stopped; those
    samples, and the few mAh passed during them, are dropped so the pulses do
    not create artificial peaks at the pulse voltages.
    """
    vs, qs, offset, v_stop = [], [], 0.0, np.inf
    for seg in c20:
        v = seg["Voltage(V)"].to_numpy(float)
        q = seg["Discharge_Capacity(Ah)"].to_numpy(float)
        keep = v < v_stop
        if not keep.any():
            continue
        first = int(np.argmax(keep))
        v, q = v[first:], q[first:] - q[first] + offset
        vs.append(v); qs.append(q)
        offset, v_stop = q[-1], v[-1]
    if not vs:
        return np.full(len(grid), np.nan)
    v = np.minimum.accumulate(np.concatenate(vs))
    q = np.concatenate(qs)
    q_grid = np.interp(grid, v[::-1], q[::-1], left=np.nan, right=np.nan)
    step = float(np.median(np.diff(grid)))
    valid = np.isfinite(q_grid)
    out = np.full(len(grid), np.nan)
    if valid.sum() > 10:
        sm = gaussian_filter1d(q_grid[valid], smooth_mv / 1000.0 / step)
        out[valid] = -np.gradient(sm, grid[valid])
    return out


def ic_peaks(ic: np.ndarray, grid: np.ndarray = IC_GRID, windows: dict[str, tuple[float, float]] = IC_WINDOWS) -> dict:
    """Position and height of the maximum of the IC curve in each window."""
    out = {}
    for name, (lo, hi) in windows.items():
        m = (grid >= lo) & (grid <= hi) & np.isfinite(ic)
        if m.sum() == 0:
            out[f"ic_pos_{name}"] = out[f"ic_height_{name}"] = np.nan
            continue
        j = np.flatnonzero(m)[np.argmax(ic[m])]
        # A maximum on the window edge is not a peak inside the window.
        edge = grid[j] <= lo + 0.004 or grid[j] >= hi - 0.004
        out[f"ic_pos_{name}"] = np.nan if edge else float(grid[j])
        out[f"ic_height_{name}"] = np.nan if edge else float(ic[j])
    return out


def rpt_features(ts: pd.DataFrame, nominal_ah: float = 4.85) -> dict:
    """Features of one capacity test with pulses (Arbin time series)."""
    segs = _segments(ts)
    kinds = [_kind(s) for s in segs]
    c20 = [s for s, k in zip(segs, kinds) if k == "c20"]
    r: dict = {"n_c20_segments": len(c20), "nonstandard": len(c20) != 4, "nonstandard_pulses": []}
    if not c20:
        return r
    dq = [float(s["Discharge_Capacity(Ah)"].max() - s["Discharge_Capacity(Ah)"].min()) for s in c20]
    r["q_c20_ah"] = float(np.sum(dq))
    r["c20_current_a"] = float(np.median(pd.concat(c20)["Current(A)"]))
    if len(c20) == 4:
        for (hi, lo), q in zip(zip(("4.20", "4.00", "3.63", "3.26"), ("4.00", "3.63", "3.26", "2.50")), dq):
            r[f"q_{hi}_{lo}"] = q
    # Pulse resistances: a discharge pulse directly after a C/20 segment.
    for i in range(1, len(segs)):
        if kinds[i] != "pulse_dis" or kinds[i - 1] != "c20":
            continue
        ref, pulse = segs[i - 1], segs[i]
        v_ref, i_ref = float(ref["Voltage(V)"].iloc[-1]), float(ref["Current(A)"].iloc[-1])
        level = min(PULSE_VOLTAGES, key=lambda x: abs(x - v_ref))
        if abs(level - v_ref) > 0.05:
            continue
        i_p = float(pulse["Current(A)"].median())
        r[f"pulse_current_{level:.2f}"] = i_p
        # C/2 relative to the C/20 current of the same test (both set from the same capacity).
        standard = abs(i_p) >= (1 - PULSE_RATE_TOLERANCE) * STANDARD_PULSE_C_RATE * abs(r["c20_current_a"]) * 20
        if not standard:
            r["nonstandard_pulses"].append(level)
        for t in PULSE_TIMES_S:
            # Resistance depends on current: a pulse off the C/2 protocol is not comparable and is not used.
            r[f"r_pulse_{level:.2f}_{t:g}s"] = (v_ref - _at(pulse, t)) / (i_ref - i_p) if standard else np.nan
    r.update(ic_peaks(ic_curve(c20)))
    temp = next((c for c in ts.columns if str(c).startswith("Aux_Temperature")), None)
    r["rpt_temp_mean"] = float(pd.concat(c20)[temp].mean()) if temp else np.nan
    return r


def eis_features(z: pd.DataFrame, voltage: float, freqs: Sequence[float] = EIS_FREQS_HZ) -> dict:
    """Features of one EIS spectrum (columns ``freq_hz``, ``z_real``, ``z_imag``; Z'' < 0 capacitive)."""
    v = f"{voltage:.2f}"
    out = {f"eis_r0_{v}": np.nan, f"eis_rarc_{v}": np.nan, f"eis_arc_peak_{v}": np.nan}
    out.update({f"eis_zre_{f:g}hz_{v}": np.nan for f in freqs})
    if z is None or len(z) < 5:
        return out
    z = z.sort_values("freq_hz", ascending=False).reset_index(drop=True)
    zr, zi, f = z["z_real"].to_numpy(float), z["z_imag"].to_numpy(float), z["freq_hz"].to_numpy(float)
    cross = np.flatnonzero((zi[:-1] > 0) & (zi[1:] <= 0))
    if len(cross):
        k = cross[0]
        out[f"eis_r0_{v}"] = float(zr[k] + (zr[k + 1] - zr[k]) * zi[k] / (zi[k] - zi[k + 1]))
    # Arc: most negative Z'' in 1-1000 Hz; valley: least negative Z'' between the arc and 0.1 Hz.
    arc = np.flatnonzero((f <= 1000) & (f >= 1))
    if len(arc) and np.isfinite(out[f"eis_r0_{v}"]):
        a = arc[np.argmin(zi[arc])]
        out[f"eis_arc_peak_{v}"] = float(-zi[a])
        low = np.flatnonzero((f < f[a]) & (f >= 0.1))
        if len(low):
            mx = argrelextrema(zi[low], np.greater_equal, order=2)[0]
            if len(mx):
                vly = low[mx[0]]
                out[f"eis_rarc_{v}"] = float(zr[vly] - out[f"eis_r0_{v}"])
    lf = np.log10(f)
    for fq in freqs:
        if f.min() <= fq * 1.01 and f.max() >= fq * 0.99:
            out[f"eis_zre_{fq:g}hz_{v}"] = float(np.interp(np.log10(fq), lf[::-1], zr[::-1]))
    return out


# ---------------------------------------------------------------------------
# Building the per-(battery, RPT) table from the archive
# ---------------------------------------------------------------------------

def _channel(name: str) -> int | None:
    import re

    m = re.search(r"_Channel_(\d+)", name)
    return int(m.group(1)) if m else None


def read_aggregate_eis(path, freq_hz: np.ndarray) -> dict[str, pd.DataFrame]:
    """Per-cell spectra from an ``*_EIS_#_<v>V_ALL.xlsx`` file (used where per-cell files are missing).

    Format checked against RPT 5, 6 and 8, where both forms exist: column pairs
    (Z', -Z'') per cell under a ``ZCURVE (<date>_<cell>_EIS...)`` header, in the
    same point order as the per-cell files and rounded to 1e-5 ohm (maximum
    difference 5e-6 ohm). Frequencies are not stored; every per-cell sweep
    uses the same 43-point grid, which is passed in as ``freq_hz``.
    """
    import re

    a = pd.read_excel(path, header=None)
    h = next(i for i in range(len(a)) if a.iloc[i].astype(str).str.contains("ZCURVE").any())
    out = {}
    for j in range(1, a.shape[1]):
        m = re.search(r"_([A-Za-z]+\d+)_EIS", str(a.iloc[h, j]))
        if not m:
            continue
        zr = a.iloc[h + 1:, j - 1].astype(float).to_numpy()
        zi = -a.iloc[h + 1:, j].astype(float).to_numpy()
        if len(zr) != len(freq_hz):
            continue
        out[m.group(1).lower()] = pd.DataFrame({"freq_hz": freq_hz, "z_real": zr, "z_imag": zi})
    return out


def build_feature_table(raw_root, cache_root, health: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features for every (battery, RPT) in the archive, plus a log of file choices.

    Selection rules (applied mechanically, logged in the second return value):

    * capacity test: the file with four C/20 segments (a complete test); if
      several, the longest. (g1 RPT 1 has a second, partial export.)
    * EIS per (battery, RPT, voltage): non-empty spectra only; prefer the file
      recorded on the same channel as the cell's capacity test in that RPT,
      then the one closest in time to the capacity test. (RPT 9 has spectra
      filed under a cell's name on another cell's channel; RPT 17 holds two
      complete EIS sessions, Oct and Nov 2024.)
    * galvanostatic spectra (``_gal`` files, w8 RPT 17 and 19) are excluded:
      every other spectrum is potentiostatic.
    * pulse resistances from pulses off the C/2 protocol (RPT 1 at 4.00 V,
      0.525 A; v4 and w10 in RPT 17-19, 1.05 A) are set to NaN and flagged.
    * RPT 7 has no per-cell EIS files; its spectra come from the aggregate
      ``_ALL`` files (``read_aggregate_eis``).
    """
    from pathlib import Path

    from src import data_pipeline as dp
    from src import raw_diagnostics as rd

    raw_root, cache_root = Path(raw_root), Path(cache_root)
    inv = rd.list_raw_files(raw_root)
    inv = inv[inv["kind"].notna()].copy()
    inv["channel"] = inv["file"].map(_channel)
    log, rows = [], {}
    freq_grid = None

    for (rpt, bat), g in inv[inv["kind"] == "RPT"].groupby(["rpt_number", dp.GROUP_COL]):
        best, best_key = None, None
        for _, f in g.iterrows():
            ts = rd.timeseries(rd.cached_sheets(f["path"], cache_root))
            feat = rpt_features(ts)
            key = (feat["n_c20_segments"] == 4, len(ts))
            if best is None or key > best_key:
                best, best_key = (f, ts, feat), key
        f, ts, feat = best
        if len(g) > 1:
            log.append({"rpt_number": rpt, dp.GROUP_COL: bat, "kind": "RPT", "chosen": f["file"],
                        "rejected": [x for x in g["file"] if x != f["file"]], "rule": "complete test, then longest"})
        rows[(int(rpt), bat)] = {dp.GROUP_COL: bat, "rpt_number": int(rpt), "rpt_file": f["file"],
                                 "rpt_start": ts["Date_Time"].iloc[0], "rpt_channel": f["channel"],
                                 **{k: v for k, v in feat.items() if k not in PROTOCOL_CURRENTS},
                                 "_currents": {k: feat.get(k) for k in PROTOCOL_CURRENTS}}

    eis = inv[inv["kind"] == "EIS"]
    for (rpt, bat, volt), g in eis.groupby(["rpt_number", dp.GROUP_COL, "eis_voltage"]):
        base = rows.get((int(rpt), bat))
        cands = []
        for _, f in g.iterrows():
            if "gal" in str(f["suffix"]).lower():
                # Galvanostatic EIS: a different measurement mode from every other spectrum.
                log.append({"rpt_number": rpt, dp.GROUP_COL: bat, "kind": f"EIS {volt:.2f} V", "chosen": None,
                            "rejected": [f["file"]], "rule": "galvanostatic mode excluded"})
                continue
            sheets = rd.cached_sheets(f["path"], cache_root)
            z = rd.impedance(sheets)
            if len(z) == 0:
                continue
            if freq_grid is None:
                freq_grid = z["freq_hz"].to_numpy()
            ts = rd.timeseries(sheets)
            t = ts["Date_Time"].iloc[0] if len(ts) else f["date"]
            same_ch = base is not None and f["channel"] == base["rpt_channel"]
            gap = abs((t - base["rpt_start"]).total_seconds()) if base is not None else 0.0
            cands.append(((not same_ch, gap), f["file"], z))
        if not cands:
            log.append({"rpt_number": rpt, dp.GROUP_COL: bat, "kind": f"EIS {volt:.2f} V", "chosen": None,
                        "rejected": g["file"].tolist(), "rule": "no non-empty spectrum"})
            continue
        cands.sort(key=lambda c: c[0])
        if len(g) > 1:
            log.append({"rpt_number": rpt, dp.GROUP_COL: bat, "kind": f"EIS {volt:.2f} V", "chosen": cands[0][1],
                        "rejected": [x for x in g["file"] if x != cands[0][1]],
                        "rule": "non-empty, same channel as capacity test, closest in time"})
        if base is not None:
            base.update(eis_features(cands[0][2], volt))
            base[f"eis_file_{volt:.2f}"] = cands[0][1]

    # RPT 7: aggregate files only.
    for path in sorted(raw_root.glob("RPT_*/**/*_ALL.xlsx")):
        import re

        m = re.search(r"_EIS_(\d+)_(\d{3})V_ALL", path.name)
        rpt, volt = int(m.group(1)), rd.EIS_VOLTAGES[m.group(2)]
        for bat, z in read_aggregate_eis(path, freq_grid).items():
            base = rows.get((rpt, bat))
            if base is None or f"eis_file_{volt:.2f}" in base:
                continue
            base.update(eis_features(z, volt))
            base[f"eis_file_{volt:.2f}"] = path.name
            log.append({"rpt_number": rpt, dp.GROUP_COL: bat, "kind": f"EIS {volt:.2f} V", "chosen": path.name,
                        "rejected": [], "rule": "no per-cell file; aggregate file"})

    table = pd.DataFrame(rows.values()).sort_values([dp.GROUP_COL, "rpt_number"]).reset_index(drop=True)
    if health is not None:
        # Handover labels are kept for comparison only; SOH_Percent is recomputed on the C/20 basis (D17).
        h = health[[dp.GROUP_COL, "RPT_Number", "Test_Date", "Capacity_Ah", "SOH_Percent"]].rename(
            columns={"RPT_Number": "rpt_number", "Capacity_Ah": "Capacity_Ah_handover", "SOH_Percent": "SOH_handover"})
        table = table.merge(h, on=[dp.GROUP_COL, "rpt_number"], how="left", validate="one_to_one")
        table = add_c20_soh(table)
    return table, pd.DataFrame(log)


def add_c20_soh(table: pd.DataFrame) -> pd.DataFrame:
    """SOH on the C/20 capacity basis (decision D17).

    ``Capacity_Ah`` is the charge discharged in the C/20 steps of the capacity
    test (``q_c20_ah``; Moy et al., 2024, Eq. (1)), excluding the pulses, whose
    size changed with the protocol. ``Reference_Capacity_Ah`` is the battery's
    first RPT on that basis, and ``SOH_Percent = Capacity_Ah /
    Reference_Capacity_Ah * 100``.
    """
    from src import data_pipeline as dp

    out = table.sort_values([dp.GROUP_COL, "rpt_number"]).copy()
    out["Capacity_Ah"] = out["q_c20_ah"]
    out["Reference_Capacity_Ah"] = out.groupby(dp.GROUP_COL)["Capacity_Ah"].transform("first")
    out["SOH_Percent"] = out["Capacity_Ah"] / out["Reference_Capacity_Ah"] * 100
    return out.reset_index(drop=True)


def health_table(table: pd.DataFrame) -> pd.DataFrame:
    """RPT health table on the C/20 basis, in the layout of ``final_battery_health_dataset.csv``."""
    from src import data_pipeline as dp

    t = table.sort_values([dp.GROUP_COL, "rpt_number"])
    return pd.DataFrame({dp.GROUP_COL: t[dp.GROUP_COL], "RPT": "RPT_" + t["rpt_number"].astype(str),
                         "RPT_Number": t["rpt_number"], "Test_Date": t["Test_Date"],
                         "Capacity_Ah": t["Capacity_Ah"], "Reference_Capacity_Ah": t["Reference_Capacity_Ah"],
                         "SOH_Percent": t["SOH_Percent"], "SOH_handover": t["SOH_handover"],
                         "rpt_temp_mean": t["rpt_temp_mean"]}).reset_index(drop=True)


def feature_columns(table: pd.DataFrame) -> list[str]:
    """Numeric health-feature columns (no identifiers, labels, files or protocol currents).

    ``q_c20_ah`` is excluded: it is the capacity the SOH label is built from (D17).
    The window charges ``q_<hi>_<lo>`` sum to it, so at the same RPT they carry
    the label; they are legitimate inputs only for forecasting later RPTs.
    """
    prefixes = ("q_", "r_pulse_", "ic_pos_", "ic_height_", "eis_r0_", "eis_rarc_", "eis_arc_peak_", "eis_zre_")
    return [c for c in table.columns if c.startswith(prefixes) and c != "q_c20_ah"]
