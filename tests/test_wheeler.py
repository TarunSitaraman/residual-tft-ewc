"""Tests for the S6 (Wheeler et al. 2025) characterisation features and samples, on hand-made checks."""

import numpy as np
import pandas as pd
import pytest

from src import data_pipeline as dp
from src import forecasting as FC
from src import wheeler as W


def _check(q=1.0, start_v=3.50, c3_start_v=3.50, drop_c25=False):
    """One characterisation: C/3 dis (11), C/3 ch (13), C/25 dis (16), C/25 ch (18); flat LFP-like plateau."""
    rows, t = [], 0.0

    def ocv(frac):                          # voltage vs discharged fraction: plateau with two steps
        return np.interp(frac, [0, 0.02, 0.3, 0.32, 0.7, 0.72, 0.98, 1.0], [3.5, 3.34, 3.33, 3.30, 3.29, 3.22, 3.18, 2.5])

    def seg(step, n, current, v, qcol, qvals):
        nonlocal t
        for k in range(n):
            t += 1
            rows.append({"DateTime": 44000 + t / 86400, "Step_Index": step, "Step_Time": k + 1.0, "Current": current,
                         "Voltage": v[k], "Charge_Capacity": qvals[k] if qcol == "c" else 0.0,
                         "Discharge_Capacity": qvals[k] if qcol == "d" else 0.0})
    n = 400
    frac = np.linspace(0, 1, n)
    v3 = ocv(frac) - 0.02
    v3[0] = c3_start_v
    seg(11, n, -q / 3, v3, "d", frac * q * 0.98)
    seg(13, n, q / 3, np.linspace(3.0, 3.65, n), "c", frac * q)
    if not drop_c25:
        v25 = ocv(frac)
        v25[0] = start_v
        seg(16, n, -q / 25, v25, "d", frac * q)
    v_ch = ocv(1 - frac) + 0.02            # charge curve above the discharge curve (hysteresis)
    v_ch[-1] = 3.65                        # reaches the charge cut-off
    seg(18, n, q / 25, v_ch, "c", frac * q * 1.005)
    return pd.DataFrame(rows)


def test_check_features_capacity_and_windows():
    r = W.check_features(_check(q=1.0))
    assert r["complete"] and r["q_c25_dis"] == pytest.approx(1.0)
    windows = [c for c in r if c.startswith("q_3.") or c.startswith("q_2.")]
    assert sum(r[c] for c in windows) == pytest.approx(r["q_c25_dis"] - W._q_at(np.array([3.5, 3.55]), np.array([0, 0]), 3.55), abs=0.02)
    assert all(r[c] >= 0 for c in windows)
    assert r["r_c3_c25"] > 0 and r["hysteresis_c25_mid"] > 0


def test_incomplete_steps_are_not_used():
    r = W.check_features(_check(c3_start_v=3.24))          # C/3 discharge not from full charge
    assert np.isnan(r["q_c3_dis"]) and np.isnan(r.get("r_c3_c25", np.nan)) and r["q_c25_dis"] == pytest.approx(1.0)
    r = W.check_features(_check(drop_c25=True))            # missing C/25 discharge: no label
    assert np.isnan(r["q_c25_dis"]) and not r["complete"]


def _table():
    rows = []
    for cell, case, fade in (("cell1a", "case1", 0.0002), ("cell1b", "case1", 0.00025), ("cell7a", "case7", 0.0004)):
        for k in range(12):
            q = 1.0 - fade * 100 * k
            rows.append({"cell": cell, dp.GROUP_COL: case, "cycle": 100 * k, "check_time": pd.Timestamp("2021-01-01") + pd.Timedelta(days=20 * k),
                         "q_c25_dis": q if (cell, k) != ("cell1a", 1) else np.nan, "q_c25_ch": q * 1.005, "q_c3_dis": 0.98 * q,
                         "ic_height_p4": 20 - k, "hysteresis_c25_mid": 0.025 + 0.001 * k})
    t = pd.DataFrame(rows)
    t["valid"] = t["q_c25_dis"].notna()
    ref = t[t["valid"]].sort_values("cycle").groupby("cell")["q_c25_dis"].first()
    t["Capacity_Ah"] = t["q_c25_dis"]
    t["Reference_Capacity_Ah"] = t["cell"].map(ref)
    t["SOH_Percent"] = t["Capacity_Ah"] / t["Reference_Capacity_Ah"] * 100
    return t


def test_samples_skip_invalid_checks_and_use_no_future():
    s = W.build_samples(_table())
    assert 100 not in set(s.loc[s["cell"] == "cell1a", "origin_cycle"]) | set(s.loc[s["cell"] == "cell1a", "target_cycle"])
    a = s[(s["cell"] == "cell7a") & (s["origin_cycle"] == 300) & (s["horizon"] == 2)].iloc[0]
    assert a["target_cycle"] == 500 and a["cycles_to_target"] == 200 and a[FC.TARGET] == pytest.approx(-8.0)
    assert a["fade_rate_pp_per_cycle"] == pytest.approx(-0.04)
    t2 = _table()
    t2.loc[t2["cycle"] > 500, "SOH_Percent"] -= 10
    b = W.build_samples(t2)
    f1 = W.feature_sets(s)["F1"]
    m = s["origin_cycle"] <= 500
    pd.testing.assert_frame_equal(s.loc[m, f1].reset_index(drop=True), b.loc[m, f1].reset_index(drop=True))
    assert not {"dsoh", "soh_target", "Capacity_Ah", "SOH_Percent"} & set(f1)


def test_tasks_are_test_cases_and_windows_are_per_cell():
    s = W.build_samples(_table())
    assert dp.groups_in(s) == ["case1", "case7"]
    tasks = FC.forecast_stream(s, ["case7", "case1"])
    assert set(tasks[1].train["cell"]) == {"cell1a", "cell1b"}
    assert s.groupby(["Battery_ID", "Cycling_Test", "Cycle_Index"]).size().max() == 1   # unique TFT window keys
    assert W.cell_case("Cell4b") == "case4"
