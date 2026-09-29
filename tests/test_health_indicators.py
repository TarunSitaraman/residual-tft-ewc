"""Tests for health-indicator extraction (D14) on a small hand-made point-level block."""

import numpy as np
import pandas as pd
import pytest

from src import data_pipeline as dp
from src import health_indicators as HI


def _block(battery="G1", test="Cycling_2", start="2023-01-01", cc_current=1.46, dis_current=-4.5, n_cc=40):
    """One standardised monthly block: steps 1-7 in cycle 1, plus one later duty cycle."""
    t0 = pd.Timestamp(start)
    rows, t = [], 0.0

    def add(step, cycle, n, current, v_from, v_to, qc0=0.0, qc_rate=0.0, qd0=0.0, qd_rate=0.0, temp=25.0, dt=60.0):
        nonlocal t
        for k in range(n):
            rows.append({"Date_Time": t0 + pd.Timedelta(seconds=t), "Test_Time(s)": t, "Step_Time(s)": (k + 1) * dt,
                         "Step_Index": step, "Cycle_Index": cycle, "Voltage(V)": v_from + (v_to - v_from) * k / max(n - 1, 1),
                         "Current(A)": current, "Charge_Capacity(Ah)": qc0 + qc_rate * (k + 1),
                         "Discharge_Capacity(Ah)": qd0 + qd_rate * (k + 1), "Discharge_Energy(Wh)": 3.7 * (qd0 + qd_rate * (k + 1)),
                         "Aux_Temperature(C)_1": temp + (2.0 * k / n if step == 5 else 0.0),
                         "Source_Folder": battery, "Cycling_Test": test})
            t += dt
    add(1, 1, 3, 0.0, 3.50, 3.50)
    add(2, 1, n_cc, cc_current, 3.50, 4.20, qc_rate=cc_current / 60)           # CC charge, Ah = I*t/3600
    q_cc = cc_current / 60 * n_cc
    add(3, 1, 20, 0.3, 4.20, 4.20, qc0=q_cc, qc_rate=0.005)                     # CV
    add(4, 1, 5, 0.0, 4.19, 4.185, qc0=q_cc + 0.1)                              # rest
    add(5, 1, 30, dis_current, 3.95, 3.50, qd_rate=-dis_current / 60)          # discharge
    add(6, 1, 5, 0.0, 3.70, 3.72)                                                # rest
    add(7, 1, 10, 0.0, 3.55, 4.00, dt=900.0)                                     # duty cycle
    add(7, 2, 10, 0.0, 3.56, 3.99, dt=900.0)
    return pd.DataFrame(rows)


def test_window_charge():
    v = np.linspace(3.5, 4.2, 71)
    q = np.linspace(0.0, 3.5, 71)          # 5 Ah per volt
    assert HI.window_charge(v, q, 3.7, 3.9) == pytest.approx(1.0)
    assert np.isnan(HI.window_charge(v, q, 3.4, 3.9))   # charge starts above the window
    assert np.isnan(HI.window_charge(v, q, 3.9, 4.3))


def test_extract_block_indicators_values():
    b = HI.extract_block_indicators(_block()).iloc[0]
    assert b[dp.GROUP_COL] == "g1" and not b["nonstandard_charge"] and b["missing_steps"] == []
    assert b["cc_current"] == pytest.approx(1.46)
    assert b["cc_charge_ah"] == pytest.approx(1.46 / 60 * 39)            # max - min over the step
    assert b["dis_charge_ah"] == pytest.approx(4.5 / 60 * 29)
    assert b["ir_proxy_ohm"] == pytest.approx((4.185 - 3.95) / 4.5)
    assert b["rest_charge_dv"] == pytest.approx(0.005)
    assert np.isfinite(b["dq_3.7_3.9"]) and b["dis_temp_rise"] > 0
    assert 3.55 <= b["duty_v_p05"] < b["duty_v_p95"] <= 4.0


def test_nonstandard_and_missing_steps_are_flagged():
    b = HI.extract_block_indicators(_block(cc_current=1.05)).iloc[0]
    assert b["nonstandard_charge"] and np.isnan(b["dq_3.7_3.9"])
    pts = _block()
    b2 = HI.extract_block_indicators(pts[pts["Step_Index"] != 5]).iloc[0]
    assert 5 in b2["missing_steps"] and np.isnan(b2.get("dis_end_v", np.nan))


def test_labels_and_protocol_scaling():
    pts = pd.concat([_block(start="2023-01-01", dis_current=-4.60), _block(test="Cycling_3", start="2023-02-01", dis_current=-4.55)])
    health = pd.DataFrame({dp.GROUP_COL: "g1", "RPT": ["RPT_1", "RPT_2", "RPT_3"], "RPT_Number": [1, 2, 3],
                           "Test_Date": pd.to_datetime(["2022-12-20", "2023-01-20", "2023-02-20"]),
                           "Capacity_Ah": [4.60, 4.55, 4.50], "Reference_Capacity_Ah": 4.60,
                           "SOH_Percent": [100.0, 4.55 / 4.60 * 100, 4.50 / 4.60 * 100]})
    blocks = HI.attach_rpt_labels(HI.extract_block_indicators(pts), health)
    assert blocks["rpt_prev"].tolist() == ["RPT_1", "RPT_2"] and blocks["rpt_next"].tolist() == ["RPT_2", "RPT_3"]
    assert blocks["days_since_first_rpt"].tolist() == [12, 43]
    ratio = HI.protocol_scaling(blocks)
    assert ratio.loc["ALL", "mean"] == pytest.approx(1.0) and ratio.loc["ALL", "count"] == 2


def test_protocol_scaled_quantities_are_not_indicators():
    assert not set(HI.PROTOCOL_SCALED) & set(HI.INDICATORS)
