"""Tests for the raw-diagnostic readers and features on small hand-made inputs."""

import numpy as np
import pandas as pd
import pytest

from src import diagnostic_features as F
from src import raw_diagnostics as rd


def _capacity_test(pulse_a=2.3, c20_a=0.23, r_ohm=0.03):
    """C/20 discharge 4.2 -> 2.5 V (linear OCV in charge) with pulse sequences at 4.00/3.63/3.26 V."""
    rows, t, q = [], 0.0, 0.0
    step = 1

    def ocv(qq):
        return 4.2 - 1.7 * qq / 4.5

    def add(n, dt, current, v_fn):
        nonlocal t, q, step
        for k in range(n):
            t += dt
            if current < 0:
                q += -current * dt / 3600
            rows.append({"Test_Time(s)": t, "Step_Time(s)": (k + 1) * dt, "Step_Index": step, "Cycle_Index": 1,
                         "Voltage(V)": v_fn(k), "Current(A)": current, "Discharge_Capacity(Ah)": q,
                         "Date_Time": pd.Timestamp("2023-01-01") + pd.Timedelta(seconds=t)})
        step += 1

    for stop in (4.00, 3.63, 3.26, 2.50):
        n = 0
        while ocv(q + c20_a * (n + 1) * 10 / 3600) - r_ohm * c20_a > stop:
            n += 1
        q0 = q
        add(n, 10.0, -c20_a, lambda k, q0=q0: ocv(q0 + c20_a * (k + 1) * 10 / 3600) - r_ohm * c20_a)
        if stop > 2.6:
            v_ref = rows[-1]["Voltage(V)"]
            add(80, 0.1, -pulse_a, lambda k: v_ref - r_ohm * (pulse_a - c20_a))
            add(100, 0.1, 0.0, lambda k: v_ref + r_ohm * c20_a)
            add(80, 0.1, pulse_a, lambda k: v_ref + r_ohm * (pulse_a + c20_a))
    return pd.DataFrame(rows)


def test_parse_raw_name_variants(tmp_path):
    p = tmp_path / "RPT_9" / "EIS_9" / "20230816_w9_EIS400V_EIS2_Channel_3.xlsx"
    f = rd.parse_raw_name(p)
    assert (f.rpt_number, f.battery, f.kind, f.eis_voltage, f.suffix) == (9, "w9", "EIS", 4.00, "_EIS2")
    f = rd.parse_raw_name(tmp_path / "RPT_4" / "20230228_sd_w9_RPT_Channel_3.xlsx")
    assert (f.kind, f.battery, f.eis_voltage) == ("RPT", "w9", None)
    assert rd.parse_raw_name(tmp_path / "RPT_5" / "20230327_EIS_5_326V_ALL.xlsx") is None
    assert rd.parse_raw_name(tmp_path / "20230327_G1_EIS326_Channel_6.xlsx") is None     # no RPT folder


def test_impedance_sign_convention():
    sheets = {"ACIM_Chan1": pd.DataFrame({"Freq": [1000.0, 1.0], "Zmod": [0.03, 0.04], "Zphz": [30.0, -30.0]})}
    z = rd.impedance(sheets)
    assert z["z_imag"].iloc[0] > 0 > z["z_imag"].iloc[1]                  # inductive at high f, capacitive at low f
    assert z["z_real"].iloc[0] == pytest.approx(0.03 * np.cos(np.deg2rad(30)))


def test_rpt_features_capacity_and_resistance():
    ts = _capacity_test()
    r = F.rpt_features(ts)
    assert r["n_c20_segments"] == 4 and not r["nonstandard"] and r["nonstandard_pulses"] == []
    # Charge within the C/20 steps only (max - min per step), pulses excluded.
    c20 = ts[ts["Current(A)"] == -0.23].groupby("Step_Index")["Discharge_Capacity(Ah)"].agg(lambda s: s.max() - s.min())
    assert r["q_c20_ah"] == pytest.approx(c20.sum())
    assert r["q_c20_ah"] < ts["Discharge_Capacity(Ah)"].max()
    assert sum(r[k] for k in ("q_4.20_4.00", "q_4.00_3.63", "q_3.63_3.26", "q_3.26_2.50")) == pytest.approx(r["q_c20_ah"])
    for v in ("4.00", "3.63", "3.26"):
        assert r[f"r_pulse_{v}_1s"] == pytest.approx(0.03, rel=1e-6)
    assert not set(F.PROTOCOL_CURRENTS) - set(r)                          # recorded for checks...
    assert not any(c.startswith(("c20_current", "pulse_current")) for c in F.feature_columns(pd.DataFrame(columns=list(r))))


def test_offprotocol_pulses_are_not_used():
    r = F.rpt_features(_capacity_test(pulse_a=1.05))
    assert r["nonstandard_pulses"] == [4.00, 3.63, 3.26]
    assert all(np.isnan(r[f"r_pulse_{v}_1s"]) for v in ("4.00", "3.63", "3.26"))


def test_ic_curve_of_linear_ocv_is_flat_and_ignores_pulses():
    ts = _capacity_test()
    c20 = [s for s in F._segments(ts) if F._kind(s) == "c20"]
    ic = F.ic_curve(c20)
    inside = (F.IC_GRID > 2.8) & (F.IC_GRID < 4.1)
    assert np.nanmedian(ic[inside]) == pytest.approx(4.5 / 1.7, rel=0.02)
    assert np.nanmax(ic[inside]) < 1.1 * 4.5 / 1.7                        # no spikes at the pulse voltages


def test_eis_features_on_randles_spectrum():
    f = np.logspace(4, -2, 43)
    w = 2 * np.pi * f
    r0, rct, cdl, l = 0.025, 0.004, 2.0, 1e-6
    z = r0 + rct / (1 + 1j * w * rct * cdl) + 1j * w * l + 0.002 * (1 - 1j) / np.sqrt(w)
    out = F.eis_features(pd.DataFrame({"freq_hz": f, "z_real": z.real, "z_imag": z.imag}), 3.63)
    assert out["eis_r0_3.63"] == pytest.approx(r0, abs=5e-4)
    assert 0.5 * rct < out["eis_rarc_3.63"] < 1.5 * rct
    assert out["eis_arc_peak_3.63"] > 0
    assert out["eis_zre_1000hz_3.63"] < out["eis_zre_0.01hz_3.63"]


def test_eis_features_empty_spectrum():
    out = F.eis_features(pd.DataFrame(columns=["freq_hz", "z_real", "z_imag"]), 3.26)
    assert all(np.isnan(v) for v in out.values())


def test_c20_soh_labels():
    from src import data_pipeline as dp

    t = pd.DataFrame({dp.GROUP_COL: ["a", "a", "a", "b", "b"], "rpt_number": [2, 1, 3, 1, 2],
                      "q_c20_ah": [4.4, 4.5, 4.3, 4.0, 3.9], "Test_Date": pd.to_datetime(["2023-02-01", "2023-01-01",
                      "2023-03-01", "2023-01-01", "2023-02-01"]), "SOH_handover": np.nan, "rpt_temp_mean": 25.0})
    out = F.add_c20_soh(t)
    assert out["Reference_Capacity_Ah"].tolist() == [4.5, 4.5, 4.5, 4.0, 4.0]          # first RPT, not first row
    assert out["SOH_Percent"].tolist() == pytest.approx([100, 4.4 / 4.5 * 100, 4.3 / 4.5 * 100, 100, 97.5])
    h = F.health_table(out)
    assert dp.validate_soh_formula(h)["ok"] and h["RPT"].iloc[0] == "RPT_1"
    assert "q_c20_ah" not in F.feature_columns(out)
