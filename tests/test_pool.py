"""Tests for the D21 pool tables (synthetic frames built in the test; never used for results)."""

import numpy as np
import pandas as pd
import pytest

from src import forecasting as FC
from src import pool


def _eoc(n_cu: int = 8, start: str = "2023-01-01", step_days: int = 20, fade: float = 0.02) -> pd.DataFrame:
    """Synthetic S7-like eoc frame: one operation row, then charge + discharge rows per check-up."""
    rows = [{"timestamp_s": pd.Timestamp(start).timestamp(), "cyc_condition": 0, "cyc_charged": 1,
             "i_dischg_min_A": -3.0, "delta_q_dischg_Ah": -2.0, "num_cycles_checkup": 0, "t_start_degC": 40.0,
             "age_type": 2, "age_temp": 40, "age_soc": 50, "age_chg_rate": 1.0, "age_dischg_rate": 1.0,
             "age_profile": 0, "v_max_target_V": 4.1, "v_min_target_V": 3.0}]
    for k in range(n_cu):
        t = (pd.Timestamp(start) + pd.Timedelta(days=step_days * k)).timestamp()
        common = {k2: rows[0][k2] for k2 in pool.S7_STATIC}
        rows.append({**common, "timestamp_s": t, "cyc_condition": 2, "cyc_charged": 1, "i_dischg_min_A": -1.0,
                     "delta_q_dischg_Ah": 0.0, "num_cycles_checkup": k, "t_start_degC": 25.0})
        rows.append({**common, "timestamp_s": t + 3600, "cyc_condition": 2, "cyc_charged": 0, "i_dischg_min_A": -1.0,
                     "delta_q_dischg_Ah": -(3.0 - fade * k), "num_cycles_checkup": k, "t_start_degC": 25.0})
    return pd.DataFrame(rows)


def test_s7_checks_use_measured_checkup_discharges_only():
    eoc = _eoc(n_cu=5)
    eoc.loc[eoc.index[-1], "i_dischg_min_A"] = -0.5          # off-protocol check-up discharge: excluded
    c = pool.s7_cell_checks(eoc, "P001-1", "P001")
    assert len(c) == 4                                        # 5 check-up discharges, one off-protocol
    assert (c["Capacity_Ah"] > 0).all()                       # absolute value of the negative stored charge
    np.testing.assert_allclose(c["Capacity_Ah"], [3.0, 2.98, 2.96, 2.94])
    assert c["age_temp"].eq(40).all() and c["v_min_target_V"].eq(3.0).all()   # static from operation rows


def test_s7_excludes_checkups_after_thermal_failure():
    eoc = _eoc(n_cu=6, start="2024-05-01", step_days=7)     # check-ups on 1, 8, 15, 22, 29 May and 5 Jun
    c = pool.s7_cell_checks(eoc, "P001-1", "P001")
    assert (c["check_time"] < pool.S7_LAST_VALID).all() and len(c) == 4


def test_soh_is_relative_to_first_valid_check():
    t = pool.add_soh(pool.s7_cell_checks(_eoc(n_cu=6), "P001-1", "P001"))
    assert t["SOH_Percent"].iloc[0] == pytest.approx(100.0)
    np.testing.assert_allclose(t["SOH_Percent"], t["Capacity_Ah"] / t["Capacity_Ah"].iloc[0] * 100)


def test_samples_targets_inputs_and_minimum_length():
    a = pool.s7_cell_checks(_eoc(n_cu=8), "P001-1", "P001")
    b = pool.s7_cell_checks(_eoc(n_cu=4), "P002-1", "P002")          # below the 5-check minimum
    t = pool.add_soh(pd.concat([a, b], ignore_index=True))
    s = pool.build_samples(t, static=["age_temp"])
    assert set(s["cell"]) == {"P001-1"}
    assert len(s) == 7 + 6 + 5                                       # h = 1, 2, 3 on 8 checks
    np.testing.assert_allclose(s[pool.TARGET], s["soh_target"] - s["soh_origin"])
    assert (s["days_to_target"] == 20 * s["horizon"]).all()
    assert s.loc[s["origin_pos"] < 2, "fade_rate_pp_per_day"].isna().all()
    # fade rate at origin k uses checks k-2..k only: constant fade gives the same slope everywhere
    rates = s.loc[s["origin_pos"] >= 2, "fade_rate_pp_per_day"]
    np.testing.assert_allclose(rates, rates.iloc[0])


def test_samples_work_with_the_purged_splits():
    t = pool.add_soh(pool.s7_cell_checks(_eoc(n_cu=20), "P001-1", "P001"))
    sp = FC.assign_split(pool.build_samples(t, static=["age_temp"]))
    tr, va, te = (sp[sp["split"] == k] for k in ("train", "val", "test"))
    assert len(tr) and len(va) and len(te)
    assert tr["target_pos"].max() <= va["origin_pos"].min() and va["target_pos"].max() <= te["origin_pos"].min()


def test_s7_excludes_cold_checkups():
    eoc = _eoc(n_cu=6)
    eoc.loc[eoc.index[-1], "t_start_degC"] = 1.3          # a check-up that started cold (as CU 11 of P00x cells)
    c = pool.s7_cell_checks(eoc, "P001-1", "P001")
    assert len(c) == 5 and (c["check_temp_c"] >= pool.S7_MIN_CHECKUP_TEMP_C).all()


def test_block_checks_average_complete_blocks_only():
    meas = pd.DataFrame({"cycle": np.arange(1, 26), "capacity_ah": np.arange(25, dtype=float),
                         "elapsed_s": np.arange(25) * 3600.0})
    b = pool.block_checks(meas, 10)
    assert len(b) == 2                                          # the incomplete third block is dropped
    np.testing.assert_allclose(b["Capacity_Ah"], [4.5, 14.5])
    assert b["cycle"].tolist() == [10, 20] and b["elapsed_s"].tolist() == [9 * 3600.0, 19 * 3600.0]


def test_s9_measurements_require_a_full_discharge():
    rows = []
    for cyc, (vmin, q) in enumerate([(2.65, 3200), (2.65, 3190), (2.90, 2500), (2.65, 3180)], start=1):
        rows += [{"cycle number": cyc, "time/s": cyc * 100.0, "Ecell/V": 4.2, "Q discharge/mA.h": 0.0},
                 {"cycle number": cyc, "time/s": cyc * 100.0 + 50, "Ecell/V": vmin, "Q discharge/mA.h": q}]
    m = pool.s9_cell_measurements(pd.DataFrame(rows))
    assert m["cycle"].tolist() == [1, 2, 4]                     # cycle 3 stopped at 2.90 V: not a full discharge
    np.testing.assert_allclose(m["capacity_ah"], [3.2, 3.19, 3.18])


def test_samples_include_planned_cycles_when_known():
    t = pool.add_soh(pd.DataFrame({"dataset": "S9", "cell": "c1", "condition": "k", "cycle": np.arange(10, 90, 10),
                                   "Capacity_Ah": np.linspace(3.0, 2.8, 8),
                                   "check_time": pd.Timestamp("2000-01-01") + pd.to_timedelta(np.arange(8), unit="D")}))
    s = pool.build_samples(t, static=[])
    assert (s["cycles_to_target"] == 10 * s["horizon"]).all() and (s["cycles_since_first"] == 10 * s["origin_pos"]).all()


def test_s9_time_restart_is_made_continuous():
    rows = []
    for cyc in range(1, 5):
        base = cyc * 100.0 if cyc <= 2 else (cyc - 2) * 100.0       # counter restarts after cycle 2
        rows += [{"cycle number": cyc, "time/s": base, "Ecell/V": 4.2, "Q discharge/mA.h": 0.0},
                 {"cycle number": cyc, "time/s": base + 50, "Ecell/V": 2.65, "Q discharge/mA.h": 3200 - cyc}]
    m = pool.s9_cell_measurements(pd.DataFrame(rows))
    assert m.attrs["time_resets"] == 1
    assert m["elapsed_s"].is_monotonic_increasing and m["cycle"].tolist() == [1, 2, 3, 4]


def test_s9_short_segments_touching_cutoff_are_not_full_discharges():
    rows = []
    for cyc in range(1, 8):
        q = 22.0 if cyc == 4 else 3200.0 - cyc                      # cycle 4: a 22 mAh segment that reaches 2.65 V
        rows += [{"cycle number": cyc, "time/s": cyc * 100.0, "Ecell/V": 4.2, "Q discharge/mA.h": 0.0},
                 {"cycle number": cyc, "time/s": cyc * 100.0 + 50, "Ecell/V": 2.65, "Q discharge/mA.h": q}]
    m = pool.s9_cell_measurements(pd.DataFrame(rows))
    assert 4 not in m["cycle"].tolist() and len(m) == 6


def test_s9_reference_cycles_at_another_current_are_dropped():
    rows = []
    for cyc in range(1, 8):
        i, q = (-2500.0, 3550.0) if cyc == 5 else (-5000.0, 2400.0 - cyc)   # cycle 5: a slow reference cycle
        rows += [{"cycle number": cyc, "time/s": cyc * 100.0, "Ecell/V": 4.2, "Q discharge/mA.h": 0.0, "<I>/mA": 2000.0},
                 {"cycle number": cyc, "time/s": cyc * 100.0 + 50, "Ecell/V": 2.5, "Q discharge/mA.h": q, "<I>/mA": i}]
    m = pool.s9_cell_measurements(pd.DataFrame(rows))
    assert m["cycle"].tolist() == [1, 2, 3, 4, 6, 7]


def test_s9_cycles_with_two_discharge_steps_are_dropped():
    rows = []
    for cyc in range(1, 7):
        rows += [{"cycle number": cyc, "time/s": cyc * 100.0, "Ecell/V": 4.2, "Q discharge/mA.h": 0.0, "<I>/mA": 2500.0},
                 {"cycle number": cyc, "time/s": cyc * 100.0 + 10, "Ecell/V": 3.0, "Q discharge/mA.h": 1200.0, "<I>/mA": -2500.0}]
        if cyc == 3:                                                  # rest, then a second discharge in the same cycle
            rows += [{"cycle number": cyc, "time/s": cyc * 100.0 + 20, "Ecell/V": 3.4, "Q discharge/mA.h": 1200.0, "<I>/mA": 0.0},
                     {"cycle number": cyc, "time/s": cyc * 100.0 + 30, "Ecell/V": 2.5, "Q discharge/mA.h": 3500.0, "<I>/mA": -2500.0}]
        else:
            rows += [{"cycle number": cyc, "time/s": cyc * 100.0 + 30, "Ecell/V": 2.5, "Q discharge/mA.h": 2400.0 - cyc, "<I>/mA": -2500.0}]
    m = pool.s9_cell_measurements(pd.DataFrame(rows))
    assert 3 not in m["cycle"].tolist() and len(m) == 5
