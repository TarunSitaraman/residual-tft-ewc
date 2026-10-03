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
