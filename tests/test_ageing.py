"""Tests for the D22 ageing-cycle responses (synthetic rows built in the test; never real data)."""

import numpy as np
import pandas as pd
import pytest

from src import ageing as A


def _cycle(cyc: int, v_end_dis: float = 3.10, skip_rest: bool = False, reach_limit: bool = True) -> list[dict]:
    rows, t = [], 44000.0 + cyc
    def add(step, i, v, n):
        nonlocal t
        for k in range(n):
            t += 1e-5
            rows.append({"DateTime": t, "Step_Index": step, "Cycle_Index": cyc, "Current": i[k] if hasattr(i, "__len__") else i,
                         "Voltage": v[k] if hasattr(v, "__len__") else v, "Step_Time": float(k), "Discharge_Capacity": 0.0,
                         "Internal_Resistance": 0.0})
    vmax = 3.65 if reach_limit else 3.62
    add(2, [0.37] * 8 + [0.2, 0.1], list(np.linspace(3.3, vmax, 8)) + [vmax, vmax], 10)   # CC then 2 s CV hold
    if not skip_rest:
        add(3, 0.0, [3.60, 3.58, 3.57], 3)                                                  # rest after charge
    add(5, -0.14, list(np.linspace(3.5, v_end_dis, 5)), 5)                                  # discharge
    rows[-1]["Discharge_Capacity"] = 0.84
    add(6, 0.0, [v_end_dis + 0.05, v_end_dis + 0.08], 2)                                    # rest after discharge
    return rows


def test_responses_of_a_valid_cycle_and_an_invalid_one():
    raw = pd.DataFrame(_cycle(1) + _cycle(2, skip_rest=True))
    c = A.cycle_responses(raw, first_cycle=101, v_limit=3.65, cccv=True)
    ok, bad = c.iloc[0], c.iloc[1]
    assert ok["valid"] and not bad["valid"]
    assert ok["global_cycle"] == 101 and bad["global_cycle"] == 102
    assert ok["v_end_charge"] == pytest.approx(3.65)
    assert ok["cv_seconds"] == pytest.approx(2.0)                   # samples at >= 3.648 V: step times 7..9
    assert ok["cv_end_current"] == pytest.approx(0.1)
    assert ok["v_end_discharge"] == pytest.approx(3.10)
    assert ok["relax_after_charge"] == pytest.approx(3.57 - 3.60)
    assert ok["relax_after_discharge"] == pytest.approx(0.03)
    assert np.isnan(ok["internal_resistance"])
    assert ok["q_discharged"] == pytest.approx(0.84)
    assert np.isnan(bad.get("v_end_charge", np.nan))


def test_cc_only_cases_have_no_cv_inputs_and_unreached_limit_gives_zero_cv():
    raw = pd.DataFrame(_cycle(1, reach_limit=False))
    cc = A.cycle_responses(raw, 1, v_limit=3.65, cccv=False).iloc[0]
    assert np.isnan(cc["cv_seconds"]) and np.isnan(cc["cv_end_current"])
    cccv = A.cycle_responses(raw, 1, v_limit=3.65, cccv=True).iloc[0]
    assert cccv["cv_seconds"] == 0.0 and np.isnan(cccv["cv_end_current"])


def test_block_features_need_enough_valid_cycles():
    rows = []
    for k in range(1, A.MIN_VALID_CYCLES + 1):
        rows += _cycle(k, v_end_dis=3.20 - 0.001 * k)
    raw = pd.DataFrame(rows)
    f = A.block_features(A.cycle_responses(raw, 1, 3.65, True))
    assert f["age_valid_cycles"] == A.MIN_VALID_CYCLES
    assert f["age_v_end_discharge_last"] == pytest.approx(3.20 - 0.001 * A.MIN_VALID_CYCLES)
    assert f["age_v_end_discharge_slope"] == pytest.approx(-0.001)
    short = A.block_features(A.cycle_responses(pd.DataFrame(_cycle(1)), 1, 3.65, True))
    assert all(np.isnan(short[n]) for n in A.feature_names())


def test_block_range_parses_file_names():
    assert A.block_range("x/WWH_18650_cell1a_aging0101to0200.csv.xz") == ("Cell1a", 101, 200)
    with pytest.raises(ValueError):
        A.block_range("WWH_18650_cell1a_initial.csv")


def test_late_life_cycle_with_split_charge_and_discharge_is_valid():
    # C, pause, C (CV hold), rest, short low-current step, rest, WLTP discharge, rest, CC discharge, rest, rest
    plan = [(2, 0.37, [3.25, 3.40, 3.65]), (3, 0.0, [3.647, 3.640]), (2, 0.06, [3.65, 3.65, 3.65]),
            (4, 0.0, [3.647, 3.600, 3.582]), (5, -0.001, [3.58, 3.55]), (6, 0.0, [3.586, 3.583]),
            (7, -0.13, [3.58, 3.30, 3.20]), (8, 0.0, [3.21, 3.23]), (9, -0.72, [3.20, 3.129]),
            (10, 0.0, [3.207, 3.220, 3.231]), (11, 0.0, [3.231, 3.231])]
    rows, t = [], 44000.0
    for step, i, volts in plan:
        for k, v in enumerate(volts):
            t += 1e-5
            rows.append({"DateTime": t, "Step_Index": step, "Cycle_Index": 1, "Current": i, "Voltage": v,
                         "Step_Time": float(k), "Discharge_Capacity": 0.0, "Internal_Resistance": 0.0})
    c = A.cycle_responses(pd.DataFrame(rows), 1, v_limit=3.65, cccv=True).iloc[0]
    assert c["valid"]
    assert c["v_end_charge"] == pytest.approx(3.65) and c["cv_seconds"] == pytest.approx(2.0)
    assert c["relax_after_charge"] == pytest.approx(3.582 - 3.647)      # the single rest step after the last charge
    assert c["v_end_discharge"] == pytest.approx(3.129)
    assert c["relax_after_discharge"] == pytest.approx(3.231 - 3.207)   # not merged with the following rest
