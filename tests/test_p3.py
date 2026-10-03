"""Tests for protocol P3 (D21): shared inputs, pretraining and the stream (synthetic data built in the test)."""

import numpy as np
import pandas as pd
import pytest

from src import forecasting as FC
from src import p3


def _checks(dataset: str, n_cells: int = 4, n_checks: int = 12, condition: str = "c0") -> pd.DataFrame:
    rows = []
    for c in range(n_cells):
        for k in range(n_checks):
            rows.append({"dataset": dataset, "cell": f"{condition}-cell{c}", "condition": condition, "cycle": 10 * (k + 1),
                         "check_time": pd.Timestamp("2000-01-01") + pd.Timedelta(days=5 * k),
                         "Capacity_Ah": 3.0 - 0.01 * (c + 1) * k})
    from src import pool
    return pool.add_soh(pd.DataFrame(rows))


def test_s9_rate_codes():
    assert p3._rate("025") == 0.25 and p3._rate("05") == 0.5 and p3._rate("1") == 1.0 and p3._rate("4") == 4.0


def test_s9_conditions_from_names():
    t = pd.DataFrame({"condition": ["S9-D1_NCA_battery-CY25-025_1", "S9-D3_NCM_NCA_battery-CY25-05_4"]})
    c = p3.conditions_s9(t)
    assert c["temp_c"].tolist() == [25.0, 25.0] and c["charge_c_rate"].tolist() == [0.25, 0.5]
    assert c["discharge_c_rate"].tolist() == [1.0, 4.0] and c["chemistry"].tolist() == ["NCA", "NMC_NCA"]


def test_harmonised_samples_have_all_shared_inputs_and_unique_window_keys():
    s = p3.harmonised_samples(_checks("S9", condition="S9-D2_NCM_battery-CY45-05_1"), "S9")
    assert set(p3.SHARED_FEATURES) <= set(s.columns)
    assert not s.duplicated(subset=["Battery_ID", "Cycling_Test", "Cycle_Index"]).any()
    assert (s["chem_NMC"] == 1).all() and (s["Battery_ID"] == "S9").all()


def _setup():
    pool_s = p3.harmonised_samples(_checks("S9", condition="S9-D2_NCM_battery-CY45-05_1"), "S9")
    s6 = pd.concat([p3.harmonised_samples(_checks("S6", n_cells=2, condition=f"case{i}"), "S6") for i in (1, 2)],
                   ignore_index=True)
    return p3.pool_task(pool_s), FC.forecast_stream(s6, ["case1", "case2"]), pd.concat([pool_s, s6], ignore_index=True)


@pytest.mark.parametrize("method", ["ewc", "tft_ewc"])
def test_pretraining_fits_on_the_pool_and_anchors_ewc(method):
    pool_t, stream, history = _setup()
    settings = {"epochs": 2, "ewc_lambda": 10.0, **({"window": 2, "window_group": ("Battery_ID", "Cycling_Test")}
                                                   if method.startswith("tft") else {})}
    s = p3.build_strategy(method, list(p3.SHARED_FEATURES), seed=0, history=history, settings=settings)
    p3.pretrain(s, pool_t)
    assert s.tasks_seen_ == ["pool"] and len(s.fishers_) == 1          # anchor = pretrained weights
    assert s.preprocessor.n_fit_rows_ == len(pool_t.train)               # fitted on the pool's training rows
    before = (s.preprocessor.median_.copy(), s.preprocessor.mean_.copy(), s.target_mean_, s.target_scale_)
    out = p3.run_stream(s, stream, pool_t)
    assert out["R"].shape == (2, 2) and np.isfinite(out["pool_error_before"]) and np.isfinite(out["pool_error_after"])
    assert len(s.fishers_) == 3                                          # pool + two stream tasks
    np.testing.assert_array_equal(s.preprocessor.median_, before[0])     # preprocessing frozen from the pool
    np.testing.assert_array_equal(s.preprocessor.mean_, before[1])
    assert (s.target_mean_, s.target_scale_) == before[2:]


def test_scratch_strategy_has_no_pool_error_before():
    pool_t, stream, history = _setup()
    s = p3.build_strategy("finetune", list(p3.SHARED_FEATURES), seed=0, history=None, settings={"epochs": 2})
    out = p3.run_stream(s, stream, pool_t)
    assert np.isnan(out["pool_error_before"]) and np.isfinite(out["pool_error_after"])
