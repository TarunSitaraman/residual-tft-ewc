"""Leakage and split tests for SOH forecasting samples (D18) on a small hand-made table."""

import numpy as np
import pandas as pd
import pytest

from src import data_pipeline as dp
from src import forecasting as FC


def _table(n_rpt=12, batteries=("g1", "v4")):
    rows = []
    for b in batteries:
        for k in range(1, n_rpt + 1):
            rows.append({dp.GROUP_COL: b, "rpt_number": k, "rpt_start": pd.Timestamp("2023-01-01") + pd.Timedelta(days=30 * k),
                         "SOH_Percent": 100 - 0.3 * (k - 1) + (0.5 if b == "v4" else 0), "eis_r0_3.63": 0.025 + 1e-4 * k,
                         "q_4.20_4.00": 1.0 - 0.01 * k})
    return pd.DataFrame(rows)


def _blocks(batteries=("g1", "v4"), n=12):
    rows = []
    for b in batteries:
        for k in range(1, n + 1):
            start = pd.Timestamp("2023-01-01") + pd.Timedelta(days=30 * k + 2)
            rows.append({dp.GROUP_COL: b, "Cycling_Test": f"Cycling_{k}", "start": start, "end": start + pd.Timedelta(days=20),
                         "temp_median": 35.0 if k % 2 else 20.0, "setpoint_c": 35.0 if k % 2 else 20.0})
    return pd.DataFrame(rows)


@pytest.fixture()
def samples():
    return FC.build_samples(_table(), _blocks())


def test_targets_and_rpt2_exclusion(samples):
    assert 2 not in set(samples["origin_rpt"]) | set(samples["target_rpt"])
    s = samples[(samples[dp.GROUP_COL] == "g1") & (samples["origin_rpt"] == 3) & (samples["horizon"] == 2)].iloc[0]
    assert s["target_rpt"] == 5 and s[FC.TARGET] == pytest.approx(-0.6)
    assert s["soh_origin"] + s[FC.TARGET] == pytest.approx(s["soh_target"])


def test_inputs_use_only_origin_and_earlier(samples):
    # Changing every SOH after an origin must not change that origin's inputs.
    t = _table()
    t2 = t.copy()
    t2.loc[t2["rpt_number"] > 6, "SOH_Percent"] += 5.0
    a = FC.build_samples(t, _blocks())
    b = FC.build_samples(t2, _blocks())
    feats = FC.feature_sets(a)["F1"]
    m = a["origin_rpt"] <= 6
    pd.testing.assert_frame_equal(a.loc[m, feats].reset_index(drop=True), b.loc[m, feats].reset_index(drop=True))
    assert not {"dsoh", "soh_target", "target_rpt"} & set(feats)


def test_fade_rate_needs_three_rpts(samples):
    g = samples[(samples[dp.GROUP_COL] == "g1") & (samples["horizon"] == 1)].sort_values("origin_rpt")
    assert g["fade_rate_pp_per_day"].iloc[:2].isna().all()                  # RPT 1 and 3: fewer than 3 valid RPTs
    assert g["fade_rate_pp_per_day"].iloc[2] == pytest.approx(-0.01, rel=1e-6)


def test_schedule_days():
    out = FC.interval_schedule(_blocks(), "g1", pd.Timestamp("2023-01-31"), pd.Timestamp("2023-03-02"))
    assert out["days_to_target"] == pytest.approx(30)
    assert out["days_35c"] == pytest.approx(20) and out["days_20c"] == pytest.approx(0)
    assert out["days_rest"] == pytest.approx(10)


def test_splits_are_purged(samples):
    s = FC.assign_split(samples)
    for b, g in s.groupby(dp.GROUP_COL):
        a, c = FC.split_cuts(int(g["n_valid_rpts"].iloc[0]))
        tr, va, te = (g[g["split"] == k] for k in ("train", "val", "test"))
        assert tr["target_pos"].max() <= va["origin_pos"].min()             # no train label after a val origin
        assert va["target_pos"].max() <= te["origin_pos"].min()             # no val label after a test origin
        assert (tr["target_pos"] <= a).all() and (te["origin_pos"] >= c).all()


def test_stream_and_leave_one_out(samples):
    tasks = FC.forecast_stream(samples, ["v4", "g1"])
    assert [t.name for t in tasks] == ["v4", "g1"] and all(len(t.test) for t in tasks)
    lo = FC.leave_one_out_task(samples, "g1")
    assert set(lo.test[dp.GROUP_COL]) == {"g1"} and "g1" not in set(lo.train[dp.GROUP_COL]) | set(lo.val[dp.GROUP_COL])


def test_baselines_shapes(samples):
    s = FC.assign_split(samples)
    train, test = s[s["split"] == "train"], s[s["split"] == "test"]
    preds = FC.baseline_predictions(train, {"t": test}, FC.feature_sets(samples))
    assert set(preds) >= {"persistence", "linear_extrapolation", "ridge_F0", "ridge_F1", "gbm_F1"}
    assert all(len(p["t"]) == len(test) for p in preds.values())
    assert np.allclose(preds["persistence"]["t"], 0.0)
    errs = FC.forecast_errors(test, preds["linear_extrapolation"]["t"])
    assert errs["RMSE"] == pytest.approx(0.0, abs=1e-9)                     # exact linear fade
