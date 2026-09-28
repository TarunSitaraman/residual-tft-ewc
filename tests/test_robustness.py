"""Tests for missing-temperature robustness tools (E11) on synthetic test data."""

import numpy as np
import pandas as pd
import pytest

from src import data_pipeline as dp
from src import experiments as X
from src import protocols as P
from src import robustness as RB
from src.models.baseline_dnn import TabularPreprocessor


def test_carry_forward_is_causal_and_per_battery(ml_train):
    df = ml_train.copy()
    g = df[df[dp.GROUP_COL] == "g1"].sort_values(["Cycle_Date", "Cycle_Index"])
    first, second = g.index[0], g.index[5]
    df.loc[first, list(RB.TEMPS)] = np.nan          # leading gap: must stay missing
    df.loc[second, list(RB.TEMPS)] = np.nan
    out = RB.carry_forward_temperature(df)
    assert out.loc[first, list(RB.TEMPS)].isna().all()
    prev = g.index[4]
    expected = out.loc[prev, "Temperature_Mean"]
    assert out.loc[second, "Temperature_Mean"] == expected
    assert out.index.equals(df.index)
    # No temperature is filled from another battery or from the future.
    assert out[RB.TEMPS[0]].notna().sum() >= df[RB.TEMPS[0]].notna().sum()
    assert np.allclose(out.loc[df[RB.TEMPS[0]].notna(), RB.TEMPS[0]], df.loc[df[RB.TEMPS[0]].notna(), RB.TEMPS[0]])


def test_mask_temperature_only_removes_values(ml_train):
    rows = pd.Series(True, index=ml_train.index)
    for mode in ("random", "block"):
        out, masked = RB.mask_temperature(ml_train, rows, 0.3, mode, seed=1)
        before, after = ml_train[list(RB.TEMPS)], out[list(RB.TEMPS)]
        assert (after.isna() | (after == before)).all().all()     # nothing added or changed
        assert after.loc[masked].isna().all().all()
        for _, g in ml_train.groupby(dp.GROUP_COL):
            assert masked.loc[g.index].sum() == round(0.3 * len(g))
        out2, masked2 = RB.mask_temperature(ml_train, rows, 0.3, mode, seed=1)
        assert masked.equals(masked2)                               # seeded


def test_block_mask_is_contiguous(ml_train):
    rows = ml_train[dp.GROUP_COL] == "w8"
    _, masked = RB.mask_temperature(ml_train, rows, 0.25, "block", seed=3)
    g = ml_train[rows].sort_values(["Cycle_Date", "Cycle_Index"])
    pos = np.flatnonzero(masked.loc[g.index].to_numpy())
    assert len(pos) > 0 and np.all(np.diff(pos) == 1)
    assert not masked[~rows].any()
    with pytest.raises(ValueError):
        RB.mask_temperature(ml_train, rows, 1.5)


def test_variants_and_hidden_indicators(ml_train):
    features = dp.model_feature_columns(ml_train)
    _, f_drop, _ = RB.prepare_variant("drop_temperature", ml_train, features)
    assert not set(f_drop) & set(RB.TEMPS) and len(f_drop) == len(features) - 3
    _, _, ind = RB.prepare_variant("impute_only", ml_train, features)
    assert ind is False
    x = TabularPreprocessor(features, indicators=False).fit(ml_train).transform(ml_train)
    assert x.shape[1] == 2 * len(features) and not x[:, len(features):].any()
    with pytest.raises(ValueError):
        RB.prepare_variant("interpolate", ml_train, features)


def test_stratified_errors(ml_train):
    flag = RB.temperature_missing(ml_train)
    res = RB.stratified_errors(ml_train, ml_train["SOH_Percent"] + 1.0, flag)
    pooled = res[res[dp.GROUP_COL] == "ALL"].set_index("subset")
    assert pooled.loc["temperature missing", "n"] == flag.sum()
    assert np.allclose(pooled["RMSE"], 1.0)


def test_tft_predictions_follow_history_swap(ml_train):
    features = dp.model_feature_columns(ml_train)
    task = P.make_task(ml_train, "g1")
    predict, cfg = X.fit_model("tft", task.train, task.val, features, mlp_kwargs={"epochs": 3, "patience": 2},
                               history=ml_train, lr_grid=[1e-2])
    base = predict(task.test)
    in_test = RB.rows_in(ml_train, task.test)
    assert in_test.sum() == len(task.test)
    masked, n_masked = RB.mask_temperature(ml_train, in_test, 1.0, "random", seed=0)
    assert n_masked.sum() == len(task.test)
    swapped = predict(task.test, history=masked)
    assert not np.allclose(base, swapped)             # masking reaches the TFT
    assert np.allclose(predict(task.test), base)       # original history restored afterwards
