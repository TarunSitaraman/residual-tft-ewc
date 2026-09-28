"""Tests for the shared MLP backbone and preprocessor (synthetic test data only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src import data_pipeline as dp
from src.models.baseline_dnn import MLPRegressor, TabularPreprocessor, set_determinism


def _group_split(ml_train: pd.DataFrame, group: str) -> pd.DataFrame:
    return dp.temporal_split(ml_train[ml_train[dp.GROUP_COL] == group])


def test_preprocessor_uses_training_statistics_only(ml_train):
    features = dp.model_feature_columns(ml_train)
    split = _group_split(ml_train, "g1")
    train, test = split[split["split"] == "train"], split[split["split"] == "test"]
    other = ml_train[ml_train[dp.GROUP_COL] == "w8"]

    pre = TabularPreprocessor(features).fit(train)
    assert pre.features == features
    np.testing.assert_allclose(pre.mean_, train[features].mean().to_numpy(), rtol=1e-10)
    np.testing.assert_allclose(pre.median_, train[features].median().to_numpy(), rtol=1e-10)

    # Transforming other data uses the frozen training statistics.
    before = (pre.mean_.copy(), pre.std_.copy(), pre.median_.copy())
    out = pre.transform(other)
    for a, b in zip(before, (pre.mean_, pre.std_, pre.median_)):
        np.testing.assert_array_equal(a, b)
    j = features.index("Voltage_Mean")
    expected = (other["Voltage_Mean"].to_numpy() - pre.mean_[j]) / pre.std_[j]
    np.testing.assert_allclose(out[:, j], expected, rtol=1e-5, atol=1e-5)

    # Fixed width and float32 regardless of which data are transformed.
    for df in (train, test, other):
        z = pre.transform(df)
        assert z.shape == (len(df), 2 * len(features))
        assert z.dtype == np.float32
        assert np.isfinite(z).all()


def test_preprocessor_missing_indicators_and_imputation(ml_train):
    features = dp.model_feature_columns(ml_train)
    temps = list(dp.TEMPERATURE_COLUMNS)
    df = ml_train.copy()
    assert df[temps].isna().any().any(), "synthetic data should contain missing temperatures"
    pre = TabularPreprocessor(features).fit(df)
    z = pre.transform(df)
    d = len(features)
    for c in temps:
        j = features.index(c)
        miss = df[c].isna().to_numpy()
        np.testing.assert_array_equal(z[:, d + j], miss.astype(np.float32))
        imputed = (pre.median_[j] - pre.mean_[j]) / pre.std_[j]
        np.testing.assert_allclose(z[miss, j], imputed, rtol=1e-5)
    complete = [features.index(c) for c in features if not df[c].isna().any()]
    assert (z[:, [d + j for j in complete]] == 0).all()


def test_preprocessor_guards():
    df = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [1.0, np.nan, 3.0]})
    pre = TabularPreprocessor(["a", "b"])
    with pytest.raises(RuntimeError):
        pre.transform(df)
    z = pre.fit_transform(df)
    assert pre.std_[0] == 1.0  # zero-variance feature left unscaled
    np.testing.assert_allclose(z[:, 0], 0.0)
    with pytest.raises(KeyError):
        pre.transform(df[["a"]])


def test_mlp_shape_and_determinism():
    set_determinism(3)
    m1 = MLPRegressor(10)
    set_determinism(3)
    m2 = MLPRegressor(10)
    x = torch.randn(7, 10)
    m1.eval(), m2.eval()
    assert m1(x).shape == (7,)
    torch.testing.assert_close(m1(x), m2(x))
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.get_num_threads() == 1
