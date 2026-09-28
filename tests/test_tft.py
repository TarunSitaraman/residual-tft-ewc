"""Tests for the TFT backbone, causal cycle windows and TFT strategies (synthetic test data)."""

import numpy as np
import pandas as pd
import pytest
import torch

from src import continual_learning as cl
from src import data_pipeline as dp
from src import experiments as X
from src import protocols as P
from src.models.baseline_dnn import MLPRegressor, TabularPreprocessor
from src.models.tft_continual import CycleWindows, TemporalFusionTransformer

FAST = {"epochs": 5, "patience": 3}


@pytest.fixture(scope="module")
def features(ml_train):
    return dp.model_feature_columns(ml_train)


def test_tft_forward_shape_and_interpretation(features):
    model = TemporalFusionTransformer(len(features), d_model=8, n_heads=2)
    x = torch.randn(5, 8, 2 * len(features) + 1)
    x[..., -1] = 0.0
    y = model(x)
    assert y.shape == (5,) and torch.isfinite(y).all()
    info = model.last_interpretation_
    assert torch.allclose(info["variable_weights"].sum(-1), torch.ones(5), atol=1e-5)
    assert torch.allclose(info["attention"].sum(-1), torch.ones(5), atol=1e-5)
    with pytest.raises(ValueError):
        model(torch.randn(2, 8, 3))


def test_tft_capacity_matches_mlp(features):
    n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    tft = n(TemporalFusionTransformer(len(features), d_model=8, n_heads=2))
    mlp = n(MLPRegressor(2 * len(features), (128, 64)))
    assert 0.8 < tft / mlp < 1.2


def test_attention_is_causal_and_ignores_padding(features):
    torch.manual_seed(0)
    model = TemporalFusionTransformer(len(features), d_model=8, n_heads=2, dropout=0.0).eval()
    x = torch.randn(1, 6, 2 * len(features) + 1)
    x[..., -1] = 0.0
    x[0, :2, -1] = 1.0  # first two steps padded
    model(x)
    attn = model.last_interpretation_["attention"][0]
    assert torch.all(attn[:2] == 0)  # padded steps receive no attention
    # Attention ignores padded steps; the LSTM still reads them, which is harmless
    # because CycleWindows always fills padded steps with zeros.


def test_windows_are_causal_and_label_free(ml_train, features):
    history = ml_train.drop(columns=["SOH_Percent"])
    windows = CycleWindows(history, features, window=4)
    assert "SOH_Percent" not in windows.history.columns
    pre = TabularPreprocessor(features).fit(ml_train)
    g = ml_train[ml_train[dp.GROUP_COL] == "g1"].sort_values(["Cycle_Date", "Cycle_Index"]).reset_index(drop=True)
    w = windows.build(g, pre)
    enc = pre.transform(g)
    f2 = 2 * len(features)
    # Last step is the row itself; previous steps are the preceding cycles.
    assert np.allclose(w[10, -1, :f2], enc[10]) and np.allclose(w[10, 0, :f2], enc[7])
    # First cycle: three padded steps.
    assert w[0, :3, f2].tolist() == [1.0, 1.0, 1.0] and w[0, 3, f2] == 0.0
    # Changing a *later* cycle's features never changes an earlier row's window.
    later = history.copy()
    idx = later.index[(later[dp.GROUP_COL] == "g1") & (later["Cycle_Index"] == g["Cycle_Index"].iloc[20])]
    later.loc[idx, features[0]] += 1000
    w_later = CycleWindows(later, features, window=4).build(g.iloc[:20], pre)
    assert np.allclose(w_later, w[:20])


def test_windows_reject_unknown_rows(ml_train, features):
    windows = CycleWindows(ml_train[ml_train[dp.GROUP_COL] == "g1"], features, window=4)
    pre = TabularPreprocessor(features).fit(ml_train)
    with pytest.raises(KeyError):
        windows.build(ml_train[ml_train[dp.GROUP_COL] == "w8"].head(3), pre)


def test_tft_strategy_drops_labels_from_history(ml_train, features):
    s = cl.make_strategy("tft_ewc", features, history=ml_train, **FAST)
    assert not set(cl.LABEL_COLUMNS) & set(s.windows.history.columns)


@pytest.mark.parametrize("name", ["tft_finetune", "tft_ewc"])
def test_tft_strategies_train_on_stream(ml_train, features, name):
    tasks = P.make_task_stream(ml_train, ["g1", "w8", "v4"])
    s = cl.make_strategy(name, features, history=ml_train, seed=0, **FAST)
    for t in tasks:
        s.fit_task(t)
    pred = s.predict(tasks[0].test)
    assert pred.shape == (len(tasks[0].test),) and np.isfinite(pred).all()
    info = s.interpret(tasks[0].test)
    assert np.isclose(info["variable_weights"].sum(), 1.0, atol=1e-4)
    assert len(info["attention_by_lag"]) == s.window


def test_tft_ewc_penalty_and_determinism(ml_train, features):
    tasks = P.make_task_stream(ml_train, ["g1", "w8"])
    a = cl.make_strategy("tft_ewc", features, history=ml_train, seed=2, **FAST).fit_task(tasks[0])
    assert float(a.penalty().detach()) == pytest.approx(0.0, abs=1e-12)
    with torch.no_grad():
        for p in a.model.parameters():
            p.add_(0.1)
    assert float(a.penalty().detach()) > 0
    b1 = cl.make_strategy("tft_finetune", features, history=ml_train, seed=2, **FAST).fit_task(tasks[0])
    b2 = cl.make_strategy("tft_finetune", features, history=ml_train, seed=2, **FAST).fit_task(tasks[0])
    assert np.array_equal(b1.predict(tasks[1].test), b2.predict(tasks[1].test))


def test_runner_with_tft_and_joint(ml_train):
    cfg = X.CLConfig(methods=("tft_ewc",), n_orders=1, seeds=(0,), common_kwargs=FAST, joint_backbones=("mlp", "tft"))
    out = X.run_cl_experiment(ml_train, cfg)
    assert out["runs"]["method"].tolist() == ["tft_ewc"]
    assert set(out["joint"]["method"]) == {"joint", "tft_joint"}
    tasks = [P.Task(t.name, t.train, t.val, t.test.iloc[0:0]) for t in P.make_task_stream(ml_train, ["g1", "w8"])]
    val = X.run_stream("tft_ewc", tasks, 0, eval_split="val",
                       history=pd.concat([ml_train], ignore_index=True), **FAST)
    assert val["R"].notna().all().all()
