"""Tests for reuse-category evaluation (E10) on hand-made and synthetic data."""

import numpy as np
import pandas as pd
import pytest

from src import data_pipeline as dp
from src import evaluation as ev
from src import experiments as X
from src import protocols as P
from src import reuse as RU


def _frame(soh, rpt, battery="g1"):
    return pd.DataFrame({dp.GROUP_COL: battery, "RPT": rpt, "SOH_Percent": soh})


def test_interval_samples_average_predictions_per_rpt():
    df = _frame([98.0, 98.0, 93.0, 93.0, 93.0], ["R1", "R1", "R2", "R2", "R2"])
    s = RU.samples(df, [97.0, 99.0, 92.0, 94.0, 96.0], "interval")
    assert s["soh_true"].tolist() == [98.0, 93.0]
    assert s["soh_pred"].tolist() == [98.0, 94.0] and s["n_cycles"].tolist() == [2, 3]
    assert len(RU.samples(df, np.zeros(5), "cycle")) == 5


def test_interval_samples_reject_unaligned_labels():
    with pytest.raises(ValueError):
        RU.samples(_frame([98.0, 97.0], ["R1", "R1"]), [98.0, 97.0], "interval")
    with pytest.raises(ValueError):
        RU.samples(_frame([98.0], ["R1"]), [98.0, 1.0], "interval")


def test_class_metrics_values():
    # measured classes: C0, C1, C2, C3 ; predicted: C0, C2, C2, C3
    m = RU.class_metrics([97.0, 92.0, 85.0, 70.0], [96.0, 88.0, 85.0, 75.0], mae=1.0)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["n_classes_present"] == 4 and not m["single_class"]
    assert m["off_by_more_than_one"] == 0.0
    assert m["confusion"].sum() == 4 and m["confusion"][1, 2] == 1
    assert 0 < m["kappa_linear"] < 1
    # boundary share: 97 is within 1 pp of 95? no; 92 no; 85 no; 70 no -> 0
    assert m["boundary_share"] == 0.0


def test_class_metrics_degenerate_single_class():
    m = RU.class_metrics([99.0, 98.0], [99.5, 97.0])
    assert m["single_class"] and m["accuracy"] == 1.0 and m["macro_f1"] == 1.0
    assert np.isnan(m["kappa_linear"]) and np.isnan(m["boundary_share"])


def test_boundary_share_uses_mae():
    m = RU.class_metrics([95.4, 90.2, 99.0], [95.4, 90.2, 99.0], mae=0.5)
    assert m["boundary_share"] == pytest.approx(2 / 3)


def test_evaluate_groups_and_pooled(ml_train):
    test = pd.concat([P.make_task(ml_train, g).test for g in ["g1", "w8"]], ignore_index=True)
    res = RU.evaluate(test, test["SOH_Percent"].to_numpy(), "interval")
    assert res[dp.GROUP_COL].tolist() == ["g1", "w8", "ALL"]
    assert (res["accuracy"] == 1.0).all() and (res["soh_mae"] == 0.0).all()


def test_baselines(ml_train):
    task = P.make_task(ml_train, "w8")
    persistence = RU.baseline_predictions(task.train, task.test, "persistence")
    last = task.train.sort_values(["Cycle_Date", "Cycle_Index"])["SOH_Percent"].iloc[-1]
    assert np.allclose(persistence, last)
    majority = RU.baseline_predictions(task.train, task.test, "majority")
    cls = ev.reuse_class(majority)
    assert len(set(cls)) == 1
    train_cls = pd.Series(ev.reuse_class(task.train["SOH_Percent"].to_numpy()))
    assert cls[0] == train_cls.value_counts().sort_index().idxmax()
    with pytest.raises(ValueError):
        RU.baseline_predictions(task.train, task.test, "oracle")


def test_category_matrix_and_forgetting_sign(ml_train):
    tasks = P.make_task_stream(ml_train, ["g1", "w8"])
    frames = {t.name: t.test for t in tasks}
    truth = {t.name: t.test["SOH_Percent"].to_numpy() for t in tasks}
    # Perfect on every task at every step, except g1 is badly predicted after step 1.
    preds = {(0, "g1"): truth["g1"], (0, "w8"): truth["w8"], (1, "w8"): truth["w8"], (1, "g1"): truth["g1"] - 30}
    M = RU.category_matrix(preds, frames, ["g1", "w8"], "accuracy")
    assert M.iloc[0, 0] == 0.0 and M.iloc[1, 0] > 0
    assert ev.cl_metrics(M)["forgetting"] > 0


def test_class_distribution_and_variants(ml_train):
    table = RU.class_distribution(ml_train)
    assert list(table.columns) == list(ev.REUSE_CLASS_NAMES)
    assert table.to_numpy().sum() == ml_train.groupby([dp.GROUP_COL, "RPT"]).ngroups
    assert RU.class_distribution(ml_train, RU.CUTPOINT_VARIANTS["S-a2 three classes"]).shape[1] == 3


def test_runner_records_reuse_metrics(ml_train):
    cfg = X.CLConfig(methods=("finetune",), n_orders=1, seeds=(0,), common_kwargs={"epochs": 3, "patience": 2})
    out = X.run_cl_experiment(ml_train, cfg)
    row = out["runs"].iloc[0]
    for col in X.REUSE_METRIC_COLS:
        assert np.isfinite(row[col])
    assert 0 <= row["reuse_final_accuracy"] <= 1
    conf = out["reuse_confusion"]["finetune"]
    assert conf.shape == (4, 4) and conf.sum() == sum(
        t.test.groupby("RPT").ngroups for t in P.make_task_stream(ml_train, list(dp.BATTERY_GROUPS)))
