"""Tests for the continual-learning strategies (synthetic test data only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src import continual_learning as cl
from src import data_pipeline as dp
from src.protocols import Task

GROUPS = ("g1", "w8", "v4")  # heterogeneous: two capacity classes
FAST = {"epochs": 30, "patience": 10, "batch_size": 64, "seed": 0}


def _tasks(ml_train: pd.DataFrame, groups=GROUPS) -> list[Task]:
    tasks = []
    for g in groups:
        s = dp.temporal_split(ml_train[ml_train[dp.GROUP_COL] == g])
        tasks.append(Task(g, *(s[s["split"] == p].reset_index(drop=True) for p in ("train", "val", "test"))))
    return tasks


@pytest.fixture(scope="module")
def tasks(ml_train):
    return _tasks(ml_train)


@pytest.fixture(scope="module")
def features(ml_train):
    return dp.model_feature_columns(ml_train)


def _rmse(strategy, task: Task) -> float:
    err = strategy.predict(task.test) - task.test["SOH_Percent"].to_numpy()
    return float(np.sqrt(np.mean(err ** 2)))


@pytest.mark.parametrize("name", sorted(cl.STRATEGIES))
def test_strategy_trains_on_stream(name, tasks, features):
    s = cl.make_strategy(name, features, **FAST)
    with pytest.raises(RuntimeError):
        s.predict(tasks[0].test)
    for t in tasks:
        s.fit_task(t)
        for u in tasks:
            p = s.predict(u.test)
            assert p.shape == (len(u.test),)
            assert np.isfinite(p).all()
    assert s.tasks_seen_ == list(GROUPS)
    assert len(s.training_time_) == len(tasks)
    assert all(t > 0 for t in s.training_time_)


def test_unknown_strategy(features):
    with pytest.raises(ValueError):
        cl.make_strategy("nope", features)


def test_determinism(tasks, features):
    preds = []
    for _ in range(2):
        s = cl.make_strategy("ewc", features, fisher_samples=50, **FAST)
        for t in tasks[:2]:
            s.fit_task(t)
        preds.append(s.predict(tasks[0].test))
    np.testing.assert_array_equal(preds[0], preds[1])
    other = cl.make_strategy("ewc", features, fisher_samples=50, **{**FAST, "seed": 1})
    for t in tasks[:2]:
        other.fit_task(t)
    assert not np.array_equal(preds[0], other.predict(tasks[0].test))


def test_preprocessor_frozen_after_first_task(tasks, features):
    s = cl.make_strategy("finetune", features, **FAST).fit_task(tasks[0])
    mean = s.preprocessor.mean_.copy()
    scale = (s.target_mean_, s.target_scale_)
    s.fit_task(tasks[1])
    np.testing.assert_array_equal(mean, s.preprocessor.mean_)
    assert scale == (s.target_mean_, s.target_scale_)
    np.testing.assert_allclose(mean, tasks[0].train[features].mean().to_numpy(), rtol=1e-10)


def test_replay_buffer(tasks, features):
    s = cl.make_strategy("replay", features, buffer_per_task=50, **FAST)
    assert s.buffer.empty
    key = [dp.GROUP_COL, "Cycle_Date"]
    for i, t in enumerate(tasks):
        # Before training task i, the buffer holds only tasks 0..i-1.
        assert len(s.buffer) == 50 * i
        if i:
            assert set(s.buffer[dp.GROUP_COL]) == set(GROUPS[:i])
        s.fit_task(t)
    buf = s.buffer
    assert len(buf) == 50 * len(tasks)
    train_keys = pd.concat([t.train[key] for t in tasks]).apply(tuple, axis=1)
    assert buf[key].apply(tuple, axis=1).isin(set(train_keys)).all()
    assert not buf.duplicated(key).any()
    # Capped at the task's training size.
    small = cl.make_strategy("replay", features, buffer_per_task=10_000, **FAST).fit_task(tasks[0])
    assert len(small.buffer) == len(tasks[0].train)


def test_ewc_penalty(tasks, features):
    s = cl.make_strategy("ewc", features, fisher_samples=50, **FAST)
    assert float(s.penalty().detach()) == 0.0
    s.fit_task(tasks[0])
    assert len(s.fishers_) == 1
    assert float(s.penalty().detach()) == pytest.approx(0.0, abs=1e-12)  # at the stored optimum
    assert all((f >= 0).all() for f in s.fishers_[0].values())
    assert sum(float(f.sum()) for f in s.fishers_[0].values()) > 0
    with torch.no_grad():
        for p in s.model.parameters():
            p.add_(0.05)
    assert float(s.penalty().detach()) > 0
    s.fit_task(tasks[1])
    assert len(s.fishers_) == len(s.optima_) == 2


def test_lwf_teacher_frozen(tasks, features):
    s = cl.make_strategy("lwf", features, **FAST).fit_task(tasks[0])
    assert s.teacher_ is None
    after_first = {n: p.detach().clone() for n, p in s.model.named_parameters()}

    original_train = s._train
    snapshots = []

    def spy(*args, **kwargs):
        snapshots.append({n: p.detach().clone() for n, p in s.teacher_.named_parameters()})
        return original_train(*args, **kwargs)

    s._train = spy
    s.fit_task(tasks[1])
    teacher = dict(s.teacher_.named_parameters())
    for n, p in after_first.items():
        torch.testing.assert_close(snapshots[0][n], p)
        torch.testing.assert_close(teacher[n].detach(), p)
        assert not teacher[n].requires_grad
    assert any(not torch.equal(p, after_first[n]) for n, p in s.model.named_parameters())


def test_joint_fit_all_and_sanity_vs_finetune(tasks, features):
    joint = cl.make_strategy("joint", features, **FAST).fit_all(tasks)
    assert joint.tasks_seen_ == list(GROUPS)
    ft = cl.make_strategy("finetune", features, **FAST)
    for t in tasks:
        ft.fit_task(t)
    j, f = _rmse(joint, tasks[0]), _rmse(ft, tasks[0])
    assert np.isfinite(j) and np.isfinite(f)
    # Loose sanity check: joint training should not forget the first task more than
    # sequential fine-tuning does (margin allows for training noise).
    assert j <= f * 1.1 + 0.25
