"""Tests for task, stream and reference-protocol construction (synthetic test data only)."""

from __future__ import annotations

import pandas as pd
import pytest

from src import data_pipeline as dp
from src import protocols as pr


def _assert_chronological(task: pr.Task) -> None:
    parts = [task.train, task.val, task.test]
    assert all(len(p) > 0 for p in parts)
    for earlier, later in zip(parts, parts[1:]):
        assert earlier["Cycle_Date"].max() <= later["Cycle_Date"].min()
    for p in parts:
        assert p["Cycle_Date"].is_monotonic_increasing
        assert "split" not in p


def _keys(frame: pd.DataFrame) -> set:
    return set(zip(frame[dp.GROUP_COL].astype(str), frame["Cycling_Test"], frame["Cycle_Index"]))


def test_make_task_is_chronological_and_disjoint(ml_train):
    for group in dp.BATTERY_GROUPS:
        task = pr.make_task(ml_train, group)
        assert task.name == group
        _assert_chronological(task)
        keys = [_keys(p) for p in (task.train, task.val, task.test)]
        assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
        n = (ml_train[dp.GROUP_COL] == group).sum()
        assert sum(len(k) for k in keys) == n
        assert set(task.train[dp.GROUP_COL]) == {group}


def test_make_task_matches_temporal_split(ml_train):
    split = dp.temporal_split(ml_train)
    task = pr.make_task(ml_train, "w9")
    expected = split[(split[dp.GROUP_COL] == "w9") & (split["split"] == "test")].drop(columns="split")
    pd.testing.assert_frame_equal(task.test, expected.reset_index(drop=True))


def test_make_task_errors(ml_train):
    with pytest.raises(ValueError, match="Unknown battery group"):
        pr.make_task(ml_train, "x1")
    tiny = ml_train[ml_train[dp.GROUP_COL] == "g1"].head(2)
    with pytest.raises(ValueError, match="empty"):
        pr.make_task(tiny, "g1")


def test_task_stream_follows_order(ml_train):
    order = ("w10", "g1", "v5")
    stream = pr.make_task_stream(ml_train, order)
    assert [t.name for t in stream] == list(order)
    with pytest.raises(ValueError, match="duplicate"):
        pr.make_task_stream(ml_train, ["g1", "g1"])
    with pytest.raises(ValueError, match="absent"):
        pr.make_task_stream(ml_train, ["g1", "zz"])
    with pytest.raises(ValueError):
        pr.make_task_stream(ml_train, [])


def test_sample_orders_deterministic_and_distinct():
    orders = pr.sample_orders(dp.BATTERY_GROUPS, 10, seed=0)
    assert orders == pr.sample_orders(dp.BATTERY_GROUPS, 10, seed=0)
    assert orders[0] == dp.BATTERY_GROUPS
    assert len(set(orders)) == 10
    assert all(sorted(o) == sorted(dp.BATTERY_GROUPS) for o in orders)
    assert pr.sample_orders(dp.BATTERY_GROUPS, 5, seed=0) == orders[:5]
    assert orders != pr.sample_orders(dp.BATTERY_GROUPS, 10, seed=1)
    assert len(set(pr.sample_orders(("a", "b", "c"), 6, seed=3))) == 6


def test_sample_orders_errors():
    with pytest.raises(ValueError, match="exceeds"):
        pr.sample_orders(("a", "b", "c"), 7, seed=0)
    with pytest.raises(ValueError, match="duplicates"):
        pr.sample_orders(("a", "a"), 1, seed=0)
    with pytest.raises(ValueError):
        pr.sample_orders(("a", "b"), 0, seed=0)


def test_leave_one_group_out(ml_train):
    for held_out in dp.BATTERY_GROUPS:
        task = pr.leave_one_group_out(ml_train, held_out)
        assert task.name == f"without_{held_out}"
        for part in (task.train, task.val):
            assert held_out not in set(part[dp.GROUP_COL])
            assert set(part[dp.GROUP_COL]) == set(dp.BATTERY_GROUPS) - {held_out}
        assert set(task.test[dp.GROUP_COL]) == {held_out}
        pd.testing.assert_frame_equal(task.test, pr.make_task(ml_train, held_out).test)
        held_rows = ml_train[ml_train[dp.GROUP_COL] == held_out]
        early = _keys(held_rows) - _keys(task.test)
        assert not early & (_keys(task.train) | _keys(task.val))
        assert not _keys(task.train) & _keys(task.val)


def test_leave_one_group_out_errors(ml_train):
    with pytest.raises(ValueError, match="Unknown"):
        pr.leave_one_group_out(ml_train, "x1")
    only_one = ml_train[ml_train[dp.GROUP_COL] == "g1"]
    with pytest.raises(ValueError, match="besides"):
        pr.leave_one_group_out(only_one, "g1")


def test_rpt_boundary_split(ml_train):
    split = pr.rpt_boundary_split(ml_train)
    assert len(split) == len(ml_train)
    assert set(split["split"]) == set(pr.SPLITS)
    check = pr.check_rpt_boundary(split)
    assert check["ok"].all()
    assert (check["rpts_spanning_splits"] == 0).all()
    assert (check[["train_rpts", "val_rpts", "test_rpts"]] >= 1).all().all()
    per_rpt = split.groupby([dp.GROUP_COL, "RPT"])["split"].nunique()
    assert (per_rpt == 1).all()
    for _, g in split.groupby(dp.GROUP_COL):
        assert g["Cycle_Date"].is_monotonic_increasing
        codes = g["split"].map({"train": 0, "val": 1, "test": 2})
        assert codes.is_monotonic_increasing
    # Synthetic data: RPT intervals of 40,40,40,40,40,30 rows -> 4/1/1 intervals is closest to 70/15/15.
    assert (check["train_rpts"] == 4).all() and (check["val_rpts"] == 1).all()


def test_check_rpt_boundary_flags_shared_rpt(ml_train):
    split = pr.rpt_boundary_split(ml_train)
    g1 = split.index[split[dp.GROUP_COL] == "g1"]
    first_val = split.loc[g1][split.loc[g1, "split"] == "val"].index[0]
    bad = split.copy()
    bad.loc[first_val - 1, "split"] = "val"  # last train row takes the val label but keeps its RPT
    bad.loc[first_val, "split"] = "train"  # and the first val row now precedes it in train
    check = pr.check_rpt_boundary(bad)
    assert not check.loc["g1", "ok"]
    assert check.loc["g1", "rpts_spanning_splits"] >= 1
    assert check.drop(index="g1")["ok"].all()


def test_rpt_boundary_split_errors(ml_train):
    g1 = ml_train[ml_train[dp.GROUP_COL] == "g1"]
    two_rpts = g1[g1["RPT"].isin(["RPT_1", "RPT_2"])]
    with pytest.raises(ValueError, match="at least 3"):
        pr.rpt_boundary_split(two_rpts)
    with pytest.raises(ValueError, match="recur"):
        recurring = g1.copy()
        recurring.loc[recurring.index[-1], "RPT"] = "RPT_1"
        pr.rpt_boundary_split(recurring)
    with pytest.raises(ValueError, match="missing"):
        pr.rpt_boundary_split(g1.assign(RPT=None))
    with pytest.raises(ValueError):
        pr.rpt_boundary_split(ml_train, train_frac=0.9, val_frac=0.2)


def test_relabel_date_only_start_changes_nothing(ml_train, health):
    out = pr.relabel_date_only(ml_train, health, "start")
    assert len(out) == len(ml_train)
    assert not out["label_changed"].any()
    base = ml_train.reset_index(drop=True)
    pd.testing.assert_frame_equal(out.drop(columns="label_changed"), base, check_dtype=False)


def test_relabel_date_only_end_changes_same_day_cycles(ml_train, health):
    base = ml_train.reset_index(drop=True)
    same_day = dp.same_day_as_date_only_rpt(base, health)
    first_rpt_day = base.groupby(dp.GROUP_COL)["Cycle_Date"].transform("min").dt.normalize() == base["Cycle_Date"].dt.normalize()
    dropped_expected = same_day & first_rpt_day  # same-day cycles of a battery's first RPT lose their label
    assert same_day.sum() > dropped_expected.sum() > 0

    out = pr.relabel_date_only(base, health, "end")
    assert len(out) == len(base) - dropped_expected.sum()
    assert out["label_changed"].sum() == (same_day & ~dropped_expected).sum()

    kept = base[~dropped_expected].reset_index(drop=True)
    assert out["label_changed"].equals(same_day[~dropped_expected].reset_index(drop=True))
    features = dp.model_feature_columns(base)
    pd.testing.assert_frame_equal(out[features], kept[features])
    assert dp.validate_soh_formula(out)["ok"]
    changed = out[out["label_changed"]]
    # Moved back to the previous RPT, 40 days earlier in the synthetic data.
    assert (changed["RPT_Number"] == kept.loc[changed.index, "RPT_Number"] - 1).all()
    assert (changed["Days_Since_RPT"] == kept.loc[changed.index, "Days_Since_RPT"] + 40).all()
    summary, _ = dp.validate_rpt_alignment(out, health, date_only_rpt="end")
    assert summary["ok"]


def test_relabel_date_only_errors(ml_train, health):
    with pytest.raises(ValueError):
        pr.relabel_date_only(ml_train, health, "noon")
    with pytest.raises(ValueError, match="missing"):
        pr.relabel_date_only(ml_train, health.drop(columns="Reference_Capacity_Ah"), "start")
