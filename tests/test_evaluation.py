"""Tests for src/evaluation.py on small hand-computed examples."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import evaluation as ev

ORDER = ["a", "b", "c"]

# Forgetting case. Rows: after step i (training a, b, c); columns: tasks a, b, c.
FORGET = {
    (0, "a"): 1.0, (0, "b"): 5.0, (0, "c"): 6.0,
    (1, "a"): 0.5, (1, "b"): 1.5, (1, "c"): 4.0,
    (2, "a"): 3.0, (2, "b"): 2.5, (2, "c"): 1.0,
}


def test_accuracy_matrix_layout():
    R = ev.accuracy_matrix(FORGET, ORDER)
    assert R.shape == (3, 3)
    assert R.index.name == "after_step" and list(R.index) == ORDER
    assert list(R.columns) == ORDER
    assert R.iloc[1, 0] == 0.5 and R.loc["c", "b"] == 2.5


def test_accuracy_matrix_upper_triangle_optional():
    lower = {k: v for k, v in FORGET.items() if ORDER.index(k[1]) <= k[0]}
    R = ev.accuracy_matrix(lower, ORDER)
    assert np.isnan(R.iloc[0, 1]) and np.isnan(R.iloc[0, 2]) and np.isnan(R.iloc[1, 2])
    m = ev.cl_metrics(R, reference_errors={"b": 6.0, "c": 5.0})
    assert np.isnan(m["forward_transfer"]) and m["forward_transfer_n"] == 0
    assert m["backward_transfer"] == pytest.approx(1.5)


@pytest.mark.parametrize("bad", [
    {**FORGET, (3, "a"): 1.0},          # step out of range
    {**FORGET, (0, "z"): 1.0},          # unknown task
    {**FORGET, (0, "a"): -1.0},         # negative error
    {**FORGET, (0, "a"): float("nan")},  # non-finite
    {k: v for k, v in FORGET.items() if k != (2, "a")},  # missing required entry
])
def test_accuracy_matrix_validation(bad):
    with pytest.raises(ValueError):
        ev.accuracy_matrix(bad, ORDER)


def test_accuracy_matrix_rejects_duplicate_order():
    with pytest.raises(ValueError):
        ev.accuracy_matrix({(0, "a"): 1.0, (1, "a"): 1.0}, ["a", "a"])


def test_cl_metrics_forgetting_case():
    R = ev.accuracy_matrix(FORGET, ORDER)
    m = ev.cl_metrics(R, reference_errors={"b": 6.0, "c": 5.0})
    assert m["n_tasks"] == 3
    assert m["final_avg_error"] == pytest.approx((3.0 + 2.5 + 1.0) / 3)
    assert m["plasticity"] == pytest.approx((1.0 + 1.5 + 1.0) / 3)
    # BWT: a: 3.0 - 1.0 = 2.0; b: 2.5 - 1.5 = 1.0
    assert m["backward_transfer"] == pytest.approx(1.5)
    assert m["backward_transfer_per_task"] == pytest.approx({"a": 2.0, "b": 1.0})
    # Forgetting: a: 3.0 - min(1.0, 0.5) = 2.5; b: 2.5 - 1.5 = 1.0
    assert m["forgetting"] == pytest.approx(1.75)
    assert m["forgetting_per_task"] == pytest.approx({"a": 2.5, "b": 1.0})
    # FWT: b: R[0,b] - 6 = -1; c: R[1,c] - 5 = -1 (negative = helpful)
    assert m["forward_transfer"] == pytest.approx(-1.0)
    assert m["forward_transfer_n"] == 2
    assert m["final_error_per_task"] == {"a": 3.0, "b": 2.5, "c": 1.0}
    assert m["diagonal_error_per_task"] == {"a": 1.0, "b": 1.5, "c": 1.0}
    assert m["backward_transfer"] > 0 and m["forgetting"] > 0


def test_cl_metrics_no_forgetting_case():
    errors = {(i, t): 1.0 for i in range(3) for t in ORDER[: i + 1]}
    m = ev.cl_metrics(ev.accuracy_matrix(errors, ORDER))
    assert m["backward_transfer"] == 0.0
    assert m["forgetting"] == 0.0
    assert m["final_avg_error"] == 1.0 and m["plasticity"] == 1.0
    assert np.isnan(m["forward_transfer"])


def test_cl_metrics_negative_backward_transfer():
    errors = {(0, "a"): 2.0, (1, "a"): 1.0, (1, "b"): 1.0}
    m = ev.cl_metrics(ev.accuracy_matrix(errors, ["a", "b"]))
    assert m["backward_transfer"] == pytest.approx(-1.0)
    assert m["forgetting"] == pytest.approx(-1.0)


def test_cl_metrics_single_task():
    m = ev.cl_metrics(ev.accuracy_matrix({(0, "a"): 0.7}, ["a"]), reference_errors={"a": 2.0})
    assert m["final_avg_error"] == 0.7 and m["plasticity"] == 0.7
    for k in ("backward_transfer", "forgetting", "forward_transfer"):
        assert np.isnan(m[k])


def test_task_error_uses_regression_metrics():
    assert ev.task_error([100.0, 98.0], [99.0, 99.0]) == pytest.approx(1.0)
    assert ev.task_error([100.0, 98.0], [99.0, 100.0], metric="MAE") == pytest.approx(1.5)


def test_bootstrap_ci_deterministic_and_contains_mean():
    x = np.random.default_rng(1).normal(2.0, 1.0, size=40)
    r1 = ev.bootstrap_ci(x, n_boot=2000, seed=3)
    r2 = ev.bootstrap_ci(x, n_boot=2000, seed=3)
    assert r1 == r2
    mean, lo, hi = r1
    assert mean == pytest.approx(x.mean())
    assert lo < mean < hi


def test_bootstrap_ci_degenerate():
    assert all(np.isnan(v) for v in ev.bootstrap_ci([]))
    assert all(np.isnan(v) for v in ev.bootstrap_ci([np.nan]))
    mean, lo, hi = ev.bootstrap_ci([np.nan, 3.0])
    assert mean == 3.0 and np.isnan(lo) and np.isnan(hi)
    assert ev.bootstrap_ci([2.0, 2.0, 2.0], n_boot=100) == (2.0, 2.0, 2.0)


def test_paired_comparison_identical():
    x = [1.0, 2.0, 3.0, 4.0]
    r = ev.paired_comparison(x, x)
    assert r["n"] == 4 and r["n_nonzero"] == 0
    assert r["mean_diff"] == 0.0 and r["ci_low"] == 0.0 and r["ci_high"] == 0.0
    assert r["p_value"] == 1.0 and r["rank_biserial"] == 0.0


def test_paired_comparison_known_values():
    a = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    b = a - np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r = ev.paired_comparison(a, b, n_boot=500)
    assert r["mean_diff"] == pytest.approx(3.0)
    assert r["wilcoxon_stat"] == 0.0
    assert r["p_value"] == pytest.approx(2 / 32)  # exact two-sided, all signs positive
    assert r["rank_biserial"] == pytest.approx(1.0)
    assert ev.paired_comparison(b, a, n_boot=500)["rank_biserial"] == pytest.approx(-1.0)


def test_paired_comparison_small_and_nan():
    r = ev.paired_comparison([1.0, np.nan], [0.0, 1.0])
    assert r["n"] == 1 and np.isnan(r["p_value"]) and np.isnan(r["rank_biserial"])
    with pytest.raises(ValueError):
        ev.paired_comparison([1.0, 2.0], [1.0])


def test_holm_correction_known_example():
    p = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005}
    # sorted: d 0.005*4=0.02, a 0.01*3=0.03, c 0.03*2=0.06, b max(0.04*1, 0.06)=0.06
    adj = ev.holm_correction(p)
    assert adj == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06, "d": 0.02})
    adj = ev.holm_correction({"x": 0.6, "y": 0.9, "z": np.nan})
    assert adj["x"] == pytest.approx(1.0) and adj["y"] == pytest.approx(1.0) and np.isnan(adj["z"])


def test_summarise_runs():
    runs = pd.DataFrame({
        "method": ["ft"] * 4 + ["ewc"] * 4,
        "order_id": [0, 0, 1, 1] * 2,
        "seed": [0, 1] * 4,
        "bwt": [1.0, 2.0, 3.0, np.nan, 0.5, 0.5, 0.5, 0.5],
    })
    out = ev.summarise_runs(runs, ["bwt"], ["method"], n_boot=500)
    assert list(out["method"]) == ["ewc", "ft"]
    ft = out.set_index("method").loc["ft"]
    assert ft["bwt_mean"] == pytest.approx(2.0) and ft["bwt_n"] == 3
    assert ft["bwt_ci_low"] <= 2.0 <= ft["bwt_ci_high"]
    ewc = out.set_index("method").loc["ewc"]
    assert ewc["bwt_ci_low"] == ewc["bwt_ci_high"] == 0.5
    with pytest.raises(KeyError):
        ev.summarise_runs(runs, ["missing"], ["method"])
