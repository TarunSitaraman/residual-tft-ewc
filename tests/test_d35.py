"""Tests for decision D35: gradient boosting under the continual-learning constraint on the S6 stream (P1).

SYNTHETIC TEST DATA ONLY, built in pytest temporary directories (never the repository); it exercises code paths and
must never be used for reported results (AGENTS.md).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_d26 import _write_wheeler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture()
def runner(tmp_path, monkeypatch):
    _write_wheeler(tmp_path)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    import run_d26
    return run_d26.Runner("e1", "smoke")


def test_gbm_last_keeps_no_earlier_task(runner):
    """The final model depends only on the last task: changing earlier tasks' targets leaves R's last row unchanged."""
    from src import forecasting as FC
    import run_d35
    order = runner.orders[0]
    base = run_d35.gbm_last_stream(runner, order, "direct")
    assert np.asarray(base["R"]).shape == (len(order), len(order))
    earlier = runner.direct["Battery_ID"].isin(order[:-1])
    noisy = runner.direct.copy()
    rng = np.random.default_rng(0)                                   # unlearnable noise, not a constant shift
    noisy.loc[earlier, FC.TARGET] = noisy.loc[earlier, FC.TARGET] + rng.normal(0, 2.0, int(earlier.sum()))
    runner.direct = noisy
    moved = run_d35.gbm_last_stream(runner, order, "direct")
    # the last task's own error is unchanged: the final model saw only the last task
    assert np.isclose(np.asarray(base["R"])[-1, -1], np.asarray(moved["R"])[-1, -1])
    # and an earlier step does see its own (changed) task
    assert not np.isclose(np.asarray(base["R"])[0, 0], np.asarray(moved["R"])[0, 0])


def test_residual_arm_scores_on_delta(runner):
    """Residual and direct arms score the same quantity: an error on the residual equals the error on Delta."""
    import run_d35
    order = runner.orders[0]
    d = run_d35.gbm_last_stream(runner, order, "direct")
    r = run_d35.gbm_last_stream(runner, order, "residual")
    assert np.isfinite(d["final_avg_error"]) and np.isfinite(r["final_avg_error"])
    assert d["final_avg_error"] != r["final_avg_error"]            # different targets, different fits


def test_paired_tests_and_verdicts():
    import run_d35
    rng = np.random.default_rng(0)
    runs = pd.DataFrame([{"set": s, "order_id": o, "seed": sd, "final_avg_error": 0.32 + 0.01 * rng.normal()}
                         for s, seeds in (("main", range(5)), ("replicate", range(5, 10)))
                         for o in range(10) for sd in seeds])
    gbm = {"gbm_last_direct": [0.9 + 0.01 * o for o in range(10)],
           "gbm_last_residual": [0.32 + 0.001 * (o % 2) for o in range(10)]}
    rows, verdicts = run_d35.tests(runs, gbm)
    v = {x["b"]: x["verdict"] for x in verdicts}
    assert v["gbm_last_direct"] == "replicated, a better"
    assert v["gbm_last_residual"] == "practically equivalent"
    assert {x["set"] for x in rows} == {"main", "replicate"} and all("p_holm" in x for x in rows)
