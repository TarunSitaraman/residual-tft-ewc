"""Tests for decision D32: gradient boosting under the continual-learning constraint (P2).

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


def test_constrained_arms_see_only_their_cases(runner):
    from src import forecasting as FC
    import run_d32
    case = runner.cases[0]
    fit, test, cols = run_d32.fit_frame(runner, case)
    order6 = runner.fold_order(runner.orders[0], case)
    warm = run_d32.gbm_warm(fit, order6, cols)
    assert warm.n_iter_ == run_d32.STAGE_ITER * len(order6)
    # GBM-LAST ignores every other case: changing their targets leaves it unchanged
    other = fit["Battery_ID"] != order6[-1]
    noisy = fit.copy()
    noisy.loc[other, FC.TARGET] = noisy.loc[other, FC.TARGET] + 5.0
    a = run_d32.gbm_on(fit[~other], cols).predict(test[cols])
    b = run_d32.gbm_on(noisy[~other], cols).predict(test[cols])
    assert np.array_equal(a, b)
    # GBM-WARM depends on the order of the cases (it is a sequential learner)
    rev = run_d32.gbm_warm(fit, tuple(reversed(order6)), cols)
    assert not np.allclose(warm.predict(test[cols]), rev.predict(test[cols]))


def test_paired_tests_and_verdicts():
    import run_d32
    rng = np.random.default_rng(0)
    runs = pd.DataFrame([{"set": s, "order_id": o, "seed": sd, "method": m,
                          "p2_error": 0.5 + 0.01 * rng.normal(), "p2_error_hplus": 0.4 + 0.01 * rng.normal()}
                         for s, seeds in (("main", range(5)), ("replicate", range(5, 10)))
                         for o in range(10) for sd in seeds for m in (run_d32.SHR, run_d32.RAW)])
    gbm = pd.DataFrame([{"method": arm, "order_id": o, "p2_error": base + 0.001 * o, "p2_error_hplus": base}
                        for arm, base in ((run_d32.WARM, 0.8), (run_d32.LAST, 0.51)) for o in range(10)])
    rows, verdicts = run_d32.tests(runs, gbm)
    v = {(x["a"], x["b"], x["metric"]): x["verdict"] for x in verdicts}
    assert v[(run_d32.SHR, run_d32.WARM, "p2_error")] == "replicated, a better"
    assert v[(run_d32.SHR, run_d32.LAST, "p2_error")] == "practically equivalent"
    assert {r["family"] for r in rows} == {"primary", "secondary"} and len(rows) == 12
