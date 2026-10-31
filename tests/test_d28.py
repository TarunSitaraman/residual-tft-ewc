"""Tests for decision D28: continual calibration (quantile tracking) and refit on an unseen condition (P2).

SYNTHETIC TEST DATA ONLY, built in pytest temporary directories (never the repository); it exercises code paths and
must never be used for reported results (AGENTS.md).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from test_d26 import _write_wheeler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SCRIPT = str(ROOT / "scripts" / "run_d28.py")


def _frame(n: int, miss_at: set[int]) -> tuple[pd.DataFrame, np.ndarray]:
    """One horizon, one sample per day, each maturing 2.5 days later; targets miss at ``miss_at``."""
    f = pd.DataFrame({"Cycle_Date": pd.date_range("2021-01-01", periods=n, freq="D"), "days_to_target": 2.5,
                      "horizon": 1, "target_delta": [5.0 if i in miss_at else 0.0 for i in range(n)]})
    return f.rename(columns={"target_delta": __import__("src.forecasting", fromlist=["TARGET"]).TARGET}), np.zeros(n)


def test_track_uses_only_matured_errors():
    f, centre = _frame(6, miss_at={0})
    q = __import__("run_d28").track(f, centre, {1: 1.0}, 0.9, c=0.5)
    # sample 0 matures on day 2.5: samples issued on days 0-2 keep q0; day 3 sees one miss: 1 + 0.5 * (1 - 0.1)
    assert np.allclose(q[:3], 1.0) and np.isclose(q[3], 1.45)
    # then each covered sample shrinks it by 0.5 * 0.1
    assert np.isclose(q[4], 1.45 - 0.05) and np.isclose(q[5], 1.45 - 0.10)
    q0 = __import__("run_d28").track(f, centre, {1: 1.0}, 0.9, c=0.0)
    assert np.allclose(q0, 1.0)                                        # c = 0 is the static interval


def test_d28_folds_reproduction_and_no_lookahead(tmp_path, monkeypatch):
    _write_wheeler(tmp_path)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    import run_d27
    import run_d28
    run = run_d28.D28("e1")
    case = run.cases[0]
    test = run.direct[run.direct["Battery_ID"] == case].reset_index(drop=True)
    fit_rows = run_d27.p2_fit_rows(run.direct, case)
    p_static = run_d27.gbm_predict(fit_rows, test, run.features)
    # the GBM-only refit path equals the D27 comparator on the same rows
    assert np.allclose(run_d28.gbm_fit_predict(fit_rows, test, run.features), p_static, atol=0, rtol=0)
    p_refit, _ = run_d28.refit_predictions(fit_rows, test, run.features, p_static)
    t, m = run_d28.times(test)
    first = t == t.min()
    assert not (m[first] <= t.min()).any() and np.array_equal(p_refit[first], p_static[first])   # nothing known yet

    folds = [run.fold(c) for c in run.cases]
    d27 = [run_d27.D27("e1").fold(c) for c in run.cases]
    ref = {"per_fold": d27, "folds": run.cases}
    assert run_d28.reproduction(folds, ref)["passed"]
    for f in folds:
        assert f["c"] in run_d28.C_GRID
        for arm in run_d28.ARMS:
            b = f["rules"][arm]["90"]
            assert 0 <= b["all"]["coverage"] <= 1 and b["n"] == f["n"]
    good = tmp_path / "d27_ok.json"
    good.write_text(json.dumps(ref))
    ref["per_fold"][0]["rmse_gbm"] += 1e-6
    assert not run_d28.reproduction(folds, ref)["passed"]

    env = {**os.environ, "DATA_ROOT": str(tmp_path), "PYTHONPATH": str(ROOT)}
    out = tmp_path / "d28.json"
    res = subprocess.run([sys.executable, SCRIPT, "--exp", "e1", "--d27", str(good), "--out", str(out), "--allow-dirty",
                          "--allow-unpinned"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert res.returncode == 0, res.stdout + res.stderr
    data = json.loads(out.read_text())
    assert data["reproduction_check"]["passed"] and set(data["summary"]) == set(run_d28.ARMS)
    bad = tmp_path / "d27_bad.json"
    bad.write_text(json.dumps(ref))
    out2 = tmp_path / "d28_bad.json"
    res = subprocess.run([sys.executable, SCRIPT, "--exp", "e1", "--d27", str(bad), "--out", str(out2), "--allow-dirty",
                          "--allow-unpinned"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert res.returncode == 3 and "STOP" in res.stdout and not out2.exists()
