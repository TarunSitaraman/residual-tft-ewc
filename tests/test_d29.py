"""Tests for decision D29: physics-structured Bayesian forecaster with a learned residual (P2).

SYNTHETIC TEST DATA ONLY, built in pytest temporary directories (never the repository); it exercises code paths and
must never be used for reported results (CLAUDE.md).
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
SCRIPT = str(ROOT / "scripts" / "run_d29.py")


def test_population_recovers_power_law_and_posterior_uses_past_only():
    import run_d29
    rows = []
    for k, a in enumerate((0.20, 0.25, 0.30)):
        for cyc in range(0, 1001, 100):
            rows.append({"cell": f"c{k}", "case": "x", "cycle": cyc, "dod": 0.7, "soh": 100 - a * (cyc * 0.7) ** 0.6})
    chk = pd.DataFrame(rows)
    pop = run_d29.fit_population(chk, "efc")
    assert pop["z"] == 0.6 and np.isclose(pop["mu"], 0.25) and pop["sigma2"] < 1e-20
    pop["sigma2"] = 1e-6                                         # a noiseless fit; any small noise for the update
    frame = pd.DataFrame({"cell": ["c1"], "origin_cycle": [500], "cycles_to_target": [200], "dod": [0.7]})
    mean, var = run_d29.phys_predict(frame, chk, pop)
    expected = -0.25 * ((700 * 0.7) ** 0.6 - (500 * 0.7) ** 0.6)
    assert np.isclose(mean[0], expected, rtol=1e-4) and var[0] > 0
    # a later check that contradicts the law must not change a forecast issued before it
    chk2 = chk.copy()
    chk2.loc[(chk2["cell"] == "c1") & (chk2["cycle"] == 600), "soh"] -= 5
    assert np.isclose(run_d29.phys_predict(frame, chk2, pop)[0][0], mean[0])


def test_d29_end_to_end(tmp_path, monkeypatch):
    _write_wheeler(tmp_path)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    import run_d27
    import run_d28
    import run_d29
    run = run_d29.D29("e1")
    folds = [run.fold(c) for c in run.cases]
    d27 = {"per_fold": [run_d27.D27("e1").fold(c) for c in run.cases], "folds": run.cases}
    assert run_d29.reproduction(folds, d27)["passed"]
    for f in folds:
        for arm in run_d29.ARMS:
            b = f["rules"][arm]["90"]
            assert 0 <= b["all"]["coverage"] <= 1 and b["n"] == f["n"]
            assert f["rules"][arm]["90"]["covered"] >= f["rules"][arm]["80"]["covered"]
    d28 = {"per_fold": [run_d28.D28("e1").fold(c) for c in run.cases], "folds": run.cases}
    d27["summary"] = {"gbm_loco": {"90": run_d27.pooled_metrics(d27["per_fold"], "gbm_loco", "90")}}
    d28["summary"] = {"refit_track": {"90": run_d27.pooled_metrics(d28["per_fold"], "refit_track", "90")}}
    p27, p28 = tmp_path / "d27.json", tmp_path / "d28.json"
    p27.write_text(json.dumps(d27))
    p28.write_text(json.dumps(d28))
    env = {**os.environ, "DATA_ROOT": str(tmp_path), "PYTHONPATH": str(ROOT)}
    out = tmp_path / "d29.json"
    res = subprocess.run([sys.executable, SCRIPT, "--exp", "e1", "--d27", str(p27), "--d28", str(p28), "--out", str(out),
                          "--allow-dirty", "--allow-unpinned"], cwd=ROOT, env=env, capture_output=True, text=True,
                         timeout=900)
    assert res.returncode == 0, res.stdout + res.stderr
    data = json.loads(out.read_text())
    assert data["reproduction_check"]["passed"] and set(data["summary"]) == set(run_d29.ARMS)
    d27["per_fold"][0]["rmse_gbm"] += 1e-6
    p27.write_text(json.dumps(d27))
    out2 = tmp_path / "d29_bad.json"
    res = subprocess.run([sys.executable, SCRIPT, "--exp", "e1", "--d27", str(p27), "--d28", str(p28), "--out", str(out2),
                          "--allow-dirty", "--allow-unpinned"], cwd=ROOT, env=env, capture_output=True, text=True,
                         timeout=900)
    assert res.returncode == 3 and "STOP" in res.stdout and not out2.exists()
