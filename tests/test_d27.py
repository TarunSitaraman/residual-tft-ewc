"""Tests for decision D27: leave-one-condition-out conformal intervals (P2).

SYNTHETIC TEST DATA ONLY, built in pytest temporary directories (never the repository); it exercises code paths and
must never be used for reported results (CLAUDE.md).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from test_d26 import _write_wheeler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SCRIPT = str(ROOT / "scripts" / "run_d27.py")


def _fake_d26(folds: list[dict], cases: list[str]) -> dict:
    """A D26-shaped reference built from D27's own fold results (the values the stop conditions compare)."""
    import run_d27
    linc = {"method": "linear_extrapolation_conformal"}
    for lv in ("90", "80"):
        m = run_d27.pooled_metrics(folds, "lin_val", lv)
        linc[f"is{lv}"], linc[f"cov{lv}"] = m["interval_score"], m["coverage_pooled"]
    return {"folds_in_tests": cases, "runs": [linc],
            "baselines_by_fold": {f["case"]: {"gbm_shared": {"rmse": f["rmse_gbm"]}} for f in folds}}


def test_d27_folds_calibration_and_stop_condition(tmp_path, monkeypatch):
    _write_wheeler(tmp_path)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    import run_d27
    run = run_d27.D27("e1")
    folds = [run.fold(c) for c in run.cases]
    for f in folds:
        n_other = int((run.direct["Battery_ID"] != f["case"]).sum())
        assert f["calibration_rows"]["gbm"] == f["calibration_rows"]["lin_all"] == n_other    # held-out case excluded
        assert 0 < f["calibration_rows"]["lin_val"] < n_other
        for rule in run_d27.RULES:
            b = f["rules"][rule]["90"]
            assert 0 <= b["all"]["coverage"] <= 1 and b["n"] == f["n"]
            assert b["hplus"]["n"] + b["hminus"]["n"] == f["n"]
            assert 0 <= b["decided_share"] <= 1
        # a wider level never covers less
        assert f["rules"]["gbm_loco"]["90"]["covered"] >= f["rules"]["gbm_loco"]["80"]["covered"]
    ref = _fake_d26(folds, run.cases)
    assert run_d27.reproduction(folds, ref)["passed"]
    ref["baselines_by_fold"][run.cases[0]]["gbm_shared"]["rmse"] += 1e-6
    assert not run_d27.reproduction(folds, ref)["passed"]

    # End to end: writes the results file when the reference reproduces, and nothing when it does not.
    env = {**os.environ, "DATA_ROOT": str(tmp_path), "PYTHONPATH": str(ROOT)}
    good = tmp_path / "d26_ok.json"
    good.write_text(json.dumps(_fake_d26(folds, run.cases)))
    out = tmp_path / "d27.json"
    res = subprocess.run([sys.executable, SCRIPT, "--exp", "e1", "--d26", str(good), "--out", str(out), "--allow-dirty",
                          "--allow-unpinned"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stdout + res.stderr
    data = json.loads(out.read_text())
    assert data["reproduction_check"]["passed"] and data["overrides"] == {"allow_dirty": True, "allow_unpinned": True}
    assert set(data["summary"]) == set(run_d27.RULES) and data["q2_test"]["n_gbm_better"] in range(len(run.cases) + 1)
    assert np.isclose(data["q1_coverage_gbm_loco_90"], data["summary"]["gbm_loco"]["90"]["coverage_pooled"])
    bad = tmp_path / "d26_bad.json"
    bad.write_text(json.dumps(ref))
    out2 = tmp_path / "d27_bad.json"
    res = subprocess.run([sys.executable, SCRIPT, "--exp", "e1", "--d26", str(bad), "--out", str(out2), "--allow-dirty",
                          "--allow-unpinned"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert res.returncode == 3 and "STOP" in res.stdout and not out2.exists()
