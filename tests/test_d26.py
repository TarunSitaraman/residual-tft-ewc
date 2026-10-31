"""Tests for decision D26: P2 folds, origin strata, the D26 runner, the summary and the night orchestrator.

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
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _write_wheeler(d: Path) -> None:
    rows = []
    for case, fade in (("case1", 0.004), ("case4", 0.012), ("case7", 0.03)):
        for c in range(2):
            for k in range(14):
                rows.append({"Battery_ID": case, "cell": f"{case[-1]}{'ab'[c]}", "cycle": 100 * k,
                             "check_time": pd.Timestamp("2001-01-01") + pd.Timedelta(days=9 * k),
                             "Capacity_Ah": 1.1 * (1 - fade * k * (1 + 0.1 * c)), "valid": True})
    t = pd.DataFrame(rows)
    t["SOH_Percent"] = t["Capacity_Ah"] / t.groupby("cell")["Capacity_Ah"].transform("first") * 100
    t.to_csv(d / "wheeler_checks.csv", index=False)


@pytest.fixture()
def runner(tmp_path, monkeypatch):
    _write_wheeler(tmp_path)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    import run_d26
    return run_d26.Runner("e1", "smoke")


def test_folds_and_strata(runner):
    import run_d26
    from src import forecasting as FC
    held = "case4"
    order6 = runner.fold_order(runner.orders[0], held)
    assert held not in order6 and len(order6) == len(runner.cases) - 1
    tasks = runner.training_tasks(order6)
    for t in tasks:
        sp = FC.assign_split(runner.s6[runner.s6["Battery_ID"] == t.name])
        assert len(t.train) == int(sp["split"].isin(["train", "test"]).sum())     # D18 P2 rule
        assert len(t.val) == int((sp["split"] == "val").sum()) and t.test.empty
        assert (t.train["Battery_ID"] == t.name).all()
    frame, direct, hplus = runner.held(held)
    assert len(frame) == int((runner.s6["Battery_ID"] == held).sum())               # every held-out sample is a test sample
    assert (hplus == direct["fade_rate_pp_per_cycle"].notna().to_numpy()).all()
    assert hplus.any() and (~hplus).any()                                           # both strata occur
    # Scores: extrapolation (zero residual) has the same error as the residual target itself; H+/H- split it.
    sc = run_d26.scores(frame, direct, np.zeros(len(frame)), hplus)
    y = frame[FC.TARGET].to_numpy(float)
    assert sc["rmse"] == pytest.approx(np.sqrt(np.mean(y ** 2)))
    assert sc["rmse_hminus"] == pytest.approx(np.sqrt(np.mean(y[~hplus] ** 2)))


def test_d26_runner_smoke_parts_collect_and_summary(tmp_path):
    _write_wheeler(tmp_path)
    env = {**os.environ, "DATA_ROOT": str(tmp_path), "PYTHONPATH": str(ROOT), "MPLBACKEND": "Agg"}
    script = str(ROOT / "scripts" / "run_d26.py")
    bad = subprocess.run([sys.executable, script, "--exp", "e1", "--part", "main_f5", "--profile", "smoke"],
                         cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert bad.returncode != 0 and "only 3 test cases" in (bad.stdout + bad.stderr)
    for part in ("main_f0", "replicate_f0", "main_f1"):
        subprocess.run([sys.executable, script, "--exp", "e1", "--part", part, "--profile", "smoke"],
                       cwd=ROOT, env=env, check=True, capture_output=True, timeout=900)
    res = subprocess.run([sys.executable, script, "--exp", "e1", "--profile", "smoke", "--collect-only"],
                         cwd=ROOT, env=env, check=True, capture_output=True, text=True, timeout=900)
    out = tmp_path / "cache" / "d26" / "forecasting_d26_e1_smoke.json"
    data = json.loads(out.read_text())
    assert data["parts_run"] == ["main_f0", "main_f1", "replicate_f0"]
    assert data["parts_not_run"] == ["main_f2", "replicate_f1", "replicate_f2"]
    assert data["folds_in_tests"] == ["case1"]                  # only the fold both seed sets finished
    assert all(r["folds"] == ["case1"] for r in data["runs"])    # main's extra fold (case4) is left out of every run
    assert {r["method"] for r in data["runs"]} == {"tft_residual_ewc", "tft_residual_ewc_shrunk", "tft_residual_joint",
                                                   "linear_extrapolation_conformal"}
    assert {b["method"] for b in data["baselines"]} == {"persistence", "linear_extrapolation", "ridge_shared",
                                                        "gbm_shared"}
    shr = [r for r in data["runs"] if r["method"] == "tft_residual_ewc_shrunk"]
    assert all(0 <= v <= 1 for r in shr for a in r["alpha_per_fold"].values() for v in a.values())
    assert all(0 <= r["cov90"] <= 1 and "is90" in r for r in shr)
    assert {v["metric"] for v in data["verdicts"] if v["family"] == "primary"} == {"p2_error_hplus", "p2_error", "is90"}
    assert set(data["coverage_check"]) == {"threshold", "mean_cov90", "adequate"}
    assert data["strata_counts"]["case1"]["hplus"] + data["strata_counts"]["case1"]["hminus"] == \
        data["strata_counts"]["case1"]["n"]
    assert not (ROOT / "results" / "forecasting_d26_e1_smoke.json").exists()   # smoke output stays outside Git
    assert "not run" in res.stdout
    summ = subprocess.run([sys.executable, str(ROOT / "scripts" / "summarise_d26.py"), "--results", str(out.parent)],
                          cwd=ROOT, env=env, check=True, capture_output=True, timeout=300)
    text = summ.stdout.decode("utf-8")
    assert "Residual TFT + EWC, shrunk" in text and "parts not run" in text and "Gradient boosting" in text


def test_night_plan_and_scheduler_settings():
    import run_d25_night
    import run_d26
    import run_d26_night as night
    assert len(night.PLAN) == 2 * len(run_d26.PARTS) == 28 and night.DEPENDS == {}
    night.configure()
    try:
        assert run_d25_night.RUNNER.name == "run_d26.py" and run_d25_night.DATA_FILES == ("wheeler_checks.csv",)
        assert run_d25_night.LOGS == night.LOGS and run_d25_night.DECISION == "D26"
    finally:                                                          # restore the D25 defaults for other tests
        import importlib
        importlib.reload(run_d25_night)


def test_night_dry_run_preflight(tmp_path):
    _write_wheeler(tmp_path)                                          # no pool tables: D26 does not need them
    env = {**os.environ, "DATA_ROOT": str(tmp_path), "PYTHONPATH": str(ROOT)}
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_d26_night.py"), "--dry-run", "--allow-dirty",
                          "--allow-unpinned"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "pre-flight passed" in res.stdout and "e2 main_f0" in res.stdout and "e1 replicate_f6" in res.stdout
