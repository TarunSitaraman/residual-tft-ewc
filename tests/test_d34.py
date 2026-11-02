"""Tests for decision D34: condition sets, folds, the D34 runner (smoke profile) and the orchestrator.

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

TYPES = [1] * 4 + [2] * 10 + [3] * 3                    # 17 synthetic conditions: calendar, cyclic, WLTP


def _write_s7(d: Path) -> None:
    rows = []
    for i, t in enumerate(TYPES):
        temp = (0, 10, 25, 40)[i % 4]
        for c in range(2):
            for k in range(9):
                soh = 100 - (0.2 + 0.05 * i) * k * (1 + 0.1 * c)
                rows.append({"dataset": "S7", "cell": f"c{i:02d}_{c}", "condition": f"S7-k{i:02d}", "check_index": k,
                             "check_time": pd.Timestamp("2022-10-12") + pd.Timedelta(days=20 * k),
                             "Capacity_Ah": 2.9 * soh / 100, "check_temp_c": 25.0, "age_type": t, "age_temp": temp,
                             "age_soc": 50.0 if t == 1 else np.nan, "age_chg_rate": 0.5 if t == 2 else np.nan,
                             "age_dischg_rate": 1.0 if t == 2 else np.nan, "age_profile": {1: np.nan, 2: 2, 3: 3}[t],
                             "v_max_target_V": 4.2 if t == 2 else np.nan, "v_min_target_V": 2.5,
                             "Reference_Capacity_Ah": 2.9, "SOH_Percent": soh, "n_valid_checks": 9})
    pd.DataFrame(rows).to_csv(d / "pool_s7_checks.csv", index=False)


def test_assign_sets_proportional_stratified_and_deterministic():
    import run_d34
    conds = pd.DataFrame({"condition": [f"x{i}" for i in range(67)], "age_type": [1] * 16 + [2] * 39 + [3] * 12,
                          "age_temp": [(0, 10, 25, 40)[i % 4] for i in range(67)], "cells": 3})
    s = run_d34.assign_sets(conds)
    tun = s[s["set"] == "tuning"]
    assert len(tun) == 7 and tun["age_type"].value_counts().to_dict() == {1: 2, 2: 4, 3: 1}   # largest remainder
    assert (tun["fold"] == -1).all()
    test = s[s["set"] == "test"]
    assert test["fold"].value_counts().to_dict() == {k: 12 for k in range(5)}
    for _, g in test.groupby("age_type"):                    # stratified: each type spread evenly over the folds
        assert g["fold"].value_counts().max() - g["fold"].value_counts().reindex(range(5), fill_value=0).min() <= 1
    assert s.equals(run_d34.assign_sets(conds))              # fixed seed: the same assignment every time


@pytest.fixture()
def runner(tmp_path, monkeypatch):
    _write_s7(tmp_path)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    import run_d34
    return run_d34.Runner("e1", "smoke")


def test_folds_tasks_orders_and_strata(runner):
    import run_d26
    from src import forecasting as FC
    tun = set(runner.tuning)
    assert len(tun) == 7
    held_all = []
    for k in range(5):
        held, train = runner.fold_conditions(k)
        held_all += held
        assert not (set(held) | set(train)) & tun and not set(held) & set(train)
        assert len(held) + len(train) == 10
        assert set(runner.orders(k)[0]) == set(train) and not set(runner.tuning_orders(k)) & set(runner.orders(k))
    assert sorted(held_all) == sorted(set(runner.sets["condition"]) - tun)       # every test condition once
    c = runner.fold_conditions(0)[1][0]
    t = runner.task(c)
    sp = FC.assign_split(runner.res[runner.res["Battery_ID"] == c])
    assert len(t.train) == int(sp["split"].isin(["train", "test"]).sum()) and t.test.empty   # D18 P2 rule
    frame, direct, hp = runner.held(runner.fold_conditions(0)[0][0])
    assert (frame["Battery_ID"] == direct["Battery_ID"]).all() and hp.any() and (~hp).any()
    lin = direct["fade_rate_pp_per_day"].fillna(0) * direct["days_to_target"]
    assert np.allclose(direct["lin_forecast"], lin)                              # D21 per-day extrapolation
    sc = run_d26.scores(frame, direct, np.zeros(len(frame)), hp)                 # extrapolation = zero residual
    assert sc["rmse"] == pytest.approx(np.sqrt(np.mean(frame[FC.TARGET].to_numpy(float) ** 2)))


def test_runner_smoke_parts_and_collect(tmp_path):
    _write_s7(tmp_path)
    env = {**os.environ, "DATA_ROOT": str(tmp_path), "PYTHONPATH": str(ROOT), "MPLBACKEND": "Agg"}
    script = str(ROOT / "scripts" / "run_d34.py")
    for part in ("tune1_f0", "tune2_f0", "main_f0", "replicate_f0", "main_f1"):
        subprocess.run([sys.executable, script, "--exp", "e1", "--part", part, "--profile", "smoke"],
                       cwd=ROOT, env=env, check=True, capture_output=True, timeout=900)
    res = subprocess.run([sys.executable, script, "--exp", "e1", "--profile", "smoke", "--collect-only"],
                         cwd=ROOT, env=env, check=True, capture_output=True, text=True, timeout=900)
    out = tmp_path / "cache" / "d34" / "forecasting_d34_e1_smoke.json"
    data = json.loads(out.read_text())
    assert data["folds_in_tests"] == [0]                     # only the fold both seed sets finished
    assert "replicate_f1" in data["parts_not_run"] and "main_f1" not in data["parts_not_run"]
    v = pd.DataFrame(data["condition_values"])
    held = sorted(v["condition"].unique())
    assert len(held) == 2 and not set(held) & set(data["condition_sets"][i]["condition"] for i in range(17)
                                                     if data["condition_sets"][i]["set"] == "tuning")
    assert set(v["method"]) == {"tft_residual_ewc", "tft_residual_ewc_shrunk", "tft_residual_joint", "gbm_last",
                                "gbm_all", "linear_extrapolation", "persistence", "ridge_shared"}
    prim = [x for x in data["verdicts"] if x["family"] == "primary"]
    assert {(x["b"], x["metric"]) for x in prim} == {("gbm_last", "rmse"), ("linear_extrapolation", "rmse_hplus"),
                                                     ("gbm_all", "rmse")}
    t = data["tuning"]["0"]
    assert set(t) >= {"lr", "ewc_lambda", "stage1", "stage2"}
    assert 0 <= data["descriptive"]["coverage"]["main"]["90"]["pooled"] <= 1
    assert not (ROOT / "results" / "forecasting_d34_e1_smoke.json").exists()   # smoke output stays outside Git
    assert "not run" in res.stdout


def test_paper_profile_refuses_without_committed_sets(tmp_path, monkeypatch):
    _write_s7(tmp_path)
    env = {**os.environ, "DATA_ROOT": str(tmp_path), "PYTHONPATH": str(ROOT)}
    # Missing, or committed for the real S7 table (other condition names): either way the paper run must refuse.
    res =subprocess.run([sys.executable, str(ROOT / "scripts" / "run_d34.py"), "--exp", "e1", "--part", "tune1_f0"],
                         cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert res.returncode != 0 and "d34_condition_sets.csv" in res.stdout + res.stderr


def test_night_plan_dependencies_and_dry_run(tmp_path):
    import run_d25_night
    import run_d34
    import run_d34_night as night
    assert len(night.PLAN) == 2 * len(run_d34.PARTS) == 40
    assert run_d34.DEPENDS["main_f3"] == "tune2_f3" and run_d34.DEPENDS["tune2_f3"] == "tune1_f3"
    night.configure()
    try:
        assert run_d25_night.RUNNER.name == "run_d34.py" and run_d25_night.DATA_FILES == ("pool_s7_checks.csv",)
    finally:
        import importlib
        importlib.reload(run_d25_night)
    _write_s7(tmp_path)
    env = {**os.environ, "DATA_ROOT": str(tmp_path), "PYTHONPATH": str(ROOT)}
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_d34_night.py"), "--dry-run", "--allow-dirty",
                          "--allow-unpinned"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    if (ROOT / "results" / "d34_condition_sets.csv").exists():
        assert res.returncode == 0 and "e2 tune1_f0" in res.stdout and "(after tune2_f4)" in res.stdout
    else:
        assert res.returncode == 2 and "d34_condition_sets.csv missing" in res.stdout
