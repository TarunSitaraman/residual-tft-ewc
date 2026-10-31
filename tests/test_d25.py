"""Tests for decision D25: horizon-wise shrinkage, conformal intervals, the D25 runner and the night orchestrator.

SYNTHETIC TEST DATA ONLY, built in the tests or in pytest temporary directories (never the repository); it exercises
code paths and must never be used for reported results (AGENTS.md).
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import calibration as C
from src import forecasting as FC
from src import p3
from src import pool

ROOT = Path(__file__).resolve().parents[1]
FAST = {"epochs": 3, "patience": 2, "window": 2, "window_group": ("Battery_ID", "Cycling_Test")}


class _Fixed:
    """Stand-in strategy whose prediction is a fixed column of the frame."""

    def __init__(self, col):
        self.col = col

    def predict(self, df):
        return df[self.col].to_numpy(float)


def _frame(y, p, h):
    return pd.DataFrame({FC.TARGET: y, "p": p, "horizon": h})


# --- calibration ------------------------------------------------------------------------------------

def test_conformal_halfwidth_finite_sample_rule():
    s = np.arange(1, 20, dtype=float)                       # n = 19
    q, flag = C.conformal_halfwidth(s, 0.9)                  # ceil(20 * 0.9) = 18th smallest
    assert q == 18.0 and not flag
    q, flag = C.conformal_halfwidth(np.arange(1, 6, dtype=float), 0.9)   # ceil(6 * 0.9) = 6 > 5: largest, flagged
    assert q == 5.0 and flag
    assert math.isnan(C.conformal_halfwidth(np.array([]), 0.9)[0])


def test_interval_score_formula():
    m = C.interval_metrics(np.array([0.0, 3.0]), np.zeros(2), np.ones(2), 0.9)
    assert m["coverage"] == 0.5 and m["width"] == 2.0
    assert m["interval_score"] == pytest.approx(np.mean([2.0, 2.0 + 20.0 * 2.0]))


def test_shrinkage_is_clipped_least_squares_per_horizon():
    rng = np.random.default_rng(0)
    p = rng.normal(size=60)
    h = np.repeat([1, 2, 3], 20)
    y = np.where(h == 1, 0.5 * p, np.where(h == 2, -p, 3 * p))    # LS factors 0.5, -1 (clip 0), 3 (clip 1)
    cal = C.HorizonCalibrator((1, 2, 3, 4)).update(_Fixed("p"), _frame(y, p, h))
    assert cal.alpha_[1] == pytest.approx(0.5) and cal.alpha_[2] == 0.0 and cal.alpha_[3] == 1.0
    assert cal.alpha_[4] == 1.0                                      # no rows: keeps its start value
    f = _frame(y, p, h)
    np.testing.assert_allclose(cal.shrink(p, f), np.select([h == 1, h == 2], [0.5 * p, 0 * p], p))
    # The conformal residuals are those of the shrunk model; extrapolation's are the residual target itself.
    m = h == 1
    assert cal.q_model_[0.9][1] == C.conformal_halfwidth(y[m] - 0.5 * p[m], 0.9)[0]
    assert cal.q_lin_[0.9][1] == C.conformal_halfwidth(y[m], 0.9)[0]
    # A later task without rows for a horizon keeps that horizon's values.
    cal.update(_Fixed("p"), _frame(y[h == 3], p[h == 3], h[h == 3]))
    assert cal.alpha_[1] == pytest.approx(0.5)


def _s6_frame(cases=("case1", "case4", "case7")):
    fades = {"case1": 0.004, "case4": 0.012, "case7": 0.03}
    rows = []
    for c in cases:
        for cell in range(2):
            for k in range(12):
                rows.append({"dataset": "S6", "cell": f"{c}-{cell}", "condition": c, "cycle": 50 * (k + 1),
                             "check_time": pd.Timestamp("2000-01-01") + pd.Timedelta(days=7 * k),
                             "Capacity_Ah": 1.1 - fades[c] * (cell + 1) * k})
    s = p3.harmonised_samples(pool.add_soh(pd.DataFrame(rows)), "S6")
    s["lin_forecast"] = 0.8 * s[FC.TARGET].to_numpy(float)          # synthetic stand-in, known only in tests
    return p3.residual_target(s)


def test_raw_path_equals_the_uncalibrated_stream():
    frame = _s6_frame()
    order = ["case4", "case1", "case7"]
    feats = list(p3.SHARED_FEATURES)
    a = p3.build_strategy("tft_ewc", feats, 0, frame, {**FAST, "ewc_lambda": 10.0})
    b = p3.build_strategy("tft_ewc", feats, 0, frame, {**FAST, "ewc_lambda": 10.0})
    plain = p3.run_stream(a, FC.forecast_stream(frame, order), None)
    cal = C.run_stream_calibrated(b, FC.forecast_stream(frame, order), C.HorizonCalibrator((1, 2, 3)))
    for k, v in plain["metrics"].items():
        assert cal["raw"]["metrics"][k] == pytest.approx(v, abs=1e-12, nan_ok=True)
    assert len(cal["calibration"]) == 3
    assert set(cal["intervals"]) == {"model_90", "lin_90", "model_80", "lin_80"}
    assert 0.0 <= cal["intervals"]["model_90"]["coverage"] <= 1.0


# --- runner and orchestrator (smoke profile on synthetic tables outside the repository) ---------------------

def _write_tables(d: Path) -> None:
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

    def checks(dataset, conds, extra):
        out = []
        for i, cond in enumerate(conds):
            for c in range(3):
                for k in range(10):
                    out.append({"dataset": dataset, "cell": f"{dataset}-{i}-{c}", "condition": cond,
                                "check_time": pd.Timestamp("2002-01-01") + pd.Timedelta(days=11 * k),
                                "Capacity_Ah": 2.0 * (1 - 0.005 * (i + 1) * k), **extra(k)})
        return pool.add_soh(pd.DataFrame(out))
    checks("S7", ["S7-cal"], lambda k: {"age_type": 1, "age_temp": 25.0, "age_chg_rate": 0.0, "age_dischg_rate": 0.0,
                                        "age_profile": 2, "v_max_target_V": 4.2, "age_soc": 50.0}
           ).to_csv(d / "pool_s7_checks.csv", index=False)
    checks("S8", ["S8-Batch-1", "S8-Batch-2"], lambda k: {"cycle": 20 * k}).to_csv(d / "pool_s8_checks.csv", index=False)
    checks("S9", ["S9-D1_a-CY25-05_1"], lambda k: {"cycle": 30 * k}).to_csv(d / "pool_s9_checks.csv", index=False)


def test_d25_runner_smoke_parts_collect_and_summary(tmp_path):
    _write_tables(tmp_path)
    env = {**os.environ, "DATA_ROOT": str(tmp_path), "PYTHONPATH": str(ROOT), "MPLBACKEND": "Agg"}
    script = str(ROOT / "scripts" / "run_d25.py")
    # pre_main before pre_tune must refuse; the rest run in dependency order.
    bad = subprocess.run([sys.executable, script, "--exp", "e1", "--part", "pre_main", "--profile", "smoke"],
                         cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert bad.returncode != 0 and "pre_tune" in (bad.stdout + bad.stderr)
    for part in ("scratch_main", "pre_tune", "pre_main"):
        subprocess.run([sys.executable, script, "--exp", "e1", "--part", part, "--profile", "smoke"],
                       cwd=ROOT, env=env, check=True, capture_output=True, timeout=900)
    res = subprocess.run([sys.executable, script, "--exp", "e1", "--profile", "smoke", "--collect-only"],
                         cwd=ROOT, env=env, check=True, capture_output=True, text=True, timeout=900)
    out = tmp_path / "cache" / "d25" / "forecasting_d25_e1_smoke.json"
    data = json.loads(out.read_text())
    assert data["parts_run"] == ["scratch_main", "pre_tune", "pre_main"]
    assert data["parts_not_run"] == ["scratch_replicate", "pre_replicate"]
    methods = {r["method"] for r in data["runs"]}
    assert methods == {"tft_residual_ewc", "tft_residual_ewc_shrunk", "tft_pre_residual_ewc",
                       "tft_pre_residual_ewc_shrunk", "linear_extrapolation_conformal"}
    shr = [r for r in data["runs"] if r["method"].endswith("_shrunk")]
    assert all(0 <= v <= 1 for r in shr for v in r["alpha_final"].values()) and all("is90" in r for r in shr)
    assert {b["method"] for b in data["baselines"]} == {"linear_extrapolation", "persistence"}
    assert data["tuning"]["pre"]["pool_cells"] == 12
    assert not (ROOT / "results" / "forecasting_d25_e1_smoke.json").exists()   # smoke output stays outside Git
    assert "not run" in res.stdout
    summ = subprocess.run([sys.executable, str(ROOT / "scripts" / "summarise_d25.py"), "--results", str(out.parent)],
                          cwd=ROOT, env=env, check=True, capture_output=True, text=True, encoding="utf-8", timeout=300)
    assert "Pretrained residual TFT + EWC, shrunk" in summ.stdout and "parts not run" in summ.stdout


def test_night_plan_dependencies():
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_d25
    import run_d25_night as night
    assert {p for _, p in night.PLAN} == set(run_d25.PARTS) and night.DEPENDS == run_d25.DEPENDS
    for i, (exp, part) in enumerate(night.PLAN):                    # every dependency is queued before its part
        if part in night.DEPENDS:
            assert night.PLAN.index((exp, night.DEPENDS[part])) < i


def test_night_dry_run_preflight(tmp_path):
    _write_tables(tmp_path)
    env = {**os.environ, "DATA_ROOT": str(tmp_path), "PYTHONPATH": str(ROOT)}
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_d25_night.py"), "--dry-run", "--allow-dirty",
                          "--allow-unpinned"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "pre-flight passed" in res.stdout and "pre_main (after pre_tune)" in res.stdout
