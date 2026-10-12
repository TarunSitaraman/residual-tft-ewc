"""Tests for decision D24: sequence baselines, shift-adaptive EWC (RC-TFT), tabular baselines and the D24 runner.

SYNTHETIC TEST DATA ONLY, built in the tests or in pytest temporary directories (never the repository); it exercises
code paths and must never be used for reported results (CLAUDE.md).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src import continual_learning as cl
from src import forecasting as FC
from src import p3
from src import pool
from src import tabular_baselines as TB
from src.models.sequence_baselines import SEQUENCE_BACKBONES, TCNRegressor, TransformerRegressor, n_parameters
from src.models.tft_continual import TemporalFusionTransformer

ROOT = Path(__file__).resolve().parents[1]
N = len(p3.SHARED_FEATURES)
FAST = {"epochs": 3, "patience": 2, "window": 2, "window_group": ("Battery_ID", "Cycling_Test")}


def _x(b=4, t=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, t, 2 * N + 1, generator=g)
    x[..., -1] = 0.0
    return x


# --- sequence backbones -----------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(SEQUENCE_BACKBONES))
def test_sequence_backbones_shape_size_and_input_check(name):
    model = SEQUENCE_BACKBONES[name](N).eval()
    y = model(_x())
    assert y.shape == (4,) and torch.isfinite(y).all()
    tft = n_parameters(TemporalFusionTransformer(N, 8, 2, 0.1))
    assert 0.8 <= n_parameters(model) / tft <= 1.2                    # D24: size matched to the TFT within 20 %
    with pytest.raises(ValueError):
        model(torch.randn(2, 4, 3))


def test_tcn_is_causal():
    model = TCNRegressor(N, dropout=0.0).eval()
    x = _x(t=6)
    h1 = model.blocks(x.transpose(1, 2))
    x2 = x.clone()
    x2[:, 4:] += 5.0                                                  # change steps 4 and 5 only
    h2 = model.blocks(x2.transpose(1, 2))
    torch.testing.assert_close(h1[:, :, :4], h2[:, :, :4])           # outputs at steps 0-3 unchanged


def test_transformer_ignores_padded_steps_and_is_causal():
    model = TransformerRegressor(N, dropout=0.0).eval()
    x = _x()
    x[:, :2, -1] = 1.0                                                # first two steps padded
    x2 = x.clone()
    x2[:, :2, :-1] += 7.0
    torch.testing.assert_close(model(x), model(x2))


# --- shift-adaptive EWC -------------------------------------------------------------------------------

def _checks(condition, n_cells=2, n_checks=12, fade=0.01, dataset="S6"):
    rows = [{"dataset": dataset, "cell": f"{condition}-cell{c}", "condition": condition, "cycle": 50 * (k + 1),
             "check_time": pd.Timestamp("2000-01-01") + pd.Timedelta(days=7 * k),
             "Capacity_Ah": 1.1 - fade * (c + 1) * k} for c in range(n_cells) for k in range(n_checks)]
    return pool.add_soh(pd.DataFrame(rows))


def _s6_frame(cases=("case1", "case4", "case7")):
    fades = {"case1": 0.004, "case4": 0.012, "case7": 0.03}
    s = pd.concat([p3.harmonised_samples(_checks(c, fade=fades.get(c, 0.01)), "S6") for c in cases], ignore_index=True)
    return s.assign(lin_forecast=0.0)


def test_shift_measure():
    a = (np.zeros(4), np.ones(4))
    assert cl.ShiftAdaptiveEWC.shift(a, a) == 0.0
    b = (np.array([2.0, 0, 0, 0]), np.ones(4))
    assert cl.ShiftAdaptiveEWC.shift(a, b) == pytest.approx(1.0)        # sqrt(4 / 4)


def test_saewc_weights_follow_shift_and_reduce_to_ewc():
    frame = _s6_frame()
    stream = FC.forecast_stream(frame, ["case1", "case4", "case7"])
    s = p3.build_strategy("tft_saewc", list(p3.SHARED_FEATURES), 0, frame, {**FAST, "ewc_lambda": 10.0})
    for t in stream:
        s.fit_task(t)
    assert [len(w) for w in s.shift_weights_] == [0, 1, 2] and len(s.fishers_) == 3
    # The third task's weights are the shifts between its training inputs and each earlier task's.
    cur = s._input_stats(stream[2].train)
    assert s.shift_weights_[2] == pytest.approx([cl.ShiftAdaptiveEWC.shift(st, cur) for st in s.input_stats_[:2]])
    # With unit weights the penalty is exactly the EWC penalty.
    s.task_weights_ = [1.0] * len(s.fishers_)
    torch.testing.assert_close(s.penalty(), cl.EWC.penalty(s))
    s.task_weights_ = [0.0] * len(s.fishers_)
    assert float(s.penalty().detach()) == 0.0


def test_saewc_is_deterministic():
    frame = _s6_frame()
    stream = FC.forecast_stream(frame, ["case1", "case4", "case7"])

    def run():
        s = p3.build_strategy("tft_saewc", list(p3.SHARED_FEATURES), 3, frame, {**FAST, "ewc_lambda": 10.0})
        for t in stream:
            s.fit_task(t)
        return s.predict(stream[0].test)
    np.testing.assert_array_equal(run(), run())


@pytest.mark.parametrize("name", [f"{b}_{m}" for b in SEQUENCE_BACKBONES for m in ("finetune", "ewc")])
def test_sequence_strategies_train_on_a_stream(name):
    frame = _s6_frame()
    assert cl.needs_history(name) and not cl.needs_history("ewc")
    extra = {"ewc_lambda": 10.0} if name.endswith("ewc") else {}
    s = p3.build_strategy(name, list(p3.SHARED_FEATURES), 0, frame, {**FAST, **extra})
    out = p3.run_stream(s, FC.forecast_stream(frame, ["case1", "case4"]), None)
    assert out["R"].shape == (2, 2) and np.isfinite(out["R"].to_numpy()).all()
    assert s.config()["backbone"] == name.split("_")[0]
    # Labels never enter the windows.
    assert FC.TARGET not in s.windows.history.columns and "SOH_Percent" not in s.windows.history.columns


# --- tabular baselines --------------------------------------------------------------------------------

@pytest.mark.parametrize("model", TB.TABULAR_MODELS)
def test_tabular_baselines_direct_and_residual(model):
    frame = _s6_frame()
    frame = frame.assign(lin_forecast=np.linspace(-0.5, 0.0, len(frame)))
    tasks = FC.forecast_stream(frame, ["case1", "case4", "case7"])
    train = pd.concat([t.train for t in tasks], ignore_index=True)
    val = pd.concat([t.val for t in tasks], ignore_index=True)
    tests = {t.name: t.test for t in tasks}
    for arm in ("direct", "residual"):
        out = TB.fit_predict(model, train, val, tests, p3.SHARED_FEATURES, arm=arm, seed=0)
        assert set(out["pred"]) == set(tests)
        assert all(len(out["pred"][k]) == len(v) and np.isfinite(out["pred"][k]).all() for k, v in tests.items())
        if model == "svr":
            assert out["config"]["C"] in TB.SVR_C_GRID
    # Deterministic.
    a = TB.fit_predict(model, train, val, tests, p3.SHARED_FEATURES, seed=0)["pred"]["case1"]
    b = TB.fit_predict(model, train, val, tests, p3.SHARED_FEATURES, seed=0)["pred"]["case1"]
    np.testing.assert_array_equal(a, b)


def test_residual_arm_adds_the_extrapolation_back():
    frame = _s6_frame().assign(lin_forecast=0.0)
    tasks = FC.forecast_stream(frame, ["case1", "case4"])
    train = pd.concat([t.train for t in tasks], ignore_index=True)
    tests = {t.name: t.test for t in tasks}
    d = TB.fit_predict("rf", train, train, tests, p3.SHARED_FEATURES, arm="direct")["pred"]
    r = TB.fit_predict("rf", train, train, tests, p3.SHARED_FEATURES, arm="residual")["pred"]
    np.testing.assert_allclose(d["case1"], r["case1"])               # lin = 0: identical arms


# --- runner (smoke profile on a synthetic S6 table outside the repository) ----------------------------

def _wheeler_table(path: Path) -> None:
    rows = []
    for case, fade in (("case1", 0.004), ("case4", 0.012), ("case7", 0.03)):
        for c in range(2):
            cap0 = 1.1
            for k in range(14):
                rows.append({"Battery_ID": case, "cell": f"{case[-1]}{'ab'[c]}", "cycle": 100 * k,
                             "check_time": pd.Timestamp("2001-01-01") + pd.Timedelta(days=9 * k),
                             "Capacity_Ah": cap0 * (1 - fade * k * (1 + 0.1 * c)), "valid": True})
    t = pd.DataFrame(rows)
    t["SOH_Percent"] = t["Capacity_Ah"] / t.groupby("cell")["Capacity_Ah"].transform("first") * 100
    t.to_csv(path, index=False)


def test_d24_runner_smoke_part_and_collect(tmp_path):
    _wheeler_table(tmp_path / "wheeler_checks.csv")
    env = {**os.environ, "DATA_ROOT": str(tmp_path), "PYTHONPATH": str(ROOT)}
    script = str(ROOT / "scripts" / "run_d24.py")
    for part in ("tft_residual_saewc", "lstm_residual"):
        subprocess.run([sys.executable, script, "--exp", "e1", "--part", part, "--profile", "smoke"],
                       cwd=ROOT, env=env, check=True, capture_output=True, timeout=900)
    res = subprocess.run([sys.executable, script, "--exp", "e1", "--profile", "smoke", "--collect-only"],
                         cwd=ROOT, env=env, check=True, capture_output=True, text=True, timeout=900)
    out = next((tmp_path / "cache" / "d24").glob("forecasting_d24_e1_smoke.json"))
    data = json.loads(out.read_text())
    assert data["parts_run"] == ["lstm_residual", "tft_residual_saewc"]
    assert set(data["parts_not_run"]) == {"tcn_residual", "transformer_residual", "lstm_direct", "tcn_direct",
                                          "transformer_direct"}
    methods = {r["method"] for r in data["runs"]}
    assert methods == {"tft_residual_saewc", "lstm_residual_finetune", "lstm_residual_ewc"}
    base = {b["method"] for b in data["baselines"]}
    assert {"linear_extrapolation", "gbm_shared", "gpr_direct", "gpr_residual", "rf_residual", "svr_direct"} <= base
    assert all(len(r["shift_weights"]) == 3 for r in data["runs"] if r["method"] == "tft_residual_saewc")
    assert not (ROOT / "results" / "forecasting_d24_e1_smoke.json").exists()   # smoke output stays outside Git
    assert "not run" in res.stdout
    gpr = next(b for b in data["baselines"] if b["method"] == "gpr_direct")["config"]
    assert gpr["n_fit_rows"] == min(gpr["n_train_rows"], 200)                    # smoke cap
    # Tables, figures and the summary work on a D24-shaped results file (written outside the repository).
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_d24_outputs.py"), "--results", str(out.parent),
                    "--out", str(tmp_path / "outputs")], cwd=ROOT, env=env, check=True, capture_output=True, timeout=600)
    comp = pd.read_csv(tmp_path / "outputs" / "tables" / "d24_comparison_e1.csv")
    assert {"tft_residual_saewc", "lstm_residual_ewc", "gpr_residual", "linear_extrapolation"} <= set(comp["method"])
    assert (tmp_path / "outputs" / "figures" / "d24_comparison_e1.png").exists()
    summ = subprocess.run([sys.executable, str(ROOT / "scripts" / "summarise_d24.py"), "--results", str(out.parent)],
                          cwd=ROOT, env=env, check=True, capture_output=True, text=True, encoding="utf-8", timeout=600)
    assert "RC-TFT" in summ.stdout and "Parts not run" in summ.stdout


def test_baselines_part_is_checkpointed(tmp_path):
    _wheeler_table(tmp_path / "wheeler_checks.csv")
    env = {**os.environ, "DATA_ROOT": str(tmp_path), "PYTHONPATH": str(ROOT)}
    script = str(ROOT / "scripts" / "run_d24.py")
    subprocess.run([sys.executable, script, "--exp", "e2", "--part", "baselines", "--profile", "smoke"],
                   cwd=ROOT, env=env, check=True, capture_output=True, timeout=900)
    ck = list((tmp_path / "cache" / "d24").rglob("baselines__*.pickle"))
    assert len(ck) == 1 and list((tmp_path / "cache" / "d24").rglob("done_baselines.json"))


def test_gpr_subsample_cap():
    frame = _s6_frame().assign(lin_forecast=0.0)
    tasks = FC.forecast_stream(frame, ["case1", "case4", "case7"])
    train = pd.concat([t.train for t in tasks], ignore_index=True)
    out = TB.fit_predict("gpr", train, train, {"x": tasks[0].test}, p3.SHARED_FEATURES, gpr_max_rows=40)
    assert out["config"]["n_fit_rows"] == 40 < out["config"]["n_train_rows"] == len(train)


def test_baselines_only_writes_a_results_file_outside_git_in_smoke(tmp_path):
    _wheeler_table(tmp_path / "wheeler_checks.csv")
    env = {**os.environ, "DATA_ROOT": str(tmp_path), "PYTHONPATH": str(ROOT)}
    subprocess.run([sys.executable, str(ROOT / "scripts" / "run_d24.py"), "--exp", "e1", "--baselines-only",
                    "--profile", "smoke"], cwd=ROOT, env=env, check=True, capture_output=True, timeout=900)
    out = json.loads((tmp_path / "cache" / "d24" / "forecasting_d24_baselines_e1_smoke.json").read_text())
    names = {b["method"] for b in out["baselines"]}
    assert {"gpr_direct", "gpr_residual", "rf_direct", "rf_residual", "svr_direct", "svr_residual",
            "linear_extrapolation", "gbm_shared"} <= names
    assert not (ROOT / "results" / "forecasting_d24_baselines_e1_smoke.json").exists()
