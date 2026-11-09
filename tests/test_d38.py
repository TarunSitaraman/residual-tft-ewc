"""Tests for decision D38: experience replay, the predict-before-training calibration, the direction rule, the
reproduction check and the D38 runner (smoke profile) with its orchestrator.

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

from test_d34 import _write_s7  # noqa: E402  (synthetic S7 check table)
from test_d36 import _write_s10  # noqa: E402  (synthetic S10 check table)


@pytest.fixture()
def runner(tmp_path, monkeypatch):
    _write_s10(tmp_path)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    import run_d38
    return run_d38.S10Runner("e1", "smoke")


def test_draw_and_stored_rows_are_seeded_and_bounded(runner):
    import run_d38 as D
    tasks = [runner.task(c) for c in runner.orders(0)[0]]
    t = tasks[0]
    a, b = D.draw(t.train, 8, 0, 3, 0), D.draw(t.train, 8, 0, 3, 0)
    assert len(a) == 8 and a.equals(b) and a.index.is_monotonic_increasing          # same draw, row order kept
    assert not a.equals(D.draw(t.train, 8, 1, 3, 0)) and not a.equals(D.draw(t.train, 8, 0, 4, 0))
    assert len(D.draw(t.train, 10 ** 6, 0, 0, 0)) == len(t.train) and D.draw(t.train, 0, 0, 0, 0).empty
    tr, va = D.stored_rows(tasks, 8, 0)
    assert all(len(f) == min(8, len(x.train)) for f, x in zip(tr, tasks))
    assert all(len(f) <= 2 for f in va)


def test_replay_model_keeps_the_stored_rows_and_uses_them(runner):
    import run_d38 as D
    from src import p3
    assert D.CL.STRATEGIES["tft_er"] is D.TFTExperienceReplay and D.CL.needs_history("tft_er")
    tasks = [runner.task(c) for c in runner.orders(0)[0][:3]]
    build = lambda **kw: p3.build_strategy("tft_er", runner.features, 0, runner.res,
                                           {**runner.settings(1e-2), "epochs": 30, "buffer_per_task": 8, **kw})
    s = build()
    assert float(s._penalty(None)) == 0.0                                           # nothing stored yet
    for t in tasks:
        s.fit_task(t)
    tr, va = D.stored_rows(tasks, 8, 0)
    assert len(s.buffer_train_) == 3 and all(a.equals(b) for a, b in zip(s.buffer_train_, tr))
    assert all(a.equals(b) for a, b in zip(s.buffer_val_, va))                      # GBM-ER sees the same rows
    assert s.config()["buffer_per_task"] == 8 and s.config()["val_buffer_per_task"] == 2
    again = build()
    for t in tasks:
        again.fit_task(t)
    frame = runner.held(runner.fold_conditions(0)[0][0])[0]
    assert np.array_equal(s.predict(frame), again.predict(frame))                   # deterministic
    plain = p3.build_strategy("tft_finetune", runner.features, 0, runner.res, {**runner.settings(1e-2), "epochs": 30})
    off = build(replay_weight=0.0, buffer_per_task=0)                               # no memory = plain fine-tuning
    for t in tasks:
        plain.fit_task(t)
        off.fit_task(t)
    assert max(plain.best_epoch_) >= 0                                              # the model did train
    assert np.array_equal(plain.predict(frame), off.predict(frame))
    assert not np.array_equal(plain.predict(frame), s.predict(frame))               # replay changes training


def test_calibration_per_horizon_and_stratum():
    import run_d38 as D
    rng = np.random.default_rng(0)
    n = 400
    h = np.tile([1, 2], n // 2)
    hp = np.repeat([True, False], n // 2)
    p = rng.normal(size=n)
    y = np.where(hp, 0.5 * p, 2.0 * p) + 0.01 * rng.normal(size=n)                  # H+: half the correction; H-: more
    cal = D.fit_calibration(p, y, h, hp, (1, 2))
    assert cal[(1, True)]["alpha"] == pytest.approx(0.5, abs=0.01) and cal[(1, False)]["alpha"] == 1.0   # clipped
    assert cal[(2, True)]["source"] == "stratum" and cal[(2, True)]["n"] == 100
    shrunk, q = D.apply_calibration(cal, p, h, hp)
    assert np.allclose(shrunk[hp], cal[(1, True)]["alpha"] * p[hp], atol=0.02 * np.abs(p[hp]).max())
    cov = np.mean(np.abs(y - shrunk) <= q[0.9])
    assert 0.88 <= cov <= 0.95                                                      # split-conformal on its own rows
    assert (q[0.9] >= q[0.8]).all()
    few = D.fit_calibration(p[:50], y[:50], h[:50], hp[:50], (1, 2), min_rows=30)   # 25 rows per horizon, all H+
    assert few[(1, True)]["source"] == "flagged" and few[(1, True)]["alpha"] == 1.0
    mix = D.fit_calibration(p, y, h, np.arange(n) < 10, (1, 2))                     # 5 H+ rows per horizon
    assert mix[(1, True)]["source"] == "horizon" and mix[(1, True)]["n"] == 200
    assert mix[(1, True)]["alpha"] == pytest.approx(mix[(1, False)]["alpha"], abs=0.05)


def test_direction_rule_flags_mean_and_rank_disagreement():
    import run_d38 as D
    row = lambda s, med, a, b: {"family": "primary", "a": "x", "b": "y", "metric": "rmse", "set": s,
                                "median_diff": med, "n_conditions_a_better": a, "n_conditions_b_better": b}
    v = {"family": "primary", "a": "x", "b": "y", "metric": "rmse", "verdict": "replicated, a better"}
    win = D.direction([row("main", -0.1, 40, 20), row("replicate", -0.2, 41, 19)], [v])[0]
    assert win["in_favour"] and not win["mean_and_rank_disagree"]
    d36_q3 = D.direction([row("main", 0.17, 12, 52), row("replicate", 0.15, 13, 51)], [v])[0]     # the D36 Q3 case
    assert not d36_q3["in_favour"] and d36_q3["mean_and_rank_disagree"]
    null = D.direction([row("main", -0.1, 40, 20), row("replicate", -0.2, 41, 19)],
                       [{**v, "verdict": "not replicated"}])[0]
    assert not null["in_favour"] and not null["mean_and_rank_disagree"]


def test_reproduction_check_compares_with_the_base_results(runner):
    import run_d38 as D
    conds = runner.fold_conditions(0)[0]
    base = pd.DataFrame([{"set": s, "method": m, "condition": c, "rmse": 1.0}
                         for s in D.SETS for m in (D.RAW, D.ALL, D.JOINT) for c in conds])
    runner._ref = {"condition_values": base.to_dict("records"), "tuning": {"0": {"lr": 0.002, "ewc_lambda": 0.5}}}
    assert runner.a0_settings(0) == (0.002, 0.5) and runner.a0_settings(1) == (runner.cfg["lr_grid"][0], 1.0)
    v = base[base["method"] != D.JOINT].assign(rmse=lambda t: t["rmse"] + np.where(t["method"] == D.RAW, 0.01, 1e-3))
    rep = D.reproduction(runner, v)
    assert rep["max_abs_diff_pp"][D.RAW] == pytest.approx(0.01) and rep["outside_tolerance"] == [D.ALL]
    assert rep["checked"] is False                                                  # smoke: recorded, not enforced
    runner._ref, runner.profile = None, "smoke"
    assert D.reproduction(runner, v) == {"checked": False, "reason": "no reference file"}


def test_preq_frame_holds_only_later_conditions(runner):
    order = runner.orders(0)[0]
    pf = runner.preq_frame(0, 0)
    assert set(pf["Battery_ID"]) == set(order[1:]) and pf["pos"].min() == 1 and pf["pos"].max() == len(order) - 1
    assert pf["hplus"].equals(pf["fade_rate_pp_per_day"].notna()) and pf["hplus"].any() and (~pf["hplus"]).any()
    held = set(runner.fold_conditions(0)[0])
    assert not held & set(pf["Battery_ID"]) and not set(runner.tuning) & set(pf["Battery_ID"])


@pytest.mark.parametrize("ds,write", [("s10", _write_s10), ("s7", _write_s7)])
def test_runner_smoke_parts_and_collect(tmp_path, ds, write):
    import run_d38 as D
    write(tmp_path)
    env = {**os.environ, "DATA_ROOT": str(tmp_path), "PYTHONPATH": str(ROOT), "MPLBACKEND": "Agg"}
    script = str(ROOT / "scripts" / "run_d38.py")
    parts = ["tune_f0", "r32_main_f0", "r32_replicate_f0", "a0_main_f0", "a0_replicate_f0", "r32_main_f1"]
    if ds == "s10":
        parts += ["r8_main_f0", "r8_replicate_f0"]
    for part in parts:
        subprocess.run([sys.executable, script, "--exp", f"{ds}_e1", "--part", part, "--profile", "smoke"],
                       cwd=ROOT, env=env, check=True, capture_output=True, timeout=900)
    res = subprocess.run([sys.executable, script, "--exp", f"{ds}_e1", "--profile", "smoke", "--collect-only"],
                         cwd=ROOT, env=env, check=True, capture_output=True, text=True, timeout=900)
    data = json.loads((tmp_path / "cache" / f"d38_{ds}" / f"forecasting_d38_{ds}_e1_smoke.json").read_text())
    assert data["decision"] == "D38" and data["folds_in_tests"] == [0] and data["v2"] == D.V2
    assert "r32_replicate_f1" in data["parts_not_run"] and "not run" in res.stdout
    arms = ["r32", "a0"] + (["r8"] if ds == "s10" else [])
    assert data["arms_in_tests"] == arms
    v = pd.DataFrame(data["condition_values"])
    want = {D.V2, D.ER["r32"], D.ER_ENS["r32"], D.GBM_ER["r32"], D.RAW, D.SHR, D.SEL, D.A0_ENS, D.A0_CAL,
            D.ALL, D.LAST, D.LIN, D.PERS, D.RIDGE}
    if ds == "s10":
        want |= {D.ER["r8"], D.ER_ENS["r8"], D.ER_CAL["r8"], D.GBM_ER["r8"]}
    assert set(v["method"]) == want and v.groupby(["set", "method"])["condition"].nunique().eq(2).all()
    assert v.loc[v["method"] == D.ER["r32"], "n_runs"].eq(4).all()                  # 2 orders x 2 seeds
    assert v.loc[v["method"] == D.V2, "n_runs"].eq(2).all()                         # one ensemble per order
    assert v.loc[v["method"] == D.V2, "cov90"].between(0, 1).all()
    prim = {(x["b"], x["metric"]) for x in data["verdicts"] if x["family"] == "primary"}
    prev = D.SEL if ds == "s10" else D.SHR
    assert prim == {(D.ALL, "rmse"), (D.GBM_ER["r32"], "rmse"), (D.LIN, "rmse_hplus"), (prev, "rmse")}
    assert all({"in_favour", "mean_and_rank_disagree", "median_main"} <= set(x) for x in data["verdicts"])
    left = {tuple(t) for t in data["tests_not_run"]}
    assert (D.V2, D.JOINT, "rmse") in left                                          # no base results in the smoke run
    assert ((D.ER_CAL["r8"], D.V2, "rmse") in left) == (ds == "s7")
    cov = data["descriptive"]["coverage"][D.V2]["main"]["90"]
    assert 0 <= cov["pooled"] <= 1 and set(cov["by_order"]) == {"0", "1"}
    assert data["descriptive"]["coverage_check"]["method"] == D.V2
    assert data["reproduction"] == {"checked": False, "reason": "no reference file"}
    assert data["tuning"]["0"]["replay"]["lr"] in data["settings"]["lr_grid"]
    cal = data["calibration"][0]
    assert {"h1_hplus", "h1_hminus", "n_rows"} <= set(cal) and 0 <= cal["h1_hplus"]["alpha"] <= 1
    assert not (ROOT / "results" / f"forecasting_d38_{ds}_e1_smoke.json").exists()   # smoke output stays outside Git


def test_night_plan_order_dependencies_and_dry_run(tmp_path):
    import importlib

    import run_d25_night
    import run_d38 as D
    import run_d38_night as night
    plan = night.plan("s10")
    assert len(plan) == 2 * len(D.PARTS) == 70
    stage = [p.split("_")[0] for _, p in plan]
    assert stage == sorted(stage, key=["tune", "r32", "a0", "r8"].index)            # the primary arm before the others
    assert plan[0] == ("s10_e2", "tune_f0") and len(night.plan("s10", r8=False)) == 50
    assert D.DEPENDS["r32_main_f3"] == "tune_f3" and D.DEPENDS["r8_replicate_f1"] == "tune_f1"
    assert not any(p.startswith("a0") for p in D.DEPENDS)
    night.configure("s7")
    try:
        assert run_d25_night.RUNNER.name == "run_d38.py" and run_d25_night.DATA_FILES == ("pool_s7_checks.csv",)
        assert run_d25_night.LOGS.parent.name == "d38_s7"
    finally:
        importlib.reload(run_d25_night)
    _write_s10(tmp_path)
    env = {**os.environ, "DATA_ROOT": str(tmp_path), "PYTHONPATH": str(ROOT)}
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_d38_night.py"), "--dataset", "s10", "--dry-run",
                          "--allow-dirty", "--allow-unpinned", "--no-r8"], cwd=ROOT, env=env, capture_output=True,
                         text=True, timeout=120)
    assert res.returncode == 0 and "s10_e2 tune_f0" in res.stdout and "(after tune_f4)" in res.stdout
    assert "r8_main" not in res.stdout


def test_paper_profile_needs_the_committed_base_results(tmp_path):
    _write_s10(tmp_path)
    env = {**os.environ, "DATA_ROOT": str(tmp_path), "PYTHONPATH": str(ROOT)}
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_d38.py"), "--exp", "s10_e1", "--part", "tune_f0"],
                         cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert res.returncode != 0 and "d36_condition_sets.csv" in res.stdout + res.stderr   # synthetic sets differ
