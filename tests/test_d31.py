"""Tests for decision D31: online adaptation of the joint residual TFT with EWC on an unseen condition (P2).

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
import torch

from test_d26 import _write_wheeler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SCRIPT = str(ROOT / "scripts" / "run_d31.py")


@pytest.fixture()
def run(tmp_path, monkeypatch):
    _write_wheeler(tmp_path)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    import run_d31
    return run_d31.D31("e1", "smoke")


def _anchored(run, case):
    import run_d31
    base, tasks = run.joint(run.fold_order(run.cases, case), 0)
    pooled = pd.concat([t.train for t in tasks], ignore_index=True)
    return base, run_d31.anchor(base, pooled, run.settings(run.lr), 0, run.s6), tasks


def test_anchor_penalty_and_independent_arms(run):
    import run_d31
    base, anchored, _ = _anchored(run, run.cases[0])
    ewc = run_d31.arm_copy(anchored, 10.0)
    assert len(ewc.fishers_) == 1 and float(ewc.penalty().detach()) == 0.0              # theta = theta* at the start
    with torch.no_grad():
        for p in ewc.model.parameters():
            p.add_(0.1)
    assert float(ewc.penalty().detach()) > 0
    same = [torch.equal(a, b) for a, b in zip(anchored.model.parameters(), base.model.parameters())]
    assert all(same)                                                         # copies do not touch the anchor
    assert run_d31.arm_copy(anchored, 0.0).ewc_lambda == 0.0


def test_online_no_lookahead(run):
    import run_d28
    import run_d31
    case = run.cases[0]
    base, anchored, _ = _anchored(run, case)
    frame = run.held(case)[0]
    pred, updates = run_d31.online(run_d31.arm_copy(anchored, 1.0), frame, 2, 0)
    t, m = run_d28.times(frame)
    before = t < m.min()                                                       # issued before anything matured
    assert before.any() and np.allclose(pred[before], base.predict(frame)[before], atol=0, rtol=0)
    assert updates == run_d31.n_updates(frame) > 0
    assert not np.allclose(pred[~before], base.predict(frame)[~before])        # later forecasts were adapted


def test_select_and_verdict_rules():
    import run_d31
    scores = {("ft_online", 5, 0.0): 1.0, ("ft_online", 20, 0.0): 1.0,
              ("ewc_online", 20, 0.1): 0.5, ("ewc_online", 20, 10.0): 0.5, ("ewc_online", 5, 1.0): 0.6}
    assert run_d31.select(scores, "ft_online") == (5, 0.0)                    # tie: fewer epochs
    assert run_d31.select(scores, "ewc_online") == (20, 10.0)                 # tie: larger lambda
    cell = lambda d, p: {"mean_diff": d, "p_value": p}
    assert run_d31.fold_verdict([cell(-0.1, 0.016)] * 4) == "positive"
    assert run_d31.fold_verdict([cell(0.1, 0.016)] * 4) == "negative"
    assert run_d31.fold_verdict([cell(0.01, 0.5)] * 4) == "as accurate as"
    assert run_d31.fold_verdict([cell(-0.1, 0.016)] * 3 + [cell(-0.1, 0.08)]) == "not established"


def test_d31_parts_collect_and_stop_condition(run, tmp_path):
    import run_d26
    parts = {c: run.run_part(f"f{k}") for k, c in enumerate(run.cases)}
    for p in parts.values():
        sel = p["selection"]
        assert sel["ft_online"]["e_ft"] in (5, 20) and sel["ewc_online"]["lambda"] in (0.1, 1.0, 10.0)
        for u in p["units"].values():
            assert u["joint_static"]["n_updates"] == 0 and u["ewc_online"]["n_updates"] == p["n_updates"]
    seeds = [int(s) for s in parts[run.cases[0]]["units"]]
    d26 = {"runs": [{"method": run_d26.JOINT, "order_id": 0, "seed": s,
                     "per_fold": {c: parts[c]["units"][str(s)]["joint_static"]["scores"]["rmse"] for c in parts}}
                    for s in seeds]}
    d28 = {"per_fold": [{"case": c, "rmse": {"refit": 0.5, "static": 0.6}} for c in parts]}
    (tmp_path / "d26.json").write_text(json.dumps(d26))
    (tmp_path / "d28.json").write_text(json.dumps(d28))
    env = {**os.environ, "DATA_ROOT": str(tmp_path), "PYTHONPATH": str(ROOT)}
    args = [sys.executable, SCRIPT, "--exp", "e1", "--profile", "smoke", "--d26", str(tmp_path / "d26.json"),
            "--d28", str(tmp_path / "d28.json")]
    out = tmp_path / "d31.json"
    res = subprocess.run(args + ["--out", str(out)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert res.returncode == 0, res.stdout + res.stderr
    data = json.loads(out.read_text())
    assert data["reproduction_check"]["passed"] and set(data["tests"]) == {"main", "replicate"}
    assert {"q1_ewc_online_vs_d28_refit", "q2_ewc_online_vs_joint_static",
            "q3_ewc_online_vs_ft_online"} <= set(data["tests"]["main"])
    d26["runs"][0]["per_fold"][run.cases[0]] += 1e-6
    (tmp_path / "d26.json").write_text(json.dumps(d26))
    out2 = tmp_path / "d31_bad.json"
    res = subprocess.run(args + ["--out", str(out2)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert res.returncode == 3 and "STOP" in res.stdout and not out2.exists()
