"""Tests for decision D36: the S10 label rules, the condition mapping, the selective shrinkage and the D36 runner.

SYNTHETIC TEST DATA ONLY, built in pytest temporary directories (never the repository); it exercises code paths and
must never be used for reported results (AGENTS.md).
"""

import gzip
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

KINDS = [(1, 1)] * 4 + [(1, 2)] * 6 + [(2, 1)] * 3 + [(2, 2)] * 4      # 17 synthetic conditions: (stage, type)


def _write_s10(d: Path) -> None:
    rows = []
    for i, (stage, t) in enumerate(KINDS):
        for c in range(2):
            for k in range(9):
                soh = 100 - (0.2 + 0.05 * i) * k * (1 + 0.1 * c)
                rows.append({"dataset": "S10", "cell": f"s{stage}_c{i:02d}_{c}", "condition": f"s{stage}_x{i:02d}",
                             "stage": stage, "lab": "L1", "check_index": k + 1, "rpt_kind": "CU",
                             "check_time": pd.Timestamp("2022-01-01") + pd.Timedelta(days=30 * k),
                             "Capacity_Ah": 4.9 * soh / 100, "check_temp_c": 23.0, "age_type": t,
                             "age_temp": (10, 23, 35, 45)[i % 4], "soc_max": 0.5,
                             "dod": 0.0 if t == 1 else 0.3, "age_chg_rate": 0.0 if t == 1 else 0.5,
                             "age_dischg_rate": 0.0 if t == 1 else 1.0, "Reference_Capacity_Ah": 4.9,
                             "SOH_Percent": soh, "n_valid_checks": 9})
    pd.DataFrame(rows).to_csv(d / "pool_s10_checks.csv", index=False)


def _csv(path: Path, blocks: list[tuple[float, float, float]]) -> None:
    """A check-up file with step_type 22 blocks of (amps, final volts, start temperature), one row per minute."""
    rows, t = [], 0.0
    for amps, v_end, temp in blocks:
        rows.append({"run_time": "00:00:00.000", "c_vol": 4.2, "c_cur": 0.0, "c_surf_temp": temp, "step_type": 0})
        for j in range(61):
            cur = -amps if j < 60 else -0.25
            rows.append({"run_time": f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:06.3f}",
                         "c_vol": 4.2 - (4.2 - v_end) * j / 60, "c_cur": cur, "c_surf_temp": temp, "step_type": 22})
            t += 60.0
    with gzip.open(path, "wt") as f:
        pd.DataFrame(rows).to_csv(f, index=False)


def test_capacity_label_rules(tmp_path):
    import build_stroebl_checks as b
    _csv(tmp_path / "a.csv.gz", [(4.9, 2.5, 23.0)])                          # one complete block of ~1 h at 4.9 A
    r = b.capacity(tmp_path / "a.csv.gz")
    assert r["valid"] and r["Capacity_Ah"] == pytest.approx(4.9, rel=0.02)
    _csv(tmp_path / "b.csv.gz", [(4.9, 2.5, 23.0), (4.0, 3.2, 23.0)])        # last block incomplete: the first is used
    assert b.capacity(tmp_path / "b.csv.gz")["Capacity_Ah"] == pytest.approx(4.9, rel=0.02)
    _csv(tmp_path / "c.csv.gz", [(4.9, 2.5, 31.0)])                          # too warm at the block start
    assert b.capacity(tmp_path / "c.csv.gz")["reason"] == "temperature"
    _csv(tmp_path / "d.csv.gz", [(4.9, 3.0, 23.0)])                          # never reaches 2.55 V
    assert b.capacity(tmp_path / "d.csv.gz")["reason"] == "no complete block"


def test_start_date_and_order_rule(tmp_path):
    import build_stroebl_checks as b
    for name, text in (("ok", "17.02.2023"), ("short", "02.02.23"), ("bad", "xx.12.2021")):
        with gzip.open(tmp_path / f"{name}.gz", "wt") as f:
            f.write(f"Laboratory identifier: X\nMeasurement start date: {text}\n")
    assert b.start_date(tmp_path / "ok.gz") == pd.Timestamp("2023-02-17")
    assert b.start_date(tmp_path / "short.gz") == pd.Timestamp("2023-02-02")
    assert pd.isna(b.start_date(tmp_path / "bad.gz")) and pd.isna(b.start_date(tmp_path / "missing.gz"))
    days = [0, 10, 20, 15, 30, 400, 100]                                     # #4 repeats the past, #6 is after AT
    kinds = ["ET_T23"] + ["CU"] * 5 + ["AT_T23"]
    t = pd.DataFrame({"cell": "x", "check_index": range(1, 8), "rpt_kind": kinds, "valid": True, "reason": "",
                      "check_time": [pd.Timestamp("2022-01-01") + pd.Timedelta(days=d) for d in days]})
    out = b.order_rule(t)
    assert out["valid"].tolist() == [True, True, True, False, True, False, True]
    assert set(out.loc[~out["valid"], "reason"]) == {"date out of order"}


@pytest.fixture()
def runner(tmp_path, monkeypatch):
    _write_s10(tmp_path)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    import run_d36
    return run_d36.Runner("e1", "smoke")


def test_mapping_sets_and_selective_shrinkage(runner):
    import run_d36
    from src import forecasting as FC
    d = runner.direct
    assert "soc_max" in runner.features and d["soc_max"].eq(0.5).all()
    cal = d["calendar"] == 1
    assert (d.loc[cal, ["charge_c_rate", "dod"]] == 0).all().all() and d.loc[~cal, "rest_soc"].isna().all()
    assert (d.filter(like="chem_") == 0).all().all()
    s = runner.sets
    assert (s["set"] == "tuning").sum() == 7 and s.loc[s["set"] == "test", "fold"].value_counts().max() == 2
    frame, direct, hp = runner.held(runner.fold_conditions(0)[0][0])
    raw, shr = np.ones(len(frame)), np.zeros(len(frame))
    sc = run_d36.Runner.unit_scores(frame, direct, raw, shr, hp)
    y = frame[FC.TARGET].to_numpy(float)
    sel = np.where(hp, 0.0, 1.0)
    assert sc["sel"]["rmse"] == pytest.approx(np.sqrt(np.mean((sel - y) ** 2)))           # shrunk on H+, raw on H-
    assert sc["sel"]["rmse_hplus"] == pytest.approx(sc["shrunk"]["rmse_hplus"])
    assert sc["sel"]["rmse_hminus"] == pytest.approx(sc["raw"]["rmse_hminus"])


def test_runner_smoke_parts_and_collect(tmp_path):
    _write_s10(tmp_path)
    env = {**os.environ, "DATA_ROOT": str(tmp_path), "PYTHONPATH": str(ROOT), "MPLBACKEND": "Agg"}
    script = str(ROOT / "scripts" / "run_d36.py")
    for part in ("tune1_f0", "tune2_f0", "main_f0", "replicate_f0"):
        subprocess.run([sys.executable, script, "--exp", "e1", "--part", part, "--profile", "smoke"],
                       cwd=ROOT, env=env, check=True, capture_output=True, timeout=900)
    subprocess.run([sys.executable, script, "--exp", "e1", "--profile", "smoke", "--collect-only"],
                   cwd=ROOT, env=env, check=True, capture_output=True, timeout=900)
    data = json.loads((tmp_path / "cache" / "d36" / "forecasting_d36_e1_smoke.json").read_text())
    assert data["decision"] == "D36" and data["folds_in_tests"] == [0]
    assert "tft_residual_ewc_selective" in {r["method"] for r in data["condition_values"]}
    prim = {(x["a"], x["b"], x["metric"]) for x in data["verdicts"] if x["family"] == "primary"}
    assert prim == {("tft_residual_ewc_selective", "gbm_last", "rmse"),
                    ("tft_residual_ewc_selective", "linear_extrapolation", "rmse_hplus"),
                    ("tft_residual_ewc_selective", "gbm_all", "rmse")}
    assert "by_lab" in data["descriptive"] and "by_stage" in data["descriptive"]
    assert not (ROOT / "results" / "forecasting_d36_e1_smoke.json").exists()
