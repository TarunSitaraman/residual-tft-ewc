"""Tests of the dashboard export (scripts/export_dashboard_data.py).

SYNTHETIC TEST DATA ONLY: every input below is built in a pytest temporary directory to exercise code paths; it is
never written into the repository and never feeds reported results (AGENTS.md).
"""

from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import export_dashboard_data as X  # noqa: E402
from src import data_pipeline as dp  # noqa: E402

ORDERS = [["case1", "case2", "case3"], ["case3", "case1", "case2"]]


def stream_json() -> dict:
    runs = []
    for oid, order in enumerate(ORDERS):
        per = {c: 0.1 * (i + 1) for i, c in enumerate(order)}
        runs.append({"set": "main", "method": "tft_residual_ewc_shrunk", "seed": 0, "order_id": oid,
                     "final_avg_error": 0.3, "forgetting": 0.05, "backward_transfer": 0.01, "plasticity": 0.29,
                     "forward_transfer": None, "rmse_h1": 0.2, "rmse_h2": 0.3, "rmse_h3": 0.4,
                     "final_error_per_task": per, "diagonal_error_per_task": per,
                     "forgetting_per_task": {c: 0.01 for c in order[:-1]},
                     "backward_transfer_per_task": {c: 0.0 for c in order[:-1]},
                     "forward_transfer_per_task": {c: -0.5 for c in order[1:]}})
    return {"orders": ORDERS, "runs": runs,
            "summary": [{"set": s, "method": "tft_residual_ewc_shrunk", "final_avg_error_mean": 0.3}
                        for s in ("main", "replicate")],
            "baselines": [{"method": "linear_extrapolation", "final_avg_error": 0.34,
                           "final_error_per_task": {"case1": 0.3, "case2": 0.35, "case3": 0.37}},
                          {"method": "persistence", "final_avg_error": 1.6}, {"method": "gbm_shared", "final_avg_error": 0.45}],
            "verdicts": [{"family": "primary", "a": "tft_residual_ewc_shrunk", "b": "linear_extrapolation",
                          "metric": "final_avg_error", "diff_main": -0.02, "diff_replicate": -0.03,
                          "p_holm_main": 0.01, "p_holm_replicate": 0.01, "verdict": "practically equivalent"}]}


def unseen_json() -> dict:
    folds = {"case1": 0.9, "case2": 1.1, "case3": 2.0}
    return {"runs": [{"set": "main", "method": "tft_residual_ewc_shrunk", "seed": 0, "order_id": 0, "p2_error": 1.0,
                      "per_fold": folds, "per_fold_hplus": folds, "cov90_per_fold": None}],
            "summary": [{"set": s, "method": "tft_residual_ewc_shrunk", "p2_error_mean": 0.92} for s in ("main", "replicate")],
            "baselines": [{"method": "gbm_shared", "p2_error": 0.54, "per_fold": folds},
                          {"method": "linear_extrapolation", "p2_error": 1.37}, {"method": "persistence", "p2_error": 2.8}],
            "tests": [{"set": "main", "a": "tft_residual_ewc_shrunk", "b": "gbm_shared", "metric": "p2_error",
                       "n": 50, "mean_diff": 0.39, "p_value": 1e-9, "p_holm": 1e-9}],
            "verdicts": [{"family": "primary", "a": "tft_residual_ewc_shrunk", "b": "gbm_shared", "metric": "p2_error",
                          "diff_main": 0.39, "diff_replicate": 0.41, "p_holm_main": 1e-9, "p_holm_replicate": 1e-9,
                          "verdict": "replicated, a worse"},
                         {"family": "primary", "a": "tft_residual_ewc_shrunk", "b": "linear_extrapolation",
                          "metric": "p2_error", "diff_main": -0.45, "diff_replicate": -0.43, "p_holm_main": 1e-9,
                          "p_holm_replicate": 1e-9, "verdict": "replicated, a better"}]}


def d32_json() -> dict:
    return {"gbm_runs": [{"method": "gbm_warm", "order_id": 0, "p2_error": 2.2, "per_fold": {"case1": 2.4}}],
            "summary": {"main_tft_residual_ewc_shrunk": {"p2_error": 0.92}, "gbm_last": {"p2_error": 0.86},
                        "gbm_all": {"p2_error": 0.54}},
            "verdicts": [{"family": "primary", "a": "tft_residual_ewc_shrunk", "b": "gbm_last", "metric": "p2_error",
                          "diff_main": 0.06, "diff_replicate": 0.08, "p_holm_main": 0.01, "p_holm_replicate": 0.01,
                          "verdict": "replicated, a worse"}]}


@pytest.fixture()
def results_dir(tmp_path: Path) -> Path:
    res = tmp_path / "results"
    (res / "tables").mkdir(parents=True)
    for e in ("e1", "e2"):
        for n in ("d23", "d25"):
            (res / f"forecasting_{n}_{e}.json").write_text(json.dumps(stream_json()))
        (res / f"forecasting_d26_{e}.json").write_text(json.dumps(unseen_json()))
        (res / f"forecasting_d32_{e}.json").write_text(json.dumps(d32_json()))
    pd.DataFrame({"dataset": ["S6"], "method": ["persistence"], "name": ["Persistence"], "rmse_pp_h1": [0.8],
                  "rmse_pp_h2": [1.5], "rmse_pp_h3": [2.3]}).to_csv(res / "tables" / "horizon.csv", index=False)
    return res


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    (data / "predictions").mkdir(parents=True)
    cap = {"g1": [4.8, 4.7, 4.5], "v4": [4.9, 4.4, 4.0]}
    rows = []
    for b, caps in cap.items():
        for k, c in enumerate(caps, start=1):
            rows.append({"Battery_ID": b, "RPT": f"RPT_{k}", "RPT_Number": k, "Test_Date": f"2024-0{k}-01",
                         "Capacity_Ah": c, "Reference_Capacity_Ah": caps[0], "SOH_Percent": c / caps[0] * 100,
                         "SOH_handover": np.nan, "rpt_temp_mean": 25.0 if k != 2 else np.nan})
    # Written out of order to check that the export sorts by cell and check.
    pd.DataFrame(rows[::-1]).to_csv(data / "rpt_health_c20.csv", index=False)
    w = pd.DataFrame({"cell": ["cell1a", "cell1a", "cell1a", "cell7b"], "Battery_ID": ["case1", "case1", "case1", "case7"],
                      "cycle": [0, 100, 200, 0], "check_time": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-01-01"],
                      "valid": [True, False, True, True], "Capacity_Ah": [2.3, np.nan, 2.1, 2.2],
                      "Reference_Capacity_Ah": [2.3, 2.3, 2.3, 2.2]})
    w["SOH_Percent"] = w["Capacity_Ah"] / w["Reference_Capacity_Ah"] * 100
    w.to_csv(data / "wheeler_checks.csv", index=False)
    preds = pd.DataFrame({"Battery_ID": ["case1", "case1"], "cell": ["cell1a"] * 2, "horizon": [1, 2],
                          "method": ["gbm_F1"] * 2, "seed": [0, 0], "soh_origin": [96.0, 96.0],
                          "soh_target": [94.0, 89.0], "soh_pred": [95.0, 91.0]})
    with gzip.open(data / "predictions" / "d30_s6_p2.csv.gz", "wt") as f:
        preds.to_csv(f, index=False)
    return data


def test_constants_match_source_modules():
    assert X.S1_GROUPS == dp.BATTERY_GROUPS
    from src import evaluation as ev
    from src import wheeler as W
    soh = np.array([101.0, 95.0, 94.99, 90.0, 85.0, 80.0, 60.0])
    np.testing.assert_array_equal(X.reuse_class(soh), ev.reuse_class(soh))
    ours = pd.DataFrame(X.S6_CASES, columns=["Battery_ID", "mode", "v_max", "charge_c_rate", "discharge", "dod"])
    theirs = W.TEST_CASES
    assert list(ours["Battery_ID"]) == list(theirs["Battery_ID"])
    np.testing.assert_allclose(ours[["v_max", "charge_c_rate", "dod"]], theirs[["v_max", "charge_c_rate", "dod"]])
    assert list(ours["mode"]) == ["CC-CV" if c else "CC" for c in theirs["cccv_charge"]]
    assert list(ours["discharge"]) == ["CC 2C" if c else "WLTP" for c in theirs["cc2c_discharge"]]


def test_reuse_class_keeps_nan():
    out = X.reuse_class([np.nan, 99.0])
    assert math.isnan(out[0]) and out[1] == 0


def test_stream_tables_rows_and_arrival_positions():
    runs, tasks, base = X.stream_tables(stream_json(), "d25", "e1")
    assert len(runs) == 2 and set(runs["experiment"]) == {"D25"}
    assert len(tasks) == 2 * 3
    t = tasks.set_index(["order_id", "group"])
    for oid, order in enumerate(ORDERS):
        for pos, c in enumerate(order, start=1):
            assert t.loc[(oid, c), "arrival_pos"] == pos
    # The last task of an order has no forgetting; the first has no forward transfer.
    assert math.isnan(t.loc[(0, "case3"), "forgetting"]) and math.isnan(t.loc[(0, "case1"), "forward_transfer"])
    assert {"all", "case1", "case2", "case3"} == set(base.loc[base["method"] == "linear_extrapolation", "group"])


def test_unseen_tables_include_baselines_per_fold():
    runs, folds = X.unseen_tables(unseen_json(), "d26", "e1")
    assert runs["is_baseline"].sum() == 3
    assert set(folds["method"]) == {"tft_residual_ewc_shrunk", "gbm_shared"}
    assert len(folds) == 6


def test_soh_table_order_relative_soh_and_missing_temperature(data_dir: Path):
    t = X.soh_table(data_dir)
    assert set(t["dataset"]) == {"S1", "S6"}
    # Each unit belongs to exactly one group (no mixing) and checks are in order within a unit.
    assert (t.groupby("unit")["group"].nunique() == 1).all()
    for _, g in t.groupby(["dataset", "unit"]):
        assert g["check_index"].is_monotonic_increasing
    # SOH relative to the first valid check of each cell.
    first = t[t["valid"]].groupby("unit").first()
    np.testing.assert_allclose(first["soh_pct"], 100.0)
    v = t[t["valid"]]
    np.testing.assert_allclose(v["soh_pct"], v["capacity_ah"] / v["reference_capacity_ah"] * 100)
    # Missing temperatures stay missing and are flagged; S6 has no per-check reading.
    s1 = t[t["dataset"] == "S1"]
    assert s1["temperature_c"].isna().sum() == 2
    assert set(s1.loc[s1["temperature_c"].isna(), "temperature_source"]) == {"missing, not filled"}
    assert t.loc[t["dataset"] == "S6", "temperature_c"].isna().all()
    # The invalid S6 check (no C/25 label) has no SOH and no class.
    bad = t[(t["unit"] == "cell1a") & (t["check_index"] == 100)].iloc[0]
    assert not bad["valid"] and math.isnan(bad["soh_pct"]) and pd.isna(bad["reuse_class"])


def test_predictions_table_classes(data_dir: Path):
    p = X.predictions_table(data_dir)
    assert list(p["class_true"]) == ["C1 moderate fade", "C2 substantial fade"]
    assert list(p["class_pred"]) == ["C1 moderate fade", "C1 moderate fade"]
    assert list(p["class_distance"]) == [0, -1]
    np.testing.assert_allclose(p["error_pp"], [1.0, 2.0])


def test_main_end_to_end(results_dir: Path, data_dir: Path, tmp_path: Path):
    out_r, out_d = tmp_path / "out_r", tmp_path / "out_d"
    assert X.main(["--results", str(results_dir), "--data-root", str(data_dir), "--out-results", str(out_r),
                   "--out-data", str(out_d)]) == 0
    for name in ("dim_group", "dim_method", "fact_stream_run", "fact_stream_task", "fact_unseen_run",
                 "fact_unseen_fold", "fact_verdicts", "fact_horizon", "fact_headline"):
        assert (out_r / f"{name}.csv").exists(), name
    assert (out_d / "fact_soh.csv").exists() and (out_d / "fact_predictions.csv").exists()
    summary = json.loads((out_r / "summary.json").read_text())   # strict JSON: no NaN
    assert {k["id"] for k in summary["kpis"]} >= {"p1_lin", "p2_gbm", "p2_proposed"}
    assert len(summary["groups"]) == 13
    manifest = json.loads((out_r / "manifest.json").read_text())
    assert manifest["row_counts"]["fact_soh"] == 6 + 4
    h = pd.read_csv(out_r / "fact_headline.csv")
    assert set(h["protocol"]) == {"Continual stream (P1)", "Unseen usage condition (P2)"}
    m = pd.read_csv(out_r / "dim_method.csv")
    assert m.loc[m["method"] == "tft_residual_ewc_shrunk", "is_proposed"].item()


def test_main_without_data_writes_results_only(results_dir: Path, tmp_path: Path):
    out_r, out_d = tmp_path / "out_r", tmp_path / "out_d"
    assert X.main(["--results", str(results_dir), "--data-root", str(tmp_path / "empty"), "--out-results", str(out_r),
                   "--out-data", str(out_d)]) == 0
    assert (out_r / "summary.json").exists() and not out_d.exists()
