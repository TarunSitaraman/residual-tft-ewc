"""Tests of the Tableau kit (scripts/build_tableau_workbook.py).

SYNTHETIC TEST DATA ONLY: inputs are written by the dashboard export from synthetic results files in a pytest
temporary directory; nothing is written into the repository or used for reported results (AGENTS.md). The D38
tests also read the committed D38 results files (read only, copied into the temporary directory) to check the
export against the run log in docs/DECISIONS.md.
"""

from __future__ import annotations

import json
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import build_tableau_workbook as T  # noqa: E402
import export_dashboard_data as X  # noqa: E402
from test_export_dashboard_data import data_dir, results_dir  # noqa: E402,F401  (fixtures)

CONTRACT = {"dataset": "Dataset", "setting": "Setting", "protocol": "Protocol", "group": "Group",
            "family": "Method Family"}


@pytest.fixture()
def exported(results_dir: Path, data_dir: Path, tmp_path: Path) -> tuple[Path, Path]:  # noqa: F811
    out_r, out_d = tmp_path / "out_r", tmp_path / "out_d"
    assert X.main(["--results", str(results_dir), "--data-root", str(data_dir), "--out-results", str(out_r),
                   "--out-data", str(out_d)]) == 0
    return out_r, out_d


def test_contract_matches_config_js():
    assert T.contract_fields() == CONTRACT


def test_sheets_use_contract_names_and_join_dims(exported):
    out_r, out_d = exported
    tables = T.sheets(out_r, out_d, CONTRACT)
    assert {"Headline", "GroupError", "SOH", "Predictions"} <= set(tables)
    ge = tables["GroupError"]
    assert {"Dataset", "Setting", "Protocol", "Group", "Method Family", "Method Label", "Group Order"} <= set(ge)
    methods = pd.read_csv(out_r / "dim_method.csv").set_index("method")["family"]
    assert (ge["Method Family"] == ge["method"].map(methods)).all()
    # Headline settings are codes (the page filters by e1/e2), with a display label beside them.
    assert set(tables["Headline"]["Setting"]) <= {"e1", "e2"}
    soh = tables["SOH"]
    assert set(soh["Dataset"]) == {"S1", "S6"} and soh.groupby("unit")["Dataset"].nunique().max() == 1
    # Invalid checks are kept (drawn as hollow points, off the line) with `valid` as a real boolean.
    assert soh["valid"].dtype == bool
    assert {"D30Metrics", "StreamRuns", "Verdicts"} <= set(tables)


def test_missing_data_folder_gives_placeholders(results_dir: Path, tmp_path: Path):  # noqa: F811
    out_r = tmp_path / "out_r"
    X.main(["--results", str(results_dir), "--data-root", str(tmp_path / "none"), "--out-results", str(out_r),
            "--out-data", str(tmp_path / "none_out")])
    tables = T.sheets(out_r, tmp_path / "none_out", CONTRACT)
    assert tables["SOH"].empty and "Group" in tables["SOH"].columns
    assert tables["Predictions"].empty and "Group" in tables["Predictions"].columns


def test_main_writes_kit_and_twb_contract(exported, tmp_path: Path, monkeypatch):
    out_r, out_d = exported
    data_root = out_d.parent / "data_root"
    (data_root / "dashboard").mkdir(parents=True)
    for p in out_d.glob("*.csv"):
        (data_root / "dashboard" / p.name).write_bytes(p.read_bytes())
    kit = tmp_path / "kit"
    assert T.main(["--results-out", str(out_r), "--data-root", str(data_root), "--out", str(kit)]) == 0
    xl = pd.ExcelFile(kit / T.XLSX_REL)
    assert {"Headline", "GroupError", "StreamTasks", "SOH", "Predictions"} <= set(xl.sheet_names)
    root = ET.parse(kit / "residual_tft_ewc.twb").getroot()
    assert [d.get("name") for d in root.iter("dashboard")] == list(T.DASHBOARDS)
    sheets = {w.get("name"): w for w in root.iter("worksheet")}
    # Every sheet placed on a dashboard exists, and every dashboard zone names a real sheet.
    for d in root.iter("dashboard"):
        names = [z.get("name") for z in d.iter("zone") if z.get("name")]
        assert names and set(names) <= set(sheets)
    # Per-group views put Group in the view (a selected mark carries it); method-level views do not split by it.
    per_group = {v["name"] for v in T._views(CONTRACT) if v.get("per_group")}
    assert per_group >= {"Error heatmap", "SOH trajectories", "Interval coverage"}
    for name, w in sheets.items():
        text = ET.tostring(w, encoding="unicode")
        assert ("[none:Group:nk]" in text) == (name in per_group), name
    # Every data source is an Excel sheet with an enabled extract of its own, as Tableau saves one: Tableau Public
    # publishes extracts only (error 3C242D89) and its server rejects direct .hyper connections. Paths are relative,
    # so they resolve both beside the .twb and inside the .twbx.
    used = {d.get("caption") for d in root.find("datasources") if d.get("name") != "Parameters"}
    for d in root.find("datasources"):
        if d.get("name") == "Parameters":
            continue
        live = d.find("connection/named-connections/named-connection/connection")
        assert live.get("class") == "excel-direct" and live.get("filename") == T.XLSX_REL
        assert list(d)[-1].tag == "extract" and d.find("extract").get("enabled") == "true"
        assert d.find("extract/connection").get("dbname") == T.hyper_rel(d.get("caption"))
    with zipfile.ZipFile(kit / "residual_tft_ewc.twbx") as z:
        assert set(z.namelist()) == {"residual_tft_ewc.twb", T.XLSX_REL} | {T.hyper_rel(s) for s in used}
    # Each extract holds all the rows of its sheet.
    from tableauhyperapi import Connection, HyperProcess, TableName, Telemetry
    with HyperProcess(Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU, parameters={"log_dir": str(tmp_path)}) as hp:
        for name in used:
            with Connection(hp.endpoint, str(kit / T.hyper_rel(name))) as con:
                n = con.execute_scalar_query(f"SELECT COUNT(*) FROM {TableName('Extract', 'Extract')}")
                assert n == len(xl.parse(name)), name
    assert (kit / "AUTHORING.md").exists()
    # Elements Tableau's schema rejected when the workbook was first opened (error D2E8DA72).
    assert root.get("source-build")
    assert not list(root.iter("simple-id")) and not list(root.iter("computed-sort"))
    assert all(w.find("cards") is not None or w.find("viewpoints") is not None for w in root.iter("window"))
    # Each dashboard window has a viewpoint for every sheet it places (error 2805CF18 otherwise).
    for d in root.iter("dashboard"):
        w = next(w for w in root.iter("window") if w.get("name") == d.get("name"))
        assert {v.get("name") for v in w.iter("viewpoint")} == {z.get("name") for z in d.iter("zone") if z.get("name")}
    # Text tables fit the dashboard width; on Overview the heatmap is taller than each table.
    text = {v["name"] for v in T._views(CONTRACT) if v["mark"] == "Text"}
    for vp in root.iter("viewpoint"):
        assert vp.find("zoom").get("type") == ("fit-width" if vp.get("name") in text else "entire-view")
    ov = {z.get("name"): int(z.get("h")) for d in root.iter("dashboard") if d.get("name") == "Overview"
          for z in d.iter("zone") if z.get("name")}
    assert ov["Error heatmap"] > ov["Headline table"]
    # Every spec view has a worksheet (WORKBOOK_SPEC.md §3).
    assert {"Verdict strip", "D30 metric per group", "Seed spread", "Within one class"} <= set(sheets)
    # The D30 view picks its metric with a parameter, shown on the Forecast accuracy dashboard.
    params = [d for d in root.iter("datasource") if d.get("name") == "Parameters" and d.get("hasconnection")]
    assert len(params) == 1 and params[0].find("column/members") is not None
    # Its sheet lists the data source before Parameters (else Tableau drops it: "no valid data source"),
    # and depends on the full parameter definition.
    d30_view = sheets["D30 metric per group"].find("table/view")
    assert [d.get("name") for d in d30_view.find("datasources")] == [T._ds_name("D30Metrics"), "Parameters"]
    pdep = next(d for d in d30_view.findall("datasource-dependencies") if d.get("datasource") == "Parameters")
    assert pdep.find("column/members") is not None and pdep.find("column/calculation") is not None
    fa = next(d for d in root.iter("dashboard") if d.get("name") == "Forecast accuracy")
    assert any(z.get("type-v2") == "paramctrl" for z in fa.iter("zone"))
    # Calculated fields used by a worksheet carry their formula in its dependencies.
    soh_ws = ET.tostring(sheets["SOH trajectories"], encoding="unicode")
    assert "IF [valid] THEN [soh_pct] END" in soh_ws


# ---------- D38 (S10, S7): the committed results files, read only; nothing is written into the repository ----------
D38_FILES = [f"forecasting_d38_{ds}_{e}.json" for ds in ("s10", "s7") for e in ("e1", "e2")]


@pytest.fixture()
def results_with_d38(results_dir: Path) -> Path:  # noqa: F811
    for name in D38_FILES:
        (results_dir / name).write_bytes((ROOT / "results" / name).read_bytes())
    return results_dir


def _d38_row(vt: pd.DataFrame, ds: str, e: str, q: str, s: str = "main") -> pd.Series:
    return vt[(vt["dataset"] == ds) & (vt["experiment"] == e) & (vt["question"] == q) & (vt["seed_set"] == s)].iloc[0]


def test_d38_export_schema_and_run_log_values():
    vt, mt = X.d38_tables(ROOT / "results")
    assert list(vt.columns) == ["dataset", "experiment", "seed_set", "question", "comparison", "mean_diff_pp",
                                "median_diff_pp", "n_favour_v2", "n_conditions", "p_holm", "verdict", "answer",
                                "coverage_90"]
    assert list(mt.columns) == ["dataset", "experiment", "seed_set", "method", "mean_rmse_pp", "median_rmse_pp",
                                "coverage_90", "coverage_80"]
    # 2 datasets x 2 experiments x 2 seed sets x 5 questions.
    assert len(vt) == 40 and set(vt["dataset"]) == {"s10", "s7"} and set(vt["experiment"]) == {"E1", "E2"}
    # docs/DECISIONS.md, D38 run log, 11 Nov 2026 (S10): Q1 E1 -0.606, median -0.029, 42 of 64, p_Holm 0.003.
    r = _d38_row(vt, "s10", "E1", "Q1")
    assert r["comparison"] == "V2 vs GBM-ALL" and round(r["mean_diff_pp"], 3) == -0.606
    assert round(r["median_diff_pp"], 3) == -0.029 and r["n_favour_v2"] == 42 and r["n_conditions"] == 64
    assert round(r["p_holm"], 3) == 0.003 and r["verdict"] == "replicated, a better" and r["answer"] == "positive"
    # 13 Nov 2026 (S7): Q1 E2 +0.773, median +0.331, 9 of 60, p_Holm 5e-6, GBM-ALL better.
    r = _d38_row(vt, "s7", "E2", "Q1")
    assert round(r["mean_diff_pp"], 3) == 0.773 and round(r["median_diff_pp"], 3) == 0.331
    assert r["n_favour_v2"] == 9 and r["n_conditions"] == 60 and float(f"{r['p_holm']:.0e}") == 5e-6
    assert r["verdict"] == "replicated, a worse" and r["answer"] == "negative"
    # Registered answers, one per dataset and question: S10 all positive; S7 Q1 and Q2 negative, Q3-Q5 positive.
    ans = vt.groupby(["dataset", "question"])["answer"].agg(lambda a: sorted(set(a)))
    assert all(len(a) == 1 for a in ans)
    assert {q: a[0] for (ds, q), a in ans.items() if ds == "s10"} == dict.fromkeys(("Q1", "Q2", "Q3", "Q4", "Q5"),
                                                                                   "positive")
    assert {q: a[0] for (ds, q), a in ans.items() if ds == "s7"} == {
        "Q1": "negative", "Q2": "negative", "Q3": "positive", "Q4": "positive", "Q5": "positive"}
    # Q4 compares with each dataset's earlier proposed model.
    assert _d38_row(vt, "s10", "E1", "Q4")["comparison"] == "V2 vs SEL (D36 model)"
    assert _d38_row(vt, "s7", "E1", "Q4")["comparison"] == "V2 vs shrunk TFT-EWC (D34 model)"
    # Q5: coverage only (0.911 / 0.903 on S10 E1); Q1-Q4: no coverage.
    q5 = vt[vt["question"] == "Q5"]
    assert q5[["mean_diff_pp", "median_diff_pp", "n_favour_v2", "n_conditions", "p_holm", "verdict"]].isna().all().all()
    assert round(_d38_row(vt, "s10", "E1", "Q5")["coverage_90"], 3) == 0.911
    assert round(_d38_row(vt, "s10", "E1", "Q5", "replicate")["coverage_90"], 3) == 0.903
    assert vt.loc[vt["question"] != "Q5", "coverage_90"].isna().all()
    # Method table: V2 S10 E1 main mean 0.782, median 0.333, coverage 0.911 / 0.820; S7 E1 GBM-ALL 3.845, no coverage.
    m = mt.set_index(["dataset", "experiment", "seed_set", "method"])
    v2 = m.loc[("s10", "E1", "main", "v2")]
    assert [round(v2[c], 3) for c in ("mean_rmse_pp", "median_rmse_pp", "coverage_90", "coverage_80")] == \
        [0.782, 0.333, 0.911, 0.820]
    g = m.loc[("s7", "E1", "main", "gbm_all")]
    assert round(g["mean_rmse_pp"], 3) == 3.845 and np.isnan(g["coverage_90"]) and np.isnan(g["coverage_80"])
    # Short ids agreed with the web dashboard; every id has a readable label for the workbook.
    assert set(mt["method"]) == set(X.D38_IDS.values()) == set(X.D38_LABELS)
    assert {"v2", "sel", "shrunk", "gbm_all", "gbm_er32", "gbm_er8", "gbm_last", "linear", "ridge", "persistence",
            "joint", "r32_ens", "r32_single", "r8_ens_cal", "a0_single", "a0_ens"} <= set(mt["method"])


def test_d38_export_writes_csvs_and_counts(results_with_d38: Path, tmp_path: Path):
    out_r = tmp_path / "out_r"
    assert X.main(["--results", str(results_with_d38), "--data-root", str(tmp_path / "none"),
                   "--out-results", str(out_r), "--out-data", str(tmp_path / "none_out")]) == 0
    v, m = pd.read_csv(out_r / "fact_d38_verdicts.csv"), pd.read_csv(out_r / "fact_d38_method.csv")
    assert tuple(v.columns) == X.D38_VERDICT_COLS and tuple(m.columns) == X.D38_METHOD_COLS
    manifest = json.loads((out_r / "manifest.json").read_text())
    summary = json.loads((out_r / "summary.json").read_text())
    for name, df in (("fact_d38_verdicts", v), ("fact_d38_method", m)):
        assert manifest["row_counts"][name] == summary["counts"][name] == len(df)
    assert set(D38_FILES) <= {Path(p).name for p in manifest["sources"]}
    # dim_method is unchanged by D38 (no D38-only ids such as v2 are added to it).
    assert "v2" not in set(pd.read_csv(out_r / "dim_method.csv")["method"])


def test_d38_sheets_placeholder_without_results(exported):
    tables = T.sheets(*exported, CONTRACT)
    assert tables["D38Verdicts"].empty and {"Dataset", "answer", "verdict"} <= set(tables["D38Verdicts"])
    assert tables["D38Methods"].empty and "mean_rmse_pp" in tables["D38Methods"]


def test_d38_sources_sheets_and_dashboard(results_with_d38: Path, tmp_path: Path):
    out_r = tmp_path / "out_r"
    X.main(["--results", str(results_with_d38), "--data-root", str(tmp_path / "none"), "--out-results", str(out_r),
            "--out-data", str(tmp_path / "none_out")])
    tables = T.sheets(out_r, tmp_path / "none" / "dashboard", CONTRACT)
    assert set(tables["D38Verdicts"]["Dataset"]) == set(tables["D38Methods"]["Dataset"]) == {"S10", "S7"}
    kit = tmp_path / "kit"
    assert T.main(["--results-out", str(out_r), "--data-root", str(tmp_path / "none"), "--out", str(kit)]) == 0
    root = ET.parse(kit / "residual_tft_ewc.twb").getroot()
    # Each D38 source is an Excel sheet with its own enabled extract (a direct .hyper source is "not an extract").
    dss = {d.get("caption"): d for d in root.find("datasources") if d.get("name") != "Parameters"}
    for name in ("D38Verdicts", "D38Methods"):
        d = dss[name]
        assert d.find("connection/named-connections/named-connection/connection").get("class") == "excel-direct"
        assert list(d)[-1].tag == "extract" and d.find("extract").get("enabled") == "true"
        assert d.find("extract/connection").get("dbname") == T.hyper_rel(name)
    with zipfile.ZipFile(kit / "residual_tft_ewc.twbx") as z:
        assert {T.hyper_rel("D38Verdicts"), T.hyper_rel("D38Methods")} <= set(z.namelist())
    sheets = {w.get("name"): w for w in root.iter("worksheet")}
    vw, mw = sheets["D38 verdicts by question"], sheets["D38 RMSE by method"]
    for w, ds in ((vw, "D38Verdicts"), (mw, "D38Methods")):
        # The sheet's data source is listed first (no Parameters here), and the sheet has a title.
        assert [d.get("name") for d in w.find("table/view/datasources")] == [T._ds_name(ds)]
        assert w.find("layout-options/title/formatted-text/run").text
    # Verdicts: dataset and experiment as columns, questions as rows, colour by the registered answer.
    cols, rows = vw.find("table/cols").text, vw.find("table/rows").text
    assert "[none:Dataset:nk]" in cols and "[none:experiment:nk]" in cols and "[none:question:nk]" in rows
    assert vw.find("table/panes/pane/encodings/color").get("column").endswith("[none:answer:nk]")
    assert "ISNULL([mean_diff_pp])" in ET.tostring(vw, encoding="unicode")
    assert "[avg:mean_rmse_pp:qk]" in mw.find("table/cols").text
    assert "[none:Method Label:nk]" in mw.find("table/rows").text
    dm = tables["D38Methods"]
    assert (dm.loc[dm["method"] == "v2", "Method Label"] == X.D38_LABELS["v2"]).all()
    assert (dm["Method Label"] != dm["method"]).all()   # every id has a readable label
    # The dashboard places both sheets, each under a plain-language caption, with the S7/S10 attribution.
    db = next(d for d in root.iter("dashboard") if d.get("name") == "D38 replay ensemble")
    assert [z.get("name") for z in db.iter("zone") if z.get("name")] == ["D38 verdicts by question",
                                                                         "D38 RMSE by method"]
    texts = [r.text for z in db.iter("zone") if z.get("type-v2") == "text" for r in z.iter("run")]
    assert any("Lower RMSE is better" in t for t in texts)
    assert any("Q1-Q4 compare" in t and "positive = V2 better" in t for t in texts)
    assert any("S7" in t and "S10" in t and "not a confirmatory" in t for t in texts)
    assert sum(int(z.get("h")) for z in db.find("zones/zone")) <= 100000
    w = next(w for w in root.iter("window") if w.get("name") == "D38 replay ensemble")
    assert {v.get("name") for v in w.iter("viewpoint")} == {"D38 verdicts by question", "D38 RMSE by method"}
    from tableauhyperapi import Connection, HyperProcess, TableName, Telemetry
    with HyperProcess(Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU, parameters={"log_dir": str(tmp_path)}) as hp:
        for name in ("D38Verdicts", "D38Methods"):
            with Connection(hp.endpoint, str(kit / T.hyper_rel(name))) as con:
                n = con.execute_scalar_query(f"SELECT COUNT(*) FROM {TableName('Extract', 'Extract')}")
                assert n == len(tables[name]), name
