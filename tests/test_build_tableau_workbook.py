"""Tests of the Tableau kit (scripts/build_tableau_workbook.py).

SYNTHETIC TEST DATA ONLY: inputs are written by the dashboard export from synthetic results files in a pytest
temporary directory; nothing is written into the repository or used for reported results (AGENTS.md).
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

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
    assert soh["valid"].all()   # invalid checks (no label) are not drawn


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
    xl = pd.ExcelFile(kit / "tableau_source.xlsx")
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
    # Each Excel source points at the written workbook file.
    files = {c.get("filename") for c in root.iter("connection") if c.get("class") == "excel-direct"}
    assert files == {str((kit / "tableau_source.xlsx").resolve())}
    assert (kit / "AUTHORING.md").exists()
