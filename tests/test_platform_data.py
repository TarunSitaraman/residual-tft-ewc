import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import platform_data as P  # noqa: E402

DF = pd.DataFrame({"cell": ["cell1a", "cell1b", "cell2a", None], "soh": [0.95, 0.81, 0.70, float("nan")], "n": [1, 2, 3, 4]})


def test_filter_comparisons_and_text():
    assert P.apply_filter(DF, "soh", ">0.8")["cell"].tolist() == ["cell1a", "cell1b"]
    assert P.apply_filter(DF, "n", "<= 2")["n"].tolist() == [1, 2]
    assert P.apply_filter(DF, "cell", "=CELL2A")["n"].tolist() == [3]
    assert P.apply_filter(DF, "cell", "1")["n"].tolist() == [1, 2]
    assert P.apply_filter(DF, "", "cell1")["n"].tolist() == [1, 2]
    assert P.apply_filter(DF, "", "")["n"].tolist() == [1, 2, 3, 4]


def test_table_page_sorts_pages_and_returns_json_safe_cells(tmp_path):
    f = tmp_path / "t.csv"
    DF.to_csv(f, index=False)
    p = P.table_page(f, offset=1, limit=2, sort="soh", desc=True)
    assert p["total"] == 4 and p["matched"] == 4
    assert [r[0] for r in p["rows"]] == ["cell1b", "cell2a"]
    last = P.table_page(f, offset=3, limit=5, sort="soh")["rows"][0]
    assert last[0] is None and last[1] is None  # missing values come last, as null


def test_only_catalogue_files_are_allowed(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "ok.csv").write_text("a\n1\n")
    (tmp_path / "secret.csv").write_text("a\n1\n")
    assert P.allowed(tmp_path, "data/ok.csv", "tables") == tmp_path / "data" / "ok.csv"
    assert P.allowed(tmp_path, "secret.csv", "tables") is None
    assert P.allowed(tmp_path, "data/../secret.csv", "tables") is None
    assert P.allowed(tmp_path, "", "tables") is None
