"""Preprocessing results table (work plan WS2.1): what each pipeline stage keeps, per dataset, with a source per number.

Every number is read from a committed file (download manifests in docs/external_data/metadata/, results/*.json);
nothing is typed by hand. When the git-ignored data tables are present (``data/`` or ``DATA_ROOT``), per-dataset
counts of the pretraining pool are added from them as well. Writes ``results/preprocessing_funnel.json``,
``results/tables/preprocessing.csv`` and ``paper/tables/preprocessing.tex``.

Usage (from the repository root)::

    python scripts/make_preprocessing_table.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import experiments as X  # noqa: E402

META = ROOT / "docs" / "external_data" / "metadata"
RES = ROOT / "results"


def j(path: Path) -> dict:
    return json.loads(path.read_text())


def gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


def rows() -> list[dict]:
    out = []

    def add(dataset, stage, value, unit, source):
        out.append({"dataset": dataset, "stage": stage, "value": value, "unit": unit, "source": source})

    # S1: Stanford second-life (raw RPT and EIS files)
    m = j(META / "osf_8jnr5_v2_diagnostic_manifest.json")
    add("S1", "raw diagnostic files (RPT, EIS), verified", m["n_files"], "files", "osf_8jnr5_v2_diagnostic_manifest.json")
    add("S1", "raw size", gb(m["total_bytes"]), "", "osf_8jnr5_v2_diagnostic_manifest.json")
    d = j(RES / "diagnostic_features.json")
    add("S1", "RPTs with C/20 label and diagnostics", d["n_rows"], "(battery, RPT)", "results/diagnostic_features.json")
    add("S1", "diagnostic features per RPT", d["n_features"], "features", "results/diagnostic_features.json")
    f = j(RES / "forecasting.json")
    add("S1", "forecasting samples (h = 1-3)", f["n_samples"], "samples", "results/forecasting.json")

    # S6: Wheeler et al. 2025, checks
    m = j(META / "wheeler2025_characterisation_manifest.json")
    add("S6 checks", "check files read from the archive, verified", m["n_files"], "files",
        "wheeler2025_characterisation_manifest.json")
    add("S6 checks", "uncompressed size of the files read", gb(m["total_bytes"]), "", "wheeler2025_characterisation_manifest.json")
    w = j(RES / "forecasting_wheeler.json")
    add("S6 checks", "capacity checks parsed", w["n_checks"], "checks", "results/forecasting_wheeler.json")
    add("S6 checks", "valid (labelled) checks", w["n_valid_checks"], "checks", "results/forecasting_wheeler.json")
    add("S6 checks", "forecasting samples (h = 1-3)", w["n_samples"], "samples", "results/forecasting_wheeler.json")
    for exp, h in (("e2", "1-6"),):
        p = RES / f"forecasting_d23_{exp}.json"
        if p.exists():
            add("S6 checks", f"forecasting samples (h = {h})", j(p)["n_samples"], "samples", f"results/{p.name}")

    # S6: ageing cycles between checks (the Spark workload)
    c = j(RES / "s6_ageing_row_count.json")
    a = j(RES / "d22_ageing_report.json")
    s = j(RES / "d22_spark_validation.json")
    valid = sum(x["valid_cycles"] for x in a["completeness"])
    invalid = sum(x["invalid_cycles"] for x in a["completeness"])
    below = sum(x["blocks_below_min_valid"] for x in a["completeness"])
    add("S6 ageing", "raw ageing files (cells x 100-cycle blocks), verified", c["n_files"], "files",
        "results/s6_ageing_row_count.json")
    add("S6 ageing", "raw size (uncompressed)", gb(c["uncompressed_bytes"]), "", "results/s6_ageing_row_count.json")
    add("S6 ageing", "raw data rows (about 1 Hz)", c["data_rows"], "rows", "results/s6_ageing_row_count.json")
    add("S6 ageing", "ageing cycles parsed (Spark map)", s["cycles"], "cycles", "results/d22_spark_validation.json")
    add("S6 ageing", "valid cycles", valid, "cycles", "results/d22_ageing_report.json")
    add("S6 ageing", "invalid cycles (excluded)", invalid, "cycles", "results/d22_ageing_report.json")
    add("S6 ageing", "block feature rows (Spark SQL reduce)", s["blocks"], "blocks", "results/d22_spark_validation.json")
    add("S6 ageing", "blocks below 50 valid cycles (inputs missing)", below, "blocks", "results/d22_ageing_report.json")
    add("S6 ageing", "ageing inputs per block", len(s["per_input"]) - 2, "inputs", "results/d22_spark_validation.json")
    add("S6 ageing", "Spark run time, full set", f"{s['seconds'] / 60:.1f} min", s["engine"],
        "results/d22_spark_validation.json")
    add("S6 ageing", "Spark output equals pandas output", "yes" if s["matches_pandas"] else "NO", s["criterion"],
        "results/d22_spark_validation.json")
    add("S6 ageing", "reduction, raw rows per block feature row", round(c["data_rows"] / s["blocks"]), "rows",
        "derived from the two lines above")

    # Pretraining pool S7-S9
    pm = j(META / "pool_downloads_manifest.json")["files"]
    p3 = j(RES / "forecasting_d21_p3.json")
    for ds in ("S7", "S8", "S9"):
        files = [x for x in pm if x["id"] == ds]
        if files:
            add(f"Pool {ds}", "downloaded archives, verified", len(files), "archives", "pool_downloads_manifest.json")
            add(f"Pool {ds}", "download size", gb(sum(x["size"] for x in files)), "", "pool_downloads_manifest.json")
    data = Path(os.environ.get("DATA_ROOT", ROOT / "data"))
    data = data if data.is_absolute() else ROOT / data
    for ds in ("S7", "S8", "S9"):
        t = data / f"pool_{ds.lower()}_checks.csv"
        if t.exists():
            df = pd.read_csv(t)
            add(f"Pool {ds}", "cells with checks", df["cell"].nunique(), "cells", f"data/{t.name} (git-ignored)")
            add(f"Pool {ds}", "capacity checks", len(df), "checks", f"data/{t.name} (git-ignored)")
    add("Pool S7-S9", "cells used (>= 5 valid checks)", p3["n_samples"]["pool_cells"], "cells",
        "results/forecasting_d21_p3.json")
    add("Pool S7-S9", "forecasting samples", p3["n_samples"]["pool"], "samples", "results/forecasting_d21_p3.json")
    for k, v in p3["pool_split_sizes"].items():
        add("Pool S7-S9", f"  {k} split", v, "samples", "results/forecasting_d21_p3.json")
    return out


def main() -> int:
    r = rows()
    X.write_results(RES / "preprocessing_funnel.json", {"work_plan": "WS2.1", "rows": r}, ROOT)
    df = pd.DataFrame(r)
    (RES / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(RES / "tables" / "preprocessing.csv", index=False)
    esc = lambda s: str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")  # noqa: E731
    fmtv = lambda v: f"{v:,}" if isinstance(v, int) else str(v)  # noqa: E731
    lines = [r"\begin{table}[t]", r"  \centering",
             r"  \caption{Data preprocessing results per dataset and stage (every number read from a committed "
             r"manifest or results file; see \texttt{results/preprocessing\_funnel.json}).}",
             r"  \label{tab:preprocessing}", r"  \scriptsize", r"  \setlength{\tabcolsep}{3pt}",
             r"  \begin{tabular}{llr}", r"    \toprule", r"    Dataset & Stage & Value \\", r"    \midrule"]
    prev = None
    for x in r:
        if x["unit"].startswith(("Apache", "|spark")):
            unit = ""
        else:
            unit = f" {x['unit']}" if x["unit"] and not str(x["value"]).endswith(("GB", "min")) else ""
        lines.append(f"    {esc(x['dataset']) if x['dataset'] != prev else ''} & {esc(x['stage'])} & "
                     f"{esc(fmtv(x['value']) + unit)} \\\\")
        prev = x["dataset"]
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    out = ROOT / "paper" / "tables" / "preprocessing.tex"
    out.write_text("\n".join(lines))
    print(df[["dataset", "stage", "value", "unit"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
