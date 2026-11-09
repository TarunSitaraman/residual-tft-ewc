"""Build the Tableau kit for the results dashboard: a Tableau-ready Excel source and a starter workbook (.twb).

Reads the tables written by ``scripts/export_dashboard_data.py`` (``results/dashboard/`` and, when present,
``<data>/dashboard/``) and writes to ``<data>/dashboard/tableau/`` (git-ignored, since it can hold rows built from
the data folder):

- ``tableau_source.xlsx``: one sheet per view. Each sheet is self-contained (method label, family and kind, group
  order and description joined in) and already filtered to what its view shows, so the workbook needs no joins,
  relationships or data-source filters. The filter fields carry the names the JS page drives
  (``dashboard/config.js`` FIELDS): Dataset, Setting, Protocol, Group, Method Family.
- ``Data/tableau_source.hyper``: the same tables as a Tableau extract (Tableau Public saves extracts only).
- ``residual_tft_ewc.twb`` and the packaged ``residual_tft_ewc.twbx``: a STARTER workbook. One extract data source
  per sheet, the calculated fields and aliases
  of dashboard/tableau/WORKBOOK_SPEC.md, one worksheet per view and the five dashboards named as the page expects.
  It is generated XML that cannot be opened in a Linux container: open it in Tableau Desktop Public Edition, check
  every view against the spec, fix what Tableau reports, then save it to Tableau Public.
- ``AUTHORING.md``: those steps.

Nothing is recomputed: rows are filtered, joined and renamed only.

    python scripts/build_tableau_workbook.py                     # data folder: $DATA_ROOT or data/
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import export_dashboard_data as X  # noqa: E402

EXCEL_MAX_ROWS = 1_048_575   # data rows below the header
CONFIG_JS = ROOT / "dashboard" / "config.js"
DASHBOARDS = ("Overview", "SOH explorer", "Forecast accuracy", "Continual learning", "Reuse & uncertainty")
ATTRIBUTION = ("SOH is relative to each cell's first valid capacity check. Data: Moy et al. 2024 (S1, CC BY 4.0); "
               "Wheeler et al. 2025 (S6, CC BY 4.0). Changes listed in docs/EXTERNAL_DATA.md.")


def contract_fields(config_js: Path = CONFIG_JS) -> dict[str, str]:
    """The FIELDS map of dashboard/config.js (filter key -> Tableau field name)."""
    text = config_js.read_text(encoding="utf-8")
    block = re.search(r"export const FIELDS = \{(.*?)\};", text, re.S).group(1)
    return dict(re.findall(r"(\w+):\s*\"([^\"]+)\"", block))


# ---------- the Excel source ----------
def _read(folder: Path, name: str) -> pd.DataFrame:
    p = folder / f"{name}.csv"
    return pd.read_csv(p, low_memory=False) if p.exists() else pd.DataFrame()


def _with_dims(df: pd.DataFrame, methods: pd.DataFrame, groups: pd.DataFrame, f: dict[str, str]) -> pd.DataFrame:
    """Join method and group attributes and apply the contract names."""
    out = df.copy()
    if "method" in out and len(methods):
        m = methods.rename(columns={"label": "Method Label", "family": f["family"], "kind": "Method Kind",
                                    "is_proposed": "Is Proposed"})
        out = out.merge(m, on="method", how="left", validate="many_to_one")
        out["Method Label"] = out["Method Label"].fillna(out["method"])
    if "group" in out and len(groups):
        g = groups[["dataset", "group", "group_order", "description"]].rename(
            columns={"group_order": "Group Order", "description": "Group Description"})
        on = ["dataset", "group"] if "dataset" in out else ["group"]
        g = g if "dataset" in out else g.drop(columns="dataset").drop_duplicates("group")
        out = out.merge(g, on=on, how="left", validate="many_to_one")
    out = out.rename(columns={"dataset": f["dataset"], "setting": f["setting"], "protocol": f["protocol"],
                              "group": f["group"]})
    if f["setting"] in out:
        out["Setting Label"] = out[f["setting"]].map(X.SETTINGS)
    return out


def sheets(results_out: Path, data_out: Path, f: dict[str, str]) -> dict[str, pd.DataFrame]:
    """One self-contained, pre-filtered table per view (sheet name -> rows)."""
    R = lambda n: _read(results_out, n)  # noqa: E731
    methods, groups = R("dim_method"), R("dim_group")
    J = lambda df: _with_dims(df, methods, groups, f) if len(df) else df  # noqa: E731
    out: dict[str, pd.DataFrame] = {}

    head = R("fact_headline")
    if len(head):
        head = head.assign(setting=head["setting"].map({v: k for k, v in X.SETTINGS.items()}),
                           protocol=np.where(head["protocol"].str.startswith("Continual"), "P1", "P2"), dataset="S6")
        out["Headline"] = J(head[head["rmse_main"].notna()].drop(columns=["diff_main", "diff_replicate", "verdict", "post_hoc"]))
        tests = head[head["verdict"].notna()].drop(columns=["rmse_main", "rmse_replicate"])
        out["HeadlineTests"] = tests.rename(columns={"dataset": f["dataset"], "setting": f["setting"],
                                                     "protocol": f["protocol"]}).assign(
            **{"Setting Label": lambda d: d[f["setting"]].map(X.SETTINGS)})

    ge = X.per_group(R("fact_stream_task"), R("fact_stream_baseline"), R("fact_unseen_fold"))
    if len(ge):
        out["GroupError"] = J(ge.drop(columns="label").assign(dataset="S6"))

    hz = R("fact_horizon")
    if len(hz):
        out["Horizon"] = J(hz.drop(columns="label"))

    d30 = R("fact_d30_group")
    if len(d30):
        out["D30Metrics"] = J(d30)   # all metrics: the view picks one with the D30 metric parameter

    st = R("fact_stream_task")
    if len(st):
        st = st[(st["experiment"] == "D24") & (st["seed_set"] == "main")]
        out["StreamTasks"] = J(st)

    sr = R("fact_stream_run")
    if len(sr):
        out["StreamRuns"] = J(sr)   # both seed sets, every stream experiment (the view splits by experiment)

    cov = R("fact_coverage")
    if len(cov):
        out["Coverage90"] = J(cov[(cov["level"] == 0.9) & (cov["group"] != "all")])

    vd = R("fact_verdicts")
    if len(vd):
        out["Verdicts"] = vd.rename(columns={"setting": f["setting"]})

    soh = _read(data_out, "fact_soh")
    # Invalid checks stay in: the line skips them (SOH valid checks is null there) and the spec draws them as
    # hollow points. `valid` is written as a real boolean so the calculated field can test it.
    out["SOH"] = J(soh.assign(valid=soh["valid"].astype(str).str.lower().isin(("true", "1")))) if len(soh) else \
        pd.DataFrame(columns=[f["dataset"], f["group"], "unit", "check_index", "soh_pct", "valid", "reuse_class",
                              "temperature_source"])
    pr = _read(data_out, "fact_predictions")
    out["Predictions"] = J(pr) if len(pr) else pd.DataFrame(
        columns=[f["dataset"], f["group"], "method", "Method Label", f["family"], "horizon", "soh_target", "soh_pred",
                 "class_true", "class_pred", "class_distance"])
    # Every view keeps its sheet: inputs not built yet become empty placeholders with the fields the view uses.
    for v in _views(f):
        if v["sheet"] not in out or out[v["sheet"]].empty:
            have = list(out[v["sheet"]].columns) if v["sheet"] in out else []
            used = [r.field for r in v["rows"] + v["cols"] + list(v["enc"].values())]
            cols = list(dict.fromkeys(have + used + ([f["group"], "Group Order"] if v.get("sort_group") else [])))
            out[v["sheet"]] = pd.DataFrame(columns=cols)
    for name, df in out.items():
        if len(df) > EXCEL_MAX_ROWS:
            raise ValueError(f"sheet {name}: {len(df):,} rows exceed Excel's limit")
    return out


def write_xlsx(tables: dict[str, pd.DataFrame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in tables.items():
            df.to_excel(xw, sheet_name=name, index=False)


HYPER_REL = "Data/tableau_source.hyper"   # relative to the .twb, and the path inside the .twbx


def _sqltype(s: pd.Series):
    from tableauhyperapi import SqlType
    return {"boolean": SqlType.bool, "integer": SqlType.big_int, "real": SqlType.double}.get(_dtype(s), SqlType.text)()


def write_hyper(tables: dict[str, pd.DataFrame], path: Path) -> None:
    """The extract Tableau Public requires: one table per view in schema Extract, same columns and types."""
    from tableauhyperapi import (NULLABLE, Connection, CreateMode, HyperProcess, Inserter, TableDefinition,
                                 TableName, Telemetry)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Hyper writes hyperd.log to log_dir (default: the working directory, i.e. the repository root).
    with HyperProcess(Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU, parameters={"log_dir": tempfile.gettempdir()}) as hp, \
            Connection(hp.endpoint, str(path), CreateMode.CREATE_AND_REPLACE) as con:
        con.catalog.create_schema("Extract")
        for name, df in tables.items():
            td = TableDefinition(TableName("Extract", name),
                                 [TableDefinition.Column(str(c), _sqltype(df[c]), NULLABLE) for c in df.columns])
            con.catalog.create_table(td)
            text = [_dtype(df[c]) == "string" for c in df.columns]
            with Inserter(con, td) as ins:
                ins.add_rows([None if pd.isna(v) else str(v) if t else v.item() if hasattr(v, "item") else v
                              for v, t in zip(row, text)] for row in df.itertuples(index=False))
                ins.execute()


def write_twbx(twb: Path, hyper: Path, path: Path) -> None:
    """Packaged workbook: the .twb plus its extract, opened directly in Tableau (File > Open)."""
    import zipfile
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(twb, twb.name)
        z.write(hyper, HYPER_REL)


# ---------- the starter workbook ----------
def _dtype(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_integer_dtype(s):
        return "integer"
    if pd.api.types.is_float_dtype(s):
        return "real"
    return "string"


def _ds_name(sheet: str) -> str:
    return f"federated.{sheet.lower()}"


class Ref:
    """A field used in a worksheet: column instance name and its declaration."""

    def __init__(self, field: str, datatype: str, agg: str | None = None, discrete: bool = True):
        self.field, self.datatype, self.agg, self.discrete = field, datatype, agg, discrete

    @property
    def instance(self) -> str:
        deriv = {"avg": "avg", "sum": "sum", "cnt": "cnt", None: "none"}[self.agg]
        kind = ("n" if self.datatype == "string" else "o") if self.discrete else "q"
        return f"[{deriv}:{self.field}:{kind}k]"

    def full(self, ds: str) -> str:
        return f"[{ds}].{self.instance}"


#: One dict per worksheet: name, source sheet, rows/cols (lists of Refs), mark, encodings {channel: Ref}.
#: ``per_group`` views draw one mark per battery group and carry Group, so a selected mark tells the page its group;
#: the other views aggregate over groups on purpose (method-level summaries) and must not split by group.
def _views(f: dict[str, str]) -> list[dict]:
    G, D, P, S = f["group"], f["dataset"], f["protocol"], f["setting"]
    s = lambda n: Ref(n, "string")  # noqa: E731
    i = lambda n: Ref(n, "integer")  # noqa: E731
    avg = lambda n: Ref(n, "real", "avg", discrete=False)  # noqa: E731
    return [
        {"name": "Headline table", "sheet": "Headline", "rows": [s("Method Label")], "cols": [s(P), s(S)],
         "mark": "Text", "enc": {"text": avg("rmse_main")}},
        {"name": "Verdicts vs baselines", "sheet": "HeadlineTests", "rows": [s("label")], "cols": [s(P), s(S)],
         "mark": "Text", "enc": {"text": s("verdict")}},
        {"name": "Verdict strip", "sheet": "Verdicts", "rows": [s("experiment")], "cols": [s("verdict")],
         "mark": "Text", "enc": {"text": Ref("verdict", "string", "cnt", discrete=False)}},
        {"name": "Error heatmap", "sheet": "GroupError", "rows": [s(P), s("Method Label")], "cols": [s(G)],
         "mark": "Square", "enc": {"color": avg("rmse_mean"), "text": avg("rmse_mean")}, "sort_group": True,
         "per_group": True},
        {"name": "Error by horizon", "sheet": "Horizon", "rows": [s(D), avg("rmse_pp")], "cols": [i("horizon")],
         "mark": "Line", "enc": {"color": s("Method Label")}},
        {"name": "D30 metric per group", "sheet": "D30Metrics", "rows": [s(D), s("Method Label")],
         "cols": [avg("D30 metric value")],
         "mark": "Bar", "enc": {"detail": s(G), "color": s(f["family"])}, "per_group": True},
        {"name": "SOH trajectories", "sheet": "SOH", "rows": [s(D), avg("SOH valid checks")],
         "cols": [Ref("check_index", "integer", None, discrete=False)], "mark": "Line",
         "enc": {"color": s(G), "detail": s("unit")}, "per_group": True},
        {"name": "Adaptation vs retention", "sheet": "StreamTasks", "rows": [s(S), avg("final_error")],
         "cols": [avg("diagonal_error")], "mark": "Circle", "enc": {"color": s(f["family"]), "detail": s("Method Label")}},
        {"name": "Forgetting by arrival", "sheet": "StreamTasks", "rows": [s(S), avg("forgetting")],
         "cols": [i("arrival_pos")], "mark": "Line", "enc": {"color": s("Method Label")}},
        {"name": "Backward transfer heatmap", "sheet": "StreamTasks", "rows": [s(S), s(G)], "cols": [i("arrival_pos")],
         "mark": "Square", "enc": {"color": avg("backward_transfer")}, "sort_group": True, "per_group": True},
        {"name": "Seed spread", "sheet": "StreamRuns",
         "rows": [s("experiment"), Ref("final_avg_error", "real", None, discrete=False)],
         "cols": [s("Method Label"), s("seed_set")], "mark": "Circle", "enc": {"color": s("seed_set")},
         "disaggregate": True},
        {"name": "Interval coverage", "sheet": "Coverage90", "rows": [s(S), avg("coverage")], "cols": [s(G)],
         "mark": "Circle", "enc": {"color": s("Method Label")}, "sort_group": True, "per_group": True},
        {"name": "Reuse-class confusion", "sheet": "Predictions", "rows": [s("class_true")], "cols": [s("class_pred")],
         "mark": "Square", "enc": {"text": Ref("soh_pred", "real", "cnt", discrete=False),
                                   "color": Ref("soh_pred", "real", "cnt", discrete=False)}},
        {"name": "Forecast parity", "sheet": "Predictions", "rows": [avg("soh_pred")], "cols": [avg("soh_target")],
         "mark": "Circle", "enc": {"detail": s(G), "color": s(f["family"])}, "disaggregate": True,
         "per_group": True},
        {"name": "Within one class", "sheet": "Predictions", "rows": [s("Method Label")],
         "cols": [avg("Within one class share")], "mark": "Bar", "enc": {"color": s(f["family"])}},
    ]


LAYOUT = {
    "Overview": ["Headline table", "Verdicts vs baselines", "Verdict strip", "Error heatmap"],
    "SOH explorer": ["SOH trajectories"],
    "Forecast accuracy": ["Error heatmap", "Error by horizon", "D30 metric per group"],
    "Continual learning": ["Adaptation vs retention", "Forgetting by arrival", "Backward transfer heatmap",
                           "Seed spread"],
    "Reuse & uncertainty": ["Reuse-class confusion", "Forecast parity", "Interval coverage", "Within one class"],
}
#: Tableau rejects a workbook without source-build; any recent Desktop build opens and re-stamps it on save.
SOURCE_BUILD = "2024.2.0 (20242.24.0613.1930)"
D30_PARAM = "[Parameters].[Parameter 1]"
PARAM_CONTROLS = {"Forecast accuracy": D30_PARAM}


def _calcs(sheet: str) -> list[tuple[str, str, str]]:
    """(caption, datatype, formula) of the calculated fields of a data source (WORKBOOK_SPEC.md §2)."""
    if sheet == "GroupError":
        return [("Skill vs persistence", "real",
                 "1 - AVG([rmse_mean]) / AVG({FIXED [Dataset], [Protocol], [Setting], [Group]: "
                 "AVG(IF [method] = 'persistence' THEN [rmse_mean] END)})")]
    if sheet == "StreamTasks":
        return [("Retention gap", "real", "[final_error] - [diagonal_error]")]
    if sheet == "Predictions":
        return [("Within one class", "boolean", "ABS([class_distance]) <= 1"),
                ("Within one class share", "real", "IIF(ABS([class_distance]) <= 1, 1.0, 0.0)")]
    if sheet == "SOH":
        return [("Reuse class band", "string", 'IF [soh_pct] > 95 THEN "C0" ELSEIF [soh_pct] > 90 THEN "C1" '
                                                'ELSEIF [soh_pct] > 80 THEN "C2" ELSE "C3" END'),
                ("SOH valid checks", "real", "IF [valid] THEN [soh_pct] END")]
    if sheet == "D30Metrics":
        return [("D30 metric value", "real", f"IF [metric] = {D30_PARAM} THEN [value] END")]
    return []


def _parameters(parent: ET.Element, metrics: list[str]) -> None:
    """The D30 metric parameter (WORKBOOK_SPEC.md §3.3), default reg_RMSE."""
    ds = ET.SubElement(parent, "datasource", hasconnection="false", inline="true", name="Parameters", version="18.1")
    default = "reg_RMSE" if "reg_RMSE" in metrics else metrics[0]
    col = ET.SubElement(ds, "column", {"caption": "D30 metric", "datatype": "string", "name": "[Parameter 1]",
                                       "param-domain-type": "list", "role": "measure", "type": "nominal",
                                       "value": f'"{default}"'})
    ET.SubElement(col, "calculation", {"class": "tableau", "formula": f'"{default}"'})
    members = ET.SubElement(col, "members")
    for m in metrics:
        ET.SubElement(members, "member", value=f'"{m}"')


def _datasource(parent: ET.Element, sheet: str, df: pd.DataFrame, f: dict[str, str]) -> None:
    """A data source on the bundled Hyper extract (Tableau Public saves extracts only)."""
    ds = ET.SubElement(parent, "datasource", caption=sheet, inline="true", name=_ds_name(sheet), version="18.1")
    conn = ET.SubElement(ds, "connection", {"class": "federated"})
    named = ET.SubElement(ET.SubElement(conn, "named-connections"), "named-connection",
                          caption="tableau_source", name="hyper.source")
    ET.SubElement(named, "connection", {"authentication": "auth-none", "author-locale": "en_US", "class": "hyper",
                                        "dbname": HYPER_REL, "default-settings": "yes", "schema": "Extract",
                                        "sslmode": "", "tablename": sheet, "username": "tableau_internal_user"})
    ET.SubElement(conn, "relation", connection="hyper.source", name=sheet, table=f"[Extract].[{sheet}]", type="table")
    ET.SubElement(ds, "aliases", enabled="yes")
    if f["setting"] in df:
        col = ET.SubElement(ds, "column", datatype="string", name=f"[{f['setting']}]", role="dimension", type="nominal")
        al = ET.SubElement(col, "aliases")
        for k, v in X.SETTINGS.items():
            ET.SubElement(al, "alias", key=f'"{k}"', value=v)
    for cap, dt, formula in _calcs(sheet):
        _calc_column(ds, cap, dt, formula)


def _calc_column(parent: ET.Element, cap: str, dt: str, formula: str) -> None:
    """A calculated field, named by its caption so worksheets refer to it like a data column."""
    role, typ = ("dimension", "nominal") if dt in ("boolean", "string") else ("measure", "quantitative")
    col = ET.SubElement(parent, "column", caption=cap, datatype=dt, name=f"[{cap}]", role=role, type=typ)
    ET.SubElement(col, "calculation", {"class": "tableau", "formula": formula})


def _shelf(refs: list[Ref], ds: str) -> str:
    parts = [r.full(ds) for r in refs]
    return parts[0] if len(parts) == 1 else "(" + " / ".join(parts) + ")"


def _worksheet(parent: ET.Element, v: dict, df: pd.DataFrame, f: dict[str, str]) -> None:
    ds = _ds_name(v["sheet"])
    ws = ET.SubElement(parent, "worksheet", name=v["name"])
    table = ET.SubElement(ws, "table")
    view = ET.SubElement(table, "view")
    calcs = {cap: (dt, formula) for cap, dt, formula in _calcs(v["sheet"])}
    refs = v["rows"] + v["cols"] + list(v["enc"].values())
    uses_param = any(D30_PARAM in calcs[r.field][1] for r in refs if r.field in calcs)
    dsl = ET.SubElement(view, "datasources")
    if uses_param:
        ET.SubElement(dsl, "datasource", name="Parameters")
    ET.SubElement(dsl, "datasource", caption=v["sheet"], name=ds)
    if uses_param:
        pdeps = ET.SubElement(view, "datasource-dependencies", datasource="Parameters")
        ET.SubElement(pdeps, "column", caption="D30 metric", datatype="string", name="[Parameter 1]",
                      role="measure", type="nominal")
    deps = ET.SubElement(view, "datasource-dependencies", datasource=ds)
    if v.get("sort_group"):
        refs.append(Ref("Group Order", "integer", "sum", discrete=False))
    seen = set()
    for r in refs:
        if r.field not in seen:
            seen.add(r.field)
            if r.field in calcs:
                _calc_column(deps, r.field, *calcs[r.field])
                continue
            dt = _dtype(df[r.field]) if r.field in df else r.datatype
            measure = dt in ("real", "integer") and r.agg is not None
            ET.SubElement(deps, "column", datatype=dt, name=f"[{r.field}]", role="measure" if measure else "dimension",
                          type="quantitative" if measure else "nominal")
    inst = set()
    for r in refs:
        if r.instance not in inst:
            inst.add(r.instance)
            ET.SubElement(deps, "column-instance", column=f"[{r.field}]",
                          derivation={"avg": "Avg", "sum": "Sum", "cnt": "Count", None: "None"}[r.agg],
                          name=r.instance, pivot="key", type="quantitative" if not r.discrete else
                          ("nominal" if r.datatype == "string" else "ordinal"))
    if v.get("sort_group"):
        g = Ref(f["group"], "string")
        ET.SubElement(view, "sort", {"class": "computed"}, column=g.full(ds), direction="ASC",
                      using=Ref("Group Order", "integer", "sum", discrete=False).full(ds))
    ET.SubElement(view, "aggregation", value="false" if v.get("disaggregate") else "true")
    ET.SubElement(table, "style")
    pane = ET.SubElement(ET.SubElement(table, "panes"), "pane", {"selection-relaxation-option": "selection-relaxation-allow"})
    ET.SubElement(ET.SubElement(pane, "view"), "breakdown", value="auto")
    ET.SubElement(pane, "mark", {"class": v["mark"]})
    enc = ET.SubElement(pane, "encodings")
    for channel, r in v["enc"].items():
        ET.SubElement(enc, "lod" if channel == "detail" else channel, column=r.full(ds))
    ET.SubElement(table, "rows").text = _shelf(v["rows"], ds)
    ET.SubElement(table, "cols").text = _shelf(v["cols"], ds)


def _dashboard(parent: ET.Element, name: str, sheets_: list[str], tables_: set[str]) -> None:
    db = ET.SubElement(parent, "dashboard", name=name)
    ET.SubElement(db, "style")
    ET.SubElement(db, "size", maxheight="1100", maxwidth="1400", minheight="700", minwidth="800")
    zones = ET.SubElement(db, "zones")
    outer = ET.SubElement(zones, "zone", {"h": "100000", "id": "1", "type-v2": "layout-flow", "param": "vert",
                                          "w": "100000", "x": "0", "y": "0"})
    # Charts get twice the height of text tables, which need only a few rows.
    weight = {s: 1 if s in tables_ else 2 for s in sheets_}
    unit, zid = (100000 - 6000) // max(sum(weight.values()), 1), 2
    title = ET.SubElement(outer, "zone", {"h": "6000", "id": str(zid), "type-v2": "text", "w": "100000", "x": "0", "y": "0"})
    run = ET.SubElement(ET.SubElement(title, "formatted-text"), "run")
    run.text = f"{name}. {ATTRIBUTION}"
    top = 6000
    for s in sheets_:
        zid += 1
        h = unit * weight[s]
        ET.SubElement(outer, "zone", {"h": str(h), "id": str(zid), "name": s, "w": "100000", "x": "0", "y": str(top)})
        top += h
    if name in PARAM_CONTROLS:   # floating control, top right, over the title band
        zid += 1
        ET.SubElement(zones, "zone", {"h": "6000", "id": str(zid), "mode": "compact", "param": PARAM_CONTROLS[name],
                                      "type-v2": "paramctrl", "w": "25000", "x": "75000", "y": "0"})


def _worksheet_window(parent: ET.Element, name: str) -> None:
    """The window of a worksheet: Tableau requires its card layout (shelves, marks, filters)."""
    w = ET.SubElement(parent, "window", {"class": "worksheet", "name": name})
    edges = ET.SubElement(w, "cards")
    left = ET.SubElement(ET.SubElement(edges, "edge", name="left"), "strip", size="160")
    for t in ("pages", "filters", "marks"):
        ET.SubElement(left, "card", type=t)
    top = ET.SubElement(edges, "edge", name="top")
    for t, size in (("columns", "2147483647"), ("rows", "2147483647"), ("title", "31")):
        ET.SubElement(ET.SubElement(top, "strip", size=size), "card", type=t)


def build_twb(tables: dict[str, pd.DataFrame], f: dict[str, str]) -> ET.ElementTree:
    wb = ET.Element("workbook", {"original-version": "18.1", "source-build": SOURCE_BUILD, "source-platform": "win",
                                 "version": "18.1",
                                 "xmlns:user": "http://www.tableausoftware.com/xml/user"})
    ET.SubElement(wb, "preferences")
    views = _views(f)
    dss = ET.SubElement(wb, "datasources")
    d30 = tables.get("D30Metrics", pd.DataFrame())
    _parameters(dss, sorted(d30["metric"].dropna().unique()) if "metric" in d30 and len(d30) else ["reg_RMSE"])
    for sheet in dict.fromkeys(v["sheet"] for v in views):
        _datasource(dss, sheet, tables[sheet], f)
    wss = ET.SubElement(wb, "worksheets")
    for v in views:
        _worksheet(wss, v, tables[v["sheet"]], f)
    dbs = ET.SubElement(wb, "dashboards")
    text_sheets = {v["name"] for v in views if v["mark"] == "Text"}
    for name in DASHBOARDS:
        _dashboard(dbs, name, LAYOUT[name], text_sheets)
    wins = ET.SubElement(wb, "windows", {"source-height": "30"})
    for v in views:
        _worksheet_window(wins, v["name"])
    for name in DASHBOARDS:
        w = ET.SubElement(wins, "window", {"class": "dashboard", "maximized": "true" if name == "Overview" else "false",
                                           "name": name})
        vps = ET.SubElement(w, "viewpoints")   # one per placed sheet: the dashboard's view of it
        for sheet in LAYOUT[name]:
            # Text tables fit the width (entire view leaves their right-hand columns cut off).
            ET.SubElement(ET.SubElement(vps, "viewpoint", name=sheet), "zoom",
                          type="fit-width" if sheet in text_sheets else "entire-view")
        ET.SubElement(w, "active", id="-1")
        ET.SubElement(w, "device-preview")
    tree = ET.ElementTree(wb)
    ET.indent(tree)
    return tree


AUTHORING = """# Tableau authoring steps (on Windows or macOS)

Files in this folder: `residual_tft_ewc.twbx` (open this: the workbook with its extract packaged in),
`residual_tft_ewc.twb` + `Data/tableau_source.hyper` (the same, unpacked) and `tableau_source.xlsx` (the same rows
as Excel, for inspection only; the workbook does not read it).

1. Install Tableau Desktop Public Edition (free) and sign in to Tableau Public.
2. File > Open `residual_tft_ewc.twbx`. Its data sources are already extracts, as Tableau Public requires.
3. The workbook is generated and has not been opened before. For every dashboard, check the views against
   `dashboard/tableau/WORKBOOK_SPEC.md` and fix what Tableau reports (missing fields, mark types, sorts). Keep the
   five dashboard names and the fields Dataset, Setting, Protocol, Group and Method Family exactly as they are:
   the web page filters by those names.
4. Finish by hand what the generator cannot write reliably (WORKBOOK_SPEC.md §3):
   - SOH trajectories: reference bands at 95 / 90 / 80 % labelled C0-C3; axis title "SOH, % of first valid
     check"; invalid checks (`valid` = false) as hollow points with no line (the line field, SOH valid checks, is
     already null there); tooltip `temperature_source`.
   - Adaptation vs retention: a diagonal 1:1 reference line. Forecast parity: the 1:1 line.
   - Seed spread: Analytics > Box Plot on the final_avg_error axis.
   - Interval coverage: a constant reference line at 0.9.
   - Backward transfer heatmap: diverging palette with grey at zero.
   - Forecast accuracy: check the D30 metric parameter control is shown.
5. Colours: proposed model in the accent blue `#2a78d6`, baselines grey, families in the fixed order of the spec.
6. File > Save to Tableau Public (no extract step: the sources are extracts already).
   For each dashboard, copy its share URL
   (`https://public.tableau.com/views/<Workbook>/<Dashboard>`) into `TABLEAU.views` in `dashboard/config.js`.
7. Save the corrected workbook as `dashboard/tableau/residual_tft_ewc.twb` (the .twb holds no data) and commit
   it. Keep the .xlsx and any .twbx out of Git.
8. Serve the page (`python -m http.server 8000` from the repository root, open /dashboard/) and check that the
   filter bar moves every embedded view and that selecting a mark opens the details panel.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-out", default=str(ROOT / "results" / "dashboard"))
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", str(ROOT / "data")))
    ap.add_argument("--out", default=None, help="default: <data-root>/dashboard/tableau")
    args = ap.parse_args(argv)
    data = Path(args.data_root)
    data = data if data.is_absolute() else ROOT / data
    out = Path(args.out) if args.out else data / "dashboard" / "tableau"
    f = contract_fields()
    tables = sheets(Path(args.results_out), data / "dashboard", f)
    if not (Path(args.results_out) / "dim_method.csv").exists():
        raise SystemExit(f"no export in {args.results_out}: run scripts/export_dashboard_data.py first")
    xlsx = (out / "tableau_source.xlsx").resolve()
    write_xlsx(tables, xlsx)
    hyper, twb = out / HYPER_REL, out / "residual_tft_ewc.twb"
    write_hyper(tables, hyper)
    build_twb(tables, f).write(twb, encoding="utf-8", xml_declaration=True)
    write_twbx(twb, hyper, out / "residual_tft_ewc.twbx")
    (out / "AUTHORING.md").write_text(AUTHORING, encoding="utf-8")
    for name, df in tables.items():
        note = "" if len(df) else "  (empty placeholder: build the data folder and re-run the export)"
        print(f"  {name}: {len(df):,} rows{note}")
    print(f"Wrote tableau_source.xlsx, Data/tableau_source.hyper, residual_tft_ewc.twb/.twbx (starter: open and check in Tableau) and AUTHORING.md "
          f"to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
