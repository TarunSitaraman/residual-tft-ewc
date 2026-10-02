"""Independent audit of the forecasting data (S1 and S6) before any model is trained on it.

Loads the data exactly as notebooks 12 and 13 do and checks, per dataset:

- labels: the SOH formula, the reference (first valid RPT/check), and agreement
  with the handover (S1) or the authors' own capacities (S6);
- time order: strictly increasing RPT/check dates per cell, and calendar gaps;
- samples: count, target definition, inputs known at the origin only, schedule
  arithmetic (S1), and purged splits;
- leakage screen: no protocol current among the inputs, and no input nearly
  identical to the target;
- coverage and missingness per cell, task and split;
- physical plausibility of the diagnostic features;
- SOH increases between consecutive measurements (recovery or measurement noise).

Every check is recorded as PASS, FAIL or INFO with its evidence in
``results/data_audit.json`` (environment and commit included). The script
exits with status 1 if any check fails. It never modifies data.

Usage (from the repository root, after the build scripts)::

    python scripts/audit_data.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_pipeline as dp  # noqa: E402
from src import diagnostic_features as F  # noqa: E402
from src import forecasting as FC  # noqa: E402
from src import wheeler as W  # noqa: E402

#: Values published in docs/RESULTS.md and the results JSON files; the audit must reproduce them.
EXPECTED = {"s1_samples": 279, "s6_samples": 2409, "s6_checks": 847, "s6_valid_checks": 843,
            "s6_invalid": {("cell1a", 100), ("cell2a", 100), ("cell2c", 100), ("cell3a", 100)}}
LEAK_RHO = 0.95          # within-cell |Spearman| of an input with the target that triggers a leakage FAIL
SOH_TOL = 1e-9


class Audit:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def check(self, dataset: str, name: str, ok: bool | None, detail) -> None:
        status = "INFO" if ok is None else ("PASS" if ok else "FAIL")
        self.rows.append({"dataset": dataset, "check": name, "status": status, "detail": detail})
        print(f"[{status}] {dataset} {name}: {detail if not isinstance(detail, (dict, list)) else json.dumps(detail, default=str)[:300]}")

    @property
    def failed(self) -> list[dict]:
        return [r for r in self.rows if r["status"] == "FAIL"]


def _jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    return str(x)


# ---------------------------------------------------------------------------
# Shared checks
# ---------------------------------------------------------------------------

def check_labels(a: Audit, ds: str, t: pd.DataFrame, cell: str, order: str, valid: pd.Series) -> None:
    v = t[valid].sort_values([cell, order])
    recomputed = v["Capacity_Ah"] / v["Reference_Capacity_Ah"] * 100
    a.check(ds, "SOH formula", bool((recomputed - v["SOH_Percent"]).abs().max() < SOH_TOL),
            f"max |SOH - Capacity/Reference*100| = {(recomputed - v['SOH_Percent']).abs().max():.2e}")
    first = v.groupby(cell)["Capacity_Ah"].first()
    ref = v.groupby(cell)["Reference_Capacity_Ah"].agg(["min", "max"])
    a.check(ds, "reference = first valid measurement",
            bool(((ref["min"] - first).abs() < SOH_TOL).all() and ((ref["max"] - first).abs() < SOH_TOL).all()),
            f"{len(first)} cells, reference constant within each cell and equal to its first valid capacity")


def check_sample_core(a: Audit, ds: str, s: pd.DataFrame, cell: str, origin: str, target: str) -> None:
    a.check(ds, "target = SOH(k+h) - SOH(k)", bool((s[FC.TARGET] - (s["soh_target"] - s["soh_origin"])).abs().max() < SOH_TOL),
            f"max deviation {(s[FC.TARGET] - (s['soh_target'] - s['soh_origin'])).abs().max():.2e}")
    a.check(ds, "target after origin", bool((s[target] > s[origin]).all() and (s["target_pos"] - s["origin_pos"] == s["horizon"]).all()),
            "every target lies h valid measurements after its origin")
    sp = FC.assign_split(s)
    bad = []
    for c, g in sp.groupby(cell):
        tr, va, te = (g[g["split"] == k] for k in ("train", "val", "test"))
        if len(va) and len(tr) and tr["target_pos"].max() > va["origin_pos"].min():
            bad.append((c, "train label after val origin"))
        if len(te) and len(va) and va["target_pos"].max() > te["origin_pos"].min():
            bad.append((c, "val label after test origin"))
        if not (len(tr) and len(va) and len(te)):
            bad.append((c, "empty split"))
    a.check(ds, "purged splits", not bad, bad or "per cell: max train target <= min val origin, max val target <= min test origin; no empty split")
    counts = sp.groupby([cell, "split"]).size().unstack(fill_value=0)
    a.check(ds, "samples per cell and split", None, counts.to_dict(orient="index"))


def check_leakage(a: Audit, ds: str, s: pd.DataFrame, features: list[str], cell: str) -> None:
    banned = [c for c in features if c in F.PROTOCOL_CURRENTS or "current" in c.lower()]
    a.check(ds, "no protocol current among inputs", not banned, banned or "none")
    flagged, table = [], {}
    for f in features:
        rhos = []
        for _, g in s.groupby(cell):
            x, y = g[f], g[FC.TARGET]
            m = x.notna() & y.notna()
            if m.sum() >= 5 and x[m].nunique() > 1:
                rhos.append(spearmanr(x[m], y[m]).statistic)
        if rhos:
            med = float(np.nanmedian(np.abs(rhos)))
            table[f] = round(med, 3)
            if med >= LEAK_RHO:
                flagged.append(f)
    a.check(ds, f"no input with median within-cell |rho| >= {LEAK_RHO} with the target", not flagged,
            flagged or f"largest: {sorted(table.items(), key=lambda kv: -kv[1])[:5]}")


def check_missingness(a: Audit, ds: str, s: pd.DataFrame, features: list[str]) -> None:
    miss = s[features].isna().mean()
    a.check(ds, "missing share per input (non-zero only)", None, {k: round(float(v), 3) for k, v in miss[miss > 0].items()})


def check_soh_increases(a: Audit, ds: str, t: pd.DataFrame, cell: str, order: str, threshold: float) -> None:
    rows = []
    for c, g in t.sort_values([cell, order]).groupby(cell):
        d = g["SOH_Percent"].diff()
        for i in np.flatnonzero(d.to_numpy() > threshold):
            rows.append({"cell": c, order: _jsonable(g[order].iloc[i]), "increase_pp": round(float(d.iloc[i]), 3)})
    a.check(ds, f"SOH increases > {threshold} pp between consecutive measurements", None, rows or "none")


# ---------------------------------------------------------------------------
# S1
# ---------------------------------------------------------------------------

def audit_s1(a: Audit, data_root: Path) -> None:
    ds = "S1"
    table = dp.read_dataset(data_root / "diagnostic_features.csv")
    table["rpt_start"] = pd.to_datetime(table["rpt_start"])
    cell = dp.GROUP_COL
    check_labels(a, ds, table, cell, "rpt_number", table["Capacity_Ah"].notna())
    a.check(ds, "RPT count", None, table.groupby(cell)["rpt_number"].agg(["count", "min", "max"]).to_dict(orient="index"))

    order_bad = [c for c, g in table.sort_values([cell, "rpt_number"]).groupby(cell) if not g["rpt_start"].is_monotonic_increasing]
    a.check(ds, "RPT dates increase with RPT number", not order_bad, order_bad or "all cells")

    hand = dp.read_dataset(data_root / "final_battery_health_dataset.csv")
    if "SOH_handover" in table:
        diff = (table["SOH_Percent"] - table["SOH_handover"]).abs()
        a.check(ds, "C/20 labels vs handover labels (D17: at most 0.089 pp)", bool(diff.max() <= 0.0895),
                f"max |difference| = {diff.max():.4f} pp over {diff.notna().sum()} RPTs")
    a.check(ds, "handover health table rows", None, f"{len(hand)} rows; diagnostic table {len(table)} rows")

    c20 = data_root / "rpt_health_c20.csv"
    if c20.is_file():
        h = dp.read_dataset(c20)
        m = h.merge(table[[cell, "rpt_number", "SOH_Percent"]], left_on=[cell, "RPT_Number"], right_on=[cell, "rpt_number"])
        a.check(ds, "rpt_health_c20.csv agrees with diagnostic table", bool(len(m) == len(h) and (m["SOH_Percent_x"] - m["SOH_Percent_y"]).abs().max() < SOH_TOL),
                f"{len(m)}/{len(h)} rows matched")

    windows = [c for c in table.columns if c.startswith("q_") and c != "q_c20_ah"]
    if windows:
        resid = (table[windows].sum(axis=1, min_count=len(windows)) - table["q_c20_ah"]).abs()
        a.check(ds, "voltage-window charges sum to the C/20 capacity", None,
                f"max |sum - q_c20| = {resid.max():.4f} Ah over {resid.notna().sum()} RPTs (windows: {windows})")

    temp_col = next(c for c in dp.read_header(data_root / "cleaned_battery_data.csv") if c.startswith("Aux_Temperature"))
    points = dp.read_dataset(data_root / "cleaned_battery_data.csv", usecols=["Date_Time", "Source_Folder", "Cycling_Test", temp_col])
    blocks = FC.cycling_blocks(points)
    s = FC.build_samples(table, blocks)
    a.check(ds, "sample count", len(s) == EXPECTED["s1_samples"], f"{len(s)} (published {EXPECTED['s1_samples']})")
    a.check(ds, "RPT 2 excluded", bool(not s["origin_rpt"].isin(FC.EXCLUDED_RPTS).any() and not s["target_rpt"].isin(FC.EXCLUDED_RPTS).any()),
            "RPT 2 is never an origin or a target")
    check_sample_core(a, ds, s, cell, "Cycle_Date", "target_date")

    sched = (s["days_35c"] + s["days_20c"] + s["days_rest"] - s["days_to_target"]).abs()
    a.check(ds, "schedule days add up to days to target", bool(sched.max() < 1e-6), f"max deviation {sched.max():.2e} days")
    over = s.loc[s["days_35c"] + s["days_20c"] > s["days_to_target"] + 1e-6]
    a.check(ds, "cycling days never exceed the interval", over.empty, f"{len(over)} samples exceed")

    # Fade rate must use only RPTs up to the origin: recompute from the table.
    t = table[~table["rpt_number"].isin(FC.EXCLUDED_RPTS)].sort_values([cell, "rpt_start"])
    worst = 0.0
    for bat, g in t.groupby(cell):
        days = ((g["rpt_start"] - g["rpt_start"].iloc[0]).dt.total_seconds() / 86400).to_numpy()
        soh = g["SOH_Percent"].to_numpy(float)
        sb = s[s[cell] == bat]
        for _, r in sb.iterrows():
            i = int(r["origin_pos"])
            ref = np.polyfit(days[i - 2:i + 1], soh[i - 2:i + 1], 1)[0] if i >= 2 else np.nan
            if np.isnan(ref) != np.isnan(r["fade_rate_pp_per_day"]):
                worst = np.inf
            elif not np.isnan(ref):
                worst = max(worst, abs(ref - r["fade_rate_pp_per_day"]))
    a.check(ds, "fade rate uses only the last three RPTs up to the origin", bool(worst < 1e-9), f"max deviation {worst:.2e}")

    sets = FC.feature_sets(s)
    check_leakage(a, ds, s, sets["F1"], cell)
    check_missingness(a, ds, s, sets["F1"])
    check_soh_increases(a, ds, t, cell, "rpt_number", 0.1)

    gaps = t.groupby(cell)["rpt_start"].diff().dt.days
    a.check(ds, "days between consecutive valid RPTs", None,
            {"median": float(gaps.median()), "max": float(gaps.max()), "n_gaps_over_60_days": int((gaps > 60).sum())})


# ---------------------------------------------------------------------------
# S6
# ---------------------------------------------------------------------------

def audit_s6(a: Audit, data_root: Path) -> None:
    ds = "S6"
    checks = dp.read_dataset(data_root / "wheeler_checks.csv")
    checks["check_time"] = pd.to_datetime(checks["check_time"])
    a.check(ds, "check count", len(checks) == EXPECTED["s6_checks"] and int(checks["valid"].sum()) == EXPECTED["s6_valid_checks"],
            f"{len(checks)} checks, {int(checks['valid'].sum())} valid (published {EXPECTED['s6_checks']}/{EXPECTED['s6_valid_checks']})")
    invalid = {(c, int(y)) for c, y in checks.loc[~checks["valid"], ["cell", "cycle"]].itertuples(index=False)}
    a.check(ds, "unlabelled checks are exactly the documented acquisition faults", invalid == EXPECTED["s6_invalid"], sorted(invalid))
    a.check(ds, "cells per test case", None, checks.groupby(dp.GROUP_COL)["cell"].nunique().to_dict())
    check_labels(a, ds, checks, "cell", "cycle", checks["valid"])

    order_bad = [c for c, g in checks.sort_values(["cell", "cycle"]).groupby("cell") if not g["check_time"].is_monotonic_increasing]
    a.check(ds, "check times increase with cycle number", not order_bad, order_bad or "all cells")

    v = checks[checks["valid"]].sort_values(["cell", "cycle"])
    g = v.groupby("cell")
    days = g["check_time"].diff().dt.total_seconds() / 86400
    cyc = g["cycle"].diff()
    per100 = (days / cyc * 100).dropna()
    a.check(ds, "calendar days per 100 cycles between consecutive valid checks (50 degC; not an input)", None,
            {"median": round(float(per100.median()), 1), "p95": round(float(per100.quantile(0.95)), 1),
             "max": round(float(per100.max()), 1),
             "by_case_median": v.assign(d=per100).groupby(dp.GROUP_COL)["d"].median().round(1).to_dict()})
    longest = v.assign(gap_days=days).nlargest(8, "gap_days")[["cell", "cycle", "gap_days"]]
    a.check(ds, "longest calendar gaps before a check", None,
            [{k: _jsonable(x) for k, x in r.items()} for r in longest.to_dict(orient="records")])

    feats = W.feature_columns(checks)
    plaus = {
        "r_c3_c25 > 0 (ohm)": (v["r_c3_c25"] <= 0).sum() if "r_c3_c25" in v else 0,
        "q_ratio_c3_c25 <= 1.02": (v["q_ratio_c3_c25"] > 1.02).sum() if "q_ratio_c3_c25" in v else 0,
        "hysteresis_c25_mid > 0 (V)": (v["hysteresis_c25_mid"] <= 0).sum() if "hysteresis_c25_mid" in v else 0,
        "C/25 capacity <= 1.2 x nominal": (v["Capacity_Ah"] > 1.2 * W.NOMINAL_AH).sum(),
    }
    a.check(ds, "physical plausibility (violations per rule)", all(int(n) == 0 for n in plaus.values()),
            {k: int(n) for k, n in plaus.items()})
    a.check(ds, "SOH range per test case", None, v.groupby(dp.GROUP_COL)["SOH_Percent"].agg(["min", "max"]).round(1).to_dict(orient="index"))

    s = W.build_samples(checks)
    a.check(ds, "sample count", len(s) == EXPECTED["s6_samples"], f"{len(s)} (published {EXPECTED['s6_samples']})")
    check_sample_core(a, ds, s, "cell", "origin_cycle", "target_cycle")
    sp = FC.assign_split(s)
    a.check(ds, "samples per test case (task) and split", None, sp.groupby([dp.GROUP_COL, "split"]).size().unstack(fill_value=0).to_dict(orient="index"))
    sets = W.feature_sets(s)
    dropped = [c for c in feats if c not in sets["F1"]]
    a.check(ds, "diagnostic features dropped as never observed (D20)", None, dropped or "none")
    check_leakage(a, ds, s, sets["F1"], "cell")
    check_missingness(a, ds, s, sets["F1"])
    check_soh_increases(a, ds, v, "cell", "cycle", 0.5)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "data"))
    ap.add_argument("--out", default=str(ROOT / "results" / "data_audit.json"))
    ap.add_argument("--datasets", nargs="+", choices=["S1", "S6"], default=["S1", "S6"])
    args = ap.parse_args()
    data_root = dp.resolve_data_root(args.data_root, ROOT)

    a = Audit()
    if "S1" in args.datasets:
        audit_s1(a, data_root)
    if "S6" in args.datasets:
        audit_s6(a, data_root)
    report = {"environment": dp.environment_info(ROOT), "n_checks": len(a.rows), "n_failed": len(a.failed), "checks": a.rows}
    Path(args.out).write_text(json.dumps(report, indent=1, default=_jsonable), encoding="utf-8")
    print(f"\n{len(a.rows)} checks, {len(a.failed)} failed -> {args.out}")
    return 1 if a.failed else 0


if __name__ == "__main__":
    sys.exit(main())
