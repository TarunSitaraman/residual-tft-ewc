"""Build the D22 ageing inputs from the downloaded S6 ageing files; check causality, completeness and leakage.

Outputs (data/ is outside Git; the small JSON report is committed):
- data/s6_ageing_cycles.csv: per-cycle validity and responses of every block present;
- data/s6_ageing_blocks.csv: per (cell, block end cycle) the F2 aggregates and valid/invalid cycle counts;
- results/d22_ageing_report.json: completeness per cell, causality check, leakage audit (D22 amendments).

Per-file results are cached in data/cache/ageing, so the script can be re-run as the download progresses.

Usage (from the repository root)::

    python scripts/build_s6_ageing_features.py [--workers 2]
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import ageing as A  # noqa: E402
from src import data_pipeline as dp  # noqa: E402
from src import experiments as X  # noqa: E402
from src import wheeler as W  # noqa: E402

RAW = ROOT / "data" / "raw" / "wheeler2025"
CACHE = ROOT / "data" / "cache" / "ageing"


def case_settings() -> dict[str, tuple[float, bool]]:
    """cell -> (charge voltage limit, CC-CV charge?) from the test-case table (Wheeler et al. 2025, Table 2)."""
    wc = dp.read_dataset(ROOT / "data" / "wheeler_checks.csv")
    tc = W.TEST_CASES.set_index("Battery_ID")
    case = wc.drop_duplicates("cell").set_index("cell")["Battery_ID"]
    return {c: (float(tc.loc[b, "v_max"]), bool(tc.loc[b, "cccv_charge"])) for c, b in case.items()}


def one_file(args: tuple[str, float, bool]) -> Path:
    path, v_limit, cccv = args
    out = CACHE / (Path(path).name.split(".csv")[0] + ".csv")
    if not out.exists():
        cell, first, last = A.block_range(path)
        c = A.cycle_responses(A.read_block(path), first, v_limit, cccv)
        c.insert(0, "cell", cell.lower())
        c.insert(1, "block_end", last)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        c.to_csv(tmp, index=False)
        tmp.replace(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    settings = case_settings()
    # The first ageing files were stored as .gz, later ones as .xz (fetch_wheeler_characterisation.py --store).
    files = sorted([*RAW.glob("Cell*/agingdata_50degC/*aging*.csv.xz"), *RAW.glob("Cell*/agingdata_50degC/*aging*.csv.gz")])
    stems = [f.name.rsplit(".", 1)[0] for f in files]
    if len(stems) != len(set(stems)):
        raise SystemExit("an ageing block is stored both as .xz and .gz")
    jobs = [(str(f), *settings[A.block_range(f)[0].lower()]) for f in files]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        outs = list(pool.map(one_file, jobs))
    cycles = pd.concat([pd.read_csv(o) for o in outs], ignore_index=True)
    blocks = pd.DataFrame([{"cell": cell, "block_end": end, "t_end": float(g["t_end"].max()), **A.block_features(g)}
                           for (cell, end), g in cycles.groupby(["cell", "block_end"], sort=True)])
    cycles.to_csv(ROOT / "data" / "s6_ageing_cycles.csv", index=False)
    blocks.to_csv(ROOT / "data" / "s6_ageing_blocks.csv", index=False)

    wc = dp.read_dataset(ROOT / "data" / "wheeler_checks.csv")
    wc["check_time"] = pd.to_datetime(wc["check_time"])
    checks = wc[wc["cycle"] > 0][["cell", "cycle", "check_time", "valid", "Capacity_Ah"]]
    m = checks.merge(blocks, left_on=["cell", "cycle"], right_on=["cell", "block_end"], how="left")
    have = m[m["block_end"].notna()].copy()
    have["block_end_time"] = A.excel_time(have["t_end"].to_numpy())
    late = have[have["block_end_time"] >= have["check_time"]]

    completeness = []
    for cell, g in m.groupby("cell"):
        completeness.append({"cell": cell, "checks_after_cycle_0": int(len(g)), "blocks_present": int(g["block_end"].notna().sum()),
                             "blocks_missing": [int(c) for c in g.loc[g["block_end"].isna(), "cycle"]],
                             "valid_cycles": int(g["age_valid_cycles"].sum()), "invalid_cycles": int(g["age_invalid_cycles"].sum()),
                             "blocks_below_min_valid": int((g["age_valid_cycles"] < A.MIN_VALID_CYCLES).sum())})

    # Leakage audit (D22): (i) per-cycle discharged charge vs the previous check's capacity, within cell;
    # (ii) each F2 input vs the origin check's capacity, within cell (Spearman), reported.
    prev = wc[["cell", "cycle", "Capacity_Ah"]].assign(block_end=lambda t: t["cycle"] + 100)
    qd = cycles.groupby(["cell", "block_end"])["q_discharged"].mean().reset_index()
    qd = qd.merge(prev[["cell", "block_end", "Capacity_Ah"]], on=["cell", "block_end"], how="inner").dropna()
    case = wc.drop_duplicates("cell").set_index("cell")["Battery_ID"]
    audit_i = [{"cell": c, "case": case[c], "n": int(len(g)), "pearson": float(g["q_discharged"].corr(g["Capacity_Ah"]))}
               for c, g in qd.groupby("cell") if len(g) >= 5]
    valid = have[have["valid"]]
    audit_ii = {}
    for f in A.feature_names():
        rho = [float(g[f].corr(g["Capacity_Ah"], method="spearman")) for _, g in valid.groupby("cell")
               if g[f].notna().sum() >= 5 and g[f].nunique() > 1]
        audit_ii[f] = {"n_cells": len(rho), "median_rho": float(np.median(rho)) if rho else None,
                       "max_abs_rho": float(np.max(np.abs(rho))) if rho else None}
    X.write_results(ROOT / "results" / "d22_ageing_report.json", {
        "decision": "D22", "files_processed": len(outs), "cells_with_files": sorted(cycles["cell"].unique()),
        "n_cycles": int(len(cycles)), "n_blocks": int(len(blocks)),
        # A block ending at or after its check matters only if that check is valid (an origin); invalid checks
        # (acquisition faults, e.g. a repeated check) are never origins.
        "causality": {"blocks_checked": int(len(have)), "blocks_ending_at_or_after_check": int(len(late)),
                      "violations_at_valid_checks": int(late["valid"].sum()),
                      "violations": late[["cell", "cycle", "valid"]].to_dict("records")},
        "completeness": completeness, "audit_discharged_charge_vs_previous_capacity": audit_i,
        "audit_f2_vs_origin_capacity": audit_ii}, ROOT)
    print(f"{len(outs)} files, {len(cycles)} cycles ({int(cycles['valid'].sum())} valid), {len(blocks)} blocks; "
          f"causality violations: {len(late)} ({int(late['valid'].sum())} at valid checks)", flush=True)
    for c in completeness:
        print(f"  {c['cell']}: {c['blocks_present']}/{c['checks_after_cycle_0']} blocks, valid cycles {c['valid_cycles']}, "
              f"invalid {c['invalid_cycles']}, blocks below {A.MIN_VALID_CYCLES} valid: {c['blocks_below_min_valid']}", flush=True)
    for a in audit_i:
        print(f"  audit (i) {a['cell']} ({a['case']}): r = {a['pearson']:.3f} over {a['n']} blocks", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
