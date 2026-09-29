"""Build the per-(battery, RPT) diagnostic feature table from the raw archive (decision D16).

Prerequisite: ``python scripts/fetch_osf_diagnostics.py`` (raw files in
``data/raw/osf_8jnr5_v2``). The first run converts every ``.xlsx`` to a pickle
cache (about 20 minutes on 4 cores); later runs take about a minute.

Outputs (derived data, git-ignored):
    data/diagnostic_features.csv       one row per (battery, RPT)
    data/diagnostic_selection_log.csv  every file choice and exclusion
    data/rpt_health_c20.csv            RPT health table on the C/20 capacity basis (D17)

Also checks the RPT 7 decoding against the authors' own high-frequency
resistance (``RPT_7/eis_7.mat``, ``EIS_7_R_HF``) and stops on a mismatch.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data_pipeline as dp  # noqa: E402
from src import diagnostic_features as F  # noqa: E402
from src import raw_diagnostics as rd  # noqa: E402


def check_rpt7_r0(table: pd.DataFrame, raw_root: Path, tol_ohm: float = 1e-6) -> float:
    """Largest |our R0 - authors' R_HF| over RPT 7 (18 values)."""
    m = sio.loadmat(raw_root / "RPT_7" / "eis_7.mat", squeeze_me=True)
    worst = 0.0
    for i, v in enumerate(int(x) / 100 for x in m["EIS_volts"]):
        for j, cell in enumerate(c.lower() for c in m["cn_fn_chan_order"]):
            ours = table.loc[(table[dp.GROUP_COL] == cell) & (table["rpt_number"] == 7), f"eis_r0_{v:.2f}"].iloc[0]
            worst = max(worst, abs(float(ours) - float(m["EIS_7_R_HF"][i, j])))
    if not worst <= tol_ohm:
        raise RuntimeError(f"RPT 7 R0 differs from the authors' R_HF by {worst:.2e} ohm")
    return worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "data"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    data_root = Path(args.data_root)
    raw_root, cache_root = data_root / rd.RAW_SUBDIR, data_root / rd.CACHE_SUBDIR
    if not raw_root.exists():
        raise SystemExit(f"{raw_root} not found: run scripts/fetch_osf_diagnostics.py first")

    inv = rd.list_raw_files(raw_root)
    n = rd.convert_all(raw_root, cache_root, workers=args.workers, files=inv.loc[inv["kind"].notna(), "path"])
    print(f"converted {n} files to the cache")

    health = dp.read_dataset(data_root / "final_battery_health_dataset.csv")
    table, log = F.build_feature_table(raw_root, cache_root, health)
    worst = check_rpt7_r0(table, raw_root)
    print(f"RPT 7 R0 vs authors' R_HF: max difference {worst:.1e} ohm")

    table = table.drop(columns=["_currents"]).assign(nonstandard_pulses=table["nonstandard_pulses"].map(
        lambda x: ";".join(f"{v:.2f}" for v in x)))
    table.to_csv(data_root / "diagnostic_features.csv", index=False)
    health_c20 = F.health_table(table)
    if not dp.validate_soh_formula(health_c20)["ok"]:
        raise RuntimeError("C/20 health table violates SOH = Capacity / Reference * 100")
    health_c20.to_csv(data_root / "rpt_health_c20.csv", index=False)
    shift = (health_c20["SOH_Percent"] - health_c20["SOH_handover"]).abs()
    print(f"C/20 labels vs handover labels: max |SOH difference| {shift.max():.3f} pp")
    log.assign(rejected=log["rejected"].map(lambda x: ";".join(x) if isinstance(x, list) else x)).to_csv(
        data_root / "diagnostic_selection_log.csv", index=False)
    feats = F.feature_columns(table)
    print(f"{len(table)} rows, {len(feats)} features; non-missing per feature: "
          f"{int(table[feats].notna().sum().min())}-{int(table[feats].notna().sum().max())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
