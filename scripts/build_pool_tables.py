"""Build the D21 pool check tables from the verified raw downloads (scripts/fetch_pool.py).

Outputs (derived data, git-ignored):
    data/pool_s7_checks.csv   one row per (S7 cell, valid check-up): capacity, SOH, static conditions
    data/pool_s8_checks.csv   one row per (S8 cell, check): mean capacity of the measurement cycles
    data/pool_s9_checks.csv   one row per (S9 cell, check): mean capacity of 10 full-discharge cycles

Usage (from the repository root)::

    python scripts/build_pool_tables.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data_pipeline as dp  # noqa: E402
from src import pool  # noqa: E402

S7_EOC = Path("raw/pool/S7/10.35097-1969/data/dataset/cell_eocv2.zip")
S8_ZIP = Path("raw/pool/S8/Battery Dataset.zip")
S9_ZIPS = [Path(f"raw/pool/S9/{n}.zip") for n in ("Dataset_1_NCA_battery", "Dataset_2_NCM_battery",
                                                    "Dataset_3_NCM_NCA_battery")]


def summarise(name: str, t, static) -> None:
    kept = t[t["n_valid_checks"] >= pool.MIN_VALID_CHECKS]
    s = pool.build_samples(t, static=static)
    print(f"{name}: {t['cell'].nunique()} cells, {len(t)} checks; {kept['cell'].nunique()} cells in "
          f"{kept['condition'].nunique()} conditions with >= {pool.MIN_VALID_CHECKS} checks ({len(kept)} checks); "
          f"{len(s)} samples; SOH {kept['SOH_Percent'].min():.1f}-{kept['SOH_Percent'].max():.1f} %")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "data"))
    args = ap.parse_args()
    root = dp.resolve_data_root(args.data_root, Path(__file__).resolve().parents[1])

    t = pool.s7_check_table(root / S7_EOC)
    t.to_csv(root / "pool_s7_checks.csv", index=False)
    kept = t[t["n_valid_checks"] >= pool.MIN_VALID_CHECKS]
    s = pool.build_samples(t, static=pool.S7_STATIC)
    print(f"S7: {t['cell'].nunique()} cells, {len(t)} valid check-ups; "
          f"{kept['cell'].nunique()} cells in {kept['condition'].nunique()} condition sets with "
          f">= {pool.MIN_VALID_CHECKS} check-ups ({len(kept)} check-ups); {len(s)} forecasting samples")
    print(f"S7: reference capacity {t.groupby('cell')['Reference_Capacity_Ah'].first().min():.3f}-"
          f"{t.groupby('cell')['Reference_Capacity_Ah'].first().max():.3f} Ah; "
          f"SOH {kept['SOH_Percent'].min():.1f}-{kept['SOH_Percent'].max():.1f} %; "
          f"check-up temperature {t['check_temp_c'].min():.1f}-{t['check_temp_c'].max():.1f} degC")
    for name, build, arg, static in (("S8", pool.s8_check_table, root / S8_ZIP, []),
                                     ("S9", pool.s9_check_table, [root / z for z in S9_ZIPS], ["temp_c", "chemistry"])):
        t = build(arg)
        # Time must increase with cycle number within every cell (checked in cycle order, not time order).
        bad = t.sort_values(["cell", "cycle"]).groupby("cell")["elapsed_s"].apply(lambda x: not x.is_monotonic_increasing)
        assert not bad.any(), f"{name}: time not increasing with cycle in {bad[bad].index.tolist()[:3]}"
        if "time_resets" in t:
            r = t.groupby("cell")["time_resets"].first()
            print(f"{name}: time counter restarted in {int((r > 0).sum())} cells: {r[r > 0].index.tolist()}")
        t.to_csv(root / f"pool_{name.lower()}_checks.csv", index=False)
        summarise(name, t, static)
    return 0


if __name__ == "__main__":
    sys.exit(main())
