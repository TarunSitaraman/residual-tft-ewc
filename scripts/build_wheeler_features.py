"""Build the S6 (Wheeler et al. 2025) check table and check it against the authors' values (D20).

Prerequisite: ``python scripts/fetch_wheeler_characterisation.py`` and the
authors' ``extractedData.mat`` in ``data/raw/wheeler2025`` (both checksummed).

Outputs (derived data, git-ignored):
    data/wheeler_checks.csv   one row per (cell, 50 degC check): capacities, features, C/25 SOH

The authors' own capacity (``extractedData.mat``: maximum C/3 charge counter of
each check) is recomputed from our files; any difference above 1e-6 Ah on a
check present in both stops the run, so a parsing error cannot pass silently.
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
from src import wheeler as W  # noqa: E402


def authors_capacities(mat_path: Path) -> pd.DataFrame:
    m = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    rows = []
    for c in m["cell"]:
        for ch, f in zip(np.atleast_1d(c.charac), np.atleast_1d(c.characfileslist)):
            name = str(f.name)
            if "charac" not in name or "initial" in name or "final" in name:
                continue
            cap = np.asarray(ch.chargecapacity, float).ravel()
            rows.append({"cell": str(c.name).lower(), "cycle": int(name.split("charac")[-1].split(".")[0]),
                         "authors_value": float(cap[0]) if cap.size == 1 else np.nan})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "data"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    root = Path(args.data_root) / W.RAW_SUBDIR
    table = W.build_table(root, args.workers)
    ref = authors_capacities(root / "extractedData.mat")
    cmp_ = table.merge(ref, on=["cell", "cycle"], how="inner")
    both = cmp_.dropna(subset=["authors_value", "authors_c3_charge_ah"])
    diff = (both["authors_value"] - both["authors_c3_charge_ah"]).abs()
    print(f"authors' C/3 charge capacity: {len(both)} checks compared, max |difference| {diff.max():.2e} Ah")
    if (diff > 1e-6).any():
        raise RuntimeError(f"{int((diff > 1e-6).sum())} checks differ from the authors' extractedData.mat")
    table.to_csv(Path(args.data_root) / "wheeler_checks.csv", index=False)
    print(f"{len(table)} checks, {int(table['valid'].sum())} with a C/25 label, {table['cell'].nunique()} cells; "
          f"SOH range {table['SOH_Percent'].min():.1f}-{table['SOH_Percent'].max():.1f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
