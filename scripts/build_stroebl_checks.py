"""Build the S10 check table (Stroebl et al. 2024) with the D36 label rules.

Input: the 23 degC check-ups kept by ``scripts/fetch_stroebl.py`` under ``data/raw/stroebl2024/<file_id>/`` and the
authors' ``experiments_meta.csv``. Output: ``data/pool_s10_checks.csv`` (valid check-ups of cells with at least five,
in the layout of the S7 table) and ``data/pool_s10_checks_all.csv`` (every check-up with its validity reason).

Label (D36): capacity = |integral of I dt| (Ah) of the last complete ``step_type`` 22 block (section A capacity
discharge, CC 1C + CV at 2.5 V) of each CU, exCU, ET_T23 and AT_T23 file; complete = minimum voltage <= 2.55 V and
final |I| <= 0.3 A. Invalid: no complete block, or cell surface temperature at the block start outside 18-28 degC.
SOH relative to the cell's first valid check-up. Check-up date = measurement start date of the file's meta.txt.

Stage: test-point names repeat across the two stages, and the internal serial in meta.txt is not unique across them,
so a zip's stage is taken from its ET_T23 date (stage 1 began Sep-Oct 2021, stage 2 Dec 2022-Feb 2023); every
(stage, cell) pair must then occur exactly once and exist in experiments_meta.csv, or the build stops.

    python scripts/build_stroebl_checks.py --workers 6
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_stroebl import MANIFEST, data_root  # noqa: E402

STAGE2_FROM = pd.Timestamp("2022-06-01")
FILE = re.compile(r"^(TP_[kz]\d\d_\d\d)_(\d\d)_(CU|exCU|ET_T23|AT_T23)\.csv\.gz$")
V_END, I_END, T_LO, T_HI, MIN_CHECKS = 2.55, 0.3, 18.0, 28.0, 5


def seconds(run_time: pd.Series) -> np.ndarray:
    p = run_time.str.split(":", expand=True).astype(float)
    return (p[0] * 3600 + p[1] * 60 + p[2]).to_numpy()


def start_date(meta_gz: Path) -> pd.Timestamp:
    """dd.mm.yyyy or dd.mm.yy; anything else (2 files: 'xx.12.2021', '22.10.222.12.233') or a missing meta.txt (1 file,
    TP_z02_02_09_CU in zip 47631133) is NaT, never guessed."""
    if not meta_gz.exists():
        return pd.NaT
    t = gzip.open(meta_gz).read().decode("latin1")
    d = re.search(r"Measurement start date:\s*(\S*)", t).group(1)
    if not re.fullmatch(r"\d\d\.\d\d\.(\d\d|\d{4})", d):
        return pd.NaT
    return pd.to_datetime(d, format="%d.%m.%Y" if len(d) == 10 else "%d.%m.%y")


def capacity(csv_gz: Path) -> dict:
    """Capacity of the last complete step_type 22 block, with the block's start temperature and validity."""
    d = pd.read_csv(csv_gz, usecols=["run_time", "c_vol", "c_cur", "c_surf_temp", "step_type"])
    m = (d["step_type"] == 22).to_numpy()
    if not m.any():
        return {"Capacity_Ah": np.nan, "check_temp_c": np.nan, "valid": False, "reason": "no step_type 22"}
    edges = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
    blocks = list(zip(edges[::2], edges[1::2]))
    for a, b in reversed(blocks):
        g = d.iloc[a:b]
        if g["c_vol"].min() <= V_END and abs(g["c_cur"].iloc[-1]) <= I_END:
            t, cur = seconds(g["run_time"]), g["c_cur"].to_numpy(float)
            q = abs(float(np.trapezoid(cur, t))) / 3600.0
            temp = float(g["c_surf_temp"].iloc[0])
            ok = T_LO <= temp <= T_HI
            return {"Capacity_Ah": q, "check_temp_c": temp, "valid": ok, "reason": "" if ok else "temperature",
                    "n_blocks": len(blocks)}
    return {"Capacity_Ah": np.nan, "check_temp_c": np.nan, "valid": False, "reason": "no complete block",
            "n_blocks": len(blocks)}


def one_zip(args) -> list[dict]:
    zid, name, folder = args
    files = sorted(p for p in Path(folder).glob("*.csv.gz") if FILE.match(p.name))
    et = [p for p in files if FILE.match(p.name).group(3) == "ET_T23"]
    if len(et) != 1:
        raise RuntimeError(f"{zid} {name}: {len(et)} ET_T23 files")
    stage = 1 if start_date(Path(str(et[0])[:-7] + "_meta.txt.gz")) < STAGE2_FROM else 2
    rows = []
    for p in files:
        cell, idx, kind = FILE.match(p.name).groups()
        if cell != name:
            raise RuntimeError(f"{zid}: file {p.name} inside {name}.zip")
        when = start_date(Path(str(p)[:-7] + "_meta.txt.gz"))
        row = {"file_id": zid, "serial": cell, "stage": stage, "check_index": int(idx), "rpt_kind": kind,
               "check_time": when, **capacity(p)}
        if pd.isna(when):
            row.update(valid=False, reason="unparseable date")
        rows.append(row)
    return rows


def order_rule(t: pd.DataFrame) -> pd.DataFrame:
    """Three cells have metadata dates out of order (e.g. TP_z25_01 stage 1: check-ups 9-29 all dated as check-up 3).
    In check-up order, a valid check-up is kept only if its date is after the last kept one and before the cell's final
    (AT_T23) check-up; the others are invalid ('date out of order'). Dates are never corrected."""
    t = t.copy()
    for _, g in t[t["valid"]].groupby("cell"):
        g = g.sort_values("check_index")
        final = g.loc[g["rpt_kind"] == "AT_T23", "check_time"].max()
        last = None
        for i, row in g.iterrows():
            ok = (last is None or row["check_time"] > last) and (
                row["rpt_kind"] == "AT_T23" or pd.isna(final) or row["check_time"] < final)
            if ok:
                last = row["check_time"]
            else:
                t.loc[i, ["valid", "reason"]] = [False, "date out of order"]
    return t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    raw = data_root() / "raw" / "stroebl2024"
    man = json.loads(MANIFEST.read_text())
    jobs = [(zid, z["name"][:-4], raw / zid) for zid, z in sorted(man["zips"].items())]
    with Pool(a.workers) as pool:
        rows = [r for rs in pool.map(one_zip, jobs) for r in rs]
    t = pd.DataFrame(rows)
    meta = pd.read_csv(raw / "experiments_meta.csv", dtype={"serial_internal": str})
    meta.columns = meta.columns.str.strip()
    pairs = t.groupby(["stage", "serial"])["file_id"].nunique()
    if (pairs != 1).any():
        raise SystemExit(f"(stage, cell) pairs with several zips: {pairs[pairs != 1].index.tolist()}")
    known = set(zip(meta["stage"], meta["serial"].str.strip()))
    missing = set(pairs.index) - known
    if missing or len(pairs) != len(meta):
        raise SystemExit(f"stage assignment does not match experiments_meta.csv: {sorted(missing)[:10]}, "
                         f"{len(pairs)} cells against {len(meta)}")
    meta = meta.assign(serial=meta["serial"].str.strip())
    t = t.merge(meta[["stage", "serial", "type", "tp", "lab", "amb_temp_tp", "soc_max_tp", "dod_tp", "c_ch_tp",
                      "c_dch_tp"]], on=["stage", "serial"], how="left", validate="many_to_one")
    t = t.assign(dataset="S10", cell="s" + t["stage"].astype(str) + "_" + t["serial"],
                 condition="s" + t["stage"].astype(str) + "_" + t["type"] + t["tp"].astype(int).map("{:02d}".format),
                 age_type=np.where(t["type"] == "k", 1, 2), age_temp=t["amb_temp_tp"].astype(float),
                 soc_max=t["soc_max_tp"].astype(float), dod=t["dod_tp"].astype(float),
                 age_chg_rate=t["c_ch_tp"].astype(float), age_dischg_rate=t["c_dch_tp"].astype(float))
    t = t.sort_values(["cell", "check_index"]).reset_index(drop=True)
    data = data_root()
    t.to_csv(data / "pool_s10_checks_all.csv", index=False)
    t = order_rule(t)
    t.to_csv(data / "pool_s10_checks_all.csv", index=False)
    v = t[t["valid"]].copy()
    v["Reference_Capacity_Ah"] = v.groupby("cell")["Capacity_Ah"].transform("first")
    v["SOH_Percent"] = v["Capacity_Ah"] / v["Reference_Capacity_Ah"] * 100
    v["n_valid_checks"] = v.groupby("cell")["cell"].transform("size")
    if (v.groupby("cell")["check_time"].apply(lambda s: s.is_monotonic_increasing) == False).any():  # noqa: E712
        raise SystemExit("check-up dates do not increase with the check-up index in some cell")
    keep = v[v["n_valid_checks"] >= MIN_CHECKS]
    cols = ["dataset", "cell", "condition", "stage", "lab", "check_index", "rpt_kind", "check_time", "Capacity_Ah",
            "check_temp_c", "age_type", "age_temp", "soc_max", "dod", "age_chg_rate", "age_dischg_rate",
            "Reference_Capacity_Ah", "SOH_Percent", "n_valid_checks"]
    keep[cols].to_csv(data / "pool_s10_checks.csv", index=False)
    print(f"{len(t)} check-ups in {t['cell'].nunique()} cells; invalid: {t.loc[~t['valid'], 'reason'].value_counts().to_dict()}")
    print(f"kept {len(keep)} valid check-ups, {keep['cell'].nunique()} cells, {keep['condition'].nunique()} conditions "
          f"(cells with >= {MIN_CHECKS}); dropped cells: {v['cell'].nunique() - keep['cell'].nunique()}")
    print(f"SOH range {keep['SOH_Percent'].min():.1f}-{keep['SOH_Percent'].max():.1f} %; reference capacity "
          f"{keep['Reference_Capacity_Ah'].min():.3f}-{keep['Reference_Capacity_Ah'].max():.3f} Ah")
    return 0


if __name__ == "__main__":
    sys.exit(main())
