"""Data loading, validation, temporal alignment, and SOH preprocessing.

The handover CSVs are not stored in Git. Every loader in this module resolves
files relative to ``DATA_ROOT`` (environment variable, default ``data``) and
fails with an explicit message when a file is missing. Nothing here fabricates
or imputes values; validation helpers only *report* problems.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATTERY_GROUPS: tuple[str, ...] = ("g1", "v4", "v5", "w8", "w9", "w10")

EXPECTED_FILES: dict[str, str] = {
    "cleaned_battery_data.csv": "Cleaned point-level cycling data",
    "rpt_capacity_data.csv": "Extracted RPT capacity measurements",
    "DATA_DICTIONARY.csv": "Column definitions",
    "ml_training_dataset.csv": "SOH-labelled subset (rows before first RPT excluded)",
    "final_ml_dataset.csv": "Complete cycle-level dataset with temporal RPT alignment",
    "final_battery_health_dataset.csv": "RPT capacity and SOH progression per battery",
    "final_battery_cycling_features.csv": "Cycle-level cycling features",
}

# Point-level file: profile in chunks instead of loading into memory.
LARGE_FILES: frozenset[str] = frozenset({"cleaned_battery_data.csv"})

DATE_COLUMNS: tuple[str, ...] = ("Cycle_Date", "Test_Date")
GROUP_COL = "Battery_ID"

ID_COLUMNS: tuple[str, ...] = ("Battery_ID", "Cycling_Test", "Cycle_Index", "Cycle_Date")

# Columns that come from the RPT / health side of the temporal join. They are
# labels or label-derived and must never be used as model inputs for SOH.
HEALTH_COLUMNS: tuple[str, ...] = (
    "RPT",
    "RPT_Number",
    "Capacity_Ah",
    "Reference_Capacity_Ah",
    "SOH_Percent",
    "Days_Since_RPT",
)

TEMPERATURE_COLUMNS: tuple[str, ...] = ("Temperature_Mean", "Temperature_Min", "Temperature_Max")

# Cycle-level features documented in README / DATA_DICTIONARY.csv. The handover
# describes "22 cycling features"; notebook 03 reconciles this list against the
# dictionary and the observed file columns rather than assuming the count.
CYCLING_FEATURE_CANDIDATES: tuple[str, ...] = (
    "Cycle_Index",
    "Voltage_Mean",
    "Voltage_Min",
    "Voltage_Max",
    "Current_Mean",
    "Current_Min",
    "Current_Max",
    "Temperature_Mean",
    "Temperature_Min",
    "Temperature_Max",
    "Charge_Capacity_Max",
    "Discharge_Capacity_Max",
    "Charge_Energy_Max",
    "Discharge_Energy_Max",
    "Power_Mean",
    "Capacity_Ratio_Mean",
    "Energy_Efficiency_Mean",
    "Data_Points",
)

# Loose physical-plausibility bounds used only to *flag* records.
PLAUSIBILITY_BOUNDS: dict[str, tuple[float, float]] = {
    "Voltage_Mean": (0.0, 5.0),
    "Voltage_Min": (0.0, 5.0),
    "Voltage_Max": (0.0, 5.0),
    "Temperature_Mean": (-40.0, 100.0),
    "Temperature_Min": (-40.0, 100.0),
    "Temperature_Max": (-40.0, 100.0),
    "Charge_Capacity_Max": (0.0, np.inf),
    "Discharge_Capacity_Max": (0.0, np.inf),
    "Charge_Energy_Max": (0.0, np.inf),
    "Discharge_Energy_Max": (0.0, np.inf),
    "Capacity_Ratio_Mean": (0.0, 1.5),
    "Energy_Efficiency_Mean": (0.0, 1.5),
    "Data_Points": (1.0, np.inf),
    "Capacity_Ah": (0.0, np.inf),
    "Reference_Capacity_Ah": (0.0, np.inf),
    "SOH_Percent": (0.0, 150.0),
    "Days_Since_RPT": (0.0, np.inf),
}


class DataFilesMissingError(FileNotFoundError):
    """Raised when DATA_ROOT or required handover files are not available."""


# ---------------------------------------------------------------------------
# Paths and file inventory
# ---------------------------------------------------------------------------


def find_repo_root(start: Path | str | None = None) -> Path:
    """Return the nearest ancestor containing ``src/data_pipeline.py``."""
    here = Path(start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "src" / "data_pipeline.py").is_file():
            return candidate
    raise RuntimeError(
        f"Could not locate the repository root from {here}. "
        "Run the notebooks from inside the cloned repository."
    )


def resolve_data_root(data_root: Path | str | None = None, repo_root: Path | None = None) -> Path:
    """Resolve ``DATA_ROOT``; relative paths are taken relative to the repo root."""
    raw = data_root if data_root is not None else os.environ.get("DATA_ROOT", "data")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (repo_root or find_repo_root()) / path
    return path


def check_data_files(
    data_root: Path,
    required: Iterable[str] | None = None,
    expected: Mapping[str, str] = EXPECTED_FILES,
) -> pd.DataFrame:
    """Inventory the expected files and raise if DATA_ROOT or a required file is missing.

    ``required`` defaults to every expected file. The returned table lists every
    expected file with presence and size so notebooks can display it.
    """
    required = list(expected) if required is None else list(required)
    hint = (
        "Provide the handover CSVs outside Git and point DATA_ROOT at them, e.g.\n"
        "    export DATA_ROOT=/mnt/project-data\n"
        "or copy them into <repo>/data/. See data/README.md."
    )
    if not data_root.is_dir():
        raise DataFilesMissingError(f"DATA_ROOT does not exist or is not a directory: {data_root}\n{hint}")

    rows = []
    for name, description in expected.items():
        path = data_root / name
        present = path.is_file()
        rows.append(
            {
                "file": name,
                "description": description,
                "present": present,
                "required": name in required,
                "size_mb": round(path.stat().st_size / 1e6, 3) if present else np.nan,
            }
        )
    inventory = pd.DataFrame(rows)

    missing = inventory.loc[inventory["required"] & ~inventory["present"], "file"].tolist()
    if missing:
        raise DataFilesMissingError(
            f"Missing required file(s) in DATA_ROOT={data_root}:\n  - "
            + "\n  - ".join(missing)
            + f"\nNo substitute data is generated.\n{hint}"
        )
    return inventory


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def read_header(path: Path) -> list[str]:
    """Return the column names of a CSV without reading its body."""
    return list(pd.read_csv(path, nrows=0).columns)


def _date_columns(columns: Sequence[str]) -> list[str]:
    return [c for c in DATE_COLUMNS if c in columns]


def read_dataset(path: Path, usecols: Sequence[str] | None = None, **kwargs) -> pd.DataFrame:
    """``pd.read_csv`` with project defaults.

    Parses ``Cycle_Date`` / ``Test_Date`` when present, uses ``low_memory=False``
    and only requests ``usecols`` that actually exist (missing ones are reported
    by raising, never silently dropped).
    """
    header = read_header(path)
    if usecols is not None:
        absent = [c for c in usecols if c not in header]
        if absent:
            raise KeyError(f"{path.name} has no column(s) {absent}. Available: {header}")
        columns = list(usecols)
    else:
        columns = header
    kwargs.setdefault("low_memory", False)
    return pd.read_csv(path, usecols=usecols, parse_dates=_date_columns(columns), **kwargs)


def profile_csv_chunked(
    path: Path,
    chunksize: int = 500_000,
    usecols: Sequence[str] | None = None,
    group_col: str = GROUP_COL,
) -> dict:
    """Row counts, per-group counts, null counts and date ranges computed chunk by chunk."""
    header = read_header(path)
    usecols = [c for c in (usecols or header) if c in header]
    date_cols = _date_columns(usecols)
    n_rows = 0
    nulls = pd.Series(0, index=usecols, dtype="int64")
    per_group: pd.Series = pd.Series(dtype="int64")
    date_min = {c: pd.NaT for c in date_cols}
    date_max = {c: pd.NaT for c in date_cols}
    dtypes: dict[str, set] = {c: set() for c in usecols}

    for chunk in pd.read_csv(path, usecols=usecols, parse_dates=date_cols, chunksize=chunksize, low_memory=False):
        n_rows += len(chunk)
        nulls = nulls.add(chunk.isna().sum(), fill_value=0).astype("int64")
        for c in usecols:
            dtypes[c].add(str(chunk[c].dtype))
        if group_col in chunk:
            per_group = per_group.add(chunk[group_col].astype(str).value_counts(), fill_value=0).astype("int64")
        for c in date_cols:
            parsed = pd.to_datetime(chunk[c], errors="coerce")
            date_min[c] = pd.Series([date_min[c], parsed.min()]).min()
            date_max[c] = pd.Series([date_max[c], parsed.max()]).max()

    return {
        "file": path.name,
        "rows": n_rows,
        "columns": header,
        "dtypes": {c: "/".join(sorted(v)) for c, v in dtypes.items()},
        "null_counts": nulls,
        "rows_per_group": per_group.sort_index(),
        "date_ranges": {c: (date_min[c], date_max[c]) for c in date_cols},
    }


def dtype_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Column-level dtype and missingness table."""
    missing = df.isna().sum()
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null": df.notna().sum(),
            "missing": missing,
            "missing_pct": (missing / max(len(df), 1) * 100).round(3),
        }
    )


def date_ranges(df: pd.DataFrame, date_col: str, group_col: str = GROUP_COL) -> pd.DataFrame:
    """Per-group min/max of a datetime column plus NaT counts."""
    return df.groupby(group_col, sort=True)[date_col].agg(
        start="min", end="max", n="size", n_missing=lambda s: s.isna().sum()
    )


# ---------------------------------------------------------------------------
# Data dictionary
# ---------------------------------------------------------------------------


def load_data_dictionary(path: Path) -> pd.DataFrame:
    """Load DATA_DICTIONARY.csv and add a normalised ``column_name`` field.

    The column that holds variable names is detected by header (``Column``,
    ``Column_Name``, ``Field``, ``Variable``, ``Name``); otherwise the first
    column is used.
    """
    dd = pd.read_csv(path, low_memory=False)
    candidates = ["column", "column_name", "columnname", "field", "variable", "name", "feature"]
    lookup = {c.strip().lower().replace(" ", "_"): c for c in dd.columns}
    name_col = next((lookup[c] for c in candidates if c in lookup), dd.columns[0])
    dd = dd.copy()
    dd["column_name"] = dd[name_col].astype(str).str.strip()
    dd.attrs["name_column"] = name_col
    return dd


def compare_dictionary(dictionary_columns: Iterable[str], observed: Mapping[str, Iterable[str]]) -> pd.DataFrame:
    """Matrix of dictionary columns vs. files, plus columns missing from the dictionary."""
    dict_cols = list(dict.fromkeys(dictionary_columns))
    all_observed = list(dict.fromkeys(c for cols in observed.values() for c in cols))
    index = dict_cols + [c for c in all_observed if c not in dict_cols]
    table = pd.DataFrame(index=pd.Index(index, name="column"))
    table["in_dictionary"] = table.index.isin(dict_cols)
    for name, cols in observed.items():
        table[name] = table.index.isin(list(cols))
    table["n_files"] = table[list(observed)].sum(axis=1)
    return table


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def check_battery_groups(values: Iterable, expected: Sequence[str] = BATTERY_GROUPS) -> dict:
    observed = sorted({str(v) for v in values if pd.notna(v)})
    return {
        "observed": observed,
        "missing": [g for g in expected if g not in observed],
        "unexpected": [g for g in observed if g not in expected],
        "ok": set(observed) == set(expected),
    }


def validate_soh_formula(df: pd.DataFrame, atol: float = 1e-6, rtol: float = 1e-6) -> dict:
    """Check ``SOH_Percent == Capacity_Ah / Reference_Capacity_Ah * 100`` on rows where all three exist."""
    cols = ["Capacity_Ah", "Reference_Capacity_Ah", "SOH_Percent"]
    absent = [c for c in cols if c not in df]
    if absent:
        return {"checked": 0, "skipped_reason": f"missing columns {absent}"}
    sub = df[cols].dropna()
    expected = sub["Capacity_Ah"] / sub["Reference_Capacity_Ah"] * 100
    err = (sub["SOH_Percent"] - expected).abs()
    bad = ~np.isclose(sub["SOH_Percent"], expected, atol=atol, rtol=rtol)
    return {
        "checked": int(len(sub)),
        "rows_with_missing_inputs": int(len(df) - len(sub)),
        "mismatches": int(bad.sum()),
        "max_abs_error": float(err.max()) if len(err) else np.nan,
        "ok": bool(len(sub) > 0 and not bad.any()),
    }


def _rpt_sort_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in (GROUP_COL, "Test_Date", "RPT_Number") if c in df]


def validate_reference_capacity(health: pd.DataFrame, atol: float = 1e-9) -> pd.DataFrame:
    """Per battery: is ``Reference_Capacity_Ah`` the first observed RPT capacity?

    "First observed" is taken chronologically (``Test_Date``), with
    ``RPT_Number`` as tie-breaker; the RPT_Number-first view is reported too.
    """
    h = health.dropna(subset=["Capacity_Ah"]).sort_values(_rpt_sort_cols(health), kind="mergesort")
    rows = []
    for battery, g in h.groupby(GROUP_COL, sort=True):
        refs = g["Reference_Capacity_Ah"].dropna().unique() if "Reference_Capacity_Ah" in g else []
        first_by_date = g["Capacity_Ah"].iloc[0]
        first_by_number = (
            g.sort_values("RPT_Number", kind="mergesort")["Capacity_Ah"].iloc[0] if "RPT_Number" in g else np.nan
        )
        ref = refs[0] if len(refs) else np.nan
        rows.append(
            {
                GROUP_COL: battery,
                "n_rpt": len(g),
                "first_rpt": g["RPT"].iloc[0] if "RPT" in g else None,
                "first_test_date": g["Test_Date"].iloc[0] if "Test_Date" in g else pd.NaT,
                "first_capacity_by_date": first_by_date,
                "first_capacity_by_rpt_number": first_by_number,
                "reference_values": len(refs),
                "Reference_Capacity_Ah": ref,
                "ok": len(refs) == 1 and np.isclose(ref, first_by_date, atol=atol),
            }
        )
    return pd.DataFrame(rows).set_index(GROUP_COL)


def align_to_rpt(
    cycles: pd.DataFrame,
    rpt: pd.DataFrame,
    cycle_time: str = "Cycle_Date",
    rpt_time: str = "Test_Date",
    rpt_cols: Sequence[str] = ("RPT", "RPT_Number", "Capacity_Ah", "Reference_Capacity_Ah", "SOH_Percent"),
    suffix: str = "_expected",
) -> pd.DataFrame:
    """Attach the most recent RPT **on or before** each cycle (backward as-of join per battery).

    Cycles without a parseable date, or before the first RPT of their battery,
    get NaN health columns. The original row order is restored.
    """
    rpt_cols = [c for c in rpt_cols if c in rpt]
    right = (
        rpt.dropna(subset=[rpt_time])[[GROUP_COL, rpt_time, *rpt_cols]]
        .rename(columns={c: f"{c}{suffix}" for c in rpt_cols})
        .rename(columns={rpt_time: f"{rpt_time}{suffix}"})
    )
    right[GROUP_COL] = right[GROUP_COL].astype(str)
    right_time = f"{rpt_time}{suffix}"
    right = right.sort_values([right_time, GROUP_COL], kind="mergesort")

    left = cycles.copy()
    left["_row"] = np.arange(len(left))
    left["_group"] = left[GROUP_COL].astype(str)
    dated = left.dropna(subset=[cycle_time]).sort_values(cycle_time, kind="mergesort")
    merged = pd.merge_asof(
        dated,
        right.rename(columns={GROUP_COL: "_group"}),
        left_on=cycle_time,
        right_on=right_time,
        by="_group",
        direction="backward",
        allow_exact_matches=True,
    )
    undated = left[left[cycle_time].isna()]
    out = pd.concat([merged, undated], ignore_index=True).sort_values("_row", kind="mergesort")
    out["Days_Since_RPT_expected"] = (out[cycle_time] - out[right_time]).dt.total_seconds() / 86400.0
    return out.drop(columns=["_row", "_group"]).reset_index(drop=True)


def validate_rpt_alignment(
    cycles: pd.DataFrame,
    health: pd.DataFrame,
    soh_atol: float = 1e-6,
    days_atol: float = 1.0,
) -> tuple[dict, pd.DataFrame]:
    """Recompute the RPT alignment and compare with what the dataset carries.

    Returns an overall summary and a per-battery table. Checks:
    * the assigned ``RPT`` equals the most recent RPT on or before ``Cycle_Date``;
    * ``SOH_Percent`` equals that RPT's SOH (both-missing counts as agreement);
    * ``Days_Since_RPT`` is non-negative and within ``days_atol`` of the recomputed gap
      (tolerant to whole-day vs fractional-day conventions);
    * no row carries health information from an RPT dated after the cycle.
    """
    aligned = align_to_rpt(cycles, health)

    def same(a: pd.Series, b: pd.Series, numeric: bool) -> pd.Series:
        both_na = a.isna() & b.isna()
        if numeric:
            eq = pd.Series(np.isclose(a.astype(float), b.astype(float), atol=soh_atol), index=a.index)
        else:
            eq = a.astype(str) == b.astype(str)
        return both_na | (eq & a.notna() & b.notna())

    checks = pd.DataFrame({GROUP_COL: aligned[GROUP_COL].astype(str)})
    if "RPT" in aligned and "RPT_expected" in aligned:
        checks["rpt_match"] = same(aligned["RPT"], aligned["RPT_expected"], numeric=False)
    if "SOH_Percent" in aligned and "SOH_Percent_expected" in aligned:
        checks["soh_match"] = same(aligned["SOH_Percent"], aligned["SOH_Percent_expected"], numeric=True)
    if "Days_Since_RPT" in aligned:
        d = aligned["Days_Since_RPT"].astype(float)
        exp = aligned["Days_Since_RPT_expected"]
        checks["days_nonnegative"] = d.isna() | (d >= 0)
        checks["days_match"] = (d.isna() & exp.isna()) | ((d - exp).abs() <= days_atol)
    # Future leakage: the dataset's RPT maps to a Test_Date later than the cycle.
    if "RPT" in aligned:
        rpt_dates = health.dropna(subset=["Test_Date"]).assign(**{GROUP_COL: lambda x: x[GROUP_COL].astype(str)})
        rpt_dates = rpt_dates.drop_duplicates([GROUP_COL, "RPT"]).set_index([GROUP_COL, "RPT"])["Test_Date"]
        key = pd.MultiIndex.from_arrays([aligned[GROUP_COL].astype(str), aligned["RPT"]])
        assigned_date = pd.Series(rpt_dates.reindex(key).to_numpy(), index=aligned.index)
        checks["no_future_rpt"] = assigned_date.isna() | (assigned_date <= aligned["Cycle_Date"])
    checks["before_first_rpt"] = aligned["RPT_expected"].isna() if "RPT_expected" in aligned else False

    flag_cols = [c for c in checks if c not in (GROUP_COL, "before_first_rpt")]
    per_battery = checks.groupby(GROUP_COL).agg(
        rows=(GROUP_COL, "size"),
        before_first_rpt=("before_first_rpt", "sum"),
        **{f"{c}_failures": (c, lambda s: int((~s).sum())) for c in flag_cols},
    )
    summary = {
        "rows": int(len(checks)),
        "undated_rows": int(aligned["Cycle_Date"].isna().sum()),
        "rows_before_first_rpt": int(checks["before_first_rpt"].sum()),
        **{f"{c}_failures": int((~checks[c]).sum()) for c in flag_cols},
    }
    summary["ok"] = all(summary[f"{c}_failures"] == 0 for c in flag_cols)
    return summary, per_battery


def validate_training_subset(
    ml_train: pd.DataFrame, final_ml: pd.DataFrame, health: pd.DataFrame
) -> pd.DataFrame:
    """Check that ml_training excludes pre-RPT rows and matches final_ml's labelled rows."""
    first_rpt = health.dropna(subset=["Test_Date"]).groupby(health[GROUP_COL].astype(str))["Test_Date"].min()
    rows = []
    for battery in sorted(set(final_ml[GROUP_COL].astype(str)) | set(ml_train[GROUP_COL].astype(str))):
        tr = ml_train[ml_train[GROUP_COL].astype(str) == battery]
        fm = final_ml[final_ml[GROUP_COL].astype(str) == battery]
        first = first_rpt.get(battery, pd.NaT)
        fm_before = fm[fm["Cycle_Date"] < first] if pd.notna(first) else fm.iloc[0:0]
        rows.append(
            {
                GROUP_COL: battery,
                "first_rpt_date": first,
                "train_rows": len(tr),
                "final_rows": len(fm),
                "final_rows_with_soh": int(fm["SOH_Percent"].notna().sum()),
                "final_rows_before_first_rpt": len(fm_before),
                "final_pre_rpt_rows_with_soh": int(fm_before["SOH_Percent"].notna().sum()),
                "train_rows_before_first_rpt": int((tr["Cycle_Date"] < first).sum()) if pd.notna(first) else len(tr),
                "train_missing_soh": int(tr["SOH_Percent"].isna().sum()),
                "train_min_cycle_date": tr["Cycle_Date"].min(),
            }
        )
    out = pd.DataFrame(rows).set_index(GROUP_COL)
    out["ok"] = (
        (out["train_rows_before_first_rpt"] == 0)
        & (out["train_missing_soh"] == 0)
        & (out["final_pre_rpt_rows_with_soh"] == 0)
        & (out["train_rows"] == out["final_rows_with_soh"])
    )
    return out


def missing_by_group(df: pd.DataFrame, columns: Sequence[str], group_col: str = GROUP_COL) -> pd.DataFrame:
    """Missing counts and percentages per group for the given columns."""
    columns = [c for c in columns if c in df]
    grouped = df[columns].isna().groupby(df[group_col])
    counts = grouped.sum().add_suffix("_missing")
    pct = (grouped.mean() * 100).round(2).add_suffix("_missing_pct")
    out = pd.concat([df.groupby(group_col).size().rename("rows"), counts, pct], axis=1)
    out["any_temperature_missing"] = df[columns].isna().any(axis=1).groupby(df[group_col]).sum()
    return out


def find_duplicates(df: pd.DataFrame, key: Sequence[str]) -> dict:
    key = [c for c in key if c in df]
    return {
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "key": key,
        "duplicate_keys": int(df.duplicated(subset=key, keep=False).sum()) if key else 0,
    }


def find_out_of_order(
    df: pd.DataFrame, time_col: str = "Cycle_Date", index_col: str = "Cycle_Index", segment_cols: Sequence[str] = (GROUP_COL, "Cycling_Test")
) -> pd.DataFrame:
    """Count temporal-order violations per battery.

    * ``date_decreases_in_file_order``: Cycle_Date goes backwards between consecutive
      rows of the same battery, in the order the file stores them.
    * ``date_decreases_by_index``: within a (Battery_ID, Cycling_Test) segment, sorted
      by Cycle_Index, Cycle_Date goes backwards.
    * ``index_repeats``: repeated Cycle_Index inside a segment.
    """
    segs = [c for c in segment_cols if c in df]
    res = pd.DataFrame(index=pd.Index(sorted(df[GROUP_COL].astype(str).unique()), name=GROUP_COL))
    g = df.groupby(df[GROUP_COL].astype(str), sort=False)[time_col]
    res["date_decreases_in_file_order"] = (g.diff().dt.total_seconds() < 0).groupby(df[GROUP_COL].astype(str)).sum()
    if index_col in df:
        s = df.sort_values([*segs, index_col], kind="mergesort")
        seg_key = [s[c].astype(str) for c in segs]
        res["date_decreases_by_index"] = (
            (s.groupby(seg_key)[time_col].diff().dt.total_seconds() < 0).groupby(s[GROUP_COL].astype(str)).sum()
        )
        res["index_repeats"] = s.duplicated(subset=[*segs, index_col]).groupby(s[GROUP_COL].astype(str)).sum()
    res["missing_dates"] = df[time_col].isna().groupby(df[GROUP_COL].astype(str)).sum()
    return res.fillna(0).astype(int)


def find_invalid_values(df: pd.DataFrame, bounds: Mapping[str, tuple[float, float]] = PLAUSIBILITY_BOUNDS) -> pd.DataFrame:
    """Count values outside loose plausibility bounds (NaNs are not counted as invalid)."""
    rows = []
    for col, (lo, hi) in bounds.items():
        if col not in df or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        s = df[col]
        rows.append(
            {
                "column": col,
                "lower": lo,
                "upper": hi,
                "below": int((s < lo).sum()),
                "above": int((s > hi).sum()),
                "non_finite": int(np.isinf(s).sum()),
                "min": s.min(),
                "max": s.max(),
            }
        )
    out = pd.DataFrame(rows).set_index("column")
    out["flagged"] = out[["below", "above", "non_finite"]].sum(axis=1)
    return out


# ---------------------------------------------------------------------------
# Features, splits and metrics
# ---------------------------------------------------------------------------


def identify_cycling_features(
    dictionary_columns: Iterable[str], feature_file_columns: Iterable[str]
) -> pd.DataFrame:
    """Classify each documented/observed column as identifier, health label, or cycling feature."""
    dict_cols = list(dictionary_columns)
    file_cols = list(feature_file_columns)
    names = list(dict.fromkeys(dict_cols + file_cols))

    def role(c: str) -> str:
        if c in HEALTH_COLUMNS:
            return "health/label (RPT join)"
        if c in ID_COLUMNS and c != "Cycle_Index":
            return "identifier/time"
        return "cycling feature"

    return pd.DataFrame(
        {
            "column": names,
            "role": [role(c) for c in names],
            "in_dictionary": [c in dict_cols for c in names],
            "in_cycling_features_file": [c in file_cols for c in names],
            "in_candidate_list": [c in CYCLING_FEATURE_CANDIDATES for c in names],
        }
    ).set_index("column")


def model_feature_columns(df: pd.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    """Numeric input columns for SOH regression: excludes identifiers and every health/label column."""
    banned = set(HEALTH_COLUMNS) | {c for c in ID_COLUMNS if c != "Cycle_Index"} | set(exclude)
    return [c for c in df.columns if c not in banned and pd.api.types.is_numeric_dtype(df[c])]


def temporal_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    time_cols: Sequence[str] = ("Cycle_Date", "Cycle_Index"),
    group_col: str = GROUP_COL,
) -> pd.DataFrame:
    """Chronological per-battery split: first ``train_frac`` → train, next ``val_frac`` → val, rest → test.

    Rows are never shuffled. Each battery contributes to every split, so the
    model sees each group's early life and is evaluated on its later life.
    Returns a copy sorted by battery and time with a ``split`` column.
    """
    if not 0 < train_frac < 1 or not 0 <= val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("Need 0 < train_frac, 0 <= val_frac and train_frac + val_frac < 1")
    order = [group_col, *[c for c in time_cols if c in df]]
    out = df.sort_values(order, kind="mergesort").reset_index(drop=True)
    pos = out.groupby(group_col).cumcount()
    n = out.groupby(group_col)[group_col].transform("size")
    frac = pos / n
    out["split"] = np.where(frac < train_frac, "train", np.where(frac < train_frac + val_frac, "val", "test"))
    return out


def check_split_ordering(split_df: pd.DataFrame, time_col: str = "Cycle_Date", group_col: str = GROUP_COL) -> pd.DataFrame:
    """Per battery: train ends no later than val starts, and val ends no later than test starts."""
    agg = split_df.groupby([group_col, "split"])[time_col].agg(["min", "max", "size"]).unstack("split")
    res = pd.DataFrame(index=agg.index)
    for s in ("train", "val", "test"):
        res[f"{s}_rows"] = agg[("size", s)] if ("size", s) in agg else 0
    res["train_end"] = agg.get(("max", "train"))
    res["val_start"] = agg.get(("min", "val"))
    res["val_end"] = agg.get(("max", "val"))
    res["test_start"] = agg.get(("min", "test"))
    res["train_before_val"] = res["val_start"].isna() | (res["train_end"] <= res["val_start"])
    res["val_before_test"] = res["test_start"].isna() | (res["val_end"].isna()) | (res["val_end"] <= res["test_start"])
    return res


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict:
    """RMSE, MAE, MAPE (%) and R²; MAPE ignores rows with y_true == 0."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    nz = y_true != 0
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) if len(y_true) else 0.0
    return {
        "n": int(len(y_true)),
        "RMSE": float(np.sqrt(np.mean(err**2))) if len(y_true) else np.nan,
        "MAE": float(np.mean(np.abs(err))) if len(y_true) else np.nan,
        "MAPE": float(np.mean(np.abs(err[nz] / y_true[nz])) * 100) if nz.any() else np.nan,
        # R² is undefined when the target is constant (common within one RPT interval).
        "R2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
    }


def metrics_by_group(
    df: pd.DataFrame, y_true: str, y_pred: str, group_col: str = GROUP_COL
) -> pd.DataFrame:
    """Regression metrics per group plus an ``ALL`` row."""
    rows = {str(g): regression_metrics(sub[y_true], sub[y_pred]) for g, sub in df.groupby(group_col, sort=True)}
    rows["ALL"] = regression_metrics(df[y_true], df[y_pred])
    return pd.DataFrame(rows).T
