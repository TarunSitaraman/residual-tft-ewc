"""Robustness to missing temperature (E11, decision D13).

Three questions:

1. **Natural missingness** — are errors higher on cycles whose temperature is
   missing? (``stratified_errors``)
2. **Handling** — does the way missing temperature is handled matter?
   (``HANDLING_VARIANTS``, ``prepare_variant``)
3. **Stress** — how does error grow when more temperature values are
   *removed* from the test periods? (``mask_temperature``)

Nothing here fabricates data. Masking only removes values (sets them missing);
carry-forward fills a gap with the same battery's most recent *earlier*
observation and is applied identically to every split, so no future or test
information enters training.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src import data_pipeline as dp

TARGET = "SOH_Percent"
TEMPS: tuple[str, ...] = dp.TEMPERATURE_COLUMNS
ORDER = ("Cycle_Date", "Cycle_Index")

#: Handling strategies compared in E11.
HANDLING_VARIANTS: dict[str, str] = {
    "impute_indicator": "Training median + missing indicator (default)",
    "impute_only": "Training median, indicators hidden from the model",
    "carry_forward": "Battery's last earlier observation, then default handling for leading gaps",
    "drop_temperature": "Temperature features removed",
}


def temperature_missing(df: pd.DataFrame, columns: Sequence[str] = TEMPS) -> pd.Series:
    """True where any temperature column is missing."""
    cols = [c for c in columns if c in df]
    return df[cols].isna().any(axis=1)


KEY = ("Battery_ID", "Cycling_Test", "Cycle_Index")


def rows_in(df: pd.DataFrame, subset: pd.DataFrame) -> pd.Series:
    """Boolean mask of ``df`` rows that appear in ``subset``, matched by cycle key (not index).

    Task frames have reset indices, so index-based matching would select the
    wrong cycles.
    """
    idx = pd.MultiIndex.from_frame(df[list(KEY)].astype({"Battery_ID": str}))
    sub = pd.MultiIndex.from_frame(subset[list(KEY)].astype({"Battery_ID": str}))
    return pd.Series(idx.isin(sub), index=df.index)


def carry_forward_temperature(df: pd.DataFrame, columns: Sequence[str] = TEMPS) -> pd.DataFrame:
    """Fill missing temperatures with the same battery's most recent earlier value (causal).

    Rows are processed in time order within each battery; values never flow
    backwards in time or across batteries. Gaps before a battery's first
    observed temperature stay missing. Row order of ``df`` is preserved.
    """
    cols = [c for c in columns if c in df]
    out = df.copy()
    order = out.sort_values([dp.GROUP_COL, *ORDER], kind="mergesort").index
    filled = out.loc[order].groupby(out.loc[order, dp.GROUP_COL].astype(str))[cols].ffill()
    out.loc[order, cols] = filled.to_numpy()
    return out


def prepare_variant(variant: str, df: pd.DataFrame, features: Sequence[str]) -> tuple[pd.DataFrame, list[str], bool]:
    """Data, feature list and indicator flag for one handling variant.

    Returns ``(data, features, indicators)``; ``data`` is transformed before any
    split, which is safe because every transformation is causal or row-wise.
    """
    features = list(features)
    if variant == "impute_indicator":
        return df, features, True
    if variant == "impute_only":
        return df, features, False
    if variant == "carry_forward":
        return carry_forward_temperature(df), features, True
    if variant == "drop_temperature":
        return df, [f for f in features if f not in TEMPS], True
    raise ValueError(f"unknown variant {variant!r}; choose from {sorted(HANDLING_VARIANTS)}")


def mask_temperature(df: pd.DataFrame, rows: pd.Series, rate: float, mode: str = "random", seed: int = 0,
                     columns: Sequence[str] = TEMPS) -> tuple[pd.DataFrame, pd.Series]:
    """Remove temperature values from a share ``rate`` of the ``rows`` (boolean mask) of each battery.

    All temperature columns of a chosen cycle are removed together, matching the
    observed pattern in which the three go missing jointly.

    * ``"random"``: cycles chosen independently (seeded).
    * ``"block"``: one contiguous run of cycles per battery in time order, a
      sensor-outage pattern (seeded start).

    Values are only ever removed. Returns the masked copy and a boolean Series
    marking the rows masked by this call.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be in [0, 1]")
    if mode not in ("random", "block"):
        raise ValueError("mode must be 'random' or 'block'")
    cols = [c for c in columns if c in df]
    rng = np.random.default_rng(seed)
    masked = pd.Series(False, index=df.index)
    eligible = df[rows.reindex(df.index, fill_value=False).astype(bool)]
    for _, g in eligible.sort_values([dp.GROUP_COL, *ORDER], kind="mergesort").groupby(dp.GROUP_COL, sort=True):
        n = len(g)
        k = int(round(rate * n))
        if k == 0:
            continue
        if mode == "random":
            chosen = g.index[np.sort(rng.choice(n, size=k, replace=False))]
        else:
            start = int(rng.integers(0, n - k + 1))
            chosen = g.index[start:start + k]
        masked.loc[chosen] = True
    out = df.copy()
    out.loc[masked, cols] = np.nan
    return out, masked


def stratified_errors(df: pd.DataFrame, pred: Sequence[float], flag: pd.Series, labels=("complete", "temperature missing")) -> pd.DataFrame:
    """RMSE/MAE (and counts) per battery group and pooled, split by a boolean row flag."""
    frame = pd.DataFrame({dp.GROUP_COL: df[dp.GROUP_COL].astype(str).to_numpy(), "y": df[TARGET].to_numpy(float),
                          "pred": np.asarray(pred, float), "flag": flag.reindex(df.index).to_numpy(bool)})
    rows = []
    for group, g in [*frame.groupby(dp.GROUP_COL, sort=True), ("ALL", frame)]:
        for value, label in zip((False, True), labels):
            sub = g[g["flag"] == value]
            m = dp.regression_metrics(sub["y"], sub["pred"]) if len(sub) else {"n": 0, "RMSE": np.nan, "MAE": np.nan}
            rows.append({dp.GROUP_COL: group, "subset": label, "n": int(m["n"]), "RMSE": m["RMSE"], "MAE": m["MAE"]})
    return pd.DataFrame(rows)
