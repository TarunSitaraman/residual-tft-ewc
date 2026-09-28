"""Task and stream construction for reference baselines and continual learning.

See docs/RESEARCH_PLAN.md §3–4. A task is one battery group split
chronologically into train / validation / test; nothing is shuffled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src import data_pipeline as dp

SPLITS: tuple[str, ...] = ("train", "val", "test")
TIME_COLS: tuple[str, ...] = ("Cycle_Date", "Cycle_Index")


@dataclass(frozen=True)
class Task:
    """One battery group's chronological train / validation / test periods."""

    name: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


# ---------------------------------------------------------------------------
# Tasks and streams (Stage B, E1)
# ---------------------------------------------------------------------------


def _groups_in(df: pd.DataFrame) -> set[str]:
    return set(df[dp.GROUP_COL].astype(str).unique())


def _periods(split_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split a frame carrying a ``split`` column into train / val / test frames (column dropped)."""
    return {
        s: split_df[split_df["split"] == s].drop(columns="split").reset_index(drop=True)
        for s in SPLITS
    }


def make_task(df: pd.DataFrame, group: str, train_frac: float = 0.70, val_frac: float = 0.15) -> Task:
    """One battery group as a task, split chronologically with :func:`dp.temporal_split`.

    Rows are ordered by ``Cycle_Date`` then ``Cycle_Index`` and never shuffled, so
    the task is evaluated on the group's later life. An empty period is an error
    rather than a silently degenerate task.
    """
    group = str(group)
    if group not in _groups_in(df):
        raise ValueError(f"Unknown battery group {group!r}; available: {sorted(_groups_in(df))}")
    rows = df[df[dp.GROUP_COL].astype(str) == group]
    periods = _periods(dp.temporal_split(rows, train_frac, val_frac, time_cols=TIME_COLS))
    empty = [s for s, frame in periods.items() if frame.empty]
    if empty:
        raise ValueError(f"Group {group!r} has an empty {'/'.join(empty)} period ({len(rows)} rows)")
    return Task(name=group, **periods)


def make_task_stream(
    df: pd.DataFrame, order: Sequence[str], train_frac: float = 0.70, val_frac: float = 0.15
) -> list[Task]:
    """One :class:`Task` per group, in ``order`` (the continual-learning stream)."""
    order = [str(g) for g in order]
    if len(order) == 0:
        raise ValueError("order must contain at least one group")
    if len(set(order)) != len(order):
        raise ValueError(f"order contains duplicate groups: {order}")
    missing = [g for g in order if g not in _groups_in(df)]
    if missing:
        raise ValueError(f"order contains groups absent from the data: {missing}")
    return [make_task(df, g, train_frac, val_frac) for g in order]


def sample_orders(groups: Sequence[str], n_orders: int, seed: int) -> list[tuple[str, ...]]:
    """Task orders shared by every method: ``groups`` as given, then distinct seeded permutations.

    Deterministic for a given ``(groups, n_orders, seed)``; a larger ``n_orders``
    extends, and never changes, the orders drawn for a smaller one.
    """
    groups = tuple(str(g) for g in groups)
    if len(set(groups)) != len(groups):
        raise ValueError(f"groups contains duplicates: {groups}")
    if n_orders < 1:
        raise ValueError("n_orders must be >= 1")
    n_possible = math.factorial(len(groups))
    if n_orders > n_possible:
        raise ValueError(f"n_orders={n_orders} exceeds the {n_possible} distinct orders of {len(groups)} groups")
    rng = np.random.default_rng(seed)
    orders = [groups]
    seen = {groups}
    while len(orders) < n_orders:
        candidate = tuple(groups[i] for i in rng.permutation(len(groups)))
        if candidate not in seen:
            seen.add(candidate)
            orders.append(candidate)
    return orders


# ---------------------------------------------------------------------------
# Reference protocols (Stage A: E3, E2, E5)
# ---------------------------------------------------------------------------


def leave_one_group_out(
    df: pd.DataFrame, held_out: str, train_frac: float = 0.70, val_frac: float = 0.15
) -> Task:
    """E3: train/validate on the other groups, test on the held-out group's test period.

    Train and validation are the other groups' chronological train and val
    periods (their test periods are unused). The test set is exactly
    ``make_task(df, held_out).test`` so E3 and E1 are scored on the same window;
    the held-out group's earlier rows are excluded entirely.
    """
    held_out = str(held_out)
    test = make_task(df, held_out, train_frac, val_frac).test
    others = df[df[dp.GROUP_COL].astype(str) != held_out]
    if others.empty:
        raise ValueError("leave_one_group_out needs at least one group besides the held-out one")
    periods = _periods(dp.temporal_split(others, train_frac, val_frac, time_cols=TIME_COLS))
    empty = [s for s in ("train", "val") if periods[s].empty]
    if empty:
        raise ValueError(f"Other groups have an empty {'/'.join(empty)} period")
    return Task(name=f"without_{held_out}", train=periods["train"], val=periods["val"], test=test)


def _rpt_cuts(sizes: np.ndarray, train_frac: float, val_frac: float) -> tuple[int, int]:
    """Number of RPT intervals in train (k1) and train+val (k2), cut closest to the target fractions.

    Constrained to 1 <= k1 < k2 < m so every split keeps at least one interval;
    ties go to the earlier boundary.
    """
    m = len(sizes)
    cum = np.cumsum(sizes) / sizes.sum()  # cum[k-1] = share of rows in the first k intervals
    k1_range = np.arange(1, m - 1)
    k1 = int(k1_range[np.argmin(np.abs(cum[k1_range - 1] - train_frac))])
    k2_range = np.arange(k1 + 1, m)
    k2 = int(k2_range[np.argmin(np.abs(cum[k2_range - 1] - (train_frac + val_frac)))])
    return k1, k2


def rpt_boundary_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15) -> pd.DataFrame:
    """E2: per-battery chronological split that cuts only where the RPT changes (D4).

    Whole RPT intervals (consecutive rows sharing an ``RPT`` value, in time
    order) go to one split, so no SOH label value seen in training reappears in
    validation or test through a shared RPT. Cut points are the RPT boundaries
    closest to the requested row fractions, subject to every split receiving at
    least one interval. Returns a copy sorted by battery and time with a
    ``split`` column.
    """
    if not 0 < train_frac < 1 or not 0 <= val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("Need 0 < train_frac, 0 <= val_frac and train_frac + val_frac < 1")
    if "RPT" not in df:
        raise ValueError("rpt_boundary_split needs an 'RPT' column")
    if df["RPT"].isna().any():
        raise ValueError("RPT has missing values; use the labelled subset (ml_training_dataset.csv)")
    order = [dp.GROUP_COL, *[c for c in TIME_COLS if c in df]]
    out = df.sort_values(order, kind="mergesort").reset_index(drop=True)
    out["split"] = ""
    for battery, g in out.groupby(out[dp.GROUP_COL].astype(str), sort=True):
        rpt = g["RPT"].astype(str)
        run = (rpt != rpt.shift()).cumsum()
        runs_per_rpt = run.groupby(rpt).nunique()
        if (runs_per_rpt > 1).any():
            raise ValueError(
                f"Battery {battery!r}: RPT(s) {runs_per_rpt[runs_per_rpt > 1].index.tolist()} "
                "recur after a later RPT in time order; cannot split on RPT boundaries"
            )
        sizes = run.value_counts(sort=False).sort_index().to_numpy()
        if len(sizes) < 3:
            raise ValueError(f"Battery {battery!r} has {len(sizes)} RPT interval(s); at least 3 are needed")
        k1, k2 = _rpt_cuts(sizes, train_frac, val_frac)
        out.loc[g.index, "split"] = np.where(run <= k1, "train", np.where(run <= k2, "val", "test"))
    return out


def check_rpt_boundary(split_df: pd.DataFrame, time_col: str = "Cycle_Date") -> pd.DataFrame:
    """Per battery: RPT intervals per split, RPTs spanning two splits, and chronological ordering."""
    key = split_df[dp.GROUP_COL].astype(str)
    n_splits = split_df.groupby([key, split_df["RPT"].astype(str)])["split"].nunique()
    shared = (n_splits > 1).groupby(level=0).sum().rename("rpts_spanning_splits")
    per_split = (
        split_df.groupby([key, "split"])["RPT"].nunique().unstack("split").reindex(columns=list(SPLITS)).fillna(0)
    ).astype(int).add_suffix("_rpts")
    ordering = dp.check_split_ordering(split_df.assign(**{dp.GROUP_COL: key}), time_col=time_col)
    res = pd.concat(
        [per_split, shared, ordering[["train_rows", "val_rows", "test_rows", "train_before_val", "val_before_test"]]],
        axis=1,
    )
    res.index.name = dp.GROUP_COL
    res["ok"] = (res["rpts_spanning_splits"] == 0) & res["train_before_val"] & res["val_before_test"]
    return res


def relabel_date_only(df: pd.DataFrame, health: pd.DataFrame, date_only_rpt: str) -> pd.DataFrame:
    """E5: recompute every health/label column under a date-only RPT convention (D3).

    Only cycles on the calendar day of a date-only RPT can change. Rows that
    fall before their battery's first effective RPT under the new convention
    lose their label and are dropped, matching how ``ml_training_dataset.csv``
    excludes pre-RPT rows. Feature columns are left untouched.

    ``Days_Since_RPT`` keeps the input's convention (whole or fractional days):
    rows whose RPT is unchanged keep their value, and reassigned rows are
    shifted by the exact gap between the old and new RPT dates. Rows without an
    input ``Days_Since_RPT`` or RPT date get the fractional-day gap from
    :func:`dp.align_to_rpt`.

    Adds ``label_changed``: True where ``SOH_Percent`` differs from the input.
    """
    rpt_cols = ("RPT", "RPT_Number", "Capacity_Ah", "Reference_Capacity_Ah", "SOH_Percent")
    absent = [c for c in (dp.GROUP_COL, "Test_Date", *rpt_cols) if c not in health]
    if absent:
        raise ValueError(f"health is missing column(s) {absent}")
    base = df.reset_index(drop=True)
    aligned = dp.align_to_rpt(base, health, rpt_cols=rpt_cols, date_only_rpt=date_only_rpt)

    out = base.copy()
    for c in rpt_cols:
        out[c] = aligned[f"{c}_expected"].to_numpy()

    new_days = aligned["Days_Since_RPT_expected"]
    if "Days_Since_RPT" in base and "RPT" in base:
        h = health.dropna(subset=["Test_Date"]).assign(_g=lambda x: x[dp.GROUP_COL].astype(str))
        rpt_date = h.drop_duplicates(["_g", "RPT"]).set_index(["_g", "RPT"])["Test_Date"]
        old_key = pd.MultiIndex.from_arrays([base[dp.GROUP_COL].astype(str), base["RPT"]])
        old_date = pd.Series(rpt_date.reindex(old_key).to_numpy(), index=base.index)
        new_date = aligned["Test_Date_expected"]
        shift = (old_date - new_date).dt.total_seconds() / 86400.0
        kept = base["Days_Since_RPT"].astype(float) + shift
        same_rpt = base["RPT"].astype(str).eq(out["RPT"].astype(str)) & base["RPT"].notna()
        kept = kept.where(~same_rpt, base["Days_Since_RPT"].astype(float))
        new_days = kept.where(kept.notna(), new_days)
    out["Days_Since_RPT"] = new_days.to_numpy()

    old_soh = base["SOH_Percent"].astype(float) if "SOH_Percent" in base else pd.Series(np.nan, index=base.index)
    new_soh = out["SOH_Percent"].astype(float)
    same = (old_soh.isna() & new_soh.isna()) | np.isclose(old_soh, new_soh, rtol=0.0, atol=1e-9)
    out["label_changed"] = ~same
    return out[out["RPT"].notna()].reset_index(drop=True)
