"""Continual-learning evaluation: accuracy matrices, CL metrics and statistics.

See docs/RESEARCH_PLAN.md §3 Stage B. All quantities are *errors* (e.g. test
RMSE in SOH percentage points, D8), so lower is better throughout. Transfer
metrics are defined so that a **positive** value is harmful (forgetting, or a
worse start than the untrained reference) and a negative value is beneficial.

Per-task errors should come from ``src.data_pipeline.regression_metrics`` (see
``task_error``) so that the same RMSE/MAE definitions are used everywhere.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from src.data_pipeline import regression_metrics

STEP_INDEX_NAME = "after_step"
TASK_COLUMNS_NAME = "task"


# ---------------------------------------------------------------------------
# Accuracy matrix and continual-learning metrics
# ---------------------------------------------------------------------------


def task_error(y_true: Sequence[float], y_pred: Sequence[float], metric: str = "RMSE") -> float:
    """One entry of the accuracy matrix: ``metric`` from ``regression_metrics``."""
    return float(regression_metrics(y_true, y_pred)[metric])


def accuracy_matrix(errors: Mapping[tuple[int, str], float], order: Sequence[str]) -> pd.DataFrame:
    """Build the T×T error matrix ``R`` from ``{(step, task): error}``.

    ``R.iloc[i, j]`` is the test error on task ``order[j]`` after training
    through step ``i`` (steps are 0..T-1, step ``i`` trains ``order[i]``). Rows
    are labelled by the task trained at that step (index name ``after_step``),
    columns by task name in stream order.

    Entries with ``j <= i`` (seen tasks, including the diagonal) are required.
    Entries with ``j > i`` (not-yet-seen tasks) are optional and stay NaN when
    missing; ``R[j-1, j]`` is used for forward transfer.
    """
    order = [str(t) for t in order]
    if not order:
        raise ValueError("order must contain at least one task")
    if len(set(order)) != len(order):
        raise ValueError(f"order contains duplicate tasks: {order}")
    T = len(order)
    pos = {t: j for j, t in enumerate(order)}

    values = np.full((T, T), np.nan)
    for key, err in errors.items():
        if not (isinstance(key, tuple) and len(key) == 2):
            raise ValueError(f"key must be (step, task), got {key!r}")
        step, task = key
        if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or not 0 <= step < T:
            raise ValueError(f"step must be an integer in [0, {T - 1}], got {step!r}")
        if str(task) not in pos:
            raise ValueError(f"task {task!r} is not in order {order}")
        err = float(err)
        if not np.isfinite(err) or err < 0:
            raise ValueError(f"error for {key!r} must be finite and non-negative, got {err}")
        values[int(step), pos[str(task)]] = err

    missing = [(i, order[j]) for i in range(T) for j in range(i + 1) if np.isnan(values[i, j])]
    if missing:
        raise ValueError(f"missing required entries (step, task) with task seen by step: {missing}")

    return pd.DataFrame(
        values,
        index=pd.Index(order, name=STEP_INDEX_NAME),
        columns=pd.Index(order, name=TASK_COLUMNS_NAME),
    )


def cl_metrics(R: pd.DataFrame, reference_errors: Mapping[str, float] | None = None) -> dict:
    """Error-based continual-learning metrics from an accuracy matrix ``R``.

    With T tasks and ``R[i, j]`` the error on task j after step i:

    - ``final_avg_error``   = mean_j R[T-1, j]
    - ``plasticity``        = mean_j R[j, j]  (error right after learning task j)
    - ``backward_transfer`` = mean_{j<T-1} (R[T-1, j] - R[j, j]);
      positive = forgetting, negative = later tasks improved task j.
    - ``forgetting``        = mean_{j<T-1} (R[T-1, j] - min_{j<=i<T-1} R[i, j]);
      positive = final error above the best error seen before the last step.
    - ``forward_transfer``  = mean_{j>0} (R[j-1, j] - b_j), with ``b_j`` the
      error of the untrained reference model on task j (``reference_errors``);
      negative = earlier tasks help on task j before it is trained.

    Transfer metrics are NaN for T = 1. Forward transfer averages the tasks
    where both R[j-1, j] and b_j are available (``forward_transfer_n``) and is
    NaN when there are none. Per-task values are returned as ``{task: value}``.
    """
    if R.shape[0] != R.shape[1] or list(R.index) != list(R.columns):
        raise ValueError("R must be square with rows and columns in the same task order")
    tasks = [str(t) for t in R.columns]
    A = R.to_numpy(dtype=float)
    T = len(tasks)
    if T == 0:
        raise ValueError("R is empty")
    if np.isnan(A[np.tril_indices(T)]).any():
        raise ValueError("R has missing entries on or below the diagonal")

    final = A[T - 1, :]
    diag = np.diag(A).copy()
    bwt_j = final[: T - 1] - diag[: T - 1]
    forget_j = np.array([final[j] - A[j : T - 1, j].min() for j in range(T - 1)])

    fwt_per_task: dict[str, float] = {}
    if reference_errors is not None:
        for j in range(1, T):
            b = reference_errors.get(tasks[j], np.nan)
            b = np.nan if b is None else float(b)
            v = A[j - 1, j] - b
            if np.isfinite(v):
                fwt_per_task[tasks[j]] = float(v)
    fwt_vals = list(fwt_per_task.values())

    def _mean(x: Iterable[float]) -> float:
        x = np.asarray(list(x), dtype=float)
        return float(x.mean()) if x.size else float("nan")

    return {
        "n_tasks": T,
        "final_avg_error": _mean(final),
        "plasticity": _mean(diag),
        "backward_transfer": _mean(bwt_j),
        "forgetting": _mean(forget_j),
        "forward_transfer": _mean(fwt_vals),
        "forward_transfer_n": len(fwt_vals),
        "final_error_per_task": dict(zip(tasks, map(float, final))),
        "diagonal_error_per_task": dict(zip(tasks, map(float, diag))),
        "backward_transfer_per_task": dict(zip(tasks[: T - 1], map(float, bwt_j))),
        "forgetting_per_task": dict(zip(tasks[: T - 1], map(float, forget_j))),
        "forward_transfer_per_task": fwt_per_task,
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _clean(values) -> np.ndarray:
    x = np.asarray(values, dtype=float).ravel()
    return x[~np.isnan(x)]


def bootstrap_ci(values, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Sample mean and percentile-bootstrap ``(1 - alpha)`` CI: ``(mean, lower, upper)``.

    NaNs are dropped. n = 0 gives all NaN; n = 1 gives the value with NaN
    bounds. Deterministic for a given ``seed``.
    """
    x = _clean(values)
    n = x.size
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(x.mean())
    if n == 1:
        return (mean, float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    chunk = max(1, 2_000_000 // n)  # bound memory for large samples
    for start in range(0, n_boot, chunk):
        stop = min(n_boot, start + chunk)
        idx = rng.integers(0, n, size=(stop - start, n))
        boot[start:stop] = x[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return (mean, float(lo), float(hi))


def paired_comparison(a, b, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Paired comparison of matched runs (same order and seed at each position).

    Differences are ``d = a - b``; for errors, a negative mean difference means
    ``a`` is better. Pairs with a NaN on either side are dropped. Returns the
    mean difference with its bootstrap CI, the Wilcoxon signed-rank statistic
    and two-sided p-value (zero differences discarded, scipy default), and the
    matched-pairs rank-biserial correlation ``r = (W+ - W-) / (W+ + W-)`` over
    non-zero differences (r > 0: ``a`` tends to be larger than ``b``).

    If every difference is zero the p-value is 1 and r = 0; with fewer than
    two non-zero differences the test and effect size are NaN.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same length, got {a.size} and {b.size}")
    keep = ~(np.isnan(a) | np.isnan(b))
    d = a[keep] - b[keep]
    n = int(d.size)
    mean, lo, hi = bootstrap_ci(d, n_boot=n_boot, alpha=alpha, seed=seed)
    nz = d[d != 0]
    out = {
        "n": n,
        "n_nonzero": int(nz.size),
        "mean_diff": mean,
        "ci_low": lo,
        "ci_high": hi,
        "median_diff": float(np.median(d)) if n else float("nan"),
        "wilcoxon_stat": float("nan"),
        "p_value": float("nan"),
        "rank_biserial": float("nan"),
    }
    if n > 0 and nz.size == 0:
        out.update(wilcoxon_stat=0.0, p_value=1.0, rank_biserial=0.0)
    elif nz.size >= 2:
        res = stats.wilcoxon(nz)
        ranks = stats.rankdata(np.abs(nz))
        w_pos, w_neg = ranks[nz > 0].sum(), ranks[nz < 0].sum()
        out.update(
            wilcoxon_stat=float(res.statistic),
            p_value=float(res.pvalue),
            rank_biserial=float((w_pos - w_neg) / (w_pos + w_neg)),
        )
    return out


def holm_correction(pvalues: Mapping[str, float]) -> dict:
    """Holm–Bonferroni adjusted p-values ``{name: p_adj}``.

    With the m non-NaN p-values sorted ascending, ``p_adj(k) = max_{l<=k}
    min(1, (m - l + 1) p(l))``. NaN inputs stay NaN and do not count in m.
    """
    names = list(pvalues)
    p = np.array([np.nan if pvalues[k] is None else float(pvalues[k]) for k in names])
    valid = np.flatnonzero(~np.isnan(p))
    adjusted = np.full(p.shape, np.nan)
    m = valid.size
    if m:
        order = valid[np.argsort(p[valid], kind="stable")]
        scaled = np.minimum(1.0, (m - np.arange(m)) * p[order])
        adjusted[order] = np.maximum.accumulate(scaled)
    return {k: float(v) for k, v in zip(names, adjusted)}


def summarise_runs(
    runs: pd.DataFrame,
    metric_cols: Sequence[str],
    group_cols: Sequence[str],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Per-group mean, bootstrap CI and n for each metric of a long run table.

    ``runs`` has one row per run (e.g. method, order_id, seed, metrics). The
    output has one row per group with columns ``<metric>_mean``,
    ``<metric>_ci_low``, ``<metric>_ci_high`` and ``<metric>_n`` (non-NaN runs).
    """
    metric_cols, group_cols = list(metric_cols), list(group_cols)
    missing = [c for c in metric_cols + group_cols if c not in runs.columns]
    if missing:
        raise KeyError(f"columns not in runs: {missing}")
    rows = []
    for key, sub in runs.groupby(group_cols, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key))
        for m in metric_cols:
            mean, lo, hi = bootstrap_ci(sub[m], n_boot=n_boot, alpha=alpha, seed=seed)
            row.update({
                f"{m}_mean": mean, f"{m}_ci_low": lo, f"{m}_ci_high": hi,
                f"{m}_n": int(sub[m].notna().sum()),
            })
        rows.append(row)
    return pd.DataFrame(rows)
