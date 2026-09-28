"""Reuse-category evaluation (E10, decisions D10 and docs/REUSE_CATEGORIES.md §4).

Categories come from thresholding SOH with ``evaluation.reuse_class``; the same
rule is applied to measured and predicted SOH, so no separate classifier is
trained for the primary result. Evaluation is reported at two units:

- ``"interval"`` (headline): one sample per (battery, RPT) in the evaluation
  period, with the measured SOH of that RPT and the mean predicted SOH over the
  interval's cycles. This removes the weighting by repeated labels (D4).
- ``"cycle"``: one sample per cycle.

Classes describe capacity fade since the first observed RPT, not suitability
for an application.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

from src import data_pipeline as dp
from src import evaluation as ev

TARGET = "SOH_Percent"
UNITS = ("interval", "cycle")

#: Pre-registered cut-point variants (docs/REUSE_CATEGORIES.md §4).
CUTPOINT_VARIANTS: dict[str, tuple[float, ...]] = {
    "primary": ev.REUSE_CUTPOINTS,
    "S-a1 minus 2.5 pp": tuple(c - 2.5 for c in ev.REUSE_CUTPOINTS),
    "S-a1 plus 2.5 pp": tuple(c + 2.5 for c in ev.REUSE_CUTPOINTS),
    "S-a2 three classes": (10.0, 20.0),
}


def samples(df: pd.DataFrame, pred: Sequence[float], unit: str = "interval") -> pd.DataFrame:
    """Evaluation samples with measured and predicted SOH at the chosen unit.

    For ``"interval"`` rows are grouped by (Battery_ID, RPT); the measured SOH
    is constant within an interval by construction (checked), and the
    predicted SOH is the mean over the interval's cycles.
    """
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {UNITS}, got {unit!r}")
    pred = np.asarray(pred, dtype=float)
    if len(pred) != len(df):
        raise ValueError(f"{len(pred)} predictions for {len(df)} rows")
    frame = pd.DataFrame({dp.GROUP_COL: df[dp.GROUP_COL].astype(str).to_numpy(),
                          "soh_true": df[TARGET].to_numpy(dtype=float), "soh_pred": pred})
    if unit == "cycle":
        return frame
    if "RPT" not in df:
        raise KeyError("interval unit needs the RPT column")
    frame["RPT"] = df["RPT"].to_numpy()
    out = frame.groupby([dp.GROUP_COL, "RPT"], sort=False).agg(
        soh_true=("soh_true", "first"), soh_max=("soh_true", "max"), soh_min=("soh_true", "min"),
        soh_pred=("soh_pred", "mean"), n_cycles=("soh_pred", "size")).reset_index()
    if ((out["soh_max"] - out["soh_min"]) > 1e-9).any():
        raise ValueError("measured SOH varies within an RPT interval; labels are not RPT-aligned")
    return out.drop(columns=["soh_max", "soh_min"])


def class_metrics(soh_true: Sequence[float], soh_pred: Sequence[float],
                  cutpoints: Sequence[float] = ev.REUSE_CUTPOINTS, mae: float | None = None) -> dict:
    """Agreement between classes from measured and predicted SOH.

    Returns accuracy; macro-F1 over the classes present in measured ∪ predicted
    (with that number, so a high macro-F1 on one class is visible); linear-weighted
    Cohen's kappa (NaN when fewer than two classes occur); the share of errors of
    more than one class; the confusion matrix over all classes; ``single_class``
    when the measured labels contain one class (macro-F1 then not interpretable);
    and ``boundary_share``, the fraction of measured SOH within ±``mae`` of a
    cut-point (NaN without ``mae``).
    """
    soh_true, soh_pred = np.asarray(soh_true, float), np.asarray(soh_pred, float)
    n_classes = len(cutpoints) + 1
    y, yhat = ev.reuse_class(soh_true, cutpoints), ev.reuse_class(soh_pred, cutpoints)
    present = sorted(set(y) | set(yhat))
    out = {
        "n": int(len(y)),
        "accuracy": float(np.mean(y == yhat)) if len(y) else np.nan,
        "macro_f1": float(f1_score(y, yhat, labels=present, average="macro", zero_division=0)) if len(y) else np.nan,
        "n_classes_present": len(present),
        "n_classes_measured": len(set(y)),
        "single_class": len(set(y)) < 2,
        "kappa_linear": float(cohen_kappa_score(y, yhat, labels=list(range(n_classes)), weights="linear"))
        if len(present) >= 2 else np.nan,
        "off_by_more_than_one": float(np.mean(np.abs(y - yhat) > 1)) if len(y) else np.nan,
        "confusion": confusion_matrix(y, yhat, labels=list(range(n_classes))) if len(y) else np.zeros((n_classes, n_classes), int),
    }
    if mae is not None and len(y):
        boundaries = 100.0 - np.asarray(cutpoints, float)
        out["boundary_share"] = float(np.mean(np.min(np.abs(soh_true[:, None] - boundaries[None, :]), axis=1) <= mae))
    else:
        out["boundary_share"] = np.nan
    return out


def evaluate(df: pd.DataFrame, pred: Sequence[float], unit: str = "interval",
             cutpoints: Sequence[float] = ev.REUSE_CUTPOINTS, by_group: bool = True) -> pd.DataFrame:
    """Class metrics per battery group (optional) and pooled (``ALL``).

    ``boundary_share`` uses the cycle-level MAE of the SOH predictions on the
    same rows, i.e. the model's own error scale.
    """
    pred = np.asarray(pred, float)
    s = samples(df, pred, unit)
    rows = []
    groups = [(g, s[s[dp.GROUP_COL] == g], df[dp.GROUP_COL].astype(str).to_numpy() == g)
              for g in sorted(s[dp.GROUP_COL].unique())] if by_group else []
    for name, sub, mask in [*groups, ("ALL", s, np.ones(len(df), bool))]:
        mae = float(np.mean(np.abs(df[TARGET].to_numpy(float)[mask] - pred[mask]))) if mask.any() else None
        m = class_metrics(sub["soh_true"], sub["soh_pred"], cutpoints, mae)
        rows.append({dp.GROUP_COL: name, "unit": unit, "soh_mae": mae, **{k: v for k, v in m.items() if k != "confusion"},
                     "confusion": m["confusion"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def _last_rpt_soh(train: pd.DataFrame) -> pd.Series:
    t = train.sort_values([dp.GROUP_COL, "Cycle_Date", "Cycle_Index"], kind="mergesort")
    return t.groupby(t[dp.GROUP_COL].astype(str))[TARGET].last()


def baseline_predictions(train: pd.DataFrame, test: pd.DataFrame, kind: str,
                         cutpoints: Sequence[float] = ev.REUSE_CUTPOINTS) -> np.ndarray:
    """SOH-valued predictions whose class equals a per-battery baseline class.

    * ``"persistence"``: the class of the battery's last training RPT (its SOH is returned).
    * ``"majority"``: the most frequent class among the battery's training cycles
      (ties to the lower class); a representative SOH inside that class is returned.

    Returning SOH values lets baselines go through the same ``evaluate`` path.
    """
    groups = test[dp.GROUP_COL].astype(str)
    if kind == "persistence":
        last = _last_rpt_soh(train)
        return groups.map(last).to_numpy(float)
    if kind == "majority":
        tr_groups = train[dp.GROUP_COL].astype(str)
        cls = pd.Series(ev.reuse_class(train[TARGET].to_numpy(float), cutpoints), index=train.index)
        majority = cls.groupby(tr_groups).agg(lambda s: s.value_counts().sort_index().idxmax())
        # A representative SOH per class: the median training SOH of that class and battery.
        rep = {}
        for g in majority.index:
            m = majority[g]
            vals = train.loc[(tr_groups == g) & (cls == m), TARGET]
            rep[g] = float(vals.median())
        return groups.map(rep).to_numpy(float)
    raise ValueError(f"kind must be 'persistence' or 'majority', got {kind!r}")


# ---------------------------------------------------------------------------
# Continual learning: category matrix per training step
# ---------------------------------------------------------------------------


def category_matrix(predictions: Mapping[tuple[int, str], np.ndarray], frames: Mapping[str, pd.DataFrame],
                    order: Sequence[str], metric: str = "macro_f1", unit: str = "interval",
                    cutpoints: Sequence[float] = ev.REUSE_CUTPOINTS) -> pd.DataFrame:
    """Category *error* matrix ``1 − metric`` after each step, for ``evaluation.cl_metrics``.

    ``predictions[(i, task)]`` are SOH predictions for ``frames[task]`` after
    training step ``i``. Using ``1 − metric`` keeps the error convention (lower
    is better, positive backward transfer = forgetting) of the RMSE matrix.
    """
    if metric not in ("macro_f1", "accuracy"):
        raise ValueError("metric must be 'macro_f1' or 'accuracy'")
    errors = {}
    for (i, task), pred in predictions.items():
        s = samples(frames[task], pred, unit)
        errors[(i, task)] = 1.0 - class_metrics(s["soh_true"], s["soh_pred"], cutpoints)[metric]
    return ev.accuracy_matrix(errors, order)


def pooled_confusion(predictions: Mapping[str, np.ndarray], frames: Mapping[str, pd.DataFrame], unit: str = "interval",
                     cutpoints: Sequence[float] = ev.REUSE_CUTPOINTS) -> np.ndarray:
    """Confusion matrix pooled over tasks (e.g. after the last step)."""
    n = len(cutpoints) + 1
    total = np.zeros((n, n), dtype=int)
    for task, pred in predictions.items():
        s = samples(frames[task], pred, unit)
        total += class_metrics(s["soh_true"], s["soh_pred"], cutpoints)["confusion"]
    return total


def class_distribution(df: pd.DataFrame, cutpoints: Sequence[float] = ev.REUSE_CUTPOINTS,
                       unit: str = "interval") -> pd.DataFrame:
    """Share of samples per measured class and battery group (shows class imbalance and degeneracy)."""
    s = samples(df, df[TARGET].to_numpy(float), unit)
    s["class"] = ev.reuse_class(s["soh_true"], cutpoints)
    names = [ev.REUSE_CLASS_NAMES[i] if len(cutpoints) == len(ev.REUSE_CUTPOINTS) else f"C{i}"
             for i in range(len(cutpoints) + 1)]
    table = pd.crosstab(s[dp.GROUP_COL], s["class"]).reindex(columns=range(len(cutpoints) + 1), fill_value=0)
    table.columns = names
    return table
