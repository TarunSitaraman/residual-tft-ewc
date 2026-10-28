"""Supplementary regression and classification metrics for the SOH forecasts (decision D30).

The headline metric stays the RMSE of the forecast SOH in percentage points (pp).
D30 adds the full metric set, computed on the same predictions:

- ``regression_metrics``: MSE, RMSE, MAE, MedAE, bias (mean error), MAPE, R²
  and NRMSE, on the SOH level (SOH relative to the first observed check), plus
  R² of the SOH *change*. The level R² is dominated by the spread of SOH between
  groups and origins, so a forecaster that predicts no change already scores
  high; the change R² measures what a forecaster adds to persistence.
- ``classification_metrics``: agreement of the reuse class (D10 cut-points) of
  the forecast and the measured SOH: accuracy, balanced accuracy, macro
  precision / recall / specificity / F1, Matthews correlation coefficient,
  Cohen's kappa (unweighted and linear-weighted), per-class values and the
  confusion matrix.
- ``binary_metrics``: the same for one binary decision, by default "fade beyond
  20 % since the first check" (SOH <= 80, class C3; the positive class), with
  AUC-ROC using the forecast SOH as the score (lower SOH = more likely
  positive). No probability model is involved, so AUC measures ranking only.

Log loss (cross-entropy) is not computed: every forecaster here gives a point
forecast, not class probabilities, and turning interval half-widths into class
probabilities would add an untested distributional assumption (D30).

Undefined values are NaN, never 0: R² when the target is constant, MAPE when a
target is 0, MCC / kappa / AUC when fewer than two classes occur, a class's
precision when it is never predicted, and its recall when it never occurs.
Macro averages run over the classes present in the measured labels.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from src import evaluation as ev

#: Binary decision for AUC-ROC and the binary metrics: SOH <= this value is positive (class C3 under D10).
EOL_SOH: float = 80.0

REGRESSION_KEYS: tuple[str, ...] = ("MSE", "RMSE", "MAE", "MedAE", "Bias", "MAPE", "R2", "R2_change", "NRMSE")
CLASS_KEYS: tuple[str, ...] = ("accuracy", "balanced_accuracy", "precision_macro", "recall_macro",
                               "specificity_macro", "f1_macro", "mcc", "kappa", "kappa_linear")
BINARY_KEYS: tuple[str, ...] = ("accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "mcc",
                                "kappa", "auc_roc")


def _nan_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float],
                       y_origin: Sequence[float] | None = None) -> dict:
    """Regression metrics of ``y_pred`` against ``y_true`` (both SOH in percent).

    Errors are ``y_pred - y_true``, so a positive ``Bias`` means the forecast
    overestimates SOH (underestimates fade). ``NRMSE`` is RMSE divided by the
    range (max - min) of ``y_true`` over the same samples; it is NaN when the
    range is 0. ``R2_change`` needs ``y_origin`` (the SOH at the forecast
    origin) and is R² of ``y_pred - y_origin`` against ``y_true - y_origin``.
    """
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    if y.shape != p.shape:
        raise ValueError(f"{len(p)} predictions for {len(y)} targets")
    n = len(y)
    out = {"n": int(n)}
    if not n:
        return {**out, **{k: float("nan") for k in REGRESSION_KEYS}}
    err = p - y
    mse = float(np.mean(err ** 2))
    nz = y != 0

    def r2(t, q):
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        return 1.0 - float(np.sum((q - t) ** 2)) / ss_tot if ss_tot > 0 else float("nan")

    out.update({
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.mean(np.abs(err))),
        "MedAE": float(np.median(np.abs(err))),
        "Bias": float(np.mean(err)),
        "MAPE": float(np.mean(np.abs(err[nz] / y[nz])) * 100) if nz.all() else float("nan"),
        "R2": r2(y, p),
        "R2_change": float("nan"),
        "NRMSE": _nan_div(np.sqrt(mse), float(np.ptp(y))),
    })
    if y_origin is not None:
        o = np.asarray(y_origin, float)
        if o.shape != y.shape:
            raise ValueError(f"{len(o)} origins for {len(y)} targets")
        out["R2_change"] = r2(y - o, p - o)
    return out


def _from_confusion(cm: np.ndarray) -> dict:
    """Multi-class metrics from a confusion matrix (rows: measured class, columns: predicted class)."""
    cm = np.asarray(cm, float)
    n = cm.sum()
    k = cm.shape[0]
    tp = np.diag(cm)
    actual, predicted = cm.sum(axis=1), cm.sum(axis=0)
    fp, fn = predicted - tp, actual - tp
    tn = n - tp - fp - fn
    present = actual > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(predicted > 0, tp / predicted, np.nan)
        recall = np.where(actual > 0, tp / actual, np.nan)
        specificity = np.where((tn + fp) > 0, tn / (tn + fp), np.nan)
        f1 = np.where((2 * tp + fp + fn) > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
    # Macro averages over the classes that occur in the measured labels (the set sklearn's balanced accuracy uses).
    # A measured class that is never predicted has undefined precision; it counts as 0 in the average (sklearn's
    # zero_division=0), so a forecaster cannot raise macro precision by never predicting a hard class.
    def macro(v):
        return float(np.mean(np.nan_to_num(v[present], nan=0.0))) if present.any() else float("nan")
    # Matthews correlation coefficient, multi-class form (Gorodkin 2004).
    cov_ytyp = float(tp.sum() * n - actual @ predicted)
    cov_ypyp = float(n ** 2 - predicted @ predicted)
    cov_ytyt = float(n ** 2 - actual @ actual)
    mcc = cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp) if cov_ytyt > 0 and cov_ypyp > 0 else float("nan")
    # Cohen's kappa, unweighted and linear-weighted, from the same matrix.
    expected = np.outer(actual, predicted) / n if n else np.zeros_like(cm)
    idx = np.arange(k)
    kappas = {}
    for name, w in (("kappa", (idx[:, None] != idx[None, :]).astype(float)),
                    ("kappa_linear", np.abs(idx[:, None] - idx[None, :]).astype(float))):
        den = float(np.sum(w * expected))
        kappas[name] = 1.0 - float(np.sum(w * cm)) / den if den > 0 else float("nan")
    return {
        "n": int(n), "n_classes_measured": int(present.sum()),
        "accuracy": _nan_div(tp.sum(), n),
        "balanced_accuracy": macro(recall),
        "precision_macro": macro(precision), "recall_macro": macro(recall),
        "specificity_macro": macro(specificity), "f1_macro": macro(f1),
        "mcc": float(mcc), **kappas,
        "per_class": {int(c): {"support": int(actual[c]), "precision": float(precision[c]), "recall": float(recall[c]),
                               "specificity": float(specificity[c]), "f1": float(f1[c])} for c in range(k)},
        "confusion": cm.astype(int),
    }


def confusion(true_class: Sequence[int], pred_class: Sequence[int], n_classes: int) -> np.ndarray:
    """Confusion counts over classes ``0 .. n_classes - 1`` (rows: measured, columns: predicted)."""
    t, p = np.asarray(true_class, int), np.asarray(pred_class, int)
    cm = np.zeros((n_classes, n_classes), int)
    np.add.at(cm, (t, p), 1)
    return cm


def classification_metrics(soh_true: Sequence[float], soh_pred: Sequence[float],
                           cutpoints: Sequence[float] = ev.REUSE_CUTPOINTS) -> dict:
    """Reuse-class agreement of forecast and measured SOH (classes from ``evaluation.reuse_class``).

    ``classification_from_confusion`` gives the same values from a confusion matrix, so matrices from several
    groups can be summed and scored as one pooled sample.
    """
    k = len(cutpoints) + 1
    cm = confusion(ev.reuse_class(soh_true, cutpoints), ev.reuse_class(soh_pred, cutpoints), k)
    return _from_confusion(cm)


def classification_from_confusion(cm: np.ndarray) -> dict:
    """``classification_metrics`` from a (possibly summed) confusion matrix."""
    return _from_confusion(np.asarray(cm))


def roc_auc(positive: Sequence[bool], score: Sequence[float]) -> float:
    """Area under the ROC curve (Mann–Whitney form, ties counted half); NaN unless both classes occur."""
    y, s = np.asarray(positive, bool), np.asarray(score, float)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    sorted_s = s[order]
    # Average ranks over ties.
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def binary_metrics(soh_true: Sequence[float], soh_pred: Sequence[float], threshold: float = EOL_SOH) -> dict:
    """Binary decision "SOH <= ``threshold``" (positive) from forecast vs measured SOH.

    Returns accuracy, balanced accuracy, precision, recall (sensitivity),
    specificity, F1, MCC, Cohen's kappa, the 2x2 confusion counts and AUC-ROC
    with ``-soh_pred`` as the score.
    """
    t = np.asarray(soh_true, float) <= threshold
    p = np.asarray(soh_pred, float) <= threshold
    m = _from_confusion(confusion(t.astype(int), p.astype(int), 2))
    pos = m["per_class"][1]
    return {"n": m["n"], "threshold": float(threshold), "n_positive": int(t.sum()),
            "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
            "precision": pos["precision"], "recall": pos["recall"], "specificity": pos["specificity"], "f1": pos["f1"],
            "mcc": m["mcc"], "kappa": m["kappa"],
            "auc_roc": roc_auc(t, -np.asarray(soh_pred, float)),
            "tn": int(m["confusion"][0, 0]), "fp": int(m["confusion"][0, 1]),
            "fn": int(m["confusion"][1, 0]), "tp": int(m["confusion"][1, 1])}


def flat(soh_true: Sequence[float], soh_pred: Sequence[float], soh_origin: Sequence[float] | None = None,
         cutpoints: Sequence[float] = ev.REUSE_CUTPOINTS, threshold: float = EOL_SOH) -> dict:
    """All D30 metrics as one flat dict of scalars (for result rows).

    Keys: ``reg_<metric>``, ``cls_<metric>``, ``cls_cm_<i>_<j>`` (confusion counts, so rows can be pooled by
    summing) and ``eol_<metric>``.
    """
    reg = regression_metrics(soh_true, soh_pred, soh_origin)
    cls = classification_metrics(soh_true, soh_pred, cutpoints)
    eol = binary_metrics(soh_true, soh_pred, threshold)
    out = {f"reg_{k}": reg[k] for k in REGRESSION_KEYS}
    out.update({f"cls_{k}": cls[k] for k in CLASS_KEYS})
    out["cls_n_classes_measured"] = cls["n_classes_measured"]
    out.update({f"cls_cm_{i}_{j}": int(v) for (i, j), v in np.ndenumerate(cls["confusion"])})
    out.update({f"eol_{k}": eol[k] for k in (*BINARY_KEYS, "n_positive", "tn", "fp", "fn", "tp")})
    return out


def pooled_from_rows(rows: Sequence[Mapping], n_classes: int = len(ev.REUSE_CUTPOINTS) + 1) -> dict:
    """Pooled reuse-class metrics from the summed ``cls_cm_*`` counts of several result rows."""
    cm = np.zeros((n_classes, n_classes), int)
    for r in rows:
        cm += np.array([[int(r[f"cls_cm_{i}_{j}"]) for j in range(n_classes)] for i in range(n_classes)])
    return _from_confusion(cm)
