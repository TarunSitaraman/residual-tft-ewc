"""Decision D25: horizon-wise shrinkage of the learned residual and split-conformal forecast intervals.

Both are fitted after each task on that task's **validation rows only** (the rows that already decide early
stopping), from the model's predictions; neither changes training, the weights, the EWC Fisher or anchor, so the
unshrunk ("raw") model of a stream is exactly the model of the same stream without them.

- **Shrinkage.** For each horizon h, ``alpha_h = clip(sum(p * y) / sum(p ** 2), 0, 1)`` over the validation rows of
  that horizon, where ``p`` is the raw prediction and ``y`` the target, both on the residual scale
  (``Delta - lin_forecast``). The shrunk forecast is ``lin_forecast + alpha_h * p``: at ``alpha_h = 0`` it is linear
  extrapolation, at 1 the raw residual model. ``alpha_h`` minimises the validation squared error of ``alpha * p``
  over [0, 1]. A horizon without validation rows in the task (or with all predictions zero) keeps its previous
  value; every horizon starts at 1.
- **Intervals.** For each level (0.9, 0.8) and horizon, the split-conformal half-width ``q_h`` is the
  ``ceil((n + 1) * level)``-th smallest absolute validation residual of the (shrunk) model; if that index exceeds
  ``n`` the largest residual is used and the horizon is flagged (``undersized``). The interval is
  ``forecast +/- q_h``. Linear extrapolation gets intervals the same way, from the same rows (its residual is the
  residual-scale target itself), so the two are compared on equal terms.
- **Interval score** (Gneiting and Raftery 2007) for a central interval of level ``1 - a``:
  ``(u - l) + (2 / a) * (l - y) * [y < l] + (2 / a) * (y - u) * [y > u]``; for ``c +/- q`` this is
  ``2 q + (2 / a) * max(|y - c| - q, 0)`` (pp; lower is better).
"""

from __future__ import annotations

import copy
import math
from typing import Sequence

import numpy as np
import pandas as pd

from src import evaluation as ev
from src.forecasting import TARGET

LEVELS: tuple[float, ...] = (0.9, 0.8)

__all__ = ["LEVELS", "HorizonCalibrator", "conformal_halfwidth", "interval_metrics", "run_stream_calibrated"]


def conformal_halfwidth(scores: np.ndarray, level: float) -> tuple[float, bool]:
    """Split-conformal half-width from absolute residuals; returns (q, undersized)."""
    s = np.sort(np.abs(np.asarray(scores, float)))
    n = s.size
    if n == 0:
        return float("nan"), True
    k = math.ceil((n + 1) * level)
    return (float(s[-1]), True) if k > n else (float(s[k - 1]), False)


def interval_metrics(y: np.ndarray, centre: np.ndarray, q: np.ndarray, level: float) -> dict:
    """Coverage, mean width (pp) and mean interval score (pp) of ``centre +/- q`` for the targets ``y``."""
    y, centre, q = (np.asarray(v, float) for v in (y, centre, q))
    dev = np.abs(y - centre)
    a = 1.0 - level
    return {"coverage": float(np.mean(dev <= q)), "width": float(np.mean(2 * q)),
            "interval_score": float(np.mean(2 * q + (2 / a) * np.maximum(dev - q, 0.0)))}


class HorizonCalibrator:
    """Per-horizon shrinkage factors and conformal half-widths, refitted after every task (see module docstring)."""

    def __init__(self, horizons: Sequence[int], levels: Sequence[float] = LEVELS) -> None:
        self.horizons = [int(h) for h in horizons]
        self.levels = [float(v) for v in levels]
        self.alpha_ = {h: 1.0 for h in self.horizons}
        self.q_model_ = {lv: {h: float("nan") for h in self.horizons} for lv in self.levels}
        self.q_lin_ = {lv: {h: float("nan") for h in self.horizons} for lv in self.levels}
        self.history_: list[dict] = []

    def _alpha_of(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["horizon"].astype(int).map(self.alpha_).to_numpy(float)

    def shrink(self, pred: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
        """Shrunk residual prediction ``alpha_h * pred`` for the rows of ``frame``."""
        return self._alpha_of(frame) * np.asarray(pred, float)

    def update(self, strategy, frame: pd.DataFrame, task: str = "") -> "HorizonCalibrator":
        """Refit on ``frame`` (a task's validation rows, residual target) with ``strategy``'s current predictions."""
        if len(frame) == 0:
            self.history_.append({"task": task, "n": 0})
            return self
        y = frame[TARGET].to_numpy(float)
        p = np.asarray(strategy.predict(frame), float)
        h = frame["horizon"].astype(int).to_numpy()
        undersized = {}
        for k in self.horizons:
            m = h == k
            if not m.any():
                continue
            den = float(np.sum(p[m] ** 2))
            if den > 0:
                self.alpha_[k] = float(np.clip(np.sum(p[m] * y[m]) / den, 0.0, 1.0))
            for lv in self.levels:
                self.q_model_[lv][k], u1 = conformal_halfwidth(y[m] - self.alpha_[k] * p[m], lv)
                self.q_lin_[lv][k], u2 = conformal_halfwidth(y[m], lv)
                if u1 or u2:
                    undersized[f"{lv}_h{k}"] = int(m.sum())
        self.history_.append({"task": task, "n": int(len(frame)), "alpha": dict(self.alpha_),
                              "q_model": copy.deepcopy(self.q_model_), "q_lin": copy.deepcopy(self.q_lin_),
                              "undersized": undersized})
        return self

    def halfwidths(self, frame: pd.DataFrame, which: str, level: float) -> np.ndarray:
        q = (self.q_model_ if which == "model" else self.q_lin_)[level]
        return frame["horizon"].astype(int).map(q).to_numpy(float)


def run_stream_calibrated(strategy, stream: Sequence, calibrator: HorizonCalibrator, eval_split: str = "test",
                          metric: str = "RMSE") -> dict:
    """``p3.run_stream`` (no pool) with the D25 calibrator refitted on each task's validation rows after training.

    The training path is that of ``p3.run_stream``, so the ``raw`` results equal those of the same stream run
    without calibration (tested). Returns the raw and shrunk CL metrics and per-horizon RMSE after the last step,
    and, after the last step, interval metrics per task and level for the shrunk model and for linear extrapolation.
    The reference errors ``b_j`` are those of the strategy (and calibrator) before the stream.
    """
    ref = copy.deepcopy(strategy).initialise(stream[0]) if strategy.model is None else strategy
    frames = {t.name: getattr(t, eval_split) for t in stream}
    y = {k: f[TARGET].to_numpy(float) for k, f in frames.items()}
    p_ref = {k: ref.predict(f) for k, f in frames.items()}
    b_raw = {k: ev.task_error(y[k], p_ref[k], metric) for k in frames}
    b_shr = {k: ev.task_error(y[k], calibrator.shrink(p_ref[k], frames[k]), metric) for k in frames}
    raw, shr = {}, {}
    for i, task in enumerate(stream):
        strategy.fit_task(task)
        calibrator.update(strategy, task.val, task.name)
        for k, f in frames.items():
            p = strategy.predict(f)
            raw[(i, k)] = ev.task_error(y[k], p, metric)
            shr[(i, k)] = ev.task_error(y[k], calibrator.shrink(p, f), metric)
    order = [t.name for t in stream]
    R_raw, R_shr = ev.accuracy_matrix(raw, order), ev.accuracy_matrix(shr, order)
    final = pd.concat(frames.values(), ignore_index=True)
    p_raw = np.concatenate([strategy.predict(frames[k]) for k in order])
    p_shr = calibrator.shrink(p_raw, final)
    intervals = {}
    for lv in calibrator.levels:
        tag = f"{round(lv * 100)}"
        for which, centre in (("model", p_shr), ("lin", np.zeros(len(final)))):
            q = calibrator.halfwidths(final, which, lv)
            per_task, start = {}, 0
            for k in order:
                n = len(frames[k])
                per_task[k] = interval_metrics(y[k], centre[start:start + n], q[start:start + n], lv)
                start += n
            intervals[f"{which}_{tag}"] = {
                **{m: float(np.mean([v[m] for v in per_task.values()])) for m in ("coverage", "width", "interval_score")},
                "per_task": per_task}
    return {"raw": {"R": R_raw, "metrics": ev.cl_metrics(R_raw, reference_errors=b_raw),
                    "horizon_rmse": _horizon_rmse(final, p_raw)},
            "shrunk": {"R": R_shr, "metrics": ev.cl_metrics(R_shr, reference_errors=b_shr),
                       "horizon_rmse": _horizon_rmse(final, p_shr)},
            "intervals": intervals, "calibration": calibrator.history_,
            "train_time_s": list(strategy.training_time_), "config": strategy.config()}


def _horizon_rmse(frame: pd.DataFrame, pred: np.ndarray) -> dict[int, float]:
    err = np.asarray(pred, float) - frame[TARGET].to_numpy(float)
    h = frame["horizon"].to_numpy()
    return {int(k): float(np.sqrt(np.mean(err[h == k] ** 2))) for k in sorted(np.unique(h))}
