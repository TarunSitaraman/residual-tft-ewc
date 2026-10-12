"""Tabular literature baselines of decision D24: Gaussian-process regression, random forest and SVR.

Like the D21 gradient boosting, each is fitted once on the training rows of every task (they are not continual
learners) and predicts every task's test rows. Preprocessing is fitted on the training rows only: median imputation
with missing indicators, removal of constant columns and standardisation. The ``residual`` arm fits the target
``Delta - lin_forecast`` and adds ``lin_forecast`` back (D23), so its errors are on the same scale as the direct arm.

- GPR (Takahashi et al. 2023 use bagged GPR; Faraji-Niri et al. 2023 EIS with machine learning): constant x Matern
  (nu = 2.5, one length scale per input) + white noise, normalised target, hyperparameters by marginal likelihood
  with 3 seeded restarts.
- Random forest: 500 trees, scikit-learn defaults otherwise, seeded.
- SVR: RBF kernel on a standardised target, epsilon 0.1; C from ``SVR_C_GRID`` by RMSE on the validation rows,
  then refitted on the training rows.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.forecasting import TARGET

TABULAR_MODELS: tuple[str, ...] = ("gpr", "rf", "svr")
SVR_C_GRID: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)


def _pipeline(model):
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), VarianceThreshold(0.0),
                         StandardScaler(), model)


def make_model(name: str, seed: int = 0, n_inputs: int | None = None, C: float = 1.0):
    """Unfitted scikit-learn pipeline for one tabular baseline."""
    if name == "gpr":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(n_inputs) if n_inputs else 1.0, nu=2.5,
                                              length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(1e-2, (1e-6, 1e1))
        return _pipeline(GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=3, random_state=seed))
    if name == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return _pipeline(RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=1))
    if name == "svr":
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR
        return _pipeline(TransformedTargetRegressor(SVR(kernel="rbf", C=C, epsilon=0.1), transformer=StandardScaler()))
    raise ValueError(f"unknown tabular baseline {name!r}; choose from {TABULAR_MODELS}")


def _n_inputs(train: pd.DataFrame, features: Sequence[str]) -> int:
    """Columns after imputation-with-indicators and constant-column removal (for GPR's per-input length scales)."""
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    return int(make_pipeline(SimpleImputer(strategy="median", add_indicator=True), VarianceThreshold(0.0))
               .fit_transform(train[list(features)]).shape[1])


def fit_predict(name: str, train: pd.DataFrame, val: pd.DataFrame, tests: Mapping[str, pd.DataFrame],
                features: Sequence[str], arm: str = "direct", seed: int = 0, target: str = TARGET) -> dict:
    """Fit ``name`` on ``train`` and predict each frame of ``tests`` (forecast of Delta in pp).

    ``val`` is used only to choose SVR's C. Returns ``{"pred": {test name: array}, "config": {...}}``.
    """
    if arm not in ("direct", "residual"):
        raise ValueError("arm must be 'direct' or 'residual'")
    feats = [f for f in features if train[f].notna().any()]
    offset = (lambda df: df["lin_forecast"].to_numpy(float)) if arm == "residual" else (lambda df: np.zeros(len(df)))
    y = train[target].to_numpy(float) - offset(train)
    config: dict = {"model": name, "arm": arm, "seed": seed, "n_features": len(feats)}
    if name == "svr":
        yv = val[target].to_numpy(float)
        scores = {}
        for c in SVR_C_GRID:
            m = make_model("svr", seed, C=c).fit(train[feats], y)
            scores[c] = float(np.sqrt(np.mean((m.predict(val[feats]) + offset(val) - yv) ** 2)))
        best = min(SVR_C_GRID, key=lambda c: (scores[c], c))
        model = make_model("svr", seed, C=best).fit(train[feats], y)
        config.update(C=best, val_rmse_by_C={str(k): v for k, v in scores.items()})
    else:
        model = make_model(name, seed, n_inputs=_n_inputs(train, feats) if name == "gpr" else None)
        model.fit(train[feats], y)
        if name == "gpr":
            config["kernel"] = str(model[-1].kernel_)
    return {"pred": {k: model.predict(v[feats]) + offset(v) for k, v in tests.items()}, "config": config}
