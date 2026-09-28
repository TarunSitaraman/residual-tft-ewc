"""Shared MLP backbone and tabular preprocessing for the Stage C methods.

Every continual-learning strategy (docs/RESEARCH_PLAN.md §3 Stage C) uses the
same preprocessor and network so that differences between methods come from
the continual-learning mechanism only. The preprocessor is fitted on training
rows only; missing values are imputed with the training median *and* flagged
by an explicit indicator column, so imputation is never silent (AGENTS.md).
"""

from __future__ import annotations

import os
import random
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

__all__ = ["TabularPreprocessor", "MLPRegressor", "set_determinism"]


class TabularPreprocessor:
    """Median imputation + standardisation + one missing-indicator per feature.

    Output width is always ``2 * len(features)`` (values, then indicators), so
    the network input dimension never changes across tasks even when a later
    task has missingness the first task did not.
    """

    def __init__(self, features: Sequence[str], indicators: bool = True) -> None:
        if len(features) == 0:
            raise ValueError("TabularPreprocessor needs at least one feature")
        if len(set(features)) != len(features):
            raise ValueError("Duplicate feature names")
        self.features: list[str] = list(features)
        # indicators=False zeroes the indicator block (width unchanged), so the model
        # cannot see which values were imputed: the E11 "impute only" ablation.
        self.indicators = bool(indicators)
        self.median_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.n_fit_rows_: int | None = None

    @property
    def is_fitted(self) -> bool:
        return self.mean_ is not None

    @property
    def output_dim(self) -> int:
        return 2 * len(self.features)

    def _values(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise KeyError(f"Missing feature columns: {missing}")
        x = df[self.features].to_numpy(dtype=np.float64, copy=True)
        if np.isinf(x).any():
            raise ValueError("Infinite feature values; clean them explicitly before preprocessing")
        return x

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        """Learn medians, means and standard deviations from ``df`` (training rows only)."""
        if len(df) == 0:
            raise ValueError("Cannot fit the preprocessor on an empty frame")
        x = self._values(df)
        all_nan = np.isnan(x).all(axis=0)
        if all_nan.any():
            # No training information: centre at 0 so imputed values are neutral;
            # the indicator column still carries the missingness.
            x[:, all_nan] = 0.0
        self.median_ = np.nanmedian(x, axis=0)
        self.mean_ = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        # Constant training features would divide by zero; leave them unscaled.
        self.std_ = np.where(std > 1e-12, std, 1.0)
        self.n_fit_rows_ = len(df)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Standardised values (NaN → standardised training median) + binary indicators."""
        if not self.is_fitted:
            raise RuntimeError("TabularPreprocessor.transform called before fit")
        x = self._values(df)
        mask = np.isnan(x)
        x = np.where(mask, self.median_, x)
        z = (x - self.mean_) / self.std_
        flags = mask.astype(np.float64) if self.indicators else np.zeros_like(z)
        return np.concatenate([z, flags], axis=1).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)


class MLPRegressor(nn.Module):
    """Small fully connected network with a scalar output (SOH regression)."""

    def __init__(self, in_dim: int, hidden: Sequence[int] = (128, 64), dropout: float = 0.1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.in_dim = in_dim
        self.hidden = tuple(hidden)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def set_determinism(seed: int) -> None:
    """Seed Python, NumPy and PyTorch and force deterministic single-threaded CPU kernels.

    Reference runs must be bit-reproducible (docs/RESEARCH_PLAN.md §6).
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
