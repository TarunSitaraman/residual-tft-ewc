"""Temporal Fusion Transformer for sequence-to-one SOH estimation (proposed model, E9).

Implements the building blocks of the Temporal Fusion Transformer (Lim et al.,
"Temporal Fusion Transformers for interpretable multi-horizon time series
forecasting", International Journal of Forecasting 37(4), 2021): gated
residual networks, variable selection, an LSTM encoder with gated skip
connections, static-enrichment GRN, interpretable multi-head attention and a
position-wise GRN (decision D12).

Adaptations for this project:

- **Estimation, not forecasting.** The input is a window of the most recent
  cycles' features for one battery, ending at the cycle being estimated; the
  output is that cycle's SOH. Past SOH labels are never inputs, so the model
  cannot copy the step-function target (D4).
- **No static covariates.** Battery identity is not an input, keeping the TFT
  task-agnostic like every other method; the static-enrichment GRN therefore
  runs without a context vector.
- **Explicit missingness.** Each variable is embedded from its standardised
  value *and* its missing indicator, so imputed values are always marked (D7).
- **Causal attention** over the window plus a padding mask for batteries with
  fewer past cycles than the window length.

Input tensor per sample: ``(window, 2 * n_features + 1)`` — standardised values,
missing indicators, then a padding flag (1 = padded step). This single-tensor
format lets the continual-learning strategies (EWC Fisher, LwF distillation)
treat the TFT exactly like the MLP backbone.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from torch import nn

__all__ = ["GatedResidualNetwork", "TemporalFusionTransformer", "CycleWindows"]


class GatedLinearUnit(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_in, 2 * d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.glu(self.linear(self.dropout(x)), dim=-1)


class GateAddNorm(nn.Module):
    """GLU(x) + skip, then LayerNorm (the TFT's gated skip connection)."""

    def __init__(self, d: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.glu = GatedLinearUnit(d, d, dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.norm(self.glu(x) + skip)


class GatedResidualNetwork(nn.Module):
    """GRN: ELU(W2 a [+ W3 c]) -> W1 -> dropout -> GLU, plus a residual and LayerNorm."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.0, d_context: int | None = None) -> None:
        super().__init__()
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.fc2 = nn.Linear(d_in, d_hidden)
        self.context = nn.Linear(d_context, d_hidden, bias=False) if d_context else None
        self.fc1 = nn.Linear(d_hidden, d_hidden)
        self.glu = GatedLinearUnit(d_hidden, d_out, dropout)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, a: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        h = self.fc2(a)
        if self.context is not None and c is not None:
            h = h + self.context(c)
        h = self.fc1(nn.functional.elu(h))
        return self.norm(self.skip(a) + self.glu(h))


class PerVariableGRN(nn.Module):
    """One GRN per variable (separate weights), evaluated for all variables in one batched op.

    Mathematically identical to ``n_vars`` independent ``GatedResidualNetwork``
    modules with ``d_in = d_hidden = d_out = d`` (identity skip), but avoids a
    Python loop over variables, which dominates training time otherwise.
    """

    def __init__(self, n_vars: int, d: int, dropout: float) -> None:
        super().__init__()
        def weight(d_in: int, d_out: int) -> nn.Parameter:
            w = torch.empty(n_vars, d_in, d_out)
            for i in range(n_vars):
                nn.init.kaiming_uniform_(w[i].T, a=math.sqrt(5))  # nn.Linear's default init
            return nn.Parameter(w)
        bound = 1 / math.sqrt(d)
        self.w2, self.b2 = weight(d, d), nn.Parameter(torch.empty(n_vars, d).uniform_(-bound, bound))
        self.w1, self.b1 = weight(d, d), nn.Parameter(torch.empty(n_vars, d).uniform_(-bound, bound))
        self.wg, self.bg = weight(d, 2 * d), nn.Parameter(torch.empty(n_vars, 2 * d).uniform_(-bound, bound))
        self.gamma, self.beta = nn.Parameter(torch.ones(n_vars, d)), nn.Parameter(torch.zeros(n_vars, d))
        self.dropout = nn.Dropout(dropout)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        # a: (B, T, V, d)
        h = torch.einsum("btvi,vio->btvo", a, self.w2) + self.b2
        h = torch.einsum("btvi,vio->btvo", nn.functional.elu(h), self.w1) + self.b1
        g = torch.einsum("btvi,vio->btvo", self.dropout(h), self.wg) + self.bg
        out = a + nn.functional.glu(g, dim=-1)
        mu = out.mean(-1, keepdim=True)
        var = out.var(-1, keepdim=True, unbiased=False)
        return (out - mu) / torch.sqrt(var + 1e-5) * self.gamma + self.beta


class VariableSelectionNetwork(nn.Module):
    """Per-timestep softmax weights over variables, applied to per-variable GRN outputs."""

    def __init__(self, n_vars: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.weights_grn = GatedResidualNetwork(n_vars * d_model, d_model, n_vars, dropout)
        self.var_grns = PerVariableGRN(n_vars, d_model, dropout)

    def forward(self, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # emb: (B, T, V, d)
        b, t, v, d = emb.shape
        weights = torch.softmax(self.weights_grn(emb.reshape(b, t, v * d)), dim=-1)  # (B, T, V)
        processed = self.var_grns(emb)  # (B, T, V, d)
        return (weights.unsqueeze(-1) * processed).sum(dim=2), weights


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention with a shared value projection and head-averaged output (TFT)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, self.d_head)
        self.out = nn.Linear(self.d_head, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, d); mask: (B, T, T) True where attention is NOT allowed.
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(x)  # shared across heads: (B, T, d_head)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)  # (B, H, T, T)
        scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        heads = self.dropout(attn) @ v.unsqueeze(1)  # (B, H, T, d_head)
        return self.out(heads.mean(dim=1)), attn.mean(dim=1)


class TemporalFusionTransformer(nn.Module):
    """TFT for sequence-to-one regression on windows of cycles.

    ``forward(x)`` takes ``(B, T, 2 * n_features + 1)`` and returns ``(B,)``,
    the (standardised) SOH estimate for the last cycle of each window. The
    most recent interpretation (variable-selection weights and attention of
    the last timestep) is kept in ``last_interpretation_``.
    """

    def __init__(self, n_features: int, d_model: int = 32, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_features, self.d_model = n_features, d_model
        # Per-variable embedding of (value, missing indicator) -> d_model.
        self.var_weight = nn.Parameter(torch.empty(n_features, 2, d_model))
        self.var_bias = nn.Parameter(torch.zeros(n_features, d_model))
        nn.init.xavier_uniform_(self.var_weight.view(n_features * 2, d_model))
        self.vsn = VariableSelectionNetwork(n_features, d_model, dropout)
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.post_lstm = GateAddNorm(d_model, dropout)
        self.enrichment = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        self.attention = InterpretableMultiHeadAttention(d_model, n_heads, dropout)
        self.post_attention = GateAddNorm(d_model, dropout)
        self.positionwise = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        self.pre_output = GateAddNorm(d_model, 0.0)
        self.head = nn.Linear(d_model, 1)
        self.last_interpretation_: dict[str, torch.Tensor] | None = None

    def _split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self.n_features
        if x.shape[-1] != 2 * f + 1:
            raise ValueError(f"expected last dimension {2 * f + 1}, got {x.shape[-1]}")
        return x[..., :f], x[..., f:2 * f], x[..., 2 * f] > 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values, missing, padded = self._split(x)
        pairs = torch.stack([values, missing], dim=-1)  # (B, T, V, 2)
        emb = torch.einsum("btvi,vid->btvd", pairs, self.var_weight) + self.var_bias
        selected, var_weights = self.vsn(emb)  # (B, T, d), (B, T, V)

        lstm_out, _ = self.lstm(selected)
        temporal = self.post_lstm(lstm_out, selected)
        enriched = self.enrichment(temporal)

        t = x.shape[1]
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        # Never attend to future or padded steps; every position may attend to itself,
        # so no row is fully masked (which would give NaN gradients through softmax).
        eye = torch.eye(t, dtype=torch.bool, device=x.device).unsqueeze(0)
        mask = (causal.unsqueeze(0) | padded.unsqueeze(1)) & ~eye
        attended, attn = self.attention(enriched, mask)
        h = self.post_attention(attended, enriched)
        h = self.positionwise(h)
        h = self.pre_output(h, temporal)

        self.last_interpretation_ = {
            "variable_weights": var_weights[:, -1].detach(),  # weights at the estimated cycle
            "attention": attn[:, -1].detach(),                # attention of the last cycle over the window
        }
        return self.head(h[:, -1]).squeeze(-1)


class CycleWindows:
    """Causal windows of recent cycles for each row, built from a label-free history.

    ``history`` holds every cycle's identifiers and features (target and RPT
    columns are dropped on construction, so labels can never enter a window).
    For a row, the window is the ``window`` most recent cycles of the same
    battery up to and including that cycle, ordered by ``Cycle_Date`` then
    ``Cycle_Index``; earlier positions are zero-padded and flagged.
    """

    KEY = ("Battery_ID", "Cycling_Test", "Cycle_Index")
    ORDER = ("Cycle_Date", "Cycle_Index")

    def __init__(self, history: pd.DataFrame, features, window: int = 16) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.features = list(features)
        self.window = int(window)
        cols = list(dict.fromkeys([*self.KEY, *self.ORDER, *self.features]))
        missing = [c for c in cols if c not in history.columns]
        if missing:
            raise KeyError(f"history lacks columns {missing}")
        h = history[cols].copy()
        h["Battery_ID"] = h["Battery_ID"].astype(str)
        if h.duplicated(subset=list(self.KEY)).any():
            raise ValueError("history has duplicate (Battery_ID, Cycling_Test, Cycle_Index) keys")
        self.history = h.sort_values(["Battery_ID", *self.ORDER], kind="mergesort").reset_index(drop=True)
        self._pos = pd.Series(
            self.history.groupby("Battery_ID").cumcount().to_numpy(),
            index=pd.MultiIndex.from_frame(self.history[list(self.KEY)].astype({"Cycle_Index": "int64"})),
        )
        self._cache: tuple[int, dict[str, np.ndarray]] | None = None

    def _encoded(self, preprocessor) -> dict[str, np.ndarray]:
        # Transform the history once per (frozen) preprocessor.
        if self._cache is None or self._cache[0] != id(preprocessor):
            enc = {}
            for battery, g in self.history.groupby("Battery_ID", sort=False):
                enc[battery] = preprocessor.transform(g)
            self._cache = (id(preprocessor), enc)
        return self._cache[1]

    def build(self, df: pd.DataFrame, preprocessor) -> np.ndarray:
        """``(len(df), window, 2 * n_features + 1)`` float32 windows for the rows of ``df``."""
        enc = self._encoded(preprocessor)
        f2 = 2 * len(self.features)
        out = np.zeros((len(df), self.window, f2 + 1), dtype=np.float32)
        out[:, :, f2] = 1.0  # padded until filled
        keys = pd.MultiIndex.from_arrays([
            df["Battery_ID"].astype(str).to_numpy(), df["Cycling_Test"].to_numpy(), df["Cycle_Index"].astype("int64").to_numpy(),
        ])
        pos = self._pos.reindex(keys).to_numpy()
        if np.isnan(pos.astype(float)).any():
            raise KeyError("some rows are not in the window history; pass the full cycle history")
        for i, (battery, p) in enumerate(zip(df["Battery_ID"].astype(str), pos.astype(int))):
            start = max(0, p - self.window + 1)
            seq = enc[battery][start:p + 1]
            out[i, self.window - len(seq):, :f2] = seq
            out[i, self.window - len(seq):, f2] = 0.0
        return out
