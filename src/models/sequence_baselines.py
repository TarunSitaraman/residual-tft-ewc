"""Sequence baselines from the literature (decision D24): BiLSTM, TCN and a Transformer encoder.

Every model takes the TFT's input contract, ``(B, T, 2 * n_features + 1)`` windows of standardised values, missing
indicators and a padding flag built by ``CycleWindows`` (inputs only, never labels), and returns ``(B,)``, the
(standardised) target for the last step of each window. Default sizes match the TFT of the D23/D24 runs
(``d_model=8``, two heads, the 19 D21 shared inputs: 12.7k parameters) to within 20 % (tested), so differences reflect
the architecture rather than model size.

- ``BiLSTMRegressor`` (Meskine et al. 2026 use BiLSTMs): a linear step embedding, one bidirectional LSTM layer over
  the window, and a head on the final states of both directions. The window holds only the origin and earlier
  origins, so reading it in both directions uses no future information.
- ``TCNRegressor`` (Yao et al. 2025, Meskine et al. 2026): residual blocks of dilated causal 1-D convolutions
  (kernel 2, dilations 1 and 2; receptive field 7 >= the window of 4) and a head on the last step.
- ``TransformerRegressor``: a linear step embedding with learned positions, one encoder layer of multi-head
  self-attention masked as in the TFT (no future or padded steps; every step may attend to itself) and a
  position-wise feed-forward block, and a head on the last step. It is the TFT without variable selection and
  gating.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["BiLSTMRegressor", "TCNRegressor", "TransformerRegressor", "SEQUENCE_BACKBONES", "n_parameters"]


def n_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _check(x: torch.Tensor, n_features: int) -> None:
    if x.shape[-1] != 2 * n_features + 1:
        raise ValueError(f"expected last dimension {2 * n_features + 1}, got {x.shape[-1]}")


class BiLSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int = 22, embed: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_features = n_features
        self.embed = nn.Linear(2 * n_features + 1, embed)
        self.lstm = nn.LSTM(embed, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check(x, self.n_features)
        _, (h, _) = self.lstm(torch.relu(self.embed(x)))
        # h: (2, B, hidden) = final forward state (after the last step) and final backward state (after the first).
        return self.head(self.dropout(torch.cat([h[0], h[1]], dim=-1))).squeeze(-1)


class _CausalBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, dilation: int, kernel: int, dropout: float) -> None:
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, C, T)
        h = self.dropout(torch.relu(self.conv1(F.pad(x, (self.pad, 0)))))   # left padding only: causal
        h = self.dropout(torch.relu(self.conv2(F.pad(h, (self.pad, 0)))))
        return torch.relu(h + self.skip(x))


class TCNRegressor(nn.Module):
    def __init__(self, n_features: int, channels: int = 36, kernel: int = 2, dilations=(1, 2),
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.n_features = n_features
        blocks, c_in = [], 2 * n_features + 1
        for d in dilations:
            blocks.append(_CausalBlock(c_in, channels, d, kernel, dropout))
            c_in = channels
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check(x, self.n_features)
        h = self.blocks(x.transpose(1, 2))                      # (B, C, T)
        return self.head(h[:, :, -1]).squeeze(-1)


class TransformerRegressor(nn.Module):
    def __init__(self, n_features: int, d_model: int = 32, n_heads: int = 2, d_ff: int = 64, max_len: int = 64,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.n_features, self.n_heads = n_features, n_heads
        self.embed = nn.Linear(2 * n_features + 1, d_model)
        self.position = nn.Parameter(torch.zeros(max_len, d_model))
        nn.init.normal_(self.position, std=0.02)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check(x, self.n_features)
        b, t, _ = x.shape
        padded = x[..., -1] > 0.5
        h = self.embed(x) + self.position[:t]
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        eye = torch.eye(t, dtype=torch.bool, device=x.device).unsqueeze(0)
        mask = (causal.unsqueeze(0) | padded.unsqueeze(1)) & ~eye             # True = not allowed (as in the TFT)
        mask = mask.repeat_interleave(self.n_heads, dim=0)                    # (B * heads, T, T)
        a, _ = self.attention(h, h, h, attn_mask=mask, need_weights=False)
        h = self.norm1(h + self.dropout(a))
        h = self.norm2(h + self.dropout(self.ff(h)))
        return self.head(h[:, -1]).squeeze(-1)


SEQUENCE_BACKBONES: dict[str, type[nn.Module]] = {
    "lstm": BiLSTMRegressor, "tcn": TCNRegressor, "transformer": TransformerRegressor,
}
