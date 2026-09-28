"""Task and stream construction for reference baselines and continual learning.

See docs/RESEARCH_PLAN.md §3–4. A task is one battery group split
chronologically into train / validation / test; nothing is shuffled.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Task:
    """One battery group's chronological train / validation / test periods."""

    name: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
