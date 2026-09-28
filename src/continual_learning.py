"""Continual-learning strategies for SOH regression over a stream of battery groups.

All methods (docs/RESEARCH_PLAN.md §3 Stage C) share one interface and one
backbone (``MLPRegressor`` on ``TabularPreprocessor`` outputs), so they differ
only in how they limit forgetting:

- ``FineTune``: plain sequential training (lower bound, E6).
- ``Replay``: rehearsal of a fixed-size random sample of each finished task (E7).
- ``EWC``: Elastic Weight Consolidation, one diagonal Fisher per task (E8a).
- ``LwF``: Learning without Forgetting, output distillation to the previous model (E8b).
- ``Joint``: training on all tasks together (upper bound, E4).

Strict continual-learning rules: the preprocessor and the target scaling are
fitted once, on the first task's training rows, then frozen, so no strategy
(except ``Joint``, by design) sees statistics from future tasks. Early stopping
uses the current task's validation period only; test rows are never touched
during training. Mini-batches are shuffled *within* the current training set
with a seeded generator; this does not affect the chronological train/val/test
boundaries.
"""

from __future__ import annotations

import copy
import time
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.models.baseline_dnn import MLPRegressor, TabularPreprocessor, set_determinism
from src.protocols import Task

__all__ = ["Strategy", "FineTune", "Replay", "EWC", "LwF", "Joint", "make_strategy", "STRATEGIES"]

# SOH is in percentage points; a first task whose training period spans a single
# RPT has (near-)zero target spread, so the target scale is floored at 1 pp to
# avoid inflating later tasks' standardised targets.
MIN_TARGET_SCALE = 1.0


class Strategy:
    """Base class: sequential training of one MLP with early stopping per task.

    Subclasses customise ``_training_frame`` (what to train on), ``_penalty``
    (an extra loss term per mini-batch) and the ``_before_task`` /
    ``_after_task`` hooks.
    """

    name = "base"

    def __init__(
        self,
        features: Sequence[str],
        target: str = "SOH_Percent",
        seed: int = 0,
        hidden: Sequence[int] = (128, 64),
        lr: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 64,
        patience: int = 20,
        weight_decay: float = 0.0,
        dropout: float = 0.1,
        standardise_target: bool = True,
    ) -> None:
        self.features = list(features)
        self.target = target
        self.seed = int(seed)
        self.hidden = tuple(hidden)
        self.lr = lr
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.standardise_target = standardise_target

        self.preprocessor: TabularPreprocessor | None = None
        self.model: MLPRegressor | None = None
        self.target_mean_: float = 0.0
        self.target_scale_: float = 1.0
        self.tasks_seen_: list[str] = []
        self.training_time_: list[float] = []
        self.best_epoch_: list[int] = []
        self.best_val_rmse_: list[float] = []

    # -- configuration / provenance ---------------------------------------
    def config(self) -> dict:
        """Hyperparameters to record with results."""
        return {
            "strategy": self.name, "target": self.target, "seed": self.seed, "hidden": list(self.hidden),
            "lr": self.lr, "epochs": self.epochs, "batch_size": self.batch_size, "patience": self.patience,
            "weight_decay": self.weight_decay, "dropout": self.dropout,
            "standardise_target": self.standardise_target, "n_features": len(self.features),
        }

    # -- setup ------------------------------------------------------------
    def _initialise(self, train: pd.DataFrame) -> None:
        """Fit the frozen preprocessor and target scaling on ``train`` and build the model."""
        set_determinism(self.seed)
        self.preprocessor = TabularPreprocessor(self.features).fit(train)
        y = self._target(train)
        if self.standardise_target:
            self.target_mean_ = float(np.mean(y))
            self.target_scale_ = float(max(np.std(y), MIN_TARGET_SCALE))
        else:
            self.target_mean_, self.target_scale_ = 0.0, 1.0
        self.model = MLPRegressor(self.preprocessor.output_dim, self.hidden, self.dropout)

    def _target(self, df: pd.DataFrame) -> np.ndarray:
        y = df[self.target].to_numpy(dtype=np.float64)
        if np.isnan(y).any():
            raise ValueError(f"NaN in target {self.target!r}; use SOH-labelled rows only")
        return y

    def _xy(self, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.preprocessor.transform(df))
        y = torch.from_numpy(((self._target(df) - self.target_mean_) / self.target_scale_).astype(np.float32))
        return x, y

    # -- hooks ------------------------------------------------------------
    def _training_frame(self, task: Task) -> pd.DataFrame:
        return task.train

    def _before_task(self, task: Task) -> None:
        pass

    def _after_task(self, task: Task) -> None:
        pass

    def _penalty(self, xb: torch.Tensor) -> torch.Tensor:
        return torch.zeros(())

    # -- training ---------------------------------------------------------
    def fit_task(self, task: Task) -> "Strategy":
        """Train on the next task of the stream (early stopping on ``task.val``)."""
        start = time.perf_counter()
        if self.model is None:
            self._initialise(task.train)
        self._before_task(task)
        self._train(self._training_frame(task), task.val, task_index=len(self.tasks_seen_))
        self._after_task(task)
        self.tasks_seen_.append(task.name)
        self.training_time_.append(time.perf_counter() - start)
        return self

    def _train(self, train: pd.DataFrame, val: pd.DataFrame, task_index: int) -> None:
        # Reseed per task so each step is reproducible irrespective of what ran before.
        set_determinism(self.seed + 1000 * task_index)
        gen = torch.Generator().manual_seed(self.seed + 1000 * task_index)
        x, y = self._xy(train)
        # Without validation rows, early stopping falls back to training RMSE.
        xv, yv = self._xy(val) if len(val) else (x, y)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        best_rmse, best_state, best_epoch, bad = self._val_rmse(xv, yv), copy.deepcopy(self.model.state_dict()), -1, 0
        n = len(x)
        for epoch in range(self.epochs):
            self.model.train()
            perm = torch.randperm(n, generator=gen)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                xb, yb = x[idx], y[idx]
                loss = loss_fn(self.model(xb), yb) + self._penalty(xb)
                opt.zero_grad()
                loss.backward()
                opt.step()
            rmse = self._val_rmse(xv, yv)
            if rmse < best_rmse - 1e-9:
                best_rmse, best_state, best_epoch, bad = rmse, copy.deepcopy(self.model.state_dict()), epoch, 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.model.load_state_dict(best_state)
        self.model.eval()
        self.best_epoch_.append(best_epoch)
        self.best_val_rmse_.append(best_rmse)

    @torch.no_grad()
    def _val_rmse(self, xv: torch.Tensor, yv: torch.Tensor) -> float:
        """Validation RMSE in target units (SOH percentage points)."""
        self.model.eval()
        err = (self.model(xv) - yv) * self.target_scale_
        return float(torch.sqrt(torch.mean(err ** 2)))

    # -- inference --------------------------------------------------------
    @torch.no_grad()
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predictions in target units, in the row order of ``df``."""
        if self.model is None:
            raise RuntimeError(f"{type(self).__name__}.predict called before any task was fitted")
        self.model.eval()
        x = torch.from_numpy(self.preprocessor.transform(df))
        out = self.model(x).numpy().astype(np.float64)
        return out * self.target_scale_ + self.target_mean_


class FineTune(Strategy):
    """Sequential fine-tuning with no protection against forgetting (lower bound)."""

    name = "finetune"


class Replay(Strategy):
    """Experience replay: a seeded random sample of each finished task's training rows.

    The buffer grows by ``buffer_per_task`` rows per task (fewer if a task has
    fewer training rows) and is concatenated with the current task's training
    rows. Only past tasks' *training* rows are stored.
    """

    name = "replay"

    def __init__(self, features: Sequence[str], buffer_per_task: int = 200, **kwargs) -> None:
        super().__init__(features, **kwargs)
        self.buffer_per_task = int(buffer_per_task)
        self.buffer_: list[pd.DataFrame] = []

    def config(self) -> dict:
        return {**super().config(), "buffer_per_task": self.buffer_per_task}

    @property
    def buffer(self) -> pd.DataFrame:
        """All stored rows (empty frame before the first task ends)."""
        return pd.concat(self.buffer_, ignore_index=True) if self.buffer_ else pd.DataFrame()

    def _training_frame(self, task: Task) -> pd.DataFrame:
        if not self.buffer_:
            return task.train
        return pd.concat([task.train, *self.buffer_], ignore_index=True)

    def _after_task(self, task: Task) -> None:
        rng = np.random.default_rng(self.seed + 1000 * len(self.tasks_seen_))
        k = min(self.buffer_per_task, len(task.train))
        idx = np.sort(rng.choice(len(task.train), size=k, replace=False))
        self.buffer_.append(task.train.iloc[idx].copy())


class EWC(Strategy):
    """Elastic Weight Consolidation with one diagonal Fisher per finished task.

    Variant: the original multi-penalty form (Kirkpatrick et al., 2017), not the
    online variant. After task k the empirical diagonal Fisher ``F_k`` is the
    mean of squared per-sample gradients of the (standardised-target) MSE loss
    over up to ``fisher_samples`` seeded training rows, and ``theta*_k`` is the
    early-stopped optimum. The penalty is

        (ewc_lambda / 2) * sum_k sum_i F_k,i * (theta_i - theta*_k,i)^2.

    Storage grows linearly with the number of tasks, which is negligible for six
    groups and a small MLP.
    """

    name = "ewc"

    def __init__(self, features: Sequence[str], ewc_lambda: float = 100.0, fisher_samples: int = 500, **kwargs) -> None:
        super().__init__(features, **kwargs)
        self.ewc_lambda = float(ewc_lambda)
        self.fisher_samples = int(fisher_samples)
        self.fishers_: list[dict[str, torch.Tensor]] = []
        self.optima_: list[dict[str, torch.Tensor]] = []

    def config(self) -> dict:
        return {**super().config(), "ewc_lambda": self.ewc_lambda, "fisher_samples": self.fisher_samples,
                "ewc_variant": "multi-penalty (one Fisher per task)"}

    def penalty(self) -> torch.Tensor:
        """Current EWC penalty (zero before the first task ends)."""
        total = torch.zeros(())
        if self.model is None:
            return total
        for fisher, optimum in zip(self.fishers_, self.optima_):
            for n, p in self.model.named_parameters():
                total = total + (fisher[n] * (p - optimum[n]) ** 2).sum()
        return 0.5 * self.ewc_lambda * total

    def _penalty(self, xb: torch.Tensor) -> torch.Tensor:
        return self.penalty()

    def _after_task(self, task: Task) -> None:
        rng = np.random.default_rng(self.seed + 1000 * len(self.tasks_seen_) + 1)
        k = min(self.fisher_samples, len(task.train))
        idx = np.sort(rng.choice(len(task.train), size=k, replace=False))
        x, y = self._xy(task.train.iloc[idx])
        params = dict(self.model.named_parameters())
        fisher = {n: torch.zeros_like(p) for n, p in params.items()}
        self.model.eval()  # no dropout noise in the curvature estimate
        for i in range(k):
            self.model.zero_grad()
            loss = (self.model(x[i:i + 1]) - y[i:i + 1]).pow(2).sum()
            loss.backward()
            for n, p in params.items():
                fisher[n] += p.grad.detach() ** 2
        self.model.zero_grad()
        self.fishers_.append({n: f / k for n, f in fisher.items()})
        self.optima_.append({n: p.detach().clone() for n, p in params.items()})


class LwF(Strategy):
    """Learning without Forgetting for regression.

    Before each new task the current model is frozen as a teacher; training adds
    ``alpha * MSE(student(x), teacher(x))`` on the current task's mini-batch
    inputs (standardised-target units), so no past data are stored.
    """

    name = "lwf"

    def __init__(self, features: Sequence[str], alpha: float = 1.0, **kwargs) -> None:
        super().__init__(features, **kwargs)
        self.alpha = float(alpha)
        self.teacher_: MLPRegressor | None = None

    def config(self) -> dict:
        return {**super().config(), "alpha": self.alpha}

    def _before_task(self, task: Task) -> None:
        if not self.tasks_seen_:
            return
        self.teacher_ = copy.deepcopy(self.model)
        self.teacher_.eval()
        for p in self.teacher_.parameters():
            p.requires_grad_(False)

    def _penalty(self, xb: torch.Tensor) -> torch.Tensor:
        if self.teacher_ is None:
            return torch.zeros(())
        with torch.no_grad():
            target = self.teacher_(xb)
        return self.alpha * torch.mean((self.model(xb) - target) ** 2)


class Joint(Strategy):
    """Joint training on several tasks at once (upper bound, E4).

    ``fit_all(tasks)`` trains a fresh model once on the concatenation of all
    tasks' training rows, with early stopping on the concatenated validation
    rows; the preprocessor and target scaling are fitted on the concatenated
    training rows (Joint is allowed to see every task by design).

    ``fit_task(task)`` is the cumulative variant: it adds ``task`` to the tasks
    seen so far and calls ``fit_all`` on them, i.e. retrains from scratch at
    every step. This gives an upper-bound row for each step of the accuracy
    matrix.
    """

    name = "joint"

    def __init__(self, features: Sequence[str], **kwargs) -> None:
        super().__init__(features, **kwargs)
        self.seen_tasks_: list[Task] = []

    def fit_all(self, tasks: Sequence[Task]) -> "Joint":
        if not tasks:
            raise ValueError("Joint.fit_all needs at least one task")
        start = time.perf_counter()
        train = pd.concat([t.train for t in tasks], ignore_index=True)
        val = pd.concat([t.val for t in tasks], ignore_index=True)
        self.model = None
        self._initialise(train)
        self._train(train, val, task_index=0)
        self.tasks_seen_ = [t.name for t in tasks]
        self.training_time_.append(time.perf_counter() - start)
        return self

    def fit_task(self, task: Task) -> "Joint":
        self.seen_tasks_.append(task)
        return self.fit_all(self.seen_tasks_)


STRATEGIES: dict[str, type[Strategy]] = {
    "finetune": FineTune, "replay": Replay, "ewc": EWC, "lwf": LwF, "joint": Joint,
}


def make_strategy(name: str, features: Sequence[str], **kwargs) -> Strategy:
    """Build a strategy by name ("finetune", "replay", "ewc", "lwf", "joint")."""
    key = name.lower()
    if key not in STRATEGIES:
        raise ValueError(f"Unknown strategy {name!r}; choose from {sorted(STRATEGIES)}")
    return STRATEGIES[key](features, **kwargs)
