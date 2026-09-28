"""Experiment runners for the reference baselines (Stage A) and continual learning (Stages B–C).

See docs/RESEARCH_PLAN.md §3. Runners return plain DataFrames so notebooks
can tabulate and plot them; ``write_results`` adds provenance (D9).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import data_pipeline as dp
from src import evaluation as ev
from src import protocols as P
from src.continual_learning import Joint, make_strategy

TARGET = "SOH_Percent"
RIDGE_ALPHAS: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
CL_METRIC_COLS: tuple[str, ...] = (
    "final_avg_error", "plasticity", "backward_transfer", "forgetting", "forward_transfer", "train_time_s",
)


# ---------------------------------------------------------------------------
# Shared model fitting
# ---------------------------------------------------------------------------


def _ridge(alpha: float):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge(alpha=alpha))


def fit_predict(
    model: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    tests: Mapping[str, pd.DataFrame],
    features: Sequence[str],
    seed: int = 0,
    mlp_kwargs: Mapping | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Fit ``model`` ("ridge" or "mlp") on ``train``, select on ``val``, predict each test frame.

    Ridge chooses alpha by validation RMSE (as notebook 04). The MLP is the
    continual-learning backbone trained jointly (``Joint``), so Stage A and
    Stage C share one architecture.
    """
    features = list(features)
    if model == "ridge":
        scores = {}
        for alpha in RIDGE_ALPHAS:
            m = _ridge(alpha).fit(train[features], train[TARGET])
            scores[alpha] = ev.task_error(val[TARGET], m.predict(val[features])) if len(val) else np.nan
        best = min(scores, key=lambda a: (np.nan_to_num(scores[a], nan=np.inf), a))
        fitted = _ridge(best).fit(train[features], train[TARGET])
        return {k: fitted.predict(t[features]) for k, t in tests.items()}, {"model": "ridge", "alpha": best}
    if model == "mlp":
        joint = Joint(features, seed=seed, **(mlp_kwargs or {}))
        joint.fit_all([P.Task("pooled", train.reset_index(drop=True), val.reset_index(drop=True), val.iloc[0:0])])
        return {k: joint.predict(t) for k, t in tests.items()}, joint.config()
    raise ValueError(f"unknown model {model!r}; expected 'ridge' or 'mlp'")


def _score_rows(experiment: str, model: str, preds: Mapping[str, np.ndarray], tests: Mapping[str, pd.DataFrame]) -> list[dict]:
    rows = []
    for group, test in tests.items():
        m = dp.regression_metrics(test[TARGET], preds[group])
        rows.append({"experiment": experiment, "model": model, dp.GROUP_COL: group, **m})
    return rows


# ---------------------------------------------------------------------------
# Stage A: reference baselines (E1–E5)
# ---------------------------------------------------------------------------


def _pooled(df: pd.DataFrame, groups: Sequence[str], train_frac: float, val_frac: float):
    tasks = [P.make_task(df, g, train_frac, val_frac) for g in groups]
    train = pd.concat([t.train for t in tasks], ignore_index=True)
    val = pd.concat([t.val for t in tasks], ignore_index=True)
    return train, val, {t.name: t.test for t in tasks}


def reference_baselines(
    df: pd.DataFrame,
    health: pd.DataFrame | None = None,
    models: Sequence[str] = ("ridge", "mlp"),
    seed: int = 0,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    mlp_kwargs: Mapping | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run E1–E5 and return one row per (experiment, model, battery group) plus details.

    * E1 within-group: one model on every group's chronological train period,
      evaluated on each group's test period (the MLP version is E4, joint training).
    * E2 RPT-boundary: as E1, but splits cut only at RPT changes, so test SOH
      states never appear in training.
    * E3 leave-one-group-out: train on the other groups, test on the held-out
      group's E1 test period (unseen-group error, RQ1).
    * E5 date-only sensitivity: E1 after relabelling same-day cycles with the
      ``"end"`` convention (needs ``health``); labels differ from E1 by design.
    """
    features = dp.model_feature_columns(df)
    groups = [g for g in dp.BATTERY_GROUPS if g in set(df[dp.GROUP_COL].astype(str))]
    rows: list[dict] = []
    details: dict = {"features": features, "groups": groups, "models": {}}

    for model in models:
        # E1 / E4
        train, val, tests = _pooled(df, groups, train_frac, val_frac)
        preds, cfg = fit_predict(model, train, val, tests, features, seed, mlp_kwargs)
        rows += _score_rows("E1_within_group", model, preds, tests)
        details["models"][f"E1_{model}"] = cfg

        # E2
        split = P.rpt_boundary_split(df, train_frac, val_frac)
        e2_train = split[split["split"] == "train"]
        e2_val = split[split["split"] == "val"]
        e2_tests = {g: split[(split["split"] == "test") & (split[dp.GROUP_COL].astype(str) == g)] for g in groups}
        preds, cfg = fit_predict(model, e2_train, e2_val, e2_tests, features, seed, mlp_kwargs)
        rows += _score_rows("E2_rpt_boundary", model, preds, e2_tests)
        details["models"][f"E2_{model}"] = cfg

        # E3
        for g in groups:
            task = P.leave_one_group_out(df, g, train_frac, val_frac)
            preds, cfg = fit_predict(model, task.train, task.val, {g: task.test}, features, seed, mlp_kwargs)
            rows += _score_rows("E3_leave_one_group_out", model, preds, {g: task.test})
            details["models"][f"E3_{model}_{g}"] = cfg

        # E5
        if health is not None:
            relabelled = P.relabel_date_only(df, health, "end")
            details["E5_rows_dropped"] = int(len(df) - len(relabelled))
            details["E5_labels_changed"] = int(relabelled["label_changed"].sum())
            train, val, tests = _pooled(relabelled, groups, train_frac, val_frac)
            preds, cfg = fit_predict(model, train, val, tests, features, seed, mlp_kwargs)
            rows += _score_rows("E5_date_only_end", model, preds, tests)
            details["models"][f"E5_{model}"] = cfg

    return pd.DataFrame(rows), details


# ---------------------------------------------------------------------------
# Stages B–C: continual learning
# ---------------------------------------------------------------------------


@dataclass
class CLConfig:
    """Continual-learning experiment grid. Defaults are the paper configuration (plan §3)."""

    methods: tuple[str, ...] = ("finetune", "replay", "ewc", "lwf")
    n_orders: int = 10
    order_seed: int = 0
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    metric: str = "RMSE"
    train_frac: float = 0.70
    val_frac: float = 0.15
    common_kwargs: dict = field(default_factory=dict)       # passed to every strategy (epochs, lr, ...)
    method_kwargs: dict = field(default_factory=dict)       # per-method extras, e.g. {"ewc": {"ewc_lambda": 100}}


def _task_errors(strategy, tasks: Sequence[P.Task], metric: str, split: str = "test") -> dict[str, float]:
    if split not in ("test", "val"):
        raise ValueError(f"split must be 'test' or 'val', got {split!r}")
    return {t.name: ev.task_error(getattr(t, split)[TARGET], strategy.predict(getattr(t, split)), metric) for t in tasks}


def run_stream(method: str, tasks: Sequence[P.Task], seed: int, metric: str = "RMSE", eval_split: str = "test", **kwargs) -> dict:
    """Train ``method`` through ``tasks`` in order, evaluating every task after every step.

    Returns the accuracy matrix (all entries, including tasks not yet seen, which
    feed forward transfer), the untrained reference errors ``b_j``, CL metrics
    and per-task training time. ``eval_split="val"`` evaluates on the validation
    periods instead, for hyperparameter tuning; test periods are then never read.
    """
    features = dp.model_feature_columns(tasks[0].train)
    reference = make_strategy(method, features, seed=seed, **kwargs).initialise(tasks[0])
    b = _task_errors(reference, tasks, metric, eval_split)

    strategy = make_strategy(method, features, seed=seed, **kwargs)
    errors: dict[tuple[int, str], float] = {}
    for i, task in enumerate(tasks):
        strategy.fit_task(task)
        for name, err in _task_errors(strategy, tasks, metric, eval_split).items():
            errors[(i, name)] = err
    order = [t.name for t in tasks]
    R = ev.accuracy_matrix(errors, order)
    metrics = ev.cl_metrics(R, reference_errors=b)
    return {"R": R, "metrics": metrics, "reference_errors": b,
            "train_time_s": list(strategy.training_time_), "config": strategy.config()}


def _warm_up(df: pd.DataFrame) -> None:
    """One tiny fit so one-off library initialisation is not charged to the first timed run."""
    group = sorted(set(df[dp.GROUP_COL].astype(str)))[0]
    task = P.make_task(df, group)
    make_strategy("finetune", dp.model_feature_columns(df), epochs=1).fit_task(task)


def run_cl_experiment(df: pd.DataFrame, config: CLConfig) -> dict:
    """Run every method over every (order, seed) pair and the joint upper bound per seed.

    Returns ``runs`` (one row per method × order × seed with scalar CL metrics),
    ``matrices`` (accuracy matrices keyed by (method, order_id, seed)),
    ``joint`` (per-task test error of joint training per seed), ``orders`` and
    the config.
    """
    groups = [g for g in dp.BATTERY_GROUPS if g in set(df[dp.GROUP_COL].astype(str))]
    orders = P.sample_orders(groups, config.n_orders, config.order_seed)
    _warm_up(df)

    runs, matrices = [], {}
    for order_id, order in enumerate(orders):
        tasks = P.make_task_stream(df, order, config.train_frac, config.val_frac)
        for seed in config.seeds:
            for method in config.methods:
                kwargs = {**config.common_kwargs, **config.method_kwargs.get(method, {})}
                out = run_stream(method, tasks, seed, config.metric, **kwargs)
                matrices[(method, order_id, seed)] = out["R"]
                m = out["metrics"]
                runs.append({
                    "method": method, "order_id": order_id, "order": ">".join(order), "seed": seed,
                    **{k: m[k] for k in CL_METRIC_COLS if k in m},
                    "train_time_s": float(np.sum(out["train_time_s"])),
                })

    tasks = P.make_task_stream(df, groups, config.train_frac, config.val_frac)
    joint_rows = []
    features = dp.model_feature_columns(df)
    for seed in config.seeds:
        joint = Joint(features, seed=seed, **config.common_kwargs).fit_all(tasks)
        for name, err in _task_errors(joint, tasks, config.metric).items():
            joint_rows.append({"method": "joint", "seed": seed, dp.GROUP_COL: name, "error": err,
                               "train_time_s": float(np.sum(joint.training_time_))})

    return {"runs": pd.DataFrame(runs), "matrices": matrices, "joint": pd.DataFrame(joint_rows),
            "orders": orders, "config": asdict(config)}


# ---------------------------------------------------------------------------
# Hyperparameter tuning on validation data (decision D11)
# ---------------------------------------------------------------------------

#: Method-specific grids, fixed before any real-data run. Shared settings
#: (learning rate, epochs, batch size, backbone) are not tuned, so no method
#: receives a larger tuning budget than another.
DEFAULT_GRIDS: dict[str, dict[str, list]] = {
    "replay": {"buffer_per_task": [50, 100, 200, 400]},
    "ewc": {"ewc_lambda": [1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0]},
    "lwf": {"alpha": [0.1, 0.5, 1.0, 2.0, 5.0]},
}


@dataclass
class TuningConfig:
    """Validation-only tuning grid.

    Tuning streams use ``n_orders`` group orders that are guaranteed disjoint
    from the evaluation orders of the matching ``CLConfig``, and are evaluated
    on the tasks' validation periods only.
    """

    grids: dict = field(default_factory=lambda: {m: {k: list(v) for k, v in g.items()} for m, g in DEFAULT_GRIDS.items()})
    n_orders: int = 3
    seeds: tuple[int, ...] = (0,)
    criterion: str = "final_avg_error"


def _grid_points(grid: Mapping[str, Sequence]) -> list[dict]:
    points = [{}]
    for name, values in grid.items():
        points = [{**p, name: v} for p in points for v in values]
    return points


def tuning_orders(df: pd.DataFrame, cl_config: CLConfig, n_orders: int) -> list[tuple[str, ...]]:
    """Orders for tuning, disjoint from ``cl_config``'s evaluation orders.

    ``sample_orders`` extends its list deterministically, so the orders after the
    first ``cl_config.n_orders`` can never coincide with an evaluation order.
    """
    groups = [g for g in dp.BATTERY_GROUPS if g in set(df[dp.GROUP_COL].astype(str))]
    return P.sample_orders(groups, cl_config.n_orders + n_orders, cl_config.order_seed)[cl_config.n_orders:]


def tune_hyperparameters(df: pd.DataFrame, cl_config: CLConfig, tuning: TuningConfig) -> dict:
    """Select each method's own hyperparameters on validation data only.

    For every method with a grid and every grid point, run the stream over the
    tuning orders and seeds with ``eval_split="val"`` and average
    ``tuning.criterion`` (default: final average validation error, which
    reflects both adaptation and retention). Ties go to the earlier grid value.
    Returns the full grid table, the selected values per method, whether each
    selected value lies at the edge of its grid (then the grid should be
    extended and tuning repeated, so the optimum is not an artefact of the grid
    range), and a ``CLConfig`` whose ``method_kwargs`` hold the selection.
    """
    orders = tuning_orders(df, cl_config, tuning.n_orders)
    streams = [P.make_task_stream(df, o, cl_config.train_frac, cl_config.val_frac) for o in orders]
    _warm_up(df)
    rows, points = [], {}
    for method in cl_config.methods:
        grid = tuning.grids.get(method)
        if not grid:
            continue
        points[method] = _grid_points(grid)
        for point_id, point in enumerate(points[method]):
            kwargs = {**cl_config.common_kwargs, **cl_config.method_kwargs.get(method, {}), **point}
            for order_id, tasks in enumerate(streams):
                for seed in tuning.seeds:
                    out = run_stream(method, tasks, seed, cl_config.metric, eval_split="val", **kwargs)
                    rows.append({"method": method, "point_id": point_id, **{f"param_{k}": v for k, v in point.items()},
                                 "order_id": order_id, "seed": seed,
                                 **{k: out["metrics"][k] for k in CL_METRIC_COLS if k in out["metrics"]}})
    grid_runs = pd.DataFrame(rows)

    selected, at_edge, table = {}, {}, []
    if len(grid_runs):
        param_cols = [c for c in grid_runs.columns if c.startswith("param_")]
        agg = (grid_runs.groupby(["method", "point_id"], as_index=False)
               .agg(**{c: (c, "first") for c in param_cols},
                    **{f"val_{m}": (m, "mean") for m in ("final_avg_error", "plasticity", "forgetting")},
                    n_runs=("seed", "size")))
        for method, g in agg.groupby("method", sort=False):
            best = g.sort_values([f"val_{tuning.criterion}", "point_id"], kind="mergesort").iloc[0]
            # Take values from the grid itself so types (int buffer sizes) are exact.
            selected[method] = dict(points[method][int(best["point_id"])])
            grid = tuning.grids[method]
            at_edge[method] = {k: len(grid[k]) > 1 and v in (min(grid[k]), max(grid[k])) for k, v in selected[method].items()}
            table.append(g.assign(selected=g["point_id"] == best["point_id"]))
    tuned = CLConfig(**{**asdict(cl_config), "method_kwargs": {**cl_config.method_kwargs,
                        **{m: {**cl_config.method_kwargs.get(m, {}), **p} for m, p in selected.items()}}})
    return {"grid_runs": grid_runs, "grid_summary": pd.concat(table, ignore_index=True) if table else pd.DataFrame(),
            "selected": selected, "at_grid_edge": at_edge, "orders": orders, "config": asdict(tuning), "tuned_cl_config": tuned}


def compare_methods(runs: pd.DataFrame, baseline: str = "finetune", metric_cols: Sequence[str] = CL_METRIC_COLS[:5]) -> pd.DataFrame:
    """Paired comparison of each method against ``baseline`` on matched (order, seed) runs.

    The Holm family is fixed in advance as all (method vs. baseline) comparisons
    for one metric; differences are ``method − baseline`` (negative = lower error
    or less forgetting than the baseline).
    """
    rows = []
    key = ["order_id", "seed"]
    base = runs[runs["method"] == baseline].set_index(key)
    for metric in metric_cols:
        pvals, block = {}, []
        for method in sorted(set(runs["method"]) - {baseline}):
            other = runs[runs["method"] == method].set_index(key)
            common = base.index.intersection(other.index)
            res = ev.paired_comparison(other.loc[common, metric].to_numpy(), base.loc[common, metric].to_numpy())
            block.append({"metric": metric, "method": method, "baseline": baseline, **res})
            pvals[method] = res["p_value"]
        adjusted = ev.holm_correction(pvals)
        for r in block:
            r["p_holm"] = adjusted[r["method"]]
        rows += block
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _jsonable(obj):
    if isinstance(obj, pd.DataFrame):
        return json.loads(obj.to_json(orient="records", date_format="iso"))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def write_results(path: Path, payload: Mapping, repo_root: Path | None = None) -> Path:
    """Write a small results JSON with generation time and environment/commit provenance (D9)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "environment": dp.environment_info(repo_root), **payload}
    path.write_text(json.dumps(_jsonable(report), indent=2, default=str))
    return path
