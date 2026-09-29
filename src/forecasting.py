"""SOH forecasting samples, purged splits and baselines (decision D18).

A sample is (battery, origin RPT k, horizon h): inputs known at RPT k, target
the SOH change to RPT k+h on the C/20 basis (D17). Samples are laid out like
the cycle-level tables so the continual-learning framework can run on them:

* ``Battery_ID``: battery (the task);
* ``Cycling_Test``: ``"h1"``, ``"h2"``, ``"h3"`` (TFT windows group by battery
  and horizon, so a window holds earlier origins of the same horizon);
* ``Cycle_Index``: origin RPT number; ``Cycle_Date``: origin RPT start time;
* ``dsoh``: target; ``soh_target``: SOH at the target RPT (evaluation only).

Nothing here reads a label from after the origin into the inputs; the tests
in ``tests/test_forecasting.py`` check this.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src import data_pipeline as dp
from src import diagnostic_features as F
from src import protocols as P

TARGET = "dsoh"
HORIZONS: tuple[int, ...] = (1, 2, 3)
EXCLUDED_RPTS: tuple[int, ...] = (2,)        # RPT 2 run at ~20 degC (D16)
SPLIT_FRACS: tuple[float, float] = (0.6, 0.8)
SETPOINTS: tuple[float, ...] = (20.0, 35.0)
#: Cycling_1 has no temperature log; its setpoint is from Moy et al. 2024, Table 3 (Batch 1 at 35 degC).
UNLOGGED_SETPOINTS: dict[str, float] = {"Cycling_1": 35.0}

#: Moy et al. 2024, Table 2: first-life charge C-rate, first-life cycles,
#: capacity at end of first life (Ah) and second-life duty cycle.
FIRST_LIFE: pd.DataFrame = pd.DataFrame(
    [("v4", 0.25, 244, 4.6024, "residential"), ("w8", 0.5, 347, 4.4568, "commercial"),
     ("w9", 1.0, 341, 4.4636, "residential"), ("v5", 1.0, 29, 4.7055, "commercial"),
     ("w10", 3.0, 350, 4.4591, "residential"), ("g1", 3.0, 212, 4.6000, "commercial")],
    columns=[dp.GROUP_COL, "fl_c_rate", "fl_cycles", "fl_end_capacity_ah", "duty"],
)

HISTORY_FEATURES: tuple[str, ...] = ("soh_origin", "days_since_first", "fade_rate_pp_per_day", "horizon")
STATIC_FEATURES: tuple[str, ...] = ("residential", "fl_c_rate", "fl_cycles", "fl_end_capacity_ah")
SCHEDULE_FEATURES: tuple[str, ...] = ("days_to_target", "days_35c", "days_20c", "days_rest")


# ---------------------------------------------------------------------------
# Schedule (known-future inputs)
# ---------------------------------------------------------------------------

def cycling_blocks(points: pd.DataFrame) -> pd.DataFrame:
    """Start, end and chamber setpoint of every (battery, cycling block) of the point-level data."""
    temp = next((c for c in points.columns if c.startswith("Aux_Temperature")), None)
    g = points.assign(**{dp.GROUP_COL: points["Source_Folder"].astype(str).str.lower()}).groupby(
        [dp.GROUP_COL, "Cycling_Test"])
    b = g.agg(start=("Date_Time", "min"), end=("Date_Time", "max"),
              temp_median=(temp, "median") if temp else ("Date_Time", "size")).reset_index()
    if not temp:
        b["temp_median"] = np.nan
    nearest = b["temp_median"].map(lambda t: min(SETPOINTS, key=lambda s: abs(s - t)) if pd.notna(t) else np.nan)
    b["setpoint_c"] = nearest.fillna(b["Cycling_Test"].map(UNLOGGED_SETPOINTS))
    if b["setpoint_c"].isna().any():
        raise ValueError(f"blocks without a setpoint: {b.loc[b['setpoint_c'].isna(), 'Cycling_Test'].tolist()}")
    return b


def interval_schedule(blocks: pd.DataFrame, battery: str, t0: pd.Timestamp, t1: pd.Timestamp) -> dict:
    """Days of cycling at each setpoint and days without cycling between ``t0`` and ``t1``."""
    b = blocks[blocks[dp.GROUP_COL] == battery]
    total = (t1 - t0).total_seconds() / 86400
    out = {"days_to_target": total}
    cyc = 0.0
    for sp, name in ((35.0, "days_35c"), (20.0, "days_20c")):
        s = b[b["setpoint_c"] == sp]
        overlap = (np.minimum(s["end"], t1) - np.maximum(s["start"], t0)).dt.total_seconds().clip(lower=0) / 86400
        out[name] = float(overlap.sum())
        cyc += out[name]
    out["days_rest"] = max(total - cyc, 0.0)
    return out


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------

def _fade_rate(days: np.ndarray, soh: np.ndarray, n: int = 3) -> float:
    """Least-squares slope (pp/day) of SOH over the last ``n`` valid RPTs; NaN with fewer."""
    if len(days) < n:
        return np.nan
    d, s = days[-n:], soh[-n:]
    return float(np.polyfit(d, s, 1)[0])


def build_samples(table: pd.DataFrame, blocks: pd.DataFrame, horizons: Sequence[int] = HORIZONS,
                  excluded_rpts: Sequence[int] = EXCLUDED_RPTS) -> pd.DataFrame:
    """Forecasting samples from the diagnostic table (with C/20 SOH) and the cycling blocks.

    ``table`` is ``data/diagnostic_features.csv`` (one row per battery and RPT,
    including ``SOH_Percent`` on the C/20 basis and ``rpt_start``).
    """
    diag = F.feature_columns(table)
    t = table[~table["rpt_number"].isin(excluded_rpts)].copy()
    t["rpt_start"] = pd.to_datetime(t["rpt_start"])
    t = t.sort_values([dp.GROUP_COL, "rpt_start"]).reset_index(drop=True)
    static = FIRST_LIFE.assign(residential=(FIRST_LIFE["duty"] == "residential").astype(float)).drop(columns="duty")
    rows = []
    for bat, g in t.groupby(dp.GROUP_COL, sort=True):
        g = g.reset_index(drop=True)
        t_first = g["rpt_start"].iloc[0]
        days = ((g["rpt_start"] - t_first).dt.total_seconds() / 86400).to_numpy()
        soh = g["SOH_Percent"].to_numpy(float)
        for i in range(len(g)):
            origin = g.iloc[i]
            base = {dp.GROUP_COL: bat, "Cycle_Index": int(origin["rpt_number"]), "Cycle_Date": origin["rpt_start"],
                    "origin_rpt": int(origin["rpt_number"]), "origin_pos": i, "n_valid_rpts": len(g),
                    "soh_origin": soh[i], "days_since_first": days[i],
                    "fade_rate_pp_per_day": _fade_rate(days[:i + 1], soh[:i + 1]),
                    **{c: origin[c] for c in diag}}
            for h in horizons:
                j = i + h
                if j >= len(g):
                    continue
                target = g.iloc[j]
                rows.append({**base, "Cycling_Test": f"h{h}", "horizon": h, "target_rpt": int(target["rpt_number"]),
                             "target_pos": j, "target_date": target["rpt_start"],
                             **interval_schedule(blocks, bat, origin["rpt_start"], target["rpt_start"]),
                             "soh_target": soh[j], TARGET: soh[j] - soh[i]})
    out = pd.DataFrame(rows).merge(static, on=dp.GROUP_COL, how="left", validate="many_to_one")
    return out.sort_values([dp.GROUP_COL, "Cycle_Date", "horizon"], kind="mergesort").reset_index(drop=True)


def feature_sets(samples: pd.DataFrame) -> dict[str, list[str]]:
    """The pre-registered feature sets F0 (history, static, schedule) and F1 (F0 + diagnostics)."""
    f0 = [*HISTORY_FEATURES, *STATIC_FEATURES, *SCHEDULE_FEATURES]
    diag = [c for c in F.feature_columns(samples) if c in samples.columns]
    return {"F0": f0, "F1": f0 + diag}


# ---------------------------------------------------------------------------
# Purged splits and task streams
# ---------------------------------------------------------------------------

def split_cuts(n_valid: int, fracs: tuple[float, float] = SPLIT_FRACS) -> tuple[int, int]:
    """Positions (in the battery's valid-RPT sequence) of cuts A and B."""
    a = int(np.floor(fracs[0] * (n_valid - 1)))
    b = max(int(np.floor(fracs[1] * (n_valid - 1))), a + 1)
    return a, b


def assign_split(samples: pd.DataFrame, fracs: tuple[float, float] = SPLIT_FRACS) -> pd.DataFrame:
    """Add ``split``: train (target <= A), val (origin >= A, target <= B), test (origin >= B); else dropped.

    Samples that straddle a cut (e.g. origin before A, target after A) belong
    to no split: using them would put a label after a later split's origin.
    """
    out = samples.copy()
    cuts = out["n_valid_rpts"].map(lambda n: split_cuts(int(n), fracs))
    a, b = cuts.str[0], cuts.str[1]
    out["split"] = np.select(
        [out["target_pos"] <= a, (out["origin_pos"] >= a) & (out["target_pos"] <= b), out["origin_pos"] >= b],
        ["train", "val", "test"], default="none")
    return out


def make_forecast_task(samples: pd.DataFrame, battery: str, fracs: tuple[float, float] = SPLIT_FRACS) -> P.Task:
    s = assign_split(samples[samples[dp.GROUP_COL] == battery], fracs)
    parts = {k: s[s["split"] == k].drop(columns="split").reset_index(drop=True) for k in ("train", "val", "test")}
    empty = [k for k, v in parts.items() if v.empty]
    if empty:
        raise ValueError(f"battery {battery!r} has empty {'/'.join(empty)} split")
    return P.Task(name=battery, **parts)


def forecast_stream(samples: pd.DataFrame, order: Sequence[str], fracs: tuple[float, float] = SPLIT_FRACS) -> list[P.Task]:
    """Task stream for protocol P1 (``stream_builder`` for ``run_cl_experiment``)."""
    return [make_forecast_task(samples, str(b), fracs) for b in order]


def leave_one_out_task(samples: pd.DataFrame, held_out: str, fracs: tuple[float, float] = SPLIT_FRACS) -> P.Task:
    """Protocol P2: fit on the other batteries (their train and test splits), early-stop on their
    validation splits, test on every sample of ``held_out``."""
    others = assign_split(samples[samples[dp.GROUP_COL] != held_out], fracs)
    fit = others[others["split"].isin(["train", "test"])].drop(columns="split")
    val = others[others["split"] == "val"].drop(columns="split")
    test = samples[samples[dp.GROUP_COL] == held_out]
    return P.Task(name=held_out, train=fit.reset_index(drop=True), val=val.reset_index(drop=True),
                  test=test.reset_index(drop=True))


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def _ridge(features: Sequence[str]):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         RidgeCV(alphas=np.logspace(-3, 3, 13)))


def _gbm(seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=5,
                                         l2_regularization=1.0, random_state=seed)


#: Columns the baselines read (S1 defaults). ``rate`` x ``span`` is the linear
#: extrapolation; ``schedule`` feeds the known-usage ridge.
BASELINE_COLUMNS: dict = {"rate": "fade_rate_pp_per_day", "span": "days_to_target", "schedule": SCHEDULE_FEATURES}


def baseline_predictions(train: pd.DataFrame, tests: Mapping[str, pd.DataFrame], sets: Mapping[str, Sequence[str]],
                         seed: int = 0, columns: Mapping | None = None) -> dict[str, dict[str, np.ndarray]]:
    """Predictions of every D18 baseline for each frame in ``tests`` (keys: baseline name, then test name).

    Learned baselines are fitted on ``train`` only (training plus validation
    rows are both acceptable: they have no early stopping). ``columns``
    overrides ``BASELINE_COLUMNS`` for another dataset (e.g. cycles instead of days).
    """
    cols_ = {**BASELINE_COLUMNS, **(columns or {})}
    y = train[TARGET].to_numpy(float)
    out: dict[str, dict[str, np.ndarray]] = {
        "persistence": {k: np.zeros(len(v)) for k, v in tests.items()},
        "linear_extrapolation": {k: np.nan_to_num(v[cols_["rate"]].to_numpy(float) * v[cols_["span"]].to_numpy(float))
                                 for k, v in tests.items()},
    }
    learned = {"ridge_schedule": (list(cols_["schedule"]), _ridge), "ridge_F0": (list(sets["F0"]), _ridge),
               "ridge_F1": (list(sets["F1"]), _ridge), "gbm_F1": (list(sets["F1"]), lambda f: _gbm(seed))}
    for name, (cols, factory) in learned.items():
        model = factory(cols).fit(train[cols], y)
        out[name] = {k: model.predict(v[cols]) for k, v in tests.items()}
    return out


def forecast_errors(frame: pd.DataFrame, pred: np.ndarray) -> dict:
    """RMSE and MAE of SOH(k+h) in pp, pooled and per horizon."""
    err = np.asarray(pred, float) - frame[TARGET].to_numpy(float)
    out = {"n": int(len(err)), "RMSE": float(np.sqrt(np.mean(err ** 2))), "MAE": float(np.mean(np.abs(err)))}
    for h in sorted(frame["horizon"].unique()):
        m = frame["horizon"].to_numpy() == h
        out[f"RMSE_h{h}"] = float(np.sqrt(np.mean(err[m] ** 2)))
    if {"soh_origin", "soh_target"} <= set(frame.columns):
        # Reuse class (D10 cut-points) of the forecast vs the measured SOH at the target (D20, secondary).
        from sklearn.metrics import f1_score

        from src import evaluation as ev

        true_c = ev.reuse_class(frame["soh_target"].to_numpy(float))
        pred_c = ev.reuse_class(frame["soh_origin"].to_numpy(float) + np.asarray(pred, float))
        out["class_accuracy"] = float(np.mean(true_c == pred_c))
        out["class_macro_f1"] = float(f1_score(true_c, pred_c, average="macro", labels=np.unique(true_c)))
    return out


# ---------------------------------------------------------------------------
# Experiment runners (protocols P1 and P2 of D18)
# ---------------------------------------------------------------------------

CL_METHODS: tuple[str, ...] = ("finetune", "replay", "ewc", "tft_finetune", "tft_ewc")
TFT_SETTINGS: dict = {"window": 4, "window_group": ("Battery_ID", "Cycling_Test")}


def cl_config(features: Sequence[str], methods: Sequence[str] = CL_METHODS, n_orders: int = 10,
              seeds: Sequence[int] = (0, 1, 2, 3, 4)):
    """The D18 continual-learning configuration (headline: feature set F1)."""
    from src.experiments import CLConfig

    return CLConfig(methods=tuple(methods), n_orders=n_orders, seeds=tuple(seeds), metric="RMSE",
                    common_kwargs={"target": TARGET, "features": tuple(features)},
                    backbone_kwargs={"tft": dict(TFT_SETTINGS)}, joint_backbones=("mlp", "tft"), reuse_metrics=False)


def _group_order(samples: pd.DataFrame) -> list[str]:
    return dp.groups_in(samples)


def run_p1(samples: pd.DataFrame, config, label: str | None = None) -> dict:
    """Protocol P1: every method over every (order, seed); per-run CL metrics and final-step errors per horizon."""
    from src import experiments as X

    orders = P.sample_orders(_group_order(samples), config.n_orders, config.order_seed)
    runs, per_task, matrices = [], [], {}
    for order_id, order in enumerate(orders):
        tasks = forecast_stream(samples, order)
        for seed in config.seeds:
            for method in config.methods:
                out = X.run_stream(method, tasks, seed, config.metric, collect_predictions=True,
                                   **X.method_settings(config, method))
                name = f"{method}{label or ''}"
                matrices[(name, order_id, seed)] = out["R"]
                runs.append({"method": name, "order_id": order_id, "order": ">".join(order), "seed": seed,
                             **{k: out["metrics"][k] for k in X.CL_METRIC_COLS if k in out["metrics"]},
                             "train_time_s": float(np.sum(out["train_time_s"]))})
                last = len(tasks) - 1
                for t in tasks:
                    per_task.append({"method": name, "order_id": order_id, "seed": seed, dp.GROUP_COL: t.name,
                                     **forecast_errors(out["eval_frames"][t.name], out["predictions"][(last, t.name)])})
    return {"runs": pd.DataFrame(runs), "final_per_task": pd.DataFrame(per_task), "matrices": matrices, "orders": orders}


def run_joint(samples: pd.DataFrame, config, label: str | None = None) -> pd.DataFrame:
    """Joint-training upper bounds (MLP and TFT) on the P1 splits, per seed and battery."""
    from src.continual_learning import Joint, TFTJoint
    from src.experiments import method_settings

    tasks = forecast_stream(samples, _group_order(samples))
    rows = []
    for backbone, cls, name in (("mlp", Joint, "joint"), ("tft", TFTJoint, "tft_joint")):
        kwargs = method_settings(config, "tft_joint" if backbone == "tft" else "joint")
        features = list(kwargs.pop("features"))
        extra = {"history": samples} if backbone == "tft" else {}
        for seed in config.seeds:
            m = cls(features, seed=seed, **extra, **kwargs).fit_all(tasks)
            for t in tasks:
                rows.append({"method": f"{name}{label or ''}", "seed": seed, dp.GROUP_COL: t.name,
                             **forecast_errors(t.test, m.predict(t.test))})
    return pd.DataFrame(rows)


def run_baselines_p1(samples: pd.DataFrame, sets: Mapping[str, Sequence[str]], columns: Mapping | None = None) -> pd.DataFrame:
    """Baselines on the P1 test splits, fitted on all batteries' training and validation splits."""
    tasks = forecast_stream(samples, _group_order(samples))
    train = pd.concat([f for t in tasks for f in (t.train, t.val)], ignore_index=True)
    preds = baseline_predictions(train, {t.name: t.test for t in tasks}, sets, columns=columns)
    return pd.DataFrame([{"method": name, dp.GROUP_COL: t.name, **forecast_errors(t.test, p[t.name])}
                         for name, p in preds.items() for t in tasks])


def run_p2(samples: pd.DataFrame, config, sets: Mapping[str, Sequence[str]], columns: Mapping | None = None) -> pd.DataFrame:
    """Protocol P2 (leave one battery out): baselines and joint-trained MLP / TFT per held-out battery.

    With a single training task, TFT + EWC reduces to the TFT, so the neural
    rows are the two backbones.
    """
    from src.continual_learning import Joint, TFTJoint
    from src.experiments import method_settings

    rows = []
    for held in _group_order(samples):
        task = leave_one_out_task(samples, held)
        fit = pd.concat([task.train, task.val], ignore_index=True)
        for name, p in baseline_predictions(fit, {"t": task.test}, sets, columns=columns).items():
            rows.append({"method": name, "seed": 0, dp.GROUP_COL: held, **forecast_errors(task.test, p["t"])})
        for backbone, cls, name in (("mlp", Joint, "mlp"), ("tft", TFTJoint, "tft")):
            kwargs = method_settings(config, "tft_joint" if backbone == "tft" else "joint")
            features = list(kwargs.pop("features"))
            extra = {"history": samples} if backbone == "tft" else {}
            for seed in config.seeds:
                m = cls(features, seed=seed, **extra, **kwargs).fit_all([task])
                rows.append({"method": name, "seed": seed, dp.GROUP_COL: held, **forecast_errors(task.test, m.predict(task.test))})
    return pd.DataFrame(rows)
