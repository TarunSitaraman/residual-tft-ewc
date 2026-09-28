"""Runner tests on synthetic test data (small epochs; code paths only)."""

import json

import numpy as np
import pytest

from src import data_pipeline as dp
from src import experiments as X

FAST = {"epochs": 5, "patience": 3}


def test_reference_baselines_structure(ml_train, health):
    res, details = X.reference_baselines(ml_train, health, models=("ridge",), mlp_kwargs=FAST)
    assert set(res["experiment"]) == {"E1_within_group", "E2_rpt_boundary", "E3_leave_one_group_out", "E5_date_only_end"}
    assert set(res[dp.GROUP_COL]) == set(dp.BATTERY_GROUPS)
    assert res.groupby("experiment").size().eq(len(dp.BATTERY_GROUPS)).all()
    assert np.isfinite(res["RMSE"]).all()
    assert details["E5_labels_changed"] > 0


def test_fit_predict_rejects_unknown_model(ml_train):
    with pytest.raises(ValueError):
        X.fit_predict("svm", ml_train, ml_train, {"x": ml_train}, dp.model_feature_columns(ml_train))


def test_run_cl_experiment_grid_and_comparison(ml_train):
    cfg = X.CLConfig(methods=("finetune", "replay"), n_orders=2, seeds=(0,), common_kwargs=FAST)
    out = X.run_cl_experiment(ml_train, cfg)
    runs = out["runs"]
    assert len(runs) == 2 * 1 * 2
    assert len(out["matrices"]) == 4 and len(out["orders"]) == 2
    R = out["matrices"][("finetune", 0, 0)]
    assert R.shape == (6, 6) and np.isfinite(R.to_numpy()).all()  # every task evaluated after every step
    assert set(out["joint"][dp.GROUP_COL]) == set(dp.BATTERY_GROUPS)
    comp = X.compare_methods(runs)
    assert set(comp["method"]) == {"replay"} and "p_holm" in comp


def test_run_stream_is_deterministic(ml_train):
    from src import protocols as P

    tasks = P.make_task_stream(ml_train, ["g1", "w8"])
    a = X.run_stream("finetune", tasks, seed=1, **FAST)
    b = X.run_stream("finetune", tasks, seed=1, **FAST)
    assert a["R"].equals(b["R"])


def test_write_results_adds_provenance(tmp_path, ml_train):
    path = X.write_results(tmp_path / "r.json", {"table": ml_train.head(2), "value": np.float64("nan")})
    report = json.loads(path.read_text())
    assert "environment" in report and "git_commit" in report["environment"]
    assert report["value"] is None and len(report["table"]) == 2


# -- hyperparameter tuning (D11) ---------------------------------------------

def test_tuning_orders_disjoint_from_evaluation(ml_train):
    from src import protocols as P

    cfg = X.CLConfig(n_orders=4)
    evaluation = set(P.sample_orders(list(dp.BATTERY_GROUPS), 4, cfg.order_seed))
    tuning = X.tuning_orders(ml_train, cfg, 3)
    assert len(tuning) == 3 and not set(tuning) & evaluation


def test_validation_evaluation_never_reads_test(ml_train):
    from src import protocols as P

    tasks = [P.Task(t.name, t.train, t.val, t.test.iloc[0:0]) for t in P.make_task_stream(ml_train, ["g1", "w8"])]
    out = X.run_stream("finetune", tasks, seed=0, eval_split="val", **FAST)
    assert out["R"].notna().all().all()
    with pytest.raises(ValueError):
        X.run_stream("finetune", tasks, seed=0, eval_split="train", **FAST)


def test_tune_hyperparameters_selects_from_grid(ml_train):
    cfg = X.CLConfig(methods=("finetune", "replay"), n_orders=1, seeds=(0,), common_kwargs=FAST)
    tuning = X.TuningConfig(grids={"replay": {"buffer_per_task": [20, 80]}}, backbone_grid={"lr": [1e-3, 3e-3]}, n_orders=1)
    out = X.tune_hyperparameters(ml_train, cfg, tuning)
    chosen = out["selected"]["replay"]["buffer_per_task"]
    assert chosen in (20, 80) and isinstance(chosen, int)
    assert out["at_grid_edge"]["replay"]["buffer_per_task"] is True
    assert out["tuned_cl_config"].method_kwargs["replay"]["buffer_per_task"] == chosen
    assert "finetune" not in out["selected"]  # no method-specific grid
    # Stage 1: one backbone (MLP) tuned by fine-tuning; its lr reaches every MLP method.
    lr = out["backbone_selected"]["mlp"]["lr"]
    assert lr in (1e-3, 3e-3) and set(out["backbone_selected"]) == {"mlp"}
    assert X.method_settings(out["tuned_cl_config"], "replay")["lr"] == lr
    assert out["grid_summary"]["selected"].sum() == 2  # one per stage
    assert set(out["grid_summary"]["stage"]) == {"backbone:mlp", "method"}


def test_backbone_settings_are_per_backbone():
    cfg = X.CLConfig(common_kwargs={"lr": 1e-3, "epochs": 5}, backbone_kwargs={"tft": {"lr": 1e-2}},
                     method_kwargs={"tft_ewc": {"ewc_lambda": 10.0}})
    assert X.method_settings(cfg, "ewc") == {"lr": 1e-3, "epochs": 5}
    assert X.method_settings(cfg, "tft_ewc") == {"lr": 1e-2, "epochs": 5, "ewc_lambda": 10.0}
    assert X.backbone_of("tft_finetune") == "tft" and X.backbone_of("lwf") == "mlp"


def test_fit_predict_neural_selects_lr_on_validation(ml_train):
    from src import protocols as P

    task = P.make_task(ml_train, "g1")
    for model in ("mlp", "tft"):
        preds, cfg = X.fit_predict(model, task.train, task.val, {"g1": task.test}, dp.model_feature_columns(ml_train),
                                   mlp_kwargs=FAST, history=ml_train, lr_grid=[1e-3, 1e-2])
        assert cfg["lr"] in (1e-3, 1e-2) and set(cfg["lr_val_rmse"]) == {1e-3, 1e-2}
        assert cfg["lr"] == min(cfg["lr_val_rmse"], key=cfg["lr_val_rmse"].get)
        assert cfg["lr_at_grid_edge"] is True and np.isfinite(preds["g1"]).all()
