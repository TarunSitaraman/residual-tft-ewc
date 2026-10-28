"""D30 metric set: hand-worked cases and agreement with scikit-learn (synthetic inputs only)."""

import numpy as np
import pandas as pd
import pytest
from sklearn import metrics as skm

from src import evaluation as ev
from src import forecasting as FC
from src import metrics as M


def _soh(seed=0, n=400):
    rng = np.random.default_rng(seed)
    origin = rng.uniform(70, 101, n)
    true = origin - rng.uniform(0, 6, n)
    pred = true + rng.normal(0.3, 2.0, n)
    return origin, true, pred


def test_regression_hand_worked():
    m = M.regression_metrics([90, 80, 100], [91, 78, 100], y_origin=[95, 85, 100])
    assert m["MSE"] == pytest.approx(5 / 3)
    assert m["RMSE"] == pytest.approx(np.sqrt(5 / 3))
    assert m["MAE"] == pytest.approx(1.0)
    assert m["MedAE"] == pytest.approx(1.0)
    assert m["Bias"] == pytest.approx(-1 / 3)                      # pred - true: (1 - 2 + 0) / 3
    assert m["MAPE"] == pytest.approx((1 / 90 + 2 / 80) / 3 * 100)
    assert m["NRMSE"] == pytest.approx(np.sqrt(5 / 3) / 20)
    assert m["R2"] == pytest.approx(1 - 5 / 200)
    # Change: true (-5, -5, 0), pred (-4, -7, 0); mean -10/3.
    ss_tot = (5 / 3) ** 2 * 2 + (10 / 3) ** 2
    assert m["R2_change"] == pytest.approx(1 - 5 / ss_tot)


def test_regression_matches_sklearn():
    origin, y, p = _soh()
    m = M.regression_metrics(y, p, origin)
    assert m["MSE"] == pytest.approx(skm.mean_squared_error(y, p))
    assert m["MAE"] == pytest.approx(skm.mean_absolute_error(y, p))
    assert m["MedAE"] == pytest.approx(skm.median_absolute_error(y, p))
    assert m["MAPE"] == pytest.approx(skm.mean_absolute_percentage_error(y, p) * 100)
    assert m["R2"] == pytest.approx(skm.r2_score(y, p))
    assert m["R2_change"] == pytest.approx(skm.r2_score(y - origin, p - origin))


def test_regression_undefined_is_nan():
    m = M.regression_metrics([90, 90], [91, 89])
    assert np.isnan(m["R2"]) and np.isnan(m["NRMSE"]) and np.isnan(m["R2_change"])
    assert np.isnan(M.regression_metrics([0.0, 1.0], [0.0, 1.0])["MAPE"])
    assert M.regression_metrics([], [])["n"] == 0


def test_persistence_has_zero_change_r2_but_high_level_r2():
    origin, y, _ = _soh()
    m = M.regression_metrics(y, origin, origin)                    # predict no change
    assert m["R2"] > 0.8
    assert m["R2_change"] < 0.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_classification_matches_sklearn(seed):
    _, y, p = _soh(seed)
    t, q = ev.reuse_class(y), ev.reuse_class(p)
    m = M.classification_metrics(y, p)
    present = np.unique(t)
    assert m["accuracy"] == pytest.approx(skm.accuracy_score(t, q))
    assert m["balanced_accuracy"] == pytest.approx(skm.balanced_accuracy_score(t, q))
    assert m["precision_macro"] == pytest.approx(skm.precision_score(t, q, labels=present, average="macro", zero_division=0))
    assert m["recall_macro"] == pytest.approx(skm.recall_score(t, q, labels=present, average="macro", zero_division=0))
    assert m["f1_macro"] == pytest.approx(skm.f1_score(t, q, labels=present, average="macro", zero_division=0))
    assert m["mcc"] == pytest.approx(skm.matthews_corrcoef(t, q))
    assert m["kappa"] == pytest.approx(skm.cohen_kappa_score(t, q, labels=list(range(4))))
    assert m["kappa_linear"] == pytest.approx(skm.cohen_kappa_score(t, q, labels=list(range(4)), weights="linear"))
    np.testing.assert_array_equal(m["confusion"], skm.confusion_matrix(t, q, labels=list(range(4))))
    # Specificity of class c = recall of "not c".
    for c in present:
        assert m["per_class"][int(c)]["specificity"] == pytest.approx(skm.recall_score(t != c, q != c))


def test_classification_single_class_and_unpredicted_class():
    m = M.classification_metrics([99, 98, 97], [99, 98, 97])          # one class, perfectly predicted
    assert m["accuracy"] == 1.0 and m["n_classes_measured"] == 1
    assert np.isnan(m["mcc"]) and np.isnan(m["kappa"])
    m = M.classification_metrics([99, 99, 85, 85], [99, 99, 99, 99])  # C2 never predicted
    assert np.isnan(m["per_class"][2]["precision"]) and m["per_class"][2]["recall"] == 0.0
    assert m["precision_macro"] == pytest.approx(0.25)               # (0.5 + 0) / 2: unpredicted class counts as 0
    assert m["balanced_accuracy"] == pytest.approx(0.5)


def test_pooled_confusion_equals_metrics_on_concatenation():
    _, y, p = _soh(3)
    rows = [M.flat(y[:150], p[:150]), M.flat(y[150:], p[150:])]
    pooled = M.pooled_from_rows(rows)
    direct = M.classification_metrics(y, p)
    for k in M.CLASS_KEYS:
        assert pooled[k] == pytest.approx(direct[k], nan_ok=True)


def test_binary_and_auc():
    _, y, p = _soh(4)
    m = M.binary_metrics(y, p)
    t, q = y <= M.EOL_SOH, p <= M.EOL_SOH
    assert m["precision"] == pytest.approx(skm.precision_score(t, q))
    assert m["recall"] == pytest.approx(skm.recall_score(t, q))
    assert m["specificity"] == pytest.approx(skm.recall_score(~t, ~q))
    assert m["f1"] == pytest.approx(skm.f1_score(t, q))
    assert m["mcc"] == pytest.approx(skm.matthews_corrcoef(t, q))
    assert m["auc_roc"] == pytest.approx(skm.roc_auc_score(t, -p))
    assert m["tp"] + m["fn"] == m["n_positive"] == int(t.sum())
    assert M.roc_auc([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)   # ties count half
    assert np.isnan(M.binary_metrics([90, 95], [70, 99])["auc_roc"])               # no measured positive


def test_flat_keys_are_scalars():
    origin, y, p = _soh(5, 50)
    f = M.flat(y, p, origin)
    assert all(np.isscalar(v) for v in f.values())
    assert {f"reg_{k}" for k in M.REGRESSION_KEYS} <= set(f)
    assert sum(v for k, v in f.items() if k.startswith("cls_cm_")) == 50


def test_forecast_errors_adds_d30_keys_without_changing_existing_ones():
    origin, y, p = _soh(6, 60)
    frame = pd.DataFrame({FC.TARGET: y - origin, "soh_origin": origin, "soh_target": y, "horizon": np.tile([1, 2, 3], 20)})
    out = FC.forecast_errors(frame, p - origin)
    assert out["RMSE"] == pytest.approx(np.sqrt(np.mean((p - y) ** 2)))
    assert out["reg_RMSE"] == pytest.approx(out["RMSE"])                          # errors in pp identical on Δ and SOH
    assert out["reg_MAE"] == pytest.approx(out["MAE"])
    assert out["cls_accuracy"] == pytest.approx(out["class_accuracy"])
    bare = FC.forecast_errors(frame.drop(columns=["soh_origin", "soh_target"]), p - origin)
    assert not any(k.startswith(("reg_", "cls_", "eol_")) for k in bare)


def test_d30_reproduction_tolerance_baselines_exact_neural_within_d21_threshold():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_d30
    from src import data_pipeline as dp

    committed = {"p2_rmse": [{"method": "ridge_F0", "g": 1.0}, {"method": "tft", "g": 1.0}]}
    def p2(ridge, tft):
        return pd.DataFrame({"method": ["ridge_F0", "tft"], dp.GROUP_COL: ["g", "g"], "RMSE": [ridge, tft]})
    assert run_d30.reproduction(p2(1.0, 1.04), None, committed)["passed"]
    assert not run_d30.reproduction(p2(1.0, 1.06), None, committed)["passed"]
    assert not run_d30.reproduction(p2(1.0 + 1e-5, 1.0), None, committed)["passed"]
    assert run_d30.reproduction(p2(1.0, 1.04), None, committed)["abs_diffs"]["p2/tft/g"] == pytest.approx(0.04)
