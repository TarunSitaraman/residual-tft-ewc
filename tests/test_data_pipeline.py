"""Smoke tests for the existing pipeline on synthetic test data."""

from src import data_pipeline as dp


def test_synthetic_data_passes_validation(synthetic_frames, ml_train, health):
    final_ml = synthetic_frames["final_ml_dataset.csv"]
    assert dp.check_battery_groups(ml_train[dp.GROUP_COL])["ok"]
    assert dp.validate_soh_formula(health)["ok"]
    assert dp.validate_reference_capacity(health)["ok"].all()
    summary, _ = dp.validate_rpt_alignment(final_ml, health)
    assert summary["ok"]
    assert dp.validate_training_subset(ml_train, final_ml, health)["ok"].all()


def test_model_features_exclude_labels(ml_train):
    features = dp.model_feature_columns(ml_train)
    assert len(features) == 18
    assert not set(features) & set(dp.HEALTH_COLUMNS)


def test_check_data_files_on_synthetic_root(synthetic_data_root):
    inventory = dp.check_data_files(synthetic_data_root, required=["ml_training_dataset.csv"])
    assert inventory.set_index("file").loc["ml_training_dataset.csv", "present"]
