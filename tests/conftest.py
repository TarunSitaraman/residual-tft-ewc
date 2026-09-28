"""Shared pytest fixtures.

SYNTHETIC TEST DATA ONLY. The generator below exists to exercise code paths.
It is created in pytest temporary directories, is never written into the
repository, and must never be used for reported results (see CLAUDE.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import data_pipeline as dp  # noqa: E402

FEATURES = [
    "Voltage_Mean", "Voltage_Min", "Voltage_Max",
    "Current_Mean", "Current_Min", "Current_Max",
    "Temperature_Mean", "Temperature_Min", "Temperature_Max",
    "Charge_Capacity_Max", "Discharge_Capacity_Max",
    "Charge_Energy_Max", "Discharge_Energy_Max",
    "Power_Mean", "Capacity_Ratio_Mean", "Energy_Efficiency_Mean", "Data_Points",
]

# Heterogeneous groups: different capacity, fade rate and temperature, so that
# continual-learning code sees a real distribution shift between tasks.
GROUP_PARAMS = {
    "g1": {"cap0": 4.80, "fade": 0.00040, "temp": 25.0},
    "v4": {"cap0": 4.90, "fade": 0.00055, "temp": 27.0},
    "v5": {"cap0": 4.85, "fade": 0.00050, "temp": 26.0},
    "w8": {"cap0": 3.10, "fade": 0.00070, "temp": 30.0},
    "w9": {"cap0": 3.00, "fade": 0.00080, "temp": 31.0},
    "w10": {"cap0": 3.05, "fade": 0.00065, "temp": 29.0},
}


def make_synthetic_frames(seed: int = 0, n_cycles: int = 240, n_rpt: int = 6) -> dict[str, pd.DataFrame]:
    """Build synthetic versions of the cycle-level handover files (test use only)."""
    rng = np.random.default_rng(seed)
    cycle_rows, health_rows = [], []
    for battery, p in GROUP_PARAMS.items():
        start = pd.Timestamp("2021-01-01") + pd.Timedelta(days=int(rng.integers(0, 20)))
        rpt_dates = [start + pd.Timedelta(days=10 + 40 * k) for k in range(n_rpt)]
        for k, d in enumerate(rpt_dates):
            days = (d - start).days
            cap = p["cap0"] if k == 0 else p["cap0"] * (1 - p["fade"] * (days - 10)) * (1 + rng.normal(0, 0.001))
            health_rows.append({
                "Battery_ID": battery, "RPT": f"RPT_{k + 1}", "RPT_Number": k + 1,
                "Capacity_Ah": cap, "Test_Date": d, "Reference_Capacity_Ah": p["cap0"],
                "SOH_Percent": cap / p["cap0"] * 100,
            })
        for i in range(n_cycles):
            fade = 1 - p["fade"] * i
            base = [3.6, 2.8, 4.2, 0.1, -5, 5, p["temp"], p["temp"] - 3, p["temp"] + 5,
                    p["cap0"] * fade, p["cap0"] * fade * 0.99, p["cap0"] * fade * 3.7,
                    p["cap0"] * fade * 3.5, 5, 0.99, 0.95, 3000]
            row = {"Battery_ID": battery, "Cycling_Test": f"Cycling_{i // 120 + 1}", "Cycle_Index": i + 1,
                   "Cycle_Date": start + pd.Timedelta(days=i, hours=6)}
            for f, b in zip(FEATURES, base):
                row[f] = b + 0.005 * abs(b) * rng.normal()
            row["Data_Points"] = int(row["Data_Points"])
            if rng.random() < 0.08:
                for t in dp.TEMPERATURE_COLUMNS:
                    row[t] = np.nan
            cycle_rows.append(row)

    cycles = pd.DataFrame(cycle_rows)
    health = pd.DataFrame(health_rows)
    final = pd.merge_asof(
        cycles.sort_values("Cycle_Date"), health.sort_values("Test_Date"),
        left_on="Cycle_Date", right_on="Test_Date", by="Battery_ID", direction="backward",
    )
    final["Days_Since_RPT"] = (final["Cycle_Date"] - final["Test_Date"]).dt.days
    final = final.drop(columns="Test_Date").sort_values(["Battery_ID", "Cycle_Date"]).reset_index(drop=True)
    ml_train = final.dropna(subset=["SOH_Percent"]).reset_index(drop=True)
    return {
        "final_battery_cycling_features.csv": cycles,
        "final_battery_health_dataset.csv": health,
        "rpt_capacity_data.csv": health[["Battery_ID", "RPT", "Capacity_Ah", "Test_Date"]],
        "final_ml_dataset.csv": final,
        "ml_training_dataset.csv": ml_train,
        "DATA_DICTIONARY.csv": pd.DataFrame({"Column": list(final.columns), "Meaning": "synthetic", "Unit": ""}),
    }


@pytest.fixture(scope="session")
def synthetic_frames() -> dict[str, pd.DataFrame]:
    """In-memory synthetic cycle-level frames (test use only)."""
    return make_synthetic_frames()


@pytest.fixture(scope="session")
def ml_train(synthetic_frames) -> pd.DataFrame:
    return synthetic_frames["ml_training_dataset.csv"].copy()


@pytest.fixture(scope="session")
def health(synthetic_frames) -> pd.DataFrame:
    return synthetic_frames["final_battery_health_dataset.csv"].copy()


@pytest.fixture(scope="session")
def synthetic_data_root(tmp_path_factory, synthetic_frames) -> Path:
    """Synthetic CSVs written to a pytest temporary directory (never the repository)."""
    root = tmp_path_factory.mktemp("synthetic_data")
    for name, df in synthetic_frames.items():
        df.to_csv(root / name, index=False)
    return root
