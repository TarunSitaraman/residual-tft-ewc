"""Descriptive interpretability of the TFT trained from scratch on the S6 stream (D21 inputs, tuned learning rate).

The TFT's variable-selection weights and attention over window lags are averaged over the S6 test periods
after the full stream (first evaluation order, seeds 0-4). This is a descriptive analysis (the plan's
interpretability criterion), not a test; it uses the P3 tuning results (learning rate) and changes nothing
in P3.

Usage (from the repository root)::

    python scripts/d21_tft_interpretation.py   # -> results/d21_tft_interpretation.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_pipeline as dp  # noqa: E402
from src import experiments as X  # noqa: E402
from src import forecasting as FC  # noqa: E402
from src import p3  # noqa: E402
from src import protocols as P  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)


def main() -> int:
    tuned = json.loads((ROOT / "results" / "forecasting_d21_p3.json").read_text())["tuning"]["tft"]
    wc = dp.read_dataset(ROOT / "data" / "wheeler_checks.csv")
    wc["check_time"] = pd.to_datetime(wc["check_time"])
    s6 = p3.harmonised_samples(p3.s6_checks(wc), "S6")
    order = P.sample_orders(sorted(s6["Battery_ID"].unique()), 1, 0)[0]
    tasks = FC.forecast_stream(s6, order)
    test = pd.concat([t.test for t in tasks], ignore_index=True)
    weights, lags = [], []
    for seed in SEEDS:
        s = p3.build_strategy("tft_finetune", list(p3.SHARED_FEATURES), seed, s6,
                              {"lr": tuned["lr"], "epochs": 200, "window": 4, "window_group": ("Battery_ID", "Cycling_Test")})
        for t in tasks:
            s.fit_task(t)
        info = s.interpret(test)
        weights.append(info["variable_weights"].rename(seed))
        lags.append(info["attention_by_lag"].rename(seed))
        print(f"seed {seed}: top inputs {info['variable_weights'].head(3).round(3).to_dict()}", flush=True)
    w, a = pd.concat(weights, axis=1), pd.concat(lags, axis=1)
    out = {"order": list(order), "seeds": list(SEEDS), "lr": tuned["lr"], "n_test_samples": len(test),
           "variable_weights": {"mean": w.mean(axis=1).sort_values(ascending=False).to_dict(),
                                "min": w.min(axis=1).to_dict(), "max": w.max(axis=1).to_dict(),
                                "rank_per_seed": w.rank(ascending=False).to_dict()},
           "attention_by_lag": {"mean": a.mean(axis=1).to_dict(), "min": a.min(axis=1).to_dict(), "max": a.max(axis=1).to_dict()}}
    X.write_results(ROOT / "results" / "d21_tft_interpretation.json", out, ROOT)
    print(pd.DataFrame({"mean": w.mean(axis=1), "min": w.min(axis=1), "max": w.max(axis=1)})
          .sort_values("mean", ascending=False).round(3).to_string())
    print(a.mean(axis=1).round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
