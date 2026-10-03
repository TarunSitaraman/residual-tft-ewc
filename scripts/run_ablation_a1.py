"""D21 addendum A1: attribute the D20 -> D21 change of the TFT from scratch (inputs x learning rate).

Usage (from the repository root)::

    python scripts/run_ablation_a1.py                   # -> results/forecasting_d21_ablation_a1.json
    python scripts/run_ablation_a1.py --profile smoke   # code check only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_pipeline as dp  # noqa: E402
from src import evaluation as ev  # noqa: E402
from src import experiments as X  # noqa: E402
from src import forecasting as FC  # noqa: E402
from src import p3  # noqa: E402
from src import protocols as P  # noqa: E402
from src import wheeler as W  # noqa: E402

TFT = {"window": 4, "window_group": ("Battery_ID", "Cycling_Test")}
LRS = (1e-2, 3e-3)
CALENDAR = ("days_since_first", "days_to_target")
PROFILES = {"paper": {"n_orders": 10, "seeds": (0, 1, 2, 3, 4), "epochs": 200},
            "smoke": {"n_orders": 1, "seeds": (0,), "epochs": 2}}


def samples() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    wc = dp.read_dataset(ROOT / "data" / "wheeler_checks.csv")
    wc["check_time"] = pd.to_datetime(wc["check_time"])
    h = p3.harmonised_samples(p3.s6_checks(wc), "S6")
    old = W.build_samples(wc)
    sets20 = W.feature_sets(old)
    h["origin_cycle"] = h["Cycle_Date"].map(dict(zip(wc["check_time"], wc["cycle"]))).astype(int)
    extra = [c for c in sets20["F1"] if c not in h.columns]
    h = h.merge(old[["cell", "origin_cycle", "horizon", *extra]], on=["cell", "origin_cycle", "horizon"],
                how="left", validate="one_to_one")
    shared = list(p3.SHARED_FEATURES)
    inputs = {"D20_F1": list(sets20["F1"]), "D20_F0": list(sets20["F0"]), "D21_shared": shared,
              "D21_shared_no_calendar": [f for f in shared if f not in CALENDAR]}
    return h, inputs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", choices=tuple(PROFILES), default="paper")
    args = ap.parse_args()
    cfg = PROFILES[args.profile]
    s6, inputs = samples()
    cases = sorted(s6["Battery_ID"].unique())
    orders = P.sample_orders(cases, cfg["n_orders"], 0)
    ckpt = FC.Checkpoint(ROOT / "data" / "cache" / "ablation_a1", [FC.Checkpoint.frame_hash(s6), args.profile])
    rows = []
    for name, feats in inputs.items():
        for lr in LRS:
            for seed in cfg["seeds"]:
                for oid, order in enumerate(orders):
                    def fit(feats=feats, lr=lr, seed=seed, order=order):
                        s = p3.build_strategy("tft_finetune", feats, seed, s6, {"lr": lr, "epochs": cfg["epochs"], **TFT})
                        out = p3.run_stream(s, FC.forecast_stream(s6, order), None)
                        return out["metrics"]
                    m = ckpt(f"{name}_lr{lr}_s{seed}_o{oid}", fit, {"features": feats, "lr": lr})
                    rows.append({"inputs": name, "lr": lr, "seed": seed, "order_id": oid, **m})
            print(f"done {name} (both learning rates)", flush=True)
    runs = pd.DataFrame(rows)
    summary = runs.groupby(["inputs", "lr"])["final_avg_error"].agg(["mean", "std", "median"]).reset_index()
    pv, tests = {}, []
    key = ["seed", "order_id"]
    pick = lambda i, l: runs[(runs["inputs"] == i) & (runs["lr"] == l)].set_index(key)["final_avg_error"]
    contrasts = [(("D21_shared", lr), (other, lr)) for lr in LRS for other in ("D20_F1", "D20_F0", "D21_shared_no_calendar")]
    contrasts += [(("D20_F0", lr), ("D20_F1", lr)) for lr in LRS]
    contrasts += [((i, 3e-3), (i, 1e-2)) for i in inputs]
    for a, b in contrasts:
        x, y = pick(*a), pick(*b).reindex(pick(*a).index)
        res = ev.paired_comparison(x.to_numpy(float), y.to_numpy(float))
        k = f"{a[0]}@{a[1]} - {b[0]}@{b[1]}"
        pv[k] = res["p_value"]
        tests.append({"contrast": k, "median_diff": float(np.median(x - y)), **res})
    holm = ev.holm_correction(pv)
    tests = [{**t, "p_holm": holm[t["contrast"]]} for t in tests]
    name = "forecasting_d21_ablation_a1.json" if args.profile == "paper" else "forecasting_d21_ablation_a1_smoke.json"
    X.write_results(ROOT / "results" / name, {"profile": args.profile, "decision": "D21 addendum A1",
                                              "inputs": inputs, "learning_rates": list(LRS),
                                              "orders": [list(o) for o in orders], "runs": runs.to_dict("records"),
                                              "summary": summary.to_dict("records"), "tests": tests}, ROOT)
    print(summary.round(3).to_string(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
