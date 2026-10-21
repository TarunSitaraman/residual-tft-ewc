"""Review-deck figures: final test RMSE on the P1 stream for S1 (P1 table) and S6 (D24, horizons 1-3).

Reads only committed results: results/tables/p1.csv, results/tables/d24_comparison_e1.csv (continual arms) and
results/forecasting_d24_baselines_e1.json (single-fit baselines in the pinned environment, as reported in the paper).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
NAVY, TEAL, ORANGE, GREY = "#13233f", "#0aa3c2", "#e8833a", "#9aa5b1"


def bar_figure(rows, title, note, path, size):
    """rows: (label, mean, sd or None, colour), drawn top to bottom."""
    fig, ax = plt.subplots(figsize=size, dpi=300)
    labels, means, sds, colours = zip(*rows)
    y = range(len(rows))[::-1]
    ax.barh(list(y), means, color=colours, height=0.62)
    for yi, m, s in zip(y, means, sds):
        if s is not None:
            ax.errorbar(m, yi, xerr=s, color=NAVY, elinewidth=1, capsize=3)
    for yi, m, s in zip(y, means, sds):
        ax.text(m + (s or 0) + 0.02, yi, f"{m:.3f}", va="center", fontsize=9, color=NAVY)
    ax.set_yticks(list(y), labels, fontsize=10)
    ax.set_xlabel("Final test RMSE (pp of SOH), lower is better", fontsize=10)
    ax.set_xlim(0, max(m + (s or 0) for m, s in zip(means, sds)) * 1.15)
    ax.set_title(title, fontsize=11, color=NAVY, loc="left")
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, note, fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def s1_rows():
    p1 = pd.read_csv(ROOT / "results/tables/p1.csv").query("dataset == 'S1'").set_index("method")
    pick = [("persistence", GREY), ("ridge_schedule", TEAL), ("gbm_F1", GREY), ("tft_ewc", ORANGE),
            ("tft_joint", GREY)]
    return [(p1.at[m, "name"], p1.at[m, "rmse_pp_mean"],
             None if pd.isna(p1.at[m, "rmse_pp_sd"]) else p1.at[m, "rmse_pp_sd"], c) for m, c in pick]


def s6_rows():
    cont = pd.read_csv(ROOT / "results/tables/d24_comparison_e1.csv").set_index("method")
    base = {b["method"]: b["final_avg_error"]
            for b in json.loads((ROOT / "results/forecasting_d24_baselines_e1.json").read_text())["baselines"]}
    rows = [("Persistence", base["persistence"], None, GREY),
            ("Ridge", base["ridge_shared"], None, GREY),
            ("Random forest", base["rf_direct"], None, GREY),
            ("Gradient boosting", base["gbm_shared"], None, GREY),
            ("SVR (residual)", base["svr_residual"], None, GREY)]
    for m, label, c in (("tft_direct_ewc", "TFT + EWC (direct output)", "#f3c29b"),
                        ("mlp_residual_ewc", "MLP + EWC (residual)", GREY),
                        ("transformer_residual_ewc", "Transformer + EWC (residual)", GREY),
                        ("lstm_direct_ewc", "BiLSTM + EWC (direct)", GREY),
                        ("tft_residual_ewc", "TFT + EWC (residual, ours)", ORANGE)):
        rows.append((label, cont.at[m, "rmse_main_mean"], cont.at[m, "rmse_main_sd"], c))
    rows.append(("Linear extrapolation", base["linear_extrapolation"], None, TEAL))
    return sorted(rows, key=lambda r: -r[1])


if __name__ == "__main__":
    bar_figure(s1_rows(), "S1: continual stream (P1), horizons 1-3",
               "Mean ± s.d. over 50 runs (joint: 5). No bar: deterministic or single fit. Source: P1 results.", OUT / "deck_s1_p1.png", (5.6, 4.6))
    bar_figure(s6_rows(), "S6: continual stream (P1), horizons 1-3",
               "Mean over 50 runs, main seeds (± s.d.). Single-fit baselines: pinned environment. Source: D24 results.",
               OUT / "deck_s6_p1.png", (5.8, 4.8))
    print("wrote", OUT / "deck_s1_p1.png", OUT / "deck_s6_p1.png")
