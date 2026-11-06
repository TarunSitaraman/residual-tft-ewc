"""Main-text figure of the many-condition tests (D34 on S7, D36 on S10), horizons 1-3, main seed set.

Each point is one held-out usage condition: the proposed model's RMSE (shrunk everywhere on S7, shrunk only at H+
origins on S10, as registered) against (a) gradient boosting refitted on the latest training condition and
(b) gradient boosting with every training condition stored. Points below the diagonal favour the proposed model;
the legend gives how many conditions do. Writes paper/figures/many_conditions.png.

    python scripts/make_many_conditions_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = (("S7", "d34", "tft_residual_ewc_shrunk", "#eb6834", "s"),
        ("S10", "d36", "tft_residual_ewc_selective", "#2a78d6", "o"))
PANELS = (("gbm_last", "(a) vs. GB, latest only"), ("gbm_all", "(b) vs. GB, all stored"))
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e6e5e0"


def per_condition(dec: str, prop: str) -> pd.DataFrame:
    d = json.loads((ROOT / "results" / f"forecasting_{dec}_e1.json").read_text())
    cv = pd.DataFrame(d["condition_values"])
    cv = cv[cv["set"] == "main"]
    return cv.pivot_table(index="condition", columns="method", values="rmse")[[prop, "gbm_last", "gbm_all"]]


def main() -> int:
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 7, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.65), sharey=True)
    data = {ds: (per_condition(dec, prop), prop, c, m) for ds, dec, prop, c, m in RUNS}
    lo, hi = 0.15, 25
    for ax, (comp, title) in zip(axes, PANELS):
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
        for ds, (t, prop, c, m) in data.items():
            wins = int((t[prop] < t[comp]).sum())
            ax.scatter(t[comp], t[prop], s=7, color=c, marker=m, alpha=0.75, lw=0, zorder=2,
                       label=f"{ds}: better on {wins}/{len(t)}")
            print(ds, comp, f"proposed better on {wins}/{len(t)}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.grid(color=GRID, lw=0.5)
        ax.set_title(title, fontsize=6.6, loc="left", color=INK)
        ax.set_xlabel("comparator RMSE (pp)")
        ax.legend(fontsize=5.6, frameon=False, loc="upper left", handletextpad=0.1, borderaxespad=0.1)
    axes[0].set_ylabel("proposed RMSE (pp)")
    path = ROOT / "paper" / "figures" / "many_conditions.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"written {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
