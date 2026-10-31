"""Paper figure of the D23 comparison (notebook 14's bar chart, current labels, two-column width).

Final average test RMSE of every D23 method on the S6 stream, main seed set (mean and s.d. over 50 runs),
for horizons 1-3 (E1) and 1-6 (E2), from results/forecasting_d23_e{1,2}.json. Writes
paper/figures/d23_comparison.png.

    python scripts/make_d23_bar_figure.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("d24_outputs", ROOT / "scripts" / "make_d24_outputs.py")
O = importlib.util.module_from_spec(spec)
spec.loader.exec_module(O)

REF = "tft_residual_ewc"
PROPOSED = "Residual TFT-EWC (proposed)"


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.15))
    for ax, exp in zip(axes, ("e1", "e2")):
        data = pd.read_json(ROOT / "results" / f"forecasting_d23_{exp}.json", typ="series").to_dict()
        c = O.comparison(data).sort_values("rmse_main_mean", ascending=False)
        c.loc[c["method"] == REF, "label"] = PROPOSED
        colours = ["#2a78d6" if m == REF else O.FAMILY_COLOUR.get(O.family(m), O.GREY) if O.family(m) else O.GREY
                   for m in c["method"]]
        bars = ax.barh(c["label"], c["rmse_main_mean"], xerr=c["rmse_main_sd"].fillna(0), color=colours,
                       height=0.65, error_kw={"elinewidth": 0.6, "ecolor": O.INK2})
        for b, m in zip(bars, c["method"]):
            b.set_alpha(1.0 if m == REF else 0.55)
        lin = c.loc[c["method"] == "linear_extrapolation", "rmse_main_mean"].iloc[0]
        ax.axvline(lin, color=O.INK, linestyle="--", linewidth=0.8)
        h = data["horizons"]
        ax.set_title(f"{exp.upper()}: horizons {h[0]}-{h[-1]}", fontsize=8)
        ax.set_xlabel("final average test RMSE (pp of SOH)", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.grid(axis="x", alpha=0.3)
        for lbl in ax.get_yticklabels():
            if lbl.get_text() == PROPOSED:
                lbl.set_fontweight("bold")
    fig.tight_layout(pad=0.4)
    fig.savefig(ROOT / "paper" / "figures" / "d23_comparison.png", dpi=300)
    print("wrote paper/figures/d23_comparison.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
