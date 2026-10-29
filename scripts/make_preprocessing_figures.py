"""Supplementary figures on S6 preprocessing and inputs, from the raw check files and data/wheeler_checks.csv.

- paper/figures/preprocessing_s6.png: (a) C/25 discharge curves of one cell at three ages; (b) incremental
  capacity -dQ/dV of one check without and with the 5 mV Gaussian smoothing used for the IC features
  (src/wheeler.py, the only noise reduction in the pipeline); (c) SOH of all 20 cells per check, with the four
  checks excluded by the validity rule marked.
- paper/figures/correlation_s6.png: Spearman correlation of the S6 F1 inputs and the target (horizon 1).
  Nothing is fitted; the figures describe the data.

    python scripts/make_preprocessing_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_pipeline as dp  # noqa: E402
from src import forecasting as FC  # noqa: E402
from src import wheeler as W  # noqa: E402

CELL = "cell1a"
C25_DISCHARGE_STEP = 16  # src/wheeler.check_features

LABELS = {
    "soh_origin": "SOH at origin", "cycles_since_first": "cycles since first", "fade_rate_pp_per_cycle": "fade rate",
    "cycles_to_target": "cycles to target", "q_3.550_3.315": "Q 3.55-3.315 V", "q_3.315_3.270": "Q 3.315-3.27 V",
    "q_3.270_2.500": "Q 3.27-2.5 V", "v_c25_dis_mid": "mid-discharge V", "hysteresis_c25_mid": "hysteresis",
    "r_c3_c25": "resistance proxy", "q_ratio_c3_c25": "Q C/3 / Q C/25", "q_c3_dis": "Q C/3", "q_c25_ch": "Q C/25 charge",
    FC.TARGET: "target: SOH change",
}


def preprocessing(table: pd.DataFrame) -> None:
    checks = W.list_checks(ROOT / "data" / W.RAW_SUBDIR)
    cell = checks[checks["cell"] == CELL].reset_index(drop=True)
    picks = cell.iloc[[0, len(cell) // 2, len(cell) - 1]]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    first = None
    for (_, row), c in zip(picks.iterrows(), ("tab:blue", "tab:orange", "tab:red")):
        d = W._step(W.read_check(row["path"]), C25_DISCHARGE_STEP, discharge=True)
        first = d if first is None else first
        ax[0].plot(d["Discharge_Capacity"], d["Voltage"], color=c, lw=1, label=f"cycle {row['cycle']}")
    ax[0].set(xlabel="discharged charge (Ah)", ylabel="voltage (V)", title=f"(a) C/25 discharge, {CELL}")
    ax[0].legend(fontsize=6.5, frameon=False)

    # Same interpolation as W.ic_curve, without the Gaussian filter.
    v = np.minimum.accumulate(first["Voltage"].to_numpy(float))
    qg = np.interp(W.IC_GRID, v[::-1], first["Discharge_Capacity"].to_numpy(float)[::-1], left=np.nan, right=np.nan)
    raw = -np.gradient(qg, W.IC_GRID)
    smooth = W.ic_curve(first)
    ax[1].plot(W.IC_GRID, raw, color="0.7", lw=0.6, label="unsmoothed")
    ax[1].plot(W.IC_GRID, smooth, color="tab:blue", lw=1.2, label=f"Gaussian, {W.IC_SMOOTH_MV:g} mV")
    ax[1].set_ylim(0, 1.05 * np.nanmax(raw[(W.IC_GRID >= 3.15) & (W.IC_GRID <= 3.40)]))
    ax[1].set_xlim(3.15, 3.40)
    ax[1].set(xlabel="voltage (V)", ylabel="-dQ/dV (Ah/V)", title="(b) incremental capacity, first check")
    ax[1].legend(fontsize=6.5, frameon=False)

    for cell_id, g in table.sort_values("cycle").groupby("cell"):
        ok = g[g["valid"]]
        ax[2].plot(ok["cycle"], ok["SOH_Percent"], lw=0.7, color="tab:blue", alpha=0.6)
    # Excluded checks have no valid capacity: marked on their cell's curve at the cycle they were taken.
    bad = table[~table["valid"]]
    ok = table[table["valid"]].sort_values("cycle")
    y = [np.interp(r["cycle"], ok.loc[ok["cell"] == r["cell"], "cycle"], ok.loc[ok["cell"] == r["cell"], "SOH_Percent"])
         for _, r in bad.iterrows()]
    ax[2].plot(bad["cycle"], y, "x", color="tab:red", ms=6, mew=1.5, label=f"excluded check ({len(bad)})")
    ax[2].axhline(80, color="k", ls=":", lw=0.8)
    ax[2].set(xlabel="cycle", ylabel="SOH (%)", title=f"(c) SOH of the {table['cell'].nunique()} cells")
    ax[2].legend(fontsize=6.5, frameon=False)
    for a in ax:
        a.grid(alpha=0.3)
        a.tick_params(labelsize=7)
        a.title.set_fontsize(8)
        a.xaxis.label.set_fontsize(7.5)
        a.yaxis.label.set_fontsize(7.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(ROOT / "paper" / "figures" / "preprocessing_s6.png", dpi=300)
    plt.close(fig)


def correlation(table: pd.DataFrame) -> None:
    samples = W.build_samples(table, horizons=(1,))
    cols = ["soh_origin", "cycles_since_first", "fade_rate_pp_per_cycle", "cycles_to_target",
            *W.feature_columns(table), FC.TARGET]
    cols = [c for c in cols if samples[c].notna().any()]  # as F1: never-observed features are dropped
    corr = samples[cols].corr(method="spearman")
    labels = [LABELS.get(c, c.replace("ic_pos_", "IC pos. ").replace("ic_height_", "IC height ")) for c in cols]
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), labels, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(cols)), labels, fontsize=6.5)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}".replace("0.", ".").replace("-.", "-."), ha="center", va="center",
                        fontsize=4.8, color="w" if abs(v) > 0.6 else "k")
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label("Spearman correlation", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    ax.set_title(f"S6 history and diagnostic inputs (F1) and target, horizon 1, {len(samples)} samples (pairwise complete)", fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "paper" / "figures" / "correlation_s6.png", dpi=300)
    plt.close(fig)


def main() -> int:
    table = dp.read_dataset(ROOT / "data" / "wheeler_checks.csv")
    preprocessing(table)
    correlation(table)
    print("wrote paper/figures/preprocessing_s6.png and paper/figures/correlation_s6.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
