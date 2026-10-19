"""Figures for D25-D27, from committed results files only (``results/forecasting_d2{5,6,7}_<exp>.json``).

Outputs (never edited by hand), in ``paper/figures/``:

- ``d25_shrinkage.png`` (paper, one column): D25, continual stream. Shrinkage factor of the TFT correction per horizon
  at the end of the stream (main seed set, median and interquartile range over the 50 runs; E1 and E2).
- ``d26_per_condition.png`` (notebook): D26, unseen usage conditions. Test RMSE per held-out condition of linear
  extrapolation, gradient boosting, the shrunk residual TFT + EWC (continual) and the jointly trained residual TFT
  (main seeds, mean).
- ``d27_coverage.png`` (paper, one column): D27. 90 % coverage per held-out condition for gradient boosting calibrated
  leave-one-condition-out, linear extrapolation calibrated on all origins and on validation rows (D27), and the
  shrunk TFT calibrated on validation rows (D26, main seeds, mean).

Styling is shared with ``make_d24_outputs.py``; the one-column figures are drawn at their printed width (3.5 in) so
no text is smaller than 6.5 pt. Usage (from the repository root)::

    python scripts/make_d25_d27_figures.py
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

import make_d24_outputs as M  # sets the Agg backend and the shared rcParams

EXPS = ("e1", "e2")
TITLE = {"e1": "E1 (horizons 1-3)", "e2": "E2 (horizons 1-6)"}
CASES = [f"case{i}" for i in range(1, 8)]
TFT, GBM, LIN, JOINT = M.FAMILY_COLOUR["tft"], M.FAMILY_COLOUR["mlp"], M.GREY, M.FAMILY_COLOUR["lstm"]
COLUMN = {"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
          "ytick.labelsize": 6.5, "legend.fontsize": 6.5}


def load(d: str, exp: str) -> dict:
    return json.loads((M.RES / f"forecasting_{d}_{exp}.json").read_text())


def runs(data: dict, method: str, st: str = "main") -> list[dict]:
    return [r for r in data["runs"] if r["method"] == method and r["set"] == st]


def fold_mean(rs: list[dict], key: str) -> list[float]:
    return [float(np.mean([r[key][c] for r in rs])) for c in CASES]


def d25() -> None:
    with plt.rc_context(COLUMN):
        fig, a = plt.subplots(figsize=(3.5, 1.9))
        for exp, ls in zip(EXPS, ("-", "--")):
            data = load("d25", exp)
            al = np.array([[r["alpha_final"][str(h)] for h in data["horizons"]]
                           for r in runs(data, "tft_residual_ewc_shrunk")])
            h = np.array(data["horizons"])
            q1, med, q3 = np.percentile(al, [25, 50, 75], axis=0)
            a.plot(h, med, ls, color=TFT, marker="o", ms=3, label=TITLE[exp])
            a.fill_between(h, q1, q3, color=TFT, alpha=0.15, lw=0)
        a.set(xlabel="Horizon (capacity checks ahead)", ylabel="Shrinkage factor $\\alpha_h$", ylim=(-0.03, 1.05))
        a.legend(loc="lower right")
        fig.tight_layout(pad=0.3)
        fig.savefig(M.FIG / "d25_shrinkage.png")
        plt.close(fig)


def d26() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharex=True)
    x = np.arange(len(CASES))
    for ax, exp in zip(axes, EXPS):
        data = load("d26", exp)
        base = {b["method"]: b for b in data["baselines"]}
        series = (("Linear extrapolation", [base["linear_extrapolation"]["per_fold"][c] for c in CASES], LIN, "s"),
                  ("Gradient boosting", [base["gbm_shared"]["per_fold"][c] for c in CASES], GBM, "D"),
                  ("Shrunk TFT + EWC (continual)", fold_mean(runs(data, "tft_residual_ewc_shrunk"), "per_fold"), TFT, "o"),
                  ("TFT, joint training", fold_mean(runs(data, "tft_residual_joint"), "per_fold"), JOINT, "^"))
        for lab, v, col, mk in series:
            ax.plot(x, v, mk, color=col, ms=4.5, label=lab)
        ax.set(xticks=x, xticklabels=[c.replace("case", "") for c in CASES], xlabel="Held-out usage condition (case)",
               title=TITLE[exp], ylim=(0, None))
    axes[0].set_ylabel("Test RMSE (pp of SOH)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(M.FIG / "d26_per_condition.png")
    plt.close(fig)


def d27() -> None:
    with plt.rc_context(COLUMN):
        fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.5), sharey=True)
        x = np.arange(len(CASES))
        for ax, exp in zip(axes, EXPS):
            s27 = load("d27", exp)["summary"]
            tft = dict(zip(CASES, fold_mean(runs(load("d26", exp), "tft_residual_ewc_shrunk"), "cov90_per_fold")))
            series = (("GBM, held-out conditions", s27["gbm_loco"]["90"]["coverage_per_fold"], GBM, "D", GBM),
                      ("Extrapolation, all origins", s27["lin_all"]["90"]["coverage_per_fold"], LIN, "s", "none"),
                      ("Extrapolation, validation rows", s27["lin_val"]["90"]["coverage_per_fold"], LIN, "x", LIN),
                      ("TFT + EWC, validation rows", tft, TFT, "o", TFT))
            for lab, v, col, mk, face in series:
                ax.plot(x, [v[c] for c in CASES], mk, color=col, mfc=face, ms=3.5, label=lab)
            ax.axhline(0.9, color=M.INK, lw=0.7, ls=":")
            ax.set(xticks=x, xticklabels=[c.replace("case", "") for c in CASES], xlabel="Held-out condition",
                   title=exp.upper(), ylim=(0.3, 1.02))
        axes[0].set_ylabel("Coverage of 90 % interval")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=2, handletextpad=0.2, columnspacing=1.0)
        fig.tight_layout(pad=0.3, rect=(0, 0.15, 1, 1))
        fig.savefig(M.FIG / "d27_coverage.png")
        plt.close(fig)


if __name__ == "__main__":
    d25()
    d26()
    d27()
    print("written: d25_shrinkage.png, d26_per_condition.png, d27_coverage.png")
