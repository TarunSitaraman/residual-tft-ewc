"""Figures for the D21 results (P3, ablation A1, TFT interpretation) and D23 (residual by horizon), from the committed results files only.

Writes PNGs to paper/figures/ (300 dpi). Colours: TFT blue #2a78d6, MLP orange #eb6834 (validated
categorical pair); baselines in neutral grey. Run from the repository root::

    python scripts/make_d21_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
BLUE, ORANGE, GREY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#8a8981", "#0b0b0b", "#52514e", "#e4e3dd"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": GREY,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "legend.frameon": False, "savefig.dpi": 300})
LABELS = {"tft_scratch_finetune": "TFT, from scratch", "tft_pre_finetune": "TFT, pretrained + fine-tuning",
          "tft_pre_ewc": "TFT, pretrained + EWC", "mlp_scratch_finetune": "MLP, from scratch",
          "mlp_pre_finetune": "MLP, pretrained + fine-tuning", "mlp_pre_ewc": "MLP, pretrained + EWC"}


def boot_ci(x: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fig_p3(d: dict) -> Path:
    runs = pd.DataFrame(d["runs"])
    order = ["tft_scratch_finetune", "tft_pre_finetune", "tft_pre_ewc", "mlp_scratch_finetune", "mlp_pre_finetune", "mlp_pre_ewc"]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for i, m in enumerate(order):
        col = BLUE if m.startswith("tft") else ORANGE
        for k, (st, marker, dy) in enumerate((("main", "o", -0.14), ("replicate", "^", 0.14))):
            x = runs[(runs["method"] == m) & (runs["set"] == st)]["final_avg_error"].to_numpy(float)
            lo, hi = boot_ci(x)
            ax.plot([lo, hi], [i + dy] * 2, color=col, linewidth=2, solid_capstyle="round")
            ax.plot(x.mean(), i + dy, marker=marker, color=col, markersize=7, markeredgecolor="white", markeredgewidth=1.2,
                    label=("main seeds 0–4" if st == "main" else "replicate seeds 5–9") if i == 0 else None, linestyle="none")
    base = {b["method"]: b["rmse_final_mean_over_tasks"] for b in d["baselines"]}
    for (name, lab), yy in zip((("linear_extrapolation", "linear extrapolation"), ("gbm_shared", "gradient boosting")), (-0.85, -0.55)):
        ax.axvline(base[name], color=GREY, linestyle="--", linewidth=1)
        ax.text(base[name], yy, f" {lab} {base[name]:.3f}", color=INK2, fontsize=7.5, va="center", ha="left")
    ax.set_yticks(range(len(order)), [LABELS[m] for m in order])
    ax.invert_yaxis()
    ax.set_ylim(len(order) - 0.4, -1.05)
    ax.set_xlabel("Final average test RMSE on S6 (SOH pp)")
    ax.set_title("P3: first-life pretraining then the S6 stream", loc="left")
    ax.grid(axis="y", visible=False)
    leg = ax.legend(loc="upper right", fontsize=8)
    for h in leg.legend_handles:
        h.set_color(INK2)
    fig.tight_layout()
    path = OUT / "d21_p3_final_error.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_retention(d: dict) -> Path:
    s = pd.DataFrame(d["runs"]).groupby(["set", "method"])[["pool_error_before", "pool_error_after"]].mean()
    pre = ["tft_pre_ewc", "tft_pre_finetune", "mlp_pre_ewc", "mlp_pre_finetune"]
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    for i, m in enumerate(pre):
        col = BLUE if m.startswith("tft") else ORANGE
        for st, marker, dy in (("main", "o", -0.12), ("replicate", "^", 0.12)):
            b, a = s.loc[(st, m)]
            ax.annotate("", xy=(a, i + dy), xytext=(b, i + dy), arrowprops={"arrowstyle": "->", "color": col, "linewidth": 1.5})
            ax.plot(b, i + dy, marker=marker, color=col, markersize=6, markerfacecolor="white", linestyle="none")
            ax.plot(a, i + dy, marker=marker, color=col, markersize=7, markeredgecolor="white", linestyle="none")
    for v, lab in ((3.001, "linear extrapolation 3.001"), (3.878, "persistence 3.878")):
        ax.axvline(v, color=GREY, linestyle="--", linewidth=1)
        ax.text(v, -0.7, f" {lab}", color=INK2, fontsize=8, va="bottom")
    ax.set_yticks(range(len(pre)), [LABELS[m] for m in pre])
    ax.invert_yaxis()
    ax.set_ylim(len(pre) - 0.5, -0.9)
    ax.set_xlabel("Pool test RMSE (SOH pp); open = before S6 stream, filled = after")
    ax.set_title("Retention of first-life knowledge (pool test split)", loc="left")
    ax.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", color=INK2, linestyle="none", label="main seeds 0–4"),
                       Line2D([], [], marker="^", color=INK2, linestyle="none", label="replicate seeds 5–9")],
              loc="upper right", bbox_to_anchor=(0.9, 0.86), fontsize=8)
    fig.tight_layout()
    path = OUT / "d21_p3_pool_retention.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_interpretation(d: dict) -> Path:
    w = d["variable_weights"]
    mean = pd.Series(w["mean"]).sort_values()
    lo, hi = pd.Series(w["min"]).reindex(mean.index), pd.Series(w["max"]).reindex(mean.index)
    constant_in_s6 = {"calendar", "rest_soc", "temp_c", "chem_LFP", "chem_NCA", "chem_NMC", "chem_NMC_NCA"}
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    y = np.arange(len(mean))
    for i, f in enumerate(mean.index):
        col = GREY if f in constant_in_s6 else BLUE
        ax.plot([lo[f], hi[f]], [i, i], color=col, linewidth=2, alpha=0.45, solid_capstyle="round")
        ax.plot(mean[f], i, "o", color=col, markersize=6, markeredgecolor="white")
    ax.set_yticks(y, [f"{f} (constant in S6)" if f in constant_in_s6 else f for f in mean.index])
    ax.set_xlabel("Variable-selection weight (dot: mean; bar: range over 5 seeds)")
    ax.set_title("TFT variable-selection weights on S6", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    path = OUT / "d21_tft_variable_weights.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_ablation(d: dict, rep: dict | None = None) -> Path:
    s = pd.DataFrame(d["summary"])
    r5 = pd.DataFrame(rep["summary"]) if rep else None
    rows = ["D20_F1", "D20_F0", "D21_shared", "D21_shared_no_calendar"]
    names = {"D20_F1": "D20 inputs (with diagnostics)", "D20_F0": "D20 inputs, no diagnostics",
             "D21_shared": "D21 shared inputs", "D21_shared_no_calendar": "D21 shared, no calendar time"}
    lrs = sorted(s["lr"].unique(), reverse=True)
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    for j, lr in enumerate(lrs):
        vals = [float(s[(s["inputs"] == r) & (s["lr"] == lr)]["mean"].iloc[0]) for r in rows]
        y = np.arange(len(rows)) + (j - 0.5) * 0.32
        ax.barh(y, vals, height=0.28, color=BLUE if j == 0 else "#86b6ef", label=f"learning rate {lr:g}, seeds 0-4")
        ends = vals
        if r5 is not None:
            reps = [float(r5[(r5["inputs"] == r) & (r5["lr"] == lr)]["mean"].iloc[0]) for r in rows]
            ax.scatter(reps, y, marker="D", s=14, color=INK, zorder=3, label="replicate, seeds 5-9" if j == 0 else None)
            ends = np.maximum(vals, reps)
        for yy, v, e in zip(y, vals, ends):
            ax.text(e + 0.015, yy, f"{v:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(rows)), [names[r] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Final average test RMSE on S6 (SOH pp), TFT from scratch")
    ax.set_title("Ablation A1: what changed the TFT between D20 and D21", loc="left")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.35, -0.22), ncol=3, fontsize=7.5)
    fig.tight_layout()
    path = OUT / "d21_ablation_a1.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_residual(d: dict) -> Path:
    """D23 E2: test RMSE by horizon after the last stream step (main seeds: mean and 95% bootstrap CI;
    replicate seeds: mean, open marker) against linear extrapolation."""
    runs = pd.DataFrame(d["runs"])
    hs = d["horizons"]
    lin = next(b for b in d["baselines"] if b["method"] == "linear_extrapolation")
    series = [("tft_residual_ewc", "TFT + EWC, residual", BLUE, "o"), ("tft_residual_finetune", "TFT fine-tuning, residual", "#86b6ef", "s"),
              ("tft_direct_ewc", "TFT + EWC, direct", ORANGE, "^")]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(hs, [lin[f"rmse_h{h}"] for h in hs], color=GREY, ls="--", lw=1.5, label="Linear extrapolation")
    for k, (m, label, c, mk) in enumerate(series):
        x = np.array(hs) + (k - 1) * 0.12
        main = runs[(runs["method"] == m) & (runs["set"] == "main")]
        rep = runs[(runs["method"] == m) & (runs["set"] == "replicate")]
        mean = [main[f"rmse_h{h}"].mean() for h in hs]
        ci = np.array([boot_ci(main[f"rmse_h{h}"].to_numpy()) for h in hs])
        ax.errorbar(x, mean, yerr=[np.array(mean) - ci[:, 0], ci[:, 1] - np.array(mean)], color=c, marker=mk, ms=4,
                    lw=1.5, capsize=2, label=label)
        ax.scatter(x, [rep[f"rmse_h{h}"].mean() for h in hs], marker=mk, s=16, facecolors="white", edgecolors=c, zorder=3)
    ax.set_xticks(hs)
    ax.set_xlabel("Horizon h (checks ahead, 100 cycles each)")
    ax.set_ylabel("Test RMSE of SOH (pp)")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    path = OUT / "d23_residual_horizon.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = ROOT / "results"
    made = [fig_p3(json.loads((res / "forecasting_d21_p3.json").read_text())),
            fig_retention(json.loads((res / "forecasting_d21_p3.json").read_text())),
            fig_interpretation(json.loads((res / "d21_tft_interpretation.json").read_text()))]
    a1, a1r = res / "forecasting_d21_ablation_a1.json", res / "forecasting_d21_ablation_a1_replicate.json"
    if a1.exists():
        rep = json.loads(a1r.read_text()) if a1r.exists() else None
        made.append(fig_ablation(json.loads(a1.read_text()), rep))
    e2 = res / "forecasting_d23_e2.json"
    if e2.exists():
        made.append(fig_residual(json.loads(e2.read_text())))
    for p in made:
        print("wrote", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
