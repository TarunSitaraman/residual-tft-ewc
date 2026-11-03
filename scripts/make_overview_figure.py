"""Main-text overview figure: where the proposed model stands and what each design step adds (S6).

(a) Final error on the continual stream (P1) against error on unseen usage conditions (P2), for the methods that keep
    no earlier condition's data: the proposed shrunk residual TFT + EWC (D25, D26), linear extrapolation (D25, D26)
    and gradient boosting refitted on the newest condition (D35 residual-target arm, post hoc; D32). Gradient
    boosting with every condition stored (D23, D26) is drawn hollow and grey as a reference outside that setting.
(b) Largest shortfall of each of those methods: max over the two settings of (its error - the best error among them).
(c) Stream error by design step: direct TFT + EWC and residual TFT + EWC (D23), residual + shrinkage (D25);
    bars main seed set, dots replicate set; dashed lines linear extrapolation.
(d) Forgetting of the residual TFT, horizons 1-6: fine-tuning against EWC (D23).

Every value is read from committed results files. Writes paper/figures/overview.png.

    python scripts/make_overview_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SHR = "tft_residual_ewc_shrunk"
BLUE, ORANGE, AQUA, GRAY, LIGHT = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85", "#b9b8b2"
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e6e5e0"


def load(name: str, exp: str) -> dict:
    return json.loads((ROOT / "results" / f"forecasting_{name}_{exp}.json").read_text())


def collect() -> dict:
    out = {}
    for e in ("e1", "e2"):
        d23, d25, d26, d32, d35 = (load(n, e) for n in ("d23", "d25", "d26", "d32", "d35"))
        s23 = {(s["set"], s["method"]): s for s in d23["summary"]}
        s25 = {(s["set"], s["method"]): s["final_avg_error_mean"] for s in d25["summary"]}
        s26 = {(s["set"], s["method"]): s["p2_error_mean"] for s in d26["summary"]}
        b23 = {b["method"]: b["final_avg_error"] for b in d23["baselines"]}
        b25 = {b["method"]: b["final_avg_error"] for b in d25["baselines"]}
        b26 = {b["method"]: b["p2_error"] for b in d26["baselines"]}
        mean2 = lambda d, m: (d[("main", m)] + d[("replicate", m)]) / 2  # noqa: E731
        out[e] = {
            "points": {  # (stream, unseen)
                "proposed": (mean2(s25, SHR), mean2(s26, SHR)),
                "lin": (b25["linear_extrapolation"], b26["linear_extrapolation"]),
                "gbm_last": (d35["summary"]["gbm_last_residual"]["final_avg_error"], d32["summary"]["gbm_last"]["p2_error"]),
                "gbm_all": (b23["gbm_shared"], b26["gbm_shared"]),
            },
            "steps": [(s23[(s, m)]["final_avg_error_mean"] for s in ("main", "replicate"))
                      for m in ("tft_direct_ewc", "tft_residual_ewc")],
            "shrunk": (s25[("main", SHR)], s25[("replicate", SHR)]),
            "forget": {m: tuple(s23[(s, m)]["forgetting_mean"] for s in ("main", "replicate"))
                       for m in ("tft_residual_finetune", "tft_residual_ewc")},
        }
        out[e]["steps"] = [tuple(x) for x in out[e]["steps"]] + [out[e]["shrunk"]]
    return out


def shortfall(points: dict) -> dict:
    keys = ("proposed", "lin", "gbm_last")
    best = [min(points[k][i] for k in keys) for i in (0, 1)]
    return {k: max(points[k][i] - best[i] for i in (0, 1)) for k in keys}


NAMES = {"proposed": "Residual TFT-EWC (proposed)", "lin": "Linear extrapolation",
         "gbm_last": "Gradient boosting, newest only", "gbm_all": "Gradient boosting, all data stored"}
STYLE = {"proposed": (BLUE, "o"), "lin": (ORANGE, "s"), "gbm_last": (AQUA, "D"), "gbm_all": (LIGHT, "D")}


def figure(data: dict, path: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.6})
    fig = plt.figure(figsize=(7.16, 2.25))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.2, 0.85, 0.95, 0.5], wspace=0.55)

    # (a) trade-off
    ax = fig.add_subplot(gs[0])
    for k in ("gbm_all", "gbm_last", "lin", "proposed"):
        c, m = STYLE[k]
        xy = [data[e]["points"][k] for e in ("e1", "e2")]
        ax.plot(*zip(*xy), color=c, lw=0.8, alpha=0.5, zorder=1, ls=(0, (2, 2)) if k == "gbm_all" else "-")
        for (x, y), lab in zip(xy, ("1–3", "1–6")):
            ax.scatter(x, y, s=34, marker=m, facecolor="white" if k == "gbm_all" else c, edgecolor=c,
                       linewidth=1.3 if k == "gbm_all" else 0.8, zorder=3)
            ax.annotate(lab, (x, y), xytext=(4, -2), textcoords="offset points", fontsize=5.6, color=MUTED)
    ax.set_xlim(0.2, 1.15); ax.set_ylim(0.3, 3.1)
    ax.set_xlabel("Error, continual stream (pp)"); ax.set_ylabel("Error, unseen conditions (pp)")
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.text(0.22, 0.36, "better in both ↙", color=MUTED, fontsize=5.8, style="italic")
    ax.set_title("(a) Two settings", fontsize=7.5, loc="left", color=INK)
    handles = [Line2D([], [], marker=STYLE[k][1], ls="", markersize=4.5, markerfacecolor="white" if k == "gbm_all"
                      else STYLE[k][0], markeredgecolor=STYLE[k][0]) for k in ("proposed", "lin", "gbm_last", "gbm_all")]
    ax.legend(handles, ["Proposed", "Lin. extrapolation", "GB, newest only", "GB, all stored (ref.)"],
              fontsize=5.4, frameon=False, loc="upper right", handletextpad=0.2, borderaxespad=0.1)

    # (b) largest shortfall
    ax = fig.add_subplot(gs[1])
    keys = ("proposed", "lin", "gbm_last")
    w = 0.38
    for i, k in enumerate(keys):
        for j, e in enumerate(("e1", "e2")):
            v = shortfall(data[e]["points"])[k]
            ax.bar(i + (j - 0.5) * (w + 0.02), v, w, color=STYLE[k][0], alpha=1.0 if j == 0 else 0.55, zorder=2)
            ax.text(i + (j - 0.5) * (w + 0.02), v + 0.015, f"{v:.2f}", ha="center", fontsize=5.4, color=INK)
    ax.set_xticks(range(3)); ax.set_xticklabels(["Prop.", "Lin.\nextrap.", "GB,\nnewest"], fontsize=5.8)
    ax.set_ylabel("Largest shortfall (pp)"); ax.set_ylim(0, 1.0)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend([Patch(color=GRAY), Patch(color=GRAY, alpha=0.55)], ["h=1–3", "h=1–6"], fontsize=5.4, frameon=False,
              loc="upper left")
    ax.set_title("(b) Worst setting", fontsize=7.5, loc="left", color=INK)

    # (c) design steps
    ax = fig.add_subplot(gs[2])
    w = 0.36
    for i in range(3):
        for j, e in enumerate(("e1", "e2")):
            mn, rp = data[e]["steps"][i]
            x = i + (j - 0.5) * (w + 0.02)
            ax.bar(x, mn, w, color=GRAY if i == 0 else BLUE, alpha=1.0 if j == 0 else 0.55, zorder=2)
            ax.scatter([x], [rp], s=7, color=INK, zorder=3)
            ax.text(x, max(mn, rp) + 0.02, f"{mn:.2f}", ha="center", fontsize=5.4, color=INK)
    for e, ls in (("e1", "--"), ("e2", ":")):
        ax.axhline(data[e]["points"]["lin"][0], color=ORANGE, lw=0.9, ls=ls, zorder=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(["Direct", "Residual", "Residual\n+ shrink."], fontsize=5.6)
    ax.set_ylabel("Error, continual stream (pp)"); ax.set_ylim(0, 1.0)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend([Patch(color=BLUE), Patch(color=BLUE, alpha=0.55), Line2D([], [], marker="o", ls="", color=INK, ms=2.5),
               Line2D([], [], color=ORANGE, ls="--", lw=0.9), Line2D([], [], color=ORANGE, ls=":", lw=0.9)],
              ["h=1–3", "h=1–6", "replicate", "lin. extrap. 1–3", "lin. extrap. 1–6"], fontsize=5.2, frameon=False,
              loc="upper right", handlelength=1.4, borderaxespad=0.0)
    ax.set_title("(c) Design steps", fontsize=7.5, loc="left", color=INK)

    # (d) forgetting
    ax = fig.add_subplot(gs[3])
    f = data["e2"]["forget"]
    for i, (m, c) in enumerate((("tft_residual_finetune", GRAY), ("tft_residual_ewc", BLUE))):
        mn, rp = f[m]
        ax.bar(i, mn, 0.6, color=c, zorder=2)
        ax.scatter([i], [rp], s=7, color=INK, zorder=3)
        ax.text(i, max(mn, rp) + 0.01, f"{mn:.2f}", ha="center", fontsize=5.4, color=INK)
    cut = [1 - f["tft_residual_ewc"][k] / f["tft_residual_finetune"][k] for k in (0, 1)]
    ax.text(0.5, 0.315, f"−{cut[0]:.0%} /\n−{cut[1]:.0%}", ha="center", fontsize=5.6, color=INK, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Fine-\ntune", "EWC"], fontsize=6)
    ax.set_ylabel("Forgetting, h=1–6 (pp)"); ax.set_ylim(0, 0.37)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_title("(d) Forgetting", fontsize=7.5, loc="left", color=INK)

    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")


def main() -> int:
    data = collect()
    path = ROOT / "paper" / "figures" / "overview.png"
    figure(data, path)
    for e in ("e1", "e2"):
        print(e, {k: [round(x, 3) for x in v] for k, v in data[e]["points"].items()},
              "shortfall", {k: round(v, 3) for k, v in shortfall(data[e]["points"]).items()})
    print(f"written {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
