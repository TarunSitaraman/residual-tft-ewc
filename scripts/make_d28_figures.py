"""Figures for D28 (continual refit and quantile tracking on an unseen S6 condition).

Outputs (never edited by hand), in ``paper/figures/``:

- ``d28_per_condition.png``: 90 % coverage (top) and 90 % interval score (bottom, log scale) per held-out condition
  for the four D28 arms, E1 and E2, from the committed results files ``results/forecasting_d28_<exp>.json``.
- ``d28_case7_timeline.png``: case7, E1, horizon 1. Absolute error of the static and the refitted forecast and the
  90 % half-widths of STATIC and REFIT+TRACK over the case's check dates. Recomputed with ``scripts/run_d28.py``
  (deterministic); the script checks that the recomputed case7 coverage equals the committed results file.

Styling is shared with ``make_d24_outputs.py``. Usage (from the repository root)::

    python scripts/make_d28_figures.py
"""

from __future__ import annotations

import json

import matplotlib.dates  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

import make_d24_outputs as M  # sets the Agg backend and the shared rcParams

EXPS = ("e1", "e2")
TITLE = {"e1": "E1 (horizons 1-3)", "e2": "E2 (horizons 1-6)"}
CASES = [f"case{i}" for i in range(1, 8)]
ARMS = (("static", "Static (D27)", M.GREY, "s", "none"), ("track", "Track only", M.GREY, "x", M.GREY),
        ("refit", "Refit only", M.FAMILY_COLOUR["mlp"], "D", "none"),
        ("refit_track", "Refit + track (proposed)", M.FAMILY_COLOUR["mlp"], "D", M.FAMILY_COLOUR["mlp"]))


def load(exp: str) -> dict:
    return json.loads((M.RES / f"forecasting_d28_{exp}.json").read_text())


def per_condition() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True)
    x = np.arange(len(CASES))
    for col, exp in enumerate(EXPS):
        s = load(exp)["summary"]
        for arm, lab, colour, mk, face in ARMS:
            axes[0, col].plot(x, [s[arm]["90"]["coverage_per_fold"][c] for c in CASES], mk, color=colour, mfc=face,
                              ms=4.5, label=lab)
            axes[1, col].plot(x, [s[arm]["90"]["interval_score_per_fold"][c] for c in CASES], mk, color=colour,
                              mfc=face, ms=4.5, label=lab)
        axes[0, col].axhline(0.9, color=M.INK, lw=0.7, ls=":")
        axes[0, col].axhline(0.8, color=M.INK, lw=0.7, ls="--")
        axes[0, col].set(title=TITLE[exp], ylim=(0.3, 1.02))
        axes[1, col].set(yscale="log", xticks=x, xticklabels=[c.replace("case", "") for c in CASES],
                         xlabel="Held-out usage condition (case)")
        axes[1, col].yaxis.set_major_locator(plt.matplotlib.ticker.FixedLocator([1, 2, 3, 5, 10, 20]))
        axes[1, col].yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter("%g"))
        axes[1, col].yaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
    axes[0, 0].set_ylabel("Coverage of 90 % interval")
    axes[1, 0].set_ylabel("90 % interval score (pp, log)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(M.FIG / "d28_per_condition.png")
    plt.close(fig)


def case7_timeline() -> None:
    import run_d28
    from src import forecasting as FC

    run = run_d28.D28("e1")
    f = run.fold("case7")
    committed = next(p for p in load("e1")["per_fold"] if p["case"] == "case7")
    assert abs(f["rules"]["refit_track"]["90"]["all"]["coverage"]
               - committed["rules"]["refit_track"]["90"]["all"]["coverage"]) < 1e-12, "case7 not reproduced"
    d = run.direct
    test = d[d["Battery_ID"] == "case7"].reset_index(drop=True)
    fit_rows = run_d28.run_d27.p2_fit_rows(d, "case7")
    p_static = run_d28.run_d27.gbm_predict(fit_rows, test, run.features)
    p_refit, _ = run_d28.refit_predictions(fit_rows, test, run.features, p_static)
    q0 = {int(k): v for k, v in f["q0"]["90"].items()}
    q_track = run_d28.track(test, p_refit, q0, 0.9, f["c"])
    y = test[FC.TARGET].to_numpy(float)
    t = run_d28.times(test)[0]
    h1 = test["horizon"].to_numpy() == 1
    o = np.argsort(t[h1], kind="stable")
    tt = t[h1][o]
    fig, a = plt.subplots(figsize=(7.2, 3.4))
    a.plot(tt, np.abs(y - p_static)[h1][o], "s", color=M.GREY, mfc="none", ms=3.5, label="|error|, static forecast")
    a.plot(tt, np.abs(y - p_refit)[h1][o], "D", color=M.FAMILY_COLOUR["mlp"], ms=3, label="|error|, refitted forecast")
    a.step(tt, np.full(tt.size, q0[1]), where="post", color=M.GREY, lw=1.2, label="90 % half-width, static")
    a.step(tt, q_track[h1][o], where="post", color=M.FAMILY_COLOUR["mlp"], lw=1.2,
           label="90 % half-width, refit + track")
    a.set(xlabel="Check date of the held-out cells", ylabel="pp of SOH",
          title="Case 7 (100 % depth of discharge) held out, horizon 1: a point above its line is not covered")
    a.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(bymonth=(1, 7)))
    a.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, handletextpad=0.3, columnspacing=1.5)
    fig.tight_layout()
    fig.savefig(M.FIG / "d28_case7_timeline.png")
    plt.close(fig)


if __name__ == "__main__":
    per_condition()
    case7_timeline()
    print("written: d28_per_condition.png, d28_case7_timeline.png")
