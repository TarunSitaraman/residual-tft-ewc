"""Comparison tables and figures for D23/D24 (work plan WS5), from committed results files only.

For each experiment (E1: horizons 1-3, E2: horizons 1-6) it uses ``results/forecasting_d24_<exp>.json`` when present
(RC-TFT, the literature baselines and the D23 runs it carries), otherwise ``results/forecasting_d23_<exp>.json``.
Outputs (never edited by hand):

- ``results/tables/d24_comparison_<exp>.csv`` and ``paper/tables/d24_comparison_<exp>.tex``: every method's final
  average test RMSE (mean and s.d. over the 50 runs of each seed set), forgetting, backward transfer, parameters and
  training time; deterministic baselines as single values.
- ``results/tables/d24_tests_<exp>.csv`` and ``.tex``: the registered paired tests against the reference with the D21
  replication verdict.
- ``results/tables/d24_per_group_<exp>.csv`` and ``.tex``: test RMSE of every S6 test case after the last stream step
  (main seeds, mean over runs), as CLAUDE.md requires per-group reporting.
- ``paper/figures/d24_comparison_<exp>.png``, ``d24_per_group_<exp>.png`` (residual + EWC arms and baselines only), ``d24_forgetting_<exp>.png`` and, for E2,
  ``d24_horizon_e2.png``; ``spark_scaling.png`` when ``results/d22_spark_scaling.json`` exists.

Colours (validated categorical slots 1-5, adjacent pairs; fixed order, never cycled): TFT blue, MLP orange, BiLSTM
aqua, TCN yellow, Transformer magenta; deterministic and tabular baselines grey. Every method is labelled on the axis
(three slots are below 3:1 contrast, so identity never rests on colour alone).

Usage (from the repository root)::

    python scripts/make_d24_outputs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, TAB, FIG, PTAB = ROOT / "results", ROOT / "results" / "tables", ROOT / "paper" / "figures", ROOT / "paper" / "tables"
FAMILY_COLOUR = {"tft": "#2a78d6", "mlp": "#eb6834", "lstm": "#1baf7a", "tcn": "#eda100", "transformer": "#e87ba4"}
FAMILY_NAME = {"tft": "TFT", "mlp": "MLP", "lstm": "BiLSTM", "tcn": "TCN", "transformer": "Transformer"}
GREY, INK, INK2, GRID = "#8a8981", "#0b0b0b", "#52514e", "#e4e3dd"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": GREY,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "legend.frameon": False, "savefig.dpi": 300})
BASELINE_NAMES = {"persistence": "Persistence", "linear_extrapolation": "Linear extrapolation",
                  "ridge_schedule": "Ridge, usage only", "ridge_shared": "Ridge", "gbm_shared": "Gradient boosting",
                  "gpr_direct": "GPR, direct", "gpr_residual": "GPR, residual", "rf_direct": "Random forest, direct",
                  "rf_residual": "Random forest, residual", "svr_direct": "SVR, direct", "svr_residual": "SVR, residual"}
EQUIV, ALPHA = 0.05, 0.05   # D21 replication rule


def label(method: str) -> str:
    if method in BASELINE_NAMES:
        return BASELINE_NAMES[method]
    if method == "tft_residual_saewc":
        return "Residual TFT + SA-EWC"
    backbone, arm, strategy = method.split("_")
    strat = {"ewc": "+ EWC", "finetune": "fine-tuning", "saewc": "+ SA-EWC"}[strategy]
    return f"{FAMILY_NAME[backbone]} {strat}, {arm}"


def family(method: str) -> str | None:
    head = method.split("_")[0]
    return head if head in FAMILY_COLOUR and method not in BASELINE_NAMES else None


def load(exp: str) -> tuple[dict, str] | None:
    """The full D24 file if present; otherwise the D23 file, with its baselines replaced by the D24 baselines part
    (``forecasting_d24_baselines_<exp>.json``: the same D23 baselines recomputed plus GPR, random forest and SVR)
    when that has been written."""
    for name in (f"forecasting_d24_{exp}.json", f"forecasting_d24_{exp}_smoke.json"):
        if (RES / name).exists():
            return json.loads((RES / name).read_text()), name
    d23 = RES / f"forecasting_d23_{exp}.json"
    if not d23.exists():
        return None
    d, src = json.loads(d23.read_text()), d23.name
    for name in (f"forecasting_d24_baselines_{exp}.json", f"forecasting_d24_baselines_{exp}_smoke.json"):
        if (RES / name).exists():
            d["baselines"] = json.loads((RES / name).read_text())["baselines"]
            src += f" + {name}"
            break
    return d, src


def reference(d: dict) -> str:
    return d.get("reference", "tft_residual_ewc")


# -- tables ---------------------------------------------------------------------------------------------

def comparison(d: dict) -> pd.DataFrame:
    runs = pd.DataFrame(d["runs"])
    params = d.get("parameter_counts", {})
    rows = []
    for m, g in runs.groupby("method"):
        row = {"method": m, "label": label(m), "kind": "continual (stream)", "family": FAMILY_NAME.get(family(m), "")}
        for st in ("main", "replicate"):
            s = g[g["set"] == st]
            row[f"rmse_{st}_mean"], row[f"rmse_{st}_sd"] = s["final_avg_error"].mean(), s["final_avg_error"].std()
            row[f"n_{st}"] = len(s)
        main = g[g["set"] == "main"]
        row.update(forgetting_main=main["forgetting"].mean(), bwt_main=main["backward_transfer"].mean(),
                   train_time_s=main["train_time_s"].mean(), parameters=params.get(m.split("_")[0]))
        rows.append(row)
    for b in d["baselines"]:
        kind = "deterministic" if b["method"] in ("persistence", "linear_extrapolation") else "fitted once on all tasks"
        rows.append({"method": b["method"], "label": label(b["method"]), "kind": kind, "family": "baseline",
                     "rmse_main_mean": b["final_avg_error"], "rmse_replicate_mean": b["final_avg_error"]})
    return pd.DataFrame(rows).sort_values("rmse_main_mean").reset_index(drop=True)


def _num(x) -> float:
    return float("nan") if x is None else float(x)


def verdict(main: pd.Series, rep: pd.Series) -> str:
    dm, dr = _num(main["mean_diff"]), _num(rep["mean_diff"])
    if np.isnan(dm) or np.isnan(dr):
        return "not testable"
    if abs(dm) < EQUIV and abs(dr) < EQUIV:
        return "practically equivalent"
    if (_num(main["p_holm"]) < ALPHA and _num(rep["p_holm"]) < ALPHA and np.sign(dm) == np.sign(dr)
            and abs(dm) >= EQUIV and abs(dr) >= EQUIV):
        return "replicated, " + ("reference better" if dm < 0 else "reference worse")
    return "not replicated"


def tests_table(d: dict) -> pd.DataFrame:
    t = pd.DataFrame(d["tests_vs_reference"])
    rows = []
    for (fam, metric, other), g in t.groupby(["family", "metric", "other"], sort=False):
        if set(g["set"]) != {"main", "replicate"}:
            continue
        m, r = g[g["set"] == "main"].iloc[0], g[g["set"] == "replicate"].iloc[0]
        rows.append({"family": fam, "metric": metric, "reference": label(reference(d)), "other": label(other),
                     "diff_main": _num(m["mean_diff"]), "ci_main": f"[{fmt(_num(m['ci_low']))}, {fmt(_num(m['ci_high']))}]",
                     "p_holm_main": _num(m["p_holm"]), "r_main": _num(m["rank_biserial"]),
                     "diff_replicate": _num(r["mean_diff"]), "p_holm_replicate": _num(r["p_holm"]),
                     "verdict": verdict(m, r)})
    return pd.DataFrame(rows)


def per_group(d: dict) -> pd.DataFrame:
    runs = pd.DataFrame(d["runs"])
    main = runs[runs["set"] == "main"]
    rows = {label(m): pd.DataFrame(list(g["final_error_per_task"])).mean() for m, g in main.groupby("method")}
    for b in d["baselines"]:
        if "final_error_per_task" in b:
            rows[label(b["method"])] = pd.Series(b["final_error_per_task"])
    t = pd.DataFrame(rows).T
    t = t[sorted(t.columns, key=lambda c: int(c.replace("case", "")))]
    t["mean"] = t.mean(axis=1)
    return t.sort_values("mean")


def tex_escape(s: str) -> str:
    return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def write_tex(path: Path, caption: str, lab: str, header: list[str], body: list[list[str]], align: str,
              escape: bool = True) -> None:
    lines = [r"\begin{table}[t]", r"  \centering", f"  \\caption{{{caption}}}", f"  \\label{{{lab}}}",
             r"  \setlength{\tabcolsep}{3pt}", r"  \scriptsize", f"  \\begin{{tabular}}{{{align}}}", r"    \toprule",
             "    " + " & ".join(header) + r" \\", r"    \midrule"]
    lines += ["    " + " & ".join(tex_escape(c) if escape else str(c) for c in row) + r" \\" for row in body]
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines))


def fmt(x, nd=3) -> str:
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def sfmt(x, nd=3) -> str:
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{nd}f}"


def tables(exp: str, d: dict, src: str) -> dict[str, pd.DataFrame]:
    comp, tst, grp = comparison(d), tests_table(d), per_group(d)
    comp.to_csv(TAB / f"d24_comparison_{exp}.csv", index=False)
    tst.to_csv(TAB / f"d24_tests_{exp}.csv", index=False)
    grp.to_csv(TAB / f"d24_per_group_{exp}.csv", float_format="%.4f")
    h = "1--3" if exp == "e1" else "1--6"
    ref = label(reference(d))
    body = []
    for r in comp.itertuples():
        det = r.kind != "continual (stream)"
        rm = fmt(r.rmse_main_mean) if det else f"{fmt(r.rmse_main_mean)} ({fmt(r.rmse_main_sd)})"
        rr = "" if det else f"{fmt(r.rmse_replicate_mean)} ({fmt(r.rmse_replicate_sd)})"
        name = rf"\textbf{{{tex_escape(r.label)}}}" if r.method == reference(d) else tex_escape(r.label)
        body.append([name, rm, rr, "" if det else fmt(r.forgetting_main), "" if det else fmt(r.bwt_main),
                     "" if det or pd.isna(r.parameters) else f"{int(r.parameters):,}"])
    body = [[b[0]] + [tex_escape(c) for c in b[1:]] for b in body]
    write_tex(PTAB / f"d24_comparison_{exp}.tex",
              f"S6 stream, horizons {h}: final average test RMSE (pp), mean (s.d.) over 50 runs per seed set "
              f"(main 0--4 / replicate 5--9); forgetting and backward transfer (BWT, pp) on main seeds. Baselines "
              f"without a spread are single fits. Source: \\texttt{{results/{tex_escape(src)}}}.",
              f"tab:d24-comparison-{exp}", ["Method", "RMSE main", "RMSE replicate", "Forget.", "BWT", "Params"],
              body, "lrrrrr", escape=False)
    tb = [[r.family, r.metric.replace("final_avg_error", "final error"), r.other, sfmt(r.diff_main),
           fmt(r.p_holm_main), sfmt(r.diff_replicate), fmt(r.p_holm_replicate), r.verdict]
          for r in tst.itertuples()]
    write_tex(PTAB / f"d24_tests_{exp}.tex",
              f"Paired tests, {tex_escape(ref)} minus each other method (pp; negative favours the reference), "
              f"horizons {h}, main / replicate seeds, Holm-adjusted Wilcoxon p, D21 replication rule.",
              f"tab:d24-tests-{exp}", ["Family", "Metric", "Other", "Diff. main", "$p$ main", "Diff. rep.", "$p$ rep.",
                                       "Verdict"], tb, "lllrrrrl")
    cases = [c for c in grp.columns if c != "mean"]
    gb = [[i, *[fmt(v) for v in r[cases]], fmt(r["mean"])] for i, r in grp.iterrows()]
    write_tex(PTAB / f"d24_per_group_{exp}.tex",
              f"Test RMSE (pp) per S6 test case after the last stream step, horizons {h}; learned methods: mean over "
              f"the 50 main-seed runs.", f"tab:d24-per-group-{exp}",
              ["Method", *[c.replace("case", "Case ") for c in cases], "Mean"], gb, "l" + "r" * (len(cases) + 1))
    return {"comparison": comp, "tests": tst, "per_group": grp}


# -- figures --------------------------------------------------------------------------------------------

def boot_ci(x: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fig_comparison(exp: str, d: dict, comp: pd.DataFrame) -> Path:
    runs = pd.DataFrame(d["runs"])
    comp = comp.sort_values("rmse_main_mean", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.6, 0.24 * len(comp) + 1.1))
    lin = comp.loc[comp["method"] == "linear_extrapolation", "rmse_main_mean"]
    if len(lin):
        ax.axvline(float(lin.iloc[0]), color=GREY, linestyle="--", linewidth=1, zorder=1)
    for y, r in comp.iterrows():
        fam = family(r["method"])
        if fam is None:
            ax.plot(r["rmse_main_mean"], y, marker="D", markersize=5, color=GREY, linestyle="none", zorder=3)
            continue
        col = FAMILY_COLOUR[fam]
        for st, dy, face in (("main", -0.15, col), ("replicate", 0.15, "white")):
            x = runs[(runs["method"] == r["method"]) & (runs["set"] == st)]["final_avg_error"].to_numpy(float)
            if not len(x):
                continue
            lo, hi = boot_ci(x)
            ax.plot([lo, hi], [y + dy] * 2, color=col, linewidth=2, solid_capstyle="round", zorder=2)
            ax.plot(x.mean(), y + dy, marker="o", markersize=5, markerfacecolor=face, markeredgecolor=col,
                    markeredgewidth=1.4, linestyle="none", zorder=3)
    ax.set_yticks(range(len(comp)))
    ax.set_yticklabels(comp["label"])
    for t, m in zip(ax.get_yticklabels(), comp["method"]):
        if m == reference(d):
            t.set_fontweight("bold")
    ax.set_xlabel("Final average test RMSE of SOH(k+h) (percentage points)")
    ax.grid(axis="y", visible=False)
    present = [f for f in FAMILY_COLOUR if any(family(m) == f for m in comp["method"])]
    handles = [Line2D([], [], marker="o", color=FAMILY_COLOUR[f], linestyle="none", label=FAMILY_NAME[f]) for f in present]
    handles += [Line2D([], [], marker="D", color=GREY, linestyle="none", label="baseline (single fit)"),
                Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=INK2, linestyle="none",
                       label="open: replicate seeds 5–9"),
                Line2D([], [], color=GREY, linestyle="--", label="linear extrapolation")]
    ax.legend(handles=handles, loc="upper right", fontsize=7)
    h = "1–3" if exp == "e1" else "1–6"
    ax.set_title(f"S6 stream, horizons {h}: mean, 95 % bootstrap CI (50 runs)", loc="left")
    fig.tight_layout()
    out = FIG / f"d24_comparison_{exp}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_per_group(exp: str, grp: pd.DataFrame) -> Path:
    # One arm per method, the same rule for every backbone: the residual + EWC arm of each learned model and each
    # baseline's residual (or only) form. Fine-tuning, direct and usage-only arms stay in the CSV and .tex tables.
    grp = grp[~grp.index.str.contains("fine-tuning|, direct|usage only")]
    cases = [c for c in grp.columns if c != "mean"]
    v = grp[cases].to_numpy(float)
    fig, ax = plt.subplots(figsize=(6.6, 0.24 * len(grp) + 1.2))
    im = ax.imshow(v, cmap="Blues", aspect="auto", vmin=0, vmax=np.nanpercentile(v, 95))
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            ax.text(j, i, f"{v[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v[i, j] > 0.6 * np.nanpercentile(v, 95) else INK)
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([c.replace("case", "Case ") for c in cases])
    ax.set_yticks(range(len(grp)))
    ax.set_yticklabels(grp.index)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Test RMSE (pp)")
    ax.set_title("Per test case after the last stream step (main seeds)\n"
                 "residual + EWC arms and baselines; other arms in the table", loc="left")
    fig.tight_layout()
    out = FIG / f"d24_per_group_{exp}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_forgetting(exp: str, d: dict, comp: pd.DataFrame) -> Path:
    c = comp[comp["kind"] == "continual (stream)"]
    ref = reference(d)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for r in c.itertuples():
        is_tft = family(r.method) == "tft"
        ax.plot(r.rmse_main_mean, r.forgetting_main, marker="o", markersize=7 if r.method == ref else 5,
                color=FAMILY_COLOUR["tft"] if is_tft else GREY, linestyle="none", zorder=3 if is_tft else 2)
        if is_tft:
            ax.annotate(r.label, (r.rmse_main_mean, r.forgetting_main), xytext=(4, 3), textcoords="offset points",
                        fontsize=6.5, fontweight="bold" if r.method == ref else "normal")
    ax.set_xlabel("Final average test RMSE (pp)")
    ax.set_ylabel("Forgetting (pp)")
    ax.legend(handles=[Line2D([], [], marker="o", color=FAMILY_COLOUR["tft"], linestyle="none", label="TFT variants"),
                       Line2D([], [], marker="o", color=GREY, linestyle="none", label="other continual methods")],
              loc="upper left", fontsize=7)
    ax.set_title("Accuracy and forgetting (main seeds, means)", loc="left")
    fig.tight_layout()
    out = FIG / f"d24_forgetting_{exp}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_horizon(d: dict) -> Path:
    runs = pd.DataFrame(d["runs"])
    hs = [int(c[6:]) for c in runs.columns if c.startswith("rmse_h") and runs[c].notna().any()]
    main = runs[runs["set"] == "main"]
    ref = reference(d)
    show = [ref, "tft_residual_ewc", "lstm_residual_ewc", "tcn_residual_ewc", "transformer_residual_ewc",
            "mlp_residual_ewc"]
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for m in [s for s in dict.fromkeys(show) if s in set(main["method"])]:
        y = [main[main["method"] == m][f"rmse_h{h}"].mean() for h in hs]
        fam = family(m)
        ax.plot(hs, y, marker="o", markersize=4, linewidth=2 if m == ref else 1.4, color=FAMILY_COLOUR[fam],
                linestyle="-" if m != "tft_residual_ewc" or ref == m else (0, (4, 2)), label=label(m))
    for b in d["baselines"]:
        if b["method"] in ("linear_extrapolation", "gbm_shared"):
            ax.plot(hs, [b[f"rmse_h{h}"] for h in hs], color=GREY, linewidth=1.2,
                    linestyle="--" if b["method"] == "linear_extrapolation" else ":", label=label(b["method"]))
    ax.set_xlabel("Horizon h (checks ahead)")
    ax.set_ylabel("Test RMSE after the last step (pp)")
    ax.set_xticks(hs)
    ax.legend(fontsize=6.5, loc="upper left")
    ax.set_title("Error by horizon, S6 stream (main seeds)", loc="left")
    fig.tight_layout()
    out = FIG / "d24_horizon_e2.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_spark_scaling() -> Path | None:
    path = RES / "d22_spark_scaling.json"
    if not path.exists():
        return None
    runs = [r for r in json.loads(path.read_text())["runs"] if r["status"] == "ok"]
    if len(runs) < 2:
        return None
    cores = [r["cores"] for r in runs]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.6))
    a1.plot(cores, [r["job_seconds"] / 60 for r in runs], marker="o", color=FAMILY_COLOUR["tft"], linewidth=2)
    a1.set_xlabel("Spark worker threads (local[N])")
    a1.set_ylabel("Wall time (min)")
    a1.set_xticks(cores)
    a1.set_ylim(bottom=0)
    a1.set_title(f"{runs[0]['blocks']} blocks, {runs[0]['cycles']:,} cycles", loc="left")
    a2.plot(cores, [r["speedup"] for r in runs], marker="o", color=FAMILY_COLOUR["tft"], linewidth=2, label="measured")
    a2.plot(cores, [c / cores[0] for c in cores], color=GREY, linestyle="--", label="ideal (linear)")
    a2.set_xlabel("Spark worker threads (local[N])")
    a2.set_ylabel(f"Speed-up vs {cores[0]} thread(s)")
    a2.set_xticks(cores)
    a2.legend(fontsize=7)
    a2.set_title("Single machine, local mode", loc="left")
    fig.tight_layout()
    out = FIG / "spark_scaling.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse
    global RES, TAB, FIG, PTAB
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=RES, help="folder with the results JSON files (default results/)")
    ap.add_argument("--out", type=Path, help="write tables and figures under this folder instead of the repository "
                                             "(testing; e.g. on smoke output)")
    args = ap.parse_args(argv)
    RES = args.results
    if args.out:
        TAB, FIG, PTAB = args.out / "tables", args.out / "figures", args.out / "paper_tables"
    for p in (TAB, FIG, PTAB):
        p.mkdir(parents=True, exist_ok=True)
    for exp in ("e1", "e2"):
        got = load(exp)
        if got is None:
            continue
        d, src = got
        t = tables(exp, d, src)
        outs = [fig_comparison(exp, d, t["comparison"]), fig_per_group(exp, t["per_group"]),
                fig_forgetting(exp, d, t["comparison"])]
        if exp == "e2":
            outs.append(fig_horizon(d))
        print(f"{exp} from {src}: " + ", ".join(o.name for o in outs), flush=True)
    s = fig_spark_scaling()
    if s:
        print(f"Spark scaling: {s.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
