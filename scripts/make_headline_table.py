"""Headline table of the paper: the proposed model against gradient boosting and linear extrapolation in both S6
settings (continual stream P1; unseen usage condition P2), with the verdicts of the registered tests.

Sources (committed results files): D25 (P1, shrunk residual TFT + EWC and extrapolation), D23 (P1 gradient
boosting), D26 (P2), D32 (P2, gradient boosting under the continual constraint). The P1 test of the shrunk model
against gradient boosting was not registered (D23 registered it for the unshrunk model, replicated in favour of the
TFT); it is computed here with the same rule and marked post hoc. Writes paper/tables/headline.tex.

    python scripts/make_headline_table.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_d25  # noqa: E402
from src import evaluation as ev  # noqa: E402

SHR = "tft_residual_ewc_shrunk"
EXPS = ("e1", "e2")
SHORT = {"replicated, a better": "better", "replicated, a worse": "worse",
         "practically equivalent": "equiv.", "not replicated": "n.r."}


def load(name: str, exp: str) -> dict:
    return json.loads((ROOT / "results" / f"forecasting_{name}_{exp}.json").read_text())


def by_method(rows: list[dict], key: str) -> dict:
    return {r["method"]: r[key] for r in rows}


def verdict(rows: list[dict], a: str, b: str, metric: str) -> dict:
    v = next(x for x in rows if x["a"] == a and x["b"] == b and x["metric"] == metric)
    return {"diff": (v["diff_main"], v["diff_replicate"]), "verdict": v["verdict"]}


def p1_vs_gbm_posthoc(d25: dict, gbm: float) -> dict:
    """Shrunk runs minus the single gradient-boosting value, D21 rule (one test per seed set, so p_Holm = p)."""
    res = {}
    for s in ("main", "replicate"):
        va = np.array([r["final_avg_error"] for r in d25["runs"] if r["method"] == SHR and r["set"] == s], float)
        c = ev.paired_comparison(va, np.full(len(va), gbm))
        res[s] = {**c, "p_holm": c["p_value"]}
    return {"diff": (res["main"]["mean_diff"], res["replicate"]["mean_diff"]),
            "verdict": run_d25.verdict(res["main"], res["replicate"]), "post_hoc": True}


def collect() -> dict:
    out = {}
    for e in EXPS:
        d23, d25, d26, d32 = (load(n, e) for n in ("d23", "d25", "d26", "d32"))
        s25 = {(r["set"], r["method"]): r["final_avg_error_mean"] for r in d25["summary"]}
        s26 = {(r["set"], r["method"]): r["p2_error_mean"] for r in d26["summary"]}
        b23, b25, b26 = (by_method(x["baselines"], k) for x, k in
                         ((d23, "final_avg_error"), (d25, "final_avg_error"), (d26, "p2_error")))
        out[e] = {
            "p1": {"proposed": (s25[("main", SHR)], s25[("replicate", SHR)]), "lin": b25["linear_extrapolation"],
                   "gbm": b23["gbm_shared"], "gbm_last": None, "persistence": b25["persistence"],
                   "v_lin": verdict(d25["verdicts"], SHR, "linear_extrapolation", "final_avg_error"),
                   "v_gbm": p1_vs_gbm_posthoc(d25, b23["gbm_shared"]), "v_gbm_last": None},
            "p2": {"proposed": (s26[("main", SHR)], s26[("replicate", SHR)]), "lin": b26["linear_extrapolation"],
                   "gbm": b26["gbm_shared"], "gbm_last": d32["summary"]["gbm_last"]["p2_error"],
                   "persistence": b26["persistence"],
                   "v_lin": verdict(d26["verdicts"], SHR, "linear_extrapolation", "p2_error"),
                   "v_gbm": verdict(d26["verdicts"], SHR, "gbm_shared", "p2_error"),
                   "v_gbm_last": verdict(d32["verdicts"], SHR, "gbm_last", "p2_error")},
        }
    return out


def cells(data: dict, key: str) -> list[str]:
    cols = [(p, e) for p in ("p1", "p2") for e in EXPS]
    vals = []
    for p, e in cols:
        v = data[e][p][key]
        if v is None:
            vals.append("--")
            continue
        best = min(x for k in ("proposed", "lin", "gbm", "gbm_last")
                   if (x := (np.mean(data[e][p][k]) if data[e][p][k] is not None else None)) is not None)
        txt = f"{v[0]:.2f}/{v[1]:.2f}" if isinstance(v, tuple) else f"{v:.2f}"
        vals.append(f"\\textbf{{{txt}}}" if abs(np.mean(v) - best) < 5e-3 else txt)
    return vals


def test_cells(data: dict, key: str) -> list[str]:
    out = []
    for p in ("p1", "p2"):
        for e in EXPS:
            v = data[e][p][key]
            if v is None:
                out.append("--")
                continue
            mark = "$^\\ast$" if v.get("post_hoc") else ""
            out.append(f"${v['diff'][0]:+.2f}$/${v['diff'][1]:+.2f}$ {SHORT[v['verdict']]}{mark}")
    return out


def table(data: dict) -> str:
    rows = [("\\textbf{Residual TFT-EWC} (proposed)", "proposed"), ("Linear extrapolation", "lin"),
            ("Gradient boosting, all conditions stored", "gbm"),
            ("Gradient boosting, latest condition only$^\\dagger$", "gbm_last"), ("Persistence", "persistence")]
    tests = [("vs.\\ linear extrapolation", "v_lin"), ("vs.\\ gradient boosting, all stored", "v_gbm"),
             ("vs.\\ gradient boosting, latest only$^\\dagger$", "v_gbm_last")]
    body = [f"    {lab} & " + " & ".join(cells(data, k)) + " \\\\" for lab, k in rows]
    tbody = [f"    {lab} & " + " & ".join(test_cells(data, k)) + " \\\\" for lab, k in tests]
    return "\n".join([
        "% Generated by scripts/make_headline_table.py from results/forecasting_d23/d25/d26/d32_e*.json",
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\caption{The proposed model against the two strongest traditional forecasters on S6, in both settings. Top:",
        "    test RMSE (pp); proposed model: mean over 50 runs, main/replicate seed sets; others: single deterministic fits",
        "    (bold: best). Bottom: proposed minus baseline (pp, main/replicate; negative favours the proposed model) and the",
        "    verdict of the replication rule (better/worse: significant after Holm in both seed sets, same sign, at least",
        "    0.05~pp; equiv.: below 0.05~pp in both; n.r.: not replicated). $^\\ast$Post hoc for the shrunk model; registered",
        "    for the unshrunk model (D23), with the same verdict. $^\\dagger$Under the continual constraint of the proposed",
        "    model (conditions one at a time, earlier data not kept); P2 only (D32).}",
        "  \\label{tab:headline}",
        "  \\setlength{\\tabcolsep}{4pt}",
        "  \\footnotesize",
        "  \\begin{tabular}{lcccc}",
        "    \\toprule",
        "    & \\multicolumn{2}{c}{Continual stream (P1)} & \\multicolumn{2}{c}{Unseen usage condition (P2)} \\\\",
        "    \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "    & $h{=}1$--3 & $h{=}1$--6 & $h{=}1$--3 & $h{=}1$--6 \\\\",
        "    \\midrule",
        *body,
        "    \\midrule",
        "    \\multicolumn{5}{l}{\\emph{Proposed minus baseline, verdict}} \\\\",
        *tbody,
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table*}",
        "",
    ])


def main() -> int:
    data = collect()
    (ROOT / "paper" / "tables" / "headline.tex").write_text(table(data), encoding="utf-8")
    print(table(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
