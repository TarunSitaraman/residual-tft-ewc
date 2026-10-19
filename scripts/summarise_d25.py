"""Markdown summary of the D25 results files (for the run report; every number is read from the files).

Usage (from the repository root)::

    python scripts/summarise_d25.py                                  # results/forecasting_d25_e1.json, ..._e2.json
    python scripts/summarise_d25.py --out data/cache/d25/d25_summary.md
    python scripts/summarise_d25.py --results <dir>                  # e.g. smoke files under <data>/cache/d25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"tft_residual_ewc": "Residual TFT + EWC (D23 model, raw)",
          "tft_residual_ewc_shrunk": "Residual TFT + EWC, shrunk",
          "tft_pre_residual_ewc": "Pretrained residual TFT + EWC, raw",
          "tft_pre_residual_ewc_shrunk": "Pretrained residual TFT + EWC, shrunk",
          "linear_extrapolation": "Linear extrapolation",
          "linear_extrapolation_conformal": "Linear extrapolation (conformal intervals)",
          "persistence": "Persistence"}


def f(x, d=3) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def one(d: dict) -> list[str]:
    exp = d["experiment"].upper()
    out = [f"## D25 {exp} (horizons {d['horizons'][0]}–{d['horizons'][-1]}), profile {d['profile']}", "",
           f"- Commit {d['environment'].get('git_commit', '?')[:7]}, dirty: {d['environment'].get('git_dirty')}; "
           f"Python {d['environment'].get('python')}, torch {d['environment'].get('torch')}",
           f"- Parts run: {', '.join(d['parts_run']) or 'none'}; **parts not run: {', '.join(d['parts_not_run']) or 'none'}**",
           f"- D23 reproduction: {d['d23_reference_reproduction']}"]
    for k, t in d.get("tuning", {}).items():
        out.append(f"- Tuning ({k}): lr {t.get('lr')}, lambda {t.get('ewc_lambda')}"
                   + (f"; pool {t.get('pool_samples')} samples, {t.get('pool_cells')} cells" if k == "pre" else ""))
    runs = pd.DataFrame(d["runs"])
    out += ["", "| Method | Error main / rep. (pp) | Forgetting main / rep. | Interval score 90 % | Coverage 90 % |",
            "|---|---|---|---|---|"]
    for m, g in runs.groupby("method", sort=False):
        cell = lambda c: " / ".join(f(g[g["set"] == s][c].mean()) if c in g else "–" for s in ("main", "replicate"))
        out.append(f"| {LABELS.get(m, m)} | {cell('final_avg_error')} | {cell('forgetting')} | {cell('is90')} | "
                   f"{cell('cov90')} |")
    for b in d["baselines"]:
        out.append(f"| {LABELS.get(b['method'], b['method'])} | {f(b['final_avg_error'])} | | | |")
    alphas = [r["alpha_final"] for r in d["runs"] if isinstance(r.get("alpha_final"), dict)]
    if alphas:
        a = pd.DataFrame(alphas)
        out += ["", "Shrinkage factor after the last task, median over runs (both models): "
                + ", ".join(f"h{k} {f(a[k].median(), 2)}" for k in a.columns)]
    out += ["", "| Family | a − b (metric) | main / replicate | p_Holm | Verdict |", "|---|---|---|---|---|"]
    for v in d["verdicts"]:
        out.append(f"| {v['family']} | {LABELS.get(v['a'], v['a'])} − {LABELS.get(v['b'], v['b'])} ({v['metric']}) | "
                   f"{f(v['diff_main'])} / {f(v['diff_replicate'])} | {f(v['p_holm_main'])} / "
                   f"{f(v['p_holm_replicate'])} | {v['verdict']} |")
    return out + [""]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    lines = ["# D25 summary", ""]
    for exp in ("e1", "e2"):
        p = next((q for q in (args.results / f"forecasting_d25_{exp}.json", args.results / f"forecasting_d25_{exp}_smoke.json")
                  if q.exists()), None)
        lines += one(json.loads(p.read_text())) if p else [f"## D25 {exp.upper()}: no results file", ""]
    text = "\n".join(lines)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")       # the Windows console code page cannot print "–" or "−"
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
