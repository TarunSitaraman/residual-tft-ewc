"""Markdown summary of the D26 results files (for the run report; every number is read from the files).

Usage (from the repository root)::

    python scripts/summarise_d26.py                                  # results/forecasting_d26_e1.json, ..._e2.json
    python scripts/summarise_d26.py --out data/cache/d26/d26_summary.md
    python scripts/summarise_d26.py --results <dir>                  # e.g. smoke files under <data>/cache/d26
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"tft_residual_ewc": "Residual TFT + EWC, raw",
          "tft_residual_ewc_shrunk": "Residual TFT + EWC, shrunk",
          "tft_residual_joint": "Residual TFT, joint",
          "linear_extrapolation": "Linear extrapolation",
          "linear_extrapolation_conformal": "Linear extrapolation (conformal intervals)",
          "gbm_shared": "Gradient boosting",
          "ridge_shared": "Ridge",
          "persistence": "Persistence"}


def f(x, d=3) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def one(d: dict) -> list[str]:
    env = d["environment"]
    out = [f"## D26 {d['experiment'].upper()} (horizons {d['horizons'][0]}–{d['horizons'][-1]}), profile {d['profile']}",
           "",
           f"- Commit {env.get('git_commit', '?')[:7]}, dirty: {env.get('git_dirty')}; Python {env.get('python')}, "
           f"torch {env.get('torch')}",
           f"- Parts run: {len(d['parts_run'])}; **parts not run: {', '.join(d['parts_not_run']) or 'none'}**",
           f"- Folds in the tests: {', '.join(d['folds_in_tests'])}",
           f"- Tuning: lr {d['tuning']['lr']}, lambda {d['tuning']['ewc_lambda']} ({d['tuning']['source']})",
           "- Strata (n / H+ / H−): " + "; ".join(f"{c} {v['n']}/{v['hplus']}/{v['hminus']}"
                                                  for c, v in d["strata_counts"].items()),
           f"- Coverage check (90 %, threshold {d['coverage_check']['threshold']}): "
           f"{d['coverage_check']['mean_cov90']} → adequate: {d['coverage_check']['adequate']}"]
    runs = pd.DataFrame(d["runs"])
    out += ["", "| Method | P2 error all main / rep. (pp) | H+ | H− | IS 90 % | Coverage 90 % |",
            "|---|---|---|---|---|---|"]
    for m, g in runs.groupby("method", sort=False):
        cell = lambda c: " / ".join(f(g[g["set"] == s][c].mean()) if c in g else "–" for s in ("main", "replicate"))
        out.append(f"| {LABELS.get(m, m)} | {cell('p2_error')} | {cell('p2_error_hplus')} | {cell('p2_error_hminus')} | "
                   f"{cell('is90')} | {cell('cov90')} |")
    for b in d["baselines"]:
        out.append(f"| {LABELS.get(b['method'], b['method'])} | {f(b['p2_error'])} | {f(b['p2_error_hplus'])} | "
                   f"{f(b['p2_error_hminus'])} | | |")
    out += ["", "Per fold (all origins), shrunk model mean over runs and the single-fit comparators:", "",
            "| Fold | Shrunk | " + " | ".join(LABELS.get(b["method"], b["method"]) for b in d["baselines"]) + " |",
            "|---|---|" + "---|" * len(d["baselines"])]
    shr = runs[runs["method"] == "tft_residual_ewc_shrunk"]
    for c in d["folds_in_tests"]:
        v = np.mean([pf[c] for pf in shr["per_fold"] if c in pf]) if len(shr) else np.nan
        out.append(f"| {c} | {f(v)} | " + " | ".join(f(b["per_fold"].get(c)) for b in d["baselines"]) + " |")
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
    lines = ["# D26 summary", ""]
    for exp in ("e1", "e2"):
        p = next((q for q in (args.results / f"forecasting_d26_{exp}.json", args.results / f"forecasting_d26_{exp}_smoke.json")
                  if q.exists()), None)
        lines += one(json.loads(p.read_text(encoding="utf-8"))) if p else [f"## D26 {exp.upper()}: no results file", ""]
    text = "\n".join(lines)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")       # the Windows console code page cannot print "−" (D25 run log)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
