"""D31 run (decision D31, "Compute"): pre-flight, the 14 fold parts, collection and the verdicts.

Stages
1. Pre-flight: clean working tree (D9); the S6 check table exists; versions match ``requirements.txt`` (a mismatch
   stops the run unless ``--allow-unpinned`` is given, and is then written to the log).
2. Paper parts (``f<k>`` for E2 then E1, heaviest folds first), ``--workers`` processes at a time (each
   single-threaded). No part starts after ``--deadline``. Every unit is checkpointed, so re-running this script
   resumes where it stopped. There is no real-data smoke stage: D31's update grid is fixed, so a smoke pass would
   cost hours; ``tests/test_d31.py`` runs every code path on synthetic data.
3. Collection: ``run_d31.py --exp e1/e2`` checks D26 and D28 and writes results/forecasting_d31_<exp>.json;
   ``run_d31.py --verdicts`` prints Q1-Q3 when both exist.

Run it in a terminal of its own (not a background shell of another tool); the estimate is about 21 h with six
workers (D31 run log). Logs: data/cache/d31/logs/ (git-ignored). Nothing is committed automatically.

Usage (from the repository root)::

    python scripts/run_d31_night.py --dry-run
    python scripts/run_d31_night.py --workers 6
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_d25_night as base  # noqa: E402  (pre-flight, scheduler and logging of the D25 orchestrator)

LOGS = ROOT / "data" / "cache" / "d31" / "logs"
PY = sys.executable
# Heaviest folds first (adapted-pass mini-batches on the real frames: case6, case4, case5, case7, case2, case1, case3).
FOLDS = ("f5", "f3", "f4", "f6", "f1", "f0", "f2")
PLAN = [(exp, part) for exp in ("e2", "e1") for part in FOLDS]


def configure() -> None:
    base.LOGS, base.PLAN, base.DEPENDS, base.DECISION = LOGS, PLAN, {}, "D31"
    base.RUNNER = ROOT / "scripts" / "run_d31.py"
    base.DATA_FILES = ("wheeler_checks.csv",)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    cpu = os.cpu_count() or 2
    ap.add_argument("--workers", type=int, default=min(6, max(1, cpu - 1)),
                    help="parallel parts (single-threaded each); 6 leaves memory headroom on the ASUS")
    ap.add_argument("--deadline", help="HH:MM local time after which no new part starts")
    ap.add_argument("--allow-dirty", action="store_true", help="testing only: results would not be reportable")
    ap.add_argument("--allow-unpinned", action="store_true", help="run with library versions other than the pins")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    configure()
    deadline = base.deadline_from(args.deadline)
    base.say(f"D31 run: {len(PLAN)} parts, {args.workers} workers, deadline {deadline or 'none'}; "
             f"Python {sys.version.split()[0]}")
    problems = base.preflight(args.allow_dirty, args.allow_unpinned)
    for p in problems:
        base.say(f"PRE-FLIGHT: {p}")
    if problems:
        return 2
    base.say("pre-flight passed")
    for i, (exp, part) in enumerate(PLAN, 1):
        base.say(f"  {i:2d}. {exp} {part}")
    if args.dry_run:
        return 0
    paper = base.run_parts(PLAN, "paper", args.workers, deadline)
    collected = []
    for exp in ("e1", "e2"):
        if all(paper.get((exp, p)) == 0 for p in FOLDS):
            if base.step([PY, str(ROOT / "scripts" / "run_d31.py"), "--exp", exp], f"collect_{exp}") == 0:
                collected.append(exp)
        else:
            base.say(f"collect_{exp}: skipped (not every fold finished; re-run this script to resume)")
    if len(collected) == 2:
        base.step([PY, str(ROOT / "scripts" / "run_d31.py"), "--verdicts"], "verdicts")
    base.say("done. Review data/cache/d31/logs/, then commit results/forecasting_d31_e*.json and record the run in "
             "the D31 run log (docs/DECISIONS.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
