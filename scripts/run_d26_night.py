"""One-night D26 run (decision D26, "Compute budget"): pre-flight, smoke, the 28 parts, collection and the summary.

Stages
1. Pre-flight: clean working tree (D9); the S6 check table and the committed D23 results (tuning source) exist; the
   libraries import; versions match ``requirements.txt`` (a mismatch stops the run unless ``--allow-unpinned`` is
   given, and is then written to the log: record it in the D26 run log).
2. Smoke: every part once in the smoke profile (not reportable; nothing written to results/). ``--skip-smoke``
   skips it.
3. Paper parts (``<set>_f<k>`` for E2 then E1), ``--workers`` processes at a time (each single-threaded). No part
   starts after ``--deadline``. Every unit is checkpointed, so re-running this script resumes where it stopped.
4. Collection: ``run_d26.py --exp e1/e2 --collect-only`` writes results/forecasting_d26_<exp>.json.
5. ``summarise_d26.py`` writes the morning summary to data/cache/d26/d26_summary.md.

Logs: data/cache/d26/logs/ (git-ignored). Nothing is committed automatically.

Usage (from the repository root, e.g. in PowerShell on the ASUS)::

    python scripts/run_d26_night.py --dry-run
    python scripts/run_d26_night.py --workers 6 --deadline 07:30
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_d25_night as base  # noqa: E402  (pre-flight, scheduler and logging of the D25 orchestrator)
from run_d26 import PARTS  # noqa: E402

LOGS = ROOT / "data" / "cache" / "d26" / "logs"
PY = sys.executable
# E2 first: its streams are longer, so the critical path starts at once.
PLAN = [(exp, part) for exp in ("e2", "e1") for part in PARTS]
DEPENDS: dict[str, str] = {}


def configure() -> None:
    """Point the D25 scheduler at this decision's logs, plan, runner and data (D26 needs no pool tables)."""
    base.LOGS, base.PLAN, base.DEPENDS, base.DECISION = LOGS, PLAN, DEPENDS, "D26"
    base.RUNNER = ROOT / "scripts" / "run_d26.py"
    base.DATA_FILES = ("wheeler_checks.csv",)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    cpu = os.cpu_count() or 2
    ap.add_argument("--workers", type=int, default=min(6, max(1, cpu - 1)),
                    help="parallel parts (single-threaded each); 6 leaves memory headroom on the ASUS")
    ap.add_argument("--deadline", help="HH:MM local time after which no new part starts")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true", help="testing only: results would not be reportable")
    ap.add_argument("--allow-unpinned", action="store_true", help="run with library versions other than the pins")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    configure()
    deadline = base.deadline_from(args.deadline)
    base.say(f"D26 night run: {len(PLAN)} parts, {args.workers} workers, deadline {deadline or 'none'}; "
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
    if not args.skip_smoke:
        smoke = base.run_parts(PLAN, "smoke", args.workers, None)
        failed = [j for j, c in smoke.items() if c != 0]
        if failed:
            base.say(f"SMOKE FAILED for {failed}; see {LOGS}; stopping before the paper run")
            return 1
        base.say("smoke passed for every part (not reportable)")
    paper = base.run_parts(PLAN, "paper", args.workers, deadline)
    collected = []
    for exp in ("e1", "e2"):
        if any(paper.get((exp, p)) == 0 for p in PARTS):
            if base.step([PY, str(ROOT / "scripts" / "run_d26.py"), "--exp", exp, "--collect-only"],
                         f"collect_{exp}") == 0:
                collected.append(exp)
        else:
            base.say(f"collect_{exp}: skipped (no part finished)")
    if collected:
        base.step([PY, str(ROOT / "scripts" / "summarise_d26.py"), "--out", str(LOGS.parent / "d26_summary.md")],
                  "summary")
    base.say("done. Review data/cache/d26/logs/night.log and data/cache/d26/d26_summary.md, then commit the results "
             "files and record the run in the D26 run log (docs/DECISIONS.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
