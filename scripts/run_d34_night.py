"""D34 run (decision D34, "Compute budget"): pre-flight, smoke, the tuning and evaluation parts, collection.

Same stages as ``run_d26_night.py`` (the D25 scheduler): pre-flight (clean tree, pinned libraries, the S7 check table),
smoke (every part in the smoke profile, not reportable), paper parts with ``--workers`` processes and the
dependencies tune1 -> tune2 -> evaluation per fold, then ``run_d34.py --exp e1/e2 --collect-only``. Every unit is
checkpointed under data/cache/d34, so re-running this script resumes where it stopped (the plan spans two nights).

Logs: data/cache/d34/logs/ (git-ignored). Nothing is committed automatically.

    python scripts/run_d34_night.py --dry-run
    python scripts/run_d34_night.py --workers 6 --deadline 07:30
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_d25_night as base  # noqa: E402  (pre-flight, scheduler and logging of the D25 orchestrator)
from run_d34 import DEPENDS, PARTS  # noqa: E402

LOGS = ROOT / "data" / "cache" / "d34" / "logs"
PY = sys.executable
# E2 first (longer streams); within an experiment the parts are listed tune1, tune2, evaluation, so the scheduler
# starts every fold's tuning at once and evaluation parts as soon as their fold's tuning is done.
PLAN = [(exp, part) for exp in ("e2", "e1") for part in PARTS]


def configure() -> None:
    base.LOGS, base.PLAN, base.DEPENDS, base.DECISION = LOGS, PLAN, DEPENDS, "D34"
    base.RUNNER = ROOT / "scripts" / "run_d34.py"
    base.DATA_FILES = ("pool_s7_checks.csv",)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    cpu = os.cpu_count() or 2
    ap.add_argument("--workers", type=int, default=min(6, max(1, cpu - 1)))
    ap.add_argument("--deadline", help="HH:MM local time after which no new part starts")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true", help="testing only: results would not be reportable")
    ap.add_argument("--allow-unpinned", action="store_true", help="run with library versions other than the pins")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    configure()
    deadline = base.deadline_from(args.deadline)
    base.say(f"D34 run: {len(PLAN)} parts, {args.workers} workers, deadline {deadline or 'none'}; "
             f"Python {sys.version.split()[0]}")
    problems = base.preflight(args.allow_dirty, args.allow_unpinned)
    if not (ROOT / "results" / "d34_condition_sets.csv").exists():
        problems.append("results/d34_condition_sets.csv missing (run_d34.py --write-sets, then commit it)")
    for p in problems:
        base.say(f"PRE-FLIGHT: {p}")
    if problems:
        return 2
    base.say("pre-flight passed")
    for i, (exp, part) in enumerate(PLAN, 1):
        base.say(f"  {i:2d}. {exp} {part}" + (f" (after {DEPENDS[part]})" if part in DEPENDS else ""))
    if args.dry_run:
        return 0
    if not args.skip_smoke:
        failed = [j for j, c in base.run_parts(PLAN, "smoke", args.workers, None).items() if c != 0]
        if failed:
            base.say(f"SMOKE FAILED for {failed}; see {LOGS}; stopping before the paper run")
            return 1
        base.say("smoke passed for every part (not reportable)")
    paper = base.run_parts(PLAN, "paper", args.workers, deadline)
    for exp in ("e1", "e2"):
        if any(paper.get((exp, p)) == 0 for p in PARTS if not p.startswith("tune")):
            base.step([PY, str(ROOT / "scripts" / "run_d34.py"), "--exp", exp, "--collect-only"], f"collect_{exp}")
        else:
            base.say(f"collect_{exp}: skipped (no evaluation part finished)")
    base.say("done. Review data/cache/d34/logs/night.log, then commit the results files and record the run in the "
             "D34 run log (docs/DECISIONS.md). If parts did not finish, re-run this script with --skip-smoke to resume.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
