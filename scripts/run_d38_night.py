"""D38 run for one dataset (decision D38, "Compute"): pre-flight, smoke, the tuning and evaluation parts, collection.

Same stages as ``run_d34_night.py`` (the D25 scheduler): pre-flight (clean tree, pinned libraries, the check table,
the committed condition sets and results of the base decision), smoke (every part in the smoke profile, not
reportable), paper parts with ``--workers`` processes, then ``run_d38.py --exp <dataset>_e1/e2 --collect-only``.
The parts run in the order tuning, r32, a0, r8 (E2 before E1 within each), so a run that is stopped still has the
primary arm; ``--no-r8`` leaves the small-memory arm out (the fallback fixed in D38). Every unit is checkpointed
under data/cache/d38_<dataset>, so re-running this script resumes where it stopped.

Logs: data/cache/d38_<dataset>/logs/ (git-ignored). Nothing is committed automatically.

    python scripts/run_d38_night.py --dataset s10 --dry-run
    python scripts/run_d38_night.py --dataset s10 --workers 6 --deadline 07:30
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_d25_night as base  # noqa: E402  (pre-flight, scheduler and logging of the D25 orchestrator)
from run_d38 import ARMS, DEPENDS, PARTS, RUNNERS  # noqa: E402

PY = sys.executable
EXPS = ("e2", "e1")                                   # E2 first (longer streams)


def plan(dataset: str, r8: bool = True) -> list[tuple[str, str]]:
    stages = ["tune"] + [a for a in ARMS if r8 or a != "r8"]
    return [(f"{dataset}_{exp}", p) for st in stages for exp in EXPS for p in PARTS if p.split("_")[0] == st]


def configure(dataset: str, r8: bool = True) -> None:
    cls = RUNNERS[dataset]
    base.LOGS = ROOT / "data" / "cache" / cls.NAME / "logs"
    base.PLAN, base.DEPENDS, base.DECISION = plan(dataset, r8), DEPENDS, "D38"
    base.RUNNER = ROOT / "scripts" / "run_d38.py"
    base.DATA_FILES = (cls.CHECKS,)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    cpu = os.cpu_count() or 2
    ap.add_argument("--dataset", choices=tuple(RUNNERS), required=True)
    ap.add_argument("--workers", type=int, default=min(6, max(1, cpu - 1)))
    ap.add_argument("--deadline", help="HH:MM local time after which no new part starts")
    ap.add_argument("--no-r8", action="store_true", help="leave out the small-memory arm (the D38 compute fallback)")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true", help="testing only: results would not be reportable")
    ap.add_argument("--allow-unpinned", action="store_true", help="run with library versions other than the pins")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    configure(args.dataset, not args.no_r8)
    cls = RUNNERS[args.dataset]
    deadline = base.deadline_from(args.deadline)
    base.say(f"D38 run on {args.dataset}: {len(base.PLAN)} parts, {args.workers} workers, deadline "
             f"{deadline or 'none'}; Python {sys.version.split()[0]}")
    problems = base.preflight(args.allow_dirty, args.allow_unpinned)
    for f in (f"{cls.BASE}_condition_sets.csv", f"forecasting_{cls.BASE}_e1.json", f"forecasting_{cls.BASE}_e2.json"):
        if not (ROOT / "results" / f).exists():
            problems.append(f"results/{f} missing (committed {cls.BASE.upper()} sets and results are reused by D38)")
    for p in problems:
        base.say(f"PRE-FLIGHT: {p}")
    if problems:
        return 2
    base.say("pre-flight passed")
    for i, (exp, part) in enumerate(base.PLAN, 1):
        base.say(f"  {i:2d}. {exp} {part}" + (f" (after {DEPENDS[part]})" if part in DEPENDS else ""))
    if args.dry_run:
        return 0
    if not args.skip_smoke:
        failed = [j for j, c in base.run_parts(base.PLAN, "smoke", args.workers, None).items() if c != 0]
        if failed:
            base.say(f"SMOKE FAILED for {failed}; see {base.LOGS}; stopping before the paper run")
            return 1
        base.say("smoke passed for every part (not reportable)")
    paper = base.run_parts(base.PLAN, "paper", args.workers, deadline)
    for e in ("e1", "e2"):
        exp = f"{args.dataset}_{e}"
        if any(paper.get((exp, p)) == 0 for p in PARTS if p.startswith("r32_")):
            base.step([PY, str(ROOT / "scripts" / "run_d38.py"), "--exp", exp, "--collect-only"], f"collect_{exp}")
        else:
            base.say(f"collect_{exp}: skipped (no r32 part finished)")
    base.say(f"done. Review {base.LOGS / 'night.log'}, then commit the results files and record the run in the D38 "
             "run log (docs/DECISIONS.md). If parts did not finish, re-run this script with --skip-smoke to resume.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
