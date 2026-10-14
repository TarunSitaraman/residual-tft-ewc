"""One-night D25 run (decision D25, "Compute budget and run plan"): pre-flight, smoke, parts with their
dependencies, collection and the summary.

Stages
1. Pre-flight: clean working tree (D9); S6 check table, the three pool check tables and the committed D23 results
   exist; the libraries import. The library versions are compared with ``requirements.txt``: a mismatch stops the
   run unless ``--allow-unpinned`` is given, and is then written to the log (record it in the D25 run log).
2. Smoke: every part once in the smoke profile (not reportable; nothing written to results/). ``--skip-smoke``
   skips it.
3. Paper parts, ``--workers`` processes at a time (each single-threaded), in the order below; a part whose
   dependency has not finished waits, and one whose dependency failed is not started. No part starts after
   ``--deadline``. Every unit is checkpointed, so re-running this script resumes where it stopped.
4. Collection: ``run_d25.py --exp e1/e2 --collect-only`` writes results/forecasting_d25_<exp>.json.
5. ``summarise_d25.py`` writes the morning summary to data/cache/d25/d25_summary.md.

Logs: data/cache/d25/logs/ (git-ignored). Nothing is committed automatically.

Usage (from the repository root, e.g. in PowerShell on the ASUS)::

    python scripts/run_d25_night.py --dry-run
    python scripts/run_d25_night.py --workers 6 --deadline 07:30
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "data" / "cache" / "d25" / "logs"
PY = sys.executable

# Longest first, so the critical path (pool tuning, then pretrained evaluation) starts at once.
PLAN = [("e2", "pre_tune"), ("e1", "pre_tune"),
        ("e2", "scratch_main"), ("e2", "scratch_replicate"), ("e1", "scratch_main"), ("e1", "scratch_replicate"),
        ("e2", "pre_main"), ("e2", "pre_replicate"), ("e1", "pre_main"), ("e1", "pre_replicate")]
DEPENDS = {"pre_main": "pre_tune", "pre_replicate": "pre_tune"}
PINNED = ("numpy", "pandas", "scikit-learn", "scipy", "torch")


def now() -> str:
    return dt.datetime.now().strftime("%H:%M:%S")


def say(msg: str) -> None:
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / "night.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def deadline_from(text: str | None) -> dt.datetime | None:
    if not text:
        return None
    h, m = (int(x) for x in text.split(":"))
    d = dt.datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
    return d if d > dt.datetime.now() else d + dt.timedelta(days=1)


def version_mismatches() -> list[str]:
    pins = dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^\s#]+)", (ROOT / "requirements.txt").read_text(), re.M))
    out = []
    if sys.version_info[:2] != (3, 11):
        out.append(f"Python {sys.version.split()[0]} (pinned 3.11)")
    for name in PINNED:
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            have = "missing"
        if name in pins and have.split("+")[0] != pins[name]:
            out.append(f"{name} {have} (pinned {pins[name]})")
    return out


def preflight(allow_dirty: bool, allow_unpinned: bool) -> list[str]:
    problems = []
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                            capture_output=True, text=True)
    if status.returncode != 0:
        problems.append("git status failed (is this the repository root?)")
    elif status.stdout.strip() and not allow_dirty:
        problems.append("tracked files differ from the commit; commit or stash first (D9):\n" + status.stdout)
    data = Path(os.environ.get("DATA_ROOT", ROOT / "data"))
    data = data if data.is_absolute() else ROOT / data
    for f in ("wheeler_checks.csv", "pool_s7_checks.csv", "pool_s8_checks.csv", "pool_s9_checks.csv"):
        if not (data / f).exists():
            problems.append(f"{data / f} missing (build_wheeler_features.py / build_pool_tables.py)")
    for f in ("forecasting_d23_e1.json", "forecasting_d23_e2.json"):
        if not (ROOT / "results" / f).exists():
            problems.append(f"results/{f} missing (D23 reference runs)")
    try:
        import sklearn  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        problems.append(f"library missing: {e}")
    mism = version_mismatches()
    if mism:
        say("environment differs from requirements.txt: " + "; ".join(mism))
        if not allow_unpinned:
            problems.append("library versions differ from requirements.txt (use the pinned .venv, or pass "
                            "--allow-unpinned and record the deviation in the D25 run log)")
    return problems


def run_parts(jobs: list[tuple[str, str]], profile: str, workers: int, deadline: dt.datetime | None) -> dict:
    """Run ``run_d25.py --part`` for ``jobs`` with at most ``workers`` processes, respecting ``DEPENDS``."""
    queue, running, result = list(jobs), {}, {}
    LOGS.mkdir(parents=True, exist_ok=True)
    status_file = LOGS / f"status_{profile}.json"
    while queue or running:
        for job in list(queue):
            if len(running) >= workers:
                break
            exp, part = job
            dep = DEPENDS.get(part)
            if dep and (exp, dep) in jobs:
                if (exp, dep) not in result:
                    continue                                   # dependency still queued or running
                if result[(exp, dep)] != 0:
                    result[job] = f"not started ({dep} did not finish)"
                    queue.remove(job)
                    say(f"{profile}: {exp} {part} not started ({dep} did not finish)")
                    continue
            if deadline and dt.datetime.now() >= deadline:
                for j in queue:
                    result[j] = "not started (deadline)"
                    say(f"{profile}: {j[0]} {j[1]} not started (deadline reached)")
                queue = []
                break
            queue.remove(job)
            log = open(LOGS / f"{profile}_{exp}_{part}.log", "w", encoding="utf-8")
            cmd = [PY, str(ROOT / "scripts" / "run_d25.py"), "--exp", exp, "--part", part, "--profile", profile]
            running[job] = (subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT), log, time.time())
            say(f"{profile}: started {exp} {part} ({len(running)} running, {len(queue)} queued)")
        time.sleep(5 if profile == "smoke" else 30)
        for job, (proc, log, t0) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            result[job] = code
            del running[job]
            say(f"{profile}: {'finished' if code == 0 else f'FAILED (exit {code})'} {job[0]} {job[1]} "
                f"in {(time.time() - t0) / 60:.1f} min")
        status_file.write_text(json.dumps({
            "updated": now(), "running": [f"{e} {p}" for e, p in running], "queued": [f"{e} {p}" for e, p in queue],
            "done": {f"{e} {p}": c for (e, p), c in result.items()}}, indent=1))
    return result


def step(cmd: list[str], name: str) -> int:
    log = LOGS / f"{name}.log"
    say(f"{name}: {' '.join(Path(c).name if i == 1 else c for i, c in enumerate(cmd))}")
    with open(log, "w", encoding="utf-8") as f:
        code = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    say(f"{name}: {'ok' if code == 0 else f'FAILED (exit {code}); see {log}'}")
    return code


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
    deadline = deadline_from(args.deadline)
    say(f"D25 night run: {len(PLAN)} parts, {args.workers} workers, deadline {deadline or 'none'}; "
        f"Python {sys.version.split()[0]}")
    problems = preflight(args.allow_dirty, args.allow_unpinned)
    for p in problems:
        say(f"PRE-FLIGHT: {p}")
    if problems:
        return 2
    say("pre-flight passed")
    for i, (exp, part) in enumerate(PLAN, 1):
        say(f"  {i:2d}. {exp} {part}" + (f" (after {DEPENDS[part]})" if part in DEPENDS else ""))
    if args.dry_run:
        return 0
    if not args.skip_smoke:
        smoke = run_parts(PLAN, "smoke", args.workers, None)
        failed = [j for j, c in smoke.items() if c != 0]
        if failed:
            say(f"SMOKE FAILED for {failed}; see {LOGS}; stopping before the paper run")
            return 1
        say("smoke passed for every part (not reportable)")
    paper = run_parts(PLAN, "paper", args.workers, deadline)
    collected = []
    for exp in ("e1", "e2"):
        if any(paper.get((exp, p)) == 0 for p in ("scratch_main", "pre_main")):
            if step([PY, str(ROOT / "scripts" / "run_d25.py"), "--exp", exp, "--collect-only"], f"collect_{exp}") == 0:
                collected.append(exp)
        else:
            say(f"collect_{exp}: skipped (no evaluation part finished)")
    if collected:
        step([PY, str(ROOT / "scripts" / "summarise_d25.py"), "--out", str(LOGS.parent / "d25_summary.md")], "summary")
    say("done. Review data/cache/d25/logs/night.log and data/cache/d25/d25_summary.md, then commit the results files "
        "and record the run in the D25 run log (docs/DECISIONS.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
