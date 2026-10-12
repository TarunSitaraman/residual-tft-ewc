"""One-night D24 run (decision D24, "Compute budget and run plan"): pre-flight, smoke, parts in priority order,
collection, optional Spark scaling run, tables and figures.

Stages
1. Pre-flight: the working tree is clean (results must trace to a commit, D9), the S6 check table exists, the
   libraries import, and (with ``--spark-scaling``) Java, PySpark and the raw ageing files are present.
2. Smoke: every part once in the smoke profile (1 order, 2 epochs; not reportable; nothing written to results/),
   so a crash shows up in minutes, not in the morning. ``--skip-smoke`` skips it.
3. Paper parts in the registered priority order, ``--workers`` processes at a time (each single-threaded). No part
   starts after ``--deadline``; a part that has started runs to the end. Every unit is checkpointed, so re-running
   this script resumes where it stopped.
4. Collection: ``run_d24.py --exp e1/e2 --collect-only`` writes results/forecasting_d24_<exp>.json from the finished
   parts and lists the others as not run.
5. Only with ``--spark-scaling`` and before the deadline: ``run_spark_scaling.py`` on the now idle machine.
6. ``make_d24_outputs.py``: comparison, statistics and per-group tables and the figures.

Logs and a status file are written to data/cache/d24/logs/ (git-ignored). Nothing is committed automatically:
review the outputs, then commit them yourself (the script prints the command).

Usage (from the repository root, e.g. in PowerShell on the ASUS)::

    python scripts/run_d24_night.py --workers 8 --deadline 07:30 --spark-scaling
    python scripts/run_d24_night.py --dry-run          # show the plan and the pre-flight result only
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

PRIORITY = [  # D24 run plan, steps 1-3 (baselines moved into the parallel phase by the D24 amendment)
    ("e1", "tft_residual_saewc"), ("e1", "lstm_residual"), ("e1", "tcn_residual"), ("e1", "transformer_residual"),
    ("e2", "tft_residual_saewc"), ("e2", "lstm_residual"), ("e2", "tcn_residual"), ("e2", "transformer_residual"),
    ("e1", "baselines"), ("e2", "baselines"),   # D24 amendment: tabular baselines in the parallel phase
    ("e1", "lstm_direct"), ("e1", "tcn_direct"), ("e1", "transformer_direct"),
]
LOGS = ROOT / "data" / "cache" / "d24" / "logs"
PY = sys.executable


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


def preflight(spark: bool, allow_dirty: bool) -> list[str]:
    problems = []
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                            capture_output=True, text=True)
    if status.returncode != 0:
        problems.append("git status failed (is this the repository root?)")
    elif status.stdout.strip() and not allow_dirty:
        problems.append("tracked files differ from the commit; commit or stash first (results must trace to a "
                        "clean commit, D9):\n" + status.stdout)
    data = Path(os.environ.get("DATA_ROOT", ROOT / "data"))
    data = data if data.is_absolute() else ROOT / data
    if not (data / "wheeler_checks.csv").exists():
        problems.append(f"{data / 'wheeler_checks.csv'} missing: run scripts/build_wheeler_features.py first")
    for f in ("forecasting_d23_e1.json", "forecasting_d23_e2.json"):
        if not (ROOT / "results" / f).exists():
            problems.append(f"results/{f} missing (D23 reference runs)")
    try:
        import sklearn  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        problems.append(f"library missing: {e}")
    if spark:
        try:
            import pyspark  # noqa: F401
        except ImportError:
            problems.append("--spark-scaling: pyspark not installed")
        if not (os.environ.get("JAVA_HOME") or shutil.which("java")):
            problems.append("--spark-scaling: Java 17/21 not found (JAVA_HOME)")
        raw = ROOT / "data" / "raw" / "wheeler2025"
        if not any(raw.glob("Cell*/agingdata_50degC/*aging*.csv.*")):
            problems.append(f"--spark-scaling: no raw ageing files under {raw}")
        if not (ROOT / "data" / "s6_ageing_blocks.csv").exists():
            problems.append("--spark-scaling: data/s6_ageing_blocks.csv (pandas reference) missing")
    return problems


def run_parts(jobs: list[tuple[str, str]], profile: str, workers: int, deadline: dt.datetime | None) -> dict:
    """Run ``run_d24.py --part`` for ``jobs`` with at most ``workers`` processes; returns {job: exit code or note}."""
    queue, running, result = list(jobs), {}, {}
    status_file = LOGS / f"status_{profile}.json"
    while queue or running:
        while queue and len(running) < workers:
            if deadline and dt.datetime.now() >= deadline:
                for job in queue:
                    result[job] = "not started (deadline)"
                    say(f"{profile}: {job[0]} {job[1]} not started (deadline reached)")
                queue = []
                break
            exp, part = queue.pop(0)
            log = open(LOGS / f"{profile}_{exp}_{part}.log", "w", encoding="utf-8")
            cmd = [PY, str(ROOT / "scripts" / "run_d24.py"), "--exp", exp, "--part", part, "--profile", profile]
            running[(exp, part)] = (subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT), log,
                                    time.time())
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


def step(cmd: list[str], name: str, timeout: int | None = None) -> int:
    log = LOGS / f"{name}.log"
    say(f"{name}: {' '.join(Path(c).name if i == 1 else c for i, c in enumerate(cmd))}")
    with open(log, "w", encoding="utf-8") as f:
        try:
            code = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            code = -1
    say(f"{name}: {'ok' if code == 0 else f'FAILED (exit {code}); see {log}'}")
    return code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    cpu = os.cpu_count() or 2
    ap.add_argument("--workers", type=int, default=8 if cpu >= 10 else max(1, cpu - 1),
                    help="parallel parts (single-threaded each); use the number of physical cores")
    ap.add_argument("--deadline", help="HH:MM local time after which no new part or the Spark run starts")
    ap.add_argument("--spark-scaling", action="store_true", help="run the Spark scaling run after the parts")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true", help="testing only: results would not be reportable")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    deadline = deadline_from(args.deadline)
    say(f"D24 night run: {len(PRIORITY)} parts, {args.workers} workers, deadline {deadline or 'none'}, "
        f"Spark scaling {'yes' if args.spark_scaling else 'no'}; Python {sys.version.split()[0]}")
    problems = preflight(args.spark_scaling, args.allow_dirty)
    for p in problems:
        say(f"PRE-FLIGHT: {p}")
    if problems:
        return 2
    say("pre-flight passed")
    for i, (exp, part) in enumerate(PRIORITY, 1):
        say(f"  {i:2d}. {exp} {part}")
    if args.dry_run:
        return 0

    if not args.skip_smoke:
        smoke = run_parts(PRIORITY, "smoke", args.workers, None)
        failed = [j for j, c in smoke.items() if c != 0]
        if failed:
            say(f"SMOKE FAILED for {failed}; see {LOGS}; stopping before the paper run")
            return 1
        say("smoke passed for every part (not reportable)")

    paper = run_parts(PRIORITY, "paper", args.workers, deadline)
    for exp in ("e1", "e2"):
        if paper.get((exp, "tft_residual_saewc")) == 0:
            step([PY, str(ROOT / "scripts" / "run_d24.py"), "--exp", exp, "--collect-only"], f"collect_{exp}")
        else:
            say(f"collect_{exp}: skipped (RC-TFT part did not finish)")

    if args.spark_scaling:
        if deadline and dt.datetime.now() >= deadline:
            say("Spark scaling: not started (deadline)")
        else:
            step([PY, str(ROOT / "scripts" / "run_spark_scaling.py")], "spark_scaling", timeout=4 * 3600)

    step([PY, str(ROOT / "scripts" / "make_d24_outputs.py")], "outputs")
    say("done. Review data/cache/d24/logs/night.log, then commit, e.g.:")
    say("  git add results/forecasting_d24_*.json results/d22_spark_scaling.json results/d22_spark_validation_scale_*.json "
        "results/tables paper/figures paper/tables && git commit -m \"D24 results\"")
    say("and record the run in the D24 run log (docs/DECISIONS.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
