"""Spark scaling run (D24 run plan, step 5; work plan WS2.2): the D22 Spark job on the same blocks at several core counts.

Runs ``spark_ageing_features.py --limit N --tag scale_c<cores>`` once per core count (the job's own ``--limit``
selection, spread over the cells), each validated against the pandas implementation by the job itself, and writes
``results/d22_spark_scaling.json`` with wall time, speed-up and parallel efficiency against one core. Run it on an
otherwise idle machine: timings measured while other jobs run are not reportable.

Usage (from the repository root)::

    python scripts/run_spark_scaling.py                     # cores 1 2 4 6, 120 blocks
    python scripts/run_spark_scaling.py --cores 1 2 4 8 --limit 120
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import experiments as X  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cores", type=int, nargs="+", default=[1, 2, 4, 6])
    ap.add_argument("--limit", type=int, default=120, help="blocks per run (same blocks at every core count)")
    ap.add_argument("--timeout", type=int, default=3 * 3600, help="seconds per run")
    args = ap.parse_args()
    runs = []
    for c in sorted(args.cores):
        tag = f"scale_c{c}"
        print(f"Spark scaling: {args.limit} blocks on local[{c}] ...", flush=True)
        t0 = time.perf_counter()
        p = subprocess.run([sys.executable, str(ROOT / "scripts" / "spark_ageing_features.py"), "--cores", str(c),
                            "--limit", str(args.limit), "--tag", tag], cwd=ROOT, timeout=args.timeout)
        wall = time.perf_counter() - t0
        path = ROOT / "results" / f"d22_spark_validation_{tag}.json"
        if p.returncode != 0 or not path.exists():
            runs.append({"cores": c, "status": f"failed (exit {p.returncode})", "process_wall_s": round(wall, 1)})
            continue
        v = json.loads(path.read_text())
        runs.append({"cores": c, "status": "ok", "job_seconds": v["seconds"], "process_wall_s": round(wall, 1),
                     "blocks": v["blocks"], "cycles": v["cycles"], "matches_pandas": v["matches_pandas"],
                     "max_abs_diff": max(d["max_abs"] for d in v["per_input"].values()), "engine": v["engine"]})
    ok = {r["cores"]: r for r in runs if r["status"] == "ok"}
    base = ok.get(min(ok)) if ok else None
    for r in runs:
        if base and r["status"] == "ok":
            r["speedup"] = round(base["job_seconds"] / r["job_seconds"], 3)
            r["efficiency"] = round(r["speedup"] / (r["cores"] / base["cores"]), 3)
    X.write_results(ROOT / "results" / "d22_spark_scaling.json", {
        "decision": "D24 run plan step 5 (WS2.2)", "limit_blocks": args.limit,
        "selection": "spark_ageing_features.py --limit: every k-th block of the sorted file list, first N",
        "baseline_cores": base["cores"] if base else None,
        "note": "job_seconds = Spark map + reduce, from the job's own timer; speed-up relative to the fewest cores run",
        "runs": runs}, ROOT)
    for r in runs:
        print(r, flush=True)
    return 0 if all(r["status"] == "ok" and r["matches_pandas"] for r in runs) else 1


if __name__ == "__main__":
    sys.exit(main())
