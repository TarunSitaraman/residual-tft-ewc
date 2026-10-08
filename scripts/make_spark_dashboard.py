"""Build the data platform page (docs/spark_dashboard.html): dataset catalogue, lineage, pipelines and run registry
(scripts/platform_catalog.py), and the Spark run views from a recorded full Spark run.

Inputs: the monitoring snapshot of a run made with ``spark_ageing_features.py --ui ... --tag dashboard``
(data/spark_ui_snapshot_dashboard.json), its outputs (data/s6_ageing_blocks_spark_dashboard.csv,
results/d22_spark_validation_dashboard.json), the first full run's validation (results/d22_spark_validation.json),
the merged download manifest and the D22 ageing report. Also writes results/d22_spark_dashboard_run.json, the task
and job timings the page shows. Run from the repository root::

    python scripts/make_spark_dashboard.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import ageing as A  # noqa: E402
from src import wheeler as W  # noqa: E402
import platform_catalog as PC  # noqa: E402

RESP = {"v_end_charge": ("End-of-charge voltage", "V", 4), "cv_seconds": ("CV hold time", "s", 1),
        "cv_end_current": ("Current at end of CV hold", "A", 4), "v_end_discharge": ("End-of-discharge voltage", "V", 4),
        "relax_after_charge": ("Relaxation after charge", "V", 4), "relax_after_discharge": ("Relaxation after discharge", "V", 4)}
AGG = {"mean": "block mean", "last": "last cycle", "slope": "slope per cycle"}


def t(s: str) -> float:
    return datetime.strptime(s.replace("GMT", ""), "%Y-%m-%dT%H:%M:%S.%f").timestamp()


def case_labels() -> dict[str, str]:
    out = {}
    for r in W.TEST_CASES.itertuples():
        charge = "C/3" if abs(r.charge_c_rate - 1 / 3) < 1e-6 else f"{r.charge_c_rate:g}C"
        out[r.Battery_ID] = (f"Case {r.Battery_ID[-1]}: {'CC-CV' if r.cccv_charge else 'CC'} {r.v_max:.2f} V, {charge} charge, "
                             f"{'2C' if r.cc2c_discharge else 'WLTP'} discharge, {int(r.dod * 100)} % DoD")
    return out


SLOT_TOLERANCE_S = 0.25   # launch times come from the driver and durations from the worker; they overlap by < 0.1 s


def pack(tasks: list[dict]) -> None:
    """Assign each task a slot: the first slot free at its start (tasks sorted by start)."""
    ends: list[float] = []
    for task in sorted(tasks, key=lambda x: x["start"]):
        for i, e in enumerate(ends):
            if e <= task["start"] + SLOT_TOLERANCE_S:
                ends[i] = task["start"] + task["dur"]
                task["lane"] = i
                break
        else:
            ends.append(task["start"] + task["dur"])
            task["lane"] = len(ends) - 1


RUNS = [("First full run (inputs used in D22)", "d22_spark_validation.json"),
        ("Re-run recorded through the monitoring API (this page)", "d22_spark_validation_dashboard.json"),
        ("Re-run with task retries and event log (History Server)", "d22_spark_validation_eventlog.json")]


def runs() -> list[dict]:
    out = []
    for label, name in RUNS:
        r = json.loads((ROOT / "results" / name).read_text())
        out.append({"label": label, "file": f"results/{name}", "date": r["generated_utc"][:10], "seconds": r["seconds"],
                    "commit": r["environment"]["git_commit"][:7], "dirty": r["environment"]["git_dirty"],
                    "compared": r["compared_blocks"], "blocks": r["blocks"], "match": r["matches_pandas"]})
    return out


def build() -> tuple[dict, dict]:
    """The page data and the task/job timing record of the recorded run."""
    snap = json.loads((ROOT / "data" / "spark_ui_snapshot_dashboard.json").read_text())
    val = json.loads((ROOT / "results" / "d22_spark_validation_dashboard.json").read_text())
    first = json.loads((ROOT / "results" / "d22_spark_validation.json").read_text())
    blocks = pd.read_csv(ROOT / "data" / "s6_ageing_blocks_spark_dashboard.csv")
    manifest = json.loads((ROOT / "data" / "raw" / "wheeler2025_ageing_manifest_all.json").read_text())["files"]
    report = json.loads((ROOT / "results" / "d22_ageing_report.json").read_text())
    wc = pd.read_csv(ROOT / "data" / "wheeler_checks.csv")
    cell_case = wc.drop_duplicates("cell").set_index("cell")["Battery_ID"].to_dict()
    raw_mb = {Path(f["path"]).name.split(".csv")[0]: f["size"] / 1e6 for f in manifest}

    # Map stage: the stage with one task per block and the largest run time.
    stages = [s for s in snap["/stages"] if str(s["stageId"]) in snap["tasks"]]
    map_stage = max(stages, key=lambda s: s.get("executorRunTime", 0))
    raw_tasks = snap["tasks"][str(map_stage["stageId"])]
    by_index = {b["index"]: b for b in snap["blocks"]}
    t0 = min(t(x["launchTime"]) for x in raw_tasks)
    tasks = []
    for x in raw_tasks:
        b = by_index[x["index"]]
        stem = b["file"].split(".csv")[0]
        tasks.append({"cell": b["cell"], "block_end": b["block_end"], "case": cell_case[b["cell"]],
                      "start": round(t(x["launchTime"]) - t0, 2), "dur": round(x["duration"] / 1000, 2), "mb": round(raw_mb[stem], 1)})
    pack(tasks)
    jobs = []
    for j in sorted(snap["/jobs"], key=lambda j: j["jobId"]):
        sec = t(j["completionTime"]) - t(j["submissionTime"]) if j.get("completionTime") else float("nan")
        jobs.append({"id": j["jobId"], "desc": j.get("description") or j["name"], "tasks": j["numTasks"],
                     "seconds": round(sec, 2), "status": j["status"]})
    reduce_s = sum(j["seconds"] for j in jobs if j["desc"].startswith("REDUCE"))

    audit = {a["cell"]: a["pearson"] for a in report["audit_discharged_charge_vs_previous_capacity"]}
    task_df = pd.DataFrame(tasks)
    cells = []
    for cell, g in blocks.groupby("cell"):
        tg = task_df[task_df["cell"] == cell]
        n = g["age_valid_cycles"].sum() + g["age_invalid_cycles"].sum()
        cells.append({"cell": cell, "case": cell_case[cell], "blocks": int(len(g)), "raw_gb": float(tg["mb"].sum() / 1000),
                      "cycles": int(n), "valid_pct": float(100 * g["age_valid_cycles"].sum() / n),
                      "task_min": float(tg["dur"].sum() / 60), "audit_r": audit.get(cell)})
    cells.sort(key=lambda c: (c["case"], c["cell"]))

    feats = [{"key": f"age_{r}_{a}", "label": f"{lab} ({AGG[a]})", "unit": (f"{u}/cycle" if a == "slope" else u),
              "digits": (-1 if a == "slope" else d)} for r, (lab, u, d) in RESP.items() for a in AGG]
    keys = [f["key"] for f in feats]
    bl = [{"cell": r["cell"], "block_end": int(r["block_end"]),
           **{k: (None if pd.isna(r[k]) else float(r[k])) for k in keys}} for _, r in blocks.iterrows()]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    data = {
        "run": {**{k: val[k] for k in ("engine", "blocks", "cycles", "seconds")}, "cores": int(val["engine"].split("local[")[1].split("]")[0])},
        "first_run": {"seconds": first["seconds"]}, "validation": {k: val[k] for k in ("compared_blocks", "criterion", "per_input")},
        "max_abs": max(d["max_abs"] for d in val["per_input"].values()),
        "raw_gb": sum(f["size"] for f in manifest) / 1e9,
        "xz_gb": sum(p.stat().st_size for p in (ROOT / "data" / "raw" / "wheeler2025").rglob("*aging*.csv.*")
                     if p.suffix in (".xz", ".gz")) / 1e9,
        "valid_cycles": int(blocks["age_valid_cycles"].sum()), "tasks": tasks, "lanes": int(max(x["lane"] for x in tasks) + 1),
        "stage_seconds": float(max(x["start"] + x["dur"] for x in tasks)), "median_dur": float(np.median([x["dur"] for x in tasks])),
        "reduce_seconds": reduce_s, "jobs": jobs, "cells": cells, "case_labels": case_labels(), "features": feats, "blocks": bl,
        "app_id": snap["/application"]["id"], "commit": commit, "runs": runs(),
        "n_inputs": len(A.feature_names()), "min_valid": A.MIN_VALID_CYCLES, "n_cells": len(cells),
        "sources": PC.sources(), "tables": PC.tables(), "registry": PC.registry()}
    record = {"app_id": data["app_id"], "engine": data["run"]["engine"], "map_stage": map_stage["stageId"], "tasks": tasks, "jobs": jobs}
    return data, record


def render(data: dict) -> str:
    tpl = (ROOT / "scripts" / "spark_dashboard_template.html").read_text(encoding="utf-8")
    return tpl.replace("__DATA__", json.dumps(data, separators=(",", ":")))


def main() -> int:
    data, record = build()
    out = ROOT / "docs" / "spark_dashboard.html"
    out.write_text(render(data), encoding="utf-8")
    (ROOT / "results" / "d22_spark_dashboard_run.json").write_text(json.dumps(record, indent=1))
    tasks = record["tasks"]
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e3:.0f} kB): {len(tasks)} tasks, {data['lanes']} slots, "
          f"stage {data['stage_seconds'] / 60:.1f} min, median task {data['median_dur']:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
