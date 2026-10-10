"""Dataset catalogue, derived tables and run registry for the data platform page (docs/spark_dashboard.html).

Everything is read from the repository and the local data folder: download manifests, files on disk, CSV headers and
results files. Descriptions, licences and which script builds what come from docs/EXTERNAL_DATA.md and data/README.md.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, RAW, META = ROOT / "data", ROOT / "data" / "raw", ROOT / "docs" / "external_data" / "metadata"
GB = 1e9

HANDOVER = ["cleaned_battery_data.csv", "rpt_capacity_data.csv", "DATA_DICTIONARY.csv", "ml_training_dataset.csv",
            "final_ml_dataset.csv", "final_battery_health_dataset.csv", "final_battery_cycling_features.csv"]


def _disk(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else (path.stat().st_size if path.exists() else 0)


def _stored(base: Path, rel_paths: list[str]) -> tuple[int, int]:
    """Bytes on disk and number of files present for manifest paths stored compressed (.xz or .gz)."""
    size = n = 0
    for rel in rel_paths:
        for suffix in (".xz", ".gz", ""):
            p = base / (rel + suffix)
            if p.exists():
                size += p.stat().st_size
                n += 1
                break
    return size, n


def _cells(csv: str, col: str) -> int:
    return int(pd.read_csv(DATA / csv, usecols=[col])[col].nunique())


def sources() -> list[dict]:
    osf = json.loads((META / "osf_8jnr5_v2_diagnostic_manifest.json").read_text())
    char = json.loads((META / "wheeler2025_characterisation_manifest.json").read_text())
    age = json.loads((RAW / "wheeler2025_ageing_manifest_all.json").read_text())
    pool = json.loads((META / "pool_downloads_manifest.json").read_text())["files"]
    char_disk, char_n = _stored(RAW / "wheeler2025", [f["path"] for f in char["files"]])
    age_disk, age_n = _stored(RAW / "wheeler2025", [f["path"] for f in age["files"]])
    hand = [DATA / f for f in HANDOVER]
    out = [
        {"id": "S1", "name": "Stanford second-life diagnostic tests", "who": "Moy et al. 2024", "doi": "10.17605/OSF.IO/8JNR5",
         "role": "Second-life cells: raw RPT capacity tests and EIS spectra", "cells": _cells("diagnostic_features.csv", "Battery_ID"),
         "files": osf["n_files"], "orig_gb": osf["total_bytes"] / GB, "stored_gb": _disk(RAW / "osf_8jnr5_v2") / GB,
         "fetch": "scripts/fetch_osf_diagnostics.py", "verify": "Archive SHA-256; CRC-32 and SHA-256 per extracted file",
         "store": "data/raw/osf_8jnr5_v2", "decisions": "D16, D17"},
        {"id": "S2", "name": "Project handover tables", "who": "processed subset of S1", "doi": "Hugging Face mirror",
         "role": "Cleaned cycling data, RPT capacities and SOH-labelled cycles", "cells": _cells("final_battery_health_dataset.csv", "Battery_ID"),
         "files": sum(p.exists() for p in hand), "orig_gb": None, "stored_gb": sum(_disk(p) for p in hand) / GB,
         "fetch": "manual download", "verify": "scripts/verify_external_data.py: derivation from S1 checked",
         "store": "data/*.csv", "decisions": "notebooks 01–10"},
        {"id": "S6", "name": "LFP first- and second-life: capacity checks", "who": "Wheeler et al. 2025", "doi": "10.57745/OLBXKT",
         "role": "C/3 and C/25 characterisation every 100 cycles, 7 usage cases", "cells": _cells("wheeler_checks.csv", "cell"),
         "files": char_n, "orig_gb": char["total_bytes"] / GB, "stored_gb": char_disk / GB,
         "fetch": "scripts/fetch_wheeler_characterisation.py", "verify": "Range reads from remote zips; CRC-32; SHA-256 manifest; authors' capacity reproduced exactly",
         "store": "data/raw/wheeler2025", "decisions": "D19–D23"},
        {"id": "S6", "name": "LFP first- and second-life: ageing cycles", "who": "Wheeler et al. 2025", "doi": "10.57745/OLBXKT",
         "role": "Every ageing cycle between the checks, 901,974,259 rows; processed with Spark", "cells": _cells("s6_ageing_blocks.csv", "cell"),
         "files": age_n, "orig_gb": age["total_bytes"] / GB, "stored_gb": age_disk / GB,
         "fetch": "scripts/fetch_wheeler_characterisation.py --kind aging", "verify": "Range reads; CRC-32; SHA-256 manifest; Spark output checked against pandas",
         "store": "data/raw/wheeler2025", "decisions": "D22", "big": True},
    ]
    names = {"S7": ("First-life pool: KIT NMC/C-SiO", "Luh and Blank 2024", "10.35097/1969", "pool_s7_checks.csv"),
             "S8": ("First-life pool: XJTU NCM", "Wang et al. 2024", "10.5281/zenodo.10963339", "pool_s8_checks.csv"),
             "S9": ("First-life pool: Tongji NCA/NCM", "Zhu et al. 2022", "10.5281/zenodo.6405084", "pool_s9_checks.csv")}
    for sid, (name, who, doi, table) in names.items():
        fs = [f for f in pool if f["id"] == sid]
        out.append({"id": sid, "name": name, "who": who, "doi": doi, "role": "Capacity check-ups for pretraining (first life only)",
                    "cells": _cells(table, "cell"), "files": len(fs), "orig_gb": sum(f["size"] for f in fs) / GB,
                    "stored_gb": _disk(RAW / "pool" / sid) / GB, "fetch": "scripts/fetch_pool.py",
                    "verify": "MD5 against the repository record; SHA-256 manifest", "store": f"data/raw/pool/{sid}", "decisions": "D21"})
    return out


TABLES = [  # file, built by, from, used by
    ("cleaned_battery_data.csv", "handover", "S2", "notebooks 01, 10, 12"),
    ("final_battery_cycling_features.csv", "handover", "S2", "notebooks 01–03"),
    ("ml_training_dataset.csv", "handover", "S2", "notebooks 01–09 (superseded current-SOH work)"),
    ("final_battery_health_dataset.csv", "handover", "S2", "notebooks 02–11"),
    ("rpt_health_c20.csv", "scripts/build_diagnostic_features.py", "S1", "C/20 SOH labels (D17)"),
    ("diagnostic_features.csv", "scripts/build_diagnostic_features.py", "S1", "notebooks 11–12 (D16)"),
    ("wheeler_checks.csv", "scripts/build_wheeler_features.py", "S6 checks", "notebook 13, D20–D23"),
    ("s6_ageing_cycles.csv", "scripts/build_s6_ageing_features.py (pandas)", "S6 ageing", "D22 audit"),
    ("s6_ageing_blocks.csv", "scripts/build_s6_ageing_features.py (pandas)", "S6 ageing", "D22 models"),
    ("s6_ageing_blocks_spark.csv", "scripts/spark_ageing_features.py (Spark)", "S6 ageing", "check against pandas"),
    ("pool_s7_checks.csv", "scripts/build_pool_tables.py", "S7", "D21 pretraining (P3)"),
    ("pool_s8_checks.csv", "scripts/build_pool_tables.py", "S8", "D21 pretraining (P3)"),
    ("pool_s9_checks.csv", "scripts/build_pool_tables.py", "S9", "D21 pretraining (P3)"),
]


def tables() -> list[dict]:
    out = []
    for name, builder, src, used in TABLES:
        p = DATA / name
        if not p.exists():
            continue
        with p.open("rb") as f:
            rows = sum(1 for _ in f) - 1
        cols = len(pd.read_csv(p, nrows=0).columns)
        out.append({"file": name, "rows": rows, "cols": cols, "mb": p.stat().st_size / 1e6, "builder": builder, "source": src, "used": used})
    return out


REGISTRY = {  # results file -> (title, kind, data)
    "preprocessing_table.json": ("Preprocessing table (notebook 03)", "Analysis", "S2"),
    "baseline_metrics.json": ("Current-SOH baselines (notebook 04)", "Model", "S2"),
    "health_indicators.json": ("Health indicators (notebook 10)", "Analysis", "S2"),
    "diagnostic_features.json": ("Diagnostic feature table (D16)", "Data build", "S1"),
    "forecasting.json": ("SOH forecasting on S1 (notebook 12)", "Model", "S1"),
    "forecasting_wheeler.json": ("SOH forecasting on S6 (notebook 13, D20)", "Model", "S6"),
    "data_audit.json": ("Independent data audit", "Audit", "S1, S6"),
    "forecasting_d21_p3.json": ("Pretrain on pool, then S6 stream (D21, P3)", "Model", "S6, S7–S9"),
    "forecasting_d21_ablation_a1.json": ("Ablation A1: inputs × learning rate (D21)", "Model", "S6"),
    "forecasting_d21_ablation_a1_replicate.json": ("Ablation A1, replicate seeds (D21)", "Model", "S6"),
    "d21_tft_interpretation.json": ("TFT interpretation (D21)", "Analysis", "S6"),
    "forecasting_d23_e1.json": ("Learned residual, experiment E1 (D23)", "Model", "S6"),
    "forecasting_d23_e2.json": ("Learned residual, experiment E2 (D23)", "Model", "S6"),
    "d22_spark_validation_test.json": ("Spark ageing extraction, 6-block check", "Spark", "S6 ageing"),
    "d22_ageing_report.json": ("Ageing extraction report, pandas (D22)", "Data build", "S6 ageing"),
    "d22_spark_validation_strict_relative.json": ("Spark run, strict relative comparison", "Spark", "S6 ageing"),
    "d22_spark_validation.json": ("Spark full run 1 (inputs used in D22)", "Spark", "S6 ageing"),
    "forecasting_d22_e1_f2.json": ("Ageing inputs in the forecasting models (D22)", "Model", "S6, S6 ageing"),
    "d22_spark_validation_dashboard.json": ("Spark full run 2, monitoring API", "Spark", "S6 ageing"),
    "d22_spark_validation_eventlog.json": ("Spark full run 3, event log and retries", "Spark", "S6 ageing"),
}


def _git_date(rel: str) -> str:
    return subprocess.run(["git", "log", "-1", "--format=%cs", "--", rel], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def registry() -> list[dict]:
    out = []
    for name, (title, kind, data) in REGISTRY.items():
        p = ROOT / "results" / name
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        d = d if isinstance(d, dict) else {}
        env = d.get("environment") if isinstance(d.get("environment"), dict) else {}
        date = d.get("generated_utc", "")[:10] or _git_date(f"results/{name}")
        out.append({"file": f"results/{name}", "title": title, "kind": kind, "data": data, "date": date,
                    "dated_by": "run" if d.get("generated_utc") else "commit", "decision": d.get("decision", ""),
                    "profile": d.get("profile", ""), "commit": (env.get("git_commit") or "")[:7], "dirty": bool(env.get("git_dirty")),
                    "kb": p.stat().st_size / 1e3})
    return sorted(out, key=lambda r: r["date"])
