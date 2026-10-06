"""Merge the per-worker S6 ageing-file manifests into one and check completeness.

The parallel download (data/raw/run_s6_ageing_fetch.ps1) writes data/raw/wheeler2025_ageing_manifest[_wN].json, each
with the path, zip, CRC-32 and SHA-256 of every file it fetched or verified. This writes the merged
data/raw/wheeler2025_ageing_manifest_all.json and exits non-zero unless every expected file is listed exactly once
(with one SHA-256) and present on disk.

Usage (from the repository root)::

    python scripts/merge_ageing_manifests.py [--expected 827]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--expected", type=int, default=827)
    args = ap.parse_args()
    parts = sorted(p for p in RAW.glob("wheeler2025_ageing_manifest*.json") if not p.name.endswith("_all.json"))
    files: dict[str, dict] = {}
    conflicts = []
    for p in parts:
        for f in json.loads(p.read_text())["files"]:
            if f["path"] in files and files[f["path"]]["sha256"] != f["sha256"]:
                conflicts.append(f["path"])
            files.setdefault(f["path"], f)
    missing_on_disk = [k for k in files
                       if not any((RAW / "wheeler2025" / (k + s)).exists() for s in (".xz", ".gz"))]
    out = {"source": "S6 (doi:10.57745/OLBXKT, v2.0, CC BY 4.0), between-check ageing files",
           "parts": [p.name for p in parts], "n_files": len(files),
           "total_bytes": sum(f["size"] for f in files.values()), "files": sorted(files.values(), key=lambda f: f["path"])}
    (RAW / "wheeler2025_ageing_manifest_all.json").write_text(json.dumps(out, indent=1))
    print(f"{len(files)} files from {len(parts)} manifests; {out['total_bytes']:,} bytes raw; "
          f"SHA-256 conflicts: {len(conflicts)}; listed but missing on disk: {len(missing_on_disk)}", flush=True)
    ok = len(files) == args.expected and not conflicts and not missing_on_disk
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
