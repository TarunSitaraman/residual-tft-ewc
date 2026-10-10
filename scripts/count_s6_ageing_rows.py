"""Count the data rows and uncompressed bytes of the S6 ageing files (D22), for the size quoted in the paper.

Each file is decompressed as a stream; rows are newline-terminated lines minus the one header line.
Writes results/s6_ageing_row_count.json (totals plus per-file counts, environment and commit).

Usage (from the repository root)::

    python scripts/count_s6_ageing_rows.py [--workers 6]
"""

from __future__ import annotations

import argparse
import gzip
import lzma
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import experiments as X  # noqa: E402

RAW = ROOT / "data" / "raw" / "wheeler2025"
CHUNK = 1 << 24


def count(path: Path) -> tuple[str, int, int]:
    """(relative path, data rows, uncompressed bytes) of one file."""
    opener = lzma.open if path.suffix == ".xz" else gzip.open
    lines = size = 0
    last = b"\n"
    with opener(path, "rb") as f:
        while chunk := f.read(CHUNK):
            lines += chunk.count(b"\n")
            size += len(chunk)
            last = chunk[-1:]
    lines += last != b"\n"  # final line without a newline
    return path.relative_to(RAW).as_posix(), lines - 1, size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    files = sorted(RAW.rglob("*_aging[0-9]*"))
    with ProcessPoolExecutor(args.workers) as pool:
        per_file = sorted(pool.map(count, files))
    rows = sum(r for _, r, _ in per_file)
    size = sum(b for _, _, b in per_file)
    out = X.write_results(ROOT / "results" / "s6_ageing_row_count.json", {
        "decision": "D22", "n_files": len(per_file), "data_rows": rows, "uncompressed_bytes": size,
        "files": [{"path": p, "data_rows": r, "uncompressed_bytes": b} for p, r, b in per_file]})
    print(f"{len(per_file)} files, {rows:,} data rows, {size:,} bytes -> {out}")


if __name__ == "__main__":
    main()
