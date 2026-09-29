"""Check that the handover data derive from the public OSF second-life dataset.

Reads only the central directory of the OSF archive (HTTP range requests, about
0.7 MB instead of 2.15 GB) and compares it with the handover CSVs:

1. every file referenced in ``rpt_capacity_data.csv`` exists in the archive,
   inside the same ``RPT_#`` folder;
2. every (cell, ``Cycling_#``) block in ``cleaned_battery_data.csv`` has a
   folder in the archive, and vice versa.

Usage (from the repository root, with the CSVs in ``DATA_ROOT``)::

    python scripts/verify_external_data.py --out docs/external_data/metadata/osf_8jnr5_v2_listing.json

The archive's size and SHA-256 are fixed below from the OSF file record
(docs/external_data/metadata/moy2024_second_life_osf.file.json). A different
size means OSF now serves another version and the registry must be updated.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

OSF_FILE_URL = "https://osf.io/download/j2ne9/?version=2"
ARCHIVE_SIZE = 2_153_883_052
ARCHIVE_SHA256 = "201b21b8e2e9623887d03e9bd6620341e5c91cd28bce34294ccbad2d4e7c6fe3"
CELLS = ("G1", "V4", "V5", "W8", "W9", "W10")


def _range(url: str, start: int, end: int, attempts: int = 5) -> bytes:
    for k in range(attempts):
        try:
            return _range_once(url, start, end)
        except urllib.error.HTTPError as err:   # OSF occasionally answers 5xx; retry with backoff
            if err.code < 500 or k == attempts - 1:
                raise
            time.sleep(2 ** (k + 1))


def _range_once(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        if resp.status != 206:
            raise RuntimeError(f"server ignored the range request (HTTP {resp.status})")
        total = resp.headers.get("Content-Range", "").rsplit("/", 1)[-1]
        if total.isdigit() and int(total) != ARCHIVE_SIZE:
            raise RuntimeError(f"archive size {total} differs from the registered v2 size {ARCHIVE_SIZE}")
        return resp.read()


def archive_listing(url: str = OSF_FILE_URL) -> list[dict]:
    """File names, uncompressed sizes and CRC-32 values from the zip central directory."""
    tail = _range(url, ARCHIVE_SIZE - 65_557, ARCHIVE_SIZE - 1)
    i = tail.rfind(b"PK\x05\x06")
    _, _, _, _, n, cd_size, cd_off, _ = struct.unpack("<IHHHHIIH", tail[i:i + 22])
    if 0xFFFFFFFF in (cd_size, cd_off) or n == 0xFFFF:
        j = tail.rfind(b"PK\x06\x06")
        n, cd_size, cd_off = struct.unpack("<QQQ", tail[j + 32:j + 56])
    cd = _range(url, cd_off, cd_off + cd_size - 1)
    out, p = [], 0
    while cd[p:p + 4] == b"PK\x01\x02":
        crc, _, usize, nl, el, cl = struct.unpack("<IIIHHH", cd[p + 16:p + 34])
        name = cd[p + 46:p + 46 + nl].decode("utf-8", "replace")
        if not name.startswith("__MACOSX") and not name.endswith("/") and not name.endswith(".DS_Store"):
            out.append({"name": name, "size": usize, "crc32": f"{crc:08x}"})
        p += 46 + nl + el + cl
    if len(out) == 0:
        raise RuntimeError("empty central directory")
    return out


def compare(listing: list[dict], data_root: Path) -> dict:
    by_base = {os.path.basename(f["name"]): f["name"] for f in listing}
    rpt = pd.read_csv(data_root / "rpt_capacity_data.csv")
    found = [by_base.get(os.path.basename(str(f))) for f in rpt["File"]]
    same_folder = sum(bool(p) and f"/{r}/" in p for p, r in zip(found, rpt["RPT"]))

    archive_blocks = {(m.group(2).lower(), f"Cycling_{m.group(1)}") for f in listing
                      for m in [re.search(r"/Cycling_(\d+)/(" + "|".join(CELLS) + r")/", f["name"])] if m}
    pts = pd.read_csv(data_root / "cleaned_battery_data.csv", usecols=["Source_Folder", "Cycling_Test"]).drop_duplicates()
    handover_blocks = {(str(a).lower(), str(b)) for a, b in pts.itertuples(index=False)}
    return {
        "archive_files": len(listing),
        "rpt_file_refs": len(rpt),
        "rpt_file_refs_found": sum(bool(p) for p in found),
        "rpt_file_refs_in_same_rpt_folder": int(same_folder),
        "handover_cycling_blocks": len(handover_blocks),
        "blocks_missing_from_archive": sorted(handover_blocks - archive_blocks),
        "archive_blocks_missing_from_handover": sorted(archive_blocks - handover_blocks),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "data"))
    ap.add_argument("--out", help="write the archive listing and the comparison to this JSON file")
    args = ap.parse_args()
    listing = archive_listing()
    result = compare(listing, Path(args.data_root))
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps({"source": OSF_FILE_URL, "archive_size": ARCHIVE_SIZE,
                                              "archive_sha256": ARCHIVE_SHA256, "comparison": result,
                                              "files": listing}, indent=1))
    ok = (result["rpt_file_refs_found"] == result["rpt_file_refs"] == result["rpt_file_refs_in_same_rpt_folder"]
          and not result["blocks_missing_from_archive"] and not result["archive_blocks_missing_from_handover"])
    print("MATCH" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
