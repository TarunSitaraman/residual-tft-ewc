"""Fetch the characterisation files of the Wheeler et al. 2025 LFP dataset (S6, decision D19).

Each of the 20 per-cell zips (0.9-1.7 GB) holds ageing files, characterisation
files and EIS files. Only the characterisation files are needed (one per 100
cycles: C/3 and C/25 charge and discharge). They are read directly from the
zips with HTTP range requests (about 3 GB compressed instead of 26 GB):

1. the zip index is read from the end of each archive;
2. each characterisation entry is fetched, decompressed and checked against
   the CRC-32 in the index;
3. the file is stored gzip-compressed under ``data/raw/wheeler2025/<cell>/``
   (git-ignored), and the SHA-256 of its uncompressed content goes into the
   manifest.

The zip sizes and MD5 values are pinned in the Dataverse record saved in the
repository; a size mismatch means the repository changed and stops the run.

Usage (from the repository root)::

    python scripts/fetch_wheeler_characterisation.py \
        --manifest docs/external_data/metadata/wheeler2025_characterisation_manifest.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import http.client
import json
import lzma
import shutil
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RECORD = Path("docs/external_data/metadata/wheeler2025_lfp_dataverse.dataset.json")
ACCESS = "https://entrepot.recherche.data.gouv.fr/api/access/datafile/{id}"
CHUNK = 2 << 20          # bytes per range request: small enough to finish before slow links drop


def _range(url: str, start: int, end: int, size: int, attempts: int = 15) -> bytes:
    for k in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                if resp.status != 206:
                    raise RuntimeError(f"range request ignored (HTTP {resp.status})")
                total = resp.headers.get("Content-Range", "").rsplit("/", 1)[-1]
                if total.isdigit() and int(total) != size:
                    raise RuntimeError(f"archive size {total} differs from the recorded {size}")
                data = resp.read()
            if len(data) != end - start + 1:
                raise ConnectionError("short read")
            return data
        except (urllib.error.URLError, ConnectionError, TimeoutError, http.client.IncompleteRead) as err:
            if k == attempts - 1:
                raise
            print(f"  retry after {err}", file=sys.stderr)
            time.sleep(min(2 ** (k + 1), 60))


def zip_index(url: str, size: int) -> list[dict]:
    """Entries of a remote zip: name, compression method, sizes, CRC-32 and local-header offset."""
    tail = _range(url, max(0, size - 65_557), size - 1, size)
    i = tail.rfind(b"PK\x05\x06")
    _, _, _, _, n, cd_size, cd_off, _ = struct.unpack("<IHHHHIIH", tail[i:i + 22])
    if 0xFFFFFFFF in (cd_size, cd_off) or n == 0xFFFF:
        j = tail.rfind(b"PK\x06\x06")
        n, cd_size, cd_off = struct.unpack("<QQQ", tail[j + 32:j + 56])
    cd = _range(url, cd_off, cd_off + cd_size - 1, size)
    out, p = [], 0
    while cd[p:p + 4] == b"PK\x01\x02":
        method, = struct.unpack("<H", cd[p + 10:p + 12])
        crc, comp, usize, nl, el, cl = struct.unpack("<IIIHHH", cd[p + 16:p + 34])
        off, = struct.unpack("<I", cd[p + 42:p + 46])
        name = cd[p + 46:p + 46 + nl].decode("utf-8", "replace")
        extra = cd[p + 46 + nl:p + 46 + nl + el]
        # Zip64 extra field: 64-bit sizes / offset replace 0xFFFFFFFF values, in this order.
        q = 0
        while q + 4 <= len(extra):
            tag, ln = struct.unpack("<HH", extra[q:q + 4])
            if tag == 0x0001:
                vals, r = [], q + 4
                for flag in (usize, comp, off):
                    if flag == 0xFFFFFFFF:
                        vals.append(struct.unpack("<Q", extra[r:r + 8])[0]); r += 8
                    else:
                        vals.append(flag)
                usize, comp, off = vals
            q += 4 + ln
        out.append({"name": name, "method": method, "compressed": comp, "size": usize, "crc32": crc, "offset": off})
        p += 46 + nl + el + cl
    return out


def fetch_entry(url: str, size: int, e: dict) -> bytes:
    head = _range(url, e["offset"], e["offset"] + 29, size)
    nl, el = struct.unpack("<HH", head[26:30])
    start = e["offset"] + 30 + nl + el
    # Fetch in pieces so a dropped connection costs one piece, not the whole entry (ageing files are ~37 MB).
    raw = b"".join(_range(url, p, min(p + CHUNK, start + e["compressed"]) - 1, size)
                   for p in range(start, start + e["compressed"], CHUNK))
    if e["method"] == 0:
        data = raw
    elif e["method"] == 8:
        data = zlib.decompressobj(-15).decompress(raw)
    else:
        raise RuntimeError(f"unsupported compression method {e['method']} for {e['name']}")
    if len(data) != e["size"] or (zlib.crc32(data) & 0xFFFFFFFF) != e["crc32"]:
        raise RuntimeError(f"CRC or size mismatch for {e['name']}")
    return data


def _is_kind(name: str, kind: str) -> bool:
    """``charac``: any path containing "charac" (the 1,007 registered files, including the EIS files
    in ``initialcharach_25degC``). ``aging``: ageing files, judged by file name only, because every
    50 degC file sits in a folder named ``agingdata_50degC``."""
    if kind == "charac":
        return "charac" in name.lower()
    base = name.rsplit("/", 1)[-1].lower()
    return "aging" in base and "charac" not in base and "eis" not in base


def _open_stored(path: Path):
    return lzma.open(path, "rb") if path.suffix == ".xz" else gzip.open(path, "rb")


def _verified_local(target: Path, e: dict) -> bytes | None:
    """Content of an already-fetched file (stored as .gz or .xz) if it matches the zip index, else None.

    A file cut short by an interrupted run fails to decompress or fails the size and
    CRC-32 check, and is fetched again.
    """
    for path in (target.with_suffix(".xz"), target.with_suffix(".gz")):
        if not path.is_file():
            continue
        try:
            with _open_stored(path) as f:
                data = f.read()
        except (OSError, EOFError, lzma.LZMAError):
            continue
        if len(data) == e["size"] and (zlib.crc32(data) & 0xFFFFFFFF) == e["crc32"]:
            return data
    return None


def _store(target: Path, data: bytes, store: str) -> None:
    """Write compressed to a temporary name, then rename, so a partial file is never left under the final name."""
    target.parent.mkdir(parents=True, exist_ok=True)
    final = target.with_suffix("." + store)
    tmp = final.with_name(final.name + ".part")
    with (lzma.open(tmp, "wb", preset=6) if store == "xz" else gzip.open(tmp, "wb", compresslevel=6)) as f:
        f.write(data)
    tmp.replace(final)


def _write_manifest(manifest: dict, path: Path) -> None:
    manifest.update({"source": "S6 in docs/EXTERNAL_DATA.md: doi:10.57745/OLBXKT, version 2.0 (CC BY 4.0)",
                     "retrieved_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "n_files": len(manifest["files"]), "total_bytes": sum(f["size"] for f in manifest["files"])})
    manifest["files"].sort(key=lambda f: f["path"])
    tmp = path.with_suffix(path.suffix + ".tmp")
    for attempt in range(10):
        try:
            tmp.write_text(json.dumps(manifest, indent=1))
            tmp.replace(path)   # atomic, so an interruption never leaves a truncated manifest
            return
        except PermissionError:
            # Windows: a scanner or indexer may hold the file briefly.
            if attempt == 9:
                raise
            time.sleep(1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dest", default="data/raw/wheeler2025")
    ap.add_argument("--manifest", default="docs/external_data/metadata/wheeler2025_characterisation_manifest.json")
    ap.add_argument("--cells", nargs="*", help="subset of cells, e.g. Cell1a (default: all 20)")
    ap.add_argument("--kind", choices=("charac", "aging"), default="charac",
                    help="characterisation files (default) or the between-check ageing files")
    ap.add_argument("--store", choices=("gz", "xz"), default="gz",
                    help="local compression: gz (as registered for the characterisation files) or xz (about 45 %% smaller)")
    ap.add_argument("--min-free-gb", type=float, default=5.0, help="stop before the disk has less free space than this")
    ap.add_argument("--shard", default="0/1", metavar="I/N",
                    help="parallel workers on one cell: take every N-th file not yet on disk, starting at I (own --manifest)")
    args = ap.parse_args()
    shard_i, shard_n = (int(v) for v in args.shard.split("/"))
    files = json.loads(RECORD.read_text())["data"]["latestVersion"]["files"]
    zips = [f["dataFile"] for f in files if f["dataFile"]["filename"].startswith("Cell") and f["dataFile"]["filename"].endswith(".zip")]
    if args.cells:
        zips = [z for z in zips if z["filename"][:-4] in set(args.cells)]
    dest = Path(args.dest)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": []}
    done = {f["path"] for f in manifest["files"]}
    # One writer thread: compressing a file (xz takes minutes) overlaps the download of the next one.
    writer = ThreadPoolExecutor(max_workers=1)
    pending: list = []

    def finish_pending() -> None:
        for fut, record in pending:
            fut.result()                      # re-raises any error from the writer
            manifest["files"].append(record)
            done.add(record["path"])
            _write_manifest(manifest, manifest_path)   # after every file, so an interrupted run resumes per file
        pending.clear()

    for z in sorted(zips, key=lambda z: z["filename"]):
        url, size = ACCESS.format(id=z["id"]), int(z["filesize"])
        entries = [e for e in zip_index(url, size) if _is_kind(e["name"], args.kind) and not e["name"].endswith("/")
                   and not e["name"].startswith("__MACOSX")]
        if shard_n > 1:
            # A fixed split of the full sorted list, so shards never overlap even when a worker restarts;
            # files already on disk are verified locally (not fetched) by _verified_local below.
            entries = sorted(entries, key=lambda e: e["name"])[shard_i::shard_n]
        print(f"{z['filename']}: {len(entries)} {args.kind} files" + (f" (shard {args.shard})" if shard_n > 1 else ""),
              flush=True)
        for e in entries:
            rel = e["name"]
            if rel in done:
                continue
            if shutil.disk_usage(dest if dest.exists() else ".").free < args.min_free_gb * 1e9:
                finish_pending()
                _write_manifest(manifest, manifest_path)
                raise SystemExit(f"stopped: less than {args.min_free_gb} GB free on disk")
            target = dest / (rel + ".gz")
            data = _verified_local(target, e)
            fetched = data is None
            if fetched:
                data = fetch_entry(url, size, e)
            finish_pending()                  # the previous file is written before this one is queued
            # Write whatever was downloaded (this also replaces a corrupt local copy); a verified local copy is kept.
            fut = writer.submit(_store, target, data, args.store) if fetched else writer.submit(lambda: None)
            record = {"path": rel, "zip": z["filename"], "zip_md5": z["checksum"]["value"], "size": e["size"],
                      "crc32": f"{e['crc32']:08x}", "sha256": hashlib.sha256(data).hexdigest()}
            pending.append((fut, record))
        finish_pending()
        _write_manifest(manifest, manifest_path)
    writer.shutdown(wait=True)
    print(f"{manifest['n_files']} files, {manifest['total_bytes']:,} bytes uncompressed; manifest {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
