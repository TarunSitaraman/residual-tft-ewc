"""Download the first-life pool datasets S7–S9 and verify them against their registered checksums.

Sources, licences and checksums are registered in docs/EXTERNAL_DATA.md (S7–S9); this
script downloads nothing that is not listed there. Each file is fetched with resumable
range requests into ``<dest>/<id>/<name>.part``, checked against the registered size and
MD5, and only then renamed. The SHA-256 of every verified file goes into the manifest.

Usage (from the repository root)::

    python scripts/fetch_pool.py                     # all files
    python scripts/fetch_pool.py --only S8 S9        # a subset
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import http.client
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ZENODO = "https://zenodo.org/api/records/{record}/files/{name}/content"

#: Registered files (docs/EXTERNAL_DATA.md, S7–S9): dataset id, file name, URL, size in bytes, MD5.
FILES: tuple[dict, ...] = (
    {"id": "S7", "name": "10.35097-1969.tar", "size": 333_402_624, "md5": "aa60c8fd08755a253ac7479dde41c93d",
     "url": "https://radar.kit.edu/radar-backend/archives/CoNQplSNoVeXExyV/versions/1/content"},
    {"id": "S8", "name": "Battery Dataset.zip", "size": None, "md5": "a635cef9678e0de21f9d5c1f78a4342c",
     "url": ZENODO.format(record=10963339, name=urllib.parse.quote("Battery Dataset.zip"))},
    {"id": "S9", "name": "Dataset_1_NCA_battery.zip", "size": None, "md5": "cbd6d7eab2a296b52385e025e8d0f42a",
     "url": ZENODO.format(record=6405084, name="Dataset_1_NCA_battery.zip")},
    {"id": "S9", "name": "Dataset_2_NCM_battery.zip", "size": None, "md5": "a05c485df92f4af560d1d8d7f659ce1b",
     "url": ZENODO.format(record=6405084, name="Dataset_2_NCM_battery.zip")},
    {"id": "S9", "name": "Dataset_3_NCM_NCA_battery.zip", "size": None, "md5": "d11e68e410a638058906af5e2f5f60f3",
     "url": ZENODO.format(record=6405084, name="Dataset_3_NCM_NCA_battery.zip")},
    {"id": "S9", "name": "Impedance raw data and fitting data.zip", "size": None, "md5": "be9a8088847edd6528ce888372b42465",
     "url": ZENODO.format(record=6405084, name=urllib.parse.quote("Impedance raw data and fitting data.zip"))},
)
CHUNK = 4 << 20
RETRYABLE = (urllib.error.URLError, ConnectionError, TimeoutError, http.client.IncompleteRead)


def _total_size(url: str) -> int:
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        total = r.headers.get("Content-Range", "").rsplit("/", 1)[-1]
    if not total.isdigit():
        raise RuntimeError(f"server gives no total size for {url}")
    return int(total)


def download(f: dict, dest: Path, attempts: int = 30) -> Path:
    """Resumable download into ``<name>.part``; returns the .part path when complete."""
    part = dest / (f["name"] + ".part")
    size = f["size"] or _total_size(f["url"])
    if f["size"] and _total_size(f["url"]) != f["size"]:
        raise RuntimeError(f"{f['name']}: server size differs from the registered {f['size']}")
    for k in range(attempts):
        have = part.stat().st_size if part.exists() else 0
        if have == size:
            return part
        if have > size:
            raise RuntimeError(f"{part} is larger than the file ({have} > {size})")
        try:
            req = urllib.request.Request(f["url"], headers={"Range": f"bytes={have}-"})
            with urllib.request.urlopen(req, timeout=300) as r, part.open("ab") as out:
                if r.status != 206:
                    raise RuntimeError(f"range request ignored (HTTP {r.status})")
                while chunk := r.read(CHUNK):
                    out.write(chunk)
        except RETRYABLE as err:
            print(f"  retry after {err}", file=sys.stderr)
            time.sleep(min(2 ** (k + 1), 60))
    raise RuntimeError(f"{f['name']}: incomplete after {attempts} attempts")


def digests(path: Path) -> tuple[str, str]:
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            md5.update(chunk)
            sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dest", default="data/raw/pool")
    ap.add_argument("--manifest", default="docs/external_data/metadata/pool_downloads_manifest.json")
    ap.add_argument("--only", nargs="*", help="dataset ids to fetch, e.g. S8 S9")
    args = ap.parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": []}
    done = {(m["id"], m["name"]) for m in manifest["files"]}
    for f in FILES:
        if args.only and f["id"] not in args.only:
            continue
        dest = Path(args.dest) / f["id"]
        final = dest / f["name"]
        if (f["id"], f["name"]) in done and final.exists():
            print(f"{f['id']} {f['name']}: already verified")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        print(f"{f['id']} {f['name']}: downloading", flush=True)
        part = download(f, dest)
        md5, sha = digests(part)
        if md5 != f["md5"]:
            raise RuntimeError(f"{f['name']}: MD5 {md5} differs from the registered {f['md5']}; file kept as {part}")
        part.replace(final)
        manifest["files"] = [m for m in manifest["files"] if (m["id"], m["name"]) != (f["id"], f["name"])]
        manifest["files"].append({"id": f["id"], "name": f["name"], "url": f["url"], "size": final.stat().st_size,
                                  "md5": md5, "sha256": sha,
                                  "retrieved_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
        manifest["files"].sort(key=lambda m: (m["id"], m["name"]))
        manifest_path.write_text(json.dumps(manifest, indent=1))
        print(f"{f['id']} {f['name']}: verified (MD5 matches registry), sha256 {sha}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
