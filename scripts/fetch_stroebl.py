"""Fetch the 23 degC check-ups of the Stroebl et al. 2024 ageing study (figshare 10.6084/m9.figshare.25975315, CC BY 4.0).

For every per-cell zip of the record: download it, check its MD5 against the figshare record, and keep only the files
D36 reads, gzip-compressed under ``data/raw/stroebl2024/<file_id>/``: the check-ups at 23 degC (``*_CU``, ``*_exCU``,
``*_ET_T23``, ``*_AT_T23``) and every ``*_meta.txt`` (it names the internal cell serial, which identifies the stage,
and the measurement start date). The zip is then deleted (it can be re-downloaded and re-checked by its MD5). Cycling
(``ZYK``) and 10/45 degC characterisation files are not kept.

``docs/external_data/metadata/stroebl2024_manifest.json`` records, per zip: its figshare id, name, size and MD5, the
full list of its entries (name, size, CRC-32) and, for each kept file, the SHA-256 of its uncompressed content. A zip
whose kept files are already in the manifest is skipped, so the script resumes.

    python scripts/fetch_stroebl.py --workers 3
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RECORD = 25975315
API = f"https://api.figshare.com/v2/articles/{RECORD}"
KEEP = re.compile(r"_(CU|exCU|ET_T23|AT_T23)\.csv$|_meta\.txt$")
MANIFEST = ROOT / "docs" / "external_data" / "metadata" / "stroebl2024_manifest.json"


def data_root() -> Path:
    import os
    d = Path(os.environ.get("DATA_ROOT", ROOT / "data"))
    return d if d.is_absolute() else ROOT / d


def get(url: str, dest: Path, attempts: int = 8) -> None:
    for k in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=300) as r, open(dest, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            return
        except OSError as err:
            if k == attempts - 1:
                raise
            print(f"  retry {url}: {err}", file=sys.stderr, flush=True)
            time.sleep(min(2 ** (k + 1), 60))


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch_one(f: dict, out: Path) -> dict:
    cell_dir = out / str(f["id"])
    cell_dir.mkdir(parents=True, exist_ok=True)
    tmp = cell_dir / f["name"]
    for attempt in range(3):
        get(f["download_url"], tmp)
        if md5(tmp) == f["supplied_md5"]:
            break
        print(f"  MD5 mismatch for {f['name']} ({f['id']}), attempt {attempt + 1}", file=sys.stderr, flush=True)
    else:
        raise RuntimeError(f"{f['name']} ({f['id']}): MD5 never matched the figshare record")
    entries, kept = [], {}
    with zipfile.ZipFile(tmp) as z:
        for e in z.infolist():
            entries.append({"name": e.filename, "size": e.file_size, "crc32": e.CRC})
            base = e.filename.rsplit("/", 1)[-1]
            if e.is_dir() or not KEEP.search(base):
                continue
            data = z.read(e)                                    # zipfile checks the CRC-32 on read
            with gzip.open(cell_dir / (base + ".gz"), "wb", compresslevel=6) as g:
                g.write(data)
            kept[base] = hashlib.sha256(data).hexdigest()
    tmp.unlink()
    return {"id": f["id"], "name": f["name"], "size": f["size"], "md5": f["supplied_md5"], "entries": entries,
            "kept_sha256": kept}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    out = data_root() / "raw" / "stroebl2024"
    out.mkdir(parents=True, exist_ok=True)
    rec = json.loads(urllib.request.urlopen(API, timeout=120).read())
    if rec["license"]["name"] != "CC BY 4.0":
        raise SystemExit(f"unexpected licence {rec['license']['name']}")
    man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {
        "record": f"10.6084/m9.figshare.{RECORD}", "version": rec.get("version"), "licence": rec["license"]["name"],
        "title": rec["title"], "zips": {}}
    meta = next(f for f in rec["files"] if f["name"] == "experiments_meta.csv")
    get(meta["download_url"], out / "experiments_meta.csv")
    if md5(out / "experiments_meta.csv") != meta["supplied_md5"]:
        raise SystemExit("experiments_meta.csv: MD5 mismatch")
    man["experiments_meta_md5"] = meta["supplied_md5"]
    todo = [f for f in rec["files"] if f["name"].endswith(".zip") and str(f["id"]) not in man["zips"]]
    print(f"{len(todo)} zips to fetch ({sum(f['size'] for f in todo) / 1e9:.2f} GB)", flush=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for i, res in enumerate(ex.map(lambda f: fetch_one(f, out), todo), 1):
            man["zips"][str(res["id"])] = res
            MANIFEST.write_text(json.dumps(man, indent=1, sort_keys=True))
            print(f"[{i}/{len(todo)}] {res['name']} ({res['id']}): kept {len(res['kept_sha256'])} files", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
