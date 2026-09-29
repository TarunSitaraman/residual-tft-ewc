"""Download the OSF second-life archive (S1, version 2) and extract its diagnostic tests.

The diagnostic tests (``diagnostic_tests/RPT_#``) hold the raw RPT capacity
tests with pulses and the EIS spectra that the handover CSVs do not contain.
Source, licence (CC BY 4.0) and citation: docs/EXTERNAL_DATA.md, S1.

Steps:
1. download the archive (resumable) and check its size and SHA-256 against the
   OSF file record; stop on any mismatch;
2. extract ``diagnostic_tests/`` only (zip CRC-32 checked on extraction);
3. write a manifest with the size and SHA-256 of every extracted file.

Raw files go to ``data/raw/`` (git-ignored); only the manifest is versioned.

Usage (from the repository root)::

    python scripts/fetch_osf_diagnostics.py \
        --dest data/raw/osf_8jnr5_v2 \
        --manifest docs/external_data/metadata/osf_8jnr5_v2_diagnostic_manifest.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_external_data import ARCHIVE_SHA256, ARCHIVE_SIZE, OSF_FILE_URL  # noqa: E402

ROOT = "SL_Dataset_SECL_INR21700-M50T/"
PREFIX = ROOT + "diagnostic_tests/"
CHUNK = 8 << 20


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, path: Path, attempts: int = 8) -> None:
    """Resumable download: continue from the bytes already on disk."""
    for k in range(attempts):
        have = path.stat().st_size if path.exists() else 0
        if have == ARCHIVE_SIZE:
            return
        if have > ARCHIVE_SIZE:
            raise RuntimeError(f"{path} is larger than the registered archive; delete it and retry")
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp, open(path, "ab") as out:
                if have and resp.status != 206:
                    raise RuntimeError(f"server ignored the resume request (HTTP {resp.status})")
                while block := resp.read(CHUNK):
                    out.write(block)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
            print(f"download interrupted at {path.stat().st_size:,} bytes ({err}); retrying", file=sys.stderr)
            time.sleep(min(2 ** (k + 1), 60))
    if path.stat().st_size != ARCHIVE_SIZE:
        raise RuntimeError("download incomplete after retries")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dest", default="data/raw/osf_8jnr5_v2")
    ap.add_argument("--manifest", default="docs/external_data/metadata/osf_8jnr5_v2_diagnostic_manifest.json")
    ap.add_argument("--keep-archive", action="store_true", help="keep the 2.15 GB zip after extraction")
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "SL_Dataset_SECL_INR21700-M50T.zip"
    download(OSF_FILE_URL, archive)
    digest = sha256_of(archive)
    if digest != ARCHIVE_SHA256:
        raise RuntimeError(f"SHA-256 mismatch: got {digest}, registered {ARCHIVE_SHA256}")
    print("archive verified:", archive.stat().st_size, "bytes, sha256", digest)

    files = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = info.filename
            if info.is_dir() or not name.startswith(PREFIX) or name.endswith(".DS_Store"):
                continue
            target = dest / name[len(ROOT):]
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:   # ZipFile checks the CRC-32 at EOF
                while block := src.read(CHUNK):
                    out.write(block)
            files.append({"path": name[len(ROOT):], "size": info.file_size,
                          "crc32": f"{info.CRC:08x}", "sha256": sha256_of(target)})
    if not args.keep_archive:
        archive.unlink()

    manifest = {
        "source": "S1 in docs/EXTERNAL_DATA.md: doi:10.17605/OSF.IO/8JNR5, file version 2 (CC BY 4.0)",
        "url": OSF_FILE_URL, "archive_size": ARCHIVE_SIZE, "archive_sha256": ARCHIVE_SHA256,
        "extracted": PREFIX, "retrieved_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_files": len(files), "total_bytes": sum(f["size"] for f in files), "files": files,
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=1))
    print(f"extracted {len(files)} files ({manifest['total_bytes']:,} bytes) to {dest}; manifest {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
