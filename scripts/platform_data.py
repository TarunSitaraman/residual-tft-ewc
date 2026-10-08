"""Read-only data access for the platform server: tables, results files, paper files and raw-file previews.

Every request names a file by its repository-relative path, and only paths in the catalogue below are served.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import lzma
import math
import re
import threading
from pathlib import Path

import numpy as np
import pandas as pd

TABLE_DIRS = {"data": "Data tables", "results/tables": "Paper result tables"}
JSON_DIR = "results"
FILE_DIRS = {"paper": {".pdf"}, "paper/figures": {".png"}, "results/figures": {".png"}, "docs/figures/spark": {".png"}}
RAW_MANIFESTS = {"S6 capacity checks": "docs/external_data/metadata/wheeler2025_characterisation_manifest.json",
                 "S6 ageing cycles": "data/raw/wheeler2025_ageing_manifest_all.json"}
RAW_BASE = "data/raw/wheeler2025"
MAX_PAGE, PREVIEW_ROWS, HIST_BINS = 500, 200, 24


def catalog(root: Path) -> dict:
    tables = [{"path": f"{d}/{p.name}", "group": g, "mb": round(p.stat().st_size / 1e6, 3)}
              for d, g in TABLE_DIRS.items() for p in sorted((root / d).glob("*.csv"))]
    results = [{"path": f"{JSON_DIR}/{p.name}", "kb": round(p.stat().st_size / 1e3, 1)} for p in sorted((root / JSON_DIR).glob("*.json"))]
    files = [{"path": f"{d}/{p.name}", "kind": p.suffix[1:]} for d, sfx in FILE_DIRS.items()
             for p in sorted((root / d).glob("*")) if p.suffix in sfx and p.is_file()]
    return {"tables": tables, "results": results, "files": files, "raw": list(RAW_MANIFESTS)}


def allowed(root: Path, rel: str, kind: str) -> Path | None:
    """The file for a catalogue path of the given kind ("tables", "results", "files"), else None."""
    if not rel or not any(e["path"] == rel for e in catalog(root)[kind]):
        return None
    return root / rel


# ---------- tables ----------
_cache: dict[str, tuple[int, pd.DataFrame]] = {}
_lock = threading.Lock()


def load(path: Path) -> pd.DataFrame:
    key, mtime = str(path), path.stat().st_mtime_ns
    with _lock:
        hit = _cache.get(key)
        if hit and hit[0] == mtime:
            return hit[1]
    df = pd.read_csv(path, low_memory=False)
    with _lock:
        if len(_cache) >= 8:
            _cache.pop(next(iter(_cache)))
        _cache[key] = (mtime, df)
    return df


_OP = re.compile(r"^\s*(>=|<=|!=|>|<|=)\s*(.+?)\s*$")


def apply_filter(df: pd.DataFrame, column: str | None, q: str) -> pd.DataFrame:
    """Rows matching q in one column (or any column if column is empty).

    q is either a comparison (">0.8", "<= 3", "=cell1a", "!=case2") or text to find (case-insensitive).
    """
    q = (q or "").strip()
    if not q:
        return df
    cols = [column] if column else list(df.columns)
    if column and column not in df.columns:
        raise KeyError(column)
    m = _OP.match(q)
    mask = pd.Series(False, index=df.index)
    for c in cols:
        s = df[c]
        if m:
            op, val = m.groups()
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                try:
                    v = float(val)
                except ValueError:
                    continue
                x = s
            else:
                v, x = val.lower(), s.astype("string").str.lower()
            hit = {">": x > v, "<": x < v, ">=": x >= v, "<=": x <= v, "=": x == v, "!=": x != v}[op]
        else:
            hit = s.astype("string").str.contains(q, case=False, regex=False)
        mask |= hit.fillna(False).astype(bool)
    return df[mask]


def _cell(v):
    if v is None or v is pd.NA or v is pd.NaT or (isinstance(v, (float, np.floating)) and not math.isfinite(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return str(v)


def table_page(path: Path, offset: int = 0, limit: int = 100, sort: str | None = None, desc: bool = False,
               column: str | None = None, q: str = "") -> dict:
    df = load(path)
    total = len(df)
    view = apply_filter(df, column, q)
    if sort:
        if sort not in df.columns:
            raise KeyError(sort)
        view = view.sort_values(sort, ascending=not desc, na_position="last", kind="stable")
    limit = max(1, min(int(limit), MAX_PAGE))
    offset = max(0, int(offset))
    page = view.iloc[offset:offset + limit]
    return {"columns": list(df.columns), "dtypes": [str(t) for t in df.dtypes], "total": total, "matched": len(view),
            "offset": offset, "rows": [[_cell(v) for v in r] for r in page.itertuples(index=False, name=None)]}


def table_stats(path: Path) -> list[dict]:
    df = load(path)
    out = []
    for c in df.columns:
        s = df[c]
        d = {"column": c, "dtype": str(s.dtype), "missing": int(s.isna().sum()), "unique": int(s.nunique(dropna=True))}
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and s.notna().any():
            v = s.dropna().astype(float)
            v = v[np.isfinite(v)]
            if len(v):
                counts, edges = np.histogram(v, bins=HIST_BINS)
                d.update(min=float(v.min()), max=float(v.max()), mean=float(v.mean()), median=float(v.median()),
                         hist=counts.tolist(), edges=[float(edges[0]), float(edges[-1])])
        else:
            top = s.astype("string").value_counts(dropna=True).head(5)
            d["top"] = [[str(k), int(n)] for k, n in top.items()]
        out.append(d)
    return out


# ---------- raw files ----------
def raw_files(root: Path, dataset: str) -> list[dict]:
    files = json.loads((root / RAW_MANIFESTS[dataset]).read_text())["files"]
    return [{"path": f["path"], "mb": round(f["size"] / 1e6, 1), "sha256": f.get("sha256", "")} for f in files]


def raw_preview(root: Path, dataset: str, rel: str, rows: int = PREVIEW_ROWS) -> dict:
    """The first rows of one raw file named in a download manifest, decompressed on the fly."""
    entry = next((f for f in raw_files(root, dataset) if f["path"] == rel), None)
    if entry is None:
        raise KeyError(rel)
    base = root / RAW_BASE
    path = next((p for p in (base / (rel + ".xz"), base / (rel + ".gz"), base / rel) if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(rel)
    opener = lzma.open if path.suffix == ".xz" else gzip.open if path.suffix == ".gz" else open
    lines = []
    with opener(path, "rb") as f:
        for i, line in enumerate(f):
            lines.append(line.decode("utf-8", "replace"))
            if i >= rows:
                break
    sample = "".join(lines)
    try:
        dialect = csv.Sniffer().sniff(lines[0] if lines else ",", delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    parsed = list(csv.reader(io.StringIO(sample), dialect))
    return {**entry, "stored": str(path.relative_to(root)).replace("\\", "/"), "stored_mb": round(path.stat().st_size / 1e6, 1),
            "header": parsed[0] if parsed else [], "rows": parsed[1:rows + 1]}
