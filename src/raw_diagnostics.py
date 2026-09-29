"""Readers for the raw diagnostic tests of the public second-life archive (decision D15).

The diagnostic tests (``diagnostic_tests/RPT_#``) are fetched by
``scripts/fetch_osf_diagnostics.py`` into ``data/raw/osf_8jnr5_v2``. Each
reference performance test (RPT) contains, per cell:

* a *capacity test with pulses* (Arbin ``.xlsx``): C/20 discharge from 4.2 V to
  2.5 V, interrupted at 4.00, 3.63 and 3.26 V by an 8 s C/2 discharge pulse,
  10 s rest and 8 s C/2 charge pulse (Moy et al., 2024, Table 5);
* from RPT 5 on, EIS at the same three voltages, 10 kHz to 0.01 Hz (Gamry,
  stored in the ``ACIM_*`` sheet of the Arbin file).

Reading the ``.xlsx`` files is slow, so each file is converted once to a pickle
cache (``data/cache/raw_diagnostics``, git-ignored). The cache stores the raw
sheet content unchanged; no values are filtered or resampled.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src import data_pipeline as dp

RAW_SUBDIR = Path("raw") / "osf_8jnr5_v2" / "diagnostic_tests"
CACHE_SUBDIR = Path("cache") / "raw_diagnostics"

_NAME = re.compile(
    r"^(?P<date>\d{8})_(?:sd_)?(?P<battery>[A-Za-z]+\d+)_"
    r"(?:(?P<rpt>RPT)|EIS_?(?P<volt>\d{3})V?)(?P<suffix>.*?)_Channel_(?P<channel>\d+)(?:_\d+)?\.xlsx$"
)
EIS_VOLTAGES = {"326": 3.26, "363": 3.63, "400": 4.00}


@dataclass(frozen=True)
class RawFile:
    """One per-cell raw file of the archive, identified from its path."""

    path: Path
    rpt_number: int
    battery: str
    date: pd.Timestamp
    kind: str                  # "RPT" or "EIS"
    eis_voltage: float | None
    suffix: str                # anything between the test name and the channel (e.g. "_2", "_gal")

    @property
    def key(self) -> str:
        return self.path.name


def parse_raw_name(path: Path | str) -> RawFile | None:
    """Identify a per-cell raw file; ``None`` for aggregate files (``*_ALL.xlsx``, ``.mat``)."""
    path = Path(path)
    m = _NAME.match(path.name)
    folder = next((p for p in path.parts if re.fullmatch(r"RPT_\d+", p)), None)
    if not m or folder is None:
        return None
    return RawFile(
        path=path,
        rpt_number=int(folder.split("_")[1]),
        battery=m["battery"].lower(),
        date=pd.to_datetime(m["date"], format="%Y%m%d"),
        kind="RPT" if m["rpt"] else "EIS",
        eis_voltage=None if m["rpt"] else EIS_VOLTAGES.get(m["volt"]),
        suffix=m["suffix"] or "",
    )


def list_raw_files(raw_root: Path) -> pd.DataFrame:
    """Inventory of per-cell raw files with parsed identity; unparsed files listed with ``kind=None``."""
    rows = []
    for p in sorted(raw_root.rglob("*.xlsx")):
        f = parse_raw_name(p)
        rows.append({"path": p, "file": p.name, "rpt_number": f.rpt_number if f else None,
                     dp.GROUP_COL: f.battery if f else None, "date": f.date if f else pd.NaT,
                     "kind": f.kind if f else None, "eis_voltage": f.eis_voltage if f else None,
                     "suffix": f.suffix if f else None})
    return pd.DataFrame(rows)


def _read_sheets(path: Path) -> dict[str, pd.DataFrame]:
    """All measurement sheets of an Arbin workbook: ``Channel_*`` (time series) and ``ACIM_*`` (EIS)."""
    xl = pd.ExcelFile(path)
    wanted = [s for s in xl.sheet_names if s.startswith("Channel_") or s.startswith("ACIM")]
    return {s: xl.parse(s) for s in wanted}


def cached_sheets(path: Path, cache_root: Path) -> dict[str, pd.DataFrame]:
    """Sheets of ``path``, read from the pickle cache or converted into it on first use."""
    target = cache_root / (path.name + ".pickle")
    if target.exists() and target.stat().st_mtime >= path.stat().st_mtime:
        return pd.read_pickle(target)
    sheets = _read_sheets(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(sheets, target)
    return sheets


def timeseries(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The Arbin time series: ``Channel_*`` sheets concatenated in order (long tests are split)."""
    parts = [sheets[k] for k in sorted((k for k in sheets if k.startswith("Channel_")),
                                       key=lambda s: [int(x) for x in re.findall(r"\d+", s)])]
    if not parts:
        return pd.DataFrame()
    ts = pd.concat(parts, ignore_index=True)
    # The unit in the temperature header is mis-encoded in some files ("(¡æ)"); normalise the name
    # and keep the sensor number.
    temps = [c for c in ts.columns if str(c).startswith("Aux_Temperature")]
    ts = ts.rename(columns={c: f"Aux_Temperature(C)_{str(c).rsplit('_', 1)[-1]}" for c in temps})
    return ts.sort_values("Test_Time(s)", kind="mergesort").reset_index(drop=True)


def impedance(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """EIS spectrum from the ``ACIM_*`` sheet: frequency (Hz), Z' and Z'' (ohm), sorted by frequency.

    The sheet stores modulus ``Zmod`` and phase ``Zphz`` in degrees. Z'' is
    ``Zmod * sin(phase)``, so capacitive behaviour has Z'' < 0 (Nyquist plots
    show -Z'').
    """
    key = next((k for k in sheets if k.startswith("ACIM")), None)
    if key is None:
        return pd.DataFrame(columns=["freq_hz", "z_real", "z_imag"])
    s = sheets[key]
    phase = np.deg2rad(s["Zphz"].astype(float))
    out = pd.DataFrame({"freq_hz": s["Freq"].astype(float),
                        "z_real": s["Zmod"].astype(float) * np.cos(phase),
                        "z_imag": s["Zmod"].astype(float) * np.sin(phase)})
    return out.sort_values("freq_hz", ascending=False, kind="mergesort").reset_index(drop=True)


def convert_all(raw_root: Path, cache_root: Path, workers: int = 4, files: Iterable[Path] | None = None) -> int:
    """Fill the cache for every per-cell raw file (parallel); returns the number of files converted."""
    from concurrent.futures import ProcessPoolExecutor

    paths = [Path(p) for p in (files if files is not None else list_raw_files(raw_root)["path"])]
    todo = [p for p in paths if not (cache_root / (p.name + ".pickle")).exists()]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(cached_sheets, todo, [cache_root] * len(todo)))
    return len(todo)
