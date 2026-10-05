"""D22 ageing inputs with Apache Spark (PySpark, local mode), validated against the pandas implementation.

Map: one Spark task per 100-cycle ageing block parses the raw file into per-cycle responses
(``src.ageing.cycle_responses``, the tested function the pandas builder uses). Reduce: Spark SQL aggregates the
valid cycles of each (cell, block) into the F2 inputs: mean, last value and least-squares slope against cycle
(covariance / variance), with the minimum of ``MIN_VALID_CYCLES`` valid cycles. The result is compared with
``data/s6_ageing_blocks.csv`` from ``build_s6_ageing_features.py`` (pandas) on the blocks both contain.

Spark runs in local mode on one machine (``--cores`` worker threads); no cluster is involved. Pandas interop
inside Spark is avoided (PySpark 4.2 does not fully support pandas 3).

Usage (from the repository root)::

    python scripts/spark_ageing_features.py [--cores 4] [--limit N]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import ageing as A  # noqa: E402
from src import experiments as X  # noqa: E402

RAW = ROOT / "data" / "raw" / "wheeler2025"
OUT = ROOT / "data" / "s6_ageing_blocks_spark.csv"
FIELDS = ["cell", "block_end", "global_cycle", "valid", "t_end", *A.RESPONSES]


def parse_block(job: tuple[str, float, bool]) -> list[tuple]:
    """Spark map task: one raw block -> per-cycle rows (NaN as None, so Spark sees nulls)."""
    from src import ageing as A   # imported inside the task: runs in the Python worker
    path, v_limit, cccv = job
    cell, first, last = A.block_range(path)
    c = A.cycle_responses(A.read_block(path), first, v_limit, cccv)
    rows = []
    for r in c.itertuples(index=False):
        d = r._asdict()
        vals = [cell.lower(), last, int(d["global_cycle"]), bool(d["valid"]), float(d["t_end"])]
        vals += [None if (d.get(k) is None or (isinstance(d.get(k), float) and math.isnan(d[k]))) else float(d[k])
                 for k in A.RESPONSES]
        rows.append(tuple(vals))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cores", type=int, default=4, help="local[N] worker threads")
    ap.add_argument("--limit", type=int, default=0, help="only the first N blocks (testing)")
    args = ap.parse_args()

    import build_s6_ageing_features as B
    files = sorted([*RAW.glob("Cell*/agingdata_50degC/*aging*.csv.xz"), *RAW.glob("Cell*/agingdata_50degC/*aging*.csv.gz")])
    settings = B.case_settings()
    jobs = [(str(f), *settings[A.block_range(f)[0].lower()]) for f in files]
    if args.limit:
        jobs = jobs[::max(1, len(jobs) // args.limit)][:args.limit]     # spread over cells

    py = sys.executable
    os.environ.update({"PYSPARK_PYTHON": py, "PYSPARK_DRIVER_PYTHON": py, "PYTHONPATH": str(ROOT)})
    from pyspark.sql import SparkSession, functions as F, types as T

    spark = (SparkSession.builder.master(f"local[{args.cores}]").appName("d22-ageing")
             .config("spark.driver.memory", "1g").config("spark.ui.enabled", "false")
             .config("spark.python.worker.reuse", "true").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    schema = T.StructType([T.StructField("cell", T.StringType()), T.StructField("block_end", T.IntegerType()),
                           T.StructField("global_cycle", T.IntegerType()), T.StructField("valid", T.BooleanType()),
                           T.StructField("t_end", T.DoubleType()),
                           *[T.StructField(k, T.DoubleType()) for k in A.RESPONSES]])
    t0 = time.perf_counter()
    cycles = spark.createDataFrame(spark.sparkContext.parallelize(jobs, len(jobs)).flatMap(parse_block), schema).cache()

    # Reduce: per (cell, block) over valid cycles; each statistic uses the cycles where that response is present.
    aggs = [F.sum(F.col("valid").cast("int")).alias("age_valid_cycles"),
            F.sum((~F.col("valid")).cast("int")).alias("age_invalid_cycles"),
            F.max("t_end").alias("t_end")]
    for r in A.RESPONSES:
        v = F.when(F.col("valid"), F.col(r))                       # null for invalid cycles
        x = F.when(v.isNotNull(), F.col("global_cycle").cast("double"))
        n = F.count(v)
        aggs += [F.avg(v).alias(f"age_{r}_mean"),
                 F.max_by(v, x).alias(f"age_{r}_last"),
                 F.when(n >= 2, F.covar_pop(x, v) / F.var_pop(x)).alias(f"age_{r}_slope")]
    blocks = cycles.groupBy("cell", "block_end").agg(*aggs)
    enough = F.col("age_valid_cycles") >= A.MIN_VALID_CYCLES
    blocks = blocks.select("cell", "block_end", "t_end", "age_valid_cycles", "age_invalid_cycles",
                           *[F.when(enough, F.col(c)).alias(c) for c in A.feature_names()])
    rows = [r.asDict() for r in blocks.orderBy("cell", "block_end").collect()]
    n_cycles = cycles.count()
    elapsed = time.perf_counter() - t0
    spark.stop()

    out = pd.DataFrame(rows).astype({c: float for c in A.feature_names()})
    out.to_csv(OUT, index=False)

    # Validation against the pandas implementation (blocks present in both).
    ref = pd.read_csv(ROOT / "data" / "s6_ageing_blocks.csv")
    m = out.merge(ref, on=["cell", "block_end"], suffixes=("_spark", "_pandas"))
    diffs = {}
    for c in ["age_valid_cycles", "age_invalid_cycles", *A.feature_names()]:
        a, b = m[f"{c}_spark"].to_numpy(float), m[f"{c}_pandas"].to_numpy(float)
        same_missing = bool(np.array_equal(np.isnan(a), np.isnan(b)))
        both = ~(np.isnan(a) | np.isnan(b))
        rel = np.abs(a[both] - b[both]) / np.maximum(np.abs(b[both]), 1e-12)
        diffs[c] = {"same_missing": same_missing, "max_abs": float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0,
                    "max_rel": float(rel.max()) if both.any() else 0.0}
    ok = all(d["same_missing"] and d["max_rel"] < 1e-9 for d in diffs.values())
    from pyspark import __version__ as spark_version
    X.write_results(ROOT / "results" / ("d22_spark_validation.json" if not args.limit else "d22_spark_validation_test.json"), {
        "decision": "D22", "engine": f"Apache Spark {spark_version} (PySpark), local[{args.cores}] on one machine",
        "blocks": len(jobs), "cycles": int(n_cycles), "seconds": round(elapsed, 1),
        "compared_blocks": int(len(m)), "identical_within_1e-9_relative": ok, "per_input": diffs}, ROOT)
    print(f"Spark local[{args.cores}]: {len(jobs)} blocks, {n_cycles} cycles in {elapsed:.0f} s; "
          f"compared {len(m)} blocks with pandas: {'identical' if ok else 'DIFFERENT'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
