# Apache Spark in this project, and how to demonstrate it

## What Spark does here

The S6 between-check ageing data (decision D22) is the project's large dataset: 827 files, one per cell and 100-cycle
block, about 132 GB raw (16 GB compressed) and about 1.4 billion rows. `scripts/spark_ageing_features.py` turns it
into the 21 block-level ageing inputs with Apache Spark 4.2 (PySpark), as a map-reduce job:

- **Map:** one Spark task per block parses the raw rows into per-cycle measured responses (end-of-charge and
  end-of-discharge voltage, CV-hold time and final current, relaxation after charge and after discharge, internal
  resistance) and a validity flag, using the tested function `src/ageing.py::cycle_responses`.
- **Cache:** the per-cycle table (about 83,000 cycles) is cached in Spark memory.
- **Reduce:** Spark SQL groups the valid cycles by (cell, block) and computes the mean, the last value and the
  least-squares slope against cycle number (covariance / variance), with at least 50 valid cycles per block.
- **Validation:** the output is compared with the pandas implementation (`scripts/build_s6_ageing_features.py`);
  `results/d22_spark_validation.json` records the comparison.

Spark runs in **local mode on one machine** (`local[N]` worker threads); there is no cluster. Say so when
presenting.

![Pipeline](../../paper/figures/spark_pipeline.png)

## Requirements

- Java 17 or 21 with `JAVA_HOME` set (this PC: Java 21).
- `pyspark==4.2.0` (in `requirements.txt`; installed in `.venv`).
- No Hadoop installation is needed. Only the optional event log / History Server needs Hadoop's `winutils.exe` on
  Windows.

## Live demo for the review (about 5 minutes)

1. Close heavy applications (the job needs about 2 GB of free memory).
2. From the repository root, start a demo run on 40 blocks spread over the cells, with the web UI kept open for
   30 minutes:

   ```powershell
   .venv\Scripts\python.exe scripts\spark_ageing_features.py --cores 4 --limit 40 --ui 1800
   ```

3. Open **http://localhost:4040** in a browser while it runs. Show, in order:
   - **Jobs:** the `MAP: parse ... ageing blocks` job and its tasks progressing, then the `REDUCE: Spark SQL
     aggregation` jobs.
   - **Stages:** one task per block in the map stage; the shuffle between the two aggregation stages.
   - **Storage / Executors:** the cached per-cycle table and the worker cores.
   - **SQL / DataFrame:** the physical plan (scan, in-memory table, hash aggregate, exchange, hash aggregate).
4. When the job finishes, the terminal prints how many blocks matched the pandas implementation. The UI stays open
   for the time given with `--ui`; a REST snapshot of it is saved to `data/spark_ui_snapshot.json`.

Screenshots from a run on 16 blocks (5 Oct 2026) are in `docs/figures/spark/` for the slides.

## Full run

```powershell
.venv\Scripts\python.exe scripts\spark_ageing_features.py --cores 8
```

writes `data/s6_ageing_blocks_spark.csv` (the D22 inputs) and `results/d22_spark_validation.json`.

## Run dashboard

`docs/spark_dashboard.html` (published at https://claude.ai/artifact/4nn8GK3uk7xyHEW59LPBua) shows a full Spark run
recorded through Spark's monitoring API: the timeline of all 827 map tasks across the six worker slots, task
durations, time against file size, the Spark jobs, data per cell, an explorer for the block-level ageing inputs, and
the validation against pandas. It stays viewable without Spark running.

To rebuild it from a new run:

```powershell
.venv\Scripts\python.exe scripts\spark_ageing_features.py --cores 6 --ui 900 --tag dashboard
.venv\Scripts\python.exe scripts\make_spark_dashboard.py
```

The re-run of 7 Oct 2026 (43 min; other processes were competing for the CPU) gave block inputs identical to the
first full run of 6 Oct (19.6 min), which supplied the D22 inputs (largest difference 0.0).
