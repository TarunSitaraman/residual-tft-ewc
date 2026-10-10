# Apache Spark in this project, and how to demonstrate it

## What Spark does here

The S6 between-check ageing data (decision D22) is the project's large dataset: 827 files, one per cell and 100-cycle
block, about 132 GB raw (about 15 GB as stored, gz/xz) and 901,974,259 data rows (`results/s6_ageing_row_count.json`). `scripts/spark_ageing_features.py` turns it
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
- No Hadoop installation is needed for the job itself. The event log and the Spark History Server need Hadoop's
  Windows helpers, which Apache does not ship: `winutils.exe` and `hadoop.dll` from the community repository
  github.com/cdarlint/winutils (folder `hadoop-3.3.6`, the newest available; PySpark 4.2 bundles the Hadoop 3.5.0
  client), placed in `C:\Users\Tarun\tools\hadoop-3.3.6\bin` and checked against the repository's own file hashes.
  SHA-256: winutils.exe `496a591eb1e67df2a620f710d529ba6ddfe1c19149e6647cc4e320bb0efd8553`, hadoop.dll
  `d7ab36a68518748cef142be2da5069b4c763c2cd764c1d2e6ac48c7200405be3`. They are used only through `HADOOP_HOME` in the
  launch scripts, not installed system-wide.

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

## Data platform page

The project's data platform page is a live web page served from this PC:

```powershell
.venv\Scripts\python.exe scripts\platform_server.py --lan   # http://localhost:8050; --lan also serves it on the local network
```

The server rebuilds the page whenever a results file, derived table, download manifest or the Spark snapshot changes,
and open pages reload themselves. A status bar shows what is running now: a Spark job with its task progress (when the
job runs with `--ui`), the History Server and its latest application, and any pipeline script from `scripts/`. Without
`--lan` only this PC can open it; with `--lan` Windows may ask to allow Python through the firewall. A static snapshot
is written to `docs/spark_dashboard.html` by `scripts/make_spark_dashboard.py`.

Besides the summaries, the served page is a read-only data browser:

- **Tables:** every CSV in `data/` and `results/tables/`, with search (text or comparisons such as `>0.8`, `=cell1a`),
  sorting, paging and per-column statistics with histograms.
- **Raw files:** the first 200 rows of any S6 capacity-check or ageing file, decompressed from the verified copy, with
  its original size and SHA-256.
- **Results:** every results JSON as an expandable tree.
- **Paper:** the compiled PDF and every figure.

Only files in the catalogue (`scripts/platform_data.py`) are served; nothing can be changed from the page.

The page Its tabs show the dataset catalogue and lineage (Overview, Datasets), every pipeline (Pipelines), all
recorded runs (Runs), the Spark run and the ageing-input explorer. The Spark views show how the pipeline is built (source, ingest, map, cache, reduce, validation), the Spark configuration, Spark UI and
History Server screenshots, the three full runs, and a full run recorded through Spark's monitoring API: the timeline of all 827 map tasks across the six worker slots, task
durations, time against file size, the Spark jobs, data per cell, an explorer for the block-level ageing inputs, and
the validation against pandas. It stays viewable without Spark running.

To rebuild it from a new run:

```powershell
.venv\Scripts\python.exe scripts\spark_ageing_features.py --cores 6 --ui 900 --tag dashboard
.venv\Scripts\python.exe scripts\make_spark_dashboard.py
```

The re-run of 7 Oct 2026 (43 min; other processes were competing for the CPU) gave block inputs identical to the
first full run of 6 Oct (19.6 min), which supplied the D22 inputs (largest difference 0.0).

## Official Spark UI for a finished run (History Server)

The Spark History Server replays a finished run in the official Spark web UI (Jobs, Stages with DAG and event
timeline, Storage, Environment, Executors, SQL), without re-running it.

1. Record a run with the event log (once; outputs are tagged so the D22 files are untouched):

   ```powershell
   $env:HADOOP_HOME = "C:\Users\Tarun\tools\hadoop-3.3.6"
   .venv\Scripts\python.exe scripts\spark_ageing_features.py --cores 6 --event-log --tag eventlog
   ```

2. Start the server and open **http://localhost:18080**:

   ```powershell
   pwsh scripts\spark_history_server.ps1
   ```

   Because the repository path contains a space, the script reads `data\spark-events` through a directory junction
   at `%USERPROFILE%\spark-events` (created if missing). Stop the server with Ctrl+C.

Spark runs as `local[N,4]` (up to 4 attempts per task): on Windows a Python worker can occasionally be slow to start
and time out; the task is then retried, which cannot change the results because each task is deterministic.
