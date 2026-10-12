# D24 night run (ASUS)

Runbook for the one-night D24 run (decision D24 in `docs/DECISIONS.md`, work plan `docs/WORK_PLAN.md`). Everything that needs the datasets happens here; everything else is already in the repository.

## What runs

| Step | What | Expected time (8 workers) |
|---|---|---|
| Pre-flight | clean tree, data present, libraries | seconds |
| Smoke | every part once, 1 order, 2 epochs (not reportable) | 5–10 min |
| Parts 1–4 | E1: RC-TFT, BiLSTM, TCN, Transformer (residual) | RC-TFT about 1.5–2 h; the others about 1 h each |
| Parts 5–8 | E2 (horizons 1–6): the same four | RC-TFT about 2.5–3 h; the others about 1.5 h each |
| Parts 9–10 | E1 and E2 baselines: persistence, extrapolation, ridge, gradient boosting, GPR (at most 1,000 rows), random forest, SVR, direct and residual | about 30–45 min each |
| Parts 11–13 | E1: BiLSTM, TCN, Transformer (direct) | about 1 h each |
| Collect | D23 reproduction check (2 units), tests, `results/forecasting_d24_e1.json`, `..._e2.json` | a few minutes |
| Spark scaling (optional) | 120 blocks at 1, 2, 4, 6 threads | about 35 min |
| Outputs | tables and figures | 1 min |

Estimates come from D23's recorded training times (TFT 23–37 s per stream in E1, 37–59 s in E2) and a benchmark of the new backbones at full epochs (BiLSTM, TCN and Transformer streams take a quarter to a third of the TFT's time). Expect about 4–5 h for the model runs.

## Before starting

1. Pull `main` (the D24 code is merged there) and check the tree is clean:
   ```powershell
   git fetch origin
   git checkout main
   git pull
   git status            # must show nothing to commit
   ```
2. Git identity (commits must be under the team member's own name):
   ```powershell
   git config user.name "Tarun Sitaraman"
   git config user.email "mfsbyo@gmail.com"
   ```
3. Environment (Python 3.11, pinned versions):
   ```powershell
   pip install -r requirements.txt
   python -m pytest -q                       # expect all tests to pass, none skipped for missing data
   ```
4. Data present (git-ignored, built earlier on this machine):
   - `data/wheeler_checks.csv` (else `python scripts/build_wheeler_features.py`)
   - for the Spark scaling run: `data/raw/wheeler2025/Cell*/agingdata_50degC/*aging*.csv.*`, `data/s6_ageing_blocks.csv`, Java 17/21 (`JAVA_HOME`)
5. Close other heavy programs and disable sleep for the night (Settings → System → Power: never sleep when plugged in). Keep the laptop on mains power.

## Optional, before the night: classical-ML baselines (about 1 h)

The GPR, random forest and SVR baselines (with persistence, linear extrapolation, ridge and gradient boosting) can run on their own, before the model parts, and their checkpoint is reused by the night run:

```powershell
python scripts/run_d24.py --exp e1 --baselines-only
python scripts/run_d24.py --exp e2 --baselines-only
python scripts/make_d24_outputs.py
git add results/forecasting_d24_baselines_e1.json results/forecasting_d24_baselines_e2.json results/tables paper/figures paper/tables
git commit -m "D24 baselines part: GPR, random forest, SVR and the D23 baselines on the S6 stream"
git push
```

## Start

From the repository root, choose the deadline as the time after which nothing new should start (for example 07:30):

```powershell
python scripts/run_d24_night.py --dry-run                                  # pre-flight and plan only
python scripts/run_d24_night.py --workers 8 --deadline 07:30 --spark-scaling
```

`--workers` should be the number of physical cores (8 on the Core 5 210H). Leave the window open.

## Watching it

- `data/cache/d24/logs/night.log`: one line per event.
- `data/cache/d24/logs/status_paper.json`: running, queued, finished parts.
- `data/cache/d24/logs/paper_<exp>_<part>.log`: one log per part.

If the machine restarts, run the same command again: every unit is checkpointed under `data/cache/d24/`, so it resumes. Do not edit any file under `src/` during the run: the checkpoints are keyed on the source code, and an edit starts everything afresh.

## In the morning

1. Read `night.log`. A part listed as `FAILED` or `not started (deadline)` is reported as not run (D24); do not shorten or re-run parts selectively to change results.
2. If the collection steps did not run (for example after a restart), run them by hand:
   ```powershell
   python scripts/run_d24.py --exp e1 --collect-only
   python scripts/run_d24.py --exp e2 --collect-only
   python scripts/make_d24_outputs.py
   python scripts/make_preprocessing_table.py
   ```
3. Check that `results/forecasting_d24_e1.json` records `"git_dirty": false` and lists the parts run and not run. Then write the report from the files only:
   ```powershell
   python scripts/summarise_d24.py --out data/cache/d24/d24_summary.md
   ```
4. Commit the outputs (no other files):
   ```powershell
   git add results/forecasting_d24_e1.json results/forecasting_d24_e2.json results/d22_spark_scaling.json results/d22_spark_validation_scale_c*.json results/preprocessing_funnel.json results/tables paper/figures paper/tables
   git commit -m "D24 results: RC-TFT and literature baselines on the S6 stream (E1, E2); Spark scaling run"
   git push
   ```
5. Add a run-log entry under D24 in `docs/DECISIONS.md` (start and end times, workers, parts not run, any failure), then write the D24 section of `docs/RESULTS.md` from the results files.

## If something fails

- **Pre-flight refuses:** fix what it names; never use `--allow-dirty` for the real run.
- **Smoke fails:** the paper run does not start. Open `data/cache/d24/logs/smoke_<exp>_<part>.log` for the error.
- **"committed D23 runs do not reproduce":** the environment differs from the D23 run (library versions). Stop and record it; do not compare D24 with D23 runs in that case.
- **A part fails during the night:** the others continue. Its log has the error; record it as not run.
