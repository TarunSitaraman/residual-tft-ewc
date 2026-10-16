# D26 night run (ASUS)

Runbook for the one-night D26 run (decision D26 in `docs/DECISIONS.md`: unseen usage conditions, P2, for the residual TFT + EWC, stratified by origin history). Everything that needs the datasets happens here; everything else is already in the repository.

## What runs

| Step | What | Expected time (6 workers) |
|---|---|---|
| Pre-flight | clean tree, S6 check table, D23 results (tuning source), libraries, versions against `requirements.txt` | seconds |
| Smoke | every part once, 1 order, 1 seed, 2 epochs (not reportable) | 15–30 min |
| 28 parts: `main_f0` … `main_f6`, `replicate_f0` … `replicate_f6`, for E2 then E1 | per held-out case: 10 orders × 5 seeds continual streams over the other six cases + 5 joint fits | E2 about 55 min each, E1 about 35 min each |
| Collect | single-fit comparators per fold, tests, coverage check, `results/forecasting_d26_e1.json`, `..._e2.json` | a few minutes |
| Summary | `data/cache/d26/d26_summary.md` | seconds |

Expect about 4 h in all (timed on synthetic data of S6 size: one continual unit 34 s (E1) / 55 s (E2), one joint fit 47 s / 94 s, single-threaded). The parts have no dependencies on each other.

## Before starting

1. Pull `main` and check the tree is clean:
   ```powershell
   git fetch origin
   git checkout main
   git pull
   git status            # must show nothing to commit
   ```
   If VS Code has re-saved `notebooks/colab_review2.ipynb` (kernel metadata only), stash it, and close VS Code for the night: it can dirty the tree again during the run.
2. Git identity:
   ```powershell
   git config user.name "Tarun Sitaraman"
   git config user.email "mfsbyo@gmail.com"
   ```
3. **Pinned environment** (the existing `.venv` is Python 3.11.15 with the pinned versions; it was used for D25):
   ```powershell
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   python -m pytest -q                       # all tests pass
   ```
   If a pinned version cannot be installed, stop and record why; only then use `--allow-unpinned`, and write the versions into the D26 run log.
4. Data present (git-ignored, built earlier on this machine): `data/wheeler_checks.csv`. D26 does not use the pool tables.
5. Close other heavy programs (VS Code, browsers), disable sleep, keep the laptop on mains power:
   ```powershell
   powercfg /change standby-timeout-ac 0
   powercfg /change hibernate-timeout-ac 0
   ```

## Start

From a standalone PowerShell window (not a terminal inside VS Code):

```powershell
python scripts/run_d26_night.py --dry-run                          # pre-flight and plan only
python scripts/run_d26_night.py --workers 6 --deadline 07:30
```

Leave the window open. Do not edit anything under `src/` during the run: checkpoints are keyed on the source code.

## Watching it

- `data/cache/d26/logs/night.log`: one line per event.
- `data/cache/d26/logs/status_paper.json`: running, queued, finished parts.
- `data/cache/d26/logs/paper_<exp>_<part>.log`: one log per part.

If the machine restarts or runs out of memory, run the same command again (lower `--workers` if memory was the problem): every unit is checkpointed under `data/cache/d26/`, so it resumes. Record every restart and its reason.

## In the morning

1. Read `night.log`. A part listed as `FAILED` or `not started` is reported as not run; do not shorten or re-run parts selectively to change results. Runs, comparators and tests use only the folds that finished in both seed sets (`folds_in_tests` in the results file).
2. If collection did not run, run it by hand:
   ```powershell
   python scripts/run_d26.py --exp e1 --collect-only
   python scripts/run_d26.py --exp e2 --collect-only
   python scripts/summarise_d26.py --out data/cache/d26/d26_summary.md
   ```
3. Check that both results files record `"git_dirty": false` (under `environment`), the parts run and not run, `folds_in_tests`, `strata_counts` and `coverage_check`.
4. Commit the results files only:
   ```powershell
   git add results/forecasting_d26_e1.json results/forecasting_d26_e2.json
   git commit -m "D26 results: P2 (unseen usage condition) for the residual TFT + EWC, by origin history (E1, E2)"
   git push
   ```
5. Add a run-log entry under D26 in `docs/DECISIONS.md`: date, start, smoke-passed, paper-run end and collection times, commit run from, workers, environment (Python and library versions), every restart and its reason, parts not run. Commit and push it separately.

## If something fails

- **Pre-flight refuses:** fix what it names. Never use `--allow-dirty` for the real run.
- **Smoke fails:** the paper run does not start; open `data/cache/d26/logs/smoke_<exp>_<part>.log`.
- **A part fails during the night:** the others continue. Record it as not run; its fold drops out of the tests for that experiment.
