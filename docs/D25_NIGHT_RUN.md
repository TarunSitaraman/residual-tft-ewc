# D25 night run (ASUS)

Runbook for the one-night D25 run (decision D25 in `docs/DECISIONS.md`, plan in `docs/WORK_PLAN.md`, section "D25"). Everything that needs the datasets happens here; everything else is already in the repository.

## What runs

| Step | What | Expected time (6 workers) |
|---|---|---|
| Pre-flight | clean tree, S6 and pool tables, D23 results, libraries, versions against `requirements.txt` | seconds |
| Smoke | every part once, 1 order, 2 epochs (not reportable) | 5–10 min |
| `pre_tune` (E2, E1) | pretrain the residual TFT on the S7–S9 pool at each learning rate, tune lr and λ on validation streams | 1–2 h each |
| `scratch_main`, `scratch_replicate` (E1, E2) | the D23 residual TFT + EWC with shrinkage and intervals, seeds 0–4 / 5–9; every raw unit must reproduce D23 | 35–50 min each |
| `pre_main`, `pre_replicate` (E1, E2) | pretrained residual TFT + EWC with shrinkage and intervals; start when that experiment's `pre_tune` is done | 1–1.5 h each |
| Collect | D23 reproduction check (all 100 scratch units per experiment), tests, `results/forecasting_d25_e1.json`, `..._e2.json` | a few minutes |
| Summary | `data/cache/d25/d25_summary.md` | seconds |

Expect about 3.5–4 h in all. Measured in the cloud rehearsal (4 workers, 15–16 Oct): scratch parts 16–18 min (E1) and 27 min (E2); `pre_tune` 51 min (E1); pretrained evaluation parts 35 min (E1) and 64–73 min (E2); about 3 h end to end with 6 workers.

Run it on the ASUS, not on other hardware: the reproduction check needs the D23 units to match to 1e-9 pp, which the cloud rehearsal's CPU did not (differences up to 0.015 pp; see the D25 run log).

## Before starting

1. Pull `main` and check the tree is clean:
   ```powershell
   git fetch origin
   git checkout main
   git pull
   git status            # must show nothing to commit
   ```
2. Git identity (commits under the team member's own name):
   ```powershell
   git config user.name "Tarun Sitaraman"
   git config user.email "mfsbyo@gmail.com"
   ```
3. **Pinned environment** (the D24 run used Python 3.12 and other library versions; D25 should not):
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   python -m pytest -q                       # all tests pass
   ```
   If Python 3.11 or a pinned version cannot be installed, stop and record why; only then use `--allow-unpinned`, and write the versions into the D25 run log.
4. Data present (git-ignored, built earlier on this machine):
   - `data/wheeler_checks.csv` (else `python scripts/build_wheeler_features.py`)
   - `data/pool_s7_checks.csv`, `data/pool_s8_checks.csv`, `data/pool_s9_checks.csv` (else `python scripts/build_pool_tables.py`)
5. Close other heavy programs, disable sleep, keep the laptop on mains power.

## Start

```powershell
python scripts/run_d25_night.py --dry-run                          # pre-flight and plan only
python scripts/run_d25_night.py --workers 6 --deadline 07:30
```

Leave the window open. Do not edit anything under `src/` during the run: checkpoints are keyed on the source code.

## Watching it

- `data/cache/d25/logs/night.log`: one line per event.
- `data/cache/d25/logs/status_paper.json`: running, queued, finished parts.
- `data/cache/d25/logs/paper_<exp>_<part>.log`: one log per part.

If the machine restarts or runs out of memory, run the same command again (lower `--workers` if memory was the problem): every unit is checkpointed under `data/cache/d25/`, so it resumes.

## In the morning

1. Read `night.log`. A part listed as `FAILED` or `not started` is reported as not run; do not shorten or re-run parts selectively to change results.
2. If collection did not run, run it by hand:
   ```powershell
   python scripts/run_d25.py --exp e1 --collect-only
   python scripts/run_d25.py --exp e2 --collect-only
   python scripts/summarise_d25.py --out data/cache/d25/d25_summary.md
   ```
3. Check that both results files record `"git_dirty": false`, the D23 reproduction (`"checked": true`, `max_abs_diff` below 1e-9) and the parts run and not run.
4. Commit the results files only:
   ```powershell
   git add results/forecasting_d25_e1.json results/forecasting_d25_e2.json
   git commit -m "D25 results: calibrated residual TFT + EWC and residual pretraining on the S6 stream (E1, E2)"
   git push
   ```
5. Add a run-log entry under D25 in `docs/DECISIONS.md`: start and end times, workers, environment (Python and library versions), parts not run, any failure or restart. Commit and push it separately.

## If something fails

- **Pre-flight refuses:** fix what it names. Never use `--allow-dirty` for the real run.
- **Smoke fails:** the paper run does not start; open `data/cache/d25/logs/smoke_<exp>_<part>.log`.
- **"raw scratch units do not reproduce the committed D23 runs":** the environment differs from D23's. Stop, record it, and do not report D25 comparisons with D23.
- **A part fails during the night:** the others continue (a failed `pre_tune` means that experiment's `pre_main`/`pre_replicate` do not start). Record it as not run.
