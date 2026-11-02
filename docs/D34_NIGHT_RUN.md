# D34 run (ASUS)

Runbook for decision D34 in `docs/DECISIONS.md` (unseen usage conditions on the KIT first-life pool S7: the continual residual TFT + EWC against gradient boosting refitted on the latest condition, linear extrapolation and gradient boosting with all conditions stored; tests over 60 held-out conditions).

## What runs

| Step | What | Expected time (6 workers) |
|---|---|---|
| Pre-flight | clean tree, `data/pool_s7_checks.csv`, committed `results/d34_condition_sets.csv`, libraries, versions against `requirements.txt` | seconds |
| Smoke | every part once, 1 order, 1 seed, 2 tuning runs, 2 epochs (not reportable) | about 30–60 min |
| `tune1_f0` … `tune1_f4` | stage 1: learning rate {1e-3, 3e-3, 1e-2}, fine-tuning, 5 tuning streams each, scored on the 7 tuning conditions | E2 about 45 min, E1 about 35 min each |
| `tune2_f0` … `tune2_f4` (after `tune1_f<k>`) | stage 2: EWC λ {0.01 … 1000} at the stage-1 rate, 5 streams each; grid extended at an edge (D11) | E2 about 1.5 h, E1 about 1.1 h each (more if extended) |
| `main_f0` … `replicate_f4` (after `tune2_f<k>`) | 5 orders × 3 seeds EWC streams over the fold's 48 conditions + 3 joint fits | E2 about 50 min, E1 about 40 min each |
| Collect | single fits (GBM-ALL, GBM-LAST, extrapolation, persistence, ridge), tests over conditions, `results/forecasting_d34_e1.json`, `..._e2.json` | a few minutes |

Timed on synthetic data of S7 size (single-threaded): one EWC stream over 48 conditions 2.2 min (E1) / 3.0 min (E2); one joint fit 2.1 / 1.3 min. About 34 h of CPU in all, **about 6 h with six workers** (about 9 h if every λ grid is extended). Real data may early-stop later than the synthetic data, so allow a second night: the run resumes from its checkpoints.

## Before starting

1. Pull `main` and check the tree is clean (`git status` shows nothing to commit). Close VS Code for the run.
2. Pinned environment: `.\.venv\Scripts\Activate.ps1`, then `python -m pytest -q tests/test_d34.py` (5 tests pass).
3. Data present (git-ignored, built earlier on this machine): `data/pool_s7_checks.csv` (`scripts/build_pool_tables.py`).
4. Check the committed assignment matches the data: `python scripts/run_d34.py --write-sets` must leave `git status` clean.
5. Disable sleep, mains power (`powercfg /change standby-timeout-ac 0`, `powercfg /change hibernate-timeout-ac 0`).

## Start

From a standalone PowerShell window:

```powershell
python scripts/run_d34_night.py --dry-run
python scripts/run_d34_night.py --workers 6 --deadline 07:30
```

To resume after the deadline or a stop: `python scripts/run_d34_night.py --workers 6 --skip-smoke --deadline 07:30`. Do not edit anything under `src/` between sessions: checkpoints are keyed on the source code.

## Watching it

- `data/cache/d34/logs/night.log`: one line per event.
- `data/cache/d34/logs/status_paper.json`: running, queued, finished parts.

## Afterwards

Commit `results/forecasting_d34_e1.json` and `results/forecasting_d34_e2.json`, and record in the D34 run log the commit, start and end times, environment, parts not run and the verdicts. A positive answer to a question needs the replicated result in **both** E1 and E2 (D34).
