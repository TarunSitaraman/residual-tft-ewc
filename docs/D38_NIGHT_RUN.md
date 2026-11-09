# D38 run (data machine)

Runbook for decision D38 in `docs/DECISIONS.md`: the residual TFT with experience replay, a seed ensemble and predict-before-training calibration (V2) on unseen usage conditions, on S10 first and S7 second. Everything that needs the datasets happens on the data machine; the code, tests and protocol are already in the repository.

The work is on the branch `claude/keen-fermi-h7a26t` until it is merged. Results are written from a clean commit of that branch.

## What runs (per dataset)

| Step | What | Parts |
|---|---|---|
| Pre-flight | clean tree, the check table, the committed condition sets and results of the base decision (D36 for S10, D34 for S7), library versions against `requirements.txt` | – |
| Smoke | every part once: 2 orders, 2 seeds, 2 tuning runs, 2 epochs (not reportable) | 70 |
| `tune_f0` … `tune_f4` | learning rate of the replay model, {1e-3, 3e-3, 1e-2}, 5 tuning streams each, scored on the 7 tuning conditions | 5 per experiment |
| `r32_main_f0` … `r32_replicate_f4` (after `tune_f<k>`) | replay with 32 + 8 stored rows per condition: 5 orders × 3 seeds | 10 per experiment |
| `a0_main_f0` … `a0_replicate_f4` | the D34 / D36 model unchanged, forecasts kept: 5 orders × 3 seeds | 10 per experiment |
| `r8_main_f0` … `r8_replicate_f4` (after `tune_f<k>`) | replay with 8 + 2 stored rows | 10 per experiment |
| Collect | seed ensembles, calibration, GBM-ER and the D34 single fits, reproduction check, tests over conditions, `results/forecasting_d38_<dataset>_e1.json`, `..._e2.json` | – |

The parts run in the order tuning, r32, a0, r8 (E2 before E1), so a stopped run still has the primary arm.

**Time.** Timed on synthetic data of S10 size (71 conditions, 3 cells, 9–17 check-ups; paper profile; one stream over 51 conditions, in the cloud session of 9 Nov 2026): a0 0.98 min (E1) and 1.13 min (E2); r32 0.73 and 0.87 min; r8 0.60 and 0.87 min. A replay stream costs no more than the EWC stream. D38 has 525 streams per experiment on S10 (75 tuning, 450 evaluation), the same number as D36 (375 + 150), whose real run took 5.5 h with six workers. **Expect about 5–7 h for S10 with six workers**, below the 14 h threshold of D38, so r8 is run. S7 has longer series (D34 took 13 h); allow two nights and use `--deadline`.

## Before starting

1. Get the branch and check the tree is clean:
   ```powershell
   git fetch origin claude/keen-fermi-h7a26t
   git checkout claude/keen-fermi-h7a26t
   git pull origin claude/keen-fermi-h7a26t
   git status            # must show nothing to commit
   ```
2. Pinned environment: `.\.venv\Scripts\Activate.ps1`, then `python -m pytest -q tests/test_d38.py` (10 tests pass, about 3 min).
3. Data present (git-ignored, built earlier on this machine): `data/pool_s10_checks.csv` (S10) and `data/pool_s7_checks.csv` (S7).
4. The committed assignment matches the data: `python scripts/run_d36.py --write-sets` (and `run_d34.py --write-sets` for S7) must leave `git status` clean.
5. The defaults under "Open points" of D38 are accepted, or changed, by T. Sitaraman, and that is written to the D38 run log and committed **before** the paper run.
6. Disable sleep, mains power (`powercfg /change standby-timeout-ac 0`, `powercfg /change hibernate-timeout-ac 0`).

## Start (S10)

From a standalone PowerShell window:

```powershell
python scripts/run_d38_night.py --dataset s10 --dry-run
python scripts/run_d38_night.py --dataset s10 --workers 6 --deadline 07:30
```

To resume after the deadline or a stop: `python scripts/run_d38_night.py --dataset s10 --workers 6 --skip-smoke --deadline 07:30`. Do not edit anything under `src/` between sessions: checkpoints are keyed on the source code. If the first night shows the run will not finish, `--no-r8` leaves the small-memory arm out (reported as not run).

S7 afterwards, only once S10 is collected: the same commands with `--dataset s7`.

## Watching it

- `data/cache/d38_s10/logs/night.log`: one line per event.
- `data/cache/d38_s10/logs/status_paper.json`: running, queued, finished parts.

## Afterwards

1. Read the `reproduction` block of each results file. The a0 arm must reproduce the committed D36 (or D34) values per condition: 0.0 is expected on the machine that ran D36, and collection stops above 0.05 pp. The deterministic baselines must agree to 1e-6 pp.
2. Commit `results/forecasting_d38_s10_e1.json` and `..._e2.json`, and record in the D38 run log: the commit, start and end times, environment, parts not run, the selected learning rates, the reproduction differences, every primary verdict with its `in_favour` and `mean_and_rank_disagree` flags, Q5 (pooled 90 % coverage of V2 in both seed sets), and the means with and without `s2_z02`.
3. A question is answered positively only if it holds in **both** E1 and E2 (D38). Every outcome is reported; nothing is re-run to change one.

## Prompts for the assistant on the data machine

Paste these one at a time, in order. Each stops and reports; none of them changes a result.

**Prompt 1: set-up and checks (no model on real data)**

```text
We are running decision D38 (docs/DECISIONS.md, runbook docs/D38_NIGHT_RUN.md). Rules: follow AGENTS.md; commit only as
Tarun Sitaraman <mfsbyo@gmail.com>, never as an assistant and with no co-author or session lines; synthetic data only
inside tests; ask me before anything the runbook does not cover.
1. Fetch and check out the branch claude/keen-fermi-h7a26t, confirm the tree is clean.
2. Activate the pinned .venv and run: python -m pytest -q tests/test_d38.py tests/test_d36.py tests/test_d34.py
3. Run python scripts/run_d36.py --write-sets and confirm git status is still clean.
4. Run python scripts/run_d38_night.py --dataset s10 --dry-run and show me the pre-flight output.
5. I accept the four defaults under "Open points" of D38. Add a dated line to the D38 run log saying so and that no
   D38 code has run on real data yet, then commit and push it.
Do not start the paper run. Report what passed and anything that did not.
```

(If you want to change a default, replace step 5 with the change you want; it is logged as an amendment.)

**Prompt 2: read-only audit of the S10 condition s2_z02**

```text
Read-only audit (D37, item 6). Do not change data/pool_s10_checks.csv or any file under data/.
For the S10 condition s2_z02 (stage 2, cyclic, 45 degC): print, for each of its cells, every row of the check table
(check_index, rpt_kind, check_time, Capacity_Ah, SOH_Percent, check_temp_c, and the valid / reason columns of the
build output if they are kept) and the measurement start dates in the cell's raw *_meta.txt files under
data/raw/stroebl2024/. Tell me: are the dates in order and plausibly spaced; does the capacity jump or collapse
between check-ups; does it look like a real capacity change or a data problem. Compare with one neighbouring 45 degC
cyclic condition. Report only; a correction would be a separate decision.
```

**Prompt 3: start the S10 run**

Start it yourself in a standalone PowerShell window with the two commands under "Start (S10)" above, as for D34 and D36. If you prefer the assistant to start it:

```text
Start the D38 S10 run exactly as docs/D38_NIGHT_RUN.md says, in a process that survives this session:
python scripts/run_d38_night.py --dataset s10 --workers 6 --deadline 07:30
Confirm the pre-flight passed and the smoke stage started, tell me where the logs are, then stop. Do not edit any
file while it runs.
```

**Prompt 4: after the S10 run**

```text
The D38 S10 run has finished (or stopped). Follow "Afterwards" in docs/D38_NIGHT_RUN.md:
1. Show the tail of data/cache/d38_s10/logs/night.log and any part that failed or did not start. If parts are
   missing, tell me and stop (I will resume with --skip-smoke); do not collect a partial run unless I say so.
2. From results/forecasting_d38_s10_e1.json and ..._e2.json report: the reproduction block; the selected replay
   learning rate per fold; mean and median RMSE per method (main / replicate); every primary and secondary verdict
   with median, counts, in_favour and mean_and_rank_disagree; pooled 90 % and 80 % coverage of V2 per seed set, by
   ageing type and by order; n_conditions_best_main; the means without s2_z02.
3. State each registered answer Q1-Q5 for S10 exactly as D38 defines it (positive only if it holds in E1 and E2).
   Report null and negative results as they are. Do not re-run, re-tune or change anything to alter an outcome.
4. Write the run-log entry in D38 (commit, times, environment, parts not run, the numbers above), commit the two
   results files and the log as Tarun Sitaraman <mfsbyo@gmail.com> with no co-author lines, and push the branch.
Then wait for me before starting S7.
```

**Prompt 5: S7 (only after S10 is collected and logged)**

```text
Run D38 on S7: python scripts/run_d34.py --write-sets must leave git status clean; then
python scripts/run_d38_night.py --dataset s7 --dry-run, and after my go-ahead the paper run with --workers 6
--deadline 07:30 (resume with --skip-smoke on the next night if needed). Afterwards do the same reporting, run log and
commit as for S10. D38 expects Q1 (against gradient boosting with all data stored) not to be met on S7; report
whatever comes out.
```
