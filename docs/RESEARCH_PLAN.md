# Research plan

Plan from the validated data pipeline (notebooks 01–04) to a submittable IEEE paper. Decisions referenced as D1–D9 are in `docs/DECISIONS.md`.

## 1. Research questions and claim

**Working title:** Continual learning for state-of-health estimation and reuse assessment of heterogeneous second-life EV batteries.

| ID | Research question | Evidence required |
|---|---|---|
| RQ1 | How much does SOH-estimation error rise when a model meets a battery group it has not seen? | Leave-one-group-out vs. within-group baselines (E3 vs. E1) |
| RQ2 | When groups arrive sequentially, how much does naive fine-tuning forget earlier groups? | Accuracy matrix, backward transfer and forgetting for fine-tuning (E6) |
| RQ3 | Which continual-learning strategy gives the best adaptation–retention trade-off on this data, and at what computational cost? | Replay, EWC and LwF vs. fine-tuning (lower bound) and joint training (upper bound), with training time (E6–E8) |
| RQ4 | Can predicted SOH be turned into reliable, interpretable reuse categories? | Category agreement and confusion analysis (E10) |

**Claim boundary** (as in `PLAN_OF_ACTION.md` §3.5): the contribution is an integrated, leakage-controlled evaluation of continual learning for heterogeneous second-life batteries — sequential adaptation, retention, temporal label integrity, missing-signal robustness and reuse mapping measured together. No component method (TFT, EWC, replay, LwF) is claimed as new.

**Hypotheses** (stated before running, so results can confirm or reject them):

- H1: Error on an unseen group is significantly higher than within-group later-life error.
- H2: Sequential fine-tuning shows positive forgetting (earlier-group error rises after later groups are learned).
- H3: At least one continual-learning strategy reduces forgetting relative to fine-tuning without increasing error on the newest group by more than a pre-set margin (0.25 SOH percentage points).

## 2. Data (complete)

Six battery groups (g1, v4, v5, w8, w9, w10); cycle-level features with SOH from the most recent RPT on or before each cycle. Validated in notebooks 01–03: SOH formula, first-RPT reference, backward alignment, training subset, missing temperatures (reported, not fabricated), 18 model inputs (D1).

**Small-sample caveat.** Six groups means six continual-learning tasks. Conclusions must rest on repeated runs over several group orders and seeds with confidence intervals, and the paper must state that results cover six groups from one handover.

## 3. Experiments

All experiments use `ml_training_dataset.csv`, the 18 inputs (D1), `"start"` date-only convention (D3) unless stated, RMSE/MAE as primary metrics (D8), and record provenance (D9).

### Stage A — reference points

| ID | Experiment | Purpose |
|---|---|---|
| E1 | Within-group chronological baseline (notebook 04, done in code) | Later-life error when every group is seen |
| E2 | RPT-boundary split: cut each battery's timeline only at RPT changes | Error on genuinely unseen SOH states (removes shared-RPT interpolation, D4) |
| E3 | Leave-one-group-out: train on five groups, test on the held-out group's test period | Unseen-group error (RQ1, H1) |
| E4 | Joint training on all groups' training periods | Upper bound for continual learning |
| E5 | D3 sensitivity: relabel same-day cycles with `"end"` and re-run E1 | Shows conclusions do not depend on the date-only convention |

### Stage B — continual-learning protocol

- **Task** = one battery group, split chronologically 70/15/15 (train/val/test) as in E1.
- **Stream** = the six tasks in a given order. Orders: the identity order plus seeded random permutations (target 10 orders), identical for every method.
- **Seeds:** 5 per order. Results are aggregated over orders × seeds.
- **Accuracy matrix** R (error, lower is better): `R[i, j]` = test RMSE on task *j* after training through step *i*.
- **Metrics** (error-based versions of the standard continual-learning metrics):
  - final average error: mean over *j* of `R[T-1, j]`;
  - plasticity (learning error): mean of `R[j, j]`;
  - backward transfer: mean over *j < T-1* of `R[T-1, j] − R[j, j]` (positive = forgetting);
  - forgetting: mean over *j < T-1* of `R[T-1, j] − min_{j ≤ i < T-1} R[i, j]`;
  - forward transfer: mean over *j > 0* of `R[j-1, j] − b_j`, where `b_j` is the error of the untrained reference model on task *j*;
  - training time per task.
- **Statistics:** 95 % bootstrap confidence intervals over orders × seeds; paired Wilcoxon signed-rank tests between methods on matched (order, seed) runs, with Holm correction.

### Stage C — methods

| ID | Method | Role |
|---|---|---|
| E6 | Sequential fine-tuning | Lower bound; measures forgetting (RQ2, H2) |
| E4 | Joint training | Upper bound |
| E7 | Experience replay (fixed-size buffer per task) | Rehearsal baseline |
| E8a | Elastic Weight Consolidation | Regularisation (the proposed model's CL mechanism) |
| E8b | Learning without Forgetting | Distillation baseline |

All methods share one backbone for Stage C: a small MLP on the 18 inputs plus missing-value indicators, with preprocessing fitted on the first task's training data only (no access to future tasks).

### Stage D — proposed model

E9: **the proposed model — Temporal Fusion Transformer (TFT) with EWC**, using sequences of cycles per battery, explicit missingness masks from the `_Missing` flags (D7) and the battery group as static context. It runs on exactly the Stage B protocol (same orders, seeds, splits, metrics, validation-only tuning of λ under D11), so its numbers are directly comparable with Stage C.

The Stage C MLP runs are the ablations that make the TFT + EWC result interpretable:

| Comparison | What it isolates |
|---|---|
| TFT + EWC vs. MLP + EWC | Contribution of the temporal architecture |
| TFT + EWC vs. TFT fine-tuning | Contribution of EWC on the TFT |
| TFT + EWC vs. MLP replay / LwF | Position against other continual-learning families |
| TFT + EWC vs. joint training | Remaining gap to the upper bound |

All outcomes are reported as measured, including if the TFT does not improve on the MLP backbone on this six-group dataset.

### Stage E — reuse assessment

E10: map predicted SOH to reuse categories and measure agreement with categories from measured SOH (accuracy, macro-F1, confusion matrix), per group and after each continual-learning step.

**Open research decision (before any code):** SOH here is relative to each battery's first RPT, not to nominal capacity (CLAUDE.md), so literature thresholds such as "80 % of nominal" do not transfer directly. Options are analysed in `docs/REUSE_CATEGORIES.md`.

### Stage F — robustness

- E11: missing temperature — compare median-imputation + indicators against masking; evaluate on cycles with missing temperature separately.
- E5 (above) for the date-only convention.

## 4. Code structure

| Module | Contents | Stage |
|---|---|---|
| `src/data_pipeline.py` | Loading, validation, alignment, splits, metrics, provenance (existing) | done |
| `src/protocols.py` | Task/stream construction, group orders, leave-one-group-out, RPT-boundary split, date-only relabelling | A, B |
| `src/evaluation.py` | Accuracy matrix, CL metrics, bootstrap CIs, paired tests, aggregation | B |
| `src/models/baseline_dnn.py` | Preprocessor fitted on first-task data, MLP regressor | C |
| `src/models/tft_continual.py` | TFT for sequence-to-one estimation, causal cycle windows (D12) | D |
| `src/continual_learning.py` | Strategy interface; fine-tune, joint, replay, EWC, LwF | C |
| `src/experiments.py` | Runner: method × order × seed → accuracy matrices + results JSON | B, C |
| `notebooks/05_reference_baselines.ipynb` | E2–E5 | A |
| `notebooks/06_continual_learning.ipynb` | E4, E6–E8 | B, C |
| `notebooks/07_tft_ewc.ipynb` | E9 with ablations, interpretation, window sensitivity | D |
| `src/reuse.py`, `notebooks/08_reuse_assessment.ipynb` | E10 reuse categories: metrics, baselines, sensitivity | E |
| `tests/` | Unit tests on synthetic data generated at test time into a temporary directory | all |

### Interfaces

```python
# src/protocols.py
@dataclass(frozen=True)
class Task:
    name: str               # battery group, e.g. "g1"
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

def make_task(df, group, train_frac=0.70, val_frac=0.15) -> Task
def make_task_stream(df, order, train_frac=0.70, val_frac=0.15) -> list[Task]
def sample_orders(groups, n_orders, seed) -> list[tuple[str, ...]]
def leave_one_group_out(df, held_out, train_frac=0.70, val_frac=0.15) -> Task
def rpt_boundary_split(df, train_frac=0.70, val_frac=0.15) -> pd.DataFrame
def relabel_date_only(df, health, date_only_rpt) -> pd.DataFrame

# src/evaluation.py
def accuracy_matrix(errors: Mapping[tuple[int, str], float], order: Sequence[str]) -> pd.DataFrame
def cl_metrics(R: pd.DataFrame, reference_errors: Mapping[str, float] | None = None) -> dict
def bootstrap_ci(values, n_boot=10_000, alpha=0.05, seed=0) -> tuple[float, float, float]
def paired_comparison(a, b) -> dict
def summarise_runs(runs: pd.DataFrame, metric_cols, group_cols) -> pd.DataFrame

# src/models/baseline_dnn.py, src/continual_learning.py
class TabularPreprocessor:   fit(df) / transform(df) -> np.ndarray   (median + indicators + scaling)
class MLPRegressor(nn.Module)
class Strategy:              fit_task(task) ; predict(df) -> np.ndarray
class FineTune, Replay, EWC, LwF(Strategy);  class Joint: fit_all(tasks) ; predict(df)
```

## 5. Workstreams and milestones

| # | Workstream | Depends on | Output | Status |
|---|---|---|---|---|
| W0 | Real-data run of notebooks 01–04 from a clean commit | Data mounted in a session | Committed `results/*.json`; data issues logged | **blocked on data access** |
| W1 | Protocols (`src/protocols.py` + tests) | — | Stage A/B building blocks | done |
| W2 | Evaluation (`src/evaluation.py` + tests) | — | CL metrics and statistics | done |
| W3 | Models and strategies (`baseline_dnn.py`, `continual_learning.py` + tests) | — | Stage C methods | done |
| W4 | Reuse-category analysis (`docs/REUSE_CATEGORIES.md`) | — | Options and recommendation for E10 | done — team decisions pending (§6 of that document) |
| W5 | Runner + notebooks 05/06, validation-only tuning (D11) | W1–W3 | E2–E8 results on synthetic data, then real data | code done (`src/experiments.py`, notebooks 05–06); real-data run blocked on W0 |
| W6 | Proposed model: TFT + EWC (E9) | W5 | TFT strategy on the shared protocol, EWC reused from `src/continual_learning.py` | code done (D12, notebook 07); real-data run blocked on W0 |
| W7 | Reuse-category evaluation (E10) | W4, W5 | `src/reuse.py`, notebook 08, per-step category metrics in notebooks 06–07 | code done (D10); real-data run blocked on W0 |
| W7b | Robustness to missing temperature (E11) | W5 | Masking vs. imputation comparison | later |
| W8 | Paper: figures, tables, methods and results draft | W5–W7 | Manuscript draft | later |

## 6. Rules for all workstreams

- Follow `CLAUDE.md` and `AGENTS.md`: no fabricated data, no shuffling of time series, preprocessing fitted on training data only, per-group reporting, provenance recorded.
- Synthetic data is generated only inside tests (temporary directories) and never feeds reported results.
- Deterministic by default: fixed seeds, `torch.use_deterministic_algorithms(True)`, single-threaded CPU for reference runs.
- Every experiment writes a small JSON under `results/` with config, seeds, orders, environment and commit.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Six groups give low statistical power | Many orders × seeds, bootstrap CIs, effect sizes; state the limitation |
| Step-function labels inflate within-group scores | E2 RPT-boundary split; report RPT-overlap share (D4) |
| Groups differ in scale (capacity, protocol) | Relative SOH target; preprocessing fitted on first task only; report per-group results |
| TFT adds cost without benefit on small data (six groups, short sequences) | Report TFT + EWC against the MLP ablations under identical conditions, with training time; state the outcome as measured |
| Reuse thresholds not grounded for relative SOH | Resolved: relative-fade classes fixed before any real-data run (D10), with pre-registered sensitivity analyses |
| Few RPT intervals per test period (E10 headline unit) | Report sample counts with every category metric, cycle-level results alongside, and treat interval-level results as indicative |
| Real data never run | W0 is the first priority once data is mounted |
