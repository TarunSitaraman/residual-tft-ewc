# 21CSC314P – Big Data Essentials: Plan of Action
## Continual Learning for Adaptive Health and Reuse Assessment of Heterogeneous Second-Life EV Batteries

**Course**: 21CSC314P – Big Data Essentials  
**Faculty**: Dr. Banu Priya Prathaban  
**Team**: Darshan D (RA2411028010056), Sohan Sandeep Krishnan (RA2411028010014), Tarun Sitaraman (RA2411028010038)  
**Created**: based on Report Review 1 analysis + dataset investigation

---

## 1. Project Initialization & Setup

### 1.1 Repository Structure
```
ML handover/
├── README.md                    ← Project overview & literature review (created)
├── data/
│   ├── raw/                     ← Original Excel source files (if available)
│   ├── cleaned_battery_data.csv
│   ├── rpt_capacity_data.csv
│   ├── DATA_DICTIONARY.csv
│   ├── ml_training_dataset.csv
│   ├── final_ml_dataset.csv
│   ├── final_battery_health_dataset.csv
│   └── final_battery_cycling_features.csv
├── notebooks/
│   ├── 01_data_ingestion.ipynb       ← Load 7 CSVs, verify column alignment
│   ├── 02_eda_analysis.ipynb         ← Battery group distributions, SOH ranges, missing data patterns
│   ├── 03_feature_engineering.ipynb  ← Derive and validate observed modelling features
│   └── 04_baseline_model.ipynb       ← Persistence, Ridge, and gradient-boosting baseline
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py              ← Preprocessing: temporal alignment, SOH calculation
│   ├── models/
│   │   ├── baseline_dnn.py           ← Future neural baseline
│   │   ├── tft_continual.py          ← Proposed Temporal Fusion Transformer + EWC
│   │   └── reuse_head.py              ← Proposed multi-task suitability classifier
│   └── continual_learning.py         ← Proposed EWC implementation
├── results/
│   ├── baseline_metrics.json
│   ├── cl_metrics.json               ← Accuracy + retention + adaptation across battery groups
│   └── reuse_metrics.json
├── .gitignore                      ← Data CSVs excluded; notebooks, src, results tracked
└── AGENTS.md                       ← Deployment and reproducibility guidance
```

### 1.2 Environment Setup
| Component | Specification |
|-----------|--------------|
| **Python** | 3.10+ |
| **PyTorch** | 2.0+ (for TFT/EWC) |
| **Key Libraries** | `pytorch-forecasting` (TFT), `numpy`, `pandas`, `scikit-learn`, `torch`, `pytorch_lightning` |
| **Hardware** | CPU is sufficient for the current baseline; GPU is recommended for the proposed TFT |
| **Data** | 7 CSV files — see Git analysis below |

### 1.3 Immediate Week 1 Tasks
| Task | Owner | Deliverable |
|------|-------|-------------|
| Repository initialization with structure above | Tarun | `git init`, `README.md`, `.gitignore`, `AGENTS.md` skeleton |
| Data ingestion & verification | Darshan/Sohan | `notebooks/01_data_ingestion.ipynb` — load 7 CSVs, print shapes, check Battery_ID groups (g1, v4, v5, w8, w9, w10), verify SOH_Percent calculation: `Capacity_Ah / Reference_Capacity_Ah * 100` |
| Missing data audit | Sohan/Tarun | Report: How many cycles have missing `Temperature_Mean/Mini/Max`? Document pattern (random? systematic?) |
| SOH calculation validation | Darshan | Verify `Reference_Capacity_Ah` = first observed RPT per battery; check `ml_training_dataset.csv` excludes rows before first RPT |

---

## 2. Data Preprocessing Pipeline

### 2.1 Loading the 7 CSV Files
| File | Role | Key Columns |
|------|------|-------------|
| `cleaned_battery_data.csv` | Point-level raw data (post-cleaning) | Profiled only; schema must be confirmed before modelling |
| `rpt_capacity_data.csv` | RPT measurements extracted | Battery_ID, RPT, Capacity_Ah, Test_Date |
| `DATA_DICTIONARY.csv` | Column definitions | 27 columns with meanings/units |
| `ml_training_dataset.csv` | SOH-labelled subset (recommended start) | 27 columns; rows before first RPT excluded |
| `final_ml_dataset.csv` | Complete time-series (all cycles) | 27 columns; some SOH NaN (before first RPT) |
| `final_battery_health_dataset.csv` | RPT capacity + SOH progression | Battery_ID, RPT, Capacity_Ah, Test_Date, Reference_Capacity_Ah, SOH_Percent |
| `final_battery_cycling_features.csv` | Cycle-level features | 18 modelling inputs plus identifiers and RPT/health columns |

### 2.2 Preprocessing Steps
1. **Load all 7 CSVs** → Verify `Battery_ID` values match across files (g1, v4, v5, w8, w9, w10)
2. **Temporal alignment** (core project requirement):
   - For each cycle in `final_battery_cycling_features.csv` or `final_ml_dataset.csv`:
   - Associate with the most recent RPT measurement **on or before** that cycle's `Cycle_Date`
   - This produces the `RPT` and `Days_Since_RPT` linkage
3. **SOH calculation verification**:
   - `SOH_Percent = Capacity_Ah / Reference_Capacity_Ah * 100`
   - `Reference_Capacity_Ah` = first observed RPT capacity per battery (from `rpt_capacity_data.csv` or `final_battery_health_dataset.csv`)
4. **Missing temperature handling**:
   - Document how many cycles have NaN in `Temperature_Mean/Mini/Max`
   - Strategy: baseline models use train-fitted median imputation with missing indicators; the proposed TFT will use explicit masks and missingness features
5. **Battery group stratification**:
   - 6 groups: g1, v4, v5, w8, w9, w10
   - Ensure train/test splits respect group identity (no leakage across groups)

### 2.3 Output of Preprocessing
- `processed/` directory with:
  - `X_train.pt`, `y_train.pt` (features + SOH targets, temporal split)
  - `X_val.pt`, `y_val.pt`
  - `X_test.pt`, `y_test.pt`
  - `metadata.json` (battery group mapping, sequence lengths, missing value counts)

### 2.4 Data Integrity Assumptions

- The dictionary contains 27 columns. The modelling code derives 18 numeric inputs from the observed files rather than assuming the handover's older count of 22.
- RPT alignment uses timestamps as stored with a backward join. A date-only `Test_Date` is interpreted at midnight, so a later cycle on the same calendar day is treated as after that RPT. This convention must be confirmed if the source system has an unknown RPT time of day.
- SOH is a step function between RPT measurements. Validation and test reports include the fraction of rows whose RPT interval was already seen during training.
- The current baseline contains every battery group in its chronological split. It does not measure unseen-group generalization; the continual-learning stage must introduce complete groups sequentially and report adaptation, retention, and forgetting separately.

---

## 3. Modelling Strategy

### 3.1 Implemented Baseline (Weeks 2-3)
| Model | Architecture | Input | Target | Notes |
|-------|-------------|-------|--------|-------|
| **Persistence reference** | Last SOH observed in each battery's training period | Battery identity only | `SOH_Percent` | Diagnostic only; can be perfect inside a repeated-label RPT interval |
| **Ridge regression** | Median imputation + missing indicators + scaling + Ridge | 18 numeric cycling inputs | `SOH_Percent` | Linear, extrapolating baseline for later-life evaluation |
| **Histogram gradient boosting** | `HistGradientBoostingRegressor` | 18 numeric cycling inputs | `SOH_Percent` | Nonlinear baseline with native NaN handling; cannot extrapolate below training target range |

**Baseline Metrics Target** (literature range, not a directly comparable guarantee):
- RMSE: 0.8–1.5% (Braco 2022: 0.6-0.8%; Takahashi 2023: <1.48%)
- MAE: proportionate to RMSE
- Validation R²: >0.70

### 3.2 Proposed Continual Learning Model (Weeks 4-8)
**Architecture: Temporal Fusion Transformer (TFT) + Elastic Weight Consolidation (EWC) + Multi-Task Reuse Head**

```mermaid
graph TD
A[Input: Variable-length sequences of 18 features plus explicit masks] --> B[TFT Encoder (time-aware attention)]
    B --> C[Latent representation]
    C --> D[Head 1: SOH_Percent Regression]
    C --> E[Head 2: Reuse Suitability Classification]
    D --> F[SOH_Percent output: %]
    E --> G[Suitability category: {like-new, moderate, limited, recycle}]
    
    style D fill:#lightgreen,stroke:#333,stroke-width:2px
    style E fill:#lightblue,stroke:#333,stroke-width:2px
```

#### 3.2.1 TFT Backbone
- **Time-series encoder** with 3 sub-networks:
  - **Value network**: Processes actual feature values
  - **Timing network**: Encodes relative time steps (irregular timestamps)
  - **Static network**: Encodes battery_id as static categorical (6 groups)
- **Explicit missing-value masks** for temperature and other incomplete signals; handling will be compared against the baseline imputation path
- **Attention-based interpretability**: Which cycles/features drive SOH prediction?

#### 3.2.2 Elastic Weight Consolidation (EWC)
- **Problem**: Catastrophic forgetting when introducing new battery groups sequentially
- **Solution**: After each battery group learns, compute Fisher information matrix; add penalty term to loss for weights important to earlier groups
- **Formula**: `L_total = L_current + λ * Σ_F_i * (θ_i - θ_i^*)^2`
  - `F_i` = Fisher diagonal for weight `i`
  - `θ_i*` = optimal weight from previous task
  - `λ` = penalty coefficient (hyperparameter)

#### 3.2.3 Multi-Task Reuse Head
Two output heads:
- **Task 1**: SOH_Percent regression (continuous health)
  - Loss: L1/L2 loss on `SOH_Percent`
- **Task 2**: Reuse suitability classification (application decision)
  - 4 categories based on SOH ranges:
    - 80-100%: "like-new" → full EV repurposing, stationary storage
    - 60-80%: "moderate" → grid support, peak shaving, commercial backup
    - 40-60%: "limited" → low-demand applications, IoT, testing
    - <40%: "recycle" → material recovery, recycling
  - Loss: Cross-entropy
- **Combined loss**: `L = L_soh + α * L_reuse` (α balances continuous vs. categorical)

#### 3.2.4 Continual Learning Flow
```
Phase 1: Introduce battery group g1
  → Train TFT-EWC on g1 only
  → EWC fisher computation after g1 training
  → Evaluate: SOH accuracy on g1, reuse category accuracy on g1

Phase 2: Introduce battery group g2
  → Train TFT-EWC on g1 + g2 jointly, with EWC penalty from g1
  → EWC fisher update after g2 training
  → Evaluate: 
     • Adaptation: SOH accuracy on g2 (new group)
     • Retention: SOH accuracy on g1 (old group, should not drop significantly)
     • Reuse accuracy: category correctness on g1 + g2

Phase 3: Introduce g3, g4, g5, g6
  → Repeat sequential introduction with EWC
  → Full retention/adaptation/reuse analysis across all 6 groups
```

### 3.3 Evaluation Metrics (Track Across All Phases)
| Metric | Definition | Target / Reference |
|--------|------------|-------------------|
| **SOH RMSE** | `sqrt(mean((pred_soh - true_soh)^2))` | 0.6-1.5% (Braco 2022: 0.6-0.8%; Cui 2024: MAPE <2.3%) |
| **SOH MAE** | `mean(|pred_soh - true_soh|)` | proportionate to RMSE |
| **SOH MAPE** | `mean(|pred_soh - true_soh| / |true_soh|) * 100` | <2.3% (Cui 2024) |
| **Catastrophic Forgetting** | `RMSE_old_after_new - RMSE_old_before_new` | Should be near 0 with EWC; negative = forgetting |
| **Adaptation Score** | `RMSE_new_group` | Lower is better; compare across groups g1-g6 |
| **Reuse Category Accuracy** | `% of predictions where suitability category matches ground-truth range` | Target >75% |
| **Interpretability Score** | % of attention heads where top features match known health indicators (Voltage_Min, Capacity_Ratio_Mean, etc.) | Qualitative analysis |
| **Computational Overhead** | Training time per battery group; EWC fisher computation cost | Compare TFT-EWC vs. retraining from scratch |

### 3.4 Comparison Baselines
| Approach | Expected Strength | Expected Weakness |
|----------|------------------|-------------------|
| **MLP (no CL)** | Fast training; simple | Severe forgetting across groups; no temporal handling |
| **LSTM (no CL)** | Handles sequences; moderate forgetting | EWC not applied; harder to adapt incrementally |
| **TFT (single-task)** | State-of-the-art time-series; handles missing values | No continual learning → must retrain on all groups |
| **TFT + EWC (proposed)** | Adaptation + retention; interpretable attention; multi-task reuse | Heavier computation; more hyperparameters (λ, α, EWC fisher update freq) |
| **Yao 2025 PNN+TCN** (literature baseline) | Explicit CL for SOC; proven across 3 public datasets | Focuses on SOC, not SOH; no reuse head; different data format |

### 3.5 Novelty and Claim Boundary

The project does not claim that TFT, EWC, or reuse classification is individually new. The intended contribution is their integration for heterogeneous second-life EV batteries, with explicit measurement of:

1. Adaptation to a newly introduced battery group.
2. Retention of previously learned groups and catastrophic forgetting.
3. Temporal alignment and leakage controls for RPT-derived SOH labels.
4. Missing-signal robustness under incomplete temperature measurements.
5. Translation from SOH estimation to interpretable reuse categories.

The current baseline is within-battery chronological evaluation. It does not establish unseen-group generalization; that must be reported separately in the continual-learning and domain-generalization experiments.

---

## 4. Expected Outcomes & Deliverables

### 4.1 Quantitative Results
| Outcome | Metric | Expected Range |
|---------|--------|----------------|
| **Baseline SOH accuracy** | RMSE on held-out cycles | 0.8–1.5% (within literature: Braco 0.6-0.8, Rashid 1.1, Takahashi <1.48) |
| **CL adaptation** | SOH RMSE on newly introduced battery group | Within 20% of baseline (demonstrates adaptation) |
| **CL retention** | Forgetting metric (RMSE change on old groups after new learning) | <0.2% increase (EWC prevents forgetting) |
| **Reuse suitability** | Category accuracy across all 6 groups | >75% correct mapping (SOH → {like-new, moderate, limited, recycle}) |
| **Feature importance** | Top 3 features by attention frequency | Expected: `Voltage_Min`, `Capacity_Ratio_Mean`, `Energy_Efficiency_Mean` (aligns with literature indicator selection theme) |

### 4.2 Qualitative Outcomes
1. **Interpretability report**: Which cycles/features does TFT attend to for SOH prediction? Do they align with known battery degradation signals (voltage hysteresis, capacity fade, efficiency drop)?
2. **Per-battery-group analysis**: Does the model perform uniformly across g1-g6, or are some groups harder (addressing literature heterogeneity theme #1)?
3. **Missing data robustness**: Does explicit masking in the proposed TFT outperform the baseline's train-fitted imputation plus missing indicators? Document temperature NaN impact.
4. **Sequential learning curve**: Plot adaptation/retention trade-off as each new battery group is introduced — this is the core novelty contribution.

### 4.3 Deliverables Checklist
- [ ] `README.md` — project overview + literature review (DONE)
- [ ] `notebooks/01-04.ipynb` — data ingestion → baseline → CL model → evaluation
- [ ] `src/models/` — baseline DNN, TFT-EWC, reuse head implementations
- [ ] `src/continual_learning.py` — EWC implementation (Fisher computation, penalty term)
- [ ] `results/` — all JSON metric files + plots (adaptation/retention curves, feature importance)
- [ ] `.gitignore` — excludes data CSVs; tracks notebooks, source, results
- [ ] `AGENTS.md` — deployment/hint configuration (for future gstack/agent workflows)
- [ ] Final report/write-up connecting results to literature gap (Section 3.2 of original report)

---

## 5. Git Suitability & Gitignore Strategy

### 5.1 Are CSV Files Too Large for Git?

**Assessment**: ✅ **YES — CSV files should be gitignored**. Here's why:

| Factor | Assessment |
|--------|------------|
| **File sizes** | Battery CSV datasets typically range from **50KB to 5MB each** depending on cycle count and resolution. The 7 files in this repo are likely manageable individually, but Git tracks history of every file. |
| **Git limitations** | Git is **not designed for large binary/semi-binary files**. Repo size grows with every commit; cloning becomes slow; file history (`git log`) becomes impractical. |
| **Collaboration impact** | Team members would download 7 CSVs on every clone; `.git` directory would balloon; CI/CD pipelines would slow. |
| **Best practice** | Data should live **outside** version control; code/notebooks/models reference the data path externally. |

**Recommendation**: Gitignoring the 7 CSV files is the **standard and professional approach** for this type of project.

### 5.2 Recommended `.gitignore` Content

```gitignore
# Data files — exclude CSVs from version control
/*.csv

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
*.egg-info/
dist/
build/
*.egg

# Notebook checkpoints
.ipynb_checkpoints/

# Results/artifacts (generated during experimentation)
results/
logs/
*.pt  # PyTorch checkpoints
*.pickle
*.h5  # Keras models

# IDE/VS Code
.vscode/
.idea/
*.swp
*.swn

# Environment
.env
.venv/
venv/
env/
```

### 5.3 Git Repository Structure (Post-Gitignore)

```
ML handover/
├── .gitignore                    ← Tracked (small, essential)
├── README.md                     ← Tracked (created, ~5KB)
├── AGENTS.md                     ← Optional (hints for future)
├── notebooks/                    ← Tracked (Python files, small)
│   ├── 01_data_ingestion.ipynb
│   ├── 02_eda_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_baseline_model.ipynb
├── src/                          ← Tracked (Python source code)
│   ├── __init__.py
│   ├── data_pipeline.py
│   ├── models/
│   │   ├── baseline_dnn.py
│   │   ├── tft_continual.py
│   │   └── reuse_head.py
│   └── continual_learning.py
├── requirements.txt              ← Tracked (dependencies)
├── .env                          ← Not tracked (local config only)
└── docs/                         ← Optional (additional documentation)
```

### 5.4 How to Handle Data in Git-Ignored Repo

**Option A: Data as External Link (Recommended for this project)**
- Keep CSVs in the local `/data/` directory (or `ML handover/` root)
- Add `/data/` to `.gitignore`
- In notebooks/code, load with relative path: `pd.read_csv("../data/ml_training_dataset.csv")`
- **Team members**: Clone repo, then `git lfs install` + pull data separately, or use the original source Excel files to regenerate CSVs

**Option B: Git LFS (Large File Storage)**
- If CSVs are critical to version (e.g., specific processed versions), use Git LFS
- `git lfs track "*.csv"` → pushes CSVs via LFS endpoint
- Requires all collaborators to have Git LFS installed
- Overhead for this project: probably unnecessary; CSVs can be regenerated from the processing flow described in `READ_me.txt`

**Option C: Data Generation Script**
- Include a `generate_data.py` script that processes raw Excel → CSVs
- Keep only raw Excel source files (or note they're from original project dataset)
- CSVs are regenerated on `python generate_data.py`
- Most sustainable for this project since the processing flow is documented

### 5.5 My Recommendation for This Project

**Option A + Option C hybrid**:
1. **Gitignores all 7 CSV files** (`/*.csv` in `.gitignore`)
2. **Keeps a `generate_data.py` or `README.md` processing flow** that explains how CSVs were created from raw Excel (already documented in `READ_me.txt`)
3. **Keeps only code/notebooks/models in Git** — the "how" of data processing, not the data itself
4. **Team members**: 
   - Clone the Git repo
   - Follow the preprocessing notes in `README.md` / `notebooks/01_data_ingestion.ipynb`
   - Either (a) place the 7 CSVs in the `data/` directory from the original handover, or (b) run the data generation script if raw Excel is available
5. **Why this works**: 
   - The project's value is in the **ML/DL methodology**, not the specific CSV binary files
   - The `READ_me.txt` already documents the processing flow completely
   - Code (TFT, EWC, baselines) is the novel contribution, not the raw data
   - Keeps repo lightweight (`<100MB` vs. potential `>500MB` with CSVs + history)
   - Standard practice for academic/ML projects

### 5.6 Git Initialization Commands

```bash
# 1. Navigate to repo root
cd /C/Users/Tarun/Downloads/ML\ handover

# 2. Create .gitignore first (before any commits)
cat > .gitignore << 'EOF'
/*.csv

__pycache__/
*.pyc
*.pyo
*.pyd
*.egg-info/
dist/
build/
.ipynb_checkpoints/
results/
logs/
*.pt
*.pickle
*.h5
.vscode/
.idea/
.env
.venv/
venv/
env/
EOF

# 3. Initialize git
git init

# 4. Add .gitignore and commit
git add .gitignore
git commit -m "Initial: gitignore setup — exclude CSVs from version control"

# 5. Add project files (README, notebooks, src)
git add README.md AGENTS.md notebooks/ src/ requirements.txt
git commit -m "Initial: project structure — README, notebooks, source code"

# 6. Optional: Add remote if pushing to GitHub/GitLab
# git remote add origin git@github.com:username/21csc314p-battery-cl.git
# git push -u origin main
```

---

## 6. Revised Timeline (with Git Integration)

| Week | Milestone | Git Status |
|------|-----------|------------|
| **W1** | Repo init + `.gitignore` + data ingestion notebook | `git init` + `git commit` (README, .gitignore); CSVs gitignored |
| **W2** | Ridge and gradient-boosting baseline + metrics | Commit baseline metrics JSON |
| **W3** | Baseline evaluation, RPT-interval analysis, and literature comparison | Commit results/plots |
| **W4** | TFT implementation (single-task) | Commit TFT code |
| **W5** | EWC layer integration | Commit CL code; track fisher computation changes |
| **W6** | Multi-task reuse head addition | Commit new head architecture |
| **W7** | Sequential battery-group introduction (g1 → g2) | Commit CL flow with EWC; track adaptation/retention |
| **W8** | Introduce g3, g4; full evaluation | Commit sequential learning results |
| **W9** | g5, g6; final comparative analysis | Commit comprehensive results |
| **W10** | Final report + thesis write-up | Commit write-up; final `AGENTS.md`; tag v1.0 |

---

## 7. Success Criteria

| Criterion | Threshold | How Measured |
|-----------|-----------|--------------|
| **Repository health** | `git clone` < 10s; repo size < 100MB (without CSVs) | Measure after init + first commit |
| **Baseline SOH accuracy** | RMSE 0.8–1.5% (within literature range) | `results/baseline_metrics.json` |
| **CL adaptation** | New battery group RMSE within 20% of baseline | Per-phase evaluation in `results/cl_metrics.json` |
| **CL retention** | Forgetting metric < 0.2% increase after each new group | `results/cl_metrics.json` — Fisher + RMSE comparison |
| **Reuse suitability** | Category accuracy > 75% across 6 groups | `results/reuse_metrics.json` |
| **Interpretability** | Top-3 attention features align with known degradation signals | Qualitative analysis notebook |
| **Git usability** | Team members can clone + set up in < 30 min | Onboarding test |

---

## 8. Next Immediate Steps (Start Here)

| Step | Action | Owner | Target |
|------|--------|-------|--------|
| 1 | Create `.gitignore` with CSV exclusion | Tarun | End of today |
| 2 | Initialize `git repo` + commit skeleton | Tarun | End of today |
| 3 | Write `notebooks/01_data_ingestion.ipynb` — load 7 CSVs, print shapes, verify Battery_ID groups | Darshan | End of Week 1, Day 1 |
| 4 | Verify `SOH_Percent = Capacity_Ah / Reference_Capacity_Ah * 100` calculation | Sohan | End of Week 1, Day 1 |
| 5 | Audit missing temperature counts per battery group | Tarun | End of Week 1, Day 2 |
| 6 | Establish `requirements.txt` with all dependencies | Darshan/Sohan | End of Week 1 |
| 7 | Extend the baseline comparison beyond Ridge and gradient boosting, if justified, using the derived feature list → SOH_Percent | Tarun | End of Week 2 |
| 8 | Train baseline + record RMSE/MAE/MAPE | All | End of Week 3 |

---

**End of Plan of Action**
