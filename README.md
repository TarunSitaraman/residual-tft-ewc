# 21CSC314P – Big Data Essentials
# Report Review 1: Continual Learning for Heterogeneous Second-Life EV Batteries

**Course**: 21CSC314P – Big Data Essentials  
**Faculty**: Dr. Banu Priya Prathaban  
**Team**:
- Darshan D – RA2411028010056
- Sohan Sandeep Krishnan – RA2411028010014
- Tarun Sitaraman – RA2411028010038

---

## 1. Project Overview

This project develops a **continual-learning framework** for adaptive health and reuse assessment of heterogeneous second-life EV batteries. The core challenge is enabling a model to learn from progressively arriving battery groups while retaining knowledge from earlier batteries (avoiding catastrophic forgetting), and ultimately supporting practical reuse decisions.

### 1.1 Motivation
- EV batteries degrade over repeated cycling, temperature variation, and ageing
- Retired EV batteries may retain useful capacity for second-life applications (stationary storage, renewable-energy storage, backup power)
- A reliable SOH (State of Health) assessment is required before reuse
- Batteries are **heterogeneous**: different chemistries, manufacturers, capacities, ageing histories, charging protocols, and incomplete first-life records
- A model trained on one battery population may fail on new groups — requiring **continual adaptation**

### 1.2 Workflow
```
Raw Excel files
       |
       v
Combined cycling data
       |
       v
Cleaned cycling data
       ├───► Cycling features (22 features: voltage, current, temperature, capacities, energy, power, efficiency, cycle count)
       └───► RPT capacity measurements
                   |
                   v
               Relative SOH (SOH_Percent = Capacity_Ah / Reference_Capacity_Ah * 100)
                   |
                   v
               Temporal alignment (most recent RPT on or before each cycle)
                   |
                   v
               final_ml_dataset.csv (complete time-series)
                   |
                   v
               ml_training_dataset.csv (SOH-labelled subset for supervised ML/DL)
                   |
                   v
               ML / DL stage
```

---

## 2. Key Technical Components

### 2.1 State of Health (SOH) Definition
- `SOH_Percent = Capacity_Ah / Reference_Capacity_Ah * 100`
- `Reference_Capacity_Ah` = **First observed RPT capacity** for each battery
- Relative reference-based SOH (not manufacturer nominal rated capacity)

### 2.2 Temporal Alignment
- Each cycling observation associates with the most recent RPT measurement **on or before** that cycle
- Prevents assigning future health information to earlier observations
- Critical for time-series integrity

### 2.3 Dataset Structure
| File | Description |
|------|-------------|
| `ml_training_dataset.csv` | Recommended starting dataset — cycle-level features with known SOH. Rows before first available RPT excluded. |
| `final_ml_dataset.csv` | Complete cycle-level dataset with temporal health alignment. Includes observations before first RPT (SOH intentionally missing). |
| `final_battery_health_dataset.csv` | RPT capacity and SOH progression per battery (Battery_ID, RPT, Capacity_Ah, SOH_Percent, Test_Date) |
| `final_battery_cycling_features.csv` | 22 cycling features per cycle (voltage, current, temperature, charge/discharge capacity/energy, power, capacity ratio, energy efficiency, cycle count, data points) |
| `cleaned_battery_data.csv` | Cleaned point-level cycling data |
| `rpt_capacity_data.csv` | Extracted RPT capacity measurements |
| `DATA_DICTIONARY.csv` | Column definitions (28 columns with meanings/units) |

**6 battery groups**: g1, v4, v5, w8, w9, w10

### 2.4 Continual Learning Framework
- **Problem**: Model must learn from new battery groups sequentially while retaining earlier knowledge
- **Key challenge**: Catastrophic forgetting — when learning new degradation patterns, previous patterns are lost
- **Approach**: Progressive learning across battery groups, evaluation on both new and previously learned batteries
- **Baseline**: Initial SOH/degradation prediction model for comparison

### 2.5 Reuse Assessment
- SOH alone does not give complete application decision
- Battery with reduced capacity may still suit lower-demand stationary applications
- Severe degradation or unsuitable characteristics → recycling
- Output: **Interpretable reuse-suitability categories** rather than just prediction accuracy

---

## 3. Literature Review (10 Key Papers)

| # | Paper (Year) | Methodology | Main Finding | Main Challenge |
|---|-------------|-------------|--------------|----------------|
| 1 | Braco (2022) | Partial-charge / real-profile SOH | RMSE 0.6–0.8%; reduced testing time | Quality depends on profile/conditions |
| 2 | Braco (2022) | Health-indicator selection (58 indicators) | Best HI: 0.2% RMSE at repurposing; partial-charge useful in second-life | Indicators change with ageing |
| 3 | Rashid (2023) | EIS + optimized GPR | SOH error ~1.1%; fast/non-destructive | EIS influenced by temperature, SOC, noise, equipment |
| 4 | Takahashi (2023) | Features + bagged GPR | Mean RMSE <1.48% worst-case | Depends on measurable feature quality |
| 5 | Cui (2024) | Online ML + clustering (15+ months testing) | MAPE <2.3%; long-duration second-life data | Limited experimental population; resource-intensive |
| 6 | Srinivasan (2024) | SOH methods review | Identifies history/heterogeneity gaps; no single adaptive model | Review-level; no new model |
| 7 | Bilevel CL (2024) | EWC + bilevel continual learning | Improved degradation prediction; lower MAE after multiple batches | Adaptation vs retention trade-off |
| 8 | Yao (2025) | PNN + TCN continual learning (3 public datasets) | Strong estimation across diverse batteries; reduces need for multiple single-task models | Focuses on SOC, not SOH/reuse |
| 9 | Li (2025) | Few-shot transfer + KAN/Transformer | Handles incomplete/data-shifted data; domain generalization | Needs target-domain adaptation |
| 10 | Meskine (2026) | BiLSTM/TCN + probabilistic model | Reliability calibration improvements; uncertainty-aware SOH | Sparse/noisy measurements; unseen-cell generalization |

### 3.1 Literature Survey Summary
Key cross-paper themes:
1. **Battery heterogeneity** — chemistry, manufacturer, capacity, ageing history, temperature, charging behaviour differ between groups
2. **Incomplete/sparse information** — first-life history often unavailable; noisy BMS-style signals
3. **Catastrophic forgetting** — learning new battery groups risks losing older knowledge
4. **SOH ≠ reuse assessment** — prediction accuracy alone doesn't dictate second-life application
5. **Accuracy vs computational cost** — complex adaptive methods must balance cost with performance

### 3.2 Research Gap
Literature progresses SOH estimation, fast screening, transfer learning, and degradation prediction **separately**. This project **integrates** three areas rarely combined:
- **Continual learning** methods for heterogeneous retired EV batteries
- **Adaptive health estimation** across progressive battery groups
- **Explicit reuse decision framework** with suitability categories

---

## 4. Project Structure & Next Steps

### 4.1 Immediate Tasks
1. **Data ingestion & preprocessing** — load the 7 CSV files, verify feature/SOH alignment, handle missing temperatures
2. **Baseline model establishment** — train initial SOH prediction model (regression on 22 cycling features → SOH_Percent)
3. **Continual learning implementation** — implement sequential battery-group learning (e.g., Elastic Weight Consolidation, Progressive Neural Networks, or similar)
4. **Evaluation framework** — compute accuracy + retention metrics across battery groups:
   - Prediction accuracy (RMSE/MAE/MAPE on test batteries)
   - Catastrophic forgetting (performance drop on previously learned groups after new learning)
   - Adaptation (performance on newly introduced groups)
5. **Reuse assessment mapping** — map SOH predictions to suitability categories (stationary storage, grid support, recycling)

### 4.2 Technical Deliverables
- [ ] Python notebooks/scripts for data pipeline (raw → features → aligned datasets)
- [ ] Baseline SOH model (architecture: DNN/TCN/GPR; evaluation on `ml_training_dataset.csv`)
- [ ] Continual learning framework with batched battery-group introduction
- [ ] Quantitative comparison: baseline vs continual-learning across accuracy, retention, adaptation
- [ ] Reuse-suitability classification model or rule set
- [ ] Report/artifact documenting experiments, results, and analysis

### 4.3 Expected Outcomes
- Demonstration that a single framework can adapt to new heterogeneous battery populations
- Quantified trade-off between adaptation (learning new degradation patterns) and retention (preserving earlier knowledge)
- Interpretable reuse categories linked to health degradation profiles
- Insights on computational overhead of continual updates vs prediction accuracy

---

## 5. Data Dictionary (from DATA_DICTIONARY.csv)

| Column | Meaning | Unit / Example |
|--------|---------|----------------|
| `Battery_ID` | Battery identifier | g1, v4, v5, w8, w9, w10 |
| `Cycling_Test` | Cycling test identifier | Cycling_1, etc. |
| `Cycle_Index` | Battery cycle number | Integer |
| `Cycle_Date` | Timestamp of cycle data | Datetime |
| `Voltage_Mean` | Mean voltage during cycle | V |
| `Voltage_Min` | Minimum voltage during cycle | V |
| `Voltage_Max` | Maximum voltage during cycle | V |
| `Current_Mean` | Mean current during cycle | A |
| `Current_Min` | Minimum current during cycle | A |
| `Current_Max` | Maximum current during cycle | A |
| `Temperature_Mean` | Mean auxiliary temperature | °C |
| `Temperature_Min` | Minimum auxiliary temperature | °C |
| `Temperature_Max` | Maximum auxiliary temperature | °C |
| `Charge_Capacity_Max` | Maximum recorded charge capacity | Ah |
| `Discharge_Capacity_Max` | Maximum recorded discharge capacity | Ah |
| `Charge_Energy_Max` | Maximum recorded charge energy | Wh |
| `Discharge_Energy_Max` | Maximum recorded discharge energy | Wh |
| `Power_Mean` | Mean electrical power | W |
| `Capacity_Ratio_Mean` | Mean discharge/charge capacity ratio | Ratio |
| `Energy_Efficiency_Mean` | Mean discharge/charge energy ratio | Ratio |
| `Data_Points` | Number of raw observations in cycle | Count |
| `RPT` | Most recent RPT available at or before cycle | RPT_1, etc. |
| `RPT_Number` | Numerical RPT index | Integer |
| `Capacity_Ah` | RPT measured discharge capacity | Ah |
| `Reference_Capacity_Ah` | First observed RPT capacity for that battery | Ah |
| `SOH_Percent` | Capacity relative to first observed RPT capacity | % |
| `Days_Since_RPT` | Days from associated RPT date to cycle | Days |

---

## 6. Technical Notes for Implementation

### 6.1 Loading & Preprocessing
```python
# Key considerations
- Reference_Capacity_Ah is battery-specific (first observed RPT)
- SOH_Percent is relative (not absolute nominal capacity)
- Temporal alignment required: each cycle → most recent RPT on or before
- ml_training_dataset.csv excludes rows before first RPT
- final_ml_dataset.csv includes all rows (some SOH may be NaN/missing)
- Missing temperatures are NOT fabricated — model must handle or mask
- 6 battery groups enable cross-group evaluation / domain adaptation
```

### 6.2 Model Input/Output
- **Input**: 22 cycling features per cycle (voltage, current, temperature, capacities, energy, power, ratios, cycle count, data points)
- **Target**: `SOH_Percent` (regression) or binned suitability categories
- **Battery ID**: Required for group-wise continual learning evaluation

### 6.3 Training Strategy
- **Time-aware train/test split**: Must respect temporal ordering (not random shuffle)
- **Sequential battery-group introduction**: Group 1 → train, Group 2 → learn progressively, etc.
- **Evaluation on both new and old groups**: After each new group introduction, test on all previously seen groups

### 6.4 Continual Learning Approaches to Explore
1. **Elastic Weight Consolidation (EWC)** — penalty-based retention of important weights
2. **Progressive Neural Networks** — add new columns of neurons per task/battery group
3. **Learning without Forgetting** — distillation-based approach
4. **Gradient projection** — project gradients to avoid updating important directions

### 6.5 Evaluation Metrics
- **Prediction accuracy**: RMSE, MAE, MAPE on SOH_Percent
- **Catastrophic forgetting**: Difference in performance on previous groups after new learning
- **Adaptation**: Performance on newly introduced battery groups
- **Reuse suitability**: Accuracy of suitability category predictions (if classification)

---

## 7. References

[1] E. Braco, I. San Martín, P. Sanchis, A. Ursúa, and D.-I. Stroe, “State of health estimation of second-life lithium-ion batteries under real profile operation,” *Applied Energy*, vol. 326, 119992, 2022.

[2] E. Braco, I. San Martín, P. Sanchis, A. Ursúa, and D.-I. Stroe, “Health indicator selection for state of health estimation of second-life lithium-ion batteries under extended ageing,” *Journal of Energy Storage*, vol. 55, 105366, 2022.

[3] M. Rashid, M. Faraji-Niri, J. Sansom, M. Sheikh, D. Widanage, and J. Marco, “Accelerated state of health estimation of second life lithium-ion batteries via electrochemical impedance spectroscopy tests and machine learning techniques,” *Journal of Energy Storage*, vol. 58, 106295, 2023.

[4] A. Takahashi, A. Allam, and S. Onori, “Evaluating the feasibility of batteries for second-life applications using machine learning,” *iScience*, vol. 26, 106547, 2023.

[5] X. Cui, M. A. Khan, G. Pozzato, S. Singh, R. Sharma, and S. Onori, “Taking second-life batteries from exhausted to empowered using experiments, data analysis, and health estimation,” *Cell Reports Physical Science*, vol. 5, 101941, 2024.

[6] V. Srinivasan, H. S. Che, J. Selvaraj, K. S. Tey, J. W. Lee, H. Shareef, and R. Errouissi, “State of Health (SoH) estimation methods for second life lithium-ion battery—Review and challenges,” *Applied Energy*, vol. 369, 123542, 2024.

[7] “Bilevel-optimized continual learning for predicting capacity degradation of lithium-ion batteries,” *Journal of Energy Storage*, vol. 86, 111187, 2024.

[8] J. Yao, B. Zheng, and J. Kowal, “Continual learning for online state of charge estimation across diverse lithium-ion batteries,” *Journal of Energy Storage*, vol. 117, 116086, 2025.

[9] X. Li, Y. Lin, S. Yang, and W. Zhang, “Intelligent domain-generalized second-life EV battery state-of-health estimation,” *Journal of Energy Storage*, vol. 140, 118989, 2025.

[10] S. Meskine, S. Al-Majeed, and H. El Asri, “Reliability Assessment of Second-Life EV Batteries Using Probabilistic Deep Learning Models for State-of-Health Prediction,” *World Electric Vehicle Journal*, vol. 17, 441, 2026.