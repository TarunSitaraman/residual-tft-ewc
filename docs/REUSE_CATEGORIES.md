# Reuse categories for relative SOH

Workstream W4. Resolves the open decision in `docs/RESEARCH_PLAN.md` Stage E (E10) and replaces the SOH bands in `PLAN_OF_ACTION.md` §3.2.3 once the team adopts it. Status: **proposal, awaiting team decision** (checklist in §6).

---

## 1. Problem

In the delivered data

```
SOH_Percent = Capacity_Ah / Reference_Capacity_Ah * 100
Reference_Capacity_Ah = first observed RPT capacity of that battery
```

The batteries were already aged when the first RPT was taken, so `SOH_Percent = 100` means "as healthy as at the start of observation", not "as healthy as new". Relative SOH measures **fade since the first RPT**. It does not measure how much of the original capacity is left.

Literature thresholds (for example, EV retirement at 70–80 % of original capacity [S1]) refer to rated or beginning-of-life capacity. Applying them to relative SOH causes three problems:

| Problem | Consequence |
|---|---|
| **Different reference.** A cell at 100 % relative SOH could be at, say, 75 % of rated capacity. | "Like-new" (80–100 %) would describe cells that are already past a first-life retirement threshold, which is factually wrong. |
| **Classes collapse.** Relative SOH stays close to 100 % in this data (D8). | With 80/60/40 bands almost every row falls in one class. Accuracy is then trivially high, macro-F1 is undefined or meaningless, and E10 would say nothing about decision reliability. |
| **Different starting points are hidden.** Each battery is normalised to its own first RPT. | Two batteries with the same relative SOH can have very different absolute capacities, so one threshold does not mean the same thing for both. |

The 80/60/40 bands in `PLAN_OF_ACTION.md` §3.2.3 applied to `SOH_Percent` should therefore **not** be used, and the application labels ("full EV repurposing", "grid support", ...) must not be attached to relative-SOH classes.

### 1.1 Provenance finding that changes the options

The six battery IDs in the handover (g1, v4, v5, w8, w9, w10) exactly match the six cells of the public Stanford second-life dataset [D2]. In that dataset, cells V4 and W8 were cycled at low C-rates, W9 and V5 at 1C, and W10 and G1 at 3C. Its six cells are INR21700-M50T cells (NMC cathode, graphite/silicon anode), previously aged for 23 months on UDDS driving profiles in the first-life dataset [D1], then cycled for 24 months on synthetic grid-storage duty cycles, alternating between 20 °C and 35 °C [D2, D3]. The nominal capacity reported for the cell is 4.85 Ah [D2]; this value comes from a search-index summary and must be confirmed against the paper or the manufacturer datasheet.

**This match is a hypothesis, not a confirmed fact:** the handover does not name its source. If it is confirmed (checks in §3, option b):

- the paper **must cite** [D1–D3] as the data origin;
- "heterogeneous" must be qualified: the six groups are the **same cell model and chemistry**, and they differ in first-life history, second-life C-rate and temperature, **not** in chemistry or manufacturer as `README.md` §1.1 currently suggests;
- absolute capacity anchoring becomes possible (option b).

---

## 2. How the literature defines reuse categories

Sources are listed in §7. Bibliographic details were checked against publisher or index records. The full texts could not be retrieved from this environment, so this document attributes to each source only what its abstract or index summary states. Every citation marked † must be checked against the full text before submission.

| Family | How categories are defined | Examples | Transfer to this project |
|---|---|---|---|
| **Capacity-based** | Remaining capacity as a percentage of rated or initial capacity. First-life EV retirement is commonly set at 70–80 %, a convention with little supporting analysis [S1]. Recent work argues that EV batteries remain usable beyond it [S1, S2]. Residual-capacity testing of retired traction batteries is standardised in GB/T 34015 part 1 [T4]. | [S1], [S2], [T4] | Needs an absolute reference, which relative SOH lacks. Requires option (b). |
| **Resistance / power-based** | Internal-resistance growth or power fade as a limit, independent of capacity. [S3] proposes internal-resistance-related end-of-life criteria for EVs. The EU Battery Regulation lists capacity fade, power fade and internal resistance among the state-of-health parameters that the BMS must hold (Art. 14, Annex VII) [T3]†. | [S3]†, [T3]† | No resistance, HPPC or EIS columns in the handover. Proxies (voltage sag, energy efficiency) are confounded by C-rate and temperature, which differ by design between groups. |
| **Degradation-rate / knee-based** | Trajectory shape: fade rate, or the onset of accelerated non-linear fade ("knee"). [S4] reviews knee definitions and mechanisms; [S5] proposes identifying the knee point and the earlier knee onset. | [S4], [S5] | Computable from RPT series without an absolute reference. Sparse RPTs limit knee detection. |
| **Application-requirement-based** | Categories follow what an application needs (energy, power, cycling, calendar life). Remaining second-life lifespan depends strongly on the application [S6]†. Reviews discuss the operational requirements of second-life applications [S7, S8, S9]. | [S6]†, [S7]†, [S8]†, [S9]† | Needs absolute capacity and power, plus requirement values that the cited reviews do not set as standard numbers. |
| **Multi-criteria sorting / grading** | Several indices (capacity, resistance, self-discharge, consistency) combined into classes, often by clustering. [S10] reviews sorting indices, classification and regrouping algorithms. UL 1974 covers the sorting and grading process for repurposed cells, modules and packs [T1]. IEC 63330-1:2024 sets general requirements for repurposing [T2]. | [S10], [T1], [T2] | Needs several indices. Only capacity is measured directly here, and there are six batteries. |
| **Learned retain-vs-recycle decisions** | A model trained to assess whether retired batteries are fit for second-life use or should be recycled [S11]. [S12] studies health estimation for second-life cells on the same Stanford cells. | [S11], [S12] | Directly related work, probably on the same cells (first life [S11]; second life [S12]). |

**No numerical second-life class boundaries were found in an accessible standard.** UL 1974, IEC 63330-1 and GB/T 34015 define procedures (testing, sorting, grading, labelling), but their full texts are paywalled and were not checked for numerical cut-offs. A commonly quoted "end of second life at about 60 % of rated capacity" could **not** be traced to a verifiable source here, so the paper must not state it as established.

---

## 3. Options for this project

Columns available: `Battery_ID`, `Cycle_Date`, `RPT`, `RPT_Number`, `Capacity_Ah`, `Reference_Capacity_Ah`, `SOH_Percent`, `Days_Since_RPT`, and the 18 cycling features (D1).

Notation: `F = 100 − SOH_Percent` is the fade in percentage points (pp) since the first RPT.

### (a) Classes on relative fade since first RPT

- **Data:** `SOH_Percent` only.
- **Strengths:** Uses only the delivered data and needs no external assumption. It is a deterministic function of SOH, so E10 isolates how SOH error becomes category error. It can be pre-registered before the real-data run: W0 has not run yet, so the cut-points are genuinely blind.
- **Weaknesses:** It describes change, not suitability. Batteries with different starting health share a class, and the cut-points are conventions.
- **Reviewer risk:** Medium. It is acceptable if the classes carry neutral names, the cut-points are fixed in advance and justified, and a sensitivity analysis is reported. It is high if the classes carry application labels.

### (b) Absolute anchoring via recovered nominal or beginning-of-life capacity

- **Data:** `Capacity_Ah` and `Reference_Capacity_Ah`, plus an external anchor `C_anchor`:
  - preferred: each cell's beginning-of-life RPT capacity from the first-life dataset [D1], measured with the lab's own protocol;
  - fallback: rated capacity from the datasheet (4.85 Ah as reported [D2], to be confirmed).
- **Formula:** `SOH_abs = Capacity_Ah / C_anchor * 100`, to which capacity-based literature conventions can be applied.
- **Required checks before use:**
  1. Cycle date range starts around 28 Oct 2022, as reported for [D2].
  2. `Voltage_Min`/`Voltage_Max` are consistent with the 3–4 V window reported for the second-life protocol [D3].
  3. Temperature alternates between about 20 °C and 35 °C [D2].
  4. `Reference_Capacity_Ah` matches the first second-life capacity in [D2] to within rounding.
  5. The capacity test C-rate matches between the anchor and the handover RPTs. A datasheet rating at a different rate biases `SOH_abs`.
  6. The dataset licence permits the use.
- **Strengths:** Makes capacity thresholds from the literature meaningful and allows application labels.
- **Weaknesses:** It rests on an external link that the handover does not state. The fallback also adds a rate mismatch.
- **Reviewer risk:** Low if provenance is confirmed and cited. Very high if the match is assumed without the checks, because that would be substituting data.

### (c) Degradation-rate and trajectory classes

- **Data:** RPT series (`Capacity_Ah`, `Test_Date` or `Cycle_Date`, `RPT_Number`).
  - Fade rate per 100 days: `r_t = 100 · ΔF / Δdays` between consecutive RPTs, or the slope of a linear fit over the last k RPTs.
  - Fade rate per equivalent full cycle (EFC): needs discharged Ah per cycle. `Discharge_Capacity_Max` may approximate it, but only if it is per-cycle and not cumulative, which must be checked.
  - Knee onset per [S4, S5].
- **Strengths:** Independent of the reference, and physically meaningful (accelerating fade signals end of useful life).
- **Weaknesses:** Few RPTs per battery make rates noisy and knee detection weak. Rates reflect the imposed C-rate and temperature, which differ by group [D2]. A rate from predicted SOH needs a trajectory, not a single point.
- **Reviewer risk:** Medium. It is best as a secondary, battery-level analysis.

### (d) Application-requirement mapping (retention × rate)

- **Data:** option (b) or (a), plus (c). Example: stationary storage requires `SOH_abs ≥ x` and rate ≤ y; backup requires a lower x with any rate; otherwise the battery goes to recycling.
- **Strengths:** It is the most decision-relevant option and matches the paper's stated goal.
- **Weaknesses:** The x and y values are not set by any source verified here (§2), and power requirements cannot be checked without resistance data.
- **Reviewer risk:** High unless x and y are cited. It is acceptable only as an illustrative scenario with explicit assumptions.

### (e) Data-driven clustering validated against physical criteria

- **Data:** per-battery or per-RPT vectors (F, fade rate, energy efficiency, voltage statistics), clustered with k-means or a Gaussian mixture model.
- **Strengths:** No imposed thresholds.
- **Weaknesses:** Six batteries is too few. Clusters are likely to recover the test protocol (C-rate pairs V4/W8, W9/V5, W10/G1 [D2]) rather than health. Clusters are unstable and hard to name.
- **Reviewer risk:** High as a primary definition. At most, use it as a descriptive check.

---

## 4. Recommendation

**Primary definition:** option (a), relative-fade classes with neutral names and cut-points fixed now. Commit this file before W0 runs, so the commit hash timestamps the pre-registration. Categories are a function of SOH only, so the same rule applies to measured and predicted SOH.

With `F = 100 − SOH` (pp since first RPT):

| Class | Rule on `SOH_Percent` (S) | Meaning |
|---|---|---|
| C0 stable | S > 95 (includes S > 100, i.e. apparent capacity recovery) | < 5 pp fade since first observation |
| C1 moderate fade | 90 < S ≤ 95 | 5–10 pp |
| C2 substantial fade | 80 < S ≤ 90 | 10–20 pp |
| C3 severe fade | S ≤ 80 | ≥ 20 pp; flag for retirement screening |

Rationale to state in the paper:

- **The 20 pp boundary is an analogy.** It applies the common "80 % of the starting value" convention [S1] to the second-life starting point. It is not an equivalence with first-life end of life.
- **The 5 pp and 10 pp boundaries are resolvable.** Class widths (≥ 5 pp) are several times the SOH error the baselines are expected to reach (≈ 1 pp or less). A boundary narrower than the model error cannot be resolved.
- **Cut-points are not tuned.** They are not chosen from the class balance of the data.

Implementation: `np.digitize(100 - soh, [5, 10, 20], right=False)` gives 0–3 with the boundaries above. Put it in one function, for example `reuse_class(soh)` in `src/evaluation.py`, and unit-test the boundary values 95, 90 and 80.

**Sensitivity analyses** (all reported, none used for selection):

| ID | Variation |
|---|---|
| S-a1 | Cut-points shifted together by −2.5 pp and +2.5 pp |
| S-a2 | Coarser 3-class rule: F < 10, 10–20, ≥ 20 |
| S-b | If the provenance checks in §3(b) pass: `SOH_abs` with the per-cell beginning-of-life anchor from [D1], then capacity bands, labelled explicitly as literature conventions, with application names allowed **only here** |
| S-c | Battery-level fade rate per 100 days at each RPT (option c), reported descriptively per group |
| S-D3 | The `"end"` date-only convention (D3) |

**E10 evaluation protocol**

1. Assign `y = reuse_class(SOH_measured)` and `ŷ = reuse_class(SOH_predicted)` on each test set. Do not train a separate classifier for the primary result. The multi-task reuse head in `PLAN_OF_ACTION.md` may be compared against thresholding the regression output, but it is secondary.
2. Evaluate at two units:
   - **rows** (cycles);
   - **RPT intervals**: one sample per (battery, RPT) in the test period, using the mean predicted SOH over the interval's rows. This removes the weighting by repeated labels (D4) and is the headline unit.
3. Report these metrics:
   - accuracy;
   - macro-F1 over the classes present in `y ∪ ŷ` for that evaluation set, with the number of classes reported next to it;
   - linear-weighted Cohen's κ (the classes are ordinal);
   - the 4×4 confusion matrix;
   - the share of errors of more than one class;
   - **boundary share**: the fraction of samples whose measured S lies within ±MAE of a cut-point.
4. Compare against these baselines: majority class of the training period, and persistence (the class of the last training RPT).
5. Report **per group and pooled**, and **after each continual-learning step**: a category matrix `A[i, j]` (macro-F1 on task *j* after step *i*) with the same backward-transfer and forgetting definitions as the RMSE matrix R (Stage B). Aggregate with bootstrap CIs over orders × seeds.
6. Handle degenerate cases explicitly. If a group's test set contains one measured class, macro-F1 is not interpretable: report accuracy and boundary share, and flag the case. Do not merge classes after seeing results.

**Limitations the paper must state**

- **Classes describe change, not suitability.** They show capacity fade relative to the first observed RPT, not absolute health or suitability for a particular application. Application labels appear only in S-b and only if provenance is confirmed.
- **Cut-points are conventions.** They were fixed before evaluation. No verified standard gives numerical second-life class boundaries.
- **Capacity is the only criterion.** Resistance and power fade [S3, T3] and safety screening [T1, T2] are not assessed.
- **Small sample.** There are six cells; if the provenance is confirmed, all are one cell model, so the results do not establish generalisation across chemistries or manufacturers.
- **Label resolution is limited.** Labels change only at RPTs.

---

## 5. What changes elsewhere, once adopted (not done in this workstream)

- `PLAN_OF_ACTION.md` §3.2.3, §3.3 and §4.1: replace the 80/60/40 bands and the ">75 % category accuracy" target, which is meaningless if classes collapse, with the rule and metrics above.
- `README.md` §1.1 and §2.5: qualify "heterogeneous" and add the data-origin citation if the provenance is confirmed.
- `docs/DECISIONS.md`: add a decision entry (D10) recording the adopted rule and sensitivity set.

---

## 6. Decisions for the team

- [ ] Adopt option (a) with cut-points 95 / 90 / 80 and neutral class names as the primary rule (or amend it) **before** the real-data run.
- [ ] Commit this document before W0 so that the pre-registration is timestamped.
- [ ] Run the provenance checks in §3(b) as soon as the data is mounted, and record the result.
- [ ] If the provenance is confirmed: cite [D1–D3], revise the "heterogeneous" wording, check the licence, and decide whether to download beginning-of-life capacities from [D1] for S-b.
- [ ] Choose the E10 headline unit: RPT interval (recommended) or row.
- [ ] Decide whether the multi-task reuse head stays in the paper or becomes a secondary comparison against thresholded regression.
- [ ] Confirm whether `Discharge_Capacity_Max` is per-cycle, which EFC-based rates (S-c) require.
- [ ] Assign someone to read the full texts of the † sources and of [T1–T4] for any numerical thresholds before submission.

---

## 7. Sources

Bibliographic data verified against publisher or index records (title, authors, venue, year, DOI or URL). † means only the abstract or summary was seen, so any statement beyond the title and abstract needs a full-text check.

**Data origin (candidate; see §1.1)**

- [D1] G. Pozzato, A. Allam, S. Onori, "Lithium-ion battery aging dataset based on electric vehicle real-driving profiles," *Data in Brief*, vol. 41, 107995, 2022. doi:10.1016/j.dib.2022.107995. Data: https://osf.io/qsabn/
- [D2] K. Moy, M. A. Khan, S. Fasolato, G. Pozzato, A. Allam, S. Onori, "Second-life lithium-ion battery aging dataset based on grid storage cycling," *Data in Brief*, vol. 57, 111046, 2024. doi:10.1016/j.dib.2024.111046. Data: https://osf.io/8jnr5/ (the nominal-capacity figure and the cell-to-C-rate assignment come from search-index summaries†)
- [D3] X. Cui, M. A. Khan, G. Pozzato, S. Singh, R. Sharma, S. Onori, "Taking second-life batteries from exhausted to empowered using experiments, data analysis, and health estimation," *Cell Reports Physical Science*, vol. 5, 101941, 2024. doi:10.1016/j.xcrp.2024.101941 (README ref. [5])

**Literature**

- [S1] S. Saxena, C. Le Floch, J. MacDonald, S. Moura, "Quantifying EV battery end-of-life through analysis of travel needs with vehicle powertrain models," *J. Power Sources*, vol. 282, pp. 265–276, 2015. doi:10.1016/j.jpowsour.2015.01.072
- [S2] M. Etxandi-Santolaya, L. Canals Casals, C. Corchero, "Extending the electric vehicle battery first life: Performance beyond the current end of life threshold," *Heliyon*, 2024, e26066. doi:10.1016/j.heliyon.2024.e26066 †
- [S3] "Redefining the EV battery end of life: internal resistance related limitations," *Energy Proceedings*, vol. 26 (ICAE 2022), 2022. doi:10.46855/energy-proceedings-10134. **Authors not verified.** †
- [S4] P. M. Attia, A. Bills, F. Brosa Planella, P. Dechent, G. dos Reis, M. Dubarry, et al., "Review—'Knees' in lithium-ion battery aging trajectories," *J. Electrochem. Soc.*, vol. 169, no. 6, 060517, 2022. doi:10.1149/1945-7111/ac6d13
- [S5] P. Fermín-Cueto, E. McTurk, M. Allerhand, E. Medina-Lopez, M. F. Anjos, J. Sylvester, G. dos Reis, "Identification and machine learning prediction of knee-point and knee-onset in capacity degradation curves of lithium-ion cells," *Energy and AI*, vol. 1, 100006, 2020. doi:10.1016/j.egyai.2020.100006
- [S6] L. Canals Casals, B. Amante García, C. Canal, "Second life batteries lifespan: Rest of useful life and environmental analysis," *J. Environmental Management*, vol. 232, pp. 354–363, 2019. doi:10.1016/j.jenvman.2018.11.046 †
- [S7] E. Martinez-Laserna, I. Gandiaga, E. Sarasketa-Zabala, J. Badeda, D.-I. Stroe, M. Swierczynski, et al., "Battery second life: Hype, hope or reality? A critical review of the state of the art," *Renew. Sustain. Energy Rev.*, vol. 93, pp. 701–718, 2018. doi:10.1016/j.rser.2018.04.035 †
- [S8] E. Hossain, D. Murtaugh, J. Mody, H. M. R. Faruque, M. S. H. Sunny, N. Mohammad, "A comprehensive review on second-life batteries: Current state, manufacturing considerations, applications, impacts, barriers & potential solutions, business strategies, and policies," *IEEE Access*, vol. 7, pp. 73215–73252, 2019. doi:10.1109/ACCESS.2019.2917859 †
- [S9] M. F. Börner, M. H. Frieges, B. Späth, K. Spütz, H. H. Heimes, D. U. Sauer, W. Li, "Challenges of second-life concepts for retired electric vehicle batteries," *Cell Reports Physical Science*, 101095, 2022. doi:10.1016/j.xcrp.2022.101095 †
- [S10] X. Lai, Y. Huang, C. Deng, H. Gu, X. Han, Y. Zheng, M. Ouyang, "Sorting, regrouping, and echelon utilization of the large-scale retired lithium batteries: A critical review," *Renew. Sustain. Energy Rev.*, vol. 146, 111162, 2021. doi:10.1016/j.rser.2021.111162 †
- [S11] A. Takahashi, A. Allam, S. Onori, "Evaluating the feasibility of batteries for second-life applications using machine learning," *iScience*, vol. 26, no. 4, 106547, 2023. doi:10.1016/j.isci.2023.106547 (README ref. [4])
- [S12] = [D3]

**Standards and regulation** (scope confirmed from publisher or catalogue pages; full texts not read, so no numerical thresholds are taken from them)

- [T1] ANSI/CAN/UL 1974, *Standard for Evaluation for Repurposing or Remanufacturing Batteries*, 2nd ed., 10 Nov 2023 (1st ed. 2018). https://webstore.ansi.org/standards/ul/ansiul19742023
- [T2] IEC 63330-1:2024, *Requirements for repurposing of secondary batteries – Part 1: General requirements*. https://webstore.iec.ch/en/publication/66561
- [T3] Regulation (EU) 2023/1542 concerning batteries and waste batteries, Art. 14 and Annex VII. https://eur-lex.europa.eu/eli/reg/2023/1542/oj (Annex VII parameter list taken from secondary summaries† and must be checked against the official text)
- [T4] GB/T 34015-2017, *Recycling of traction battery used in electric vehicle – Test of residual capacity*; GB/T 34015.3-2021, *Echelon use – Part 3: Echelon using requirement*. Catalogue: https://www.codeofchina.com/standard/GBT34015-2017.html
