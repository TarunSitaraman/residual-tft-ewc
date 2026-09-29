# External data registry

This file lists **every external data source** that this repository uses, or that its results depend on. For each one it records the verified citation, licence, version, integrity hashes and how the source is used. It is the single reference for the paper's data statement, citations and acknowledgements.

**Rules**
- A source may feed reported results only if it is listed here with status **Used**.
- **Nothing is downloaded until its row is complete.** A complete row gives:
  - name, authors, publication DOI, and data DOI or URL;
  - version or access date, licence and any required citation text;
  - file checksums;
  - the experiments that use it.
- Save each new source's metadata records in `docs/external_data/metadata/`.
- Raw external files stay outside Git, under `data/`, like the handover CSVs.
- Every field below was checked on **29 Sep 2026** against primary records, never against summaries or memory:
  - the Crossref record for each article;
  - the article's full text in PubMed Central;
  - the DataCite record for the dataset DOI;
  - the OSF API records for the project, file and licence.
- The saved records and their SHA-256 values are listed in [§5](#5-evidence-saved-in-this-repository).

| ID | Source | Role | Status | Used by |
|---|---|---|---|---|
| [S1](#s1-stanford-second-life-dataset-moy-et-al-2024) | Stanford second-life dataset, OSF `8jnr5` (Moy et al., 2024) | Origin of all data used in this work | **Used** (via S2; diagnostic tests downloaded directly) | everything below; raw RPT/EIS files in `data/raw/osf_8jnr5_v2/` (not yet used by any experiment) |
| [S2](#s2-project-handover-and-its-hugging-face-mirror) | Project handover CSVs, mirrored on Hugging Face | Processed form of S1 that the pipeline reads | **Used** | notebooks 01–10; `results/*.json`; `scripts/verify_external_data.py` |
| [S3](#s3-first-life-dataset-pozzato-et-al-2022) | Stanford first-life dataset, OSF `qsabn` (Pozzato et al., 2022) | First-life history of the same cells | Cited for background; data **not used** | none (candidate for S-b) |
| [S4](#s4-synthetic-duty-cycles-moy-and-onori-2023) | Synthetic grid-storage duty cycles (Moy and Onori, 2023) | Origin of the second-life cycling profiles | Cited; no data used | methods text only |
| [S5](#s5-cui-et-al-2024) | Cui et al., 2024 | Related second-life study from the same laboratory | Candidate citation; relation to S1 **unverified** | none |
| [C1–C4](#candidates-not-verified-nothing-downloaded) | NASA, Oxford, CALCE, MIT–Stanford ageing datasets | Suggested for more varied fade | **Not verified; nothing downloaded** | none |

---

## S1. Stanford second-life dataset (Moy et al., 2024)

### Citation (cite both the article and the dataset)

**Data article**
- K. Moy, M. A. Khan, S. Fasolato, G. Pozzato, A. Allam and S. Onori, "Second-life lithium-ion battery aging dataset based on grid storage cycling," *Data in Brief*, vol. 57, Art. no. 111046, Dec. 2024.
  - doi:[10.1016/j.dib.2024.111046](https://doi.org/10.1016/j.dib.2024.111046)
  - PMCID PMC11639452; PMID 39678446
- BibTeX key: `moy2024secondlife`.
- Received 15 Jul 2024; revised 7 Oct 2024; accepted 10 Oct 2024.
- ORCIDs:
  - Moy: 0000-0002-3476-408X
  - Khan: 0009-0008-7506-9690
  - Pozzato: 0000-0001-5253-0271
  - Onori: 0000-0002-6556-2608

**Dataset**
- K. Moy *et al.*, "Second-life lithium-ion battery aging dataset based on grid storage cycling," Open Science Framework, 2024, file version 2 (27 Feb 2025).
  - doi:[10.17605/OSF.IO/8JNR5](https://doi.org/10.17605/OSF.IO/8JNR5), https://osf.io/8jnr5/
- BibTeX key: `moy2024secondlifedata`.
- **Required citation text:** the OSF project sets no custom citation (`custom_citation: null`). Its description asks users to refer to the accompanying data article. So the primary citation is the Moy et al. 2024 article (not Cui et al. 2024), together with the dataset DOI.
- The DataCite record lists only S. Onori and M. A. Khan as creators. The OSF project description refers users to the data article, so the dataset citation uses the article's author list.

### Licences
| Item | Licence | Source of the statement |
|---|---|---|
| Dataset (OSF `8jnr5`) | **CC BY 4.0** | DataCite `rightsList` (SPDX `cc-by-4.0`); OSF licence record `563c1cf88c5e4a3877f9e96a` ("CC-By Attribution 4.0 International") |
| Data article | CC BY-NC 4.0 | Crossref licence record |

Under CC BY 4.0, reuse requires:
- credit to the creators;
- a link to the licence;
- an indication of whether changes were made.

See [§6](#6-attribution-obligations).

### Version and integrity
| Field | Value |
|---|---|
| OSF project | `8jnr5`, public, created 2024-07-11, modified 2025-06-18 |
| File | `SL_Dataset_SECL_INR21700-M50T.zip` (OSF file GUID `j2ne9`) |
| Version used | **2**, dated 2025-02-27 |
| Size | 2,153,883,052 bytes (2.21 GB uncompressed; 648 data files) |
| SHA-256 | `201b21b8e2e9623887d03e9bd6620341e5c91cd28bce34294ccbad2d4e7c6fe3` |
| MD5 | `9f8aab95c3e9bded5b823af179dbe5a1` |
| Version 1 | 2024-07-11, 1,323,917,500 bytes. It matches the article's description: `Cycling_1`–`15`, data up to 6 Oct 2024 |
| Version 2 contents | `Cycling_1`–`18`, `RPT_1`–`19` (the article notes that the repository is updated as experiments continue) |

The article describes version 1. Version 2 extends the same campaign, and this work uses version 2. The paper must therefore cite the file version, not only the DOI, and say that it extends the period described in the article.

### What the source contains (from the data article)
- **Laboratory:** Stanford Energy Control Laboratory, Stanford University, CA, USA.
- **Equipment:**
  - Arbin LBT21024 cycler
  - Amerex IC500R thermal chamber
  - Gamry EIS 1010E
  - T-type thermocouples
- **Cells:**
  - Six LG Chem INR21700-M50T cells, a subset of the ten cells aged in first life (S3).
  - Chemistry: NMC (LiNiMnCoO₂) cathode, graphite–silicon anode.
  - Nominal capacity 4.85 Ah, nominal voltage 3.63 V.
  - Cut-offs: 4.2 V charge, 2.5 V discharge, 50 mA cut-off current.
  - Standard charge current 1.455 A.
  - Specifications are from the manufacturer's sheet, as cited in the article.
- **First life, before this dataset** (Table 2 of the article):

  | Cell | First-life charge rate | First-life cycles | Capacity at end of first life (Ah) | Second-life duty cycle |
  |---|---|---|---|---|
  | V4 | C/4 | 244 | 4.6024 | Residential |
  | W8 | C/2 | 347 | 4.4568 | Commercial |
  | W9 | 1C | 341 | 4.4636 | Residential |
  | V5 | 1C | 29 | 4.7055 | Commercial |
  | W10 | 3C | 350 | 4.4591 | Residential |
  | G1 | 3C | 212 | 4.6000 | Commercial |

- **Second-life protocol:**
  - Cycling starts 28 Oct 2022 with synthetic residential and commercial peak-shaving duty cycles of 48 h each (S4).
  - Cycling alternates between 20 °C and 35 °C.
  - RPTs at 25 °C: a capacity test with pulses at 4.00, 3.63 and 3.26 V, plus EIS at those voltages on most RPTs.
  - Calendar ageing occurs during stops.
- **Funding acknowledged in the article:**
  - Bits and Watts Initiative, Precourt Institute for Energy, Stanford University;
  - Stanford Chevron Energy Fellowship.

### Evidence that S2 is derived from S1 (checked 29 Sep 2026)
`scripts/verify_external_data.py` reads the central directory of the version-2 archive by HTTP range requests, without downloading it, and compares it with the handover CSVs. The output is saved in `docs/external_data/metadata/osf_8jnr5_v2_listing.json`.

| Check | Result |
|---|---|
| File names referenced in `rpt_capacity_data.csv` found in the archive | **337 / 337** |
| … and inside the same `RPT_#` folder | **337 / 337** |
| (cell, `Cycling_#`) blocks in `cleaned_battery_data.csv` present in the archive | **103 / 103** |
| Archive blocks absent from the handover | **none** (e.g. V4 and W10 lack `Cycling_16` and W9 stops after `Cycling_15`, in both) |
| Cell IDs | G1, V4, V5, W8, W9, W10 in both |
| Start date | 28 Oct 2022 in both |
| Standard CC charge current | 1.455 A (article) vs 1.46 A (handover step 2) |

**Conclusion:** the handover data are a processed subset of S1, version 2. Remaining limits:
- Measured values were not compared number by number, because the raw `.xlsx` files were not downloaded. The derivation of `Capacity_Ah` from the raw RPT files is therefore not independently reproduced yet.
- The handover author (Darshan D) should confirm which archive version was processed.

### Parts of S1 not in the handover
The archive also holds everything below. All of it is covered by the same licence and can be used with the same citation.
- **237 EIS spectrum files** (`diagnostic_tests/RPT_#/…EIS…xlsx`).
- **112 raw RPT files** (full capacity-test and pulse curves).
- Processed `.mat` files and the authors' MATLAB script `data_analysis.m`.

### Downloaded directly: the diagnostic tests
- **Script:** `scripts/fetch_osf_diagnostics.py`. It downloads the whole version-2 archive, checks its size and SHA-256 against the OSF file record above, and stops on any mismatch. It then extracts only `diagnostic_tests/`, which holds all RPT capacity tests with pulses and all EIS spectra.
- **Local copy:** the files go to `data/raw/osf_8jnr5_v2/diagnostic_tests/`, which is git-ignored.
- **Manifest (versioned):** `docs/external_data/metadata/osf_8jnr5_v2_diagnostic_manifest.json` gives the path, size, CRC-32 and SHA-256 of every file, plus the retrieval time.
- **Use:** no experiment uses these files yet. Before any result depends on them, record which experiments use them in the table at the top.

---

## S2. Project handover and its Hugging Face mirror

**What it is.** Seven CSV files produced within this project (by Darshan D, before this repository's pipeline) from the S1 archive:
- merged raw cycling data;
- extracted RPT capacities;
- SOH labels;
- cycle-level features.

The processing script is **not** in this repository; see `README.md` §1.2 for the described workflow and `docs/DATA_REPORT.md` §2 for where the delivered files differ from their description.

**Where it is.**
- https://huggingface.co/datasets/TarunSitaraman/second-life-ev-battery-datasets (public).
- Revision `eb9d95b2fed78fbfd300a09009ab999cc0e6716b`, last modified 2026-09-29T07:44:58Z.
- It is a convenience mirror for cloud sessions, not an independent source. The paper cites S1, not the mirror.

**Files as used (SHA-256, local copies in `data/`, git-ignored)**

| File | Bytes | SHA-256 |
|---|---|---|
| `cleaned_battery_data.csv` | 66,856,524 | `98328f91d41747d585bb9a06ebf5b86dbf4df2f30432142bf5056088b1bd5503` |
| `final_battery_cycling_features.csv` | 74,307,658 | `32fab8035474fde7010f55a895dac4c8df7c8ced530f7682316f243af5c98865` |
| `final_battery_health_dataset.csv` | 8,429 | `99b478f148a1a023f023281b4e318752f654d74a4e11ce6951a0d964a739bf18` |
| `final_ml_dataset.csv` | 255,358 | `e55a29ed1cbba4a630dc246a7da7278ceb704c550ae67ab830a14a98016e93f5` |
| `ml_training_dataset.csv` | 245,270 | `6ae398a68b9122ebcc1ccb8687e3d2961313700c433910aff4c5c2c2351ec9ad` |
| `rpt_capacity_data.csv` | 27,140 | `df28cd3aa4001423994defb414f86a392827ae1ccbdab4bbcdb5a525367ab3f6` |
| `DATA_DICTIONARY.csv` | 1,435 | `b12518a1d719f4b17795584ae46c94c69a2f2b9c3ed9371c01a744ae0cb0520c` |

The two large files match the Git LFS object IDs of the mirror.

**Open compliance issue.** The mirror has **no dataset card** (`README.md` returns 404). It therefore redistributes CC BY 4.0 data without the required attribution, licence link or indication of changes. Before the paper is submitted, and ideally now, add a dataset card to the mirror that:
1. names S1 with both citations and the DOI 10.17605/OSF.IO/8JNR5;
2. states the licence (CC BY 4.0, with a link) and sets `license: cc-by-4.0` in the card metadata;
3. states that the files are derived from, and are a processed subset of, S1 version 2, and lists the changes (merging, resampling as delivered, RPT capacity extraction, SOH labelling, feature aggregation);
4. states that the mirror is not endorsed by the original authors.

Alternatively, make the mirror private and point users to OSF.

---

## S3. First-life dataset (Pozzato et al., 2022)

- **Citation:** G. Pozzato, A. Allam and S. Onori, "Lithium-ion battery aging dataset based on electric vehicle real-driving profiles," *Data in Brief*, vol. 41, Art. no. 107995, Apr. 2022.
  - doi:[10.1016/j.dib.2022.107995](https://doi.org/10.1016/j.dib.2022.107995)
  - PMCID PMC8892076; PMID 35252504
  - BibTeX key: `pozzato2022firstlife`
- **Article licence:** CC BY 4.0.
- **Data:**
  - Repository `Dataset_SECL_INR21700-M50T` at `osf.io/qsabn`, linked from the article through a view-only URL.
  - The OSF API returns 401 for this node, so its data licence and version **could not be verified**.
  - The article describes ten INR21700-M50T cells tested at 23 °C for 23 months with UDDS discharge and CC–CV charging at C/4 to 3C (`Cycling_1`–`8`).
- **Use here:**
  - Cited for the cells' first-life history. The first-life values in the S1 table above come from the S1 article, not from these data.
  - **No data from S3 are used.** If the planned sensitivity analysis S-b needs beginning-of-life capacities from S3, first register its licence and version here.

## S4. Synthetic duty cycles (Moy and Onori, 2023)

- **Citation:** K. Moy and S. Onori, "Synthetic grid storage duty cycles for second-life lithium-ion battery experiments," *SAE Int. J. Adv. & Curr. Prac. in Mobility*, vol. 6, no. 1, pp. 261–269, 2023 (SAE Technical Paper 2023-01-0516, published 11 Apr 2023).
  - doi:[10.4271/2023-01-0516](https://doi.org/10.4271/2023-01-0516)
  - BibTeX key: `moy2023dutycycles`
- **Use here:** cited when describing the residential and commercial profiles (reference [2] of the S1 article). No data used.

## S5. Cui et al., 2024

- **Citation:** X. Cui, M. A. Khan, G. Pozzato, S. Singh, R. Sharma and S. Onori, "Taking second-life batteries from exhausted to empowered using experiments, data analysis, and health estimation," *Cell Reports Physical Science*, vol. 5, no. 5, Art. no. 101941, May 2024.
  - doi:[10.1016/j.xcrp.2024.101941](https://doi.org/10.1016/j.xcrp.2024.101941)
  - Article licence CC BY-NC 4.0
  - BibTeX key: `cui2024secondlife`
- **Status:**
  - The metadata are verified.
  - Whether it analyses the same six cells is **not** verified: the publisher's full text refused automated access, and Europe PMC has no record.
  - Read the paper before citing it as prior work on this dataset. It is a likely direct competitor for any SOH-estimation claim on these cells.

## Candidates (not verified, nothing downloaded)

These were suggested as additional data with larger and more varied fade (`docs/DATA_REPORT.md` §5, option 2). None has been checked or downloaded. Each needs a full row (S-format) before any use.

| ID | Dataset | Known so far |
|---|---|---|
| C1 | NASA Ames (PCoE) battery ageing data | Nothing verified |
| C2 | Oxford battery degradation data | Nothing verified |
| C3 | CALCE battery data (University of Maryland) | Nothing verified |
| C4 | MIT–Stanford–Toyota fast-charging data (Severson et al., 2019) | Nothing verified |

---

## 4. Checklist for the paper

- [ ] **Data section:** cite `moy2024secondlife` and `moy2024secondlifedata`, and name the file version 2 (27 Feb 2025) and its SHA-256.
- [ ] **Data section:** state that the analysis extends beyond the period described in the article (to Jan 2025).
- [ ] **Data section:** state that the processed CSVs were derived by the authors.
- [ ] **Data section:** describe the cells from S1 (table above) and cite `moy2023dutycycles` for the duty cycles and `pozzato2022firstlife` for first life.
- [ ] **Heterogeneity:** narrow "heterogeneous" to first-life charge rate and ageing, plus residential vs commercial duty cycle. All cells are one model and chemistry, from one laboratory.
- [ ] **Acknowledgements:** thank the Stanford Energy Control Laboratory for making the data public.
- [ ] **Data availability statement:** the raw data are at doi:10.17605/OSF.IO/8JNR5 under CC BY 4.0. Give the processing code and the commit hash.
- [ ] **Hugging Face mirror:** fix the dataset card (S2), or remove the mirror, before submission.
- [ ] **Handover version:** confirm with the handover author that version 2 of the archive was processed.
- [ ] **Capacity labels:** reproduce `Capacity_Ah` from the raw RPT files, or state the handover's extraction as a limitation.
- [ ] **Cui et al. 2024:** read it and decide whether it is prior work on the same cells (S5).
- [ ] **Comparisons with the literature:** do not compare with numbers from other datasets or SOH definitions without saying so (see `CLAUDE.md`).

## 5. Evidence saved in this repository

All records are in `docs/external_data/metadata/` and were retrieved on 29 Sep 2026. The PMC full texts are redistributed under the articles' licences: S1 CC BY-NC 4.0 and S3 CC BY 4.0, © the authors.

| File | Record | SHA-256 |
|---|---|---|
| `moy2024_second_life_dib.crossref.json` | Crossref, S1 article | `368dedc92120bbcf76202e343285246b5a8a0ffa54bab96c3f9b8766aae1edbc` |
| `moy2024_second_life_dib.fulltext.xml` | Europe PMC full text, S1 article | `eafd0e81fe836141c96cc397bd70dc0813f5583d45f220154450d525426b64e5` |
| `moy2024_second_life_osf.datacite.json` | DataCite, S1 dataset DOI | `d3eac24b456057810efbbf8feb329b1ce7749d52a814b52625030aaec0f477e7` |
| `moy2024_second_life_osf.osf_node.json` | OSF API, project `8jnr5` | `20ffa6c51963b2e85a876767e5a99442e65755e94587d5e44fa8f0a90168553e` |
| `moy2024_second_life_osf.file.json` | OSF API, archive file (version, size, hashes) | `a02bfed51799e7b0f06448a5dc8c241afe42e5b6f3ebe59c5bb284b3b0ee9bcb` |
| `moy2024_second_life_osf.license.json` | OSF API, licence record | `32fb3d7b32be0843803f96424fe7739f69574481211a2d323e16f8da8536c635` |
| `osf_8jnr5_v2_listing.json` | Archive file listing and handover comparison (`scripts/verify_external_data.py`) | regenerated by the script |
| `osf_8jnr5_v2_diagnostic_manifest.json` | Checksums of the extracted diagnostic-test files (`scripts/fetch_osf_diagnostics.py`) | regenerated by the script |
| `pozzato2022_first_life_dib.crossref.json` | Crossref, S3 article | `992367445f74b4ac41a277357aded9bb0863a41656349d1cb06efc7e132692f7` |
| `pozzato2022_first_life_dib.fulltext.xml` | Europe PMC full text, S3 article | `6dd38d6a8c5bf581164b7fe06f064f703cac49f6b69d6e9dec1381237c90a283` |
| `moy2023_synthetic_duty_cycles_sae.crossref.json` | Crossref, S4 | `fca9170f6d3b734481623af611d68545f3dc7e7614d3a6a1b025e296a8d3e176` |
| `cui2024_second_life_crps.crossref.json` | Crossref, S5 | `ba227bc78dcdca6a309c22de72e0645aa20d243787dee45bf46e089a62279ed5` |

The remaining references in `paper/references.bib` that have DOIs (TFT, EWC, LwF, LSTM, Wilcoxon, Cohen, Kerby) were also checked against Crossref on the same date. Entries without DOIs still need checking against the publisher.

## 6. Attribution obligations

| Where | Required text or action |
|---|---|
| Paper | Cite S1 article + dataset DOI. State the licence (CC BY 4.0) and that the data were processed (subset, derived features, labels) |
| Repository `README.md` | Data-source paragraph citing S1 and the licence (done) |
| Hugging Face mirror | Dataset card as in S2 (**open**) |
| Any redistributed derived file | Same attribution, licence link and change notice |
