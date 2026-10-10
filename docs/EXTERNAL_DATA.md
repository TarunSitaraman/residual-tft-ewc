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
| [S1](#s1-stanford-second-life-dataset-moy-et-al-2024) | Stanford second-life dataset, OSF `8jnr5` (Moy et al., 2024) | Origin of all data used in this work | **Used** (via S2; diagnostic tests downloaded directly) | everything below; raw RPT/EIS files in `data/raw/osf_8jnr5_v2/` feed `scripts/build_diagnostic_features.py`, notebook 11 and `results/diagnostic_features.json` |
| [S2](#s2-project-handover-and-its-hugging-face-mirror) | Project handover CSVs, mirrored on Hugging Face | Processed form of S1 that the pipeline reads | **Used** | notebooks 01–10; `results/*.json`; `scripts/verify_external_data.py` |
| [S3](#s3-first-life-dataset-pozzato-et-al-2022) | Stanford first-life dataset, OSF `qsabn` (Pozzato et al., 2022) | First-life history of the same cells | Cited for background; data **not used** | none (candidate for S-b) |
| [S4](#s4-synthetic-duty-cycles-moy-and-onori-2023) | Synthetic grid-storage duty cycles (Moy and Onori, 2023) | Origin of the second-life cycling profiles | Cited; no data used | methods text only |
| [S5](#s5-cui-et-al-2024-and-the-stanfordrelyion-dataset) | Cui et al., 2024, and the Stanford–Relyion dataset, OSF `fns57` | A different second-life dataset (8 retired Nissan Leaf LMO cells) | Cited as related work; data **not used**, since no licence is declared (permission to be requested) | none |
| [S6](#s6-lfp-first--and-second-life-dataset-wheeler-et-al-2025) | Wheeler et al., 2025, Recherche Data Gouv `10.57745/OLBXKT` | Second dataset (D19): 20 LFP cells, 7 ageing conditions, down to 30–50 % SOH | **Used** (D19, D20): characterisation files downloaded (checksummed) | `scripts/fetch_wheeler_characterisation.py`, `scripts/build_wheeler_features.py`, notebook 13, `results/forecasting_wheeler.json` |
| [C1–C4](#candidates-screened-not-used) | NASA, Oxford, CALCE, MIT–Stanford ageing datasets | Suggested for more varied fade | **Not verified; nothing downloaded** | none |
| [C5–C6](#candidates-screened-not-used) | Beatty et al. 2025 (Loughborough); Mowri et al. 2025 | Second-life datasets screened for D19 | Metadata verified; **rejected** (reasons below) | none |
| [S7](#s7-s9-first-life-pretraining-pool-registered-3-oct-2026) | Luh and Blank, 2024, KIT RADAR `10.35097/1969` | First-life pool (D21 draft): 228 NMC/C+SiO cells, calendar/cyclic/driving ageing, periodic check-ups | **Downloaded and verified** (3 Oct 2026; MD5 matches the registry) | `scripts/fetch_pool.py`; D21 (draft) |
| [S8](#s7-s9-first-life-pretraining-pool-registered-3-oct-2026) | Wang et al., 2024 (XJTU), Zenodo `10.5281/zenodo.10963339` | First-life pool: 55 NCM cells, 6 protocol batches | **Downloaded and verified** (3 Oct 2026; MD5 matches the registry) | `scripts/fetch_pool.py`; D21 (draft) |
| [S9](#s7-s9-first-life-pretraining-pool-registered-3-oct-2026) | Zhu et al., 2022 (Tongji), Zenodo `10.5281/zenodo.6405084` | First-life pool: 130 cells, NCA / NCM / NCM+NCA | **Downloaded and verified** (3 Oct 2026; MD5 matches the registry) | `scripts/fetch_pool.py`; D21 (draft) |
| [C7–C8](#candidates-screened-not-used) | Severson/Attia (MIT–Stanford–TRI, data.matr.io); Preger et al. 2020 (Sandia, batteryarchive.org) | First-life candidates for the pool | **Not usable yet:** no data licence verifiable from a primary record | none |

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
- **Capacity labels reproduced (29 Sep 2026):** all 112 RPT capacity rows in `rpt_capacity_data.csv` (111 RPTs; g1's RPT_1 spans two files) were recomputed exactly from the raw files (maximum absolute difference 0.0 Ah; `scripts/verify_external_data.py --raw-root`).
  - Definition: `Capacity_Ah` is the maximum of the Arbin cumulative `Discharge_Capacity(Ah)` counter over the whole RPT. It therefore includes the small pulse discharges, not only the constant-current capacity step.
  - This is the handover's definition. It is superseded for all reported results by D17: labels use the C/20 discharge capacity with the pulses excluded (`data/rpt_health_c20.csv`). The paper defines capacity that way.
- **Archive version settled by evidence:** the handover contains `Cycling_16`–`18`, `RPT_16`–`19` and RPTs dated up to 29 Jan 2025. Version 1 ends at `Cycling_15` and 6 Oct 2024, so the handover was produced from version 2 (or a later one with identical file names). No confirmation from the handover author is needed.

### Parts of S1 not in the handover
The archive also holds everything below. All of it is covered by the same licence and can be used with the same citation.
- **237 EIS spectrum files** (`diagnostic_tests/RPT_#/…EIS…xlsx`).
- **112 raw RPT files** (full capacity-test and pulse curves).
- Processed `.mat` files and the authors' MATLAB script `data_analysis.m`.

### Downloaded directly: the diagnostic tests
- **Script:** `scripts/fetch_osf_diagnostics.py`. It downloads the whole version-2 archive, checks its size and SHA-256 against the OSF file record above, and stops on any mismatch. It then extracts only `diagnostic_tests/`, which holds all RPT capacity tests with pulses and all EIS spectra.
- **Local copy:** the files go to `data/raw/osf_8jnr5_v2/diagnostic_tests/`, which is git-ignored.
- **Manifest (versioned):** `docs/external_data/metadata/osf_8jnr5_v2_diagnostic_manifest.json` gives the path, size, CRC-32 and SHA-256 of every file, plus the retrieval time.
- **Use:** diagnostic features (decision D16): `scripts/build_diagnostic_features.py`, notebook 11, `results/diagnostic_features.json`.

---

## S2. Project handover and its Hugging Face mirror

**What it is.** Seven CSV files produced within this project (by Darshan D, before this repository's pipeline) from the S1 archive:
- merged raw cycling data;
- extracted RPT capacities;
- SOH labels;
- cycle-level features.

The processing script is **not** in this repository. See `README.md` §1.2 for the described workflow and `docs/DATA_REPORT.md` §2 for where the delivered files differ from their description.

**Status (29 Sep 2026): frozen.** The handover is complete, and the team now owns all data processing (decision D15).
- These CSVs remain valid inputs: their RPT capacities are reproduced exactly from the raw files, and their blocks match the archive.
- Any new processed data (EIS features, raw-curve indicators, re-extracted cycling data) is produced by code in this repository from S1 directly, never by hand.

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

## S5. Cui et al., 2024, and the Stanford–Relyion dataset

**Article.**
- X. Cui, M. A. Khan, G. Pozzato, S. Singh, R. Sharma and S. Onori, "Taking second-life batteries from exhausted to empowered using experiments, data analysis, and health estimation," *Cell Reports Physical Science*, vol. 5, no. 5, Art. no. 101941, May 2024.
  - doi:[10.1016/j.xcrp.2024.101941](https://doi.org/10.1016/j.xcrp.2024.101941)
  - Article licence CC BY-NC 4.0; arXiv:2402.18859
  - BibTeX key: `cui2024secondlife`

**What it is (verified 29 Sep 2026 from the arXiv full text).** A **different** dataset from S1:
- eight retired LMO/graphite pouch cells (32.5 Ah nominal) from two Nissan Leaf packs, with unknown first-life history;
- 15 months of grid peak-shaving cycling between 3 and 4 V;
- RPTs including capacity tests and pulses.

The earlier question (whether it analyses the S1 cells) is resolved: it does not. It is related prior work on second-life SOH estimation, from the same laboratory.

**Dataset.**
- "Stanford-Relyion Second-Life Li-ion Battery Dataset," OSF, 2024, doi:[10.17605/OSF.IO/FNS57](https://doi.org/10.17605/OSF.IO/FNS57).
- DataCite creators: S. Singh, R. Sharma, G. Pozzato, X. Cui, S. Onori, M. A. Khan.
- Created 2024-01-12, updated 2025-05-01.

**Licence: none declared.**
- The DataCite `rightsList` is empty and the OSF node has no licence. The project description only asks that use be "appropriately referenced".
- Without a licence, reuse rights are not granted. **Do not download or use it** until the authors grant permission in writing (corresponding author: S. Onori). File that permission under `docs/external_data/`.
- If permission is granted, S5 becomes the most direct complement to S1: real retired EV cells on a grid duty cycle, with a different chemistry and format.

## S6. LFP first- and second-life dataset (Wheeler et al., 2025)

**Citation (cite both the article and the dataset).**
- **Data article:** W. Wheeler, Y. Bultel, P. Venet and A. Sari, "An ageing study of twenty 18650 lithium-ion Graphite/LFP cells in first and second life use," *Scientific Data*, vol. 12, Art. no. 392, 2025.
  - Published 6 Mar 2025
  - doi:[10.1038/s41597-025-04712-7](https://doi.org/10.1038/s41597-025-04712-7); PMCID PMC11885844
  - Article licence CC BY 4.0
- **Dataset:** W. Wheeler, P. Venet, Y. Bultel and A. Sari, "Aging study on twenty A123 18650 Graphite/LFP 1.1 Ah cells," Recherche Data Gouv, version 2.0 (released 11 Mar 2025).
  - doi:[10.57745/OLBXKT](https://doi.org/10.57745/OLBXKT)
  - Licence **CC BY 4.0** (dataset terms of use; DataCite `rightsList` is empty)
- **Funding acknowledged:** ANRT, CIFRE grant 2018/1625.

**What it contains (from the article).**
- **Cells:** 20 A123 18650 Graphite/LFP cells, 1.1 Ah nominal, 3.2 V nominal, 2.0–3.65 V limits.
- **Test cases:** seven, with three cells each except **case 3, which has two** (cells 3a and 3b). The article contradicts itself here: its text says "Test case 4 had only two cells", but its Table 2 lists cells 4a–4c and 3a–3b only, and the table's footnote says "except for test case 3". The data archive agrees with Table 2 (`Cell4c.zip` exists; there is no `Cell3c.zip`). Cite the table, not the sentence. They vary the charge (CC-CV or CC; 3.65 or 3.55 V; C/3 or 2C), the discharge (WLTP at C/3 on average, or CC 2C) and the depth of discharge (70 % or 100 %).
- **Ageing:** at 50 °C (accelerated) for 24 months, ending at 50–30 % SOH, including the "ageing knee".
- **Characterisation every 100 cycles:** C/3 and C/25 (pseudo-OCV) charge and discharge. EIS at 0/50/100 % SOC before and after ageing. Initial and final characterisation at 25 °C.
- **"Second life" here** means continued ageing below 80 % SOH under the same protocol, not a repurposed stationary duty cycle. The paper must say so.

**Version and integrity.**
- 24 files, 26,298,927,784 bytes in total: 20 per-cell zips of 0.93–1.65 GB, plus the MATLAB script, `extractedData.mat` (459,258 bytes) and the readme.
- MD5 checksums of every file are in `docs/external_data/metadata/wheeler2025_lfp_dataverse.dataset.json`; downloads are checked against them.

**Why it was selected (D19).** It has a clear CC BY 4.0 licence and adds several kinds of heterogeneity that S1 lacks:
- another chemistry (LFP);
- seven controlled usage conditions (groups for continual learning);
- deep fade with knees, so the reuse classes C1–C3 occur.

Its C/25 characterisation matches S1's C/20 capacity test, so the capacity labels (D17) and IC features (D16) carry over.

**Downloaded (29 Sep 2026).**
- **Characterisation files:** `scripts/fetch_wheeler_characterisation.py` read only these from the 20 per-cell zips, using HTTP range requests on the zip index. That is 1,007 files: every 50 °C check plus the initial and final 25 °C characterisations and their EIS files, 19,978,037,377 bytes uncompressed.
  - Each file was CRC-32-checked against the zip index and is stored gzip-compressed in `data/raw/wheeler2025/` (git-ignored, 3.6 GB).
  - The SHA-256 of each file's uncompressed content, with its source zip and that zip's MD5, is in `docs/external_data/metadata/wheeler2025_characterisation_manifest.json`.
- **Authors' summary files:** `readme.txt`, `extractedData.mat` and `Matlab_BasicImportAndDisplay.m` were downloaded directly and checked against their repository MD5 values.
- **Parsing check:** `scripts/build_wheeler_features.py` recomputes the authors' own capacity (the C/3 charge counter in `extractedData.mat`). It matched exactly (difference 0.0 Ah) on all 843 checks present in both.

**Use:** forecasting protocol D20 (notebook 13).

## S7–S9. First-life pretraining pool (registered 3 Oct 2026)

**Why.** S1 and S6 together have 26 cells, too few for a Temporal Fusion Transformer to be the right model (`docs/RESULTS.md`). These three datasets are candidates for a first-life pretraining pool, followed by continual learning on the second-life data (S1, and S6 below 80 % SOH). Their use is fixed in decision D21 (draft) before any model is trained on them. They are first-life data: no second-life claim rests on them.

**Verification (all on 3 Oct 2026, from primary records saved in `docs/external_data/metadata/`).** For each: the data repository's own record (licence, version, files, checksums), the data article on Crossref, and the article abstract on Europe PMC (cell counts).

### S7. Luh and Blank 2024 (KIT)
- **Article:** M. Luh and T. Blank, "Comprehensive battery aging dataset: capacity and impedance fade measurements of a lithium-ion NMC/C-SiO cell," *Scientific Data*, vol. 11, Art. no. 1004, 2024. doi:10.1038/s41597-024-03831-x. Article licence CC BY 4.0 (Crossref).
- **Data:** "… [dataset – version 2: result data]," Karlsruhe Institute of Technology, RADAR4KIT, doi:[10.35097/1969](https://doi.org/10.35097/1969). Parent dataset doi:10.35097/1947 (version 2 raw log data, 81.9 GB, not needed: doi:10.35097/kww7jv8ajuvchcah).
- **Licence:** **CC BY 4.0** (DataCite `rightsList`, SPDX `cc-by-4.0`, for both 10.35097/1969 and 10.35097/1947).
- **Content (article abstract):** 228 commercial NMC/C+SiO cells aged for more than a year under calendar, cyclic and driving-cycle conditions; result data are the capacity and impedance measured in check-ups.
- **File:** `10.35097-1969.tar`, 333,402,624 bytes, MD5 `aa60c8fd08755a253ac7479dde41c93d` (RADAR landing page; excerpt saved in `luh2024_kit_result_data.landing.txt`). Download URL: `https://radar.kit.edu/radar-backend/archives/CoNQplSNoVeXExyV/versions/1/content` (supports range requests).

### S8. Wang et al. 2024 (XJTU)
- **Article:** F. Wang, Z. Zhai, Z. Zhao, Y. Di *et al.*, "Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis," *Nature Communications*, 2024. doi:10.1038/s41467-024-48779-z. Article licence CC BY 4.0.
- **Data:** Zenodo doi:[10.5281/zenodo.10963339](https://doi.org/10.5281/zenodo.10963339), version V1, published 2024-04-12, creator F. Wang. **Licence CC BY 4.0** (Zenodo record).
- **Content:** 55 NCM cells (article abstract) in six protocol batches (`Batch-1` … `Batch-6`, 8+15+8+8+8+8 `.mat` files, read from the zip index without downloading). The archive holds only these authors' cells.
- **File:** `Battery Dataset.zip`, 2.44 GB, MD5 `a635cef9678e0de21f9d5c1f78a4342c` (Zenodo).

### S9. Zhu et al. 2022 (Tongji)
- **Article:** J. Zhu, Y. Wang, Y. Huang, R. Bhushan Gopaluni *et al.*, "Data-driven capacity estimation of commercial lithium-ion batteries from voltage relaxation," *Nature Communications*, 2022. doi:10.1038/s41467-022-29837-w. Article licence CC BY 4.0.
- **Data:** Zenodo doi:[10.5281/zenodo.6405084](https://doi.org/10.5281/zenodo.6405084), published 2022-04-01. **Licence CC BY 4.0** (Zenodo record).
- **Content:** 130 commercial cells in three datasets (article abstract): NCA (66 files), NCM (55), NCM+NCA blend (9).
- **Files (MD5, Zenodo):** `Dataset_1_NCA_battery.zip` `cbd6d7eab2a296b52385e025e8d0f42a`; `Dataset_2_NCM_battery.zip` `a05c485df92f4af560d1d8d7f659ce1b`; `Dataset_3_NCM_NCA_battery.zip` `d11e68e410a638058906af5e2f5f60f3`; `Impedance raw data and fitting data.zip` `be9a8088847edd6528ce888372b42465`. 1.67 GB in total.

**Downloaded and verified (3 Oct 2026).** `scripts/fetch_pool.py` fetched every file above with resumable range requests, refused any file whose MD5 differs from the value registered here, and wrote the SHA-256 of each verified file to `docs/external_data/metadata/pool_downloads_manifest.json`. All six files matched. Files are in `data/raw/pool/` (git-ignored).

**What the files contain (read 3 Oct 2026):**
- **S7** is a BagIt package; all 14 payload files match its internal `manifest-md5.txt`. Its `Readme v2.pdf` states that the experiment ran from 12 Oct 2022, with regular check-ups CU 1–28 (CU 28 starting 17 Apr 2024). A thermal-management failure on 27 May 2024 ended the experiment, and a manual CU 29 was run at room temperature with air cooling only (to be excluded, like RPT 2 of S1). `cell_eocv2.zip` holds 240 per-cell result files: the 228 cells plus 12 records (`P000_0_S20_*`) from a reversible-loss side experiment in which one cell per calendar condition was held at 10 % SoC for about 2.5 days in January 2024 (Readme v2, Table 1).
- **S8** holds one `.mat` file per cell with per-cycle time series (voltage, current, capacity, power, temperature) and a per-cycle summary (charge and discharge capacity, energy, median and mean voltages), plus the authors' data description (English and Chinese PDFs).
- **S9** holds one CSV per cell with the raw time series (time, voltage, current, charge and discharge capacity, cycle number). File names encode the test temperature (`CY25`, `CY35`, `CY45`) and the current settings.

The mapping of each dataset to the forecasting task (what counts as a capacity check, which temperature, which cycles) is fixed in D21 before any model is trained on these data.

## Candidates screened, not used

| ID | Dataset | Result |
|---|---|---|
| C1 | NASA Ames (PCoE) battery ageing data | Not screened; first life only |
| C2 | Oxford battery degradation data | Not screened; first life only |
| C3 | CALCE battery data (University of Maryland) | Not screened; first life only |
| C4 | MIT–Stanford–Toyota fast-charging data (Severson et al., 2019) | Not screened; first life only, no periodic RPT |
| C5 | M. Beatty, D. Strickland, J. Warren, J. Chan, P. Ferreira, "Long-Term Sweat Testing Dataset for Second-Life Batteries," *Sci. Data* 12, 1068 (2025), doi:10.1038/s41597-025-05360-7; data doi:10.17028/rd.lboro.28732490.v2 (CC BY 4.0) | **Rejected:** capacity measured only at start and end of test (periodic checks failed), so there is no SOH series to forecast; about 34 GB |
| C6 | S. T. Mowri, A. Barai, M. Sheikh, S. Moharana, J. Marco, "Dataset for assessing the impact of first-life lithium-ion battery degradation on second-life performance," *Data in Brief* 63, 112208 (2025), doi:10.1016/j.dib.2025.112208 (Mendeley Data) | **Rejected for now:** second life covers only 7 monthly cycles; data licence not yet verified |
| C7 | K. A. Severson *et al.*, *Nature Energy* 4, 383 (2019), doi:10.1038/s41560-019-0356-8; P. M. Attia *et al.*, *Nature* 578, 397 (2020), doi:10.1038/s41586-020-1994-5; data at data.matr.io | **Not usable yet (3 Oct 2026):** articles verified on Crossref, but the data portal is a JavaScript application that exposes no licence in a retrievable record, and Crossref lists only the publisher's text-and-data-mining licence for the articles |
| C8 | Y. Preger *et al.*, *J. Electrochem. Soc.* 167, 120532 (2020), doi:10.1149/1945-7111/abae37 (Sandia); data at batteryarchive.org | **Not usable yet (3 Oct 2026):** article verified (CC BY-NC-ND 4.0), but no data licence found on batteryarchive.org |
| — | Zenodo record 6379165 (listed by a review as a Tongji dataset) | Not a valid record (the API returns no record); S9 is the verified Tongji record |

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
- [x] **Handover version:** version 2, settled by the folders and dates it contains (S1, "Evidence").
- [x] **Capacity labels:** reproduce `Capacity_Ah` from the raw RPT files. Done: 112/112 RPT capacity rows exact, i.e. 111 RPTs (S1, "Evidence"). The handover capacity includes the pulses; the paper defines capacity as the C/20 discharge capacity with the pulses excluded (D17).
- [x] **Cui et al. 2024:** a different dataset (8 retired Nissan Leaf LMO cells); cite it as related work (S5).
- [ ] **Stanford–Relyion data (S5):** request permission to reuse; no licence is declared.
- [ ] **Second dataset (S6):** cite both the article and the dataset DOI; state that "second life" there means ageing below 80 % SOH under the same protocol, at 50 °C.
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
| `cui2024_arxiv.atom.xml` | arXiv API record, S5 article | `9ce11d248c80a4c46043d10ca76898492602ce3bf9584cae8787fde1daa95c77` |
| `cui2024_relyion_osf.datacite.json` | DataCite, S5 dataset (no licence) | `960b92755252082af4278568d03eab11351cf36e236137d6b375085f0497a1c8` |
| `cui2024_relyion_osf.osf_node.json` | OSF API, project `fns57` (no licence) | `1635a63226193c3fe4930b22c624ea48d95b33e006db5e07ef47ccb782140993` |
| `cui2024_relyion_osf.files.json` | OSF API, top-level file list | `e694ca8bdf6741ca942c898ecc49002c5317b1ae2e2acefd87042d837f779bf2` |
| `wheeler2025_first_second_life_sd.crossref.json` | Crossref, S6 article | `6b312d5b98833ef58e974a35e102a98de71f94bc421818d730421b529ec5fdc6` |
| `wheeler2025_first_second_life_sd.fulltext.xml` | Europe PMC full text, S6 article (CC BY 4.0) | `f164ff7efce7fcf1fc63425d1c3ddb1f87068a7edf604f48153fe51d92c4cbbf` |
| `wheeler2025_lfp_dataverse.datacite.json` | DataCite, S6 dataset | `84de95a91194555d907731a74330214ba35a806635e8526dcd1aed80a8a02ca0` |
| `wheeler2025_lfp_dataverse.dataset.json` | Dataverse API, S6 dataset (version, licence, files, MD5) | `0ad5ef469fdc6301ea0283a613fc5ae0d62f47dd89d4cb79992008a695ad49c4` |
| `wheeler2025_characterisation_manifest.json` | Checksums of the 1,007 fetched S6 characterisation files | regenerated by `scripts/fetch_wheeler_characterisation.py` |
| `luh2024_kit_result_data.datacite.json` | DataCite, S7 result data (10.35097/1969) | `4d903b8001e0a0cd40086dc884c1227fa8561f40668a106122cea3654563854a` |
| `luh2024_kit_dataset.datacite.json` | DataCite, S7 parent dataset (10.35097/1947) | `18228e21a6850eeec913e24fbf3cd755cf86b183c0b2ca5c21f8eda95a43ec39` |
| `luh2024_kit_result_data.landing.txt` | RADAR landing page excerpt: archive size, MD5, download URL (S7) | `a183016c66a6b71ac139468bc5374d07cf174250cf437f46abacfbed7798363b` |
| `luh2024_sd.crossref.json` | Crossref, S7 article | `8839c09f46386601bd8a30daec086588d94e13f3211a132af9e3a96c6ea026cd` |
| `wang2024_xjtu_zenodo.record.json` | Zenodo API, S8 record (licence, files, MD5) | `d35058254ce835727b7f8b0328a11f220c150b0c6598aed1c392a365b339c062` |
| `wang2024_nc.crossref.json` | Crossref, S8 article | `36d48918b5bef29d6b26485f731e7237e4616b03927923c5c026f33934fd3186` |
| `zhu2022_tju_zenodo.record.json` | Zenodo API, S9 record (licence, files, MD5) | `0c70dd5bb07e1bcb1215d81f5543af50e018cb1135763cf0f249ee6073d76526` |
| `zhu2022_nc.crossref.json` | Crossref, S9 article | `da1ee88c7d802ab486e0b05f755395ee54ee029e1f91c0ee9146b9cc92d5e91c` |

The remaining references in `paper/references.bib` that have DOIs (TFT, EWC, LwF, LSTM, Wilcoxon, Cohen, Kerby) were also checked against Crossref on the same date. Entries without DOIs still need checking against the publisher.

## 6. Attribution obligations

| Where | Required text or action |
|---|---|
| Paper | Cite S1 article + dataset DOI. State the licence (CC BY 4.0) and that the data were processed (subset, derived features, labels) |
| Repository `README.md` | Data-source paragraph citing S1 and the licence (done) |
| Hugging Face mirror | Dataset card as in S2 (**open**) |
| Any redistributed derived file | Same attribution, licence link and change notice |
