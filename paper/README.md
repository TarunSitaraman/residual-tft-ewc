# Paper

IEEE conference paper (`IEEEtran`; 8 pages including references, cut to the usual limit on 19 Oct 2026 after the D25–D27 update). To make room, the framework, Spark-pipeline, P3, ablation and D23 per-horizon figures and the literature table were removed (the files stay in `figures/` and the notebooks), the D27 interval table became `figures/d27_coverage.png`, and the text was condensed. The 8 pages were checked with Tectonic (TeX Gyre Termes, metric-compatible with the Times of pdfLaTeX); the last page is full, so check the page count after any edit. The team's draft (title, authors, headings, introduction, related work) is merged with the methods, results and discussion written from the implementation and the committed results. The consolidated results are in `docs/RESULTS.md`.

**Quick look:** `paper_single.pdf` is the compiled paper with all figures and tables (8 pages; the 19 Oct build used Tectonic 0.15.0), and `paper_single.tex` is the same paper as one LaTeX file (it needs only `references.bib` and `figures/` beside it). Both are generated; after editing the sources, refresh them from the repository root:

```bash
python scripts/flatten_paper.py                       # rewrites paper_single.tex
cd paper && latexmk -pdf paper_single.tex             # or: tectonic paper_single.tex
python scripts/flatten_paper.py --check               # fails if paper_single.tex is stale
```

The longer 15-page version (full dataset descriptions, feature lists, all P1 tables and figures) is kept at the git tag `paper-long-draft`.

Build from this directory:

```bash
latexmk -pdf main.tex      # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
tectonic main.tex          # alternative; main.tex switches to TeX Gyre Termes under XeTeX
```

On Overleaf: upload the contents of this directory (main.tex, references.bib, sections/, tables/, figures/) and compile with pdfLaTeX.

- `sections/methods.tex`: methodology, written from the implementation and `docs/DECISIONS.md` (D11, D12, D14, D16–D21).
- `sections/results.tex`: results. Every number comes from committed results files produced on clean commits; the sources are listed at the top of the file.
- `tables/*.tex` and the notebook figures are **generated** by `scripts/make_results_tables.py`; the `d21_*` and `d23_*` figures by `scripts/make_d21_figures.py`; the `d25_*`, `d26_*` and `d27_*` figures by `scripts/make_d25_d27_figures.py`; `figures/framework.png` by `scripts/make_framework_figure.py`; `figures/d23_comparison.png` by `scripts/make_d23_bar_figure.py`; `figures/epoch_curves.png` by `scripts/run_epoch_curves.py`; `figures/preprocessing_s6.png` and `correlation_s6.png` by `scripts/make_preprocessing_figures.py`; `tables/d26_classes.tex` and `figures/d26_confusion.png` by `scripts/make_d26_class_outputs.py`. The system and model diagrams are TikZ sources (`figures/system_architecture.tex`, `figures/proposed_model.tex`). Do not edit them by hand. `tables/hyper.tex`, `horizon.tex` and `tests.tex` are generated but not used in the short version.
- `references.bib`: entries with a DOI were checked against Crossref. For refs 3, 6, 7 and 9 no open abstract could be read, so the related work states only what their titles state; extend those descriptions only after reading the full texts.
- `\todo{...}` marks open items; none remain (the repository is public and source-available, and the code-availability sentence in `sections/methods.tex` points to it). The submitted version must contain no `\todo`.

Build outputs (aux files, and PDFs other than `paper_single.pdf` and `supplement.pdf`) are not committed. Licence: see the root `LICENSE` (source-available, all rights reserved).
