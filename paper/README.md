# Paper

IEEE conference paper (`IEEEtran`, 7 pages including references). The team's draft (title, authors, headings, introduction, related work) is merged with the methods, results and discussion written from the implementation and the committed results. The consolidated results are in `docs/RESULTS.md`.

**Quick look:** `paper_single.pdf` is the compiled paper with all figures and tables, and `paper_single.tex` is the same paper as one LaTeX file (it needs only `references.bib` and `figures/` beside it). Both are generated; after editing the sources, refresh them from the repository root:

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
- `tables/*.tex` and the notebook figures are **generated** by `scripts/make_results_tables.py`; the `d21_*` figures by `scripts/make_d21_figures.py`; `figures/framework.png` by `scripts/make_framework_figure.py`. Do not edit them by hand. `tables/hyper.tex`, `horizon.tex` and `tests.tex` are generated but not used in the short version.
- `references.bib`: entries with a DOI were checked against Crossref. For refs 3, 6, 7 and 9 no open abstract could be read, so the related work states only what their titles state; extend those descriptions only after reading the full texts.
- `\todo{...}` marks what the team still has to decide: repository visibility and code licence. The submitted version must contain no `\todo`.

Build outputs (aux files, and PDFs other than `paper_single.pdf`) are not committed.
