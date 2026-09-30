# Paper skeleton

The manuscript prose (abstract, introduction, related work, discussion, conclusion) is written by the team. This directory holds only the parts that come straight from the implementation and the results: the methods record, the results section, the generated tables and figures, and the verified references. The consolidated results are in `docs/RESULTS.md`.

IEEE conference format (`IEEEtran`). Build from this directory:

```bash
latexmk -pdf main.tex      # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

- `sections/methods.tex`: methodology, written from the implementation and `docs/DECISIONS.md` (in particular D11, D12, D14, D16–D20).
- `sections/results.tex`: results. Every number in the text comes from committed results files produced on clean commits (`results/diagnostic_features.json`, `results/forecasting.json`, `results/forecasting_wheeler.json`).
- `tables/*.tex` and `figures/*.png` are **generated** (together with `results/tables/*.csv` and the tables in `docs/RESULTS.md`); do not edit them by hand. Regenerate them after re-running notebooks 11–13:

  ```bash
  python scripts/make_results_tables.py    # from the repository root
  ```

  Each table records the commit of the results file it was built from. The script refuses results written from a dirty working tree.
- `references.bib`: entries with a DOI were checked against Crossref. The literature entries at the end are generated from the Crossref records saved in `docs/literature/crossref/`. Data sources are documented in `docs/EXTERNAL_DATA.md`.
- `\todo{...}` marks what the team still has to write or decide: the abstract and remaining prose sections, affiliations and author order, and repository visibility and code licence. The submitted version must contain no `\todo`.

Build outputs (PDF, aux files) are not committed.
