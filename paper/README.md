# Paper draft

IEEE conference-format manuscript (`IEEEtran`). Build from this directory:

```bash
latexmk -pdf main.tex      # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

- `sections/intro.tex`, `related.tex`, `discussion.tex`, `conclusion.tex`: text sections. Their numbers come from the same results files as the results section.
- `sections/methods.tex`: methodology, written from the implementation and `docs/DECISIONS.md` (in particular D11, D12, D14, D16–D20).
- `sections/results.tex`: results. Every number in the text comes from committed results files produced on clean commits (`results/diagnostic_features.json`, `results/forecasting.json`, `results/forecasting_wheeler.json`).
- `tables/*.tex` and `figures/*.png` are **generated**; do not edit them by hand. Regenerate them after re-running notebooks 11–13:

  ```bash
  python scripts/make_paper_tables.py    # from the repository root
  ```

  Each table records the commit of the results file it was built from. The script refuses results written from a dirty working tree.
- `references.bib`: entries with a DOI were checked against Crossref. The literature entries at the end are generated from the Crossref records saved in `docs/literature/crossref/`. Data sources are documented in `docs/EXTERNAL_DATA.md`.
- `\todo{...}` marks what still needs a team decision: affiliations and author order, and repository visibility and code licence. The submitted version must contain no `\todo`.

Build outputs (PDF, aux files) are not committed.
