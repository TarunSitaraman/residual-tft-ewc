# Paper draft

IEEE conference-format manuscript (`IEEEtran`). Build from this directory:

```bash
latexmk -pdf main.tex      # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

- `sections/methods.tex` — methodology, written from the implementation and `docs/DECISIONS.md` (D1–D13).
- `references.bib` — every entry must be checked against the publisher record before submission; entries marked `VERIFY` were not checked against the full text. The dataset entries are cited only if the data provenance is confirmed.
- `\todo{...}` marks values that depend on the real-data run or on a team decision. Fill them only from committed `results/*.json` produced on a clean commit (`git_dirty: false`). The submitted version must contain no `\todo`.

Build outputs (PDF, aux files) are not committed.
