# CLAUDE.md

@AGENTS.md

## Project context

This is not only a course project. The work is intended to become a **published IEEE paper**. Treat every result, figure and claim as something that will be peer-reviewed:

- **Reproducibility.** Fix random seeds, record configuration and library versions with results, and make every reported number traceable to code in this repository and a specific commit.
- **Data integrity.** Never fabricate, impute silently or substitute data. Synthetic data may be used only to test code paths, must stay outside the repository, and must never feed reported results.
- **Evaluation rigour.** Keep battery groups separate, respect temporal ordering, avoid target leakage, report per-group metrics, and compare against simple baselines. State limitations and unresolved assumptions explicitly.
- **Honest claims.** Describe SOH as relative to the first observed RPT capacity, not nominal capacity. Do not compare directly to literature numbers obtained on different datasets or SOH definitions without saying so.
- **Publication-ready outputs.** Figures should be clear and legible in print (labelled axes with units, legends, readable fonts); tables and metrics should be exportable for the manuscript.

## Commits and pull requests

- **Never mention Claude, Anthropic or AI assistance** in commit messages, pull request titles or descriptions, code comments, or any other content pushed to the repository.
- Do **not** add `Co-Authored-By` trailers, session links, "Generated with" footers or similar attribution lines.
