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

## Branches

- Keep `main` as the single long-lived branch. Work branches (including those created by cloud sessions) are temporary.
- **Once a branch's work is in `main`, delete the branch** on the remote (and locally).
- **Delete stagnant branches** too: a cloud-session branch that is already contained in `main`, or whose work has been abandoned, should be removed. Before deleting a branch with commits not in `main`, confirm with the team that the work is not needed.
- **Cloud sessions cannot delete remote branches** (the session's git proxy rejects deletes with HTTP 403). A session should try the delete once; if it is refused, do not retry or work around it. Instead, at the end of the task, list each branch that is ready to delete, with its last commit and whether it is fully contained in `main` (`git merge-base --is-ancestor <branch> origin/main`), so a team member can remove it from the GitHub Branches page or with `git push origin --delete <branch>`.
- **Resolve conflicts when merging into `main`** rather than leaving a branch unmerged: merge `main` into the branch (or rebase an unshared branch), fix the conflicts, re-run the notebooks/checks affected, then merge. Never discard someone else's changes to get a clean merge.
