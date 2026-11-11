# Project Conventions

This repository contains the continual-learning workflow for SOH estimation and second-life suitability assessment of heterogeneous EV batteries.

## Conventions

- Keep datasets outside Git under `data/`.
- Preserve battery-group identity and temporal ordering in evaluation.
- Document missing-temperature handling; do not fabricate values silently.
- Keep generated checkpoints and experiment outputs out of version control.
- Synthetic data may only test code paths: it stays outside the repository and never feeds reported results.
- Report metrics per battery group, compare against simple baselines, and describe SOH as relative to the first observed capacity check, not nominal capacity.
- Forecasting work (D16–D20) uses the C/20 labels in `data/rpt_health_c20.csv` and the features built from the raw archives (`data/README.md`). `ml_training_dataset.csv` is only the input of the superseded current-SOH notebooks 01–09.

## Invariants

- `SOH_Percent = Capacity_Ah / Reference_Capacity_Ah * 100`, where `Capacity_Ah` is the C/20 discharge capacity of the RPT, pulses excluded (`data/rpt_health_c20.csv`, decision D17)
- A cycle uses only the most recent RPT on or before its cycle date.
- Continual-learning evaluation reports adaptation and retention.

<!-- coord:start -->
## Team coordination

This repo is set up for team coordination between agents working in shared
sessions. Follow these rules:

- Before starting non-trivial work, publish a plan with `publish_plan`.
- Keep step status current as you go, using `update_step`.
- If the injected context shows an overlap warning, coordinate before you
  edit (spec Law 3, Rev 28) — never silently duplicate the work. If it is the
  same task and you have not started, join their session: register on their
  branch, or offer `propose_work_request` — the server auto-accepts only
  when every step is provably unstarted and unassigned, so a safe join needs
  no human. If the work is related, claim only unstarted, unassigned steps
  the same way. Never take started or assigned work. Stop and ask your human
  only when your permission mode requires approval or no safe action exists.
- When you discover something non-obvious (a gotcha, a decision, a fact
  someone else would need), write it down with `remember` so the team
  keeps it.
- When something is a teammate’s job rather than yours — their area, their
  decision, a question only they can answer — hand it to them with
  `assign_work` instead of asking your human to relay a message. Use
  `list_people` to turn a name into an assignee id, put the whole brief
  in the note, and tell your human who you handed it to. Their own agent is
  told at its next prompt. This is free: no model call, no sponsor, nothing
  to pay for.
- When you hit a bug, a blocker, or something that plainly wants a different
  specialist, open it with `report_opportunity` and close it with
  `resolve_opportunity` when it is handled. Nothing reads your transcript
  looking for these, so a problem you only describe in your answer reaches
  nobody; reporting it is what lets the team's Brain offer it to whoever fits.
  Reporting is not notifying — it may reach no one — so do not tell your human
  a teammate was alerted.
<!-- coord:end -->
