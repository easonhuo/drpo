# Reviewer / Gatekeeper Role

This document defines the independent review and merge-gating role for DRPO code
and experiment changes. It is an engineering workflow note, not a second
research master document.

## Responsibilities

The reviewer/gatekeeper is responsible for:

- preparing or approving the scope contract before implementation;
- checking the dev branch base, head commit, and provenance;
- reviewing the diff for unauthorized scientific-variable changes;
- verifying tests, liveness gates, run logs, and result/failure bundles;
- separating raw-complete, terminal-audited, packaged, delivered, and
  repository-applied states;
- deciding `merge_ready`, `request_changes`, or `reject`;
- interpreting scientific evidence only after result provenance and terminal
  audits are sufficient.

## Review order

Review must proceed in this order:

1. Confirm repository, branch, base commit, dev branch `HEAD`, experiment ID or
   claim, and remaining uncertainties.
2. Compare the actual diff with the approved scope contract.
3. Check that forbidden variables were not changed.
4. Check tests and liveness gates.
5. Check experiment result provenance and commit equality.
6. Check failure inventory and terminal-audit classifications.
7. Only then discuss method performance, rankings, or paper-facing conclusions.

## Merge gates

A dev branch may be merged only when all applicable gates pass:

- scope contract exists and matches the diff;
- no unauthorized changes to scientific variables, seeds, budgets, thresholds,
  datasets, formulas, or terminal criteria;
- tests and liveness gates ran on the reviewed commit;
- experiment outputs are tied to the reviewed commit;
- failed runs are preserved and classified;
- documentation and package artifacts are internally consistent;
- `main` has not advanced since the last rebase/merge test, or the branch has
  been refreshed and retested.

If the reviewer lacks repository write access, the reviewer must not claim that
`main` was updated. The reviewer may provide a merge/reject decision, commands,
or a canonical bundle-backed update package.

## Fail-closed defaults

Do not merge when any of the following is unresolved:

- the result was run from a dirty or different commit without a recorded launch
  snapshot and explicit pilot classification;
- the implementation agent self-approved its own changes;
- liveness evidence is missing for a large sweep;
- a raw-complete bundle is presented as a formal completed result;
- task-performance collapse, support/variance boundary events, and NaN/Inf
  numerical failures are conflated;
- the dev branch modified `docs/handoff.md` or `experiments/registry.yaml`
  outside the registered handoff-delta workflow.
