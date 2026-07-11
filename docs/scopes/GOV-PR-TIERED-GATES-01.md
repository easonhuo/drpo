# GOV-PR-TIERED-GATES-01 Phase 1 scope and rollback

## Purpose

Introduce a read-only shadow plan for risk-tiered PR testing before any existing
gate is removed, weakened, or made conditional. The current full compile, shell,
handoff, formal-channel, governance, pytest, and Ruff steps remain unchanged.

This phase gathers real PR evidence for the existing deterministic selector. It
does not execute the selected fast plan and does not yet reduce CI coverage or
runtime.

## Base and claim

- Base commit: `75516c5ced6714b8e462fd1e2cf2b15c9292e9f3`
- Claim: `GOV-PR-TIERED-GATES-01`
- Phase: `phase_1_shadow_only`
- Research experiment impact: none

## Allowed changes

- add one shadow-only selector-plan step to `.github/workflows/pr-gate-log.yml`;
- fetch full Git history in the workflow so PR base and head SHAs are available;
- write the selector JSON to the native job log and step summary;
- add a static workflow contract test;
- add this scope/rollback record and its governance authorization.

## Explicitly excluded

- do not pass `--execute` to the selector;
- do not remove, skip, condition, or reorder the existing full gates;
- do not change `tools/drpo-update/test_selection.py`,
  `tools/drpo-update/test_impact_map.json`, or `scripts/select_update_tests.py`;
- do not change branch protection or required-check settings;
- do not change handoff, registry, experiment code, configs, scientific
  variables, seeds, thresholds, results, or execution order;
- do not start Phase 2 enforcement in this PR.

## Acceptance

The PR is acceptable only when:

1. the workflow remains read-only;
2. selector failure is observe-only and cannot suppress the legacy full gates;
3. the selector receives the exact PR base and head SHAs;
4. the selector runs without `--execute`;
5. the legacy full gate commands remain present;
6. the static workflow contract test, governance validators, full pytest, and
   Ruff pass on the GitHub PR merge ref.

## Rollback

Revert the squash-merge commit. The repository then returns to the previous
single-path full-gate workflow with no data migration or scientific rollback.

## Phase 2 entry condition

Phase 2 is not authorized by this change. It requires a separate user-approved
scope after several real PRs show that shadow fast selections do not miss
failures found by the retained full suite.
