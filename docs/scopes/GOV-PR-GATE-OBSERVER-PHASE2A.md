# GOV-PR-GATE-OBSERVER-PHASE2A Scope and Rollback Plan

## Identity

- Claim: `GOV-PR-GATE-OBSERVER-PHASE2A`
- User request: connect the Phase-1 observer to pull requests as a non-blocking, read-only GitHub Actions workflow.
- Authoritative `main` at planning time: `30b6a9cfd9da2feca1e5be22f5c002d1459298ff`.
- Phase-1 dependency commit: `209caab5409173e9406cb70b62be3be9a4ac09c1`.
- Dev branch: `dev/gov-pr-gate-observer-phase2a`.
- Stacked PR base while Phase 1 remains unmerged: `dev/gov-pr-gate-observer-01`.
- Reviewer/gatekeeper: independent review required; the implementation agent must not self-merge.

## Classification

- Change class: **Red / control-plane**, explicitly authorized by the user on 2026-07-10.
- Risk level: medium and reversible.
- Rationale: `.github/workflows/**` changes when repository automation runs, but this phase grants no write capability and does not change branch protection or gate-selection policy.

## Allowed changes

- `docs/scopes/GOV-PR-GATE-OBSERVER-PHASE2A.md` — freeze scope, safety boundaries, and rollback.
- `.github/workflows/pr-gate-observer.yml` — run the existing Phase-1 observer on pull-request events.
- `tests/test_pr_gate_observer_workflow.py` — enforce the workflow trigger, permission, concurrency, artifact, and no-write contract.

## Forbidden changes

- Phase-1 observer behavior unless a separately reviewed compatibility defect is found.
- `tools/drpo-update/test_selection.py`, `tools/drpo-update/test_impact_map.json`, and `scripts/select_update_tests.py`.
- Branch protection, required-status-check configuration, repository settings, secrets, labels, comments, merge behavior, or scheduled execution.
- `AGENTS.md`, `docs/handoff.md`, `experiments/registry.yaml`, or `docs/handoff_deltas/**`.
- Scientific code, configs, datasets, seeds, budgets, optimizers, thresholds, formulas, convergence criteria, experiment order, or result status.

## Required workflow behavior

- Trigger only for `pull_request` actions `opened`, `synchronize`, and `reopened`.
- Use only `contents: read` permission.
- Check out full history so exact PR base/head SHAs are available to the observer.
- Cancel an older in-progress observer run when the same PR receives a newer commit.
- Execute the Phase-1 observer with exact event base/head SHAs.
- Publish `gate_summary.md` to the job summary when available.
- Upload `gate_report.json`, `gate_summary.md`, and `logs/` for 14 days.
- Preserve observe-only semantics: selected gate failures are recorded without changing the observer exit code; observer infrastructure failure may fail this non-required workflow.
- Do not configure or claim this workflow as a required branch-protection check.

## Validation plan

```bash
python -m pytest -q tests/test_pr_gate_observer.py tests/test_pr_gate_observer_workflow.py
ruff check scripts/run_pr_gate_observer.py tests/test_pr_gate_observer.py tests/test_pr_gate_observer_workflow.py
python scripts/handoff_authority.py verify --repo-root .
python scripts/validate_governance_pipeline_stage_status.py --repo-root .
python -m pytest -q
ruff check .
```

The first real GitHub Actions run must additionally confirm checkout, exact SHA resolution, artifact upload, job summary publication, and cancellation of a superseded run.

## Rollback

1. Disable or remove `.github/workflows/pr-gate-observer.yml` in a dedicated rollback PR.
2. Keep Phase-1 observer code and historical artifacts; they do not alter existing gate enforcement when not invoked.
3. Do not modify selector, impact map, branch protection, or scientific files as part of rollback.
4. Verify that no `PR Gate Observer` workflow run is triggered after rollback lands.

This rollback returns repository automation to the Phase-1 state without changing existing test gates or experiment behavior.

## Merge criteria

- Diff relative to the stacked base contains only the three allowed paths.
- Static workflow-contract tests pass.
- Full repository validators and tests pass, or any failure is independently classified and resolved before merge.
- A real PR-triggered run produces the expected summary and artifact without write access.
- Phase 1 is independently approved and merged before this PR is retargeted to and merged into `main`.
- Independent reviewer returns `merge_ready`.
