# Pipeline handoff documents

This directory contains design and implementation handoffs for repository tooling. These files are not research masters and do not override `AGENTS.md`, `docs/handoff.md`, `experiments/registry.yaml`, or `docs/governance_pipeline_stage_status.yaml`.

## Dev branch → main integration

- `DEV_BRANCH_INTEGRATION_PIPELINE_BUILD_BRIEF.md`: long-term blueprint, historical E8 integration failure analysis, and full capability roadmap.
- `DEV_BRANCH_INTEGRATION_PIPELINE_V1_SPEC.md`: current lightweight V1 implementation contract. Use this file for V1 scope, state machine, non-goals, architecture, acceptance, and rollout.
- `../scopes/GOV-DEV-BRANCH-INTEGRATION-01.md`: user-approved claim, implementation scope, exclusions, acceptance criteria, and rollback plan.

The V1 spec intentionally stops at a local ready commit. Automatic push, PR creation, CI polling, merge, and default-route changes require separate approval after shadow evidence.
