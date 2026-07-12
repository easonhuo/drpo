# Pipeline handoff documents

This directory contains design, implementation, and closure handoffs for repository tooling. These files are not research masters and do not override `AGENTS.md`, `docs/handoff.md`, `experiments/registry.yaml`, or `docs/governance_pipeline_stage_status.yaml`.

## Default repository route

The connected GitHub App route remains the normal publication path: current `main` → dedicated dev branch → Draft PR → applicable GitHub Actions → review → explicit approved merge. A local shell clone is optional; shell DNS or clone failure does not activate the offline package fallback while the GitHub App can perform the required repository operations.

`drpo-update` and bundle-backed ZIP delivery remain preserved only as an explicitly activated emergency/offline fallback or when the user specifically requests an offline package.

## Dev branch → main integration tooling

- `DEV_BRANCH_INTEGRATION_PIPELINE_BUILD_BRIEF.md`: long-term blueprint, historical E8 integration failure analysis, and full capability roadmap.
- `DEV_BRANCH_INTEGRATION_PIPELINE_V1_SPEC.md`: accepted local transaction-tool contract.
- `DEV_BRANCH_INTEGRATION_PIPELINE_V1_CLOSURE.md`: implementation lineage, real-shadow evidence, operational adoption boundary, gate policy, exclusions, and rollback.
- `../scopes/GOV-DEV-BRANCH-INTEGRATION-01.md`: historical user-approved V1 implementation scope, exclusions, acceptance criteria, and rollback plan.
- `../scopes/GOV-DEV-BRANCH-INTEGRATION-01-BATCH3-NOTE.md`: exact two-shadow and rollback acceptance contract.
- `../scopes/GOV-DIRECT-GITHUB-CUTOVER-01.md`: connected-GitHub publication route and emergency-fallback boundary.

The V1 transaction tool is the normal integration route for long-lived, stale-base, scientific, registry/handoff, provenance-sensitive, or selectively imported dev snapshots. Fresh-main isolated low-risk changes may still use a direct reviewed GitHub App PR. V1 intentionally stops at a local ready commit; direct GitHub App publication remains a reviewer-controlled action under `AGENTS.md`. Autonomous in-repository push, PR creation, CI polling, or merge is not authorized by V1 acceptance.
