# Pipeline handoff documents

This directory contains design and implementation handoffs for repository tooling. These files are not research masters and do not override `AGENTS.md`, `docs/handoff.md`, `experiments/registry.yaml`, or `docs/governance_pipeline_stage_status.yaml`.

## Default repository route

The connected GitHub App route is the normal development path: current `main` → dedicated dev branch → Draft PR → applicable GitHub Actions → review → explicit user-approved merge. A local shell clone is optional; shell DNS or clone failure does not activate the offline package fallback while the GitHub App can perform the required repository operations.

`drpo-update` and bundle-backed ZIP delivery remain preserved only as an explicitly activated emergency/offline fallback or when the user specifically requests an offline package.

## Dev branch → main integration tooling

- `DEV_BRANCH_INTEGRATION_PIPELINE_BUILD_BRIEF.md`: long-term blueprint, historical E8 integration failure analysis, and full capability roadmap.
- `DEV_BRANCH_INTEGRATION_PIPELINE_V1_SPEC.md`: optional local transaction-tool contract. It is not the default ChatGPT development route and does not block direct GitHub App development.
- `../scopes/GOV-DEV-BRANCH-INTEGRATION-01.md`: historical user-approved V1 implementation scope, exclusions, acceptance criteria, and rollback plan.
- `../scopes/GOV-DIRECT-GITHUB-CUTOVER-01.md`: superseding default-route cutover and emergency-fallback boundary.

The V1 tool intentionally stops at a local ready commit. Direct GitHub App publication is performed by the reviewing assistant under `AGENTS.md`; autonomous in-repository push, PR creation, CI polling, or merge remains separate future automation and is not implied by this cutover.
