# GOV-DIRECT-GITHUB-CUTOVER-01

## Decision

The connected GitHub App is the default repository-development transport for DRPO.

Normal route:

```text
current main
→ dev/<claim>
→ Draft PR
→ applicable GitHub Actions
→ reviewer audit
→ explicit user-approved merge
```

This supersedes the prior practice of treating bundle-backed `drpo-update` delivery as a parallel normal route.

## User approval

Approved in conversation on 2026-07-11 after the user explicitly requested retirement of the old default route and directed that defects be fixed in the new route rather than bypassed through the old one.

## Base

`a61e5b11af1e09ea880f1738991cb9e1b0e3ca1a`

## Required behavior

- Use the GitHub App whenever it can resolve `main`, write a dev branch, open a PR, inspect Actions, and merge after approval.
- Treat local clone/fetch as optional implementation convenience.
- Do not interpret shell DNS, `git clone`, or `git fetch` failure as loss of GitHub App write access.
- Do not create dummy write probes at every session start.
- Repair or iterate the direct route when it exposes a defect.
- Do not request a bundle while the GitHub App can complete the task.

## Emergency fallback

Bundle-backed `drpo-update` delivery remains available only when:

1. the GitHub App has been actually checked and lacks an operation required by the task; or
2. the user explicitly requests an offline update package.

When activated, the existing canonical package and verification rules remain unchanged.

## Exclusions

This cutover does not:

- change any scientific claim, experiment status, seed, threshold, budget, method, result, or execution order;
- change formal experiment source-provenance requirements;
- delete historical package tooling or provenance;
- authorize autonomous in-repository push, PR creation, CI polling, approval, or merge;
- modify any Stage 1, Stage 2, or Stage 5 protected implementation file.

## Rollback

Revert the `AGENTS.md` routing section, this scope record, and the pipeline README clarification together. Historical `drpo-update` tooling remains intact throughout.
