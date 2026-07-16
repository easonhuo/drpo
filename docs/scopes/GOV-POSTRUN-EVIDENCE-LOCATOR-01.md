# GOV-POSTRUN-EVIDENCE-LOCATOR-01

## Problem

The results-repository delivery path returns an immutable run ID, source commit, results commit, result path, and manifest SHA-256, but post-run registration does not yet have a uniform fail-closed locator contract. A compact handoff conclusion can therefore become detached from the detailed result evidence.

## Authorized scope

- define one registry `evidence_locator` schema for canonical `drpo-results` deliveries;
- add a transition-aware validator and a blocking pull-request workflow;
- grandfather untouched historical entries without destructive rewriting;
- require changed delivered entries to carry a complete locator;
- make locator records append-only and immutable;
- add documentation, template, and focused tests.

## Explicitly unchanged

- no experiment is run;
- no scientific status, conclusion, method ranking, seed, threshold, data size, horizon, or convergence criterion changes;
- no `docs/handoff.md` or `experiments/registry.yaml` after-image is modified in this implementation PR;
- no result is fabricated or moved;
- no change to RunSpec execution, packaging, delivery, registry rendering, or handoff authority responsibilities;
- no automatic result interpretation or registration.

## Change classification

Maintenance bugfix. It closes a discoverability/provenance gap under the existing requirements for durable artifacts, authoritative commit resolution, and chat-independent evidence. It does not introduce another authority or transaction.

## Acceptance

- valid canonical results-repository locators pass;
- changed delivered entries without locators fail;
- malformed repository, branch, commit, path, or manifest fields fail;
- existing locator records cannot be changed, reordered, or removed;
- reruns append a new record and may update the primary run ID;
- untouched historical delivered entries remain grandfathered;
- full repository tests and Ruff pass on the exact PR head.

## Rollback

Revert the workflow, validator, documentation, template, and tests as one change. Preserve any locator data later added to the registry because it is immutable provenance. The existing results delivery and Stage 5 authority remain unchanged.
