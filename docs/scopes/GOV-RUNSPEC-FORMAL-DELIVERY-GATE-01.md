# GOV-RUNSPEC-FORMAL-DELIVERY-GATE-01

## Status

- Change class: Stage 2 default-policy hardening.
- Governance authorization: explicit user approval in the 2026-07-19 project conversation.
- Base commit: `85b0a68d77ed085a7f6e67771fb0f7672c43da09`.
- Scientific experiment status: unchanged; this change does not launch or rerun an experiment.
- Target: every governed RunSpec that may produce formal evidence.

## Problem

The repository already contains the RunSpec packaging and `drpo-results` delivery
implementation, but a training entry can omit or disable `delivery` and still pass
validation. This permits a formal experiment to complete locally without durable
result delivery. Direct one-click training remains an internal experiment entrypoint;
it must not become a second formal execution channel.

## Authorized behavior

1. `policy.formal_evidence_allowed` remains the formal/non-formal declaration.
2. `false` is the only explicit local-only opt-out. It is reserved for liveness,
   smoke, engineering probes, and other runs that cannot form formal evidence.
3. `true` requires automatic results-repository delivery.
4. A missing declaration fails closed as delivery-required, so omission cannot
   silently downgrade a potentially formal run to local-only.
5. Delivery-required RunSpecs must use the existing contract:
   - `delivery.enabled: true`;
   - `delivery.auto: true`;
   - `delivery.mode: results_repo`;
   - `delivery.repository: easonhuo/drpo-results`;
   - lane-bound branch `ingest/<lane>`;
   - export profile `manifest_text_v1`.
6. The shared pre-entrypoint delivery-policy gate is called by static validation,
   safe lane claim, direct claimed execution, artifact packaging, and the canonical
   manual uploader. Invalid specifications remain unclaimed and no training starts.
7. Delivery failures retain the existing lifecycle semantics: training/package may
   pass, overall status is `PARTIAL`, local evidence is preserved, and only upload
   is retried. Training is not rerun.

## Compatibility boundary

- Historical/non-formal RunSpecs must declare
  `policy.formal_evidence_allowed: false` when local-only execution is intended.
- A pilot may still opt into durable delivery; `false` permits local-only but does
  not forbid delivery.
- No automatic upload logic is added to one-click experiment scripts. They remain
  internal entrypoints called by the governed RunSpec executor.
- No scientific variable, dataset, seed, threshold, horizon, method, result status,
  registry entry, or handoff conclusion changes.

## Acceptance

- missing formal declaration plus missing/disabled delivery is rejected;
- `formal_evidence_allowed: true` plus missing/disabled/manual delivery is rejected;
- `formal_evidence_allowed: false` permits explicit local-only execution;
- a delivery-required spec accepts only the canonical `drpo-results` repository and
  lane-bound branch;
- rejection occurs before claimed-state creation;
- existing append-only/idempotent delivery tests continue to pass;
- Python compilation, focused pytest, full pytest, Ruff, handoff authority, formal
  execution-channel validation, and governance-stage validation pass on the exact
  PR head.

## Rollback

Revert the shared formal-delivery check, tests, and documentation together. Preserve
all already-delivered result commits and local experiment evidence. Do not rerun or
delete any scientific experiment during rollback.

## Remaining uncertainties

- Historical templates not exercised by the current ready queue may rely on an
  omitted formal declaration to mean local-only; they will now fail closed until
  explicitly classified.
- Server credentials and scoped wrappers are environment concerns and are not
  changed by this repository update.
