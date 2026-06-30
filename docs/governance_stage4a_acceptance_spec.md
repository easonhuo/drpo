# Stage 4A Final Acceptance Specification

**Policy / claim:** `GOV-HANDOFF-INDEX-01`
**Evaluated repository base:** `9674cb167080dfdeecb353c9f328ad86b74f87c5`
**Authority:** machine-verifiable shadow acceptance only. `docs/handoff.md` and `experiments/registry.yaml` remain authoritative.

## 1. Scope

This acceptance closes the Stage 4A implementation composed of:

1. the static Stage 4A schema and inventories;
2. the dynamic semantic shadow graph;
3. the deterministic minimal-context module builder and semantic contracts.

It does **not** build the Stage 4B lossless physical split, activate Stage 4B, validate Stage 4C real context assembly, or authorize any authority cutover.

## 2. Hard acceptance gates

Stage 4A passes only when all of the following are true:

- all canonical experiments and development registrations are mapped; no unmapped-object suggestion remains;
- `global_core_governance`, `execution_status_gates`, and `terminal_audit` satisfy their exact source-scoped semantic contracts;
- all six registered context-closure targets pass their include/exclude boundaries;
- the static inventory and dynamic semantic graph are current, deterministic, cycle-free, and have no review-queue or rejected candidates;
- source extraction has no ambiguous partial overlap, duplicate marker ambiguity, stale marker, or missing source;
- repeated builds are byte-identical, and the second no-op build reuses every module;
- context building does not modify `AGENTS.md`, `docs/handoff.md`, or `experiments/registry.yaml`;
- generated outputs are stale/tamper detectable;
- every registered fault-injection case behaves fail-closed and provides an actionable diagnostic;
- the Stage 4 tests, governance validators, Ruff, compilation checks, and full repository test suite all pass.

Any failed hard gate leaves Stage 4A unaccepted and Stage 4B blocked.

## 3. Required semantic contracts

### `global_core_governance`

- unique master document;
- document before experiment;
- non-destructive history;
- terminal-state audit governance;
- controlled-mechanism versus external-validity boundary.

### `execution_status_gates`

- formal evidence separated from development/smoke evidence;
- one registered execution order;
- blocked states preserved until their predecessor or protocol gate is met;
- current formal route preserved;
- no unregistered experiment.

### `terminal_audit`

- convergence versus persistent drift/runaway;
- at least `2x` continuation;
- false-plateau checks;
- task-performance collapse;
- support/variance boundary;
- NaN/Inf numerical failure;
- separate reporting of the three failure classes.

A module name, dependency edge, or accidental keyword elsewhere does not satisfy a contract. Every topic requires source-scoped evidence in that module's extracted authoritative content.

## 4. Context-closure targets

The registered targets are:

- `continuous_e4_extrapolation`;
- `continuous_e4_taper`;
- `categorical_e6_generalization`;
- `hopper_e7`;
- `countdown_e8`;
- `paper_rewrite`.

Each target must contain every required dependency and exclude every prohibited responsibility recorded in `DEPENDENCIES.yaml`.

## 5. Fault-injection standard

The acceptance runner must cover at least:

- missing semantic-contract topic;
- source-scoped evidence mismatch;
- authority promotion or automatic structure policy;
- unknown, self, or cyclic dependency;
- Hopper/Countdown closure leakage;
- unmapped canonical, formal-development, and pilot-development registrations;
- stale registry mapping;
- partial source overlap;
- duplicate marker block;
- tampered or missing generated output;
- full-coverage source deduplication with provenance;
- authoritative-input isolation.

Negative cases must exit non-zero or raise the expected validation error. Positive deduplication and authority-isolation controls must pass explicitly.

## 6. Evidence and after-image

The canonical acceptance command is:

```bash
python3 scripts/run_stage4a_acceptance.py \
  --repo-root . \
  --gate-evidence /path/to/gate_evidence.json \
  --write
```

It writes:

- `ACCEPTANCE_REPORT.json`;
- `ACCEPTANCE_SUMMARY.md`;
- `AFTER_IMAGE.json`;
- `FAULT_INJECTION_REPORT.json`;
- `CHECKSUMS.sha256`.

`AFTER_IMAGE.json` freezes the accepted Stage 4A implementation. Later modifications require an explicit bugfix, compatibility, or clarification authorization, or a formal reopen. Acceptance evidence files themselves are excluded from the after-image to avoid circular hashing.

## 7. Stage transition on PASS

A PASS changes only the governance ledger:

- `stage_4a_schema_inventory: accepted`;
- `stage_4b_lossless_candidate: ready_for_authorization`;
- `stage_4c_context_assembly_shadow_validation: blocked_by_stage_4b_acceptance`.

Stage 4B remains inactive until a separate user-approved authorization. Stage 5 and authority cutover remain blocked.

## 8. Non-blocking advisories

Module-length or future-granularity suggestions may remain in the report. They cannot override a semantic failure and do not block acceptance when all hard gates pass.
