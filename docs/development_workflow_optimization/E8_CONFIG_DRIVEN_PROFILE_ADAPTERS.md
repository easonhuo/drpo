# E8 Config-Driven Profile Adapters

## Status

- claim: `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`
- scope: workflow engineering only
- scientific execution: not authorized
- completed-result changes: forbidden

## Objective

Make reviewed YAML the sole source for concrete E8 sweep points, seed offsets,
point count, cell count, and matrix digest across the existing paper-aligned scan
runtime. Python continues to own scientific semantics, loss implementations,
method-family validation, trainer behavior, evaluation boundaries, and provenance.

## Supported adapter families

The runtime must support the current E8 parameter structures through thin adapters:

1. `asymre_scan`: explicit `delta_v` points crossed with paired seeds;
2. `legacy_exp`: explicit `(alpha, c)` points crossed with paired seeds, including
   Positive-only, Global, and exponential cells under the existing validator;
3. `reciprocal_screen`: explicit `(family, alpha, coefficient)` points crossed with
   paired seeds for reciprocal-linear and reciprocal-quadratic cells.

Adapters normalize configuration values and construct one runtime profile. They do
not duplicate or replace scientific validators.

## Default rule for future methods

A new E8 method with a genuinely new parameter structure must add one thin adapter
that maps its reviewed configuration into the common runtime-profile fields:

- `kind`;
- ordered `parameter_points`;
- ordered `seed_offsets`;
- `expected_points`;
- `expected_cells`;
- `matrix_digest`;
- any method-family metadata required by the existing `Cell` construction path.

A later experiment using an already-supported structure must not add another
adapter, Python parameter tuple, `_PROFILES` entry, expected-count constant, or
parameter-specific test. It should add only its reviewed config, scope, RunSpec,
and registration/closure material required by the existing governance path.

New scientific semantics, loss formulas, cell meanings, or frozen-variable rules
still require explicit Python implementation and validator review. The adapter is
not an escape hatch for changing scientific responsibilities.

## Frozen invariants

This migration must preserve, for every historical profile:

- point order and value identity;
- seed order and pairing;
- cell order, names, family, method, alpha, coefficient, `c`, and `delta_v`;
- point and cell counts;
- trainer, optimizer, scheduler, bank, horizon, evaluation, reporting, and delivery;
- historical result identity and completed evidence;
- separate reporting of task performance, structure/support boundaries, and NaN/Inf;
- liveness and terminal-audit requirements.

## Acceptance gate

Before activation, replay all existing E8 configurations through both the frozen
historical profile definitions and the config-driven adapters. Equality is required
for normalized points and cells. The replay must cover:

- paper-aligned EXP round 1;
- EXP `c` extension;
- reciprocal shape screen;
- reciprocal high-lambda extension;
- reciprocal-quadratic dense curve;
- AsymRE scan and AsymRE boundary-dense scan.

A new candidate config for each supported structure must also expand without a new
Python profile. Static checks, focused replay, full pytest, Ruff, and exact-head
repository gates must pass. CUDA liveness and scientific sweeps remain separate and
must not be represented as completed by this workflow change.
