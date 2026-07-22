> [!WARNING]
> **SUSPENDED ON 2026-07-22.** The config-driven E8 runtime path is not active on
> `main`. The repository uses the preceding fixed-profile/fixed-matrix execution
> path until a separately reviewed real CUDA liveness and end-to-end training
> validation explicitly authorizes reactivation. This document is retained as
> historical design evidence and is not execution authority.

# E8 Config-Driven Adapter Contract

## Status

- parent claim: `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`
- implementation scope: `GOV-E8-CONFIG-DRIVEN-ADAPTERS-01`
- environment role: Countdown external validity only
- scientific execution: not authorized by this contract
- default applies to: the existing paper-aligned E8 scan runtime

## Design rule

Concrete parameter values, seed offsets, declared point counts, declared cell
counts, and the deterministic matrix digest come from the reviewed grid config.
Python retains scientific meanings, loss implementation, method-family validation,
identity binding, execution gates, aggregation, and reporting boundaries.

Adapters are defined per parameter structure, not per experiment ID. The current
runtime supports:

1. `legacy_exp`: explicit `(alpha, c)` points crossed with paired seeds. This
   includes Positive-only, Global, and continuous EXP cells according to the
   existing `Cell.method` semantics.
2. `reciprocal_screen`: explicit `(family, alpha, coefficient)` points crossed
   with paired seeds, where family is Reciprocal-Linear or
   Reciprocal-Quadratic.
3. `asymre_scan`: explicit `delta_v` points crossed with paired seeds.

A new scan that uses one of these structures must not add a concrete parameter
tuple, expected-count constant, or experiment-specific `_PROFILES` entry to
Python. Its routine repository delta is config, scope, RunSpec, and the existing
registration/provenance artifacts required by governance.

## New-method default

A genuinely new method or parameter structure must include, in the same reviewed
development scope:

1. a thin structure adapter in the nearest existing runtime;
2. a method-specific scientific validator or validator branch;
3. deterministic cell identity and run-identity binding;
4. Replay A/B against the predecessor or another frozen semantic reference;
5. an unknown-experiment candidate proving that no concrete Python profile is
   required after the adapter exists;
6. a real runtime `plan` check and the normal liveness gate before any larger
   run.

Correctness equivalence is evaluated before workflow efficiency. Replay must
compare point order, seed order, cell order, names, method/family fields,
parameters, declared counts, and matrix digest. A faster path with different
launch semantics fails.

## Size boundary

A family parser should normally remain approximately 20--40 production lines.
Shared profile construction belongs in the existing common builder. An adapter
above 60 family-specific lines requires a scope review for duplicated validation
or hidden scientific semantics. The science-agnostic matrix helper must not
learn E8 loss, model, bank, evaluation, or method-ranking rules.

## Preserved boundaries

This contract does not:

- alter any completed E8 result or result status;
- authorize CUDA liveness or a scientific sweep;
- change the trainer, optimizer, bank, seeds, horizon, evaluation split, or
  reporting semantics;
- make Countdown a substitute for controlled C-U1 or D-U1 evidence;
- establish convergence, steady state, significance, or method ranking;
- change the global repository development route in `AGENTS.md`.

Task-performance degradation, valid-structure or support boundary events, and
NaN/Inf numerical failure remain separate.
