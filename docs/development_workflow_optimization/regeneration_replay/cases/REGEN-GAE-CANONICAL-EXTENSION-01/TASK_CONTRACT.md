# REGEN-GAE-CANONICAL-EXTENSION-01 — Generator Task Contract

Task packet status: **frozen before Arm A or Arm B generation**.

Historical repository base:
`aedbe8cc7e50f730cb6fcde90ca1816b97d79173`

Target experiment identity:
`EXT-H-E7-SQEXP-GAE-01`

## Objective

Implement a code-first development pilot that compares matched one-step TD
advantages with behavior-trajectory GAE under a shared frozen critic, using the
repository's existing E7 scientific and execution infrastructure.

The implementation must be executable and testable, but it does not register or
claim a scientific result. No actor branch or formal sweep may be described as
completed unless it actually runs.

## Frozen scientific matrix

- datasets:
  - `hopper-medium-expert-v2`;
  - `walker2d-medium-v2`;
  - `walker2d-medium-replay-v2`;
- development seeds: `200,201,202,203`;
- held-out seeds: `204,205,206,207`, untouched;
- advantage estimators:
  - one-step TD;
  - behavior-trajectory GAE with `gamma=0.99`, `lambda=0.95`;
- actor update modes:
  - canonical A2C;
  - PPO clip with epsilon `0.2` and old-policy cadence `K=4`;
- negative controls:
  - Positive-only;
  - squared-remoteness EXP `c=64,128,256`;
- one shared frozen critic per dataset/seed;
- critic training horizon: `100000` updates;
- actor horizon: `1000000` updates;
- total branches: `192`;
- fixed horizon is not convergence or steady-state evidence.

No dataset, seed, coefficient, critic budget, actor horizon, learning rate,
expectile, gamma, lambda, PPO setting, evaluation horizon, terminal/timeout/tail
rule, or reporting responsibility may be changed.

## Advantage contract

For each transition, one-step TD is computed from the shared critic. GAE is
accumulated over the ordered behavior trajectory.

Boundary semantics:

- true terminal: no bootstrap and stop recursive carry;
- timeout: bootstrap and stop recursive carry;
- final stored nonterminal row: bootstrap and stop recursive carry because no
  following row exists;
- terminal and timeout flags may not overlap;
- `lambda=0` must equal one-step TD;
- no advantage normalization or clipping.

Prepared actor-facing advantages remain float32. Numerical validation must not
confuse intended float32 storage quantization with a disagreement between
independent implementations.

## Shared-critic contract

For each dataset/seed pair:

- train or load exactly one shared critic artifact under the frozen budget;
- bind the dataset, critic, and advantage artifact identities;
- every matching TD/GAE, A2C/PPO, and control branch uses that same critic;
- the critic is frozen throughout actor training;
- actor and critic parameters are disjoint;
- terminal evidence verifies critic immutability.

## Repository integration contract

Before coding, inspect the existing E7 planning, branch lifecycle, trainer,
A2C/PPO, taper, evaluation, checkpoint, resume, aggregation, RunSpec, lane, and
artifact-delivery capabilities available at the historical base.

Extend existing responsibilities where compatible. A new production module is
allowed only when its responsibility is not already owned by an existing module
and the implementation records the closest inspected predecessor.

Do not introduce an unrelated generic training framework, duplicate execution
lane, duplicate scheduler, or duplicate publication path.

## Required deliverables

- frozen experiment configuration or equivalent protocol input;
- GAE computation and boundary validation;
- shared-critic/prepared-advantage identity checks;
- adapter into both canonical actor modes;
- exact 192-branch expansion;
- resumable plan/run entrypoint using repository conventions;
- terminal engineering audit;
- focused tests;
- one existing-schema RunSpec or equivalent repository-authorized launch record;
- one operator entrypoint;
- concise scope/status documentation.

## Required reporting boundaries

Separately report:

1. task-performance degradation or collapse;
2. support or variance-boundary events;
3. NaN/Inf numerical failure;
4. terminal and late-window behavior;
5. paired GAE-minus-TD comparisons without failed-cell imputation.

This implementation cannot establish universal GAE superiority, universal A2C
or PPO superiority, convergence, steady state, controlled causal identification,
or OOD generalization. Hopper remains external-validity evidence.

## Validation available to the generator

The generator may run ordinary repository inspection, focused tests, Python
compilation, shell syntax checks, full pytest, and Ruff when available.

The generator is not given the hidden evaluator implementation or any later
historical implementation, review, or runtime bug explanation.

## Terminal submission

Submit a complete repository patch/tree with:

- changed-file summary;
- production and test code-size accounting;
- reused modules and symbols;
- tests actually run and their results;
- tests or liveness not run;
- remaining uncertainty.

A smoke test, static check, plan expansion, or finite short pilot is not a formal
scientific result.
