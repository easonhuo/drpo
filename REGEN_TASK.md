# Frozen Regeneration Task Packet

This file is immutable benchmark input and is excluded from code-size scoring.
Do not inspect historical pull requests, later implementations, the benchmark
orchestrator branch, or any other regeneration run.

Historical repository base before this packet:
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

- datasets: `hopper-medium-expert-v2`, `walker2d-medium-v2`, and
  `walker2d-medium-replay-v2`;
- development seeds: `200,201,202,203`;
- held-out seeds: `204,205,206,207`, untouched;
- advantage estimators: one-step TD and behavior-trajectory GAE with
  `gamma=0.99`, `lambda=0.95`;
- actor update modes: canonical A2C and PPO clip with epsilon `0.2`, old-policy
  cadence `K=4`;
- controls: Positive-only and squared-remoteness EXP `c=64,128,256`;
- one shared frozen critic per dataset/seed;
- critic horizon: `100000` updates;
- actor horizon: `1000000` updates;
- total branches: `192`;
- fixed horizon is not convergence or steady-state evidence.

No dataset, seed, coefficient, critic budget, actor horizon, learning rate,
expectile, gamma, lambda, PPO setting, evaluation horizon, terminal/timeout/tail
rule, or reporting responsibility may be changed.

## Advantage contract

One-step TD is computed from the shared critic. GAE is accumulated over the
ordered behavior trajectory.

- true terminal: no bootstrap and stop recursive carry;
- timeout: bootstrap and stop recursive carry;
- final stored nonterminal row: bootstrap and stop recursive carry;
- terminal and timeout may not overlap;
- `lambda=0` must equal one-step TD;
- no advantage normalization or clipping;
- actor-facing advantages remain float32;
- numerical checks must separate float32 storage quantization from independent
  implementation disagreement.

## Shared critic

For each dataset/seed pair, bind the dataset, critic, and advantage identities.
All matching branches use that same critic. It remains frozen during actor
training, actor and critic parameters remain disjoint, and terminal evidence
verifies critic immutability.

## Repository integration

Before coding, read `AGENTS.md`, `docs/handoff.md` section 0,
`experiments/registry.yaml` when present, and inspect existing E7 planning,
branch lifecycle, trainer, A2C/PPO, taper, evaluation, checkpoint, resume,
aggregation, RunSpec, lane, and artifact-delivery capabilities.

Extend existing responsibilities where compatible. Add a production module only
when its responsibility is not already owned by an existing module, and record
the closest inspected predecessor. Do not create an unrelated generic training
framework, duplicate execution lane, duplicate scheduler, or duplicate
publication path.

## Required deliverables

- frozen experiment configuration or equivalent protocol input;
- GAE computation and boundary validation;
- shared-critic and prepared-advantage identity checks;
- adapter into both canonical actor modes;
- exact 192-branch expansion;
- resumable plan/run entrypoint using repository conventions;
- terminal engineering audit;
- focused tests;
- one existing-schema RunSpec or equivalent authorized launch record;
- one operator entrypoint;
- concise scope/status documentation.

## Reporting boundaries

Separately report task-performance collapse, support/variance-boundary events,
NaN/Inf failure, terminal and late-window behavior, and paired GAE-minus-TD
comparisons without failed-cell imputation.

Do not claim universal GAE, A2C, PPO, or taper superiority, convergence, steady
state, controlled causal identification, or OOD generalization. Hopper is
external-validity evidence.

## Terminal submission

Submit one complete repository patch/tree with changed-file summary, production
and test code-size accounting, reused modules/symbols, tests actually run, tests
or liveness not run, and remaining uncertainty.

You have one initial complete attempt and at most two repair attempts. Do not
start a formal scientific run.
