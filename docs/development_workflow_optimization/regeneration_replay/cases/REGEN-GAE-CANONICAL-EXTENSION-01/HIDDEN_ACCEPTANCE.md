# REGEN-GAE-CANONICAL-EXTENSION-01 — Hidden Acceptance Contract

Evaluator-only. Freeze status: **closed before generation**.

## A. Scope and provenance

- checkout base is exactly `aedbe8cc7e50f730cb6fcde90ca1816b97d79173`;
- no `docs/handoff.md` or `experiments/registry.yaml` authority mutation;
- no scientific result, completion, convergence, steady-state, or method-ranking
  claim;
- held-out seeds `204–207` absent from executable branches;
- all changed scientific values match the task contract exactly.

## B. Matrix

The plan expands exactly:

`3 datasets × 4 seeds × 2 estimators × 2 actor modes × 4 controls = 192`

with unique branch identities.

Required controls are Positive-only and squared EXP `c=64,128,256`. Required
actor modes are canonical A2C and PPO-K4. Required estimators are one-step TD and
GAE(`lambda=0.95`).

## C. GAE and boundaries

Hidden vectors cover:

- `lambda=0` exact equality with one-step TD;
- terminal without bootstrap and without cross-episode carry;
- timeout with bootstrap and without cross-episode carry;
- final nonterminal row with bootstrap and no future carry;
- terminal/timeout overlap rejection;
- non-boundary trajectory discontinuity rejection;
- NaN/Inf rejection;
- independent float64-reference agreement;
- actor-facing float32 artifact equals the float32 cast of the reference;
- storage quantization reported separately from implementation disagreement.

## D. Shared critic and prepared artifacts

- one critic identity per dataset/seed pair;
- TD and GAE artifacts bind dataset identity, transition count, seed, critic, and
  advantage hashes;
- every branch sharing dataset/seed loads the same critic identity;
- critic parameters are disjoint from actor parameters;
- critic initial and final state hashes match;
- prepared advantage reconstruction or adapter use does not alter the advantage.

## E. Reuse and architecture

The evaluator inspects whether the historical base already owns:

- planning and branch lifecycle;
- canonical trainer/minibatch path;
- A2C and PPO injection;
- squared-remoteness kernel;
- evaluation and checkpoints;
- resume and identity;
- aggregation and terminal diagnostics;
- RunSpec lane and artifact policy.

Acceptance does not require one exact file layout. It rejects a parallel generic
trainer/runner/scheduler/aggregation stack when the same responsibilities can be
satisfied by existing modules plus a bounded experiment-specific extension.

A legitimately new GAE recurrence, prepared-artifact verifier, thin adapter,
matrix definition, or terminal audit may be accepted when responsibility and
closest predecessor are documented.

## F. Tests and static checks

Mandatory when available in the benchmark environment:

- Python compilation for all changed Python;
- shell syntax for changed shell scripts;
- focused GAE, matrix, adapter, artifact, and audit tests;
- full repository pytest;
- Ruff;
- handoff authority;
- formal execution channel;
- governance inventory and stage checks;
- RunSpec validation.

A candidate that passes only its own new tests but breaks existing tests fails.

## G. Real liveness

Before an implementation is judged execution-complete, run from the exact frozen
attempt:

1. one A2C + one-step-TD branch reaching at least one actor update;
2. one PPO-K4 + GAE branch reaching at least one actor update.

Both must:

- load the prepared artifact;
- preserve the registered branch identity;
- record the correct estimator and actor mode;
- verify unchanged critic identity;
- avoid parser/argument failure before actor update;
- remain non-scientific liveness evidence.

External asset unavailability is reported as `ENVIRONMENT_INVALID` only when the
asset was predeclared unavailable to both arms. It cannot convert a code failure
into an environment failure.

## H. Terminal classification

`ACCEPTED` requires A–G.

A candidate is `REJECTED_INCOMPLETE` for missing required functionality or
liveness, `REJECTED_UNSAFE` for scope/scientific drift or weakened checks, and
`REJECTED_GATE_FALSE_POSITIVE` only when all task checks pass but the candidate
gate still rejects the final attempt.

## I. Size and ROI collection

Size does not affect correctness classification. After classification, collect:

- production Python churn;
- test Python churn;
- total changed files;
- new production Python files;
- duplicated responsibilities;
- attempt count;
- gate runtime and evidence size;
- generation, test, liveness, review, and total wall time;
- token/tool counts when exposed.
