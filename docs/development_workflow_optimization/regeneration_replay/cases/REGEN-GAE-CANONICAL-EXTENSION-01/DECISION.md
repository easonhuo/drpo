# REGEN-GAE-CANONICAL-EXTENSION-01 Decision

## Decision

**NO CAUSAL A/B VERDICT. RETAIN BOTH TRAJECTORIES.**

The first fresh-context GAE regeneration did not produce a valid candidate-gate A/B pair.

- `replay/regen-gae-run-01` submitted one complete repository attempt at `75b30004fad742e0bb07daa059fe757b183d44eb`.
- `replay/regen-gae-run-02` remained at its task-packet commit `8af0cd1bfd7410c5c3a7cb62f4efe99709052abf`; no implementation commit or push exists.
- The randomized treatment assigned the candidate gate to run 02 only after a complete submitted attempt. Because no such attempt existed, no candidate-gate verdict or repair feedback was delivered.

This outcome cannot estimate candidate-gate code reduction, repair probability, false rejection, unsafe pass, or wall-time ROI.

## Run 01 audit

Run 01 is useful evidence, but it is not acceptable as an execution-complete implementation.

Repository-verified size, excluding the frozen task packet:

- 19 changed files;
- 1,783 production-Python added lines;
- 475 test-Python added lines;
- 14 new production Python files;
- 2,438 total added lines across implementation, tests, configuration, launch record, and documentation.

The static matrix is correct: 12 shared critic/preparation jobs and 192 unique actor branches over the frozen datasets, development seeds, estimators, actor modes, and controls. The focused GAE tests exercise terminal, timeout, final-nonterminal, lambda-zero, float32 storage, matrix, artifact identity, and a synthetic tiny actor path.

However, the hidden acceptance result is `REJECTED_INCOMPLETE`:

1. **Reuse/architecture fails.** The attempt creates a new actor, critic, objective, actor trainer, critic preparation path, coordinator, evaluation module, aggregation module, artifact helpers, and identity layer. This is a parallel experiment stack rather than a bounded GAE extension over the historical base's canonical E7 trainer, lifecycle, checkpoint, resume, evaluation, and terminal diagnostics.
2. **RunSpec integration is not demonstrated.** The added `experiments/run_specs/...yaml` is an ad hoc launch record; no existing-lane RunSpec validation result exists.
3. **Repository validation is incomplete.** The generator reported 14 focused tests and Python compilation, but no full repository pytest, Ruff, handoff authority, formal execution channel, governance checks, or external CI ran on the submitted commit.
4. **Real liveness is absent.** Neither the required A2C+TD actor update nor the PPO-K4+GAE actor update ran from the exact submitted commit.
5. **Resume is only branch-granular.** An interrupted million-step actor branch is rejected as an incomplete output path instead of resuming through the existing canonical checkpoint path.

No scientific variable drift, held-out-seed use, authority mutation, or scientific result claim was found. Therefore the classification is incomplete rather than unsafe.

## Run 02 audit

Run 02 is classified `NO_SUBMISSION_TIMEOUT`.

Its final session report described uncommitted work with 2,389 production-Python additions, 453 test-Python additions, 13 focused tests, a 12/192 plan, and Python compilation. Those claims are retained as trajectory metadata but cannot be independently audited because no tree, commit, patch, or branch update was produced.

The failure happened before candidate treatment. It is therefore an orchestration/submission failure, not evidence that the candidate gate accepted, rejected, repaired, slowed, or reduced an implementation.

## Descriptive historical context

The submitted run-01 tree is 52.13% smaller in total additions than the historical 5,093-line overgrown PR #92, indicating that the frozen task packet's explicit reuse language helped. It is still 78.87% larger in total additions than the later 1,363-line canonical PR #107 and independently reconstructs responsibilities that the canonical E7 path already owned.

These comparisons are descriptive only because bases, available infrastructure, and development histories differ.

## What this result supports

It supports three narrow conclusions:

1. Prompt-level reuse instructions alone reduced but did not eliminate GAE code bloat.
2. Both fresh generators still moved toward large standalone `e7_sqexp_gae*` stacks; a mechanical pre-implementation or post-attempt reuse gate remains justified for testing.
3. Manual dual-session orchestration is not reliable enough for a solid benchmark: only one of two scheduled runs produced a repository submission.

## What this result does not support

It does not support:

- candidate-gate effectiveness;
- candidate-gate repair-at-1 or repair-at-2;
- a candidate-versus-baseline code-size effect;
- a candidate-versus-baseline time or token ROI;
- adoption, merge, shadow-default, or hard-block activation.

## Required next benchmark state

Preserve this pair as the first failed operational trajectory. Do not delete or relabel it as a successful A/B.

The next valid GAE comparison must ensure that both isolated workers automatically emit an immutable attempt artifact even on timeout, including the working-tree patch, changed-file inventory, test log, elapsed time, and terminal status. Treatment feedback may be routed only after that immutable first-attempt artifact exists. A fully automated regeneration orchestrator is therefore an experimental-control requirement, not merely a convenience optimization.
