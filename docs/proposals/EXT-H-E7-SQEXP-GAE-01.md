# EXT-H-E7-SQEXP-GAE-01 — Frozen-Critic Behavior-Trajectory GAE Pilot

## Status

- Experiment class: code-first development pilot.
- Current result status: **not run**.
- Formal branch count started by this implementation change: **0**.
- This document is an implementation scope record, not an authoritative registry or handoff update. Authoritative registration remains subject to the code-first pilot-registration fastpath after the implementation SHA is frozen.

## Claim

Test whether replacing matched one-step TD advantages with behavior-trajectory GAE changes the finite-horizon performance of the existing E7 squared-remoteness actor update under a shared frozen critic.

This is an external-validity pilot. It does not replace C-U1/D-U1 controlled mechanism evidence and does not identify a universal actor-update or taper ranking.

## Frozen matrix

- Datasets:
  - `hopper-medium-expert-v2`
  - `walker2d-medium-v2`
  - `walker2d-medium-replay-v2`
- Development seeds: `200, 201, 202, 203`
- Held-out seeds reserved and unused: `204, 205, 206, 207`
- Actor update modes:
  - canonical A2C
  - PPO clip, epsilon `0.2`, old-policy cadence `K=4`
- Advantage modes:
  - matched frozen-critic one-step TD
  - behavior-trajectory GAE with `gamma=0.99`, `lambda=0.95`
- Negative controls:
  - Positive-only
  - squared-remoteness EXP `c=64`
  - squared-remoteness EXP `c=128`
  - squared-remoteness EXP `c=256`
- Actor horizon: fixed `1,000,000` updates.
- Total branches: `3 datasets × 4 seeds × 2 advantage modes × 2 actor modes × 4 controls = 192`.

The coefficient shortlist intentionally contains several previously strong high-`c` settings. No single `c` is selected before this pilot, and no per-dataset post-hoc selection is allowed.

## Shared frozen critic

For each dataset/seed pair, preparation trains exactly one canonical expectile critic for `100,000` updates with:

- batch `256`;
- learning rate `3e-4`;
- `gamma=0.99`;
- expectile `tau=0.7`;
- canonical network preset and source contract.

That critic checkpoint is reused by every actor/advantage/control branch sharing the dataset and seed. It is loaded strictly and is not updated during actor training. The actor wrapper preserves the existing A2C/PPO objective implementations and feeds the prepared advantage through the trainer's existing `ep_ret` slot.

## Ordered-trajectory contract

The preparation stage fails closed unless:

1. all arrays have the same nonzero length and finite values;
2. terminal and timeout flags never overlap;
3. for every non-boundary row `t`, `next_observation[t]` matches `observation[t+1]` within the frozen tolerance;
4. the ordered transition table has a deterministic identity hash;
5. runtime trainer rewards, terminals, and timeouts exactly match the prepared hashes.

### Terminal, timeout, and tail semantics

The one-step residual is

`delta_t = r_t + gamma * (1 - terminal_t) * V(s_{t+1}) - V(s_t)`.

GAE is

`A_t = delta_t + gamma * lambda * continue_t * A_{t+1}`,

where `continue_t` is false for terminals, timeouts, and the final stored row.

Therefore:

- terminal: no bootstrap; stop recursion;
- timeout: bootstrap; stop recursion;
- incomplete dataset tail: bootstrap when nonterminal; stop recursion because no following row exists.

No advantage normalization or clipping is applied.

## Bug gates

The implementation includes the following required regressions:

- `lambda=0` GAE is bit-identical to matched one-step TD;
- synthetic terminal behavior has no bootstrap and no cross-episode trace;
- synthetic timeout behavior bootstraps but does not leak into the next episode;
- nonterminal dataset tail bootstraps but stops recursion;
- non-boundary trajectory discontinuity fails closed;
- terminal/timeout overlap fails closed;
- prepared array and checkpoint hashes are verified before every plan/run;
- external advantages are reconstructed inside the unchanged parent A2C/PPO update to tolerance `1e-6`;
- critic parameters are checked for exact immutability after every actor update;
- actor and critic parameter sets must be disjoint;
- nonuniform return-weighted sampling is prohibited.

## Reporting boundaries

The fixed 1M horizon is not convergence or steady-state evidence. Any eventual result must separately report:

1. task-performance degradation or collapse;
2. support/variance-boundary events;
3. NaN/Inf numerical failure;
4. terminal checkpoint and late-window behavior;
5. GAE-minus-one-step paired differences without failed-cell imputation.

No claim that GAE, PPO, A2C, or any `c` is universally better is authorized by this implementation.
