# EXT-C-E8-ORACLE-OFFLINE-V2-REPRO-RNG-AUDIT-0.5B-01

## Status

Code-first development pilot. Not authoritatively registered and not started.
The implementation SHA must be frozen before registration closure through the
reviewed pilot-registration fastpath.

## Claim

Determine whether historical high and low Countdown continuous-EXP cells
reproduce under the historical in-process evaluation RNG behavior, and measure
how an opt-in evaluation save/restore boundary changes the same six cells.

This is a reproducibility/protocol audit, not a method-ranking experiment.

## Frozen cells per protocol

- `(alpha=0.5,c=1,seed_offset=3000)`;
- `(alpha=0.5,c=1,seed_offset=4000)`;
- `(alpha=0.5,c=1,seed_offset=13000)`;
- `(alpha=0.5,c=1,seed_offset=16000)`;
- `(alpha=1,c=8,seed_offset=13000)`;
- `(alpha=1,c=8,seed_offset=16000)`.

Two protocol phases run sequentially on the identical explicit GPU pool:

1. `legacy_contaminated_v1`: historical evaluator behavior, unchanged;
2. `rng_isolated_v2`: evaluation reseeds internally but restores Python, NumPy,
   Torch CPU, and all visible CUDA RNG states before training resumes.

Total: 12 cells. One cell per GPU. Test access is forbidden.

## Inherited scientific protocol

Qwen2.5-0.5B-Instruct, fresh LoRA per cell, frozen E8 V2 bank, all
first-occurrence unique negatives, `u=d/2`, `w=alpha*exp(-c*u^2)`, unique-count
denominator, 1200 steps, no early stop, Greedy/Pass@8 every 100, Pass@64 every
200, validation only. No near/far selection, hidden scale, budget matching,
dynamic alpha, SFT warmstart, on-policy sampling, or replay refresh.

## Primary audit outputs

- exact per-cell late-window Pass@8 over steps 800--1200;
- terminal Pass@8 at step 1200;
- paired isolated-minus-legacy differences;
- Pass@64, Greedy, and valid-rate diagnostics;
- task performance, valid-rate structure proxy, and NaN/Inf reported separately.

Best checkpoint cannot select a protocol. Fixed 1200 steps is not convergence.
No steady-state, controlled-mechanism, OOD, or method-ranking claim is allowed.

## Historical result-registration backlog

The completed 62-cell grid, alpha=1 c-scan, high-c scan, and log-c boundary scan
must be closed in a separate authoritative registration transaction after this
implementation SHA is frozen. Their original run commits, package checksums,
terminal audits, validation-only status, and non-claims must remain distinct;
this audit must not rewrite or merge their evidence identities.
