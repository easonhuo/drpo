# E8 Joint Fitted-Reference beta-TOPR pilot result

## Status and scientific identity

- Experiment: `EXT-C-E8-ORACLE-OFFLINE-V2-JOINT-FITTED-REFERENCE-TOPR-0.5B-01`
- RunSpec: `E8_JOINT_FITTED_REFERENCE_TOPR_20260722_01`
- Result status: **pilot**
- Scientific role: Countdown / Transformer external-validity diagnostic
- Method identity: **Joint Fitted-Reference beta-TOPR**, not canonical frozen-behavior TOPR
- Frozen horizon: 1200 optimizer steps; this is not convergence or steady-state evidence
- Development seeds: `4000`, `5000`
- Test split: unused

The policy adapter is `default` and the jointly fitted reference adapter is `reference` on one frozen Qwen2.5-0.5B-Instruct backbone. The negative-branch weight is

```text
exp(beta * min(sum_logpi - sum_logmu, 0))
```

where the full-completion log-ratio is detached. The task objective uses mean-token log probability and the reference objective assigns branch mass 0.5 to positives and 0.5 in total to unique negatives, uniformly within the negative branch. Policy and reference are updated one-to-one.

## Durable evidence and provenance

The complete text-first result record is stored in `easonhuo/drpo-results`:

- results commit: `68ea4980ed9c8ebb79e02f7d2b40a7e2a8ee0461`
- result path: `runs/e8/E8_JOINT_FITTED_REFERENCE_TOPR_20260722_01/`
- result-manifest SHA-256: `7cffa3cb179151ae721daa25f091855c80572686d887531c047ac5c2da7eece8`
- source-artifact ZIP SHA-256: `b8775255f453253846ae750e095ed2f49995d2d43ea14e0a7472cd488bd0876a`
- completed cells: `16/16`
- failed cells: `0`
- NaN/Inf failures: `0`

The run records source commit `f3712fdb6dd3ec16807cee72dc2afe752ee6c90c`, branch `dev/ext-c-e8-joint-fitted-reference-topr-01`, and `dirty=false`. That commit is currently not resolvable through the authoritative `easonhuo/drpo` GitHub commit API. The result package preserves protected-source SHA-256 values, but no source-equivalence audit has yet promoted those hashes to an authoritative repository commit. Therefore this result remains a provenance-limited pilot and must not be upgraded to finite-step validated or formal evidence.

The immutable data identities are:

- bank SHA-256: `3887fe4b13b1ff89e904e0816491e13a74ecd1d1dfe71dc1211b0dcae2f69519`
- structurally disjoint held-out evaluation SHA-256: `b21069c90746f7e7207d7b61079b28c6a71935da782bb696e771edbcda4fd004`
- base-config SHA-256: `492f063f0cb41dded82355bb4bd9c3d0524f607d670599a5f48a51a5c0099866`
- grid-config SHA-256: `b80daeec3b193dd387089d9afc344335c93a5afc7e710f61c607458ae18ab956`

## Frozen beta response

The run evaluated

```text
beta = {0, 0.25, 0.5, 0.75, 1, 1.5, 2, 4}
```

with both paired development seeds. Best-checkpoint Pass@8 is included only as a supplementary diagnostic. Terminal Pass@8 and terminal valid rate are the primary finite-horizon values shown below.

| beta | best Pass@8 mean | terminal Pass@8, seed 4000 | terminal Pass@8, seed 5000 | terminal Pass@8 mean | terminal valid-rate mean |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.078 | 0.032 | 0.016 | 0.024 | 0.252 |
| 0.25 | 0.165 | 0.170 | 0.160 | 0.165 | 0.995 |
| 0.5 | 0.186 | 0.126 | 0.140 | 0.133 | 0.994 |
| 0.75 | 0.172 | 0.142 | 0.138 | 0.140 | 0.993 |
| 1 | 0.176 | 0.128 | 0.150 | 0.139 | 0.996 |
| 1.5 | 0.191 | 0.150 | 0.138 | 0.144 | 0.992 |
| 2 | 0.187 | 0.154 | 0.144 | 0.149 | 0.998 |
| 4 | 0.190 | 0.142 | 0.148 | 0.145 | 0.995 |

## Interpretation

The `beta=0` no-ratio-taper boundary exhibits severe finite-horizon task-performance and output-validity degradation: mean terminal Pass@8 is `0.024` and mean terminal valid rate is `0.252`. Every tested nonzero point from `beta=0.25` through `beta=4` keeps terminal valid rate near `0.99` and terminal Pass@8 in the approximate `0.133--0.165` range.

This localizes a large transition somewhere between `beta=0` and `beta=0.25`; the original grid is too coarse to resolve its shape. The observed nonzero points form a broad finite-horizon plateau rather than a sharply identified optimum. With only two development seeds, no beta is declared best and no significance claim is permitted.

Several cells have supplementary best-checkpoint values above their terminal values. Combined with the fixed 1200-step horizon, this prevents a saturation, convergence, or steady-state claim. The raw per-checkpoint curves remain the authority for later trajectory analysis.

## Failure taxonomy

- **Task performance:** strongly degraded at `beta=0`; nonzero points retain substantially higher held-out Pass@8.
- **Support/structure boundary:** no formal boundary threshold was registered. Valid rate is reported only as a structure/output-validity diagnostic and must not be relabeled as a formal support-boundary event.
- **NaN/Inf numerical failure:** `0/16` cells.

These three categories must remain separate in all paper-facing and follow-up reporting.

## Successor experiment

The evidence motivates the fixed-profile successor

`EXT-C-E8-ORACLE-OFFLINE-V2-JOINT-FITTED-REFERENCE-BETA-TOPR-DENSE-0.5B-01`,

which freezes

```text
beta = {0, 0.01, 0.02, 0.04, 0.08, 0.125, 0.25, 0.5}
seeds = {4000, 5000}
```

for 16 cells. It changes only beta resolution in the `0--0.25` transition region while retaining the same model, bank, seeds, learning rates, optimizers, schedulers, reference target, one-to-one update frequency, 1200-step horizon, and held-out evaluation protocol. It uses a dedicated fixed-profile runtime and does not reactivate the suspended generic E8 config-driven execution surface.

## Prohibited claims

This pilot does not authorize:

- canonical frozen-behavior TOPR reproduction;
- convergence, saturation, or steady-state ranking;
- a statistically established best beta;
- formal method ranking or cross-task superiority;
- state-distribution OOD generalization;
- substitution for D-U1 controlled causal identification.
