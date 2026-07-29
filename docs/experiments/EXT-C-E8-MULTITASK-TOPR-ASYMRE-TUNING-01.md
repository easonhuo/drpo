# EXT-C-E8-MULTITASK-TOPR-ASYMRE-TUNING-01

## Status and role

- Status: implementation candidate / not run
- Role: external-validity development tuning across the eight new discrete tasks
- Parent: `EXT-C-E8-MULTITASK-EXP-TUNING-01`
- Causal authority: unchanged; D-U1 remains the categorical controlled-identification environment
- Formal registration: deferred until implementation SHA freeze and review

This experiment answers only whether the declared TOPR and AsymRE parameter grids show useful validation response curves on the eight new tasks. It does not establish convergence, significance, universal ranking, or causal identification.

## Shared data and initialization

The experiment does not construct an independent data split. It consumes the exact artifacts from the Exp tuning preparation:

- the same deterministic `5000/500/500` split for each P0 bank;
- the same train and validation prompt hashes;
- the same train-only 100-update positive reference adapter for each task;
- zero validation/test exposure during reference preparation;
- the same Qwen2.5-0.5B model, LoRA initialization, optimizer, learning-rate schedule, update horizon, evaluation cadence, generation settings, and late-window definition.

The runner fails closed on any identity mismatch. Test paths may exist in the parent split manifest for identity separation, but this experiment never opens them.

Countdown is not rerun. Its already completed dense TOPR and AsymRE development scans remain separate historical external evidence. Positive-only is not rerun on the second server; its paired task values come from the Exp tuning experiment after both implementations and shared inputs are audited.

## Frozen 80-cell matrix

For every one of the eight new tasks:

```text
joint fitted-reference beta-TOPR:
    beta in [0, 0.04, 0.08, 0.25, 0.5]

AsymRE:
    delta_v in [-1.0, -0.9, -0.7, -0.5, 0.0]
```

Therefore:

```text
8 tasks x (5 TOPR + 5 AsymRE) = 80 cells
16 concurrent cells x 5 waves = 80 cells
```

Every point is predetermined. There is no result-conditioned refinement in this stage.

## Joint fitted-reference beta-TOPR

This is explicitly the existing fitted-reference extension, not canonical frozen-behavior TOPR.

The policy and reference adapters start with identical parameters copied from the same train-only task reference. During every update:

1. the reference adapter receives a branch-balanced density-fitting objective over oracle positives and all 16 unique verifier-wrong negatives per prompt;
2. the reference outputs used in the policy ratio are detached;
3. the policy adapter receives the signed task objective;
4. both adapters have independent AdamW optimizer state but inherit the same learning-rate schedule and update horizon.

For a negative completion:

```text
log_ratio = sum_token_log_pi - sum_token_log_mu
weight = exp(beta * min(log_ratio, 0))
policy_loss = -mean(log_pi_positive) + mean(weight * log_pi_negative)
```

The task loss uses mean-token log probability while the ratio coordinate uses full-completion summed log probability, matching the existing Countdown fitted-reference protocol. At initialization, policy and reference parameters must be identical and the maximum absolute ratio must pass the frozen tolerance.

`beta=0` is the uncontrolled-negative ratio boundary. It is retained to show the response transition but is excluded from active TOPR parameter selection.

## AsymRE

AsymRE uses all 16 unique negatives and no remoteness coordinate, near/far selection, taper, critic, or learned value baseline.

With equal positive and negative branch mass and signed rewards `+1/-1`:

```text
A = R - delta_v
positive coefficient = 1 - delta_v
negative repulsion coefficient = 1 + delta_v

loss = -(1-delta_v) * mean(log_pi_positive)
       + (1+delta_v) * mean(log_pi_negative)
```

`delta_v=-1` makes the negative coefficient zero. It is a zero-negative boundary, not an active AsymRE setting, and is excluded from active AsymRE parameter selection.

## Training and evaluation

All training and evaluation values are inherited from the Exp configuration rather than duplicated here. The required contract includes:

- exactly 1200 optimizer updates;
- no early stopping;
- evaluation every 100 updates;
- late window at updates `800, 900, 1000, 1100, 1200`;
- greedy validation on 500 prompts;
- Pass@8 validation on the frozen 128-prompt subset;
- terminal adapters always saved;
- best-validation adapters retained only as supplementary recovery/diagnostic artifacts.

Each method must pass a real two-update liveness gate before any full wave. TOPR liveness must show finite nonzero policy and reference gradients, changed policy and reference parameters, and successful policy-adapter reload. AsymRE liveness must show finite nonzero policy gradient, changed policy parameters, and successful reload. Liveness is engineering evidence only.

## Selection and reporting

The primary tuning metric is validation late-window mean Pass@8, subject to finite execution and terminal greedy valid rate at least 0.95. Tie-breaking follows terminal Pass@8, late-window greedy success, terminal greedy success, then the weaker absolute parameter.

Active candidates are selected separately by task and method:

- TOPR excludes `beta=0`;
- AsymRE excludes `delta_v=-1`.

The report must explicitly state when all active candidates fall below their boundary control. A selected TOPR value at `beta=0.5` leaves the high-beta boundary unclosed. A selected AsymRE value at `delta_v=-0.9` leaves the weak-negative boundary unclosed. Any extension requires a new registered protocol; it cannot be added privately after inspecting the curve.

The terminal audit separately reports:

1. task-performance trajectories;
2. valid-format/structure diagnostics;
3. NaN/Inf numerical failure.

No post-hoc task-collapse threshold is introduced. A 1200-update result remains a finite-horizon pilot and is not a convergence or steady-state result.
