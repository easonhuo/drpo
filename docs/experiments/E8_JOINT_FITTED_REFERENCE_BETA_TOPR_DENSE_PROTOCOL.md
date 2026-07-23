# E8 Joint Fitted-Reference beta-TOPR dense transition protocol

## Identity and status

- Experiment ID: `EXT-C-E8-ORACLE-OFFLINE-V2-JOINT-FITTED-REFERENCE-BETA-TOPR-DENSE-0.5B-01`.
- Current status: `dev_code_first_unregistered`, `not_run`.
- Environment role: Countdown external-validity development pilot only.
- Method identity: Joint Fitted-Reference beta-TOPR, not canonical frozen-behavior TOPR.

The previous eight-point pilot showed a severe failure at `beta=0` and recovery for every tested positive beta, while `beta=0.25` was already in a strongly attenuated regime. The previous run remains historical development evidence, but its recorded source commit `f3712fdb6dd3ec16807cee72dc2afe752ee6c90c` is not currently resolvable through the repository. This successor therefore reruns `beta=0`, `0.25`, and `0.5` as internal anchors and adds five low-beta points under a clean, reviewable fixed-profile implementation.

## Frozen question

Where is the low-beta transition from uncontrolled-negative degradation to stable behavior-relative tapering on the frozen E8 V2 bank?

The experiment does not test longer-horizon saturation. It cannot establish convergence, steady state, a best beta, statistical significance, or formal method ranking.

## Frozen matrix

\[
\beta\in\{0,0.01,0.02,0.04,0.08,0.125,0.25,0.5\}.
\]

Each beta uses seed offsets `4000` and `5000`, for `8 x 2 = 16` cells. The only scanned scientific variable is beta.

The following remain frozen from the predecessor:

- model and LoRA parameterization;
- bank, validation split, and deduplication;
- policy and reference learning rates and schedulers;
- one reference update per policy update;
- `0.5/0.5` positive/negative reference branch mass;
- 1200 steps, no early stopping;
- evaluation every 100 steps and Pass@64 every 200 steps;
- late-window and terminal reporting;
- development seeds and no `test.jsonl` access.

## Objective

One frozen Qwen2.5-0.5B-Instruct backbone carries two identically initialized LoRA adapters:

- policy: `default`;
- fitted reference: `reference`.

The fitted reference objective assigns half of its probability-mass target to the positive completion and half uniformly across the unique negative completions for each prompt.

For a negative completion, the detached weight is

\[
w_{\beta}=
\exp\left(\beta\min(\log\pi-\log\mu,0)\right),
\]

where both log probabilities are full-completion sums. The policy task objective retains the existing mean completion-token log-probability scale and unique-negative-per-prompt denominator. There is no value network, learned baseline, absolute-surprisal taper, near/far selection, or weight-sum normalization.

## Fixed-profile execution boundary

The repository-wide generic E8 config-driven runtime remains suspended. This successor authorizes only the exact fixed profile and fixed matrix named above through:

- `configs/countdown_e8_oracle_offline_v2_joint_fitted_reference_beta_topr_dense_0p5b.yaml`;
- `scripts/run_countdown_e8_joint_fitted_reference_beta_topr_dense.sh`.

The entrypoint rejects a dirty checkout, validates the exact matrix, uses GPU `0,1` with one cell per GPU, and runs eight waves.

Before the full matrix, it must complete:

1. the two-step real Qwen/PEFT/CUDA liveness at `beta=0.25`;
2. initial full-sequence ratio max-abs `<=1e-5`;
3. finite nonzero policy and reference gradients and updates;
4. dual-adapter checkpoint save;
5. a fresh-process reload of both `default` and `reference` adapters;
6. finite forward passes after switching to each reloaded adapter.

The liveness and reload gates are engineering evidence, not scientific evidence.

## Reporting and terminal audit

Primary reporting is mean Pass@8 over steps `800,900,1000,1100,1200`; terminal Pass@8 is secondary. Best-checkpoint metrics are supplementary only. Report beta, reference loss, reference positive and negative log probability, log-ratio quantiles, weight quantiles, clipped-at-one fraction, both raw-gradient norms, both update norms, Pass@8, Pass@64, greedy success, and valid-expression rate.

Task-performance degradation, valid-expression or structure degradation, and NaN/Inf numerical failure must be reported separately. A fixed 1200-step horizon is not convergence or saturation evidence.

## Governance sequence

1. Freeze implementation SHA and exact-head CI.
2. Prepare the RunSpec bound to that SHA.
3. Complete normal schema-v3 code-first registration.
4. Run real liveness and dual-adapter reload verification.
5. Claim the registered RunSpec explicitly.
6. Run the 16 cells under foreground supervision.
7. Complete terminal audit, package, and durable delivery before scientific closure.

This document and the RunSpec do not by themselves authorize an unregistered launch or merge to `main`.
