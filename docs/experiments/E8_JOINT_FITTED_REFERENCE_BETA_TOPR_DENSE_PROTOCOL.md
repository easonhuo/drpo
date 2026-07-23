# E8 Joint Fitted-Reference beta-TOPR dense transition protocol

## Identity and status

- Experiment ID: `EXT-C-E8-ORACLE-OFFLINE-V2-JOINT-FITTED-REFERENCE-BETA-TOPR-DENSE-0.5B-01`.
- Result status: completed `pilot`, provenance-limited closure.
- Durable result: `easonhuo/drpo-results@712c36c5e858182de4e93c48dfe917bd42198c67`.
- Environment role: Countdown external-validity development pilot only.
- Method identity: Joint Fitted-Reference beta-TOPR, not canonical frozen-behavior TOPR.
- Scientific exploration status: closed; no further beta sweep is required.

The previous eight-point pilot showed a severe failure at `beta=0` and recovery for every tested positive beta, while `beta=0.25` was already in a strongly attenuated regime. The previous run remains historical development evidence, but its recorded source commit `f3712fdb6dd3ec16807cee72dc2afe752ee6c90c` is not currently resolvable through the repository. This successor reran `beta=0`, `0.25`, and `0.5` as internal anchors and added five low-beta points.

The completed successor reports clean local source commit `3733ee28cc1517b67ad235afa10f2e855f2dde33`, which is also not currently resolvable from the authoritative remote. The deposited result therefore remains a provenance-limited pilot. Protected source hashes, input hashes, terminal audit, result manifest, and artifact hashes are preserved in the result package and the repository provenance audit.

## Frozen question

Where is the low-beta transition from uncontrolled-negative degradation to stable behavior-relative tapering on the frozen E8 V2 bank?

The experiment does not test longer-horizon saturation. It cannot establish convergence, steady state, a universally best beta, statistical significance, or formal method ranking.

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

The repository-wide generic E8 config-driven runtime remains suspended. The reviewed fixed-profile implementation is already present on `main` through PR #261, with the required `runtime_scope` repair through PR #262. The canonical repository entrypoints remain:

- `configs/countdown_e8_oracle_offline_v2_joint_fitted_reference_beta_topr_dense_0p5b.yaml`;
- `scripts/run_countdown_e8_joint_fitted_reference_beta_topr_dense.sh`.

The completed server run recorded eight visible H20 GPUs and two runtime slots per GPU, completing all 16 cells in one wave. That is a recorded wall-clock scheduling fact, not a scientific-variable change and not a new universal hardware requirement. Per the user's earlier explicit instruction, the separate local eight-GPU scheduling patch is not promoted by this closure; the reviewed repository fixed profile remains unchanged.

Before the full matrix, the run recorded:

1. a two-step real Qwen/PEFT/CUDA liveness at `beta=0.25`;
2. initial full-sequence ratio max-abs within `1e-5`;
3. finite nonzero policy and reference gradients and updates;
4. dual-adapter checkpoint save;
5. a fresh-process reload gate required before matrix execution.

`CHECKPOINT_RELOAD_GATE.json` was generated and hashed, but its body was excluded from the compact deposited result package. The exact hash and size are preserved in the provenance audit. Liveness and reload gates are engineering evidence, not scientific evidence.

## Reporting and terminal audit

Primary reporting is mean Pass@8 over steps `800,900,1000,1100,1200`; terminal Pass@8 is secondary. Best-checkpoint metrics are supplementary only. The deposited result reports beta, reference loss, reference positive and negative log probability, log-ratio quantiles, weight quantiles, clipped-at-one fraction, both raw-gradient norms, both update norms, Pass@8, Pass@64, greedy success, and valid-expression rate.

Task-performance degradation, valid-expression or structure degradation, and NaN/Inf numerical failure are reported separately. The terminal audit reports 16/16 cells, zero failed cells, zero NaN/Inf failures, no test-data use, and PASS. A fixed 1200-step horizon is not convergence or saturation evidence.

## Result interpretation and closure

The completed response curve supports these bounded findings:

- `beta=0` repeatedly exhibits severe task-performance and valid-expression degradation without NaN/Inf;
- very small positive beta recovers much of the task metric before valid-expression stability is fully restored;
- the main structure-stability transition lies approximately between `beta=0.04` and `0.08` in this pilot;
- `beta=0.08--0.5` forms a broad high-validity plateau;
- the point ordering between `beta=0.25` and `0.5` is not robust across the predecessor and dense scans.

For the later budget-matched comparison against the main model, freeze:

```text
method = joint_fitted_reference_topr
beta = 0.5
selection metric = dense-scan late-window Pass@8 mean
retuning after main-model or test observation = forbidden
```

Beta `0.5` is selected because it has the largest registered dense-scan late-window Pass@8 mean while remaining in the approximately 99% valid-rate plateau. This is a transparent frozen baseline-selection rule, not a universal or significant optimality claim.

No further TOPR beta scan is required. A future comparison may evaluate the frozen configuration once under the same budget, seeds, checkpoint policy, and evaluation protocol as the main model, without reopening tuning.

## Evidence pointers

- Full result and interpretation: `docs/experiments/E8_JOINT_FITTED_REFERENCE_BETA_TOPR_DENSE_RESULT.md`.
- Provenance and gate limitations: `docs/experiments/E8_JOINT_FITTED_REFERENCE_BETA_TOPR_DENSE_PROVENANCE_AUDIT.json`.
- Durable result package: `easonhuo/drpo-results@712c36c5e858182de4e93c48dfe917bd42198c67`.
- Predecessor package: `easonhuo/drpo-results@68ea4980ed9c8ebb79e02f7d2b40a7e2a8ee0461`.
