# Scope: EXT-C-E8-ORACLE-OFFLINE-V2-JOINT-FITTED-REFERENCE-TOPR-0.5B-01

## Claim and current state

Implement, validate, and freeze the code identity for a **Joint Fitted-Reference TOPR** candidate on the existing E8 oracle-offline V2 bank. The candidate uses a policy LoRA and a jointly fitted branch-balanced reference LoRA on one frozen backbone.

Current state: `dev_code_first_unregistered`, not run. This scope does not register or launch the pilot and does not claim canonical TOPR reproduction.

## Authorized development paths

- `src/drpo/countdown_e8_alpha1_highc_scan_common.py`
- `src/drpo/countdown_e8_alpha1_c_scan_trainer.py`
- `scripts/run_countdown_e8_oracle_offline_v2_alpha1_highc_scan_auto.py`
- `tests/test_countdown_e8_oracle_offline_v2_alpha1_highc_scan.py`
- `configs/countdown_e8_oracle_offline_v2_joint_fitted_reference_topr_0p5b.yaml`
- `docs/experiments/E8_JOINT_FITTED_REFERENCE_TOPR_PROTOCOL.md`
- this scope file

No new Python path is authorized or needed.

## Frozen implementation responsibilities

- Reuse `Cell`, the existing common profile mechanism, `train_cell`, the runtime, evaluation, checkpoint, and aggregation paths.
- Add one `joint_fitted_reference_topr` profile with two paired development cells.
- Use one frozen backbone and two identically initialized LoRA adapters named `default` and `reference`.
- Fit the reference to equal positive/negative branch mass over the frozen bank.
- Use detached `exp(min(sum_logpi-sum_logmu,0))` negative weights with full completion sums.
- Keep the task loss on the existing mean completion-token log-probability scale.
- Evaluate only `default`; preserve both adapters in local checkpoints.
- Preserve all existing EXP, reciprocal, and AsymRE behavior.

## Excluded scope

- handoff, registry, schema-v3 delta, RunSpec, or formal-channel mutation;
- experiment launch, GPU pilot, result aggregation, or result closure;
- canonical frozen-behavior TOPR claim;
- new data generation or logged behavior-policy reconstruction;
- new Python files, workflows, dependencies, services, or training frameworks;
- changes to the bank, seeds, evaluation split, task horizon, thresholds, or historical results outside the unregistered candidate config;
- merge to `main` without repository-owner approval.

## Required gates before implementation freeze

- exact current-`main` parent and a dedicated dev branch;
- reviewed 7-file Narrow-M0 after-image transaction, otherwise fall back to the normal direct route without padding scope;
- Python compilation;
- focused tests for profile construction, branch-balanced reference loss, summed-sequence ratio weights, initial-ratio identity, and config drift rejection;
- existing E8 scan regression tests;
- exact-head repository CI and diff review;
- no experiment execution and no result-status promotion.
