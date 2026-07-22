# Scope: EXT-C-E8-ORACLE-OFFLINE-V2-JOINT-FITTED-REFERENCE-TOPR-0.5B-01

## Claim and current state

Implement, validate, and freeze an eight-point **Joint Fitted-Reference beta-TOPR response curve** on the existing E8 oracle-offline V2 bank. The candidate uses a policy LoRA and a jointly fitted branch-balanced reference LoRA on one frozen backbone.

Current state: `dev_code_first_unregistered`, formal RunSpec prepared but not claimed, not run. This scope does not register or launch the pilot and does not claim canonical TOPR reproduction, a best beta, convergence, steady state, or formal method ranking.

## Authorized development paths

- `src/drpo/countdown_e8_alpha1_highc_scan_common.py`
- `src/drpo/countdown_e8_alpha1_c_scan_trainer.py`
- `scripts/run_countdown_e8_oracle_offline_v2_alpha1_highc_scan_auto.py`
- `tests/test_countdown_e8_oracle_offline_v2_alpha1_highc_scan.py`
- `configs/countdown_e8_oracle_offline_v2_joint_fitted_reference_topr_0p5b.yaml`
- `docs/experiments/E8_JOINT_FITTED_REFERENCE_TOPR_PROTOCOL.md`
- `runspecs/ready/E8_JOINT_FITTED_REFERENCE_TOPR_20260722_01.yaml`
- this scope file

No new Python path is authorized or needed.

The repository owner explicitly authorized the eight-point beta curve in the active conversation on 2026-07-22. The frozen grid is:

\[
\beta\in\{0,\ 0.25,\ 0.5,\ 0.75,\ 1,\ 1.5,\ 2,\ 4\},
\]

paired with seed offsets `4000` and `5000`, for 16 cells total.

## Frozen implementation responsibilities

- Reuse `Cell`, the existing common profile mechanism, `train_cell`, the runtime, evaluation, checkpoint, aggregation, launcher, and RunSpec paths.
- Encode beta as the TOPR-family cell coefficient and include beta in every cell name and run identity.
- Use detached `exp(beta*min(sum_logpi-sum_logmu,0))` negative weights with full completion sums.
- Treat `beta=1` as the original TOPR ratio-rule anchor with a fitted reference.
- Treat `beta=0` as the no-ratio-taper boundary control, not as canonical TOPR.
- Treat all remaining beta values as tempered Joint Fitted-Reference beta-TOPR variants.
- Use one frozen backbone and two identically initialized LoRA adapters named `default` and `reference`.
- Fit the reference to equal positive/negative branch mass over the frozen bank.
- Keep policy and reference learning rates, schedulers, and 1:1 update frequency unchanged.
- Keep the task loss on the existing mean completion-token log-probability scale.
- Evaluate only `default`; preserve both adapters in local checkpoints.
- Preserve all existing EXP, reciprocal, and AsymRE behavior.
- Keep the bank, seed offsets, 1200-step horizon, evaluation split, evaluation cadence, and thresholds unchanged.
- Bind the formal RunSpec to the final scientific implementation commit.
- Require automatic text-first result delivery to `easonhuo/drpo-results` branch `ingest/e8`.

## Excluded scope

- handoff, registry, schema-v3 delta, or formal-channel mutation;
- learning-rate, reference-target, update-frequency, optimizer, horizon, seed, data, or evaluation tuning;
- claiming or launching the RunSpec before registration and the required real Qwen/CUDA liveness gate;
- experiment execution, GPU pilot, result aggregation, or result closure in this update;
- canonical frozen-behavior TOPR claim;
- best-beta selection or formal method-ranking claim;
- new data generation or logged behavior-policy reconstruction;
- new Python files, workflows, dependencies, services, or training frameworks;
- merge to `main` without repository-owner approval.

## Required gates before pilot launch

- the exact scientific implementation commit remains resolvable and protected paths remain unchanged;
- repository exact-head CI remains green after the RunSpec update;
- pilot registration is completed through the normal schema-v3 code-first transaction;
- two-step real Qwen/PEFT/CUDA dual-adapter liveness at `beta=1` passes and remains non-scientific evidence;
- initial full-sequence ratio maximum absolute value is at most `1e-5`;
- both adapters receive finite nonzero gradient and update norms;
- checkpoint inventory records both adapters and policy evaluation uses only `default`;
- the RunSpec is explicitly claimed only after the preceding gates;
- all 8 beta points and 16 cells are present in the terminal audit;
- fixed 1200 steps is not treated as convergence or steady state;
- task performance, valid-expression or structure diagnostics, and NaN/Inf numerical failure remain separate;
- no result-status promotion, best-beta claim, or method-ranking claim before terminal audit and durable result closure.
