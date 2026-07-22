# Scope: EXT-C-E8-ORACLE-OFFLINE-V2-JOINT-FITTED-REFERENCE-TOPR-0.5B-01

## Claim and current state

Implement, validate, and freeze the code identity for a **Joint Fitted-Reference TOPR** candidate on the existing E8 oracle-offline V2 bank. The candidate uses a policy LoRA and a jointly fitted branch-balanced reference LoRA on one frozen backbone.

Current state: `dev_code_first_unregistered`, formal RunSpec prepared, not run. This scope does not register or launch the pilot and does not claim canonical TOPR reproduction.

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

The repository owner explicitly authorized creation of the exact RunSpec path above in the active conversation on 2026-07-22. The RunSpec is a frozen task contract only: it remains `registration.mode: deferred`, does not change result status, and does not authorize claim, launch, or merge by itself.

## Frozen implementation responsibilities

- Reuse `Cell`, the existing common profile mechanism, `train_cell`, the runtime, evaluation, checkpoint, and aggregation paths.
- Add one `joint_fitted_reference_topr` profile with two paired development cells.
- Use one frozen backbone and two identically initialized LoRA adapters named `default` and `reference`.
- Fit the reference to equal positive/negative branch mass over the frozen bank.
- Use detached `exp(min(sum_logpi-sum_logmu,0))` negative weights with full completion sums.
- Keep the task loss on the existing mean completion-token log-probability scale.
- Evaluate only `default`; preserve both adapters in local checkpoints.
- Preserve all existing EXP, reciprocal, and AsymRE behavior.
- Bind the formal RunSpec to implementation commit `05bb65f33594e667d0e38d4c5799373bf526262b`.
- Require automatic text-first result delivery to `easonhuo/drpo-results` branch `ingest/e8`.

## Excluded scope

- handoff, registry, schema-v3 delta, or formal-channel mutation;
- claiming or launching the RunSpec before registration and the required real Qwen/CUDA liveness gate;
- experiment execution, GPU pilot, result aggregation, or result closure in this update;
- canonical frozen-behavior TOPR claim;
- new data generation or logged behavior-policy reconstruction;
- new Python files, workflows, dependencies, services, or training frameworks;
- changes to the bank, seeds, evaluation split, task horizon, thresholds, or historical results outside the unregistered candidate config;
- merge to `main` without repository-owner approval.

## Required gates before pilot launch

- exact implementation commit remains resolvable and protected paths remain unchanged;
- repository exact-head CI remains green after the RunSpec update;
- pilot registration is completed through the normal schema-v3 code-first transaction;
- two-step real Qwen/PEFT/CUDA dual-adapter liveness passes and remains non-scientific evidence;
- initial full-sequence ratio maximum absolute value is at most `1e-5`;
- both adapters receive finite nonzero gradient and update norms;
- checkpoint inventory records both adapters and policy evaluation uses only `default`;
- the RunSpec is explicitly claimed only after the preceding gates;
- fixed 1200 steps is not treated as convergence or steady state;
- task performance, valid-expression or structure diagnostics, and NaN/Inf numerical failure remain separate;
- no result-status promotion or method-ranking claim before terminal audit and durable result closure.
