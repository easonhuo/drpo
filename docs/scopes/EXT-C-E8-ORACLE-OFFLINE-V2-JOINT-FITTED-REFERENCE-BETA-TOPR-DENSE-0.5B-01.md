# Scope: EXT-C-E8-ORACLE-OFFLINE-V2-JOINT-FITTED-REFERENCE-BETA-TOPR-DENSE-0.5B-01

## Approved scientific change

Implement an exact fixed-profile, fixed-matrix successor that densifies the fitted-reference beta-TOPR curve near zero:

`beta = [0, 0.01, 0.02, 0.04, 0.08, 0.125, 0.25, 0.5]`

with paired seed offsets `4000,5000`, for 16 cells. The user explicitly approved this dense scan and requested a directly runnable implementation plus a RunSpec in the active conversation.

The sole scientific change is the beta grid. Learning rates, optimizers, schedulers, reference target, update frequency, data, seeds, 1200-step horizon, evaluation cadence, and held-out split remain frozen.

## Authorized paths

- `src/drpo/countdown_e8_alpha1_highc_scan_common.py`
- `src/drpo/countdown_e8_alpha1_c_scan_trainer.py`
- `scripts/run_countdown_e8_oracle_offline_v2_alpha1_highc_scan_auto.py`
- `scripts/run_countdown_e8_joint_fitted_reference_beta_topr_dense.sh`
- `configs/countdown_e8_oracle_offline_v2_joint_fitted_reference_beta_topr_dense_0p5b.yaml`
- `docs/experiments/E8_JOINT_FITTED_REFERENCE_BETA_TOPR_DENSE_PROTOCOL.md`
- `docs/development_workflow_optimization/E8_TOPR_DENSE_FIXED_PROFILE_REACTIVATION_20260723.md`
- `runspecs/ready/E8_JOINT_FITTED_REFERENCE_BETA_TOPR_DENSE_20260723_01.yaml`
- this scope file

No new Python path is authorized or needed.

## Runtime boundary

The generic E8 config-driven runtime remains suspended. This scope authorizes only the exact dense TOPR fixed profile and exact shell entrypoint above. It does not authorize arbitrary new E8 configs, dynamically inferred profiles, experiment-matrix replay, M0 publication, or any other suspended runtime surface.

The runtime-slot provenance key is frozen as:

`execution.runtime_scope: GOV-RUNTIME-E8-GPU-SLOT-HOTFIX-01`

This required metadata repairs `RUNTIME_SLOTS.json` generation only. It does not change beta values, seeds, GPU allocation, cells per GPU, wave count, training, data, or evaluation.

The full matrix is blocked until the exact implementation passes:

- exact-head repository CI;
- real Qwen/PEFT/CUDA two-step liveness at `beta=0.25`;
- initial policy/reference ratio tolerance;
- finite nonzero policy and reference gradient/update diagnostics;
- dual-adapter checkpoint inventory;
- fresh-process reload and finite forward checks for both adapters;
- normal schema-v3 pilot registration;
- explicit RunSpec claim.

## Excluded scope

- modifying `docs/handoff.md`, `experiments/registry.yaml`, or schema-v3 authority in the implementation commit;
- changing scientific variables other than beta;
- changing GPU count, cells per GPU, wave count, or other execution allocation in this metadata hotfix;
- changing the old result package or rewriting its unresolved source provenance;
- claiming canonical TOPR, convergence, saturation, a best beta, significance, or formal ranking;
- accessing `test.jsonl`;
- executing the scientific matrix in this implementation task;
- merging to `main` without explicit repository-owner approval.

## Required result reporting

- all 16 cells must be present;
- fixed late-window and terminal metrics must be reported;
- best checkpoints remain supplementary;
- task-performance degradation, valid-expression or structure degradation, and NaN/Inf failure remain separate;
- 1200 steps must not be described as convergence or saturation;
- results must be terminal-audited, packaged, and delivered durably before closure.
