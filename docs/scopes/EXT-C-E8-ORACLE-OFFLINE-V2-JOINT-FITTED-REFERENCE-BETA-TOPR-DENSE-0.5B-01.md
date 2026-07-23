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
- `docs/experiments/E8_JOINT_FITTED_REFERENCE_BETA_TOPR_DENSE_RESULT.md`
- `docs/experiments/E8_JOINT_FITTED_REFERENCE_BETA_TOPR_DENSE_PROVENANCE_AUDIT.json`
- `docs/development_workflow_optimization/E8_TOPR_DENSE_FIXED_PROFILE_REACTIVATION_20260723.md`
- `runspecs/ready/E8_JOINT_FITTED_REFERENCE_BETA_TOPR_DENSE_20260723_01.yaml`
- this scope file.

No new Python path is authorized or needed.

## Runtime boundary

The generic E8 config-driven runtime remains suspended. This scope authorizes only the exact dense TOPR fixed profile and exact shell entrypoint above. It does not authorize arbitrary new E8 configs, dynamically inferred profiles, experiment-matrix replay, M0 publication, or any other suspended runtime surface.

The runtime-slot provenance key is frozen as:

`execution.runtime_scope: GOV-RUNTIME-E8-GPU-SLOT-HOTFIX-01`

This required metadata repairs `RUNTIME_SLOTS.json` generation only. It does not change beta values, seeds, training, data, or evaluation. The metadata hotfix itself is engineering evidence only, not liveness or scientific evidence.

The completed run recorded an execution allocation of eight visible GPUs, two runtime slots per GPU, sixteen total slots, and one full wave. That allocation is recorded in the provenance audit as a wall-clock scheduling fact. It is not promoted by this closure into a new canonical resource requirement. Per the user's earlier explicit instruction, the separate local eight-GPU scheduling patch is not synchronized in this closure; the reviewed fixed-profile implementation already merged through PR #261 and its required runtime-scope repair already merged through PR #262.

## Result-closure authorization

The user explicitly approved closing the TOPR tuning line, depositing the completed result, and merging the reviewed closure to `main`.

The closure must preserve the following evidence boundaries:

- result status remains `pilot`;
- all 16 cells, fixed late-window metrics, terminal metrics, and supplementary best checkpoints remain visible;
- task-performance degradation, valid-expression/structure degradation, and NaN/Inf numerical failure remain separate;
- 1200 steps are not convergence, saturation, or steady-state evidence;
- no statistical-significance or formal cross-method ranking claim is allowed;
- this method remains Joint Fitted-Reference beta-TOPR, not canonical frozen-behavior TOPR;
- Countdown remains an external-validity environment only.

Scientific exploration on this beta line is closed. No further dense beta sweep is required. For the later budget-matched comparison against the main method, `beta=0.5` is frozen as the validation-selected comparison configuration because it has the highest dense-scan late-window Pass@8 mean while remaining inside the stable high-validity plateau. This is a baseline-selection rule, not a claim that beta=0.5 is a universally or significantly optimal TOPR parameter. The parameter must not be changed after observing the main-model comparison or any future test result.

## Provenance boundary

The durable result is bound to `drpo-results` commit `712c36c5e858182de4e93c48dfe917bd42198c67`.

The run reports clean local source commit `3733ee28cc1517b67ad235afa10f2e855f2dde33`, but that commit is not currently resolvable from the authoritative remote repository. The closure must therefore:

- record the unresolved commit honestly;
- preserve the protected source SHA-256 inventory embedded in every cell summary;
- record the result-package manifest and artifact hashes;
- record that `CHECKPOINT_RELOAD_GATE.json` was generated and hashed but excluded from the compact deposited package;
- avoid claiming direct-commit provenance resolution unless the exact source commit is later recovered.

This closure archives a provenance-limited pilot result and does not directly edit `docs/handoff.md` or `experiments/registry.yaml`. Promotion to a direct-commit-resolved formal result would require recovering the exact run source and using the authoritative schema-v3 delta route.

## Excluded scope

- changing the beta grid, seeds, data, optimizer, learning rates, objective, jointly fitted reference target, update ratio, horizon, evaluation cadence, or checkpoint reporting policy;
- synchronizing the separate local eight-GPU scheduling patch in this closure;
- changing the old result package or rewriting its unresolved source provenance;
- claiming canonical TOPR, convergence, saturation, universal best beta, significance, or formal ranking;
- accessing `test.jsonl`;
- opening another TOPR beta scan as part of this closure;
- reactivating the generic config-driven E8 runtime;
- merging to `main` without exact-head CI success.

## Required result reporting

- all 16 cells must be present;
- fixed late-window and terminal metrics must be reported;
- best checkpoints remain supplementary;
- task-performance degradation, valid-expression or structure degradation, and NaN/Inf failure remain separate;
- 1200 steps must not be described as convergence or saturation;
- results must be terminal-audited, packaged, delivered durably, and linked from the repository closure record.
