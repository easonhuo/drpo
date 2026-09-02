# E8 config-driven sweep repair invariants

Status: latest correctness repair applied / focused final-head checks passed / independent review pending / not a scientific result.

Scope: harden and simplify the `eight_task_coldstart_lambda_v1` orchestration refactor introduced by PR #340 without changing any currently tracked historical scientific config, the canonical paper trainer/loss, or the interpretation of existing results.

## Locked repair invariants

1. `experiment_id` is metadata selected by the YAML config. The shell runner and bootstrap must not maintain an independent default identity. An explicit `E8_COLDSTART_EXPERIMENT_ID` override is permitted only as a fail-closed equality assertion against the config value. The bootstrap/runner identity reader must accept the same safe ID scalar whether it is written as a plain, single-quoted, or double-quoted top-level YAML scalar, with an optional YAML comment after the scalar; YAML presentation must not create a second identity rule.
2. Historical experiment IDs remain immutable scientific/provenance identities. The existing 208-cell cold-start, 199-cell lambda-completion, and 140-cell lambda-curve-completion IDs must reject changes to their historical config identity.
3. A generic new cold-start-family ID is allowed only for the `eight_task_coldstart_lambda_v1` profile. New generic IDs are not authorized for the RHO or DENSE profiles.
4. Countdown remains bound to the locked paper runtime for this implementation family. A transfer-task uncontrolled endpoint is represented explicitly as `METHOD_GLOBAL` / `lambda=0`, not as an exponential cell with a non-positive lambda. Positive Exp lambdas remain strictly positive.
5. Canonical engineering liveness is independent of the scientific sweep grid. The locked round-1 paper runtime itself defines the two-update smoke representative as `c=0.693147181`, `seed_offset=4000`. Recovery and resume gates must reject a stored liveness manifest that does not match that engineering identity. This smoke is not a scientific result.
6. A zero-cell cold-start sweep is invalid and must fail during config validation rather than later in wave construction. Empty task lambda grids mean zero scientific cells for that task; Countdown additionally requires empty scheduled Countdown seeds when its scientific grid is empty.
7. Aggregate, task-result, audit, recovery, and nominal-wave geometry must be derived from the configured/built cells; task-performance events, support/structure diagnostics, and NaN/Inf events remain reported separately.
8. Run identity must not silently inherit the historical base cold-start Run ID when another config is selected. The default config `configs/e8_multitask_exp_coldstart.yaml` may retain its frozen default Run ID `E8_MULTITASK_EXP_COLDSTART_20260820_02`. Every non-default config must provide an explicit `E8_COLDSTART_RUN_ID`; otherwise the runner must fail before constructing guard, output, recovery, or package paths. A Run ID must match `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`.
9. Bootstrap self-test source identity must be explicit. The historical hard-coded `refs/pull/309/head` fallback is not an authority for PR #340 or future same-family sweeps and must not remain. Full mode may retain `refs/heads/main` as its default target; self-test mode must require an explicit validated branch or PR-head ref.
10. Temporary CI trigger files, temporary repair scripts, and temporary workflow overrides must not remain in the final PR tree.

## 2026-09-01 authority-boundary override

This section supersedes any older wording in this repair document that required a **new** cold-start-family experiment to repeat historical scientific scalar values in Python. It does not mutate any historical config and does not itself authorize a scientific run with changed values.

1. For a new `eight_task_coldstart_lambda_v1` experiment ID, the reviewed, repository-tracked YAML config is the scientific experiment specification. Scientific values such as optimizer horizon, optimizer scalars, split/init seeds, sampling temperature/top-p, LoRA dimensions, task runtime lengths, evaluation sizes, and task grids are not re-approved by comparing them to historical constants in the training module.
2. Python validation for a new cold-start-family ID is limited to: schema/type validity; finite/range checks needed for meaningful execution; implementation capability; config/matrix self-consistency; duplicate/zero-cell rejection; profile/experiment-ID scope; and engineering safety. Protocol/review decides whether a scientific change is approved; config records the approved value; runtime code executes it.
3. Historical IDs remain immutable. Their protection should be compact config identity (canonical config path plus immutable content identity) rather than field-by-field duplication of the historical config inside Python. The 208/199/140 historical matrices must remain unchanged under their canonical configs.
4. RHO remains bound to `EXT-C-E8-MULTITASK-EXP-TUNING-01`; DENSE remains bound to `EXT-C-E8-MULTITASK-EXP-LAMBDA-DENSE-01`. Only the cold-start profile may use a new well-formed generic experiment ID.
5. Scientific-kernel implementation capability remains code-bound. A config cannot name an unimplemented method or silently replace the canonical paper trainer. Canonical source/dispatch integrity is an implementation-integrity gate, not a requirement that a new experiment copy every old scientific scalar.
6. The runner must make its tracked-config claim true with an actual Git tracked-file check after repo-relative normalization. Exact commit/clean checkout, path-safe Run IDs, identity-checked resume, fail-closed recovery, and the no-automatic-scientific-mutation OOM policy remain engineering safety responsibilities.
7. The repository owner explicitly approved exactly two new Python paths for this refactor: `src/drpo/e8_experiment_config.py`, responsible for E8 config interpretation/schema/self-consistency/profile-scope/historical-identity checks without implementing the scientific loss/trainer; and `scripts/preflight_e8_multitask_config.py`, responsible for standalone fail-fast reporting of the resolved tracked config and expanded experiment plan without model training/evaluation. No additional new Python path is approved by this record.
8. Preflight must run before expensive setup/GPU work and report at least experiment ID, profile, config path/content identity, active tasks, cell/wave counts, and the effective training/evaluation values consumed by the experiment. It must exit nonzero on invalid/untracked config and must never claim a scientific result.
9. A config-representative resource liveness gate is explicitly deferred. The existing canonical Countdown liveness remains a kernel/dispatch engineering smoke, not proof that the heaviest configured cell fits resources.
10. Replacing grep-based provenance-source discovery with explicit metadata is also deferred unless separately approved; this refactor must not expand scope merely to redesign provenance enumeration.

## 2026-09-01 runtime-propagation closure

This section closes a gap found after the authority-boundary simplification. The previous synthetic acceptance test proved only that changed values survived config validation; it did **not** prove that the byte-locked canonical trainer actually consumed those values. That weaker interpretation is superseded by the rules below.

1. The acceptance invariant is end-to-end: `reviewed tracked YAML -> effective runtime specification -> materialized canonical runtime inputs -> canonical trainer/evaluator`. A value that is accepted by config validation may not be silently ignored downstream.
2. For a new cold-start-family experiment ID, every accepted scientific field must satisfy exactly one of two outcomes: (a) the effective runtime consumed by the canonical worker/trainer/evaluator contains and uses that configured value; or (b) validation rejects the value with an explicit implementation-capability or self-consistency error before launch. "Accepted by validator but old default used at runtime" is a merge-blocking failure.
3. The byte-locked canonical paper source files and their Git-blob identities remain unchanged. This repair must not edit the historical arena, taper common/runtime, trainer, historical base config, or historical grid configs merely to make a new experiment configurable. The paper loss/objective, all-unique-negative consumer, taper formula, cell coefficient, and canonical dispatch remain implementation-identity gates.
4. The wrapper may materialize per-task derived runtime copies of the historical base/grid YAMLs. Those derived files are execution inputs, not replacements for the historical canonical source files. Their identities and the resolved effective runtime must be recorded in canonical-input/cell provenance.
5. Effective runtime propagation must cover the values already declared configurable by the authority-boundary override: initialization seed; optimizer horizon; micro-batch and gradient accumulation; learning rate, weight decay, warmup ratio, gradient clipping, and evaluation cadence; LoRA rank/alpha/dropout; task model lengths; evaluation batch/budgets/pass-k interface; sampling temperature/top-p; and evaluation generation seed. Split sizes/hash seed continue to be consumed by the wrapper split implementation.
6. Where the historical implementation contains a literal interface default rather than a parameter (for example fresh-LoRA dimensions, AdamW weight decay, or sampled-generation temperature/top-p), the wrapper may use a temporary process-local adapter around that interface. Such an adapter must be restored after the cell, must not reimplement or alter loss mathematics, and must be covered by a regression test showing the canonical call receives the configured value.
7. Historical experiment IDs remain byte-for-byte config identities and therefore continue to execute their historical effective runtime values. This propagation repair authorizes no scientific parameter change to those configs and must not retroactively reinterpret their results.
8. Any legacy grid validator that protects historical paper-grid science may be adapted only for a derived runtime grid by proving that the derived file differs from its canonical source solely in explicitly propagated runtime fields. All non-runtime grid drift must still fail closed.
9. Preflight must report the **resolved effective runtime** that will be materialized for each active task, not merely echo raw YAML sections. The effective-runtime resolver used by preflight must be the same resolver used to materialize canonical runtime inputs.
10. Regression tests must inspect the materialized base/grid YAMLs and the legacy-interface adapter inputs. A test that only mutates an in-memory config, calls `validate_config`, and asserts that the dictionary still contains the new value is insufficient and must not be treated as runtime-propagation evidence.
11. Canonical paper-profile activation must use the immutable canonical source grid. A derived runtime grid is passed to the worker/trainer only after canonical activation and only inside the runtime bridge described above. This prevents the historical strict grid validator from rejecting a reviewed runtime-only change before the bridge can verify and consume it.

## Required regression checks after the authority-boundary simplification

The final branch must:

- preserve the canonical 208/199/140 historical cell matrices and wave geometry;
- reject historical IDs when their canonical config identity is not used;
- reject a generic experiment ID under RHO or DENSE and accept a well-formed generic ID only under cold-start;
- demonstrate with a synthetic new cold-start config that reviewed scientific scalars differ from historical values without editing the canonical paper sources, and verify those scalars in the effective runtime and materialized canonical inputs;
- verify process-local adapters deliver configured LoRA dimensions, optimizer weight decay, and sampled-generation temperature/top-p to the legacy interfaces rather than silently using historical literals;
- verify canonical profile activation uses the immutable source grid before a derived runtime grid enters the process-local bridge;
- reject duplicate/invalid lambda cells, malformed seeds/booleans/numbers where their type is required for execution, unsupported methods, inconsistent `expected_cells`, zero-cell matrices, and accepted-but-unconsumable runtime combinations;
- preserve optional transfer Global endpoints without treating `lambda=0` as Exp;
- preserve the exact `c=0.693147181`, seed-4000 Countdown engineering liveness identity and reject stale/mislabeled liveness manifests during recovery;
- verify that selected configs are actually Git-tracked and repository-relative before launch;
- verify runner/bootstrap config identity resolution and mismatch rejection for supported YAML scalar presentation;
- verify safe/non-inherited Run IDs and explicit bootstrap self-test target refs;
- verify preflight prints the resolved effective experiment/runtime plan and performs no model/GPU work;
- run Python compilation for changed Python files, shell syntax checks for changed shell launchers, the focused E8 test suite, `git diff --check`, and repository PR gates.

## Final runtime-propagation implementation evidence

- Base main commit: `6ed153d5ad7361a4e52348610b86b51b71e25e47`.
- Runtime-propagation implementation commit: `27762da8e52f0948b776d2ff1efeec1c121b21df`.
- Activation-order and subtractive-cleanup implementation commit: `68c5ce06ceb961bbaf83df7169c3a31815b18844`.
- GitHub Actions runtime-propagation run `33486287587` completed successfully before the activation-order follow-up; the subsequent activation-order review found that its green tests were not sufficient evidence of executable derived-grid activation.
- Final focused engineering run `33487047873`, job `99789460451`, completed successfully after that follow-up. It self-removed the temporary workflow and repair script from the branch tree.
- Complete focused `tests/test_e8_multitask_p0.py`: **63 passed in 4.95s**.
- `python -m py_compile` passed for `src/drpo/e8_experiment_config.py`, `scripts/preflight_e8_multitask_config.py`, `src/drpo/e8_multitask_exp_tuning.py`, and `tests/test_e8_multitask_p0.py`.
- `bash -n` passed for the cold-start runner, bootstrap, and lambda-completion launcher.
- Ruff passed for `src/drpo/e8_experiment_config.py`, `scripts/preflight_e8_multitask_config.py`, and the focused E8 tests; `ruff format` was applied to the touched core/test files. This is not a claim that unrelated pre-existing warnings in the scientific core were repaired.
- `git diff --check` and the staged diff check both passed.
- Generic cold-start runtime propagation is checked at four layers: effective-runtime resolution, per-task base/grid materialization, process-local legacy-interface bridging, and canonical source-grid activation before the derived grid reaches the trainer.
- The byte-locked canonical arena/common/runtime/trainer files were not edited; the wrapper changes interfaces and runtime YAML materialization without reimplementing the loss.
- The old cold-start field-by-field hard-lock block in `e8_multitask_exp_tuning.validate_config` was deleted after delegation to `e8_experiment_config`; RHO/DENSE retain their historical exact validation path. The activation/cleanup commit itself was **57 additions / 641 deletions** including removal of its temporary workflow/script.
- Exact base-to-implementation-head diff at `68c5ce06...`: seven files, `+2283/-612` (net `+1671`). `src/drpo/e8_multitask_exp_tuning.py` is `+603/-514`, net **+89** relative to main; there is no line-count quota and this measured value is reported only as evidence that the validator-heavy core was subtractively reduced.
- RHO/DENSE generic IDs remain rejected; historical cold-start-family IDs remain protected by canonical config path/content identity; canonical 208/199/140 matrix regression tests remain in the focused suite.
- No model training, GPU sweep, task-performance result, support/variance-boundary result, convergence result, significance result, or method ranking was produced by these engineering checks.
- Normal repository PR gates must run on the final documentation head after this evidence update. Independent review remains mandatory before merge.

## Authority-boundary implementation evidence

- Base main commit: `6ed153d5ad7361a4e52348610b86b51b71e25e47`.
- Simplified implementation commit: `f61b5ef5ceba8e555a4345de5c218fb0393ee902`.
- GitHub Actions one-shot engineering run `33470365713`, job `99738720935`, completed successfully and self-removed its temporary workflow/script before the implementation commit was pushed.
- Complete focused `tests/test_e8_multitask_p0.py`: `60 passed in 5.70s`.
- `python -m py_compile` passed for `src/drpo/e8_experiment_config.py`, `scripts/preflight_e8_multitask_config.py`, `src/drpo/e8_multitask_exp_tuning.py`, and `tests/test_e8_multitask_p0.py`.
- `bash -n` passed for the cold-start runner, bootstrap, and lambda-completion launcher.
- Ruff passed for the new config/preflight files and the focused E8 tests; this is not a claim that every pre-existing warning in the scientific core were repaired.
- `git diff --check` passed.
- The synthetic generic cold-start test changes optimizer horizon, stochastic-evaluation values, split/init seeds, LoRA rank, and a task runtime length without editing the training core, and validation accepts the reviewed-config values.
- RHO/DENSE generic IDs are rejected; historical cold-start-family IDs are protected by canonical config path/content identity.
- The runner now performs a real Git-tracked config check and invokes standalone preflight before expensive model/GPU setup.
- GitHub compare confirms the PR head is descended from the base (`status=ahead`, `merge_base=6ed153d5ad7361a4e52348610b86b51b71e25e47`). An intermediate Code Change Budget message claiming no shared ancestor was therefore a shallow-fetch gate artifact, not a disconnected branch; the repository gate itself has not been modified under this protocol.
- Diff at `f61b5ef5...` relative to base: `+1581/-242` across seven files (net `+1339`). The scientific core `src/drpo/e8_multitask_exp_tuning.py` is `+298/-144`, down substantially from the pre-simplification validator-heavy version; config interpretation is isolated in the approved `src/drpo/e8_experiment_config.py`.
- The repository owner's explicit approval for the two new Python paths and the large/structural config-authority refactor is preserved in PR #340 discussion using `GOV-NEW-PYTHON-FILE-ORAL-APPROVAL-02` fields.

**Runtime-propagation correction:** the bullet above describing the synthetic test is retained as historical engineering evidence only. It proves validator acceptance, not canonical-runtime consumption. It must not be cited as satisfying the end-to-end config-authority invariant. Runtime-propagation evidence is superseded by the stronger final evidence section above.

This evidence is engineering-only. No scientific experiment, GPU training sweep, task-performance result, support/variance-boundary result, convergence/significance claim, or method-ranking claim was produced. Final-head repository gates and independent review remain mandatory before merge.

## Prior engineering closure evidence (pre-simplification)

- Base main commit for PR #340: `6ed153d5ad7361a4e52348610b86b51b71e25e47`.
- Tenth-pass implementation commit: `e528e8beb3b7885ebffa2dd9ac7e7210f9f39fdf`.
- Tenth-pass GitHub Actions run `33461027051` completed successfully.
- Complete `tests/test_e8_multitask_p0.py`: `74 passed, 4 skipped`.
- `python -m py_compile` passed for `src/drpo/e8_multitask_exp_tuning.py` and `tests/test_e8_multitask_p0.py`.
- `bash -n` passed for the cold-start runner, bootstrap, and lambda-completion launcher.
- `git diff --check` passed.
- One-shot repair workflows, repair scripts, and trigger files were removed by the validated cleanup commit.

That evidence applies only to the pre-simplification implementation and must not be substituted for the final-head evidence above. No scientific experiment was executed by these checks; no task-performance, support/variance-boundary, convergence, significance, or method-ranking claim is produced here. Merge remains blocked until the repository's independent reviewer gate is satisfied and final-head repository gates are reviewed.

No smoke test, static check, focused unit test, engineering liveness result, or CI regression result produced under this protocol is a scientific result.

## 2026-09-01 second adversarial audit closure

A second end-to-end audit found additional accepted-but-unconsumable and provenance gaps. These rules are part of the same authority-boundary repair and supersede any weaker interpretation above.

1. A generic cold-start-family experiment ID must be genuinely new with respect to the execution identities known to this runner. In particular it may not reuse the P0, RHO, or DENSE experiment IDs. The three historical cold-start-family IDs remain protected by exact canonical config identity.
2. Redundant configuration fields may not silently diverge from the field that actually drives execution. While the current schema retains them, `sweep.tuning_seed` must equal the transfer Exp seed `sweep.task_transfer_seed_offset`; `sweep.countdown_sentinel_coefficients` must equal the configured Countdown `task_lambda` sequence; and `execution.expected_waves` must equal the nominal wave count derived from the expanded cell matrix and configured capacity.
3. Split sizes are configurable only within the data volume actually supplied by the locked source pipelines. The canonical P0 bank contains exactly 6000 rows per task, so `p0_train_rows + p0_validation_rows + p0_test_rows` must equal 6000 and the training partition must be non-empty. The canonical cold-start Countdown path forbids wrapper subsampling and supplies exactly 6000 training rows and 500 validation rows, so a config that requests different Countdown counts is not consumable by this implementation and must fail preflight rather than fail later in `prepare`.
4. An active task's configured evaluation prompt budget may not exceed its configured validation partition. Otherwise the legacy evaluator would silently slice to the available rows while preflight reported the larger requested budget. Such configurations must fail before launch. Countdown remains subject to its canonical equal greedy/pass-k budget interface.
5. The canonical trainer implements warmup as at least one optimizer step. Therefore a zero `training.warmup_ratio` cannot represent a true zero-warmup experiment through this implementation and must be rejected rather than reported as faithfully consumed.
6. Standalone config preflight must execute before full dependency installation, model download, GPU capacity checks, or other expensive setup on a fresh runtime. A minimal bootstrap environment may install only the lightweight dependencies needed to run the existing preflight; this does not authorize a second validator or duplicated config rules.
7. Formal guarded execution provenance must include every Python path that participates in config interpretation or launch gating, including `src/drpo/e8_experiment_config.py` and `scripts/preflight_e8_multitask_config.py`. Delivery-preflight provenance is not a substitute for a complete raw guarded-run source manifest.
8. Regression tests must cover all findings above. Passing the earlier 63-test suite is historical engineering evidence only and does not close this second-audit set.

## 2026-09-01 third adversarial audit closure

This section records the final design decisions reached after inspecting the failed V3 job and the remaining config/recovery identity path. It supersedes any implication that the V3 repair was partially applied.

### V3 non-execution correction

1. The V3 workflow failed before its repair script executed because the temporary repair code raised `SyntaxError: unterminated triple-quoted string literal`. Therefore V3 changed no validator, runner, test, provenance path, or cleanup state and produced no validated implementation commit. The current pre-repair branch state must be treated as a stranded repair state, not a half-completed V3 repair.
2. Earlier CI evidence in this document remains historical engineering evidence for the exact commits on which it ran. It must not be cited as final-head evidence for the third-audit closure.
3. No further V4/V5-style one-shot self-repair workflow is authorized. The remaining repair must be made as normal repository changes, with the temporary repair workflows/scripts removed from the final tree.

### Closed reviewed-config schema

4. Every **new generic** `eight_task_coldstart_lambda_v1` reviewed config is recursively closed-world. Every top-level section and nested mapping has an explicit implemented key set. A key not known to the schema is rejected before launch even if its value would otherwise survive YAML parsing and enter the config hash. There is no open-ended scientific or metadata section.
5. Closed-world applies to metadata as well as scientific/runtime sections. `reporting`, grid provenance, canonical-integrity declarations, and execution metadata may contain only explicitly defined fields. A historical-only field such as `historical_curve_anchor` is not thereby made part of the generic schema merely because it exists in an immutable predecessor config.
6. Historical cold-start-family IDs remain grandfathered only by their exact canonical path/content identity. The closed generic schema must not retroactively reject or reinterpret a frozen historical YAML blob.
7. The runtime-generated engineering self-test config is not a reviewed scientific config. Its additional `engineering_self_test` mapping is allowed only on the internal derived-config path and itself has an exact closed key set; an external reviewed config may not use that internal-only section.

### Strict YAML authority and loader boundary

8. There is one shared strict YAML parse boundary for this E8 config family. Duplicate mapping keys at any nesting level are an error; parser last-key-wins behavior is forbidden for reviewed or internal derived configs.
9. Parsing, schema validation, semantic/capability validation, and historical-ID identity validation must not be independently reimplemented in preflight and runtime. Preflight and runtime use the same config module.
10. An external CLI launch must resolve the selected config inside the repository and require it to be Git-tracked before execution. The direct Python module entrypoint is subject to the same tracked-config rule as the shell runner and standalone preflight.
11. A runtime-generated internal config, including `engineering_self_test_config.yaml`, is parsed through the same strict schema/semantic validator but is not required to be Git-tracked. Internal parsing must be an explicit API; it may not weaken the external launch path.

### Explicit generic scientific matrix semantics

12. Historical compatibility defaults for omitted `countdown_include_positive_only` and `include_global_endpoint` remain permitted only for immutable historical configs. A new generic reviewed config must explicitly specify both fields because they alter the scientific cell matrix.
13. For a new generic config, `sweep.task_lambda` means the **active scientific grid for this run**, not predecessor/sentinel metadata. If `countdown_seed_offsets` is empty, `task_lambda.countdown` and `countdown_sentinel_coefficients` must both be empty and `countdown_include_positive_only` must be false. Historical configs that preserve inactive Countdown metadata remain valid only through exact frozen identity.
14. For a transfer task, an empty `task_lambda[task]` means zero scientific cells for that task. `task_grid_provenance` and the currently retained derived `task_grid_hashes` entry may still describe that inactive task, but they must not be interpreted as an active grid.
15. `task_runtime` remains a task-interface/capability declaration and may remain present for an inactive task. Cell activity is determined by the sweep expansion, not by deleting task-interface metadata.

### Unified execution identity

16. `RUN_ID` is an instance/directory name and is not a scientific or execution identity. Reuse is permitted only by an explicit `execution_identity_hash` computed from one normalized payload.
17. The normalized execution identity contains at least: experiment ID; reviewed config repository path and Git blob identity; semantic/effective config hash; reviewed source commit; verified base-model repository/revision plus one content-derived model snapshot hash; a runtime fingerprint; backend identity (`real_canonical` versus `engineering_placeholder`); and run class (`formal` versus `pilot`).
18. The runtime fingerprint is intentionally narrow and execution-relevant rather than a full `pip freeze`: Python, PyTorch, PyTorch CUDA runtime, `transformers`, `peft`, `accelerate`, and `numpy` versions are sufficient for this family unless a later protocol revision documents another dependency.
19. Model snapshot identity is computed from the actual resolved model snapshot content, including the foundation-model weight files, once after setup/download and verified again when an existing setup is reused. Individual scientific cells carry the resulting snapshot hash; they must not re-read and re-hash the full foundation weights for every cell.
20. The same execution identity hash must propagate through setup state, source provenance, cell manifests, recovery state/snapshots/imports, scheduler/final manifests, and reusable-complete-attempt checks. A successful cell or complete attempt from another execution identity is stale and must not be hard-linked, resumed, or returned as current output.
21. The engineering placeholder backend must have its own explicit execution identity and may never satisfy a real-canonical execution identity. Likewise a pilot execution may not satisfy a formal execution identity solely because `RUN_ID`, source commit, and config hash happen to match.

### Source authority, recovery, and final-artifact closure

22. Authoritative target-ref resolution occurs **before** execution identity construction or reuse decisions on every bootstrap invocation. A previously completed bootstrap may not substitute its local checkout `HEAD` for a fresh resolution of the configured authoritative remote ref. If the authoritative ref advanced, the old setup/attempt identity is stale.
23. Recovery checkpoint packaging must include the reviewed config and every code path that interprets or launch-gates it, including `src/drpo/e8_experiment_config.py` and `scripts/preflight_e8_multitask_config.py`. Raw guarded-run provenance and recovery-checkpoint provenance must both be complete; delivery preflight is not a substitute.
24. Finalization and package verification must prove that scheduler, aggregate, terminal audit, and final manifests all bind the same current execution identity. A terminal package that is merely internally self-consistent under an old config/runtime identity is not current-run evidence.
25. The second-audit fixes remain mandatory in the same normal repair: reject P0/RHO/DENSE ID reuse for generic cold-start, enforce redundant-field consistency, validate split/evaluation capacity, reject zero warmup that the canonical trainer cannot express, execute preflight before expensive setup, and include complete formal provenance.
26. The five stranded one-shot repair artifacts (`.github/workflows/e8-second-audit-fix-once.yml`, `.github/workflows/e8-second-audit-fix-v2-once.yml`, `.github/workflows/e8-second-audit-fix-v3-once.yml`, `scripts/.tmp_e8_second_audit_fix.sh`, and `scripts/.tmp_e8_second_audit_fix_v3.sh`) must be removed as part of the normal repair. Their deletion is cleanup of temporary engineering artifacts, not deletion of research history; the audit history is retained in this protocol and Git history.

### Third-audit regression gates

27. Regression coverage must additionally reject duplicate YAML keys, unknown generic keys, internal-only engineering fields on reviewed configs, untracked direct Python CLI configs, omitted generic matrix booleans, inactive generic Countdown metadata, mismatched execution identities, stale completed-attempt reuse, stale recovery imports, pilot/formal backend confusion, model snapshot drift, and completed-bootstrap remote-ref drift.
28. Final-head validation must include focused E8 tests, the full repository pytest suite, Ruff on touched Python paths, shell syntax checks for touched shell launchers, `git diff --check`, the repository PR gates, and an independent reviewer. Static checks, smoke/liveness, and engineering self-tests remain non-scientific evidence only.
29. No scientific experiment may be launched merely to validate this repair. Scientific execution remains blocked until the engineering implementation, terminal identity audit, and independent review are complete.


## 2026-09-02 gate-liquidation override

The repository owner clarified that adding or strengthening a gate is itself a governance decision and requires explicit prior approval. General requests to fix, audit, harden, simplify, or finish an implementation are not authorization to invent additional blockers.

For PR #340, second/third-audit additions that were not explicitly owner-approved are therefore superseded and removed from the implementation after-image. In particular, this liquidation removes strict duplicate-YAML/closed-world-schema rejection, model/runtime execution-identity gating, stale-execution/recovery identity gating, extra data-volume/capacity rejection, zero-warmup strengthening, inactive-Countdown explicitness rules, exact Countdown liveness coefficient/seed identity, mandatory non-default RUN_ID and self-test target-ref inputs, and runtime rejection of reporting-policy declarations. Historical audit text above remains preserved as provenance, not as current authority.

The retained implementation scope is limited to the owner-approved config-driven behavior and the two approved Python responsibilities: schema/type/range parsing needed by the consumer, implementation capability, matrix self-consistency, profile/experiment-ID scope, compact historical config identity, repository-path/tracked-config safety, preflight plan reporting, and propagation of reviewed config values into the existing canonical runtime without reimplementing the scientific loss/trainer. The pre-existing repository review/merge workflow remains governed by its prior main-branch policy.


## 2026-09-02 final correctness after-image correction

This section supersedes the remaining inconsistent implementation details found after gate liquidation. It does not change scientific variables, historical configs, the canonical paper trainer/loss, or any scientific result.

- Formal raw guarded-run provenance includes both approved config-authority Python paths: `src/drpo/e8_experiment_config.py` and `scripts/preflight_e8_multitask_config.py`; delivery-preflight provenance is no longer the only place that records them.
- Residual runtime rejection of reporting/selection-policy declarations is removed. The actual canonical selection/reporting behavior remains implemented by the existing runtime; this change only removes the unapproved second policy lock.
- The post-audit hard-coded Countdown liveness coefficient `0.693147181` is reverted to the pre-audit/main runner value `0.916290732`; no new exact liveness identity is introduced.
- `e8_multitask_exp_tuning.sweep_profile()` delegates to the approved config-authority module rather than re-parsing the same field.
- This remains engineering-only evidence. No model training, GPU sweep, task-performance result, support/variance-boundary result, numerical-collapse result, convergence result, significance result, or method ranking is produced.

Latest correctness validation run: `33603556843`. Focused E8: `65 passed in 5.20s`; Countdown subset: `15 passed, 12 deselected in 4.34s`. Python compilation, Ruff, shell syntax, three real config preflights, handoff authority, governance-stage validation, `git diff --check`, and the #340 no-diff check for `.github/workflows/code-change-budget.yml` all passed in the same run.
