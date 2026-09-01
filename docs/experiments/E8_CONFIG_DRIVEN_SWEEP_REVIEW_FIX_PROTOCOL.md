# E8 config-driven sweep repair invariants

Status: authority-boundary simplification implemented / final-head repository gates pending / independent review pending / not a scientific result.

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

## Required regression checks after the authority-boundary simplification

The final branch must:

- preserve the canonical 208/199/140 historical cell matrices and wave geometry;
- reject historical IDs when their canonical config identity is not used;
- reject a generic experiment ID under RHO or DENSE and accept a well-formed generic ID only under cold-start;
- demonstrate with a synthetic new cold-start config that a reviewed scientific scalar can differ from the historical value without editing `e8_multitask_exp_tuning.py`, while malformed types/ranges still fail;
- reject duplicate/invalid lambda cells, malformed seeds/booleans/numbers where their type is required for execution, unsupported methods, inconsistent `expected_cells`, and zero-cell matrices;
- preserve optional transfer Global endpoints without treating `lambda=0` as Exp;
- preserve the exact `c=0.693147181`, seed-4000 Countdown engineering liveness identity and reject stale/mislabeled liveness manifests during recovery;
- verify that selected configs are actually Git-tracked and repository-relative before launch;
- verify runner/bootstrap config identity resolution and mismatch rejection for supported YAML scalar presentation;
- verify safe/non-inherited Run IDs and explicit bootstrap self-test target refs;
- verify preflight prints the effective experiment plan and performs no model/GPU work;
- run Python compilation for changed Python files, shell syntax checks for changed shell launchers, the focused E8 test suite, `git diff --check`, and repository PR gates.

## Authority-boundary implementation evidence

- Base main commit: `6ed153d5ad7361a4e52348610b86b51b71e25e47`.
- Simplified implementation commit: `f61b5ef5ceba8e555a4345de5c218fb0393ee902`.
- GitHub Actions one-shot engineering run `33470365713`, job `99738720935`, completed successfully and self-removed its temporary workflow/script before the implementation commit was pushed.
- Complete focused `tests/test_e8_multitask_p0.py`: `60 passed in 5.70s`.
- `python -m py_compile` passed for `src/drpo/e8_experiment_config.py`, `scripts/preflight_e8_multitask_config.py`, `src/drpo/e8_multitask_exp_tuning.py`, and `tests/test_e8_multitask_p0.py`.
- `bash -n` passed for the cold-start runner, bootstrap, and lambda-completion launcher.
- Ruff passed for the new config/preflight files and the focused E8 tests; this is not a claim that every pre-existing warning in the scientific core was repaired.
- `git diff --check` passed.
- The synthetic generic cold-start test changes optimizer horizon, stochastic-evaluation values, split/init seeds, LoRA rank, and a task runtime length without editing the training core, and validation accepts the reviewed-config values.
- RHO/DENSE generic IDs are rejected; historical cold-start-family IDs are protected by canonical config path/content identity.
- The runner now performs a real Git-tracked config check and invokes standalone preflight before expensive model/GPU setup.
- GitHub compare confirms the PR head is descended from the base (`status=ahead`, `merge_base=6ed153d5ad7361a4e52348610b86b51b71e25e47`). An intermediate Code Change Budget message claiming no shared ancestor was therefore a shallow-fetch gate artifact, not a disconnected branch; the repository gate itself has not been modified under this protocol.
- Diff at `f61b5ef5...` relative to base: `+1581/-242` across seven files (net `+1339`). The scientific core `src/drpo/e8_multitask_exp_tuning.py` is `+298/-144`, down substantially from the pre-simplification validator-heavy version; config interpretation is isolated in the approved `src/drpo/e8_experiment_config.py`.
- The repository owner's explicit approval for the two new Python paths and the large/structural config-authority refactor is preserved in PR #340 discussion using `GOV-NEW-PYTHON-FILE-ORAL-APPROVAL-02` fields.

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