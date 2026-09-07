# Development scope: E8 multitask baseline capabilities

## Status

Code-only development scope. No scientific experiment is launched by this change, and no final baseline-comparison experiment ID, hyperparameter grid, initialization choice, RunSpec, ranking, or result claim is frozen here.

Base repository state for this task: `easonhuo/drpo@8a21fd16e63ea96a0cde2a477353d24562624655` on `main`. This development branch is based on the already validated AsymRE capability commit `384c6141f330b8e440d9dae532d97b1115dc94e7`.

## Approved responsibility

Complete the existing E8 multitask cold-start orchestration so successor configs can select all three baseline families required for the final eight-task comparison:

- AsymRE, using the existing canonical implementation already integrated in the parent capability commit;
- Joint Fitted-Reference beta-TOPR, using the existing canonical TOPR implementation without reimplementing its objective in the multitask orchestration;
- canonical DPO, using the historical audited DPO implementation semantics and exposing initialization policy as configuration rather than freezing the final scientific estimand in this development change.

The multitask layer owns method/config/cell representation, planning, dispatch, reference lifecycle plumbing, manifest/provenance, resume identity, liveness, and regression tests. Scientific objectives must remain in the canonical method implementations or be ported with exact audited semantics when no canonical main-branch implementation exists.

## Allowed implementation files

- `src/drpo/e8_multitask_exp_tuning.py`
- `src/drpo/e8_experiment_config.py`
- existing canonical runtime/common files only if required to expose already-audited DPO functionality without changing its mathematics
- extensions to existing `tests/test_e8_multitask_p0.py`
- this development scope record
- temporary validation workflow files under `.github/workflows/`, provided they are non-gating and removed before final review

No new Python path is approved or required.

## Required design properties

1. Historical EXP cold-start configs and immutable identities continue to validate unchanged.
2. Existing AsymRE behavior remains unchanged and canonical/import-only.
3. TOPR dispatch uses the existing Joint Fitted-Reference beta-TOPR semantics, including detached fitted-reference ratio weighting and branch-balanced reference fitting; no TOPR objective is copied into the multitask orchestration.
4. DPO supports config-defined beta and a config-defined initialization policy. Capability may support both audited `cold_start` and `shared_sft` initialization modes, but this change must not choose which one belongs in the final 176-cell experiment.
5. DPO scientific semantics must match the audited historical canonical DPO implementation: frozen exact initial reference for the DPO stage; chosen oracle completion; unique verifier-wrong rejected completions; full-completion summed policy/reference log-probability margin; sigmoid/softplus DPO loss with zero label smoothing; prompt-balanced mean over unique negatives; zero initial pair-margin check.
6. Plan/cell identity records method-specific hyperparameters and DPO initialization explicitly enough to prevent collisions and support exact resume/provenance checks.
7. Task data, bank selection, verifier/evaluator adapters, training horizon, evaluation semantics, numerical-failure reporting, and terminal-audit responsibilities are unchanged unless a method's audited canonical lifecycle explicitly requires a reference or warm start.
8. Successor configs define method, hyperparameters, seed offsets, initialization mode where applicable, and expected cell count without experiment-specific grids hard-coded in Python.
9. Liveness becomes method-aware for AsymRE, TOPR, and DPO. Liveness/smoke/static checks remain engineering validation only and cannot be reported as scientific results.

## Explicit exclusions

This change must not:

- freeze the final five AsymRE `delta_v` values, three TOPR beta values, or three DPO beta values;
- choose the final DPO initialization estimand;
- choose the final experiment ID or claim wording;
- alter frozen seeds, data scale, negative-bank construction, task runtime overrides, convergence criteria, evaluation thresholds, or final experiment responsibility;
- start a smoke, pilot, or formal scientific run;
- claim any performance result or method ranking;
- modify `docs/handoff.md` or `experiments/registry.yaml` until a later protocol-freeze task explicitly authorizes it.

## Acceptance criteria

- Historical EXP and the existing AsymRE successor tests remain green.
- Synthetic config-driven TOPR and DPO successor configs expand exactly the requested cells and seeds without hard-coded final grids.
- TOPR cell identity/plan/manifest expose beta and method identity and dispatch to the existing canonical TOPR family.
- DPO cell identity/plan/manifest expose beta plus initialization mode and dispatch to audited canonical DPO semantics.
- Regression tests verify TOPR does not reimplement its loss in the multitask layer.
- Regression tests verify DPO pair-margin construction, frozen-reference behavior, prompt-balanced unique-negative aggregation, and zero initial pair-margin guard against the audited implementation contract.
- `py_compile`, focused pytest, diff checks, and changed-line lint pass before a Draft PR is opened.
