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

## Follow-up protocol-expression closure (2026-09-08)

The owner selected the auditable representation target **one successor config = the complete multi-method, multi-seed baseline matrix**, rather than two mirrored single-seed configs. This is a configuration/orchestration capability extension only; it does not freeze any scientific hyperparameter value or initialization estimand.

Accordingly, this same development scope additionally authorizes:

- plural transfer method seeds while retaining byte-for-byte compatibility with historical singular `task_transfer_seed_offset` configs;
- a config-level baseline matrix container that composes AsymRE, Joint Fitted-Reference beta-TOPR, and canonical DPO cells in one experiment identity;
- method-aware liveness, dispatch, aggregation, resume identity, and plan generation for that composed matrix;
- synthetic regression coverage for the target arithmetic `8 tasks x 2 seeds x (5 + 3 + 3) = 176` cells, where synthetic test values are capability fixtures and **must not** be interpreted as the future frozen protocol grid.

This follow-up still does **not** authorize selecting the five/three/three production hyperparameters, imposing a new AsymRE upper-domain gate, declaring TOPR beta `1.0` historical-canonical, choosing DPO fresh-LoRA versus shared-SFT for the final estimand, changing formal artifact status semantics, editing `docs/handoff.md` / `experiments/registry.yaml`, or launching any scientific run.

## Post-review P1-P4 hardening (2026-09-09)

The owner explicitly approved the four post-review hardening items below on the same active development branch. They remain capability/engineering changes and do not authorize a scientific run or a final protocol choice.

### P1 — baseline-matrix tail regression coverage

Add behavior-level regression coverage for the composed matrix after cell expansion, including method-aware liveness, mixed-method dynamic scheduling, task-local result materialization, mixed-method aggregation, and recovery/reuse identity. Synthetic manifests may exercise the full 176-cell geometry because they execute no model, optimizer, GPU, or scientific metric. This coverage records already intended orchestration semantics; it must not introduce a new scientific acceptance threshold, result-selection rule, or independent mandatory launch gate.

### P2 — conditional shared-SFT DPO identity/provenance binding

When `dpo.initialization_mode=shared_sft_adapter` is selected, capability code must support an exact config-defined adapter contract rather than accepting an adapter solely because rank/alpha/dropout happen to match. The contract may bind the logical base model and revision, the adapter's recorded base-model field, LoRA target modules, `modules_to_save`, bias, the complete adapter-config SHA-256, exact adapter-weight file/SHA-256, and an exact provenance JSON file/SHA-256 plus expected source fields. These identities must enter DPO cell provenance/resume identity.

This approval does **not** name or freeze a production SFT adapter, adapter SHA, source run, checkpoint, or final DPO initialization. Synthetic tests may use synthetic hashes/provenance only. If the final experiment chooses fresh LoRA, the shared-SFT contract is irrelevant to that run. Most importantly, this hardening must not alter the historical Countdown canonical DPO scientific algorithm: chosen/rejected construction, full-completion summed log probabilities, policy/reference margin, beta placement, sigmoid/softplus loss, prompt-balanced rejected-pair aggregation, exact frozen reference, zero label smoothing, and optimizer semantics remain unchanged.

### P3 — single authority for method vocabulary

The multitask runner's AsymRE, TOPR, DPO, Exponential, and baseline-matrix method constants should alias `e8_experiment_config.py` rather than independently repeating the same protocol strings. This is maintenance-only and changes no method identity or scientific behavior.

### P4 — canonical AsymRE/TOPR grid provenance consistency, non-gating

Centralize the runtime provenance record for the extra canonical AsymRE/TOPR grid files so path plus actual SHA-256 are reported consistently by single-method and matrix aggregation. This remains **non-gating**: no new expected-Git-blob mismatch rejection is authorized here. Turning those grid files into a new immutable expected-blob hard gate would require a separate explicit governance approval.
