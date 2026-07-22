# GOV-E8-CONFIG-DRIVEN-ADAPTERS-01 Scope

## Identity

- parent claim: `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`
- exact base: `main@c323577eb170e260b7916b669b4a3e1d6bac2841`
- development branch: `dev/gov-experiment-matrix-e8-replayab-01`
- initial implementation head: `c4ec718ea426f91a4b616a83106706ced8b8e028`
- scientific result status: unchanged
- execution class: repository workflow engineering only

The repository owner authorized expanding the approved E8 config-driven pilot to
all current parameter structures and making adapter plus Replay comparison the
default requirement when a genuinely new E8 method structure is introduced.

## Goal

Remove concrete experiment-grid duplication from the current paper-aligned E8
runtime while preserving every scientific validator, cell identity, execution
gate, provenance rule, and reporting boundary.

## Authorized paths

- `src/drpo/countdown_e8_alpha1_highc_scan_runtime.py`
- `scripts/run_countdown_e8_oracle_offline_v2_alpha1_highc_scan_auto.py`
- `scripts/run_e8_asymre_matrix_replay_ab.sh`
- `docs/development_workflow_optimization/E8_CONFIG_DRIVEN_ADAPTER_CONTRACT.md`
- `docs/scopes/GOV-E8-CONFIG-DRIVEN-ADAPTERS-01.md`
- PR descriptions and review comments for PR #250 and its Replay evidence

No new Python path is authorized or required. The already approved
`src/drpo/experiment_matrix.py` keeps its original science-agnostic
responsibility unchanged.

## Required behavior

1. Historical EXP/Global/Positive-only, Reciprocal, and AsymRE configs must
   reproduce their exact point and cell manifests.
2. Grid config is the sole source for concrete points, seed offsets, and declared
   counts at runtime.
3. Existing profile entries may be used only as semantic templates for historical
   compatibility; their concrete arrays must be overwritten from config before
   validation or planning.
4. A new experiment ID using an existing parameter structure must reach the
   canonical runtime `plan` path without a Python profile edit.
5. Unknown or mixed structures fail closed with an instruction to add a reviewed
   adapter and Replay A/B.
6. Run identity continues to bind the matrix helper implementation.
7. The auto launcher must install the config-driven profile before resource
   planning, not only inside the child runtime.

## Replay matrix

Historical replay covers:

- paper-aligned EXP Round 1: 16 points / 32 cells;
- EXP c extension: 8 points / 16 cells;
- Reciprocal shape screen: 8 points / 16 cells;
- Reciprocal high-lambda extension: 8 points / 16 cells;
- Reciprocal-Quadratic dense curve: 16 points / 32 cells;
- AsymRE scan: 8 points / 16 cells;
- AsymRE boundary-dense scan: 8 points / 16 cells.

Candidate replay adds one unregistered experiment ID for each of the three
supported structures and requires runtime-plan success without a Python profile
entry.

## Exclusions

This scope does not authorize changes to:

- E8 trainer or loss formulas;
- existing experiment configs, frozen variables, seeds, data, bank, budgets,
  thresholds, horizons, evaluation, or results;
- `docs/handoff.md`, `experiments/registry.yaml`, or schema-v3 authority;
- GitHub workflow files, merge policy, or automatic merge;
- E7 code or experiments;
- CUDA liveness or a full scientific run.

## Delivery gates

- local Python compile for modified Python files;
- shell syntax for the Replay entrypoint;
- isolated three-structure adapter checks;
- exact-head full pytest, Ruff, handoff authority, formal execution channel,
  governance, and evidence locator;
- full-family Replay execution when a stable full checkout executor is
  available.

Smoke, plan, static checks, and Replay are engineering evidence only.
