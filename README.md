# DRPO / SNA2C

Research codebase for studying far-field negative-gradient dynamics,
repulsive optimization, stability, and generalization in off-policy
policy optimization.

This repository contains controlled continuous and categorical
environments, external-validity experiments, experiment configurations,
tests, result manifests, and the research handoff document.

## Source of Truth

Before modifying code, designing an experiment, or running an experiment,
read the following files in order:

1. [`AGENTS.md`](AGENTS.md)
2. Section 0 of [`docs/handoff.md`](docs/handoff.md)
3. [`experiments/registry.yaml`](experiments/registry.yaml), when populated

`docs/handoff.md` is the unique research master document.

It defines:

* locked scientific conclusions;
* terminology overrides;
* experiment responsibilities;
* frozen variables and seeds;
* execution gates and ordering;
* convergence and terminal-audit requirements;
* the current status of every experiment.

This README is only a repository entry point. If this README conflicts
with `docs/handoff.md`, the handoff takes precedence.

## Research Scope

The project studies how negative-advantage updates behave under repeated
off-policy optimization.

The central research questions are:

1. Where do anomalously large negative gradients come from?
2. When do far-field negative gradients causally transmit into policy
   drift, support contraction, and task-performance collapse?
3. When can controlled negative gradients improve performance beyond a
   positive-only imitation ceiling?
4. How can global or selective control preserve useful negative
   information while preventing runaway dynamics?
5. How do the continuous Gaussian and categorical policy cases relate
   through policy-relative remoteness or surprisal?

## Experiment Responsibilities

The following environments must not be conflated.

### Product-manifold mechanism experiments

These experiments identify the source of far-field gradient
amplification while separating sample quality from policy-relative
distance.

They answer:

> Where do large negative gradients come from?

They do not by themselves establish causal policy collapse.

### Nonlinear Gaussian causal experiments

These experiments use targeted near/far interventions to determine
whether anomalously large far-field negative gradients causally transmit
into drift and collapse.

They answer:

> Do far-field negative gradients cause the observed instability in the
> controlled environment?

### C-U1 and D-U1

* `C-U1` is the primary controlled continuous contextual-bandit
  environment.
* `D-U1` is the primary controlled categorical contextual-bandit
  environment.

These environments provide controlled mechanism identification and
ground truth.

### Hopper and Countdown

* Hopper/D4RL provides continuous-control external validity under real
  offline data and a learned critic.
* Countdown/Qwen provides categorical or sequence-model external
  validity under shared Transformer parameters.

External-validity experiments do not replace controlled causal
identification.

## Terminology

C-U1 training states and test states are independently sampled from the
same state distribution.

Current C-U1 results may be described as:

* held-out-context generalization;
* unseen-context generalization;
* generalization to unseen states;
* 同分布未见状态泛化.

Current C-U1 results must not be described as:

* OOD generalization;
* distribution-shift generalization;
* out-of-distribution state generalization.

OOD terminology is allowed only after a separate explicit
distribution-shift protocol has been registered and executed.

Reports must also distinguish:

1. task-performance collapse;
2. support or variance-boundary events;
3. NaN/Inf numerical collapse.

## Current Status

The active state of the project is maintained in `docs/handoff.md`.

At the current recorded checkpoint:

* the unified C-U1 E1–E4 one-click script has been reconstructed;
* Python syntax checks have passed;
* a CPU development smoke run has passed;
* the formal 4096-train-state / 4096-test-state multi-seed run has not
  yet been completed;
* smoke tests and static checks are not formal experimental results;
* D-U1/E6 remains gated according to the handoff;
* Hopper and Countdown remain external-validity experiments and follow
  the execution order in the handoff.

Do not infer a formal result from the existence of code or from a
successful smoke test.

## Repository Layout

```text
drpo/
├── AGENTS.md
├── README.md
├── pyproject.toml
│
├── docs/
│   └── handoff.md
│
├── experiments/
│   └── registry.yaml
│
├── configs/
│   └── hopper_single_seed.yaml
│
├── scripts/
│   └── inspect_dataset.py
│
├── src/
│   └── drpo/
│       ├── __init__.py
│       ├── config.py
│       ├── datasets.py
│       ├── models.py
│       ├── seeding.py
│       ├── drpo_cu1_e1_e4_oneclick.py
│       └── README_DRPO_CU1_ONECLICK.md
│
├── tests/
│   ├── test_config.py
│   └── test_models.py
│
├── data/
└── outputs/
```

The repository will continue to evolve. The actual checked-in tree and
`docs/handoff.md` override this illustrative layout.

## Installation

Python 3.10 or newer is required.

```bash
git clone <YOUR-REPOSITORY-URL>
cd drpo

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Core dependencies are declared in `pyproject.toml`.

## Basic Validation

Run unit tests:

```bash
pytest -q
```

Run static lint checks:

```bash
ruff check .
```

Run both before reporting a coding task as complete.

Passing these checks does not constitute a formal experiment.

## C-U1 E1–E4 One-Click Runner

The current unified continuous runner is:

```text
src/drpo/drpo_cu1_e1_e4_oneclick.py
```

Run it from the repository root with:

```bash
python src/drpo/drpo_cu1_e1_e4_oneclick.py
```

The runner is designed to execute the registered C-U1 audit and E1–E4
pipeline without requiring users to edit source code or manually set
hyperparameters.

Read the detailed runner documentation first:

```text
src/drpo/README_DRPO_CU1_ONECLICK.md
```

The runner records configuration, environment audits, per-seed
trajectories, summaries, plots, reference regressions, and completion
status.

Before treating any output as a formal result, verify:

* the active Git commit SHA;
* the registered experiment protocol;
* the exact seeds;
* the data scale;
* the convergence or runaway criteria;
* the terminal-state audit;
* that the run completed without silently skipping required seeds.

## Experiment Registration

Every formal experiment must be registered before it starts.

The registration should record at least:

```yaml
id: C-U1-E4-v1
claim: >
  The scientific claim tested by this experiment.
environment: C-U1
code_entry: src/drpo/drpo_cu1_e1_e4_oneclick.py
config: null
development_seeds: []
held_out_seeds: []
metrics: []
controls: []
stopping_criteria: null
terminal_audit: required
status: not_run
code_commit: null
result_path: null
notes: null
```

A formal experiment must be consistent with both:

* `docs/handoff.md`;
* `experiments/registry.yaml`.

Do not silently change frozen seeds, thresholds, data geometry,
comparison methods, or stopping criteria.

## Allowed Result Statuses

Use only the project-approved result statuses:

* `analytically_proven`;
* `long_run_validated`;
* `finite_step_validated`;
* `pilot`;
* `not_run`;
* `rejected`;
* `superseded`.

Static inspection, unit tests, smoke tests, and short debugging runs must
not be upgraded into formal multi-seed results.

## Result and Provenance Requirements

Formal results should preserve:

* the Git commit SHA;
* complete configuration;
* seeds;
* raw per-step trajectories;
* per-seed summaries;
* aggregate summaries;
* plots and tables;
* failed runs;
* environment and dependency metadata;
* stopping and convergence diagnostics;
* terminal checkpoints;
* best-validation checkpoints when applicable.

When a prior conclusion is corrected, preserve the historical record and
document:

1. the old statement;
2. the identified problem;
3. the new evidence;
4. the replacement conclusion.

Do not destructively delete historical experiments or provenance.

## Method-Comparison Discipline

Do not assume in advance that any of the following is superior:

* Distance control;
* exponential tapering;
* global negative scaling;
* SBRC;
* Hybrid control;
* positive-only training.

Where applicable, method comparisons should use:

* matched negative-gradient budgets;
* paired seeds;
* identical starting checkpoints;
* shared datasets;
* long-run or convergence checks;
* terminal-state audits;
* mechanism diagnostics;
* held-out-context task metrics.

A best-checkpoint result must not be used to conceal poor terminal
dynamics.

## Data and Large Artifacts

The repository is intended to track code, configurations, documentation,
tests, and compact summaries.

Do not commit:

* model checkpoints;
* large HDF5 datasets;
* raw Qwen weights;
* large replay buffers;
* multi-gigabyte logs;
* credentials, API keys, or access tokens;
* private or proprietary production data.

The existing `.gitignore` excludes common dataset, checkpoint, run, and
output formats.

Large artifacts should remain on local or remote storage. The experiment
registry should record their external paths and compact summary files.

## ChatGPT Project Workflow

The intended workflow is:

```text
GitHub main
    ↓
ChatGPT Project reads AGENTS.md, handoff, registry, and relevant code
    ↓
ChatGPT prepares complete modified files and a unified patch
    ↓
The user applies and tests the patch locally
    ↓
The user commits and pushes the verified changes
    ↓
Future sessions read the updated GitHub main branch
```

When direct GitHub write access is unavailable, a code-change delivery
should include:

* complete modified files;
* a unified diff patch;
* `BASE_COMMIT.txt`;
* `CHANGE_SUMMARY.md`;
* `APPLY_AND_TEST.sh`, or equivalent application and test commands.

No session should claim that code was committed or pushed unless the
operation actually succeeded.

## Development Workflow

Before starting work:

```bash
git checkout main
git pull --ff-only origin main
git status
git rev-parse HEAD
```

After making changes:

```bash
pytest -q
ruff check .
git diff
git status
```

After review:

```bash
git add <changed-files>
git commit -m "<descriptive commit message>"
git push origin main
```

For larger or riskier changes, use a feature branch and review the diff
before merging.

## Governance

The governing rules are defined in `AGENTS.md` and `docs/handoff.md`.

The central rules are:

* document before experiment;
* preserve historical evidence;
* do not modify frozen protocols without approval;
* distinguish controlled mechanism experiments from external validity;
* distinguish smoke tests from formal results;
* perform terminal audits for dynamics, collapse, and method rankings;
* bind formal results to a Git commit SHA;
* state uncertainty explicitly.

## Repository Visibility

Before committing any research material, confirm that its visibility is
appropriate for the repository.

Do not publish confidential data, company-internal material, private
credentials, or unpublished artifacts that are not approved for public
release.
