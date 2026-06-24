# AGENTS.md

## Project identity

This repository contains the DRPO / SNA2C far-field negative-gradient dynamics research project.

The repository is the source of truth for:

* code;
* experiment configurations;
* experiment registry;
* result manifests;
* research handoff documents.

Chat history is not a source of truth.

## Mandatory startup reading

Before changing code or running an experiment, read:

1. `docs/HANDOFF.md`, especially Section 0;
2. `docs/MASTER_STATUS.md`;
3. `experiments/registry.yaml`;
4. the nearest directory-specific `AGENTS.md`, if present.

Summarize the active claim, experiment status, and relevant constraints before implementation.

## Locked research boundaries

Do not conflate:

* product-manifold gradient-source experiments;
* nonlinear Gaussian causal-collapse experiments;
* C-U1 controlled continuous experiments;
* D-U1 controlled categorical experiments;
* Hopper/D4RL external validation;
* Countdown/Transformer external validation.

Product-manifold experiments identify where large negative gradients originate.

Nonlinear Gaussian intervention experiments identify whether far-field negative gradients causally transmit into drift and collapse.

External experiments do not replace controlled causal identification.

## Execution order

Current priority:

1. C-U1 stable extrapolation and generalization;
2. categorical bandit;
3. Hopper and Countdown external validation;
4. paper rewrite and broader method evaluation.

Do not start a lower-priority experiment merely because its code is easier to run.

## Document-before-experiment rule

Before starting a new experiment, the following must be registered:

* experiment ID;
* claim being tested;
* environment and dataset;
* compared methods;
* controls;
* metrics;
* seeds;
* stopping or convergence criteria;
* expected output paths;
* result status.

Do not launch an unregistered formal experiment.

## Result-status vocabulary

Only use:

* analytically proven;
* long-run validated;
* finite-step validated;
* pilot;
* not run;
* rejected or superseded.

Do not upgrade a result status without evidence.

## Coding requirements

* Preserve old experiments and provenance.
* Do not destructively delete historical code or results.
* Add replacement records when correcting an old conclusion.
* Save configs, seeds, raw curves, summaries, logs, and failures.
* Bind each formal result to a Git commit SHA.
* Run relevant tests before reporting completion.
* Never fabricate successful runs when hardware or dependencies are unavailable.

## Method-comparison discipline

Do not assume that Distance, Exp, Global scaling, SBRC, or Hybrid is superior.

Use:

* matched negative-gradient budgets where relevant;
* paired seeds;
* long-run or convergence checks;
* ID and OOD task metrics;
* mechanism diagnostics in addition to final reward.

## Completion report

For every completed coding task, report:

* files changed;
* commands run;
* tests run;
* experiment IDs affected;
* result files created;
* remaining uncertainties;
* current Git status and commit SHA.
