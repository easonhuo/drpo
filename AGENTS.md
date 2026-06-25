# AGENTS.md

## Project identity

This repository contains the DRPO / SNA2C far-field negative-gradient dynamics research project.

The repository is the source of truth for:

* source code;
* experiment configurations;
* experiment registry;
* result manifests;
* research handoff documents.

Chat history alone is not a source of truth.

## Mandatory startup protocol

Before changing code, designing a new experiment, or running an experiment:

1. Read `docs/handoff.md`.
2. Read Section 0 of `docs/handoff.md` first and inherit all locked conclusions, terminology rules, execution gates, and experiment priorities.
3. Read `experiments/registry.yaml` if it exists.
4. Read the nearest directory-specific `AGENTS.md`, if present.
5. Inspect the current Git branch and commit SHA when the environment provides Git access.
6. Summarize the active experiment, its current status, relevant constraints, and remaining uncertainties before implementation.

`docs/handoff.md` is the unique research master document. Do not introduce a second competing master-status document.

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

C-U1 and D-U1 provide controlled mechanism identification and ground truth.

Hopper and Countdown provide external validity and do not replace controlled causal identification.

## Terminology discipline

Follow the newest terminology override in `docs/handoff.md`.

In particular:

* C-U1 train and test states are independently sampled from the same state distribution.
* C-U1 results may be described as held-out-context generalization, unseen-context generalization, or generalization to unseen states.
* Do not describe current C-U1 results as OOD generalization or distribution-shift generalization.
* Use OOD terminology only when an explicit distribution-shift protocol has been registered and executed.
* Distinguish task-performance collapse from numerical collapse.
* Distinguish support or variance-boundary events from NaN/Inf numerical failure.

## Execution order and gates

Always follow the latest execution order and gates recorded in `docs/handoff.md`.

Do not start a lower-priority experiment merely because it is easier to run.

Do not run an experiment that the handoff marks as paused, unapproved, or awaiting protocol review.

## Document-before-experiment rule

Before starting a new formal experiment, register:

* experiment ID;
* claim being tested;
* environment and dataset;
* code entry point;
* compared methods;
* controls;
* metrics;
* development seeds;
* held-out seeds;
* stopping or convergence criteria;
* expected output paths;
* result status.

The registration must appear in `docs/handoff.md` and, when applicable, in `experiments/registry.yaml`.

Do not launch an unregistered formal experiment.

## Allowed result statuses

Use only the following statuses:

* analytically proven / 已解析证明;
* long-run validated / 已长期验证;
* finite-step validated / 有限训练步数验证;
* pilot;
* not run / 尚未运行;
* rejected or superseded / 已否定或已替换.

Do not upgrade a result status without supporting evidence.

Static inspection, unit tests, and smoke tests do not constitute a formal multi-seed experimental result.

## Coding and provenance requirements

* Preserve historical experiments and provenance.
* Do not destructively delete historical code, results, or conclusions.
* When correcting an error, record the old statement, the problem, the new evidence, and the replacement conclusion.
* Save configurations, seeds, raw curves, summaries, logs, and failed runs.
* Bind every formal result to a Git commit SHA.
* Run relevant tests before reporting completion.
* Never claim that an experiment ran successfully when hardware, dependencies, or data were unavailable.
* Do not silently change frozen variables, seeds, thresholds, data geometry, or convergence criteria.
* Do not treat a fixed training horizon as convergence without the terminal-state audit required by the handoff.

## Formal experiment supervision and durable artifacts

Formal experiments in ephemeral runtimes require active supervision and durable delivery.

* Do not launch a formal experiment as an unattended background process and then end the working turn.
* Use `scripts/run_experiment_guard_hardened.py` or an equivalent foreground supervisor that records a heartbeat, streams logs, captures exit status, preserves partial outputs, and packages success or failure.
* Treat `registered`, `running`, `raw_complete`, `terminal_audited`, `packaged`, `delivered`, and `applied_to_repository` as separate execution/evidence states.
* `raw_complete` is not a completed formal result. Do not claim completion until a verified durable package has been generated and delivered.
* Do not start the next formal experiment ID until the current experiment has been packaged and delivered. In particular, package E3 before starting E4.
* For runs expected to exceed 30 minutes, create a durable checkpoint artifact at least every five formal seeds or at another interval registered before launch.
* If a run, aggregation, plotting, or audit step fails, preserve the completed raw outputs, logs, traceback, source commit, and missing-output inventory in an `experiment-failed` package before repair or rerun.
* Files written only to an ephemeral path such as `/mnt/data` are not durable evidence. Chat messages and process counters are not evidence either.
* A final experiment package must contain raw outputs, aggregate results, logs, a run manifest, `RUN_COMPLETE.json`, a terminal audit, source provenance, checksums, and the repository update files required below.
* Follow `docs/formal_experiment_artifact_protocol.md` for package kinds, lifecycle semantics, stage boundaries, size policy, and canonical commands.

## Method-comparison discipline

Do not assume that Distance, Exp, Global scaling, SBRC, Hybrid, or any other method is superior.

Use, where relevant:

* matched negative-gradient budgets;
* paired seeds;
* long-run or convergence checks;
* held-out-context task metrics;
* explicit distribution-shift metrics only when separately registered;
* mechanism diagnostics in addition to final reward;
* terminal checkpoints in addition to best validation checkpoints.

## ChatGPT patch-delivery protocol

When direct GitHub write access is unavailable, provide one verified downloadable ZIP compatible with the local `drpo-update` workflow. It must contain:

* `update.patch`, a unified patch applicable with `git apply`;
* `BASE_COMMIT.txt`, containing only the full base commit SHA;
* `CHANGE_SUMMARY.md`;
* `TEST_COMMANDS.sh`, with executable non-placeholder commands;
* `modified_files/`, containing complete modified files with repository-relative paths;
* experiment artifacts and checksums when the task includes results.

Run `git apply --check update.patch` against the confirmed base whenever the environment permits. If it cannot be run, state that explicitly.

Never state or imply that changes were pushed to GitHub unless the push actually occurred.

## Completion report

A formal experiment is not complete merely because its process exited. It is complete only after required audit, packaging, verification, and durable delivery. Repository closure additionally requires an actual applied commit.

For every completed coding task, report:

* files changed;
* commands run;
* tests run;
* experiment IDs affected;
* result files created;
* result status;
* remaining uncertainties;
* current Git status and commit SHA, when available.

## Formal commit and artifact hardening

The following controls are mandatory for formal runs under `GOV-EXP-ARTIFACT-02`:

* Resolve the current commit with Git (`git ls-remote origin refs/heads/main` and `git rev-parse HEAD`), not a cached web commits page. If authoritative remote resolution is unavailable, report that limitation rather than inventing a current SHA.
* A formal experiment may start only from a clean worktree whose full commit SHA is recorded. A dirty worktree is permitted only for an explicitly labelled pilot with `--allow-dirty`, and its tracked, staged, and bounded untracked launch snapshot must be captured before process start.
* At process exit, re-check HEAD and worktree status. A changed HEAD or dirty formal worktree marks provenance as compromised and forces failed-run handling even if the child process returned zero.
* Main artifact ZIPs are first built as candidates, internally verified, and atomically published only after checksum, safe-path, base-commit, and `git apply --check` validation succeeds.
* Failed, checkpoint, and raw-complete recovery packages are lightweight evidence packages by default. Large model adapters, checkpoints, optimizer states, datasets, and caches are indexed rather than copied into the main ZIP.
* Necessary large checkpoints must be delivered as a separate sidecar with size, SHA-256, role, and persistence status recorded in the main manifest.
* Formal large-model experiments must pre-register an artifact budget and checkpoint-retention policy. Do not save foundation-model weights, redundant checkpoints, or optimizer state unless the registered recovery plan requires them.
* Packaging must never follow result-directory symbolic links. External links are rejected; internal links are recorded as references and their targets are packaged at most once.
* The default main-package hard limit is 25 MiB and the default single-file main-package limit is 10 MiB. Crossing either limit must trigger exclusion/sidecar handling or fail before final publication; a post-hoc warning is insufficient.
* The public guard, package, and verify entry points must all use the same hardened implementation and fail closed if that implementation is missing; silently falling back to a legacy protocol is prohibited.
* Before delivering a `drpo-update` ZIP, apply it in a fresh clean checkout of the confirmed base, preserve executable file modes, run the package's own `TEST_COMMANDS.sh`, run `git diff --check`, and withhold the final ZIP if any step fails.
* The fresh checkout must come from an authoritative Git clone/fetch or a GitHub source archive pinned to the full commit SHA. Never reconstruct a base tree from web snippets, parsed pages, copied files, or a hand-built repository. If the exact base cannot be acquired, do not label the package verified and do not deliver it as ready for blind application.
* Pre-delivery validation must exercise the same merge path as the user: verify `BASE_COMMIT.txt`, run `git apply --check`, apply the patch in the fresh checkout, execute `TEST_COMMANDS.sh`, and confirm the declared modified-file set and Git modes. The user-side `drpo-update` run is a final environment check, not the first integration test.
