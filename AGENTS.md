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

## Default repository development route

For coding and repository-document changes, the normal path is the connected GitHub App:

1. Resolve the current `main` SHA through the GitHub repository API.
2. Create or update a dedicated `dev/<claim>` branch from that exact SHA.
3. Commit only the approved scope to the dev branch.
4. Open a Draft PR, run the applicable GitHub Actions checks, review the diff and results, and merge only after explicit user approval.

A local `git clone` or shell network path is an optional implementation convenience, not a prerequisite for this route. Failure of shell DNS, `git clone`, or `git fetch` does not mean direct GitHub write access is unavailable and must not trigger a bundle request while the GitHub App can still perform the required branch, file, PR, Actions, and merge operations.

Do not create dummy branches, commits, files, or PRs merely to test permissions at every session start. Read repository permissions and the current `main` SHA first; perform write operations only for an actual approved task.

If the direct GitHub route exposes a defect, repair or iterate that route rather than silently falling back to the retired package workflow. The offline package path below is an emergency fallback only.

## Temporary code-first pilot-registration transition

Until `GOV-DEV-PILOT-REGISTRATION-FASTPATH-01` is merged and separately activated on `main`, follow `docs/development_workflow_transitions/GOV-DEV-PILOT-REGISTRATION-FASTPATH-TRANSITION-01.md` for any new or modified authoritative E7, E8, or other scientific-pilot registration.

A code-first pilot may proceed through scientific implementation, command-contract validation, already-authorized liveness, and implementation-SHA freeze. After the implementation SHA is frozen, do not construct a new authoritative registry/handoff update through an ad hoc sequence of per-file writes, temporary workflows, repeated remote generation, or unreviewed branch reconstruction.

Registration closure must instead use a reviewed exact-commit fastpath shadow, wait for activation on `main`, or obtain an explicit user-approved exception with a recorded reason, frozen SHA, logical commit structure, review plan, and rollback plan.

This temporary rule does not block execution of experiments that are already authoritatively registered and otherwise allowed by the handoff. It does not make an unmerged dev-branch tool production authority, relax document-before-experiment requirements, or permit an unregistered formal launch.

## Epistemic independence and anti-sycophancy

User agreement, disagreement, confidence, praise, criticism, or repeated insistence must not directly determine a research or engineering verdict. User feedback is a signal to re-audit the judgment, not evidence that decides the judgment.

Before making an evaluative or comparative judgment:

1. Lock the exact objects, versions, files, branches, and commits being judged.
2. Define the evaluation criteria before considering the user's preferred conclusion.
3. Separate repository facts, experimental evidence, inference, user preference, and unresolved uncertainty.
4. Treat a challenge as a request to re-audit, not as an instruction to agree, disagree, or reverse the verdict.
5. Do not default to agreement, and do not use automatic disagreement as a substitute for independent judgment.
6. Change a prior verdict only when the compared object, evidence, evaluation criteria, or identified reasoning error has materially changed.
7. When changing a verdict, state exactly which premise, evidence, criterion, or object changed.
8. If nothing material changed, preserve the prior conclusion despite pressure in either direction.

## Governance pipeline stage closure

* `docs/governance_pipeline_stage_status.yaml` is the machine-readable canonical stage map for the governance pipeline. Read it before modifying any protected governance file, and run `python3 scripts/validate_governance_pipeline_stage_status.py --repo-root .` before delivery.
* Stage 1, Stage 2, and Stage 5 are `closed_maintenance_only`. Their protected after-images must match the stage ledger and an authorization record under `docs/governance_stage_authorizations/`; new features, architecture expansion, responsibility changes, or default-policy changes require an explicit user-approved `reopen` authorization and rollback plan.
* Stage 5 activated production schema-v3 delta authority at commit `e33a3d1ce8de8ebaf0969a2ec9830a031f7a6c04`. `docs/handoff.md` remains the read master but may not be edited directly. An update that changes the research handoff or `experiments/registry.yaml` must add exactly one schema-v3 delta under `docs/handoff_deltas/<update_id>/HANDOFF_DELTA.yaml`; the trusted current-main normalizer materializes `docs/handoff.md`, validates the registry event, and refreshes Stage 4A generated views. Code-only updates must normalize as a verified no-op. Run `python3 scripts/handoff_authority.py verify --repo-root .` and the governance-stage validator before delivery.
* Stage 3 shadow reports and Full Acceptance records remain immutable provenance for the engine that Stage 5 promoted; they no longer make the manual handoff the write authority.

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

## Manuscript hierarchy and cascade rule

For paper-writing tasks, the canonical hierarchy is:

1. outline;
2. paragraph blueprint;
3. prose.

The outline is the structural contract. The blueprint derives from the outline, and prose derives from the blueprint. When a user reports a manuscript problem, always inspect the matching outline paragraph first, then the blueprint only if the outline is correct, then prose only if both parents are correct. The first failing layer determines the edit root. Propagate every parent-layer correction through all configured downstream layers, and do not rewrite an upstream layer that was explicitly verified as correct.

A child-parent mismatch is evidence against the child by default, not evidence that the parent is wrong. In particular, when the user says the blueprint does not match an already approved outline, the outline must be recorded as `pass` and the blueprint as the first failing layer unless the outline content is independently shown to be wrong. Never change an outline merely to make a mismatched blueprint validate. If a useful optimization is discovered while reviewing a blueprint, re-triage the corresponding outline content first: an explicitly approved outline revision cascades to blueprint and prose; otherwise only the blueprint and its downstream prose may change.

Do not independently add, remove, reorder, split, merge, or rename paragraphs in a blueprint or prose file. Use stable paragraph IDs, exact cross-layer order/title matching, and parent SHA-256 fingerprints as specified in `docs/manuscript_cascade/README.md`. Before delivering a manuscript update, run `scripts/manuscript_cascade.py` for issue triage, artifact hierarchy validation, and Git cascade validation whenever the relevant manuscript artifacts have been registered in a hierarchy config.

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

## Emergency offline patch-delivery fallback

This is a deprecated emergency path, not a normal development route, and it must not be selected automatically. Use it only when the connected GitHub App has been actually checked and cannot perform an operation required by the task, or when the user explicitly requests an offline update package. Shell DNS, `git clone`, or local-container network failure alone is insufficient. While the GitHub App can create branches, write files, open PRs, inspect Actions, and merge after approval, do not request a bundle and do not produce a `drpo-update` package.

When this emergency path is explicitly activated, provide one verified downloadable ZIP compatible with the local `drpo-update` workflow. **All newly produced code-update packages must use the canonical bundle-backed format** and must contain:

* `update.patch`, a unified patch applicable with `git apply`;
* `BASE_COMMIT.txt`, containing only the full base commit SHA;
* `CHANGE_SUMMARY.md`;
* `TEST_COMMANDS.sh`, with executable non-placeholder commands;
* `modified_files/`, containing complete modified files with repository-relative paths;
* `change.bundle` and `PATCH_COMMIT.txt`, whose patch commit has the package base as its unique parent and is tree-equivalent to `update.patch`;
* `UPDATE_PACKAGE_MANIFEST.json`, generated by the canonical producer;
* experiment artifacts and checksums when the task includes results.

Use `scripts/package_update.py` as the canonical producer and `scripts/verify_update_package.py` as the producer-side verifier. Do not manually assemble a new patch-only ZIP. Historical exact-base patch-only packages remain supported by `drpo-update` as a consumption-compatibility path only; that compatibility is not permission to produce new legacy packages.

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

The following controls are mandatory for formal runs and code delivery under `GOV-EXP-ARTIFACT-02`, `GOV-EXP-ARTIFACT-03`, and `GOV-BASE-FRESHNESS-01`:

* Resolve the current commit with Git (`git ls-remote origin refs/heads/main` and `git rev-parse HEAD`), not a cached web commits page. If authoritative remote resolution is unavailable, report that limitation rather than inventing a current SHA.
* A formal experiment may start only from a clean worktree whose full commit SHA is recorded. A dirty worktree is permitted only for an explicitly labelled pilot with `--allow-dirty`, and its tracked, staged, and bounded untracked launch snapshot must be captured before process start.
* At process exit, re-check HEAD and worktree status. A changed HEAD or dirty formal worktree marks provenance as compromised and forces failed-run handling even if the child process returned zero.
* Main artifact ZIPs are first built as candidates, internally verified, and atomically published only after checksum, safe-path, base-commit, and `git apply --check` validation succeeds. The apply check must use an isolated index loaded from the immutable base commit, not the caller's possibly staged index.
* Failed, checkpoint, and raw-complete recovery packages are lightweight evidence packages by default. Large model adapters, checkpoints, optimizer states, datasets, and caches are indexed rather than copied into the main ZIP.
* Real model weights, adapters, optimizer states, and checkpoints remain on persistent training-server storage by default, regardless of individual file size. The main ZIP records path, size, SHA-256, role, and persistence status; a sidecar is disabled by default and is created only for explicitly selected, pre-registered files with a declared cross-machine transfer, restart, or independent-audit purpose. Foundation-model weights are never copied.
* The generic CLI default for large-file persistence is `persistent_local`. Runs in ephemeral containers must explicitly override it to `ephemeral` or `unknown`; the label may never be used merely because a file currently exists on local disk.
* Formal large-model experiments must pre-register an artifact budget and checkpoint-retention policy. Do not save foundation-model weights, redundant checkpoints, or optimizer state unless the registered recovery plan requires them.
* Packaging must never follow result-directory symbolic links. External links are rejected; internal links are recorded as references and their targets are packaged at most once.
* The default main-package hard limit is 25 MiB and the default single-file main-package limit is 10 MiB. Crossing either limit must trigger exclusion/sidecar handling or fail before final publication; a post-hoc warning is insufficient.
* The public guard, package, and verify entry points must all use the same hardened implementation and fail closed if that implementation is missing; silently falling back to a legacy protocol is prohibited.
* Before delivering a `drpo-update` ZIP, apply it in a fresh clean checkout of the confirmed base, preserve executable file modes, run the package's own `TEST_COMMANDS.sh`, run `git diff --check`, and withhold the final ZIP if any step fails.
* The merge-equivalent base must come from an authoritative immutable full-SHA source. Prefer a real Git checkout/fetch or Git bundle containing the expected commit object. For code modification and `drpo-update` pre-delivery testing, an official GitHub archive pinned to the full SHA may be promoted to a verified source capsule only after safe extraction, archive SHA-256 recording, complete-tree inventory, file-mode preservation, and a second independent extraction/apply/test pass. An unpinned branch ZIP, user-created ZIP without immutable provenance, web snippets, parsed pages, copied files, or a hand-built partial repository are not valid bases. Archive-only capsule mode does not automatically satisfy a formal experiment launch unless the supervisor explicitly validates that capsule mode.
* A formal guard launch must be anchored either by an explicit full `--expected-commit` available in the local Git object database or by a live authoritative `origin/main` match. In an offline clone or Git-bundle checkout, always pass the explicit full SHA. Merely trusting whatever local `HEAD` happens to contain is not a formal source preflight.
* Every requested `--source-file` must be a repository-relative path present in the launch commit and is checked before the child process starts. A missing source snapshot input is a preflight failure, not a post-training packaging surprise.
* Pre-delivery validation must exercise the same merge path as the user: verify `BASE_COMMIT.txt`, run `git apply --check`, apply the patch in the fresh checkout, execute `TEST_COMMANDS.sh`, and confirm the declared modified-file set and Git modes. The user-side `drpo-update` run is a final environment check, not the first integration test.
* Candidate-package failure must never delete or overwrite an already verified final package. The old final remains until the new candidate and any explicitly requested sidecar have passed verification. Sidecars use new versioned filenames rather than overwriting an existing sidecar; if main publication fails after sidecar publication, the newly published orphan sidecar is removed.
* Sidecar verification must bind the sidecar manifest to one valid experiment ID, one full base commit, one registered purpose, and the exact payload inventory. Each selected member's path, size, and SHA-256 must match; undeclared or missing payload members are hard failures.
* The supervisor must preserve failed-run evidence even when the experiment command cannot start. `Popen`/working-directory/permission failures require `RUN_FAILED.json`, a traceback log, launch commit, and an attempted lightweight recovery package.
* Every supervised attempt must use a new or empty output root. Existing files may not be reused to satisfy required-output checks or be silently mixed into a new artifact; resume input must be read from a separately declared persistent checkpoint path.
* Every recovery package is bound to the launch commit, not the packaging-time HEAD. If HEAD changes during a run, source snapshots must be read from the launch commit and the package must record both launch and packaging commits.
* `experiment-final` packages require parseable `RUN_COMPLETE.json`, terminal audit, `run_manifest.json`, and at least one log file. Their experiment ID and base commit must agree before publication.
* Formal source availability is a launch preflight. Browser-only code viewing and a plain source ZIP are sufficient for review but are not exact formal checkouts. Before asking the user to upload anything, exhaust the automated acquisition ladder: an existing exact checkout/bundle, shell clone/fetch, an environment download tool for a full-SHA Git bundle or Git-metadata-bearing verified source capsule, and project-persistent source storage. Only if all automated paths fail may the user be asked for one complete Git bundle/source capsule; never request ad hoc files or treat a plain Source code ZIP as a formal checkout.
* User upload is not the preferred or default source path and is not required when an automated path yields an exact commit-bearing checkout, Git bundle, or verified capsule. A formal run without any user source upload is fully valid once the expected full SHA exists in the local Git object database and all normal clean-worktree/source-file preflights pass.
* The exact-checkout rule for patch generation remains stricter than read-only code review. Earlier experiments are not retroactively reclassified solely because the newer provenance gate did not yet exist; their status remains whatever the handoff and surviving evidence support.
* Experiment IDs and archive paths must be validated before filesystem use; traversal, root/parent symlinks, duplicate ZIP members, malformed manifests, unknown package kinds, and runtime paths overlapping tracked repository files are hard failures.
* Every result package kind—not only final packages—must bind `run_manifest.json` and its status marker to the same experiment ID and launch commit. Small raw NumPy evidence remains in the main ZIP when below the size limit; model/checkpoint state remains index-only regardless of size.
* Result files are hashed during discovery and revalidated after copying. A file that changes between scan and ZIP assembly invalidates the candidate rather than producing a manifest/content mismatch.
* A stale child receives SIGTERM once and is escalated to SIGKILL after the registered grace period if it remains alive. Supervisor setup, monitoring, and end-provenance failures all enter the same failed-evidence path.

* Base freshness is checked three times: at session start, immediately before locking a formal execution or update-package build, and immediately before final ZIP delivery. Use `scripts/resolve_main_commit.py --phase ... --ledger ...`; record UTC time, local HEAD, selected base, authoritative remote SHA, and resolution method at every checkpoint.
* A session must discover a newer `main` itself whenever an authoritative channel is available. `git ls-remote` is first choice; if shell networking fails, use an official GitHub commit/API channel through the available web or download bridge and pass the independently resolved SHA with its method. A SHA supplied by the user is a useful hint or emergency fallback, but it is not a substitute for an available authoritative check.
* If authoritative `main` changes between freshness checkpoints or no longer matches the selected base, stop before execution or delivery, refresh/rebase to the new SHA, reread `AGENTS.md`, handoff Section 0, and the registry, then rerun `git apply --check`, the patch application, `TEST_COMMANDS.sh`, and ZIP verification. Do not deliver a package known to be stale and do not wait for the user to notice the new commit.
* Shell DNS failure alone is not grounds to ask the user for source. Inspect the session's available tools and exhaust existing checkout, Git clone/fetch, official full-SHA archive/download bridge, and project-persistent source storage. Only after those paths actually fail may one complete Git bundle or verified source capsule be requested; never request repeated ad hoc files.
